"""Inference for the AI-Agent-Failure-Predictor.

Loads the trained champion + early-window artefacts and scores agent runs:

  * `predict_run(run)`      — full-run failure probability, calibrated, with a hard
                              label at the deployed P>=0.80 threshold and a risk band.
  * `explain_run(run)`      — per-feature SHAP contributions grouped by telemetry family
                              (Error/Retry first — it's the load-bearing signal).
  * `early_window_curve(run)` — calibrated failure risk using only the first k steps,
                              for every k in the grid → the "failure in N steps" timeline.
  * `predict_batch(df)`     — vectorised scoring of many runs.

A "run" is a mapping with the run-level aggregates AND the per-step traces (see
feature_engineering.TRACE_COLS), e.g. a row from data_pipeline.generate_traces or a
record from results/ui_examples.json.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Mapping

import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import (
    assemble_features, early_window_features, featurize_one,
    ALL_FEATURE_ORDER, FEATURE_GROUPS, EW_FEATURE_ORDER,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

RISK_BANDS = [  # (upper_bound_exclusive, label)
    (0.15, "Low"), (0.40, "Elevated"), (0.65, "High"), (1.01, "Critical"),
]


@lru_cache(maxsize=1)
def load_champion(path: str | None = None) -> dict:
    path = path or os.path.join(MODELS_DIR, "champion.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.train` first to build the champion.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_early_window(path: str | None = None) -> dict:
    path = path or os.path.join(MODELS_DIR, "early_window.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `python -m src.train` first.")
    return joblib.load(path)


def risk_band(prob: float) -> str:
    for upper, label in RISK_BANDS:
        if prob < upper:
            return label
    return "Critical"


# ---------------------------------------------------------------------------------------
# Full-run scoring
# ---------------------------------------------------------------------------------------
def predict_proba_matrix(X: pd.DataFrame, champ: dict | None = None) -> np.ndarray:
    champ = champ or load_champion()
    X = X.reindex(columns=champ["features"], fill_value=0.0)
    return champ["calibrated_model"].predict_proba(X)[:, 1]


def predict_run(run: Mapping, champ: dict | None = None) -> dict:
    champ = champ or load_champion()
    X = featurize_one(run)
    prob = float(predict_proba_matrix(X, champ)[0])
    thr = champ["threshold"]
    return {
        "failure_probability": prob,
        "threshold": thr,
        "predicted_failure": bool(prob >= thr) if thr is not None else bool(prob >= 0.5),
        "risk_band": risk_band(prob),
    }


def predict_batch(df_traces: pd.DataFrame, champ: dict | None = None) -> pd.DataFrame:
    champ = champ or load_champion()
    X = assemble_features(df_traces)
    prob = predict_proba_matrix(X, champ)
    thr = champ["threshold"] if champ["threshold"] is not None else 0.5
    return pd.DataFrame({
        "failure_probability": prob,
        "predicted_failure": (prob >= thr).astype(int),
        "risk_band": [risk_band(p) for p in prob],
    }, index=df_traces.index)


# ---------------------------------------------------------------------------------------
# Explanation — CatBoost native SHAP on the (uncalibrated) ranker twin.
# ---------------------------------------------------------------------------------------
def explain_run(run: Mapping, champ: dict | None = None, top_n: int = 8) -> dict:
    """Return SHAP feature contributions (log-odds units) and a family-level rollup.

    Uses the saved `shap_model` (plain CatBoost). Calibration is a monotone post-map, so
    attributions are the ranker's — directionally identical to the served probability.
    """
    champ = champ or load_champion()
    shap_model = champ.get("shap_model")
    X = featurize_one(run)[champ["features"]]
    if shap_model is None:  # graceful fallback if a champion was saved without the twin
        return {"available": False, "features": [], "groups": []}

    from catboost import Pool
    raw = shap_model.get_feature_importance(Pool(X), type="ShapValues")[0]
    contribs = raw[:-1]          # last entry is the expected-value/bias term
    base = float(raw[-1])

    per_feat = sorted(
        ({"feature": f, "shap": float(s), "value": float(X.iloc[0][f])}
         for f, s in zip(champ["features"], contribs)),
        key=lambda d: abs(d["shap"]), reverse=True,
    )

    feat_to_group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    group_tot: dict[str, float] = {}
    for f, s in zip(champ["features"], contribs):
        group_tot[feat_to_group.get(f, "Other")] = group_tot.get(feat_to_group.get(f, "Other"), 0.0) + float(s)
    groups = sorted(({"group": g, "shap": v} for g, v in group_tot.items()),
                    key=lambda d: abs(d["shap"]), reverse=True)

    return {"available": True, "base_value": base, "features": per_feat[:top_n],
            "all_features": per_feat, "groups": groups}


# ---------------------------------------------------------------------------------------
# Early-window timeline — "predicted failure N steps before it happened".
# ---------------------------------------------------------------------------------------
def early_window_curve(run: Mapping, ew: dict | None = None) -> dict:
    """Risk estimate if we had only observed the first k steps, for each k in the grid
    (clipped to the run's actual length). Returns the per-k probabilities plus, if the
    run's final champion verdict is 'failure', the earliest k whose risk already crosses
    a chosen alert level — i.e. how many steps of lead time the early model buys."""
    ew = ew or load_early_window()
    df = pd.DataFrame([dict(run)])
    n_steps = int(run["num_steps"])
    points = []
    for k in ew["k_grid"]:
        if k > n_steps:
            break
        F = early_window_features(df, k).reindex(columns=ew["features"], fill_value=0.0)
        p = float(ew["models"][k].predict_proba(F)[:, 1][0])
        points.append({"k": int(k), "prob": p, "pct_of_full": _pct_full(ew, k)})
    return {"points": points, "n_steps": n_steps, "full_auprc": ew["full_auprc"]}


def _pct_full(ew: dict, k: int) -> float:
    for row in ew.get("table", []):
        if row["k"] == k:
            return float(row["pct_of_full"])
    return float("nan")


def early_warning_lead(run: Mapping, alert_prob: float = 0.5, ew: dict | None = None) -> dict:
    """Lead time: the earliest step k at which the early-window risk >= alert_prob, and how
    many steps that is before the run's end."""
    curve = early_window_curve(run, ew)
    n = curve["n_steps"]
    crossed = next((pt for pt in curve["points"] if pt["prob"] >= alert_prob), None)
    if crossed is None:
        return {"alerted": False, "alert_step": None, "steps_early": None, "n_steps": n,
                "curve": curve["points"]}
    return {"alerted": True, "alert_step": crossed["k"], "steps_early": n - crossed["k"],
            "n_steps": n, "alert_prob": crossed["prob"], "curve": curve["points"]}


if __name__ == "__main__":
    import json
    ex_path = os.path.join(ROOT, "results", "ui_examples.json")
    examples = json.load(open(ex_path))["examples"]
    champ = load_champion()
    print(f"champion: {champ['champion']} · threshold {champ['threshold']:.4f} · "
          f"test AUPRC {champ['test_metrics']['auprc']:.4f}")
    for ex in examples[:4]:
        out = predict_run(ex, champ)
        lead = early_warning_lead(ex)
        print(f"\n true={ex['_true_label']} reason={ex.get('failure_reason')!s:18s} "
              f"P(fail)={out['failure_probability']:.3f} [{out['risk_band']}] "
              f"pred_fail={out['predicted_failure']} · "
              f"early-warn at step {lead['alert_step']} ({lead['steps_early']} steps early)"
              if lead["alerted"] else
              f"\n true={ex['_true_label']} P(fail)={out['failure_probability']:.3f} "
              f"[{out['risk_band']}] — no early alert")
        exp = explain_run(ex, champ, top_n=3)
        if exp["available"]:
            top = ", ".join(f"{g['group']} ({g['shap']:+.2f})" for g in exp["groups"][:3])
            print(f"   top families: {top}")
