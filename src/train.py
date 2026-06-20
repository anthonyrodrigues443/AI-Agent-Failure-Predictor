"""Production training pipeline for the AI-Agent-Failure-Predictor.

Reproduces the Phase-4 champion deterministically (no Optuna at deploy time — the tuned
hyperparameters are frozen in config/config.yaml) and trains the early-window
("failure in N steps") models. Saves three artefacts under models/:

  * champion.joblib      — calibrated ranker + SHAP twin + frozen P>=0.80 threshold
  * early_window.joblib  — {k: calibrated model} for the step-by-step risk timeline
  * (results/ui_examples.json — curated held-out runs for the Streamlit demo)

Run:  python -m src.train            (uses config defaults)
      python -m src.train --n 20000  (override sample size)

The shared box deadlocks under OpenMP oversubscription, so we pin to single-threaded
numerics up front (each fit is still only ~0.2-4s).
"""
from __future__ import annotations

import argparse
import json
import os
import time

# Pin BEFORE importing numpy/sklearn/catboost (see notebooks — OpenMP deadlock guard).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss  # noqa: E402
from sklearn.model_selection import train_test_split, cross_val_predict  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

from src.data_pipeline import generate_traces, build_and_save  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    assemble_features, early_window_features, ALL_FEATURE_ORDER, EW_FEATURE_ORDER,
)
from src.utils import evaluate, recall_at_precision  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CB_THREADS = 2  # CatBoost honours thread_count internally (not the OMP env var)


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "config", "config.yaml")
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:  # pragma: no cover - config is committed; defaults are a safety net
        print(f"[train] config load failed ({e}); using baked-in defaults")
        return {
            "seed": 42,
            "data": {"n_runs": 20000, "test_size": 0.25},
            "metrics": {"operating_precision": 0.80},
            "champion": {
                "calibration": "sigmoid",
                "catboost_params": dict(iterations=543, learning_rate=0.02490643969382439, depth=4,
                                        l2_leaf_reg=5.475344508142733, random_strength=1.320457481218804,
                                        bagging_temperature=0.12203823484477883, border_count=158),
                "expected_test_auprc": 0.62406, "expected_threshold": 0.6323,
            },
            "early_window": {"k_grid": [2, 3, 4, 5, 6, 8, 10, 12],
                             "histgbm_params": dict(max_iter=400, learning_rate=0.05,
                                                    max_leaf_nodes=31, l2_regularization=1.0)},
        }


def _catboost(params: dict, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False,
                              thread_count=CB_THREADS, **params)


def _histgbm(params: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)


def train(config_path: str | None = None, n_runs: int | None = None, seed: int | None = None,
          save_examples: bool = True) -> dict:
    cfg = load_config(config_path)
    seed = seed if seed is not None else cfg.get("seed", 42)
    n_runs = n_runs or cfg["data"]["n_runs"]
    test_size = cfg["data"]["test_size"]
    op_prec = cfg["metrics"]["operating_precision"]
    champ_cfg = cfg["champion"]
    ew_cfg = cfg["early_window"]
    # The expected-AUPRC/threshold reproduction asserts only hold for the frozen default
    # run; a custom --n/--seed legitimately produces different numbers, so skip them there.
    is_frozen_default = (n_runs == cfg["data"]["n_runs"] and seed == cfg.get("seed", 42))

    models_dir = os.path.join(ROOT, "models")
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[train] generating {n_runs:,} agent runs (seed={seed}) ...")
    t0 = time.time()
    tr = generate_traces(n_runs, seed).reset_index(drop=True)
    # Persist the run-level parquet too (the simulator's canonical aggregate table).
    build_and_save(n_runs, seed, os.path.join(ROOT, "data", "processed", "agent_runs.parquet"))
    y = tr["failure"].values

    print("[train] assembling +ALL features ...")
    X = assemble_features(tr)
    assert list(X.columns) == ALL_FEATURE_ORDER, "feature order drifted from canonical schema"

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    tr_idx, te_idx = Xtr.index, Xte.index
    prevalence = float(yte.mean())
    print(f"[train] split: train {Xtr.shape} · test {Xte.shape} · prevalence {prevalence:.3f} "
          f"({time.time()-t0:.0f}s elapsed)")

    # --- Champion: tuned CatBoost, sigmoid-calibrated ----------------------------------
    cat_params = champ_cfg["catboost_params"]
    cal_method = champ_cfg["calibration"]
    print(f"[train] fitting calibrated champion (CatBoost {cal_method}) ...")
    t1 = time.time()
    calibrated = CalibratedClassifierCV(_catboost(cat_params, seed), method=cal_method, cv=5)
    calibrated.fit(Xtr, ytr)

    # SHAP twin: a plain CatBoost refit on full train. Calibration is a monotone post-map,
    # so it leaves the ranker's feature attributions intact — this twin powers the UI's
    # contributing-factor breakdown via CatBoost's native ShapValues.
    shap_model = _catboost(cat_params, seed)
    shap_model.fit(Xtr, ytr)

    p_test = calibrated.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(yte, p_test, threshold=0.5, operating_precision=op_prec)
    test_metrics["brier"] = float(brier_score_loss(yte, p_test))
    print(f"[train] champion fit in {time.time()-t1:.0f}s · test AUPRC {test_metrics['auprc']:.5f} "
          f"· ROC {test_metrics['roc_auc']:.4f} · Brier {test_metrics['brier']:.4f}")

    # --- Honest operating threshold: chosen on OOF train, frozen, applied to test -------
    print("[train] freezing P>=0.80 threshold on OOF train predictions ...")
    t2 = time.time()
    oof = cross_val_predict(
        CalibratedClassifierCV(_catboost(cat_params, seed), method=cal_method, cv=5),
        Xtr, ytr, cv=5, method="predict_proba")[:, 1]
    rec_oof, thr_star, prec_oof = recall_at_precision(ytr, oof, op_prec)
    yhat = (p_test >= thr_star).astype(int) if thr_star is not None else np.zeros_like(yte)
    from sklearn.metrics import precision_score, recall_score
    honest_prec = float(precision_score(yte, yhat, zero_division=0))
    honest_rec = float(recall_score(yte, yhat, zero_division=0))
    print(f"[train] frozen threshold {thr_star:.4f} (OOF P={prec_oof:.3f} R={rec_oof:.3f}) -> "
          f"test P={honest_prec:.3f} R={honest_rec:.3f}  ({time.time()-t2:.0f}s)")

    # --- Verify we reproduced the research champion (frozen default run only) -----------
    exp_auprc = champ_cfg.get("expected_test_auprc")
    exp_thr = champ_cfg.get("expected_threshold")
    if is_frozen_default:
        if exp_auprc is not None:
            assert abs(test_metrics["auprc"] - exp_auprc) < 5e-4, \
                f"champion AUPRC {test_metrics['auprc']:.5f} != expected {exp_auprc} (>5e-4 drift)"
        if exp_thr is not None and thr_star is not None:
            assert abs(thr_star - exp_thr) < 5e-3, f"threshold {thr_star:.4f} != expected {exp_thr}"
        print("[train] reproduction check passed (AUPRC + threshold within tolerance)")
    else:
        print(f"[train] custom run (n={n_runs}, seed={seed}) — skipping frozen-default "
              f"reproduction asserts")

    champion = {
        "calibrated_model": calibrated,
        "shap_model": shap_model,
        "threshold": float(thr_star) if thr_star is not None else None,
        "features": ALL_FEATURE_ORDER,
        "champion": "CatBoost tuned (+ALL, sigmoid-calibrated)",
        "calibration": cal_method,
        "primary_metric": "average_precision",
        "test_metrics": test_metrics,
        "operating_point": {"threshold": float(thr_star) if thr_star is not None else None,
                            "honest_precision": honest_prec, "honest_recall": honest_rec,
                            "operating_precision": op_prec},
        "prevalence": prevalence,
        "n_runs": int(n_runs),
        "seed": int(seed),
    }
    joblib.dump(champion, os.path.join(models_dir, "champion.joblib"))
    print(f"[train] wrote models/champion.joblib ({test_metrics['auprc']:.4f} AUPRC)")

    # --- Early-window models: one calibrated HistGBM per k -----------------------------
    print("[train] training early-window models ...")
    t3 = time.time()
    base_auprc = test_metrics["auprc"]
    ew_models, ew_table = {}, []
    hist_params = ew_cfg["histgbm_params"]
    for k in ew_cfg["k_grid"]:
        F = early_window_features(tr, k)
        Ftr, Fte = F.loc[tr_idx], F.loc[te_idx]
        cal = CalibratedClassifierCV(_histgbm(hist_params, seed), method="sigmoid", cv=5)
        cal.fit(Ftr, ytr)
        p = cal.predict_proba(Fte)[:, 1]
        auprc = float(average_precision_score(yte, p))
        r60, _, _ = recall_at_precision(yte, p, 0.60)
        ew_models[k] = cal
        ew_table.append({"k": int(k), "auprc": auprc, "pct_of_full": auprc / base_auprc * 100,
                         "recall_at_p60": r60, "brier": float(brier_score_loss(yte, p))})
        print(f"    k={k:2d}: AUPRC {auprc:.4f} ({auprc/base_auprc*100:.0f}% of full) · R@P60 {r60:.3f}")
    ew_artifact = {"models": ew_models, "features": EW_FEATURE_ORDER, "k_grid": list(ew_cfg["k_grid"]),
                   "full_auprc": base_auprc, "table": ew_table, "calibration": "sigmoid"}
    joblib.dump(ew_artifact, os.path.join(models_dir, "early_window.joblib"))
    print(f"[train] wrote models/early_window.joblib ({time.time()-t3:.0f}s)")

    # --- Curated demo runs for the Streamlit app ---------------------------------------
    if save_examples:
        _save_ui_examples(tr, te_idx, p_test, yte, thr_star, results_dir)

    summary = {"test_metrics": test_metrics, "threshold": thr_star,
               "honest_precision": honest_prec, "honest_recall": honest_rec,
               "early_window": ew_table, "prevalence": prevalence,
               "elapsed_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(results_dir, "phase6_train_summary.json"), "w"), indent=2)
    print(f"[train] DONE in {time.time()-t0:.0f}s — summary -> results/phase6_train_summary.json")
    return summary


def _save_ui_examples(tr: pd.DataFrame, te_idx, p_test, yte, thr_star, results_dir: str,
                      n_per_bucket: int = 3) -> None:
    """Pick representative held-out runs (failures by reason, successes, borderline,
    telemetry-light miss) so the demo always has compelling, real examples."""
    te = tr.loc[te_idx].copy().reset_index(drop=True)
    te["prob"] = p_test
    te["y"] = np.asarray(yte)
    thr = thr_star if thr_star is not None else 0.5
    rng = np.random.default_rng(42)

    picks: list[int] = []

    def take(mask, n, sort_by=None, ascending=False):
        idx = te.index[mask]
        if sort_by is not None:
            idx = te.loc[idx].sort_values(sort_by, ascending=ascending).index
        else:
            idx = pd.Index(rng.permutation(idx.to_numpy()))
        for i in idx[:n]:
            if i not in picks:
                picks.append(int(i))

    # Confident, correct catches by failure reason (the "it works" cases).
    for reason in ["stuck_retry_loop", "context_overflow", "cascade_failure"]:
        take((te.y == 1) & (te.failure_reason == reason) & (te.prob >= thr), n_per_bucket,
             sort_by="prob", ascending=False)
    # Confident correct passes.
    take((te.y == 0) & (te.prob < 0.15), n_per_bucket, sort_by="prob", ascending=True)
    # Borderline (near the deployed threshold) — both labels.
    take((te.prob.sub(thr).abs() < 0.05), 2, sort_by=None)
    # The honest miss: telemetry-light failure the model can't see (capability gap).
    take((te.y == 1) & (te.failure_reason == "latent_capability"), 1, sort_by="prob", ascending=True)
    take((te.y == 1) & (te.failure_reason == "early_exogenous"), 1, sort_by=None)

    cols_agg = [c for c in te.columns if not c.startswith("trace_") and c not in ("prob", "y")]
    examples = []
    for i in picks:
        row = te.loc[i]
        rec = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in cols_agg}
        for tc in ["trace_ctx_pct", "trace_tokens", "trace_latency",
                   "trace_err", "trace_retry", "trace_tool", "trace_loop"]:
            rec[tc] = [float(x) for x in row[tc]]
        rec["_true_label"] = int(row["y"])
        rec["_champion_prob"] = float(row["prob"])
        rec["_correct"] = int((row["prob"] >= thr) == bool(row["y"]))
        examples.append(rec)

    out = {"threshold": float(thr), "examples": examples}
    json.dump(out, open(os.path.join(results_dir, "ui_examples.json"), "w"), indent=2)
    print(f"[train] wrote results/ui_examples.json ({len(examples)} curated runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the agent-failure champion + early-window models")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="number of runs to simulate")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-examples", action="store_true")
    args = ap.parse_args()
    train(args.config, args.n, args.seed, save_examples=not args.no_examples)
