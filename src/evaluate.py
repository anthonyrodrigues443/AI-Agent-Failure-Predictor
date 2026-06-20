"""Evaluation suite for the trained champion.

Regenerates the held-out test split (deterministically — same seed/split as training),
scores the saved champion, and reports the full metric bundle + a per-prediction latency
benchmark. This is the number that backs the "147,000x faster than Opus" headline: the
served model decides in microseconds on CPU.

Run:  python -m src.evaluate
Writes results/phase6_eval.json.
"""
from __future__ import annotations

import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from src.data_pipeline import generate_traces  # noqa: E402
from src.feature_engineering import assemble_features  # noqa: E402
from src.predict import load_champion, predict_proba_matrix  # noqa: E402
from src.utils import evaluate as metric_bundle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cfg() -> dict:
    try:
        import yaml
        return yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    except Exception:
        return {"seed": 42, "data": {"n_runs": 20000, "test_size": 0.25},
                "metrics": {"operating_precision": 0.80}}


def build_test_split(n_runs: int, seed: int, test_size: float):
    tr = generate_traces(n_runs, seed).reset_index(drop=True)
    X = assemble_features(tr)
    y = tr["failure"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return tr, Xte, yte


def benchmark_latency(X, champ, n_warmup: int = 50, n_iter: int = 300) -> dict:
    """Single-row inference latency (featurised row already in hand)."""
    model = champ["calibrated_model"]
    rows = [X.iloc[[i % len(X)]] for i in range(n_warmup + n_iter)]
    for r in rows[:n_warmup]:
        model.predict_proba(r)
    t0 = time.perf_counter()
    for r in rows[n_warmup:]:
        model.predict_proba(r)
    per = (time.perf_counter() - t0) / n_iter
    # Batched throughput (the realistic serving mode).
    t1 = time.perf_counter()
    model.predict_proba(X)
    batch = (time.perf_counter() - t1) / len(X)
    return {"single_row_ms": per * 1e3, "batched_ms_per_row": batch * 1e3,
            "throughput_rows_per_s": 1.0 / batch}


def main() -> dict:
    cfg = _load_cfg()
    test_size, op = cfg["data"]["test_size"], cfg["metrics"]["operating_precision"]
    champ = load_champion()
    # Rebuild the SAME split the champion was trained on (from its stored metadata), not the
    # current config — so evaluation stays valid even if config drifts or --n/--seed was used.
    seed = champ.get("seed", cfg.get("seed", 42))
    n_runs = champ.get("n_runs", cfg["data"]["n_runs"])

    print(f"[evaluate] champion: {champ['champion']} · threshold {champ['threshold']:.4f}")
    print(f"[evaluate] rebuilding held-out split (n={n_runs}, seed={seed}) ...")
    tr, Xte, yte = build_test_split(n_runs, seed, test_size)

    p = predict_proba_matrix(Xte, champ)
    m = metric_bundle(yte, p, threshold=0.5, operating_precision=op)
    m["brier"] = float(brier_score_loss(yte, p))
    # Hard metrics at the DEPLOYED threshold (not 0.5).
    thr = champ["threshold"]
    yhat = (p >= thr).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    deployed = {"threshold": float(thr),
                "precision": float(precision_score(yte, yhat, zero_division=0)),
                "recall": float(recall_score(yte, yhat, zero_division=0)),
                "f1": float(f1_score(yte, yhat, zero_division=0))}

    lat = benchmark_latency(Xte, champ)

    # Per-failure-reason recall at the deployed threshold (where does it work / not work?).
    reasons = tr.loc[Xte.index, "failure_reason"].reset_index(drop=True)
    yte_s, yhat_s = np.asarray(yte), yhat
    fails = yte_s == 1
    import pandas as pd
    rr = (pd.DataFrame({"reason": reasons[fails].values, "caught": yhat_s[fails]})
          .groupby("reason")["caught"].agg(n="count", recall="mean").reset_index()
          .sort_values("n", ascending=False))

    report = {
        "champion": champ["champion"],
        "test_prevalence": float(yte.mean()),
        "n_test": int(len(yte)),
        "threshold_free": {k: m[k] for k in ["auprc", "roc_auc", "brier"]},
        "at_threshold_0.5": {k: m[k] for k in ["f1", "precision", "recall", "accuracy"]},
        "recall_at_p80": m["recall_at_p80"],
        "deployed_operating_point": deployed,
        "latency": lat,
        "recall_by_reason": rr.round(4).to_dict(orient="records"),
    }

    print("\n=== held-out test metrics ===")
    print(f"  AUPRC {m['auprc']:.4f} · ROC-AUC {m['roc_auc']:.4f} · Brier {m['brier']:.4f}")
    print(f"  deployed (thr {thr:.3f}): P={deployed['precision']:.3f} "
          f"R={deployed['recall']:.3f} F1={deployed['f1']:.3f}")
    print(f"  recall @ P=0.80: {m['recall_at_p80']:.3f}")
    print("\n=== latency (CPU, single-threaded) ===")
    print(f"  single-row {lat['single_row_ms']:.3f} ms · batched {lat['batched_ms_per_row']*1e3:.1f} us/row "
          f"· {lat['throughput_rows_per_s']:,.0f} rows/s")
    # Headline framing vs the Phase-5 LLM head-to-head (Opus ~10.3s/call).
    speedup = 10.3 / (lat["batched_ms_per_row"] / 1e3)
    print(f"  ≈ {speedup:,.0f}x faster than Claude Opus zero-shot (10.3 s/call, Phase 5)")
    print("\n=== recall by failure reason (deployed threshold) ===")
    print(rr.round(3).to_string(index=False))

    out = os.path.join(ROOT, "results", "phase6_eval.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[evaluate] wrote {out}")
    return report


if __name__ == "__main__":
    main()
