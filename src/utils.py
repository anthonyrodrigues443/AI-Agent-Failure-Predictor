"""Shared evaluation helpers — reused across all phases so every comparison table
ranks on the same primary metric (AUPRC) and the same operating point (Recall@P=0.80)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, precision_recall_curve,
)


def recall_at_precision(y_true, y_score, target_precision: float = 0.80):
    """Max recall achievable while precision >= target_precision.

    Returns (recall, threshold, precision_at_threshold). If no threshold reaches the
    target precision, returns (0.0, +inf, achieved_precision_at_that_point).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thr)+1 points; align by dropping the last p/r.
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= target_precision
    if not ok.any():
        # target precision unreachable -> no usable threshold (None, not inf: keep JSON strict)
        return 0.0, None, float(prec.max())
    idx_candidates = np.where(ok)[0]
    best = idx_candidates[np.argmax(rec[idx_candidates])]
    return float(rec[best]), float(thr[best]), float(prec[best])


def evaluate(y_true, y_score, threshold: float = 0.5, operating_precision: float = 0.80) -> dict:
    """Full metric bundle for a probabilistic classifier.

    `y_score` are positive-class probabilities/scores; `threshold` produces the hard labels
    used for F1/precision/recall/accuracy. AUPRC and ROC-AUC are threshold-free.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    rec_at_p, thr_at_p, prec_at_p = recall_at_precision(y_true, y_score, operating_precision)
    # thr_at_p is None when the operating precision is unreachable -> keep JSON strict-valid.
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_at_p80": rec_at_p,
        "threshold_at_p80": thr_at_p,
        "n": int(len(y_true)),
    }


def fmt_row(name: str, m: dict) -> str:
    return (f"{name:28s} AUPRC={m['auprc']:.4f}  ROC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"R@P80={m['recall_at_p80']:.3f}")
