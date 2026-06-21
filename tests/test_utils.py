"""Tests for the shared metric helpers — every comparison table in the project ranks on
these, so their edge cases (unreachable precision, perfect separation) must be nailed down."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import evaluate, recall_at_precision  # noqa: E402

EVAL_KEYS = {"auprc", "roc_auc", "f1", "precision", "recall", "accuracy",
             "recall_at_p80", "threshold_at_p80", "n"}


def test_evaluate_returns_full_bundle():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    m = evaluate(y, s)
    assert EVAL_KEYS.issubset(m)
    assert 0.0 <= m["auprc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["recall_at_p80"] <= 1.0
    assert m["n"] == 200


def test_evaluate_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    m = evaluate(y, s)
    assert m["auprc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_at_p80"] == pytest.approx(1.0)


def test_evaluate_threshold_drives_hard_labels():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.4, 0.9, 0.6, 0.3])
    hi = evaluate(y, s, threshold=0.95)   # nothing predicted positive
    lo = evaluate(y, s, threshold=0.05)   # everything predicted positive
    assert hi["recall"] == 0.0
    assert lo["recall"] == 1.0


def test_recall_at_precision_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    rec, thr, prec = recall_at_precision(y, s, 0.80)
    assert rec == pytest.approx(1.0)
    assert prec >= 0.80
    assert thr is not None


def test_recall_at_precision_unreachable_returns_none_threshold():
    # a score with no signal can never reach precision 0.95 -> contract: (0.0, None, <prec>)
    y = np.array([1, 0] * 50)
    s = np.full(100, 0.5)
    rec, thr, prec = recall_at_precision(y, s, 0.95)
    assert rec == 0.0
    assert thr is None              # None, not +inf -> JSON-strict
    assert prec <= 0.95


def test_recall_at_precision_threshold_separates():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.2, 0.3, 0.55, 0.6, 0.7, 0.9])
    rec, thr, prec = recall_at_precision(y, s, 0.80)
    # at the returned threshold, the achieved precision really is >= target
    yhat = (s >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    assert tp / (tp + fp) >= 0.80 - 1e-9
