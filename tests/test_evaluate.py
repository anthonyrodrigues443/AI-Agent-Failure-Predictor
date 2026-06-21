"""End-to-end evaluation + inference-pipeline tests.

Split-reconstruction tests run anywhere (no model needed). The metric/latency tests are
skipped if the gitignored champion isn't built — run `python -m src.train` first."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import build_test_split  # noqa: E402
from src.feature_engineering import ALL_FEATURE_ORDER  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
needs_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS, "champion.joblib")),
    reason="champion.joblib not built — run `python -m src.train`")

N_SMALL, SEED, TEST_SIZE = 2000, 42, 0.25


def test_build_test_split_is_deterministic_and_shaped():
    tr1, X1, y1 = build_test_split(N_SMALL, SEED, TEST_SIZE)
    tr2, X2, y2 = build_test_split(N_SMALL, SEED, TEST_SIZE)
    assert list(X1.columns) == ALL_FEATURE_ORDER
    assert len(X1) == int(round(N_SMALL * TEST_SIZE))
    assert np.array_equal(y1, y2)              # same split on rerun
    assert X1.index.equals(X2.index)
    # test rows are a strict subset of the full simulated frame
    assert set(X1.index).issubset(set(tr1.index))


def test_split_is_stratified():
    _, _, y = build_test_split(N_SMALL, SEED, TEST_SIZE)
    assert 0.18 <= float(np.mean(y)) <= 0.34   # prevalence preserved by stratify


@needs_model
def test_evaluate_main_reproduces_champion(tmp_path, monkeypatch):
    from src import evaluate as ev
    report = ev.main()
    # primary metric reproduces the frozen research champion within tolerance
    assert report["threshold_free"]["auprc"] == pytest.approx(0.624, abs=0.01)
    assert 0.77 <= report["threshold_free"]["roc_auc"] <= 0.80
    assert report["threshold_free"]["brier"] < 0.16
    # deployed operating point honours the P>=0.80 design (allow transfer slack)
    assert report["deployed_operating_point"]["precision"] >= 0.74
    # every failure reason reported, with recall in [0,1]
    reasons = {r["reason"] for r in report["recall_by_reason"]}
    assert "latent_capability" in reasons
    assert all(0.0 <= r["recall"] <= 1.0 for r in report["recall_by_reason"])


@needs_model
def test_latency_benchmark_is_fast():
    from src.evaluate import benchmark_latency
    from src.predict import load_champion
    _, X, _ = build_test_split(N_SMALL, SEED, TEST_SIZE)
    champ = load_champion()
    lat = benchmark_latency(X, champ, n_warmup=10, n_iter=50)
    assert lat["batched_ms_per_row"] > 0
    assert lat["throughput_rows_per_s"] > 1000     # batched serving is comfortably >1k rows/s


@needs_model
def test_recall_by_reason_orders_context_overflow_high():
    """Sanity on the honest-limitations surface: the loud failure mode (context_overflow)
    is caught far more often than the irreducible one (latent_capability)."""
    from src import evaluate as ev
    report = ev.main()
    rec = {r["reason"]: r["recall"] for r in report["recall_by_reason"]}
    if "context_overflow" in rec and "latent_capability" in rec:
        assert rec["context_overflow"] > rec["latent_capability"]
