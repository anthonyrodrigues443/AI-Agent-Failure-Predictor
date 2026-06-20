"""Contract tests for the canonical featuriser — the train/serve schema must not drift."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import generate_traces  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    assemble_features, early_window_features, featurize_one, synthesize_run,
    ALL_FEATURE_ORDER, EW_FEATURE_ORDER, TRACE_COLS, FEATURE_GROUPS,
)


@pytest.fixture(scope="module")
def traces():
    return generate_traces(400, seed=7).reset_index(drop=True)


def test_all_feature_order_is_49_unique():
    assert len(ALL_FEATURE_ORDER) == 49
    assert len(set(ALL_FEATURE_ORDER)) == 49


def test_assemble_features_schema(traces):
    X = assemble_features(traces)
    assert list(X.columns) == ALL_FEATURE_ORDER          # exact order
    assert len(X) == len(traces)
    assert not X.isnull().any().any()                    # no NaN
    assert np.isfinite(X.to_numpy()).all()               # no inf


def test_feature_groups_cover_known_columns():
    grouped = {f for fs in FEATURE_GROUPS.values() for f in fs}
    # every grouped feature is a real model feature (no typos in the UI rollup)
    assert grouped.issubset(set(ALL_FEATURE_ORDER))


def test_early_window_schema(traces):
    F = early_window_features(traces, k=3)
    assert list(F.columns) == EW_FEATURE_ORDER
    assert not F.isnull().any().any()


def test_synthesize_run_is_featurizable():
    run = synthesize_run(num_steps=10, task_type="deep_research", model_tier="small",
                         tool_error_rate=0.5, max_consecutive_retries=2,
                         context_max_pct=0.8, reasoning_loops=1)
    assert all(tc in run for tc in TRACE_COLS)
    assert all(len(run[tc]) == run["num_steps"] for tc in TRACE_COLS)
    X = featurize_one(run)
    assert list(X.columns) == ALL_FEATURE_ORDER
    assert X.shape == (1, 49)


def test_synthesize_run_clamps_steps():
    short = synthesize_run(1, "code_gen", "mid", 0.0, 0, 0.2, 0)
    assert short["num_steps"] == 3                        # clamped to minimum
    long = synthesize_run(99, "code_gen", "mid", 0.0, 0, 0.2, 0)
    assert long["num_steps"] == 45                        # clamped to MAX_STEPS


def test_single_row_missing_category_still_49():
    # a run whose task/tier are the dropped baseline levels -> all dummies 0, still 49 cols
    run = synthesize_run(8, "code_gen", "frontier", 0.0, 0, 0.3, 0)
    X = featurize_one(run)
    assert X.shape == (1, 49)
    assert X[["task_type_data_analysis", "model_tier_mid"]].sum().sum() == 0
