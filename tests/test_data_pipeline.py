"""Generator-contract tests for the causal agent-run simulator.

These lock the four design invariants the whole project rests on (see data_pipeline
docstring): determinism, trace/aggregate consistency, run-length decoupled from outcome,
and no single-feature label leakage. If any of these breaks, every downstream metric is
suspect — so they are cheap insurance against silently re-introducing the Phase-1 leak.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import (  # noqa: E402
    generate_dataset, generate_traces, MAX_STEPS, TASK_TYPES, MODEL_TIERS,
)
from src.feature_engineering import (  # noqa: E402
    TRACE_COLS, assemble_features, ALL_FEATURE_ORDER,
)

N = 3000
SEED = 11
KNOWN_REASONS = {
    "none", "early_exogenous", "context_overflow", "stuck_retry_loop",
    "cascade_failure", "degenerate_loop", "latent_capability",
}


@pytest.fixture(scope="module")
def df():
    return generate_dataset(N, SEED)


@pytest.fixture(scope="module")
def traces():
    return generate_traces(N, SEED)


def test_determinism_same_seed_identical(df):
    pd.testing.assert_frame_equal(df, generate_dataset(N, SEED))


def test_different_seed_changes_data(df):
    other = generate_dataset(N, SEED + 1)
    # prevalence is similar, but the rows are not the same draw
    assert not df["num_steps"].equals(other["num_steps"])


def test_traces_aggregates_byte_identical_to_dataset(df, traces):
    """The documented guarantee: attaching per-step traces consumes no extra rng draws,
    so the aggregate columns must be element-for-element identical to generate_dataset."""
    pd.testing.assert_frame_equal(traces[df.columns], df)


def test_trace_lengths_equal_num_steps(traces):
    for tc in TRACE_COLS:
        lengths = traces[tc].map(len).to_numpy()
        assert np.array_equal(lengths, traces["num_steps"].to_numpy()), tc


def test_label_is_binary_and_reason_known(df):
    assert set(df["failure"].unique()).issubset({0, 1})
    assert set(df["failure_reason"].unique()).issubset(KNOWN_REASONS)
    # a failure must carry a non-'none' reason, and a success must be 'none'
    assert (df.loc[df.failure == 1, "failure_reason"] != "none").all()
    assert (df.loc[df.failure == 0, "failure_reason"] == "none").all()


def test_prevalence_in_calibrated_band(df):
    # generator is calibrated to ~0.26 failure rate; allow sampling slack at n=3000
    assert 0.20 <= df["failure"].mean() <= 0.32


def test_num_steps_bounds_and_categories(df):
    assert df["num_steps"].between(3, MAX_STEPS).all()
    assert set(df["task_type"].unique()).issubset(set(TASK_TYPES))
    assert set(df["model_tier"].unique()).issubset(set(MODEL_TIERS))


def test_run_length_decoupled_from_outcome(df):
    """num_steps must NOT by itself separate the classes — that was the original leak
    (successes terminated early, only failures accumulated telemetry)."""
    auc = roc_auc_score(df["failure"], df["num_steps"])
    assert max(auc, 1 - auc) < 0.62, f"num_steps alone scores AUC {auc:.3f} — length leak?"


def test_no_single_feature_leaks_the_label(traces):
    """No single MODEL-INPUT feature should near-perfectly encode the label (the Phase-1 bug
    produced a feature with AUC 1.0). A real causal generator caps single-feature AUC. This
    guard runs over the FULL 49-feature `+ALL` matrix the champion actually trains on — base
    aggregates AND the engineered LEAD/DOM interactions (the strongest, ix_retry_casc, is the
    one to watch) — not just the raw aggregates."""
    X = assemble_features(traces)
    assert list(X.columns) == ALL_FEATURE_ORDER          # exactly the model's inputs
    y = traces["failure"].to_numpy()
    aucs = {}
    for c in X.columns:
        if X[c].nunique() < 2:
            continue
        a = roc_auc_score(y, X[c])
        aucs[c] = max(a, 1 - a)
    worst_feat = max(aucs, key=aucs.get)
    assert aucs[worst_feat] < 0.85, f"single-feature leak: {worst_feat} AUC {aucs[worst_feat]:.3f}"


def test_derived_aggregates_are_consistent(df):
    # max_consecutive_retries counts consecutive retry-eligible error STEPS (one per step),
    # so it is bounded by num_steps — NOT by num_retries (a step can log an error with a
    # Poisson(0) retry draw, so the streak can exceed the total retry count).
    assert (df["max_consecutive_retries"] <= df["num_steps"]).all()
    assert (df["max_consecutive_retries"] >= 0).all()
    assert (df["num_retries"] >= 0).all()
    assert df["tool_error_rate"].between(0, 1).all()
    assert (df["tool_error_count"] <= df["num_tool_calls"]).all()
    assert df["context_max_pct"].between(0, 1).all()
    assert (df["context_mean_pct"] <= df["context_max_pct"] + 1e-9).all()


def test_exogenous_failures_are_telemetry_light(df):
    """early_exogenous = the catastrophic-early-death channel: it breaks at step 3..7, so
    these runs are short by construction (they overlap with quick successes — the
    irreducible error)."""
    exo = df[df.failure_reason == "early_exogenous"]
    assert len(exo) > 0
    assert exo["num_steps"].max() <= 7
    # they fail with low accumulated context (the dashboard alarm can't see them)
    assert exo["context_max_pct"].mean() < df["context_max_pct"].mean()


def test_headline_failures_below_context_threshold(df):
    """The project headline: the majority of failures occur below context 0.80, invisible
    to the industry `context > 0.80` rule."""
    fails = df[df.failure == 1]
    frac_below = (fails["context_max_pct"] < 0.80).mean()
    assert frac_below > 0.60, f"only {frac_below:.0%} of failures below ctx 0.80"
