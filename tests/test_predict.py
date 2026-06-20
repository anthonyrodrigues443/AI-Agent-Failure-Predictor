"""Inference-pipeline tests. Skipped if the model artefacts aren't built yet
(`python -m src.train`); CI without the gitignored artefacts still collects cleanly."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import synthesize_run  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
needs_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS, "champion.joblib")),
    reason="champion.joblib not built — run `python -m src.train`")


def _trouble_run():
    return synthesize_run(14, "deep_research", "small", 0.7, 3, 0.95, 2)


def _clean_run():
    return synthesize_run(6, "multi_hop_qa", "frontier", 0.0, 0, 0.2, 0)


@needs_model
def test_predict_run_keys_and_range():
    from src.predict import predict_run
    out = predict_run(_trouble_run())
    assert set(out) >= {"failure_probability", "threshold", "predicted_failure", "risk_band"}
    assert 0.0 <= out["failure_probability"] <= 1.0
    assert out["risk_band"] in {"Low", "Elevated", "High", "Critical"}


@needs_model
def test_trouble_scores_higher_than_clean():
    from src.predict import predict_run
    assert (predict_run(_trouble_run())["failure_probability"]
            > predict_run(_clean_run())["failure_probability"])


@needs_model
def test_predict_batch_shape():
    from src.predict import predict_batch
    df = pd.DataFrame([_trouble_run(), _clean_run()])
    res = predict_batch(df)
    assert len(res) == 2
    assert {"failure_probability", "predicted_failure", "risk_band"}.issubset(res.columns)
    assert res["failure_probability"].between(0, 1).all()


@needs_model
def test_explain_run_groups_sum_present():
    from src.predict import explain_run
    exp = explain_run(_trouble_run(), top_n=5)
    assert exp["available"]
    assert len(exp["features"]) == 5
    groups = {g["group"] for g in exp["groups"]}
    assert "Error / Retry / Loop" in groups


@needs_model
def test_early_window_curve_monotone_length():
    from src.predict import early_window_curve, early_warning_lead
    run = _trouble_run()
    curve = early_window_curve(run)
    ks = [p["k"] for p in curve["points"]]
    assert ks == sorted(ks)                       # increasing k
    assert all(0.0 <= p["prob"] <= 1.0 for p in curve["points"])
    lead = early_warning_lead(run, alert_prob=0.5)
    if lead["alerted"]:
        assert lead["steps_early"] == lead["n_steps"] - lead["alert_step"]
