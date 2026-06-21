"""HTTP-surface tests for the FastAPI serving layer. /health needs no model; the scoring
endpoints are skipped until the champion is built (`python -m src.train`)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from src.serve import app  # noqa: E402
from src.feature_engineering import synthesize_run  # noqa: E402

client = TestClient(app)

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
needs_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS, "champion.joblib")),
    reason="champion.joblib not built — run `python -m src.train`")


def test_health_no_model_needed():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@needs_model
def test_model_meta():
    r = client.get("/model")
    assert r.status_code == 200
    meta = r.json()
    assert meta["n_features"] == 49
    assert meta["primary_metric"] == "average_precision"
    assert 0.0 < meta["threshold"] < 1.0


@needs_model
def test_whatif_trouble_scores_higher_than_clean():
    trouble = {"num_steps": 14, "task_type": "deep_research", "model_tier": "small",
               "tool_error_rate": 0.8, "max_consecutive_retries": 3,
               "context_max_pct": 0.95, "reasoning_loops": 2}
    clean = {"num_steps": 6, "task_type": "multi_hop_qa", "model_tier": "frontier",
             "tool_error_rate": 0.0, "max_consecutive_retries": 0,
             "context_max_pct": 0.2, "reasoning_loops": 0}
    pt = client.post("/predict/whatif", json=trouble).json()
    pc = client.post("/predict/whatif", json=clean).json()
    assert pt["failure_probability"] > pc["failure_probability"]
    assert pt["risk_band"] in {"Low", "Elevated", "High", "Critical"}
    assert 0.0 <= pt["failure_probability"] <= 1.0


@needs_model
def test_predict_raw_run():
    run = synthesize_run(12, "code_gen", "mid", 0.5, 2, 0.7, 1)
    r = client.post("/predict", json={"run": run})
    assert r.status_code == 200
    body = r.json()
    assert {"failure_probability", "predicted_failure", "risk_band", "top_factors"}.issubset(body)


@needs_model
def test_predict_rejects_run_without_traces():
    r = client.post("/predict", json={"run": {"num_steps": 5, "task_type": "code_gen"}})
    assert r.status_code == 422


def test_whatif_validation_rejects_out_of_range():
    # context_max_pct > 1.0 must be rejected by the pydantic schema (422), model or not
    r = client.post("/predict/whatif", json={"context_max_pct": 5.0})
    assert r.status_code == 422
