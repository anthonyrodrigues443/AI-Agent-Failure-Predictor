"""FastAPI serving layer for the AI-Agent-Failure-Predictor.

Wraps the same `src.predict` functions the Streamlit app and tests use, so the HTTP
surface scores runs with the *identical* featuriser + champion — no serving skew. The
champion is lazy-loaded on first use, so the app imports (and `/health` answers) even
before `python -m src.train` has written the artefacts.

Endpoints
  GET  /health              — liveness + whether the model artefacts are loaded
  POST /predict             — score a raw run record (aggregates + per-step traces)
  POST /predict/whatif      — score from interpretable sliders (synthesises the trace)
  GET  /model               — champion metadata (metric, threshold, op-point, prevalence)

Run:  uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from typing import Any

# Single-threaded numerics (shared-box OpenMP guard) before sklearn/catboost import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.feature_engineering import TRACE_COLS, synthesize_run  # noqa: E402

app = FastAPI(
    title="AI-Agent Failure Predictor",
    version="1.0.0",
    description="Calibrated failure-risk scoring for autonomous LLM-agent runs, with an "
                "early-window 'failure in N steps' alarm.",
)


# ---------------------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------------------
class WhatIfRequest(BaseModel):
    """Interpretable sliders — the trace is synthesised deterministically (a labelled
    hypothetical, not sampled telemetry)."""
    num_steps: int = Field(10, ge=3, le=45)
    task_type: str = Field("deep_research")
    model_tier: str = Field("small")
    tool_error_rate: float = Field(0.4, ge=0.0, le=1.0)
    max_consecutive_retries: int = Field(2, ge=0, le=10)
    context_max_pct: float = Field(0.6, ge=0.0, le=1.0)
    reasoning_loops: int = Field(1, ge=0, le=20)
    prompt_tokens: float = Field(600.0, ge=0.0)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class RunRecord(BaseModel):
    """A raw run: the run-level aggregates AND the per-step traces (lists, length =
    num_steps). Same shape as a record in results/ui_examples.json."""
    run: dict[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        run = dict(self.run)
        missing = [tc for tc in TRACE_COLS if tc not in run]
        if missing:
            raise HTTPException(422, f"run is missing per-step traces: {missing}")
        if "num_steps" not in run:
            raise HTTPException(422, "run is missing 'num_steps'")
        bad = [tc for tc in TRACE_COLS if len(run[tc]) != run["num_steps"]]
        if bad:
            raise HTTPException(422, f"trace length != num_steps for: {bad}")
        return run


class Prediction(BaseModel):
    failure_probability: float
    predicted_failure: bool
    risk_band: str
    threshold: float | None
    top_factors: list[dict[str, Any]]
    early_warning: dict[str, Any]


# ---------------------------------------------------------------------------------------
# Lazy model access — keep import + /health cheap and artefact-free.
# ---------------------------------------------------------------------------------------
def _champion():
    from src.predict import load_champion
    try:
        return load_champion()
    except FileNotFoundError as e:
        raise HTTPException(503, f"model not loaded — run `python -m src.train`. ({e})")


def _score(run: dict) -> Prediction:
    from src.predict import predict_run, explain_run, early_warning_lead
    champ = _champion()
    out = predict_run(run, champ)
    exp = explain_run(run, champ, top_n=5)
    factors = (
        [{"group": g["group"], "shap": round(g["shap"], 4)} for g in exp["groups"][:4]]
        if exp.get("available") else []
    )
    lead = early_warning_lead(run, alert_prob=0.5)
    early = {
        "alerted": lead["alerted"],
        "alert_step": lead.get("alert_step"),
        "steps_early": lead.get("steps_early"),
        "n_steps": lead.get("n_steps"),
    }
    return Prediction(
        failure_probability=round(out["failure_probability"], 4),
        predicted_failure=out["predicted_failure"],
        risk_band=out["risk_band"],
        threshold=out["threshold"],
        top_factors=factors,
        early_warning=early,
    )


# ---------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    loaded = os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "champion.joblib"))
    return {"status": "ok", "model_loaded": loaded, "service": "agent-failure-predictor"}


@app.get("/model")
def model_meta() -> dict:
    champ = _champion()
    return {
        "champion": champ["champion"],
        "primary_metric": champ["primary_metric"],
        "test_metrics": champ["test_metrics"],
        "threshold": champ["threshold"],
        "operating_point": champ["operating_point"],
        "prevalence": champ["prevalence"],
        "n_features": len(champ["features"]),
    }


@app.post("/predict", response_model=Prediction)
def predict(req: RunRecord) -> Prediction:
    return _score(req.as_mapping())


@app.post("/predict/whatif", response_model=Prediction)
def predict_whatif(req: WhatIfRequest) -> Prediction:
    run = synthesize_run(
        num_steps=req.num_steps, task_type=req.task_type, model_tier=req.model_tier,
        tool_error_rate=req.tool_error_rate, max_consecutive_retries=req.max_consecutive_retries,
        context_max_pct=req.context_max_pct, reasoning_loops=req.reasoning_loops,
        prompt_tokens=req.prompt_tokens, temperature=req.temperature,
    )
    return _score(run)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
