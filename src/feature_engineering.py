"""Canonical feature engineering for the AI-Agent-Failure-Predictor.

This is the single source of truth for the 49-feature `+ALL` representation the
Phase-4 champion (sigmoid-calibrated, Optuna-tuned CatBoost) was trained on. The exact
same functions are imported by `train.py`, `predict.py`, and `app.py`, so a run is
featurised identically at train and inference time — no notebook/serving skew.

Feature blocks (49 total, fixed order — see ALL_FEATURE_ORDER):
  * BASE   (26): 20 run-level telemetry aggregates + 6 one-hot dummies (task/model).
  * LEAD   (16): leading-indicator trajectory features (Phase 3). The signal lives in
                 the *shape* of the run (velocity/accel/early-warning), not the endpoint.
  * DOM     (7): explicit domain interactions a medicinal-... er, an agent-ops engineer
                 would hand-pick (ctx x depth, retry x cascade, ...).

A "run record" is a dict / row carrying both the aggregates AND the per-step traces
(`trace_ctx_pct`, `trace_tokens`, `trace_latency`, `trace_err`, `trace_retry`,
`trace_tool`, `trace_loop`) produced by `data_pipeline.generate_traces`.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

EPS = 1e-9

# ---------------------------------------------------------------------------------------
# Schema — the canonical column order the champion expects. Reindexing every assembled
# frame to this list makes single-row inference robust to absent dummy categories.
# ---------------------------------------------------------------------------------------
BASE_NUMERIC = [
    "num_steps", "context_max_pct", "context_mean_pct", "context_growth_rate", "max_tool_depth",
    "num_tool_calls", "tool_error_count", "tool_error_rate", "num_retries", "max_consecutive_retries",
    "error_count_subtotal", "reasoning_loop_count", "tool_calls_per_step", "error_rate_per_step",
    "tokens_per_step_mean", "tokens_per_step_growth", "mean_step_latency_ms", "distinct_tools_used",
    "temperature", "prompt_tokens",
]
BASE_CATEG = ["task_type", "model_tier"]

# get_dummies(drop_first=True) drops the alphabetically-first level of each category:
# task_type -> drops 'code_gen'; model_tier -> drops 'frontier'.
TASK_LEVELS = ["code_gen", "data_analysis", "deep_research", "multi_hop_qa", "web_navigation"]
TIER_LEVELS = ["frontier", "mid", "small"]
BASE_DUMMIES = (
    [f"task_type_{t}" for t in TASK_LEVELS[1:]]
    + [f"model_tier_{m}" for m in TIER_LEVELS[1:]]
)

LEAD_FEATURES = [
    "ctx_velocity_mean", "ctx_accel", "ctx_late_minus_early", "tokens_accel", "tokens_early_slope",
    "tokens_cv", "lat_p95_p50", "lat_slope", "err_slope", "err_var", "err_lag1ac",
    "err_late_minus_early", "retry_burst", "retry_step_frac", "time_to_first_err", "loop_late_frac",
]
DOM_FEATURES = [
    "ix_ctx_depth", "ix_retry_casc", "err_x_depth", "retries_per_toolcall",
    "toolerr_x_ctx", "loop_x_ctx", "depth_x_steps",
]

ALL_FEATURE_ORDER = BASE_NUMERIC + BASE_DUMMIES + LEAD_FEATURES + DOM_FEATURES  # 49

# Feature -> human group, used by the UI to attribute risk to a telemetry family. The
# Error/Retry/Loop family is the load-bearing one (Phase-5 ablation: -0.0222 AUPRC when
# dropped, 6x the next family).
FEATURE_GROUPS = {
    "Error / Retry / Loop": [
        "tool_error_count", "tool_error_rate", "num_retries", "max_consecutive_retries",
        "error_count_subtotal", "reasoning_loop_count", "error_rate_per_step",
        "err_slope", "err_var", "err_lag1ac", "err_late_minus_early", "retry_burst",
        "retry_step_frac", "time_to_first_err", "loop_late_frac",
        "ix_retry_casc", "err_x_depth", "retries_per_toolcall", "loop_x_ctx",
    ],
    "Context pressure": [
        "context_max_pct", "context_mean_pct", "context_growth_rate",
        "ctx_velocity_mean", "ctx_accel", "ctx_late_minus_early",
        "ix_ctx_depth", "toolerr_x_ctx",
    ],
    "Tool depth / chaining": [
        "max_tool_depth", "num_tool_calls", "tool_calls_per_step", "distinct_tools_used",
        "depth_x_steps",
    ],
    "Token / latency dynamics": [
        "tokens_per_step_mean", "tokens_per_step_growth", "tokens_accel", "tokens_early_slope",
        "tokens_cv", "mean_step_latency_ms", "lat_p95_p50", "lat_slope",
    ],
    "Capability proxy (tier/task)": [
        "model_tier_mid", "model_tier_small",
        "task_type_data_analysis", "task_type_deep_research",
        "task_type_multi_hop_qa", "task_type_web_navigation",
        "temperature",
    ],
    "Run shape": ["num_steps", "prompt_tokens"],
}

TRACE_COLS = ["trace_ctx_pct", "trace_tokens", "trace_latency",
              "trace_err", "trace_retry", "trace_tool", "trace_loop"]


# ---------------------------------------------------------------------------------------
# Small numeric helpers (mirrors the Phase-3/4 notebook definitions exactly).
# ---------------------------------------------------------------------------------------
def _slope(a) -> float:
    a = np.asarray(a, float)
    n = len(a)
    return float(np.polyfit(np.arange(n), a, 1)[0]) if n >= 2 else 0.0


def _lag1ac(a) -> float:
    """Lag-1 autocorrelation — the 'critical slowing down' early-warning signal (ecology)."""
    a = np.asarray(a, float)
    if len(a) < 3:
        return 0.0
    x, z = a[:-1], a[1:]
    if x.std() < EPS or z.std() < EPS:
        return 0.0
    return float(np.corrcoef(x, z)[0, 1])


def _first(a, n=None):
    a = list(a)
    k = max(1, len(a) // 3) if n is None else min(n, len(a))
    return a[:k]


def _last(a):
    a = list(a)
    k = max(1, len(a) // 3)
    return a[-k:]


# ---------------------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------------------
def build_base(df: pd.DataFrame) -> pd.DataFrame:
    """20 numeric aggregates + 6 one-hot dummies, reindexed to the canonical order."""
    base = pd.get_dummies(df[BASE_NUMERIC + BASE_CATEG], columns=BASE_CATEG, drop_first=True)
    # Reindex so a single row missing a category still yields all dummy columns (= 0).
    return base.reindex(columns=BASE_NUMERIC + BASE_DUMMIES, fill_value=0).astype(float)


def _lead_row(r: Mapping) -> dict:
    ctx, tok, lat = list(r["trace_ctx_pct"]), list(r["trace_tokens"]), list(r["trace_latency"])
    err, ret, loop = list(r["trace_err"]), list(r["trace_retry"]), list(r["trace_loop"])
    dctx = np.diff(ctx) if len(ctx) > 1 else np.array([0.0])
    n = max(len(ctx), 1)
    ttfe = next((i for i, e in enumerate(err) if e), None)
    return {
        "ctx_velocity_mean": float(np.mean(dctx)),
        "ctx_accel": _slope(dctx),
        "ctx_late_minus_early": float(np.mean(_last(ctx)) - np.mean(_first(ctx))),
        "tokens_accel": _slope(np.diff(tok) if len(tok) > 1 else [0.0]),
        "tokens_early_slope": _slope(_first(tok)),
        "tokens_cv": float(np.std(tok) / (np.mean(tok) + EPS)),
        "lat_p95_p50": float(np.percentile(lat, 95) / (np.percentile(lat, 50) + EPS)) if lat else 1.0,
        "lat_slope": _slope(lat),
        "err_slope": _slope(err),
        "err_var": float(np.var(err)),
        "err_lag1ac": _lag1ac(err),
        "err_late_minus_early": float(np.mean(_last(err)) - np.mean(_first(err))),
        "retry_burst": r["max_consecutive_retries"] / (r["num_retries"] + 1.0),
        "retry_step_frac": float(np.mean([1 if x > 0 else 0 for x in ret])) if ret else 0.0,
        "time_to_first_err": (ttfe / n) if ttfe is not None else 1.0,
        "loop_late_frac": (sum(_last(loop)) / (sum(loop) + 1.0)),
    }


def build_lead(df_traces: pd.DataFrame) -> pd.DataFrame:
    rows = [_lead_row(r) for _, r in df_traces.iterrows()]
    return pd.DataFrame(rows, index=df_traces.index)[LEAD_FEATURES].astype(float)


def build_dom(df: pd.DataFrame) -> pd.DataFrame:
    ctx_clip = np.clip(df["context_max_pct"] - 0.75, 0, None) / 0.25
    depth_n = np.clip(df["max_tool_depth"], 0, 8) / 8.0
    retry_n = np.clip(df["max_consecutive_retries"], 0, 5) / 5.0
    out = pd.DataFrame({
        "ix_ctx_depth": ctx_clip * depth_n,
        "ix_retry_casc": retry_n * df["error_rate_per_step"],
        "err_x_depth": df["tool_error_rate"] * df["max_tool_depth"],
        "retries_per_toolcall": df["num_retries"] / (df["num_tool_calls"] + 1.0),
        "toolerr_x_ctx": df["tool_error_rate"] * df["context_max_pct"],
        "loop_x_ctx": df["reasoning_loop_count"] * df["context_max_pct"],
        "depth_x_steps": df["max_tool_depth"] * df["num_steps"],
    }, index=df.index)
    return out[DOM_FEATURES].astype(float)


def assemble_features(df_traces: pd.DataFrame) -> pd.DataFrame:
    """Full 49-feature `+ALL` matrix from a frame carrying aggregates + traces."""
    X = pd.concat([build_base(df_traces), build_lead(df_traces), build_dom(df_traces)], axis=1)
    return X.reindex(columns=ALL_FEATURE_ORDER, fill_value=0.0).astype(float)


# ---------------------------------------------------------------------------------------
# Early-window features — the "predict failure N steps before it happens" representation.
# Featurise using only the first k steps of every trace. Phase 3/4: k=3 recovers ~78% of
# the full-run AUPRC.
# ---------------------------------------------------------------------------------------
EW_FEATURES = [
    "ew_ctx_last", "ew_ctx_max", "ew_ctx_slope", "ew_tok_mean", "ew_tok_slope",
    "ew_lat_mean", "ew_lat_slope", "ew_err_count", "ew_err_rate", "ew_retry_count",
    "ew_tool_count", "ew_loop_count", "ew_retry_per_tool",
]
EW_START = ["prompt_tokens", "temperature"] + BASE_DUMMIES  # known at step 0
EW_FEATURE_ORDER = EW_FEATURES + EW_START


def _ew_row(r: Mapping, k: int) -> dict:
    ctx, tok, lat = list(r["trace_ctx_pct"]), list(r["trace_tokens"]), list(r["trace_latency"])
    err, ret, tool, loop = list(r["trace_err"]), list(r["trace_retry"]), list(r["trace_tool"]), list(r["trace_loop"])
    m = min(k, len(ctx))
    c, t, l = ctx[:m], tok[:m], lat[:m]
    e, rt, to, lp = err[:m], ret[:m], tool[:m], loop[:m]
    return {
        "ew_ctx_last": c[-1] if c else 0.0, "ew_ctx_max": max(c) if c else 0.0,
        "ew_ctx_slope": _slope(c), "ew_tok_mean": float(np.mean(t)) if t else 0.0,
        "ew_tok_slope": _slope(t), "ew_lat_mean": float(np.mean(l)) if l else 0.0,
        "ew_lat_slope": _slope(l), "ew_err_count": float(sum(e)),
        "ew_err_rate": sum(e) / max(m, 1), "ew_retry_count": float(sum(rt)),
        "ew_tool_count": float(sum(to)), "ew_loop_count": float(sum(lp)),
        "ew_retry_per_tool": sum(rt) / (sum(to) + 1.0),
    }


def early_window_features(df_traces: pd.DataFrame, k: int) -> pd.DataFrame:
    ew = pd.DataFrame([_ew_row(r, k) for _, r in df_traces.iterrows()], index=df_traces.index)
    start = pd.get_dummies(
        df_traces[["prompt_tokens", "temperature", "task_type", "model_tier"]],
        columns=["task_type", "model_tier"], drop_first=True,
    ).reindex(columns=EW_START, fill_value=0)
    out = pd.concat([ew, start], axis=1)
    return out.reindex(columns=EW_FEATURE_ORDER, fill_value=0.0).astype(float)


# ---------------------------------------------------------------------------------------
# Single-run convenience wrappers (used by predict.py / app.py).
# ---------------------------------------------------------------------------------------
def run_to_frame(run: Mapping) -> pd.DataFrame:
    """Wrap one run record (aggregates + traces) into a 1-row DataFrame."""
    return pd.DataFrame([dict(run)])


def featurize_one(run: Mapping) -> pd.DataFrame:
    return assemble_features(run_to_frame(run))


# ---------------------------------------------------------------------------------------
# Trace synthesiser — for the UI "what-if" builder. Given high-level knobs, lay down a
# plausible per-step trace consistent with those aggregates so the featuriser/model can
# score it. This is a deterministic schedule (NOT the stochastic simulator) and is clearly
# labelled in the UI as a hand-built hypothetical, not sampled telemetry.
# ---------------------------------------------------------------------------------------
CTX_BUDGET = {"small": 16000, "mid": 48000, "frontier": 160000}


def synthesize_run(num_steps: int, task_type: str, model_tier: str,
                   tool_error_rate: float, max_consecutive_retries: int,
                   context_max_pct: float, reasoning_loops: int,
                   prompt_tokens: float = 600.0, temperature: float = 0.7,
                   tool_calls_per_step: float = 0.6) -> dict:
    """Build a run record from interpretable sliders. Errors/retries are front-loaded into a
    burst (the realistic failure mode); context ramps smoothly toward the chosen ceiling."""
    n = int(np.clip(num_steps, 3, 45))
    step = np.arange(1, n + 1)

    # Context: smooth ramp from a low start to the chosen max (slight ease-out curve).
    ctx = context_max_pct * (1 - np.exp(-2.2 * step / n))
    ctx = (ctx / (ctx.max() + EPS)) * context_max_pct
    trace_ctx = list(np.round(ctx, 4))

    # Tool calls spread evenly; tokens grow gently with context pressure.
    num_tool_calls = int(round(tool_calls_per_step * n))
    tool_steps = set(np.linspace(0, n - 1, num=max(num_tool_calls, 0), dtype=int).tolist()) if num_tool_calls else set()
    trace_tool = [1 if i in tool_steps else 0 for i in range(n)]

    base_tok = prompt_tokens / 4 + 220
    trace_tokens = list(np.round(base_tok * (1 + 0.5 * ctx) + 30 * np.arange(n), 1))
    base_lat = 320.0
    trace_lat = list(np.round(base_lat + 0.05 * np.asarray(trace_tokens) + 12 * ctx * 100, 1))

    # Errors: a burst over the first `n_err` tool/reason steps, sized by tool_error_rate.
    n_err = int(round(tool_error_rate * max(num_tool_calls, 1)))
    err_steps = sorted(tool_steps)[:n_err] if tool_steps else list(range(min(n_err, n)))
    trace_err = [1 if i in set(err_steps) else 0 for i in range(n)]
    tool_error_count = sum(1 for i in err_steps if trace_tool[i])
    error_count_subtotal = sum(trace_err) - tool_error_count

    # Retries piled onto the first error steps up to the requested consecutive max.
    trace_retry = [0] * n
    rem = int(max_consecutive_retries)
    for i in err_steps:
        if rem <= 0:
            break
        trace_retry[i] = min(2, rem + 1)
        rem -= trace_retry[i]
    num_retries = int(sum(trace_retry))

    # Reasoning loops late in the run (where they actually occur).
    trace_loop = [0] * n
    for j in range(int(reasoning_loops)):
        idx = n - 1 - j
        if 0 <= idx < n:
            trace_loop[idx] = 1

    max_depth = 1 + int(round(2 * tool_error_rate)) + (max_consecutive_retries > 1)
    total_err = tool_error_count + error_count_subtotal
    return {
        "task_type": task_type, "model_tier": model_tier,
        "num_steps": n,
        "context_max_pct": float(max(trace_ctx)),
        "context_mean_pct": float(np.mean(trace_ctx)),
        "context_growth_rate": _slope(trace_ctx),
        "max_tool_depth": int(max_depth),
        "num_tool_calls": int(num_tool_calls),
        "tool_error_count": int(tool_error_count),
        "tool_error_rate": tool_error_count / max(num_tool_calls, 1),
        "num_retries": int(num_retries),
        "max_consecutive_retries": int(max_consecutive_retries),
        "error_count_subtotal": int(error_count_subtotal),
        "reasoning_loop_count": int(reasoning_loops),
        "tool_calls_per_step": num_tool_calls / max(n, 1),
        "error_rate_per_step": total_err / max(n, 1),
        "tokens_per_step_mean": float(np.mean(trace_tokens)),
        "tokens_per_step_growth": float(np.mean(trace_tokens[-max(1, n // 3):]) /
                                        (np.mean(trace_tokens[:max(1, n // 3)]) + EPS)),
        "mean_step_latency_ms": float(np.mean(trace_lat)),
        "distinct_tools_used": int(min(num_tool_calls, 6)),
        "temperature": float(temperature),
        "prompt_tokens": float(prompt_tokens),
        "trace_ctx_pct": trace_ctx, "trace_tokens": trace_tokens, "trace_latency": trace_lat,
        "trace_err": trace_err, "trace_retry": trace_retry, "trace_tool": trace_tool,
        "trace_loop": trace_loop,
    }
