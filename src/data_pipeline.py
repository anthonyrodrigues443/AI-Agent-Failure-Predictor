"""
Causal, literature-calibrated simulator of AI-agent run telemetry.

One row = one agent run. Design principles (and the bugs they avoid):

  * Run LENGTH is decoupled from OUTCOME. Every run executes ~Poisson(task-horizon) steps
    regardless of whether it will ultimately be labelled a failure. (If length were "run
    until the plan is complete", then step-count features would trivially encode the label —
    a synthetic-leakage trap.)
  * The OUTCOME is a NOISY function of accumulated "trouble" telemetry PLUS latent factors
    that are NOT observable in telemetry (difficulty - competence) PLUS irreducible Bernoulli
    noise. So two runs with identical telemetry can land on different labels — this caps the
    Bayes-optimal AUPRC well below 1.0, as in real failure-prediction problems.
  * An exogenous early-failure channel (misconfigured tool / API outage / one catastrophic
    error) produces telemetry-LIGHT failures that overlap with quick successes.

Calibration anchors (see data/README.md for sources):
  - MAST taxonomy: most failures are tool/coordination driven, BEFORE context saturates.
  - ~31% of production failures involve tool misuse / wrong args (often upfront).
  - Context-retention degrades 15-30% past ~10 turns -> error prob ramps once ctx_pct > 0.7.
  - Cascading corruption: a bad tool call at step N poisons context for N+1..end.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Task / model profiles
# ----------------------------------------------------------------------------------------
TASK_PROFILES = {
    "code_gen":       dict(tool_intensity=0.55, chain_lambda=1.0, horizon=9,  diff_offset=0.10),
    "deep_research":  dict(tool_intensity=0.72, chain_lambda=1.5, horizon=13, diff_offset=0.05),
    "data_analysis":  dict(tool_intensity=0.50, chain_lambda=0.8, horizon=8,  diff_offset=0.00),
    "web_navigation": dict(tool_intensity=0.68, chain_lambda=1.8, horizon=10, diff_offset=0.12),
    "multi_hop_qa":   dict(tool_intensity=0.30, chain_lambda=0.6, horizon=6,  diff_offset=0.08),
}
TASK_TYPES = list(TASK_PROFILES)
TASK_WEIGHTS = np.array([0.24, 0.20, 0.20, 0.18, 0.18])

MODEL_PROFILES = {
    "small":    dict(competence=0.66, ctx_budget=16000,  plan_skill=0.48),
    "mid":      dict(competence=0.77, ctx_budget=48000,  plan_skill=0.66),
    "frontier": dict(competence=0.87, ctx_budget=160000, plan_skill=0.83),
}
MODEL_TIERS = list(MODEL_PROFILES)
TIER_WEIGHTS = np.array([0.30, 0.40, 0.30])

MAX_STEPS = 45
_TOOL_NAMES = ["search", "browser", "python", "sql", "file_io", "api_call",
               "calculator", "retriever", "shell", "vision"]

# Outcome model — logit weights on accumulated trouble. Tuned so overall failure ~0.24 and
# the best achievable AUPRC sits ~0.82-0.88 (room for Phase 2+ to beat the linear floor).
# Most weight sits on OBSERVED trouble (err/retry/ctx) so a good model can reach a high
# ceiling; `latent` and `noise` are the irreducible part; the two `ix_*` interactions are
# nonlinear, so tree ensembles can beat a linear model (the Phase-2 hypothesis).
_W = dict(b0=-3.70, err=2.3, retry=1.45, ctx=2.0, casc=2.0, loop=1.0,
          ix_ctxdepth=1.6, ix_retrycasc=2.0, latent=1.1, noise_sd=0.70)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


@dataclass
class _RunAgg:
    task_type: str
    model_tier: str
    ctx_budget: float
    temperature: float
    prompt_tokens: float
    num_steps: int = 0
    num_tool_calls: int = 0
    tool_error_count: int = 0
    num_retries: int = 0
    max_consecutive_retries: int = 0
    error_count_subtotal: int = 0
    reasoning_loop_count: int = 0
    max_tool_depth: int = 0
    distinct_tools: set = field(default_factory=set)
    ctx_pct_trace: list = field(default_factory=list)
    tokens_per_step: list = field(default_factory=list)
    latency_trace: list = field(default_factory=list)
    # Per-step event traces (Phase 3). Recording these consumes NO rng draws, so the
    # aggregate columns produced by generate_dataset() are byte-identical with/without them.
    step_tool_trace: list = field(default_factory=list)   # 1 if this step issued a tool call
    step_err_trace: list = field(default_factory=list)     # 1 if this step had any error
    step_retry_trace: list = field(default_factory=list)   # # retries fired this step
    step_loop_trace: list = field(default_factory=list)    # 1 if a reasoning loop fired
    failure: int = 0
    failure_reason: str = "none"


def _simulate_run(rng: np.random.Generator) -> _RunAgg:
    task_type = rng.choice(TASK_TYPES, p=TASK_WEIGHTS)
    model_tier = rng.choice(MODEL_TIERS, p=TIER_WEIGHTS)
    tp, mp = TASK_PROFILES[task_type], MODEL_PROFILES[model_tier]

    difficulty = float(np.clip(rng.beta(2.2, 3.0) + tp["diff_offset"], 0.02, 0.98))
    competence = float(np.clip(mp["competence"] - 0.36 * difficulty + rng.normal(0, 0.05), 0.05, 0.99))
    plan_skill = float(np.clip(mp["plan_skill"] + rng.normal(0, 0.08), 0.05, 0.99))
    ctx_budget = mp["ctx_budget"]
    prompt_tokens = float(np.clip(rng.lognormal(6.4, 0.6), 80, 8000))
    temperature = float(np.clip(rng.normal(0.7, 0.3), 0.0, 1.5))

    agg = _RunAgg(task_type, model_tier, ctx_budget, temperature, prompt_tokens)

    # Run length ~ task horizon, INDEPENDENT of the eventual outcome.
    n_target = int(np.clip(rng.poisson(tp["horizon"]) * (0.85 + 0.3 * rng.random()), 3, MAX_STEPS))
    # Exogenous early death -> telemetry-light failure.
    p_run_exo = float(np.clip(0.035 + 0.05 * difficulty + 0.04 * (1 - competence), 0.01, 0.20))
    # exo deaths start at step 3 (= the minimum a successful run can have), so no num_steps
    # value is deterministically a failure (avoids an outcome->num_steps truncation leak).
    exo_step = int(rng.integers(3, 8)) if rng.random() < p_run_exo else 10 ** 9

    context_tokens = prompt_tokens
    cascade = 0.0
    consecutive_retries = 0
    base_latency = float(rng.uniform(180, 520))
    died_exo = False

    for step in range(1, n_target + 1):
        agg.num_steps = step
        ctx_pct = min(context_tokens / ctx_budget, 1.0)
        ctx_pressure = _sigmoid((ctx_pct - 0.72) / 0.08)
        step_tokens = float(rng.lognormal(5.4, 0.45))
        s_tool = s_err = s_retry = s_loop = 0   # per-step event flags (Phase 3 traces)

        if rng.random() < tp["tool_intensity"]:
            s_tool = 1
            agg.num_tool_calls += 1
            depth = 1 + rng.poisson(tp["chain_lambda"] * (1.0 + 0.8 * cascade))
            agg.max_tool_depth = max(agg.max_tool_depth, depth)
            agg.distinct_tools.add(rng.choice(_TOOL_NAMES))
            step_tokens += depth * float(rng.lognormal(5.7, 0.45))
            p_tool_fail = np.clip(
                (1.0 - competence) + 0.07 * (depth - 1) + 0.55 * cascade
                + 0.30 * ctx_pressure - 0.06, 0.01, 0.97)
            if rng.random() < p_tool_fail:
                s_err = 1
                agg.tool_error_count += 1
                n_retry = int(min(rng.poisson(0.6 + 1.4 * (1.0 - plan_skill)), 4))
                s_retry = n_retry
                agg.num_retries += n_retry
                consecutive_retries += 1
                agg.max_consecutive_retries = max(agg.max_consecutive_retries, consecutive_retries)
                step_tokens += n_retry * float(rng.lognormal(5.5, 0.4))
                if rng.random() < (0.35 + 0.30 * (1.0 - plan_skill)):
                    cascade = min(cascade + 0.05 + 0.03 * (depth - 1) * (1.0 - plan_skill), 0.9)
            else:
                consecutive_retries = 0
        else:
            p_reason_err = np.clip(
                (1.0 - competence) * 0.6 + 0.50 * cascade + 0.40 * ctx_pressure - 0.04, 0.01, 0.95)
            if rng.random() < p_reason_err:
                s_err = 1
                agg.error_count_subtotal += 1
                if rng.random() < (0.25 + 0.45 * ctx_pressure + 0.30 * cascade):
                    s_loop = 1
                    agg.reasoning_loop_count += 1
                cascade = min(cascade + 0.02, 0.9)
            else:
                consecutive_retries = 0

        context_tokens += step_tokens
        agg.ctx_pct_trace.append(min(context_tokens / ctx_budget, 1.0))
        agg.tokens_per_step.append(step_tokens)
        agg.latency_trace.append(base_latency + 0.04 * step_tokens + rng.normal(0, 90))
        agg.step_tool_trace.append(s_tool)
        agg.step_err_trace.append(s_err)
        agg.step_retry_trace.append(s_retry)
        agg.step_loop_trace.append(s_loop)

        if step >= exo_step:
            agg.failure, agg.failure_reason, died_exo = 1, "early_exogenous", True
            break

    if not died_exo:
        # Noisy outcome from accumulated trouble + latent (unobserved) factors + Bernoulli noise.
        ctx_max = max(agg.ctx_pct_trace) if agg.ctx_pct_trace else 0.0
        t_err = agg.tool_error_count / max(agg.num_tool_calls, 1)
        t_retry = agg.max_consecutive_retries / 5.0
        t_ctx = max(0.0, (ctx_max - 0.75) / 0.25)
        t_casc = cascade
        t_loop = agg.reasoning_loop_count / 4.0
        depth_norm = min(agg.max_tool_depth, 8) / 8.0
        ix_cd = _W["ix_ctxdepth"] * t_ctx * depth_norm      # context bites only with deep chains
        ix_rc = _W["ix_retrycasc"] * t_retry * t_casc       # retries bite under active corruption
        latent = _W["latent"] * (difficulty - competence)   # NOT in telemetry -> irreducible
        noise = rng.normal(0, _W["noise_sd"])               # irreducible
        logit = (_W["b0"] + _W["err"] * t_err + _W["retry"] * t_retry + _W["ctx"] * t_ctx
                 + _W["casc"] * t_casc + _W["loop"] * t_loop + ix_cd + ix_rc + latent + noise)
        if rng.random() < _sigmoid(logit):
            agg.failure = 1
            channels = {"context_overflow": _W["ctx"] * t_ctx + ix_cd,
                        "stuck_retry_loop": _W["retry"] * t_retry,
                        "cascade_failure": _W["casc"] * t_casc + ix_rc,
                        "degenerate_loop": _W["loop"] * t_loop}
            top, val = max(channels.items(), key=lambda kv: kv[1])
            # low-trouble failure => capability gap the process telemetry never shows
            agg.failure_reason = top if val > 0.20 else "latent_capability"
    return agg


def _slope(y: list) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    return float(np.polyfit(np.arange(n), y, 1)[0])


def _aggregate(agg: _RunAgg) -> dict:
    ctx, tps, n = agg.ctx_pct_trace, agg.tokens_per_step, agg.num_steps
    first_third = tps[: max(1, n // 3)]
    last_third = tps[-max(1, n // 3):]
    tps_growth = (np.mean(last_third) / (np.mean(first_third) + 1e-9)) if tps else 1.0
    total_err = agg.tool_error_count + agg.error_count_subtotal
    return {
        "task_type": agg.task_type,
        "model_tier": agg.model_tier,
        "num_steps": n,
        "context_max_pct": float(max(ctx)) if ctx else 0.0,
        "context_mean_pct": float(np.mean(ctx)) if ctx else 0.0,
        "context_growth_rate": _slope(ctx),
        "max_tool_depth": agg.max_tool_depth,
        "num_tool_calls": agg.num_tool_calls,
        "tool_error_count": agg.tool_error_count,
        "tool_error_rate": agg.tool_error_count / max(agg.num_tool_calls, 1),
        "num_retries": agg.num_retries,
        "max_consecutive_retries": agg.max_consecutive_retries,
        "error_count_subtotal": agg.error_count_subtotal,
        "reasoning_loop_count": agg.reasoning_loop_count,
        "tool_calls_per_step": agg.num_tool_calls / max(n, 1),
        "error_rate_per_step": total_err / max(n, 1),
        "tokens_per_step_mean": float(np.mean(tps)) if tps else 0.0,
        "tokens_per_step_growth": float(tps_growth),
        "mean_step_latency_ms": float(np.mean(agg.latency_trace)) if agg.latency_trace else 0.0,
        "distinct_tools_used": len(agg.distinct_tools),
        "temperature": agg.temperature,
        "prompt_tokens": agg.prompt_tokens,
        "failure": agg.failure,
        "failure_reason": agg.failure_reason,   # EDA only — excluded from model inputs (would leak)
    }


def generate_dataset(n_runs: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Simulate `n_runs` agent runs and return a run-level telemetry DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([_aggregate(_simulate_run(rng)) for _ in range(n_runs)])


def generate_traces(n_runs: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Same runs as `generate_dataset` (identical rng order -> identical aggregates) but with
    the per-step event traces attached as list-columns, for Phase-3 leading-indicator feature
    engineering. The aggregate columns are guaranteed to match the committed parquet exactly."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_runs):
        agg = _simulate_run(rng)
        row = _aggregate(agg)
        row["trace_ctx_pct"] = list(agg.ctx_pct_trace)
        row["trace_tokens"] = list(agg.tokens_per_step)
        row["trace_latency"] = list(agg.latency_trace)
        row["trace_tool"] = list(agg.step_tool_trace)
        row["trace_err"] = list(agg.step_err_trace)
        row["trace_retry"] = list(agg.step_retry_trace)
        row["trace_loop"] = list(agg.step_loop_trace)
        rows.append(row)
    return pd.DataFrame(rows)


def build_and_save(n_runs: int = 20000, seed: int = 42,
                   out_path: str = "data/processed/agent_runs.parquet") -> pd.DataFrame:
    import os
    df = generate_dataset(n_runs, seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.replace(".parquet", ".csv")
        df.to_csv(out_path, index=False)
    print(f"[data_pipeline] wrote {len(df):,} runs -> {out_path} "
          f"(failure rate {df['failure'].mean():.3f})")
    return df


if __name__ == "__main__":
    build_and_save()
