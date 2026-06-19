"""
Frontier-LLM head-to-head harness for the AI-Agent Failure Predictor (Phase 5).

One agent run = one classification: given the run's telemetry, predict whether it
ultimately FAILED. We send a compact, human-readable telemetry description to each
frontier model via its local CLI (zero-shot, no fine-tuning) and compare against the
calibrated CatBoost champion on the SAME stratified test sample.

Design mirrors the proven Fraud-Detection-System harness (`src/mark_phase5_*`):
  * deterministic stratified sample (cached to JSON for reproducibility),
  * append-after-every-call caching so a partial run is recoverable / idempotent,
  * defensive parsing (label on line 1, failure-probability on line 2),
  * per-(llm, model) cache files so several models can warm concurrently without
    racing on a single JSON.

This module is reusable infrastructure (like `data_pipeline.py` / `utils.py`); the
actual experiment — sampling, scoring, head-to-head tables, plots — lives in
`notebooks/phase5_advanced.ipynb`, which imports these functions.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# CLI resolution (macOS / Linux). Resolve once at import; fall back to known install paths.
# ----------------------------------------------------------------------------------------
def _resolve(cmd: str, fallbacks: list[str]) -> str:
    found = shutil.which(cmd)
    if found:
        return found
    for f in fallbacks:
        if os.path.exists(f):
            return f
    return cmd  # last resort — let subprocess raise a clear error


CLAUDE_CMD = _resolve("claude", [os.path.expanduser("~/.local/bin/claude")])
CODEX_CMD = _resolve("codex", [
    os.path.expanduser("~/.nvm/versions/node/v24.13.0/bin/codex"),
    "/opt/homebrew/bin/codex",
])

# ----------------------------------------------------------------------------------------
# Prompt + feature formatting
# ----------------------------------------------------------------------------------------
LLM_PROMPT_TEMPLATE = (
    "You are an AI-agent reliability analyst. Given telemetry from ONE autonomous "
    "LLM-agent run, predict whether the run ultimately FAILED to complete its task.\n"
    "Reply with EXACTLY one word on the first line: FAIL or PASS.\n"
    "Then on a new line: a single number 0.0-1.0 = probability the run FAILED. "
    "No explanation, no other text.\n\n"
    "Run telemetry:\n{features}"
)

# Raw aggregate telemetry an operator would actually see on a dashboard (no engineered
# interaction terms — those are the champion's private advantage; the LLM reasons over
# the same observable signals a human SRE would).
_LLM_FEATURES = [
    ("task_type", "{}", lambda v: v),
    ("model_tier", "{}", lambda v: v),
    ("num_steps", "{:d}", int),
    ("context_max_pct", "{:.3f}", float),
    ("context_mean_pct", "{:.3f}", float),
    ("context_growth_rate", "{:.4f}", float),
    ("max_tool_depth", "{:d}", int),
    ("num_tool_calls", "{:d}", int),
    ("tool_error_count", "{:d}", int),
    ("tool_error_rate", "{:.3f}", float),
    ("num_retries", "{:d}", int),
    ("max_consecutive_retries", "{:d}", int),
    ("error_count_subtotal", "{:d}", int),
    ("reasoning_loop_count", "{:d}", int),
    ("error_rate_per_step", "{:.3f}", float),
    ("tokens_per_step_growth", "{:.3f}", float),
    ("mean_step_latency_ms", "{:.0f}", float),
    ("distinct_tools_used", "{:d}", int),
    ("temperature", "{:.2f}", float),
    ("prompt_tokens", "{:.0f}", float),
]


def format_run_for_llm(row: dict) -> str:
    """Compact bulleted telemetry description for one agent run."""
    lines = []
    for key, fmt, cast in _LLM_FEATURES:
        try:
            lines.append(f"  - {key}=" + fmt.format(cast(row[key])))
        except (KeyError, ValueError, TypeError):
            lines.append(f"  - {key}={row.get(key, '?')}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# CLI callers — return (text, latency_seconds); "__ERROR__:<reason>" on failure.
# ----------------------------------------------------------------------------------------
def call_claude(prompt: str, model: str = "haiku", timeout: float = 120.0) -> tuple[str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [CLAUDE_CMD, "--print", "--model", model, "--no-session-persistence",
             "--disable-slash-commands"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return f"__ERROR__:rc={proc.returncode}:{proc.stderr[:200]}", elapsed
        return proc.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "__ERROR__:timeout", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__:exc:{type(e).__name__}:{str(e)[:200]}", time.time() - t0


def call_codex(prompt: str, timeout: float = 200.0) -> tuple[str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [CODEX_CMD, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return f"__ERROR__:rc={proc.returncode}:{proc.stderr[:200]}", elapsed
        out = proc.stdout
        # codex output wraps the response in session metadata; slice the last response
        # block between the final "codex\n" marker and the "tokens used" footer.
        if "codex\n" in out:
            tail = out.rsplit("codex\n", 1)[1]
            if "tokens used" in tail:
                tail = tail.split("tokens used")[0]
            return tail.strip(), elapsed
        return out.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "__ERROR__:timeout", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__:exc:{type(e).__name__}:{str(e)[:200]}", time.time() - t0


def parse_llm_response(text: str) -> tuple[Optional[int], Optional[float]]:
    """Extract (label, failure_probability). label 1=FAIL, 0=PASS, None=parse error."""
    if not text or text.startswith("__ERROR__"):
        return None, None
    import re
    upper = text.upper()
    first = upper.split("\n")[0]
    label = None
    if "FAIL" in first:
        label = 1
    elif "PASS" in first:
        label = 0
    elif "FAIL" in upper:
        label = 1
    elif "PASS" in upper:
        label = 0

    prob = None
    for ln in text.split("\n"):
        m = re.search(r"\b(0?\.\d+|1\.0+|0\.0+|0|1)\b", ln.strip())
        if m:
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    prob = v
                    break
            except ValueError:
                pass
    if prob is None and label is not None:
        prob = 0.85 if label == 1 else 0.15
    return label, prob


# ----------------------------------------------------------------------------------------
# Split + stratified-sample reconstruction (identical to Phase 2-4)
# ----------------------------------------------------------------------------------------
NUMERIC = ["num_steps", "context_max_pct", "context_mean_pct", "context_growth_rate",
           "max_tool_depth", "num_tool_calls", "tool_error_count", "tool_error_rate",
           "num_retries", "max_consecutive_retries", "error_count_subtotal",
           "reasoning_loop_count", "tool_calls_per_step", "error_rate_per_step",
           "tokens_per_step_mean", "tokens_per_step_growth", "mean_step_latency_ms",
           "distinct_tools_used", "temperature", "prompt_tokens"]
CATEG = ["task_type", "model_tier"]
TARGET = "failure"


def load_test_and_sample(root: str, n: int = 50, seed: int = 42):
    """Reconstruct the Phase-2 test split, attach the calibrated champion's cached test
    probabilities, build a deterministic stratified n-row sample (n/2 fail + n/2 pass),
    and persist the sample indices. Returns (test_df, yte, champ_proba, sample_idx).

    test_df is reset-index (position-aligned with yte and champ_proba); sample_idx are
    positions into that test_df, so the notebook and the CLI warmer agree exactly.
    """
    from sklearn.model_selection import train_test_split

    results = os.path.join(root, "results")
    df = pd.read_parquet(os.path.join(root, "data", "processed", "agent_runs.parquet")).reset_index(drop=True)
    X0 = pd.get_dummies(df[NUMERIC + CATEG], columns=CATEG, drop_first=True)
    y = df[TARGET].values
    Xtr0, Xte0, ytr, yte = train_test_split(X0, y, test_size=0.25, random_state=seed, stratify=y)
    TE_IDX = Xte0.index
    cached = np.load(os.path.join(results, "phase2_test_idx.npy"))
    assert np.array_equal(TE_IDX.to_numpy(), cached), "split drifted from Phase-2 cache!"

    test_df = df.loc[TE_IDX].reset_index(drop=True)
    yte = np.asarray(yte)
    champ_proba = np.load(os.path.join(results, "phase4_champion_test_proba.npy"))
    assert len(champ_proba) == len(test_df), "champion proba length mismatch"

    half = n // 2
    rng = np.random.default_rng(seed)
    pos = np.where(yte == 1)[0]
    neg = np.where(yte == 0)[0]
    samp = np.concatenate([rng.choice(pos, half, replace=False),
                           rng.choice(neg, n - half, replace=False)])
    rng.shuffle(samp)
    sample_idx = samp.astype(int)

    cache_dir = Path(results) / "phase5_llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    json.dump(sample_idx.tolist(), open(cache_dir / "llm_sample_idx.json", "w"))
    return test_df, yte, champ_proba, sample_idx


# ----------------------------------------------------------------------------------------
# Eval driver (cached, idempotent)
# ----------------------------------------------------------------------------------------
def run_llm_eval(test_df: pd.DataFrame, sample_idx, cache_path, llm: str = "claude",
                 model: str = "haiku", verbose: bool = True) -> pd.DataFrame:
    """Classify each sampled run with one LLM. Cache to JSON, appending after every call
    so a partial / interrupted run resumes without re-billing completed rows."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = json.load(open(cache_path)) if cache_path.exists() else []
    seen = {(c["llm"], c["model"], c["test_idx"]) for c in cached}
    out_rows = list(cached)

    for i, idx in enumerate(sample_idx):
        key = (llm, model, int(idx))
        if key in seen:
            continue
        row = test_df.iloc[int(idx)].to_dict()
        prompt = LLM_PROMPT_TEMPLATE.format(features=format_run_for_llm(row))
        if llm == "claude":
            text, elapsed = call_claude(prompt, model=model)
        elif llm == "codex":
            text, elapsed = call_codex(prompt)
        else:
            raise ValueError(f"unknown llm: {llm}")
        label, prob = parse_llm_response(text)
        out_rows.append(dict(
            llm=llm, model=model, test_idx=int(idx),
            true_label=int(row[TARGET]), pred_label=label, pred_prob=prob,
            latency_s=round(elapsed, 2), raw=(text[:300] if text else ""),
        ))
        json.dump(out_rows, open(cache_path, "w"), indent=1)
        if verbose and (i + 1) % 5 == 0:
            print(f"  [{llm}/{model}] {i+1}/{len(sample_idx)} (last {elapsed:.1f}s)", flush=True)
    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    # Thin cache-warmer so several models can be warmed concurrently in the background
    # before the notebook runs (the notebook calls run_llm_eval again -> instant cache hit).
    import argparse
    ap = argparse.ArgumentParser(description="Warm the Phase-5 LLM head-to-head cache.")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--llm", required=True, choices=["claude", "codex"])
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()

    test_df, yte, champ, sample_idx = load_test_and_sample(a.root, n=a.n, seed=a.seed)
    tag = f"{a.llm}_{a.model}".replace("/", "_")
    cache = a.cache or os.path.join(a.root, "results", "phase5_llm_cache", f"llm_{tag}.json")
    print(f"[warm] {a.llm}/{a.model} -> {cache} · {len(sample_idx)} rows "
          f"(claude={CLAUDE_CMD}, codex={CODEX_CMD})", flush=True)
    t0 = time.time()
    run_llm_eval(test_df, sample_idx, cache, llm=a.llm, model=a.model)
    print(f"[warm] {a.llm}/{a.model} DONE in {time.time()-t0:.0f}s", flush=True)
