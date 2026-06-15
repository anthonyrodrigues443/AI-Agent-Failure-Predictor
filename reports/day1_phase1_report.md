# Phase 1: Domain Research + Dataset + EDA + Baselines — AI-Agent Failure Predictor
**Date:** 2026-06-15
**Session:** 1 of 7

## Objective
Can run-level **telemetry** (context utilization, tool-chain depth, retries, error counts)
predict whether an autonomous LLM-agent run will **fail** — and does the alert rule that
every observability dashboard actually deploys (`context_usage > 80%`) work? Establish the
dataset, the primary metric, and the baseline floor for the rest of the week.

## Research & References
Domain research was done **before** touching data (sources below):
1. **MAST failure taxonomy** (1,600+ real multi-agent traces) — failures split *Specification
   41.8% / Coordination 36.9% / Verification 21.3%*. **Implication we tested:** context
   saturation is *not* the dominant failure cause; tool/coordination problems are.
2. **TRAIL** (Patronus, arXiv 2505.08638) — 148 human-annotated agent traces, 841 errors, a
   20+-category error taxonomy from GAIA + SWE-Bench. Confirms real failure-trace datasets are
   *small* and *trace-localization* shaped — not large tabular telemetry→outcome sets.
3. **Observability "12 signals that predict incidents"** + **Datadog State of AI Engineering
   2026** — the operational signal set (context saturation, tool errors, retries, tail latency,
   retrieval recall) and the 2026 reliability backdrop (>40% of agentic projects forecast
   cancelled by 2027 over cost/monitoring gaps).

**How research shaped the work:** because no large public *telemetry→failure* tabular dataset
exists, I built a **causal** simulator calibrated to (1)–(3) — failure *emerges* from latent
step-by-step dynamics rather than being assigned from a feature (leakage-checked). The literature
prediction "most failures fire before context saturates" became the central hypothesis to test.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 agent runs |
| Features | 20 numeric + 2 categorical telemetry signals |
| Target | `failure` (1 = run did not complete its task) |
| Class distribution | 5,202 failure (26.0%) / 14,798 success (74.0%) |
| Train/Test split | 75 / 25 stratified (seed 42) |
| Missing values | 0 (synthetic) · `failure_reason` excluded from inputs (leak guard) |

**Why simulated (documented honestly):** the closest real datasets — TRAIL (n=148), Who&When
(n=127), MAST, TracerTraj-2.5K — are small human-annotated trace-localization sets, none shaped
as a 20k-row telemetry→outcome problem. The generator (`src/data_pipeline.py`) is a step-level
causal process: run length is drawn from the task horizon **independent of outcome**, and the
outcome is a *noisy* function of accumulated trouble **plus unobserved capability gaps plus
Bernoulli noise** — so two runs with identical telemetry can differ in label (realistic
irreducible error). Failure-mode mix (of failures): retry-loop **47.3%**, exogenous tool error
**24.0%**, cascade **13.0%**, context-overflow **12.7%**, latent-capability 1.6%, degenerate-loop 1.5%.

## Experiments

### Experiment 1.1 — Causal generator + EDA literature check
**Hypothesis:** the literature-predicted drivers (tool depth, retries, cascade, context pressure)
will show up in the data, and context will be a *minority* failure cause.
**Method:** simulate 20k runs; bin each driver and plot empirical failure rate; measure the share
of failures occurring below context 0.80.
**Result:** failure rate rises monotonically with `max_tool_depth` (0.16→0.47), with
`max_consecutive_retries` (0.14→0.94), and only at the *top* context decile (0.31→0.62).
**Interpretation — the headline:** **84% of all failures occur with `context_max_pct < 0.80`.**
Retry/cascade failures fail at a **mean context utilization of just 0.30**; only the 12.7%
context-overflow failures sit near the top (mean 0.97). Context is a *symptom*, not the cause —
exactly the MAST prediction.

### Experiment 1.2 — Per-feature signal & nuisance sanity check
**Hypothesis:** tool/retry signals lead; runtime nuisances (temperature, latency, prompt length)
are near-uninformative.
**Method:** single-feature ROC-AUC for all 20 numeric features.
**Result:** top signals `error_rate_per_step` 0.734, `tool_error_rate` 0.730,
`max_consecutive_retries` 0.707, then tool-error/retry counts. Nuisances: temperature ~0.50,
latency ~0.52, prompt_tokens ~0.49.
**Interpretation:** the generator is well-behaved — the literature-predicted features carry the
signal and the nuisances carry none. Max single-feature AUC **< 0.74** ⇒ no leakage.

### Experiment 1.3 — Three baselines
**Method:** B1 majority class; B2 the deployed `context_max_pct > 0.80` rule; B3 standardized
Logistic Regression (`class_weight='balanced'`). Same split, ranked by AUPRC.
**Result / Interpretation:** see table. B2's rule is *precise but blind* — when it fires it's
right 86% of the time, but it only fires on a tiny slice, catching **15% of failures** at one
fixed operating point. B3 lifts AUPRC +24% over the rule and is a *dial*: 68% recall at its
default threshold, 20% recall if held to 80% precision. (B2's AUPRC/ROC are threshold-free on the
continuous `context_max_pct` score — generous to the baseline; F1/P/R are the fixed 0.80 alert.)

## Head-to-Head Comparison (test set, ranked by AUPRC)
| Rank | Model | AUPRC | ROC-AUC | F1 | Precision | Recall | Recall@P=0.80 | Notes |
|------|-------|------:|--------:|---:|----------:|-------:|--------------:|-------|
| 1 | **B3 LogReg (balanced)** | **0.599** | **0.773** | 0.548 | 0.460 | 0.678 | **0.197** | learned floor for Phase 2 |
| 2 | B2 context score (rule@0.80) | 0.482 | 0.644 | 0.258 | 0.860 | 0.152 | 0.172 | the deployed dashboard alert |
| 3 | B1 majority class | 0.260 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | sanity floor (= prevalence) |

## Key Findings
1. **The `context > 80%` rule is a strawman everyone ships.** 84% of failures happen below
   0.80 context — structurally invisible to it. It catches only **15% of failures** at one
   un-tunable operating point; retry/cascade failures fail at a mean context of just **0.30**.
2. **Failures are tool-driven, not context-driven.** Retry (47.3%) + cascade (13.0%) + upfront
   exogenous tool errors (24.0%) ≈ **84%** of failures; context-overflow is only 12.7%.
3. **A 1-line LogReg already clears AUPRC 0.60** (+24% over the rule) with zero feature
   engineering — and unlike the rule, it's a dial (68% recall achievable). This is the floor.
4. **The ceiling is honest, not perfect.** Best single feature < 0.74 AUC; the model tops out
   ~0.77 ROC-AUC because failure also depends on *unobserved* capability gaps and exogenous
   errors. A 0.99 here would have meant leakage — and the first two generator drafts *did* leak
   (perfect LogReg), which I caught and fixed (see below). A reviewer (Codex) additionally caught
   a residual `num_steps==2`-only-failure tell from exogenous truncation, now removed.

## Error Analysis
- B2 (rule) errors are almost all **false negatives**: it misses the 84% of failures with low
  context. Its few alerts are accurate (precision 0.83) but operationally useless in volume.
- B3 (LogReg) trades the other way at its default threshold — recall 0.68, precision 0.46 — i.e.
  it over-alerts. The PR curve shows it dominates B2 everywhere, but at strict precision 0.80
  recall collapses to 0.20: the ~24% telemetry-light exogenous failures are simply unpredictable
  from process telemetry, capping achievable recall at high precision.

## Next Steps (Phase 2)
- RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting vs the LogReg floor. **Hypothesis:**
  trees gain because failure is driven by the *context × tool-depth* and *retry × cascade*
  interactions (baked into the generator) that a linear model cannot represent. A quick
  GradientBoosting probe already lifts AUPRC ~0.61→~0.68 on a subset — to be confirmed properly.
- Investigate the operating-point question: is Recall@P=0.80 too strict given ~24% irreducible
  telemetry-light failures? Consider a cost-weighted operating point in Phase 4. Also freeze
  operating thresholds on a validation split (Codex review point) before reporting test P/R.

## References Used Today
- [1] MAST failure taxonomy — *Why Do Multi-Agent LLM Systems Fail?* (failure-mode analysis of 1,600+ traces).
- [2] TRAIL: Trace Reasoning and Agentic Issue Localization — Patronus AI, arXiv:2505.08638.
- [3] "12 LLM observability signals that predict incidents"; Datadog *State of AI Engineering* 2026.
- [4] AI agent failure-mode taxonomies (reasoning/tool/memory/orchestration), 2025–2026 surveys.

## Code Changes
- `src/data_pipeline.py` — causal step-level agent-run simulator (NEW; 3 iterations to remove a
  perfect-separation leak: decoupled run length from outcome, added latent + Bernoulli noise,
  dropped leaky `plan_steps`/`plan_deviation` for honest observability ratios).
- `src/utils.py` — shared metric helpers (`evaluate`, `recall_at_precision`).
- `config/config.yaml`, `data/README.md`, `requirements.txt` — project scaffold.
- `notebooks/phase1_eda_baseline.ipynb` — executed, 23 cells, 0 errors, 5 plots.
- `results/metrics.json`, `results/EXPERIMENT_LOG.md`, `results/phase1_*.png` (5 figures).
