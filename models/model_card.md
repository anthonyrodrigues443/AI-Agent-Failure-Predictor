# Model Card — AI-Agent Failure Predictor (champion)

A calibrated gradient-boosted classifier that predicts whether an LLM-agent run will
**fail**, from run-level telemetry, and — via a companion early-window model — raises the
alarm several steps **before** the run actually ends.

| | |
|---|---|
| **Model** | Optuna-tuned CatBoost, sigmoid-calibrated (Platt), `+ALL` 49-feature set |
| **Task** | Binary classification — `failure` (positive) vs `success` |
| **Primary metric** | AUPRC (positive class is the minority, prevalence 0.26) |
| **Artefact** | `models/champion.joblib` (~1.1 MB) — calibrated model + SHAP twin + frozen threshold |
| **Companion** | `models/early_window.joblib` — calibrated HistGBM per first-`k`-steps window |
| **Framework** | scikit-learn 1.8 `CalibratedClassifierCV` over `catboost` 1.2 |
| **Reproduce** | `python -m src.train` (deterministic, seed 42; no Optuna at deploy time) |

## Intended use
Operational monitoring of autonomous LLM-agent systems: score a run's telemetry to flag
likely failures for human review or automatic intervention (checkpoint, escalate, abort).
The early-window model supports **pre-emptive** intervention — acting on a 50%-risk alert
that fires, on average, **8–14 steps** before a failing run terminates.

**Out of scope:** this is *not* a root-cause tool, not a safety classifier, and not a
judge of output quality. It estimates *failure risk* from process telemetry only.

## Inputs
One agent run = 49 features assembled by `src/feature_engineering.py`:
- **26 base** — run-level telemetry aggregates (steps, tool calls, tool/reason error
  counts and rates, retries, max consecutive retries, reasoning loops, context max/mean/
  growth, tokens-per-step mean/growth, latency, distinct tools, temperature, prompt
  tokens) + one-hot `task_type` (5) and `model_tier` (3).
- **16 LEAD** — leading-indicator *trajectory* features: context velocity/acceleration,
  token acceleration, latency p95/p50, error slope/variance, **lag-1 autocorrelation of
  the error trace** (the "critical slowing down" early-warning signal from ecology), retry
  burstiness, time-to-first-error.
- **7 DOM** — explicit domain interactions (context×depth, retry×cascade, retries per
  tool call, tool-error×context, loop×context, depth×steps).

No `failure_reason` or any post-hoc label is used as input (leakage-guarded).

## Training data
20,000 simulated agent runs from `src/data_pipeline.py`, a causal, literature-calibrated
telemetry simulator (anchors: the **MAST** agent-failure taxonomy; ~31% of production
failures are tool-misuse, often upfront; context-retention degrades past ~10 turns;
cascading context corruption). Run **length is decoupled from outcome** and the outcome
is a noisy function of accumulated telemetry **plus a latent (unobservable) capability
term plus Bernoulli noise** — so the Bayes-optimal AUPRC is capped well below 1.0, as in
real failure-prediction problems. Split: 75/25 stratified (seed 42), prevalence 0.260.

> **Why synthetic?** No public dataset of labelled agent-run telemetry with per-step
> traces exists. The simulator is calibrated to published agent-failure patterns and the
> data card (`data/README.md`) documents every anchor. This is a known limitation.

## Performance (held-out test, n=5,000, prevalence 0.260)

| Metric | Value |
|---|---|
| **AUPRC** (primary) | **0.624** |
| ROC-AUC | 0.784 |
| Brier (calibration) | 0.148 |
| Recall @ Precision=0.80 | 0.249 |
| Deployed op. point (thr 0.632) | Precision 0.785 · Recall 0.267 · F1 0.398 |

The operating threshold is frozen on **out-of-fold train** predictions (honest), not the
test set. Recall by failure mode at the deployed threshold:

| Failure reason | % of failures | Recall |
|---|--:|--:|
| context_overflow | ~11% | **0.97** |
| cascade_failure | ~13% | 0.41 |
| stuck_retry_loop | ~48% | 0.15 (→0.52 at thr 0.5 — a precision dial, not a blind spot) |
| early_exogenous | ~25% | 0.13 |
| latent_capability | ~1% | **0.00 at any threshold** (the irreducible core) |

### Early-window ("failure in N steps")
Calibrated HistGBM on the first-`k` steps only. Recovers **76% of full-run AUPRC at k=3**
(runs average ~11 steps) and 93% by k=12 — failure is visible early.

### vs. frontier LLMs (Phase-5 zero-shot head-to-head, n=50 balanced)

| Model | AUPRC | Latency/run | Cost/1k |
|---|--:|--:|--:|
| **This model** | **0.833** | ~32 µs (batched) | $0.0001 |
| Claude Opus (zero-shot) | 0.738 | 10.3 s | $4.50 |
| Claude Haiku (zero-shot) | 0.709 | 23.9 s | $0.30 |

The 1 MB tree out-ranks both frontier LLMs on identical rows, ~5–6 orders of magnitude
faster and cheaper. The LLMs over-predict failure (recall ~0.95, precision ~0.5); the
tree is calibrated. *(F1/AUPRC are real predictions; LLM latency includes CLI overhead.)*

## Latency
CPU, single-threaded: **~32 µs/row batched** (~31,000 rows/s); ~10 ms for a single
isolated `predict_proba` call (Python + 5-fold calibration ensemble overhead — amortised
to µs under batching). No GPU required.

## Limitations & ethical considerations
- **Synthetic training data** — calibrated to literature, but real deployment requires
  re-fitting on the operator's own telemetry. Treat absolute numbers as in-distribution.
- **Signal-bound ceiling (~0.62 AUPRC).** Confirmed five independent ways (model class,
  feature engineering, Optuna, ablation, stacking). The residual is a latent
  capability-gap term that leaves *no telemetry fingerprint* — `latent_capability`
  failures are caught 0% at any threshold. Do not market this as catching "all" failures.
- **Calibration drift** — recalibrate if the agent stack, tool mix, or model tiers shift.
- **Not a safety system.** A low risk score is not a guarantee of a safe or correct run.
- **Operating point is a policy choice.** P≥0.80 minimises false alarms; lower it to catch
  more retry-loop failures at the cost of precision (recall rises 0.15→0.52 at thr 0.5).

## Maintenance
Retrain with `python -m src.train`; evaluate with `python -m src.evaluate`. The training
script asserts the reproduction (test AUPRC within 5e-4 of 0.624, threshold within 5e-3 of
0.632) and fails loudly on drift.
