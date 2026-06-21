# Phase 7: Testing, Serving & Consolidation — AI-Agent Failure Predictor
**Date:** 2026-06-21
**Session:** 7 of 7 (final)

## Objective
Turn five phases of research + one phase of productionisation into a **credible, reproducible
deliverable**: expand the test suite to lock the invariants the whole project rests on, add a
deployable HTTP surface, and consolidate all seven phases into one report. No new modelling —
this is the day the project earns trust.

## Research & References
1. **Mitchell et al., 2019 — Model Cards for Model Reporting.** The card must ship the
   per-failure-mode recall, not just an aggregate — done in Phase 6, reaffirmed here as the
   honest-limitations surface (`latent_capability` 0.00 at any threshold).
2. **Train/serve-skew avoidance (Google ML pipelines guidance).** One canonical featuriser
   imported by train/predict/serve — the FastAPI layer reuses `src.predict.*` verbatim, so the
   HTTP path scores with the identical 49-feature schema. No re-implementation.
3. **Contract testing of data generators.** The load-bearing test is a *single-feature
   no-leakage guard* (`max AUC < 0.85`): the Phase-1 leak (a feature with AUC 1.0) would fail it
   immediately. Generator invariants are tested, not assumed.

How research influenced today's work: the three references map to the three deliverables —
model card (already shipped) → tests that *enforce* the honest surface; skew-avoidance → reuse
predict.py in serve.py; generator contract testing → `test_data_pipeline.py`.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 simulated agent runs (regenerated, seed 42) |
| Features | 49 (`+ALL`: 26 base + 16 LEAD + 7 DOM) |
| Target variable | `failure` (binary, positive = failure) |
| Class distribution | 26.0% failure / 74.0% success |
| Train/Test split | 15,000 / 5,000 stratified (seed 42) |

## Experiments / Work Items

### 7.1: Champion reproduction (deterministic rebuild)
**Hypothesis:** the frozen pipeline reproduces the research champion bit-for-bit on this box.
**Method:** `python -m src.train` (single-threaded OMP) from a clean worktree off `origin/main`.
**Result:** test **AUPRC 0.62406**, ROC 0.7840, Brier 0.1484; frozen threshold **0.6323**
(OOF P=0.800 R=0.283 → test P=0.785 R=0.267). Reproduction asserts passed. Early-window k=3 →
76% of full AUPRC.
**Interpretation:** the deploy-time pipeline is honest — the asserts in `train.py` would have
failed loudly on any drift. Numbers match Phase 4/6 exactly.

### 7.2: Test-suite expansion (14 → 47)
**Method:** four new test files + the two existing ones. (The single-feature leakage guard and
the four extra `test_serve` cases below were strengthened in response to the Codex review — §7.5.)
| File | Tests | What it locks |
|---|--:|---|
| `test_data_pipeline.py` | 12 | determinism, trace==aggregate byte-identity, run-length decoupled from outcome (AUC<0.62), **no single-feature leak over all 49 inputs (AUC<0.85; worst = `ix_retry_casc` 0.755)**, exo failures telemetry-light, 84%-below-context headline |
| `test_feature_engineering.py` | 9 | 49-col schema/order, single-row dummy encoding, synth-run consistency |
| `test_utils.py` | 6 | metric bundle keys/ranges, perfect-separation, **unreachable precision → `None` threshold** |
| `test_evaluate.py` | 5 | split determinism + stratification, **champion reproduction (AUPRC≈0.624, hermetic — writes to tmp)**, latency >1k rows/s, reason-recall ordering |
| `test_predict.py` | 5 | predict_run / batch / explain / early-window contracts |
| `test_serve.py` | 10 | `/health` (+early-window status), `/model`, `/predict`, `/predict/whatif`, unknown-category + missing-aggregate + out-of-range → 422 |
**Result:** **47 passed, 0 skipped** in ~51 s (model-gated tests ran because the artefact is built).
**Interpretation:** the suite enforces the project's two riskiest failure modes — data leakage
and train/serve drift — in code.

### 7.3: FastAPI serving layer + Dockerfile
**Method:** `src/serve.py` wraps `predict_run`/`explain_run`/`early_warning_lead`; lazy-loads
the champion so `/health` answers artefact-free; `Dockerfile` trains the model into the image.
**Result:** `/predict/whatif` on a trouble run → **P(fail)=0.896 [Critical]**, early-warning
alerted; `/model` returns champion meta (49 features, threshold 0.632, AUPRC 0.6241). Out-of-range
input → 422. All via `fastapi.testclient` (no live server needed in tests).
**Interpretation:** a deployable surface with zero serving skew — same featuriser, same model.

### 7.4: Consolidation
`reports/final_report.md` (244 lines) — domain context, the synthetic-data honesty, the master
experiment table (rule 0.48 → LogReg 0.60 → champion 0.62), the **five-angle ceiling** table, the
frontier-LLM head-to-head, production/serving, and limitations. README updated to Phase 7/7
complete with the new iteration block.

### 7.5: Codex second-model review — applied in-branch
The PR's automated Codex (GPT-5.4) review returned 5 findings; **all 5 were valid and applied**
before the cron triage (this project's established pattern — builder pre-lands, cron verifies):
| # | Finding | Fix applied |
|---|---|---|
| 1 | `/predict` 500s if only `champion.joblib` is present (`early_warning_lead` needs `early_window.joblib`) | `_score` degrades gracefully (`early.available=False`); `/health` now reports `early_window_loaded` |
| 2 | `RunRecord` accepts an untyped dict → `KeyError`/`TypeError` as 500 on malformed payloads | `as_mapping` validates required aggregate fields + list-typed traces → 422 |
| 3 | `task_type`/`model_tier` unconstrained → unknown category silently scores as the baseline | `Literal` types on `WhatIfRequest` + explicit category check on raw `/predict` → 422 |
| 4 | `test_evaluate` calls `ev.main()` which writes `results/phase6_eval.json` (mutates repo, breaks hermetic CI) | `monkeypatch ev.ROOT → tmp_path`; the test no longer touches the repo |
| 5 | the no-leakage guard only checked `BASE_NUMERIC`, but the champion trains on all 49 features | guard now runs over the full `+ALL` matrix; worst single feature is the engineered `ix_retry_casc` at **AUC 0.755** (< 0.85) — a stronger, more honest test |

Added 4 `test_serve` cases locking the new validation (unknown-category, missing-aggregate,
early-window health status). Copilot's review was quota-limited (no actionable output), consistent
with prior phases. Net: suite 43 → 47, all green; the serving layer is robuster and the headline
leakage guard now covers exactly what the README claims it does.

## Head-to-Head Comparison (final master table, held-out test, ranked by AUPRC)
| Rank | Model / config | AUPRC | ROC-AUC | Brier | R@P=0.80 |
|---|---|---:|---:|---:|---:|
| 1 | **CatBoost tuned `+ALL` (champion)** | **0.6237** | 0.784 | **0.147** | 0.254 |
| 6 | HistGBM (Phase-2 champ) | 0.6175 | 0.782 | 0.148 | 0.255 |
| 8 | LogReg balanced (floor) | 0.5987 | 0.773 | 0.190 | 0.197 |
| 9 | context > 0.80 rule (industry) | 0.4825 | 0.644 | — | 0.172 |
| 10 | majority class | 0.2600 | 0.500 | — | 0.000 |

## Key Findings
1. **The consolidation thesis:** rule → learned score is **+24%** AUPRC; learned score →
   fully-tuned/engineered/ensembled champion is **+4%**. The regime change is the baseline swap,
   not the sophistication after it.
2. **A test suite is where leakage gets caught for good** — the single-feature AUC guard is the
   one test that would have flagged the Phase-1 bug at commit time.
3. **Reproduction asserts + one canonical featuriser = no train/serve skew** — the FastAPI layer
   inherits it for free.

## Frontier Model Comparison (carried from Phase 5, unchanged)
| Model | AUPRC | Latency/run | Cost/1k | Winner |
|---|--:|--:|--:|---|
| **This model** | **0.833** | ~32 µs | $0.0001 | **✓** |
| Claude Opus (zero-shot) | 0.738 | 10.3 s | $4.50 | |
| Claude Haiku (zero-shot) | 0.709 | 23.9 s | $0.30 | |

## Error Analysis (the shipped blind spot, re-verified this session)
Per-reason recall at the deployed threshold (fresh evaluate run): context_overflow **0.973**,
`latent_capability` **0.000** (n=15 in this test split). The irreducible core is caught 0% at any
threshold — stated in the model card, the dashboard footer, and now `test_evaluate.py`.

## Next Steps
- Rotation complete after 2026-06-21 (per `PROJECT_ROTATION.md`). The project is finished and
  reproducible end-to-end.
- Future (out of rotation): re-fit on a real operator's telemetry; A/B the early-window alarm's
  intervention value (does acting on the 8–14-step lead actually reduce wasted spend?).

## References Used Today
- [1] Mitchell et al., 2019. *Model Cards for Model Reporting.* arXiv:1810.03993.
- [2] Google Cloud — *ML pipelines & training-serving skew* (architecture guidance).
- [3] Project Phase-1 leakage post-mortem (`reports/day1_phase1_report.md`) — the bug the
      generator-contract tests now guard against.

## Code Changes
- `tests/test_data_pipeline.py`, `tests/test_utils.py`, `tests/test_evaluate.py`,
  `tests/test_serve.py` (new) — +33 tests (14 → 47, incl. the §7.5 review-response additions).
- `src/serve.py` (new) — FastAPI scoring service reusing `src.predict`.
- `Dockerfile` (new) — containerised service; trains the model into the image.
- `reports/final_report.md` (new) — consolidated 7-phase research report.
- `README.md` — Phase 7/7 complete, new iteration block, repo layout + reproduce + roadmap.
- `requirements.txt` — `fastapi` / `uvicorn` / `httpx`.
