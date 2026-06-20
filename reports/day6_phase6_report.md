# Phase 6: Production Pipeline + Real-Time Risk Dashboard — AI-Agent Failure Predictor
**Date:** 2026-06-20
**Session:** 6 of 7

## Objective
Turn the five-phase research champion into a clean, reproducible **production system** and a
**screenshot-worthy Streamlit dashboard** — without losing the headline (a 1 MB tree that
out-ranks Claude Opus & Haiku at agent-failure prediction). Concretely: lift the feature
engineering out of the notebooks into one importable module, make `train.py` reproduce the
champion deterministically (no Optuna at deploy time), ship `predict.py`/`evaluate.py`, build
the early-window "failure in N steps" serving path, and benchmark inference latency to back
the cost/speed story with a real number.

## Research & References
1. **Mitchell et al., 2019 — "Model Cards for Model Reporting" (Google).** Followed the
   intended-use / data / metrics / limitations structure for `models/model_card.md`;
   crucially the "out-of-scope" and "ethical considerations" sections (state the
   `latent_capability` 0%-recall core honestly, not bury it).
2. **scikit-learn `CalibratedClassifierCV` (Platt/sigmoid) docs + Niculescu-Mizil & Caruana,
   2005.** Confirms calibrating an already-well-ranked tree is a monotone post-map → it
   leaves SHAP attributions intact, which justifies the "SHAP twin" design (explain with the
   uncalibrated CatBoost, score with the calibrated one).
3. **Scheffer et al., 2009 — "Early-warning signals for critical transitions" (Nature).** The
   lag-1 autocorrelation ("critical slowing down") feature carried from Phase 3 is the
   theoretical basis for the early-window timeline; production-ised here as a per-`k` serving
   model so the dashboard can show risk rising before the run ends.
4. **MAST agent-failure taxonomy (carried from Phase 1).** Grounds the per-reason recall
   table and the dashboard's failure-reason labels.

How it shaped the session: research-grade artefacts (the 0.624 champion, the frozen 0.632
threshold, the k-sweep) already existed in notebooks; the work was **packaging without
drift**. Every number below is asserted against the Phase-4 research values in `train.py`.

## Dataset
| Metric | Value |
|--------|-------|
| Total runs | 20,000 simulated (MAST-calibrated) |
| Features | 49 (`+ALL`: 26 base + 16 LEAD + 7 DOM) |
| Target | `failure` (positive = run failed) |
| Class distribution | 26.0% failure / 74.0% success |
| Train/Test split | 15,000 / 5,000, stratified, seed 42 (matches Phase-2 cache) |

## Experiments / Build steps

### 6.1 — Canonical featuriser (`src/feature_engineering.py`)
**Hypothesis:** the notebook feature code can be lifted verbatim and reproduce the champion's
predictions exactly. **Method:** ported `build_lead`/`build_dom`/early-window helpers into one
module with a frozen 49-name schema; scored the *existing* `phase4_champion.joblib` through the
new pipeline on the same split. **Result:** test AUPRC **0.62406** and `max|Δproba| = 0.00e+00`
vs the cached Phase-4 probabilities. **Interpretation:** zero train/serve skew — the module is
a byte-identical reimplementation, safe to serve.

### 6.2 — Deterministic training (`src/train.py`)
**Method:** generate → featurise → fit Optuna-tuned CatBoost (frozen params from config) →
sigmoid-calibrate → freeze the P≥0.80 threshold on **OOF train** predictions → save champion +
SHAP twin + early-window models + curated demo runs. Built-in reproduction asserts.
**Result:** rebuilt from scratch in **212 s**, test **AUPRC 0.62406**, ROC 0.7840, Brier 0.1484,
frozen threshold **0.6323** (OOF P=0.800 R=0.283) → test **P=0.785 R=0.267** — identical to
Phase 4. The assert (`|AUPRC−0.624|<5e-4`, `|thr−0.632|<5e-3`) passed.

### 6.3 — Early-window serving models
**Method:** one calibrated HistGBM per `k ∈ {2,3,4,5,6,8,10,12}`, scored on first-`k`-step
features. **Result:**

| k (steps seen) | AUPRC | % of full-run | Recall@P=0.60 |
|--:|--:|--:|--:|
| 2 | 0.457 | 73% | 0.169 |
| 3 | 0.474 | 76% | 0.235 |
| 5 | 0.503 | 81% | 0.273 |
| 8 | 0.554 | 89% | 0.367 |
| 12 | 0.580 | 93% | 0.399 |

**Interpretation:** the production timeline confirms Phase 3/4 — most of the signal is present
in the first few steps; the dashboard's 50%-risk alert fires **8–14 steps before** failing runs
end (verified on the curated examples).

### 6.4 — Inference latency benchmark (`src/evaluate.py`)
**Method:** warmup + timed `predict_proba`, single-row and batched, CPU single-threaded.
**Result:** **31.7 µs/row batched** (~31,500 rows/s); ~9.9 ms for one isolated call (Python +
5-fold calibration-ensemble overhead, amortised away under batching). **Interpretation:** this
is the real number behind the speed headline — **~324,000× faster than Claude Opus zero-shot
(10.3 s/call)** in batched serving, on a laptop CPU.

### 6.5 — Streamlit dashboard (`app.py`)
Live-scoring risk gauge + verdict card, **early-window risk-vs-step timeline** ("failure in N
steps"), **SHAP-by-telemetry-family** bar (Error/Retry/Loop highlighted as the load-bearing
signal), two input modes (16 curated real held-out runs + a what-if slider builder), and a
sidebar carrying the frontier-LLM head-to-head as the selling point. Screenshot:
`results/ui_screenshot.png`.

## Head-to-Head Comparison (carried from Phase 5, now the dashboard's headline)
| Model | AUPRC (n=50) | Latency/run | Cost/1k | Winner |
|-------|--:|--:|--:|--------|
| **This model (champion)** | **0.833** | ~32 µs batched | $0.0001 | **✓ ranking + speed + cost** |
| Claude Opus (zero-shot) | 0.738 | 10.3 s | $4.50 | |
| Claude Haiku (zero-shot) | 0.709 | 23.9 s | $0.30 | |

## Key Findings
1. **Zero train/serve skew is achievable and worth asserting.** The lifted featuriser
   reproduced the champion to `0.00e+00` proba difference; `train.py` fails loudly if AUPRC or
   threshold drifts. Production credibility = the pipeline *proves* it matches the research.
2. **The early-window model productionises cleanly into a lead-time signal.** k=3 → 76% of
   full AUPRC; on real failing runs the 50% alert lands 8–14 steps early — the actual product.
3. **The honest-limitation surface is the differentiator.** Per-reason recall at the deployed
   threshold (context_overflow 0.97, cascade 0.41, retry 0.15, latent_capability **0.00**) is
   shipped in the model card and the dashboard footer — not hidden behind an aggregate.

## Frontier Model Comparison
See table above — the 1 MB calibrated tree out-ranks both frontier LLMs on identical rows at
5–6 orders of magnitude less latency/cost; the LLMs over-predict failure (recall ~0.95,
precision ~0.5) because they are uncalibrated. (Numbers from Phase 5; carried into the UI.)

## Error Analysis (deployed threshold 0.632, held-out)
| Failure reason | n | recall |
|---|--:|--:|
| context_overflow | 147 | 0.97 |
| cascade_failure | 164 | 0.41 |
| stuck_retry_loop | 629 | 0.15 |
| early_exogenous | 324 | 0.13 |
| degenerate_loop | 21 | 0.00 |
| latent_capability | 15 | 0.00 |

The misses are the *quiet* failures (retry/exogenous at the precision dial) plus the
irreducible telemetry-light core (`latent_capability`, 0% at any threshold) — consistent with
the Phase-4 finding that misses are magnitude-driven, not reason-driven.

## Next Steps
- **Phase 7 (Sun 2026-06-21):** full pytest suite expansion + final README polish + consolidate
  all phases into `reports/final_report.md`; optional FastAPI `predict` endpoint and a Dockerfile
  for the dashboard.
- Consider a hybrid serving mode (route only the tree's uncertain band to an LLM) — Phase 5
  showed model+Opus together beat either alone (exploratory, n=50).

## References Used Today
- [1] Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT*. https://arxiv.org/abs/1810.03993
- [2] Niculescu-Mizil & Caruana (2005), *Predicting Good Probabilities With Supervised Learning*, ICML. + sklearn `CalibratedClassifierCV` docs.
- [3] Scheffer et al. (2009), *Early-warning signals for critical transitions*, Nature 461. https://www.nature.com/articles/nature08227
- [4] MAST: Multi-Agent System failure Taxonomy (carried from Phase 1 `data/README.md`).

## Code Changes
- `src/feature_engineering.py` (new) — canonical 49-feature pipeline + early-window + trace synthesiser; verified `0.00e+00` proba drift vs champion.
- `src/train.py` (new) — deterministic training, reproduction asserts, champion + SHAP twin + early-window + `ui_examples.json`.
- `src/predict.py` (new) — `predict_run`/`predict_batch`/`explain_run` (grouped SHAP)/`early_window_curve`/`early_warning_lead`.
- `src/evaluate.py` (new) — held-out metric bundle + latency benchmark + per-reason recall → `results/phase6_eval.json`.
- `app.py` (new) — Streamlit real-time risk dashboard.
- `models/model_card.md` (new), `config/config.yaml` (+champion/early_window blocks), `requirements.txt` (+plotly, +pytest).
- `tests/test_feature_engineering.py`, `tests/test_predict.py` (new) — 12 passing.
- `results/ui_screenshot.png`, `results/phase6_eval.json`, `results/phase6_train_summary.json`, `results/ui_examples.json` (new).
