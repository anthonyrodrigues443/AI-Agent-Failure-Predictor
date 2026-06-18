# Experiment Log — AI-Agent Failure Predictor

## Phase 1 — Dataset + EDA + Baselines (2026-06-15)

Dataset: 20,000 simulated runs, failure rate 0.260. Primary metric AUPRC; operating metric Recall@P=0.80.

**Headline:** 84.0% of failures occur below context 0.80 (invisible to the industry rule).

| Baseline | AUPRC | ROC-AUC | F1 | Recall@P=0.80 |
|---|---|---|---|---|
| B1 majority | 0.2600 | 0.5000 | 0.000 | 0.000 |
| B2 context>0.80 | 0.4825 | 0.6443 | 0.258 | 0.172 |
| B3 LogReg | 0.5987 | 0.7725 | 0.548 | 0.197 |

## Phase 2 — Multi-Model Head-to-Head (2026-06-16)

7 models / 3 paradigms, identical 75/25 split, ranked on AUPRC (5-fold CV + held-out test).
**Headline: the trees barely beat the linear floor (+0.019 AUPRC); LightGBM loses to it.** The
real win is calibration, not ranking.

| Rank | Model | Paradigm | AUPRC | ROC-AUC | F1 | R@P=0.80 | Brier | Δ vs floor |
|---|---|---|---|---|---|---|---|---|
| 1 | **HistGBM** | boosting | **0.6175** | 0.7818 | 0.446 | **0.255** | **0.148** | **+0.019** |
| 2 | CatBoost | boosting | 0.6077 | 0.7746 | 0.456 | 0.235 | — | +0.009 |
| 3 | ExtraTrees | bagging | 0.6054 | 0.7686 | 0.517 | 0.236 | — | +0.007 |
| 4 | XGBoost | boosting | 0.6036 | 0.7711 | 0.452 | 0.254 | — | +0.005 |
| 5 | RandomForest | bagging | 0.6027 | 0.7688 | 0.473 | 0.222 | 0.153 | +0.004 |
| 6 | LogReg (floor) | linear | 0.5987 | 0.7725 | 0.548 | 0.197 | 0.190 | — |
| 7 | LightGBM | boosting | 0.5964 | 0.7654 | 0.454 | 0.238 | — | −0.002 |

Probes:
- **Interaction:** LogReg + `ctx×depth` + `retry×cascade(proxy)` → 0.6067, recovers **42%** of the floor→champion gap (mechanism real, but small; cascade term is latent).
- **Imbalance:** XGBoost `scale_pos_weight∈{1,2.84,5}` → AUPRC spread 0.005, recall@0.5 spread 0.39 (threshold knob, not signal).
- **CatBoost native cats:** +0.0036 AUPRC over one-hot (negligible, 2 low-cardinality cats).
- **Frozen P=0.80 threshold:** lands at 0.769 precision on test (target 0.80) — ~3pt transfer drift.

Champion = **HistGBM** (best AUPRC + best Brier + tied-best R@P=0.80). Cached to
`models/phase2_champion.joblib` + `results/phase2_champion_test_proba.npy` for the Phase-5 LLM head-to-head.

## Phase 3 — Feature Engineering on the Leading Edge (2026-06-17)

Engineered 3 families on top of the 20 baseline features — **LEAD** (16 trajectory rate/EWS/latency-tail
features), **DOM** (7 explicit interactions/ratios), **EW(k)** (early-window, first-*k*-step only).
Simulator extended to emit per-step traces with **zero RNG impact** (aggregates byte-identical to the
committed parquet — asserted). Top-3 carry-forward models, identical 75/25 split, ranked on AUPRC.

**Headline: the ~0.62 ceiling is signal-bound, the signal lives in the trajectory, and 3 steps in we
already recover 78% of the full-run AUPRC.**

Feature-set × model (top of table):
| Feature set | Model | n_feat | AUPRC | Brier | R@P80 | Δ vs its FS0 |
|---|---|--:|--:|--:|--:|--:|
| **+ALL** | **CatBoost** | 49 | **0.6208** | 0.1481 | 0.249 | **+0.0131** |
| +LEAD | HistGBM | 42 | 0.6191 | 0.1478 | 0.245 | +0.0016 |
| FS0 | HistGBM | 26 | 0.6175 | 0.1481 | **0.255** | — |
| +DOM | HistGBM | 33 | 0.6174 | 0.1480 | 0.249 | −0.0002 |
| +ALL | HistGBM | 49 | 0.6169 | 0.1483 | 0.245 | −0.0007 |
| +LEAD | CatBoost | 42 | 0.6154 | 0.1494 | 0.250 | +0.0077 |
| FS0 | CatBoost | 26 | 0.6077 | 0.1508 | 0.235 | — |
| FS0 | ExtraTrees | 26 | 0.6054 | 0.1574 | 0.236 | — |

New project-best = **CatBoost+ALL 0.6208** (+0.0033 over Phase-2 champ, within the noise band). The
lift concentrates on *weaker* models; the strongest (HistGBM) barely moves — trees already reconstruct
the interactions from raw telemetry.

Probes:
- **Level vs rate:** LEVEL-only 0.6038 · RATE-only 0.5677 · BOTH 0.6043 → **rate alone recovers 94%**.
  The signal is in the *shape of the trajectory*, not the endpoint.
- **Early-window recovery (HistGBM, first-*k*-step features only):** k=2 0.467 (76%) · **k=3 0.482 (78%)**
  · k=5 0.511 (83%) · k=7 0.545 (88%) · k=10 0.575 (93%). At step 3 (mean run ~9-13 steps) the model
  equals the *full-run* accuracy of the industry `context>0.80` alarm (Phase-1 B2 = 0.4825).
- **Univariate:** explicit `ix_retry_casc` is the single strongest feature in the pool (0.739 AUC),
  above the best raw feature — yet adds ~0 to trees (redundant) and ~+0.013 to CatBoost.
- **Permutation importance (CatBoost+ALL):** engineered features in top-10 = `ix_ctx_depth`,
  `ix_retry_casc`, `toolerr_x_ctx`, and the EWS feature **`err_lag1ac`** (lag-1 autocorrelation —
  "critical slowing down" transfers from ecology to agent telemetry).

Cached `results/phase3_best_test_proba.npy` (CatBoost+ALL) on the same test idx for the Phase-5 LLM
head-to-head. Two contenders go into Phase-4 tuning: CatBoost+ALL and HistGBM.
