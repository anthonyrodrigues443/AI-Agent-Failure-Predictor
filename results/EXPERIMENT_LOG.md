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

## Phase 4 — Optuna Tuning + Calibration + Error Analysis (2026-06-18)

Tuned both contenders on `+ALL` with TPE over research-informed ranges, **selecting on train-only CV**
(no test leakage), scored once on the identical held-out test. Then calibrated the champion, froze the
operating threshold on held-out (OOF) data, gave the early-window model a deployable operating point, and
ran a full error analysis.

**Headline: Optuna does not break the ceiling (+0.0029 test AUPRC) — confirmed from a 4th angle — and the
residual error splits into a recoverable precision-tradeoff band and a small irreducible telemetry-light core.**

Tuned vs default on test (ranked on AUPRC):
| Model | test AUPRC | ROC-AUC | Brier | R@P80 | Δ vs its default |
|---|--:|--:|--:|--:|--:|
| **CatBoost tuned** (champion) | **0.6237** | 0.7842 | 0.1471 | 0.254 | **+0.0029** |
| CatBoost default | 0.6208 | 0.7807 | 0.1481 | 0.249 | — |
| HistGBM tuned | 0.6198 | 0.7837 | 0.1475 | 0.241 | +0.0030 |
| HistGBM default | 0.6169 | 0.7802 | 0.1483 | 0.245 | — |

CV gain (HistGBM 0.631→0.638, CatBoost 0.624→0.637) was +0.007–0.013 but **did not transfer** to test —
fold-overfit in the noise band. TPE chose strong regularization (CatBoost depth 4, lr 0.025; HistGBM 13
leaf nodes, lr 0.013), re-deriving Phases 2-3's "simpler is better on low-signal telemetry".

Probes:
- **Calibration is a no-op on the tuned tree:** raw Brier **0.1471** (best), sigmoid 0.1484, isotonic 0.1472;
  AUPRC flat (monotone maps). Refines Phase 2 — the regularized booster is already calibrated; "always
  calibrate" is the wrong reflex (we ship the sigmoid wrapper only for a stable operating point).
- **Honest operating point:** frozen-on-OOF threshold (0.632) lands at **P=0.785 / R=0.267** on test vs the
  in-sample-optimal P=0.800 / R=0.249. The optimism is a **precision shortfall (~1.5 pt), not a recall drop**.
- **Early-window (sigmoid-calibrated):** at k=3 → R@P60 0.265 / R@P80 0.065; k=10 → 0.404 / 0.165. Failure is
  rankable at step 3 but a high-precision early alarm costs recall — alarm at P≈0.60 early, tighten later.
- **Error analysis:** recall by reason at the deployable point — context_overflow 0.97, cascade 0.41,
  stuck_retry_loop 0.15 (48% of failures!), early_exogenous 0.13, latent_capability 0.00. FN telemetry ≈
  successes on every signal (retries 1.6 vs 4.6 for caught). At thr 0.5 the retry/cascade/exogenous misses
  recover (0.15→0.26, 0.41→0.52, 0.13→0.19) but `latent_capability` stays **0.00** — the irreducible core.
  Subgroup blind spot: frontier models recall 0.05, multi_hop_qa 0.01 — the quiet/high-capability regime.

Champion artifact `models/phase4_champion.joblib` (sigmoid-calibrated tuned CatBoost + frozen threshold +
feature list); `results/phase4_champion_test_proba.npy` cached on the same test idx for the Phase-5 LLM
head-to-head.
