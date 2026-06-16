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
