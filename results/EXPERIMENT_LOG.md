# Experiment Log — AI-Agent Failure Predictor

## Phase 1 — Dataset + EDA + Baselines (2026-06-15)

Dataset: 20,000 simulated runs, failure rate 0.260. Primary metric AUPRC; operating metric Recall@P=0.80.

**Headline:** 84.0% of failures occur below context 0.80 (invisible to the industry rule).

| Baseline | AUPRC | ROC-AUC | F1 | Recall@P=0.80 |
|---|---|---|---|---|
| B1 majority | 0.2600 | 0.5000 | 0.000 | 0.000 |
| B2 context>0.80 | 0.4825 | 0.6443 | 0.258 | 0.172 |
| B3 LogReg | 0.5987 | 0.7725 | 0.548 | 0.197 |
