# Phase 2: Multi-Model Head-to-Head — AI-Agent Failure Predictor
**Date:** 2026-06-16
**Session:** 2 of 7

## Objective
Phase 1 left a **linear floor**: a balanced LogisticRegression reaches AUPRC 0.599 / ROC 0.773.
The generator bakes in two *nonlinear* failure channels (`ctx_pressure × tool_depth` and
`retries × cascade`) that a linear model cannot represent, and a quick Phase-1 GradientBoosting
probe hinted at ~0.68 AUPRC. **Today's question:** do tree ensembles actually beat the floor on
honest evaluation, *and can we prove why* — or was the 0.68 probe a mirage? Plus three secondary
questions: bagging vs boosting, does class re-weighting add ranking signal, and which model is
calibrated enough to threshold at the P=0.80 operating point.

## Research & References
1. **Springer 2025 benchmark (20 models / 111 datasets)** & **TALENT (300+ datasets)** — both
   reconfirm GBDTs (XGBoost/LightGBM/CatBoost) match-or-beat deep nets on tabular data; this set
   the model lineup. *Implication tested:* GBDTs should top the table — but "top" turned out to
   mean +0.02, not the expected landslide.
2. **Canonical Path Deviation as a Causal Mechanism of Agent Failure** (arXiv 2602.19008) — frames
   cascade/drift as the causal mechanism of long-horizon agent failure, i.e. the exact
   `retry × cascade` channel; motivated the §7 interaction probe.
3. **XGBoost imbalanced-classification guidance** (MachineLearningMastery / forecastegy) — `scale_pos_weight`
   reweights the positive class; we tested whether that buys *ranking* (AUPRC) or only shifts the threshold.

**How research shaped the work:** the GBDT-dominance literature justified a 7-model, 3-paradigm
comparison ranked on AUPRC with 5-fold CV; the path-deviation paper justified treating the tree
advantage as a *mechanistic* question (which interactions), not a leaderboard.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 agent runs (Phase-1 simulator, unchanged) |
| Features | 20 numeric + 2 categorical (one-hot → 26 design cols); `failure_reason` excluded (leak guard) |
| Target | `failure` (26.0%) |
| Train/Test split | 75 / 25 stratified, seed 42 (identical to Phase 1) |
| Primary / operating metric | AUPRC / Recall@Precision=0.80 |

## Experiments

### Experiment 2.1 — 5-fold CV ranking (train only)
**Hypothesis:** boosting tops a robust out-of-fold ranking, clearly above the floor.
**Method:** StratifiedKFold(5) on train; per-fold AUPRC, mean ± std, for all 7 models.
**Result:** HistGBM 0.6295 ± 0.012 > CatBoost 0.6221 > ExtraTrees 0.6200 > XGBoost 0.6178 >
RandomForest 0.6158 > **LogReg 0.6120** > LightGBM 0.6082. The whole field spans **0.021 AUPRC**.
**Interpretation:** the ranking is *stable* (HistGBM #1 in CV and test) but the *spread is tiny* —
the linear floor is only 0.018 below the best and beats LightGBM.

### Experiment 2.2 — Test head-to-head (ranked on AUPRC)
**Result:** see table below. Champion **HistGBM** AUPRC 0.6175, **+0.019 over the floor** (+3.1% rel).
LightGBM *underperforms* the floor. All 7 PR curves are nearly superimposed.
**Interpretation — the headline:** the trees do **not** crush the floor. The Phase-1 "~0.68 probe"
did not survive 5-fold CV + a held-out split; it was optimistic. The shared ceiling (~0.62 AUPRC,
~0.78 ROC) is set by **irreducible latent factors** baked into the generator (latent
difficulty−competence, Bernoulli noise, ~24% telemetry-light exogenous failures), not model class.

### Experiment 2.3 — CatBoost native categoricals vs one-hot
**Result:** native +0.0036 AUPRC over one-hot (0.6113 vs 0.6077). **Interpretation:** real but
negligible with only two low-cardinality categoricals — not worth abandoning the uniform one-hot.

### Experiment 2.4 — Mechanistic interaction probe (the "why")
**Hypothesis:** if the tree edge *is* the two interactions, adding them to LogReg recovers most of it.
**Method:** add `ix_ctx_depth`, `ix_retry_casc` (+ `ctx_clip`, `depth_norm`) to LogReg, re-fit.
**Result:** LogReg 0.5987 → **0.6067** with interactions, recovering **42%** of the floor→champion gap.
**Interpretation:** the interactions are a *real but partial* explanation. `ix_ctx_depth` is fully
observable; `ix_retry_casc` uses an **observable proxy** for the latent `cascade` state, so the
residual 58% is partly the un-observable part of cascade and partly many small nonlinearities the
trees mop up. Honest version of the Phase-1 claim: the interactions matter, but the *whole* effect
is small.

### Experiment 2.5 — Imbalance probe (does re-weighting add signal?)
**Method:** XGBoost with `scale_pos_weight ∈ {1, 2.84, 5}`; AUPRC vs threshold-dependent metrics.
**Result:** AUPRC spread **0.005** (and monotonically *down*); recall@0.5 spread **0.39**.
**Interpretation:** re-weighting is a **threshold knob, not new signal** — it slides the operating
point without improving ranking. Sets up the Phase-5 SMOTE/ADASYN ablation (expectation: same lesson).

### Experiment 2.6 — Calibration (the real differentiator)
**Method:** reliability curves + Brier for LogReg, RandomForest, HistGBM.
**Result:** Brier LogReg **0.190** ≫ RandomForest 0.153 > HistGBM **0.148**. LogReg's reliability
curve sits far below the diagonal (it predicts 0.6 when the true rate is ~0.34).
**Interpretation:** `class_weight="balanced"` *distorts* LogReg's probabilities — it ranks fine but
can't be thresholded honestly. HistGBM is the only model that's both top-ranked **and** calibrated,
and it lifts **Recall@P=0.80 from 0.197 → 0.255 (+29% rel)**. **That**, not AUPRC, is why it wins.

### Experiment 2.7 — Frozen-threshold operating point (rigor, Codex Phase-1 point)
**Method:** pick the P=0.80 threshold on a validation slice of train, freeze, apply to test.
**Result:** frozen threshold 0.58 → test recall 0.262 at **precision 0.769** (target was 0.80);
the test-curve "optimistic" R@P80 is 0.245. **Interpretation:** thresholds don't transfer perfectly
— precision drifts ~3 points off target. Phase 4 should set the threshold under a cost model.

## Head-to-Head Comparison (test set, ranked by AUPRC)
| Rank | Model | Paradigm | AUPRC | ROC-AUC | F1 | Precision | Recall | R@P=0.80 | Δ vs floor |
|------|-------|----------|------:|--------:|---:|----------:|-------:|---------:|-----------:|
| 1 | **HistGBM** | boosting | **0.6175** | 0.7818 | 0.446 | 0.699 | 0.327 | **0.255** | **+0.019** |
| 2 | CatBoost | boosting | 0.6077 | 0.7746 | 0.456 | 0.685 | 0.342 | 0.235 | +0.009 |
| 3 | ExtraTrees | bagging | 0.6054 | 0.7686 | 0.517 | 0.584 | 0.463 | 0.236 | +0.007 |
| 4 | XGBoost | boosting | 0.6036 | 0.7711 | 0.452 | 0.664 | 0.342 | 0.254 | +0.005 |
| 5 | RandomForest | bagging | 0.6027 | 0.7688 | 0.473 | 0.651 | 0.371 | 0.222 | +0.004 |
| 6 | LogReg (floor) | linear | 0.5987 | 0.7725 | 0.548 | 0.460 | 0.678 | 0.197 | — |
| 7 | LightGBM | boosting | 0.5964 | 0.7654 | 0.454 | 0.638 | 0.352 | 0.238 | −0.002 |

*(CV ranking is consistent: HistGBM #1, LightGBM last. Fit/CV times are distorted by concurrent CPU
load during this run and are NOT a fair speed comparison — see the notebook timing caveat.)*

## Key Findings
1. **Model class barely matters here — the bottleneck is signal, not the algorithm.** Best tree is
   +0.019 AUPRC over a 1-line LogReg; LightGBM loses to it; all 7 sit in a 0.022 band. The Phase-1
   0.68 probe was optimistic and did not replicate. (The "complex barely beats simple" finding.)
2. **"Boosting > bagging > linear" is false at this signal ceiling** — boosting occupies both #1 and
   #7; a bagging model (ExtraTrees) outranks three boosters.
3. **The champion's value is calibration, not ranking.** LogReg's balanced weighting wrecks its
   probabilities (Brier 0.190); HistGBM (0.148) is the model you can actually threshold, lifting
   R@P=0.80 by +29% relative.
4. **Re-weighting ≠ signal.** `scale_pos_weight` moves AUPRC by 0.005 but recall@0.5 by 0.39 — a pure
   threshold lever.

## What Didn't Work (and why)
- **LightGBM** under these defaults landed *below* the linear floor (0.5964). With `num_leaves=31` +
  subsampling on a low-signal problem it likely over-fragments; it would need tuning (Phase 4) to
  match HistGBM. Honest result: out-of-the-box, more boosting machinery ≠ better here.
- **Cranking `scale_pos_weight`** — added zero ranking signal and slightly *hurt* AUPRC.
- **The expected landslide** — didn't happen; the generator's irreducible latent term caps everyone.

## Error Analysis
- At threshold 0.5 the champion is *precise but low-recall* (P 0.70 / R 0.33): it's confident only on
  the high-trouble runs and misses the telemetry-light failures — the same ~24% exogenous/latent
  failures that capped Phase 1.
- The floor (LogReg) inverts this (P 0.46 / R 0.68) because its balanced weighting inflates
  probabilities — high recall but unusable precision and bad calibration.

## Next Steps (Phase 3 — feature engineering)
- The lever is **features/signal**, not model family. Engineer **leading-indicator** features:
  context *growth rate* vs level, retry *burstiness*, early-window error slope, tokens/step
  acceleration — features that fire *before* failure (the project's headline thesis). Test whether
  they lift the shared ~0.62 ceiling on the top-3 (HistGBM, CatBoost, ExtraTrees).
- Carry the **calibration** lesson forward: report Brier alongside AUPRC; prefer calibrated models.
- Phase 4: set the operating threshold under an explicit cost model instead of a fixed precision.

## References Used Today
- [1] *Implementation and Performance Comparison of Gradient Boosting Algorithms for Tabular Data
  Classification* — Springer 2025 (20 models / 111 datasets). https://link.springer.com/chapter/10.1007/978-981-97-4533-3_36
- [2] *Canonical Path Deviation as a Causal Mechanism of Agent Failure in Long-Horizon Tasks* — arXiv 2602.19008.
- [3] *How to Configure XGBoost for Imbalanced Classification* — MachineLearningMastery; *scale_pos_weight* guide — forecastegy.
- [4] TALENT tabular benchmark (300+ datasets) — GBDT dominance on tabular.

## Code Changes
- `notebooks/phase2_multimodel.ipynb` — NEW, 28 cells, 0 errors, executed on the `agent-failure`
  kernel: 7-model CV + test head-to-head, interaction probe, imbalance probe, calibration,
  frozen-threshold operating point, persistence.
- `results/metrics.json` — appended `phase2` block (CV ranking, test table, all probes).
- `results/phase2_model_comparison.{png,csv}`, `results/phase2_interaction_recovery.png`,
  `results/phase2_calibration.png`, `results/phase2_champion_confusion.png` — 4 figures + CSV.
- `results/phase2_champion_test_proba.npy`, `results/phase2_test_idx.npy`,
  `models/phase2_champion.joblib` — cached champion (HistGBM) for the Phase-5 LLM head-to-head.
