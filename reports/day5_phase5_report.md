# Phase 5: Advanced Techniques, Ablation & the Frontier-LLM Head-to-Head — AI-Agent Failure Predictor
**Date:** 2026-06-19
**Session:** 5 of 7

## Objective
Phases 1-4 established, from four independent angles (generator design, model class, feature
engineering, Optuna), that this honest agent-telemetry problem has a **signal-bound ~0.62 AUPRC
ceiling** — set by an irreducible latent capability gap + Bernoulli noise, not by the model. The
deployed champion is a sigmoid-calibrated, Optuna-tuned **CatBoost on the 49-feature `+ALL` set**
(test AUPRC 0.624, Brier 0.147, frozen P=0.785 / R=0.267).

Phase 5 stress-tests that champion the way a skeptical reviewer would, with four questions:
1. **Does oversampling help?** SMOTE/ADASYN are the reflexive fix for class imbalance — test whether
   they actually lift AUPRC, or just trade precision + calibration for a meaningless recall@0.5 bump.
2. **Does a real stacking ensemble break the ceiling** that single models couldn't?
3. **Which telemetry family is load-bearing?** A group-level ablation (drop a whole mechanism), not a
   single-feature one.
4. **Can a frontier LLM out-predict an 8 MB tree on agent telemetry?** Claude Opus, Claude Haiku, and
   Codex (GPT-5.4), zero-shot, on the *same* stratified sample — head to head on accuracy, AUPRC,
   latency, and \$/1k.

**Primary metric:** AUPRC. **Operating metric:** Recall @ Precision = 0.80. **Secondary:** ROC-AUC, Brier.

## Research & References
1. **"To SMOTE, or not to SMOTE?"** (Elor & Averbuch-Elor, 2022) and **Fernández et al. (2018, JAIR,
   "SMOTE for Learning from Imbalanced Data: 15-year anniversary")** — synthetic oversampling
   interpolates in feature space; on **overlapping** class distributions it manufactures minority
   points inside the majority manifold, inflating recall while degrading precision and probability
   calibration. For strong, already-calibrated learners (boosted trees) resampling rarely improves
   ranking metrics. → motivated ranking the imbalance sweep on CV AUPRC and reading the *cost* off
   precision + Brier, not recall@0.5.
2. **Wolpert (1992), "Stacked Generalization"** + **van der Laan et al. (2007), "Super Learner"** —
   stacking only helps when base learners are diverse/decorrelated, and must be trained on out-of-fold
   predictions to avoid leakage. → motivated the leak-free OOF stack and a base-learner correlation
   check as the diagnostic for *why* stacking would or wouldn't help here.
3. **Project mandate (frontier-LLM head-to-head)** — race the custom model against both Claude and
   Codex zero-shot, comparing accuracy/AUPRC **and** the operational axes (latency, \$/1k) that decide
   whether a predictor can run on every step of every agent in production.

*How it shaped the work:* the literature predicts oversampling won't help an already-calibrated tree
on an overlapping problem and that highly-correlated base learners leave a meta-learner nothing to
combine — so Phase 5 is designed to *measure those predicted null results cleanly* rather than to
chase a lift the theory says isn't there, and to put the real win (5-6 orders of magnitude on
cost/latency vs frontier LLMs) on the same page as the accuracy numbers.

## Dataset
| Metric | Value |
|--------|-------|
| Total runs | 20,000 (causal literature-calibrated simulator, Phase 1) |
| Features | `+ALL` = 49 (FS0 26 telemetry/one-hot + 16 LEAD + 7 DOM, Phase 3) |
| Target | `failure` (binary; positive = run failed) |
| Class balance | 26.0% failure (AUPRC prevalence floor 0.260) |
| Train/Test | 15,000 / 5,000 stratified — **identical indices to Phases 2-4** (asserted vs `phase2_test_idx.npy`) |
| LLM sample | 50 test runs, stratified 25 fail / 25 pass (deterministic, `rng=42`, cached) |

## Experiments

### Experiment 5.1 — Does oversampling help? (SMOTE / ADASYN / SMOTE+Tomek / undersampling / cost-sensitive)
**Hypothesis:** on this overlapping, already-calibrated problem, synthetic oversampling will *not* lift
CV AUPRC and will *hurt* precision@0.5 and Brier; class-weighting slides the operating point without
adding ranking signal (the Phase-2 `scale_pos_weight` result).
**Method:** each method wrapped in an `imblearn` pipeline so resampling acts **inside each CV fold only**
(never on the held-out test); rank on 5-fold CV AUPRC, read the operating point off the held-out test
at threshold 0.5. Estimator = the Phase-4 champion (tuned CatBoost on `+ALL`).
**Result:** Confirmed, decisively. **Every** resampling method *loses* to the plain calibrated champion
on CV AUPRC — the best alternative trails by −0.0016 and the synthetic-oversampling family by −0.011 to
−0.013.

| Method | CV AUPRC | Δ vs champ | test AUPRC | Brier | Prec@0.5 | Verdict |
|--------|---------:|-----------:|-----------:|------:|---------:|---------|
| **None (champion)** | **0.6362** | **0** | **0.6237** | **0.1471** | **0.698** | best ranking + best calibration |
| Cost-sensitive (class-weight) | 0.6346 | −0.0016 | 0.6216 | 0.1808 | 0.491 | calibration wrecked (Brier ↑23%), precision halved |
| RandomUnderSample | 0.6319 | −0.0043 | 0.6194 | 0.1842 | 0.479 | throws away data; Brier worse |
| SMOTE | 0.6251 | −0.0110 | 0.6112 | 0.1520 | 0.623 | ranking ↓, no real gain |
| SMOTE+Tomek | 0.6250 | −0.0112 | 0.6094 | 0.1522 | 0.635 | ranking ↓ |
| BorderlineSMOTE | 0.6244 | −0.0117 | 0.6089 | 0.1526 | 0.617 | ranking ↓ |
| ADASYN | 0.6233 | −0.0129 | 0.6079 | 0.1530 | 0.618 | worst — interpolates into the overlap |

Two distinct failure modes: **synthetic oversampling** (SMOTE/ADASYN/Borderline/Tomek) lowers the
*ranking* (AUPRC −0.011 to −0.013) because it interpolates minority points straight into the dense
class-overlap this generator bakes in; **reweighting/undersampling** preserves AUPRC better but
**destroys calibration** (Brier 0.181–0.184 vs 0.147) and halves precision — exactly the Phase-2
`scale_pos_weight` finding. The right tool for imbalance here is the thing we already shipped: a
*calibrated* model + a threshold chosen for the precision the business needs.

### Experiment 5.2 — A real stacking ensemble vs the champion
**Hypothesis:** base learners are ~0.9+ correlated (Phase-2's 0.022 AUPRC band) → a meta-learner has
little to combine; stacking won't break the ceiling.
**Method:** four base learners (tuned CatBoost, tuned HistGBM, ExtraTrees, standardized LogReg) →
out-of-fold predictions via `cross_val_predict` (leak-free) → logistic meta-learner; compared to the
champion alone and a naive equal-weight average. Base-proba correlation matrix as the diagnostic.
**Result:** Stacking does **not** break the ceiling — the 5th independent confirmation it is signal-bound.

| Model | test AUPRC | ROC-AUC | Brier | Δ AUPRC vs champ |
|-------|-----------:|--------:|------:|-----------------:|
| **Champion (CatBoost+ALL)** | **0.6237** | 0.7842 | 0.1471 | **0** |
| Stacked (LogReg meta, OOF) | 0.6215 | 0.7842 | 0.1485 | −0.0022 |
| Mean ensemble (4 bases) | 0.6205 | 0.7831 | 0.1475 | −0.0032 |

The base learners' test-probability correlations are **all ≥ 0.92** (min 0.92) — they make the same
predictions, so the meta-learner has nothing to combine. The stack's ROC ties the champion to 4
decimals while its AUPRC is *lower*; added complexity, no ranking gain. The ~0.62 ceiling now holds
across generator → model class → features → Optuna → **ensembling**.

### Experiment 5.3 — Group ablation: which telemetry family is load-bearing?
**Hypothesis:** the Error/Retry/Loop family dominates (Phase-1: failure is 47% retry + 13% cascade);
dropping the entire engineered LEAD+DOM block barely moves AUPRC (Phase-3: trees reconstruct the
interactions from raw telemetry).
**Method:** partition the 49 features into 5 mechanistic families; drop each in turn, retrain the
champion on the remainder, measure ΔAUPRC on the held-out test; also drop all engineered features.
**Result:** One family dominates, and the entire engineered block is nearly free to remove.

| Dropped family | n kept | test AUPRC | ΔAUPRC | Verdict |
|----------------|-------:|-----------:|-------:|---------|
| **Error/Retry/Loop (18)** | 31 | 0.6016 | **−0.0222** | load-bearing — *this is the signal* |
| Meta/Run-shape (9) | 40 | 0.6062 | −0.0175 | 2nd — `model_tier` proxies the latent competence gap |
| Context (9) | 40 | 0.6203 | −0.0034 | small — context is a symptom (Phase-1 result) |
| ALL engineered (LEAD+DOM, 23) | 26 | 0.6218 | −0.0019 | **negligible — trees reconstruct interactions** |
| Tool/Depth (5) | 44 | 0.6244 | +0.0007 | redundant |
| Tokens/Latency (8) | 41 | 0.6256 | +0.0018 | mildly *helps* to drop (noisy) |

The Error/Retry/Loop family carries 6× the signal of the next mechanistic group and ~12× the Context
family — consistent with Phase 1 (failure is 47% retry-loop + 13% cascade, not context). The second
result is subtle and new: **Meta/Run-shape is the 2nd most load-bearing family** because `model_tier`
is the only observable correlate of the *latent capability* that drives the irreducible failures —
the model leans on it as a proxy. Dropping the whole 23-feature engineered block costs just −0.0019,
re-confirming Phase 3: gradient-boosted trees already extract the interactions from raw telemetry.

### Experiment 5.4 — Frontier-LLM head-to-head (the headline)
**Hypothesis:** a frontier LLM cannot out-rank a calibrated tree on quiet agent telemetry, and loses by
orders of magnitude on latency and cost.
**Method:** the same 50-run stratified sample sent zero-shot to Claude Opus, Claude Haiku, and Codex
(GPT-5.4) via their local CLIs, each given the *raw* telemetry an operator sees (no engineered terms);
the calibrated champion scored on the identical rows; compared on accuracy/F1/precision/recall/AUPRC +
latency/run + \$/1k. (Codex via the agentic CLI averaged ~130 s/call → a smaller, documented sample;
both Claude models cover the full 50.)
**Result:** On the identical 50 runs, the **1 MB calibrated CatBoost out-ranks both Claude models**
(AUPRC 0.833 vs Opus 0.738 vs Haiku 0.709) — at ~70 microseconds/run and ~$0.0001/1k.

| Model | n | Accuracy | F1 | Precision | Recall | AUPRC | Latency/run | Cost/1k |
|-------|--:|---------:|---:|----------:|-------:|------:|------------:|--------:|
| **Champion CatBoost @0.5** | 50 | 0.640 | 0.438 | **1.000** | 0.280 | **0.833** | **70 µs** | **$0.0001** |
| Champion CatBoost @0.63 (frozen) | 50 | 0.620 | 0.387 | **1.000** | 0.240 | **0.833** | **70 µs** | **$0.0001** |
| Claude Opus (zero-shot) | 50 | 0.520 | 0.667 | 0.511 | 0.96 | 0.738 | 10.3 s | $4.50 |
| Claude Haiku (zero-shot) | 50 | 0.640 | 0.719 | 0.590 | 0.92 | 0.709 | 23.9 s | $0.30 |
| Codex GPT-5.4 (zero-shot) | 11* | 0.727 | 0.800 | 0.857 | 0.75 | 0.899* | 100 s | $50 |

\* Codex via the agentic CLI averaged ~100 s/call (it re-scans the workspace each call), so it only
completed 11 of the 50 rows (8 fail / 3 pass) before being stopped — its high numbers are on a smaller,
slightly easier subset and are **not** comparable to the n=50 rows; reported for completeness, not as a
win. The two Claude models cover the full 50.

**Three real findings:**
1. **The tree out-ranks the frontier models** on the same data (AUPRC 0.833 vs 0.738 / 0.709). On quiet
   agent telemetry, a calibrated tree beats zero-shot Opus and Haiku at *ranking* failure risk.
2. **The LLMs cry wolf.** Recall 0.92–0.96 but precision 0.51–0.59 — on a balanced sample they label
   nearly everything FAIL. The champion is the opposite (precision 1.000, recall 0.24–0.28 at its
   deployed threshold tuned for 26% prevalence). The tree is *calibrated*; the LLMs are not.
3. **Operational gulf.** 70 µs/run vs 10.3 s (Opus) / 23.9 s (Haiku) → **~147,000× / ~341,000× faster**;
   $0.0001/1k vs $4.50 / $0.30 → **45,000× / 3,000× cheaper**. This is the difference between a predictor
   you can run on *every step of every agent* and one you cannot.

### Experiment 5.5 — Hybrid: model + LLM on the borderline
**Hypothesis:** routing only the champion's borderline cases to an LLM won't beat the champion alone on
this problem (the LLM is weaker on the quiet telemetry where the champion is also uncertain).
**Method:** route runs with calibrated champion prob in a band around 0.5 to the best fully-covered LLM,
keep the champion elsewhere; compare hybrid accuracy to each component alone (n=50, exploratory).
**Result:** The combination beats *either* component alone on the balanced sample.

| Borderline band ±  | n routed to Opus | Hybrid accuracy | Hybrid F1 |
|--------------------|-----------------:|----------------:|----------:|
| 0.05 | 3 | 0.640 | 0.471 |
| 0.10 | 5 | 0.680 | 0.556 |
| **0.15** | **9** | **0.720** | **0.650** |
| 0.20 | 13 | 0.680 | 0.619 |

Champion-only accuracy 0.640; Opus-only 0.520; **best hybrid (±0.15 band, 9/50 routed) 0.720** — neither
alone clears 0.64. The champion is precise-but-conservative; routing only its uncertain band to
high-recall Opus recovers the failures it under-fires on. **Caveat:** n=50 and the sample is balanced;
at the real 26% prevalence Opus's 0.51 precision would generate false alarms on the (now-majority) PASS
runs, so this gain must be re-measured at production prevalence before it ships. Framed as a direction,
not a verdict.

## Head-to-Head Comparison
All approaches ranked on the project's primary metric (test-set AUPRC, prevalence 0.260):

| Rank | Approach | test AUPRC | Brier | R@P80 | Notes |
|------|----------|-----------:|------:|------:|-------|
| 1 | **CatBoost tuned `+ALL` (champion)** | **0.6237** | **0.1471** | **0.254** | Phase-4 champion — unbeaten |
| 2 | Stacked (CatBoost+HistGBM+ExtraTrees+LogReg, OOF meta) | 0.6215 | 0.1485 | — | bases ≥0.92 correlated |
| 3 | Cost-sensitive CatBoost (class-weight) | 0.6216 | 0.1808 | — | calibration wrecked |
| 4 | Mean ensemble (4 bases) | 0.6205 | 0.1475 | — | no gain |
| 5 | RandomUnderSample + CatBoost | 0.6194 | 0.1842 | — | data thrown away |
| 6 | SMOTE + CatBoost | 0.6112 | 0.1520 | — | interpolates into overlap |
| 7 | BorderlineSMOTE + CatBoost | 0.6089 | 0.1526 | — | — |
| 8 | ADASYN + CatBoost | 0.6079 | 0.1530 | — | worst |

**Nothing in Phase 5 beat the plain calibrated champion.** Every advanced technique either matched it
(stacking, to 3 decimals on ROC) or hurt ranking/calibration (all resampling).

## Key Findings
1. **SMOTE/ADASYN are the wrong reflex on an overlapping, calibrated problem** — they cut CV AUPRC by
   0.011–0.013; reweighting/undersampling hold AUPRC but blow up Brier (0.18 vs 0.147). The shipped
   answer — calibrated model + business-chosen threshold — strictly dominates.
2. **The ~0.62 ceiling held a 5th time** (now via ensembling): base learners are ≥0.92 correlated, so a
   stack/average has nothing to combine. The bottleneck is *signal*, not the combiner.
3. **The Error/Retry/Loop telemetry family is load-bearing** (−0.0222 AUPRC when dropped, 6× the next
   family); the entire 23-feature engineered block costs only −0.0019 to remove. New nuance: `model_tier`
   (in Meta/Run-shape, the 2nd most important family) is the model's only proxy for the latent
   capability gap that causes the irreducible failures.
4. **A 1 MB calibrated tree out-ranks Claude Opus and Haiku at failure prediction** (AUPRC 0.833 vs
   0.738 / 0.709 on the same 50 runs) — at ~147,000× / 341,000× the speed and 45,000× / 3,000× the cost.
   The LLMs cry wolf (0.92–0.96 recall, 0.51–0.59 precision); they are not calibrated.
5. **Model + LLM together beat either alone (exploratory):** hybrid 0.72 acc vs 0.64 (tree) / 0.52
   (Opus) — routing only the tree's borderline cases to high-recall Opus. Needs re-validation at real
   prevalence before shipping.

## Frontier Model Comparison
| Model | Metric | Custom (CatBoost) | Claude Opus | Claude Haiku | Codex GPT-5.4 | Winner |
|-------|--------|------------------:|------------:|-------------:|--------------:|--------|
| n=50 sample | AUPRC | **0.833** | 0.738 | 0.709 | 0.899 (n=11)* | **Custom** (on equal n) |
| n=50 sample | Precision | 1.000 | 0.511 | 0.590 | 0.857* | **Custom** |
| n=50 sample | Recall | 0.28 | 0.96 | 0.92 | 0.75* | LLMs (but cry-wolf) |
| Operational | Latency/run | **70 µs** | 10.3 s | 23.9 s | 100 s | **Custom** |
| Operational | Cost/1k | **$0.0001** | $4.50 | $0.30 | $50 | **Custom** |

\* Codex on an 11-row subset — reported, not comparable. *AUPRC/precision/latency are real; LLM latency
includes CLI + agent startup overhead and would be lower via direct API — but the cost math holds and
the custom model's microsecond inference is genuine.*

## Error Analysis / Caveats
- **Balanced LLM sample.** The 50-run head-to-head is 25/25, not the real 26% prevalence — accuracy is on
  a coin-flip baseline and these are *not* full-test AUPRC numbers; the same-50-rows AUPRC column is the
  apples-to-apples comparator. The champion's full-test AUPRC remains 0.624.
- **Why the champion looks "low recall" here.** Its threshold is frozen for P=0.80 at 26% prevalence; on
  a balanced sample that threshold is conservative (precision 1.0, recall 0.24–0.28). At its operating
  prevalence it catches 27% of failures at 78% precision (Phase 4).
- **Codex coverage.** 11/50 rows due to ~100 s/call agentic overhead — documented; both Claude models
  (the more relevant frontier comparison) cover the full 50.
- **CLI latency.** LLM latency includes process + agent startup; direct-API would be far lower. Stated
  honestly — the headline rests on AUPRC + cost, not on the inflated CLI timing.

## Next Steps
- Phase 6 (Sat 2026-06-20): production pipeline (`src/train.py`, `predict.py`, `evaluate.py`) + a
  polished Streamlit real-time risk dashboard (risk gauge, contributing factors, "predicted failure in
  N steps" using the early-window model), model card.
- Carry the Phase-5 result into the UI: ship the calibrated champion + frozen threshold, show the
  Error/Retry family as the top contributing factors, and surface the cost/latency advantage vs an LLM
  judge as a selling point.

## References Used Today
- [1] Elor, Y. & Averbuch-Elor, H. (2022). *To SMOTE, or not to SMOTE?* arXiv:2201.08528.
- [2] Fernández, A. et al. (2018). *SMOTE for Learning from Imbalanced Data: Progress and Challenges,
  Marking the 15-year Anniversary.* JAIR 61:863-905.
- [3] Wolpert, D. (1992). *Stacked Generalization.* Neural Networks 5(2):241-259.
- [4] van der Laan, M., Polley, E. & Hubbard, A. (2007). *Super Learner.* Stat. Appl. Genet. Mol. Biol.
- [5] imbalanced-learn docs — SMOTE/ADASYN/BorderlineSMOTE/SMOTETomek/RandomUnderSampler pipelines.

## Code Changes
- `src/llm_eval.py` (new) — frontier-LLM CLI harness (Claude/Codex callers, defensive parser, split +
  stratified-sample reconstruction, cached idempotent eval driver, background cache-warmer entrypoint).
- `notebooks/phase5_advanced.ipynb` (new) — 5 experiments end-to-end (imbalance ablation, stacking,
  group ablation, LLM head-to-head, hybrid) with plots + metrics.
- `requirements.txt` — added `optuna`, `imbalanced-learn`.
- `results/` — `phase5_imbalance_ablation.{csv,png}`, `phase5_stacking.{csv,png}`,
  `phase5_group_ablation.{csv,png}`, `phase5_llm_vs_custom.{csv,png}`, `metrics.json[phase5]`,
  `phase5_llm_cache/`; `results/EXPERIMENT_LOG.md` (+Phase 5 section).
