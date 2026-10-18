# Sections 5 and 6: pre-processing and modeling (made by Claude)

> **Who decided what.** On 16 Oct 2026 the user asked Claude to choose the pre-processing and modeling decisions ("the best decisions that are defensible and give the best results"), build sections 5 and 6 of `f1_podium_prediction.ipynb` for **both model A and model B**, and then look for other decisions that could improve the models. **Every decision in this file was made by Claude, not reviewed by the user yet.** Each one has its reason and a pointer to where it lives in the code, so it can be checked or reversed.
>
> All the code is in `tools/build_notebook.py` (sections 5 and 6), which generates the notebook. Numbers below come from the notebook run of 16 Oct 2026.

---

## 1. What was built

| Notebook section | Content |
|---|---|
| **5.1** | `PodiumPreprocessor` (one scikit-learn transformer, fitted on the training seasons only), the model-A and model-B tables, a leak-free check |
| **5.2** | Before/after plots and a range table (brief: "effect of the step on data distribution/range/values") |
| **5.3** | Trial: `log(grid)` for the linear models |
| **5.4** | Three pre-processing-order experiments (E1, E2, E3) |
| **6** (intro) | Models, metrics, rules |
| **6.1–6.4** | Grid-only reference, baseline logistic regression, random forest, gradient boosting, neural network: tuned for **A and B** on validation |
| **6.5** | Six further improvement attempts (I1–I6), validation only |
| **6.6** | Final comparison on train / validation / **test** (all 5 metrics), precision-recall curves, final models |
| **6.7** | Limitations |

Everything runs top to bottom (Run All) without errors. It takes several minutes, mostly the grid searches in 6.2–6.4. Results are deterministic (fixed seeds): two separate runs gave identical outputs.

---

## 2. Pre-processing decisions (section 5)

**The rule:** every statistic used to transform the data (medians, percentiles, means, standard deviations) is learned from the **training seasons only**, inside each model's pipeline, then applied unchanged to validation and test.

| # | Decision | Reason | Where to change it |
|---|---|---|---|
| P1 | `reached_q3` = 0 before 2006; `grid_minus_quali` = 0 when qualifying is missing | the value has a real meaning (no Q3 existed / no known penalty); the `no_q3_format` and `quali_missing` flags say why | `PodiumPreprocessor._base` |
| P2 | New flags `drv_no_history` (driver's first race) and `cons_no_history` (team's first race) | the model can treat "unknown" differently from "average" | `_base` |
| P3 | `drv_experience` → log(1 + starts) | long tail (0–324 starts in model-A training) | `_base` |
| P4 | Cap `gap_q1_pct`, `drv_champ_pos`, `drv_avg_finish_5`, `drv_age` at the training 1st/99th percentile | extreme rows (wet Q1 up to 33%, 28th in the championship) would pull a linear model; percentiles come from the data, not hand-picked | `CAP_COLS`, `fit` |
| P5 | Remaining gaps → training **median** | robust to outliers; flags keep the "was missing" information | `_impute` |
| P6 | Standard scaling (mean 0, std 1) | regularised logistic regression and the neural network need comparable scales | `transform` |
| P7 | **Per model:** LR and NN use all steps; random forest uses P1–P5 (no scaling/capping, but sklearn's forest needs no NaN); gradient boosting uses P1–P3 and keeps NaN (native support) | trees don't need scaling; gradient boosting learns the best direction for missing values itself | `MODEL_SPECS` |
| P8 | **`log(grid)`** for LR and NN | **trial 5.3:** validation PR-AUC A 0.7244 → 0.7313, B 0.7202 → 0.7283 | `LOG_GRID` |
| P9 | **No resampling** (no SMOTE / undersampling) for the ~14% podium rate; imbalance handled by PR-AUC, the validation threshold, and `class_weight` as a hyperparameter | resampling breaks the "3 podiums per race" structure and distorts probabilities | — |

**Effect (5.2, model-A training rows):** `gap_q1_pct` 0–33% → 0–7.9%; `drv_champ_pos` 1–28 → 1–25; `drv_avg_finish_5` 1–24 → 1.6–20.2; `drv_age` 17–44 → 19–41; experience 0–324 → 0–5.8 (log). Missing before: 1,061 `reached_q3`, 113 `gap_q1_pct`, 72 driver form, 33 team form, 19 `grid_minus_quali` → **0 after**. Output: 20 columns (18 features + 2 flags).

### Pre-processing-order experiments (5.4), scored with logistic regression, validation PR-AUC

| Experiment | Chosen order | Alternative | A | B |
|---|---|---|---|---|
| **E1** where the pipeline is fitted | training seasons only: 0.7313 / 0.7283 | all seasons incl. validation + test (leaky): | 0.7305 | 0.7280 |
| **E2** missing-value flags | created before filling: 0.7313 / 0.7283 | created after filling (all 0): | 0.7314 | 0.7286 |
| **E3** capping vs scaling | cap, then scale: 0.7313 / 0.7283 | scale, then cap: | 0.7313 | 0.7294 |

**All differences are below 0.002**, far below the validation noise (±0.047). Why: medians and percentile caps barely move when the future seasons are added (median `gap_q1_pct` 1.36% → 1.22%); only ~1% of rows lack form history; capping/scaling only rescale a column, which logistic regression absorbs. **The correct order was kept** (it matters for correctness even when the scores don't move). *Possible criticism:* the brief may expect experiments with a visible effect. Leakage experiments with a bigger effect would need a non-robust step (e.g. mean imputation, or target-based encoding), which this pipeline deliberately avoids.

---

## 3. Modeling decisions (section 6)

| # | Decision | Reason |
|---|---|---|
| M1 | **Naive reference:** logistic regression on `grid` + `pit_lane_start` only | shows what the other features add |
| M2 | **Baseline (required):** logistic regression on all 18 features | the simplest reasonable model |
| M3 | **Statistical models:** random forest, gradient boosting (`HistGradientBoostingClassifier`) | bagging vs boosting, two different tree families |
| M4 | **Shallow NN:** `MLPClassifier`, 1–2 hidden layers, early stopping (random 10% of training rows) | allowed by the brief; already installed |
| M5 | Same rows and same 18 features for every model within a version | brief: comparable scores |
| M6 | Hyperparameters by grid search on **validation PR-AUC**, fixed seeds | the test seasons are never used for choices |
| M7 | **No refit on train + validation** before the test | the threshold chosen on validation would otherwise belong to a different model |
| M8 | **Primary metric PR-AUC**, secondary ROC-AUC; also F1, precision, recall | podiums are ~14% of rows; PR-AUC focuses on ranking the real podium finishers on top |
| M9 | **Threshold:** maximises F1 on validation; compared with **top 3 per race** | brief: threshold from train/validation only |

**Grids** (`MODEL_SPECS`):
- LR: C ∈ {0.01, 0.1, 1, 10} × class_weight {None, balanced}.
- RF (400 trees): depth {4, 6, 8, 12} × min leaf {20, 50} × max_features {√, 0.5} × class_weight {None, balanced_subsample}.
- GB: learning rate {0.03, 0.1} × rounds {100, 300} × depth {3, None} × leaves {7, 15} × min leaf {40, 100}, L2 = 1.
- NN: layers {(16), (32, 16), (64, 32)} × alpha {0.001, 0.01, 0.1, 1}.

**Grid history:** the first grids were too narrow. The best settings sat at the edges (smallest RF depth, largest NN alpha), and gradient boosting overfitted (training 0.82 vs validation 0.70). The grids were widened once. A second check **beyond** the new edges (GB learning rate 0.01, depth 2, 50–200 rounds; RF depth 3, min leaf 100–200) gained at most **+0.0007**, so the final grids are wide enough. That check ran outside the notebook and isn't in it.

### Tuned validation PR-AUC

| | grid only | logistic regression | random forest | gradient boosting | neural network |
|---|---|---|---|---|---|
| **A** | 0.677 | 0.733 (C 0.1) | **0.742** (depth 8, leaf 50, √, balanced) | 0.740 (depth 3, 7 leaves, 100 rounds, lr 0.03, leaf 40) | 0.734 ((64, 32), alpha 0.1) |
| **B** | 0.677 | 0.730 (C 10, balanced) | 0.734 (depth 4, leaf 50, 0.5, balanced) | 0.729 (depth 3, 7 leaves, 100 rounds, lr 0.03, leaf 100) | **0.740** ((64, 32), alpha 0.01) |

All tuned models are within ~0.01: **not a significant difference** with 60 validation races.

---

## 4. Further attempts to improve (6.5): all rejected

Tried in order, each starting from the current best model; adopted only if it improves the best model of **both** A and B (B-only ideas: must improve B). Validation only.

| Attempt | Result (validation PR-AUC) | Decision |
|---|---|---|
| I1 race-relative features (rank within the race of team points share, team form, driver form, driver average finish) | A 0.742 → 0.727, B 0.740 → 0.718 (intervals below 0) | reject: duplicates information the features already have |
| I2 per-race normalisation (probabilities of a race sum to 3) | A → 0.672, B → 0.714 | reject: destroys the comparison **across** races that PR-AUC rewards |
| I3 recency weights for model B (half-life 10 years) | small gain per family (GB 0.729 → 0.732), best weighted 0.734 < 0.740 | reject |
| I4 model-B training window | 1950: 0.740, 1970: 0.723, 1980: 0.698, 1994: 0.734 | keep 1950 (more data helps) |
| I5 monotonic constraints for gradient boosting | A 0.7397 → 0.7392, B 0.7289 → 0.7242 | reject |
| I6 ensemble of LR + RF + GB + NN | A 0.739, B 0.735 | reject (slightly below the best single models) |

Also tried during section 5/6 and **adopted**: `log(grid)` (P8), wider grids (section 3).

**Conclusion:** the 18 pre-race features carry about this much information. Different modeling tricks don't add more; gains would need **new information** (ideas in section 7).

---

## 5. Final results: test seasons 2022–2024 (used once, at the end)

| Version | Model | Test PR-AUC | Test ROC-AUC | Test F1 | Precision | Recall | Threshold (from validation) | F1 top 3 per race |
|---|---|---|---|---|---|---|---|---|
| A | grid only | 0.647 | 0.914 | 0.649 | 0.650 | 0.647 | 0.408 | 0.652 |
| A | logistic regression | 0.710 | 0.939 | 0.677 | 0.628 | 0.735 | 0.350 | 0.667 |
| A | **random forest (selected)** | **0.706** | **0.937** | **0.668** | 0.647 | 0.691 | 0.754 | 0.637 |
| A | gradient boosting | 0.718 | 0.938 | 0.654 | 0.642 | 0.667 | 0.390 | 0.652 |
| A | neural network | 0.709 | 0.938 | 0.650 | 0.653 | 0.647 | 0.442 | 0.652 |
| B | grid only | 0.647 | 0.914 | 0.649 | 0.650 | 0.647 | 0.360 | 0.652 |
| B | logistic regression | 0.723 | 0.941 | 0.674 | 0.609 | 0.755 | 0.740 | 0.662 |
| B | random forest | 0.718 | 0.938 | 0.661 | 0.607 | 0.725 | 0.779 | 0.637 |
| B | gradient boosting | 0.721 | 0.939 | 0.687 | 0.624 | 0.765 | 0.346 | 0.647 |
| B | **neural network (selected)** | **0.704** | **0.937** | **0.660** | 0.659 | 0.662 | 0.405 | 0.647 |

**Reading the results:**
- Every model beats the grid-only reference by **+0.06 to +0.08 PR-AUC** on unseen seasons.
- Test is 0.02–0.04 below validation: the 2022 rule change, plus a slightly optimistic validation from choosing the best of many settings.
- **A ≈ B.** 50 extra seasons don't clearly help or hurt; the qualifying data matters more than more years.
- **The model selected on validation isn't the best on test** (GB for A, LR for B score higher), but all differences are within noise. The simple logistic regression is as good as the complex models.
- The top-3-per-race rule doesn't beat the F1 threshold.
- No overfitting: training PR-AUC (A ~0.70) isn't above validation (0.74). Model B's training score (~0.61) is lower because old seasons are harder to predict.

**In plain words:** on races it has never seen, the model's podium predictions are right about **2 times out of 3**, and it finds about **2 of every 3** real podium finishers. Starting position alone gets about 0.65.

---

## 6. Decisions you may want to revisit

1. **Which final model to report.** The rule "best on validation" picks the random forest (A) and the neural network (B). Since all models are statistically tied, **logistic regression** could be defended as the main model: simplest, easiest to explain, and among the best on test. Choosing it *because* of the test scores would be test-set selection, though; that argument can only be "simplicity", decided without looking at the test.
2. **The order experiments show almost no effect.** That's honest, but an evaluator may prefer an experiment with a visible effect.
3. **Class weights:** the selected random forest uses balanced class weights, so its probabilities are inflated (threshold 0.75). If calibrated probabilities matter (e.g. for the inference function), use the unweighted forest (0.737 validation) or calibrate.
4. **No refit on train + validation (M7).** Refitting would use 3 more seasons for the final model, but the validation threshold wouldn't strictly apply any more.
5. **Neural network early stopping** uses a random 10% of training rows, not a season-based split.

## 7. Ideas not tried (would need new features, i.e. section 4 changes)

- Round 1: use the **previous season's** final standings instead of "no standing yet".
- **Team-mate comparison** (qualifying gap to the team-mate): isolates driver skill from the car.
- **Circuit character** (e.g. the historical grid-to-finish correlation at that circuit; overtaking difficulty) without using the circuit id itself.
- **Sprint result** (2021+), deferred earlier by the user's decision.
- Probability **calibration** (doesn't change PR-AUC, but makes probabilities usable as real chances).

## 8. How to change a decision

Edit `tools/build_notebook.py` (sections 5–6), then:

```bash
cd ACL-2
python tools/build_notebook.py f1_podium_prediction.ipynb
python tools/run_notebook.py  f1_podium_prediction.ipynb
```

and update the result notes in the notebook and this file if numbers change.
