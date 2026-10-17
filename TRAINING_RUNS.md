# Training runs log: F1 podium prediction

Every model-training run in the project, in order: what went in, what came out, and what changed from the previous run and why.
All numbers come from the notebook outputs (`f1_podium_prediction.ipynb`) and the project log (`README.md`). Nothing here is a new run.

**Reading guide**
- **Target:** `podium` = finished 1st–3rd. About 14–15% of rows.
- **Main score: PR-AUC** (how well real podium finishers are ranked at the top; random guess ≈ 0.15). Also ROC-AUC, F1, Brier score (probability error, lower is better), and the **top-3-per-race** pick.
- **Seasons:**

  | Set | Model A | Model B |
  |---|---|---|
  | Train | 2003–2018 | 1950–2018 |
  | Validation | 2019–2021 (60 races) | same |
  | Test | 2022–2024 (68 races) | same |

- **CV:** forward-chaining cross-validation inside the training seasons. Four folds score 2011–12, 2013–14, 2015–16 and 2017–18, each trained only on earlier seasons.
- **Intervals:** "[a, b]" is a 95% interval from resampling races. A difference counts only if it excludes 0.

---

## Inputs used across the runs

**The 18 final features** (section 4):

| Group | Features |
|---|---|
| grid | `grid`, `pit_lane_start` |
| qualifying | `gap_q1_pct`, `grid_minus_quali`, `reached_q3`, `no_q3_format`, `quali_missing` |
| standings | `drv_champ_pos`, `cons_points_share`, `drv_no_standing_yet`, `cons_no_standing_yet` |
| form | `drv_podium_rate_5`, `drv_avg_finish_5`, `drv_dnf_rate_5`, `cons_podium_rate_5` |
| context | `n_starters`, `drv_experience`, `drv_age` |

**The 21 model inputs** (section 5 onwards) = the 18 features + 3 missing-value flags: `q1_time_missing`, `drv_form_missing`, `cons_form_missing`.

**Rejected candidates** (section 4): `quali_pos`, `drv_points_share` and `cons_champ_pos` (near-duplicates of kept features); `home_race` (lowered the score).

---

## Run 1: Section 4 feature-selection trials (logistic regression)

**Setup:** model A rows, scored on validation. Temporary preprocessing: median fill + "was missing" flags + standardise everything. Groups were added one at a time.

| Trial | Inputs | Val PR-AUC (first version) | Val PR-AUC (after review fixes) | Decision |
|---|---|---|---|---|
| 1 | grid group (`grid`, `pit_lane_start`) | 0.6765 | 0.6765 | baseline |
| 2 | + qualifying (gap from Q1) | 0.7208 | 0.7210 | keep (+0.045; the only gain clearly above noise) |
| 2b | qualifying with gap from best-of-Q1/Q2/Q3 instead | 0.7093 | 0.7099 | keep the Q1 gap (difference within noise; 2006–09 race-fuel Q3) |
| 3 | + standings | 0.7339 | 0.7343 | keep (+0.013, interval includes 0) |
| 4 | + form (last 5 races) | 0.7362 | 0.7382 | keep (+0.004, interval includes 0) |
| 4b | form window 3 / 5 / 10 races | 0.7341 / 0.7362 / 0.7301 | 0.7351 / 0.7382 / 0.7302 | keep 5 as a middle value (differences within noise) |
| 5 | + context (`n_starters`, `drv_experience`, `drv_age`) | 0.7376 | **0.7384** | keep → **18 features** |
| 6 | + `home_race` | 0.7343 | 0.7351 | **drop** (−0.003, interval entirely below 0) |

**Redundancy choices** (training seasons):
- **`grid` kept over `quali_pos`:** they correlate at 0.97. The penalty signal stays in `grid_minus_quali` and the pace signal in `gap_q1_pct`.
- **Driver: championship position kept** over points share (0.854 vs 0.852 on its own).
- **Team: points share kept** over championship position (0.856 vs 0.826).

**What changed between the two versions (review fixes):**
- **`reached_q3`** used to mean "has a Q3 lap time". It now also counts drivers who advanced to Q3 without setting a time, which changed 114 rows. 2016 had 8 Q3 places; Q3 was cancelled at the 2015 US GP.
- **Team form** could use decades-old results when a team ID came back (Mercedes 2010 ← 1955). It now restarts after a gap of more than 550 days, which changed 141 rows.

The scores moved only slightly (0.7376 → 0.7384), and the final 18 features stayed the same.

---

## Run 2: Section 5 preprocessing experiments (logistic regression, 21 inputs)

**Setup:** the 18 features with the final preprocessing:
- missing-value flags;
- a missing Q1 time filled with a "slow lap" (training 95th percentile), and no qualifying record filled with the median;
- form filled with medians;
- `gap_q1_pct` capped at the 99th percentile of observed laps;
- continuous features standardised, flags left as 0/1.

Scored on validation.

| Experiment | Version A | Version B | Difference [95% interval] | Conclusion |
|---|---|---|---|---|
| Consistency check: median fill → scale vs scale → median fill | 0.7331 | 0.7330 | −0.0000 [−0.0007, +0.0009] | nearly equivalent; not counted as an experiment |
| **E1:** KNN fill → scale vs scale → KNN fill | 0.7368 | 0.7370 | +0.0002 [−0.0016, +0.0021] | the filled values change, the score doesn't |
| **E2 (model B):** cap on observed laps → fill vs fill → cap | 0.7236 | 0.7245 | +0.0009 [−0.0017, +0.0052] | the cap moves (9.02% vs 6.54%); score within noise. Kept: cap on observed laps |
| E2 (model A) | 0.7350 | 0.7348 | −0.0002 | nearly identical |
| Era: preprocessing fitted on 2003–2018 vs 1950–2018 | 0.7350 | 0.7350 | −0.0000 | fitted values move, score doesn't |
| INVALID demonstration: fitted on train + validation | 0.7350 | 0.7350 | −0.0000 | leak demonstration only |

**Transformation check** (each vs the final pipeline, 0.7350): `log(grid)` 0.7350, `1/grid` 0.7139, `log(1+experience)` 0.7286, `log(1+gap)` 0.7336, RobustScaler 0.7345. Every interval includes 0, so no transformation was added.

**Change from run 1:** the temporary preprocessing (0.7384) was replaced by the final 21-input pipeline (0.7350). The temporary version had 7 mostly duplicate flags and scaled the 0/1 flags too. The difference is within noise; the change was made for correctness, not score.

---

## Run 3: Section 6 development runs (not kept)

| Dev run | What happened | What was changed |
|---|---|---|
| Tie rule v1: unpaired one-standard-error rule | Every setting tied in every family, so the simplest was always chosen. Logistic regression got `C` 0.01: CV 0.674 vs best 0.704 | Switched to a **paired** rule: compare settings fold by fold |
| Tie rule v2: paired, but not monotone | A worse setting (LR `C` 1 + balanced weights, 0.7015) was kept while a better one (0.7032) was excluded. Probabilities summed to ~6.3 per race | **Closed upwards:** any setting scoring at least as well as a tie is also a tie (final rule v3) |
| Random forest with all CPU cores | Predictions differed in the last decimal places between calls | Random forest runs on 1 core (bit-for-bit reproducible) |
| CV jobs run one after another | The Model B searches took about 25 minutes | Settings × folds run in parallel processes (identical results) |

---

## Run 4: Section 6, first freeze (v1)

**Setup:**
- Four model families × models A and B, all on the 21 inputs.
- Settings chosen by CV with the paired one-SE rule (v3), simplest of the ties.
- Thresholds: best F1 on the out-of-fold predictions.
- Validation used once per model.

**Grids searched:**
- **Logistic regression (8):** `C` {0.01, 0.1, 1, 10} × class weight {none, balanced}.
- **Random forest (18):** depth {6, 10, none} × min leaf {5, 20, 50} × features {sqrt, 0.5}; 500 trees.
- **Gradient boosting (16):** learning rate {0.05, 0.1} × leaves {7, 15} × min leaf {20, 80} × trees {100, 300}.
- **Network (12, PyTorch):** hidden {(16), (32), (32,16)} × dropout {0, 0.2} × weight decay {1e-4, 1e-3}. Early stopping on the last two seasons of each fold's training period.

**Model A results:**

| Model | Chosen setting | CV PR-AUC | Val PR-AUC | Val ROC-AUC | Threshold | Val F1 | Val top-3 F1 |
|---|---|---|---|---|---|---|---|
| Rule "grid ≤ 3" | – | (CV F1 0.651) | – | – | – | 0.717 | – |
| Grid-only LR (baseline) | `grid` only | 0.646 | 0.676 | 0.917 | 0.363 | 0.705 | 0.717 |
| **Logistic regression (primary)** | `C` 1, no weights | 0.703 | **0.735** | 0.937 | 0.381 | 0.716 | 0.700 |
| Random forest | depth 6, leaf 50, sqrt | 0.684 | 0.733 | 0.936 | 0.299 | 0.712 | 0.700 |
| Gradient boosting | lr 0.05, 7 leaves, leaf 80, 100 trees | 0.693 | 0.721 | 0.938 | 0.387 | 0.725 | 0.717 |
| Neural network | (16), dropout 0, wd 1e-3, 144 epochs | 0.702 | 0.716 | 0.936 | 0.377 | 0.719 | 0.700 |

**Model B** (CV / validation PR-AUC):

| Model | Setting | CV | Val |
|---|---|---|---|
| Logistic regression | `C` 0.1 | 0.690 | 0.721 |
| Random forest | depth 6, leaf 50, sqrt | 0.684 | 0.731 |
| Gradient boosting | as model A | 0.692 | 0.727 |
| Network | (16), wd 1e-4, 112 epochs | 0.704 | 0.731 |

No A-vs-B difference is clear.

**Supporting runs (validation, descriptive only):**
- **Training start year** (1950 / 1970 / 1980 / 1994 / 2003 / 2010, all ending 2018):
  - Logistic regression: 0.724 / 0.724 / 0.723 / 0.723 / **0.735** / 0.732.
  - Gradient boosting: 0.727 / 0.723 / 0.722 / 0.723 / 0.721 / 0.720.
- **Same size as model A, mixed eras** (5 race samples): logistic regression 0.716 ± 0.009, gradient boosting 0.712 ± 0.010. At equal size, modern seasons are more useful.
- **Network seeds 42 / 43 / 44:** CV 0.700 ± 0.003, validation 0.721 ± 0.010.

**Outcome:**
- Primary model: logistic regression, model A.
- Fingerprint `53625c66…` (`outputs/frozen_models.json`, `outputs/model_artifacts/`).

---

## Run 5: Section 7, re-tuning and the second freeze (v2)

**What changed from v1:** a one-step grid extension wherever v1's best or chosen setting sat on the edge of its grid (rule fixed before running). The new values were:
- **Logistic regression:** `C` 100.
- **Random forest:** depth 4, min leaf 100.
- **Gradient boosting:** learning rate 0.02, 3 leaves, min leaf 10 / 160, 50 / 600 trees.
- **Network:** hidden (8), dropout 0.4, weight decay 1e-5 / 1e-2.

Only the new settings were run: 2 + 14 + 128 + 36 per model. They were chosen with the same folds and rule as v1. Validation wasn't used for any choice.

| Model | v1 setting → v2 setting | CV v1 → v2 | Val v1 → v2 |
|---|---|---|---|
| LR A | unchanged (`C` 1) | 0.703 → 0.703 | 0.735 → 0.735 |
| RF A | depth 6, leaf 50, sqrt → depth 4, leaf 100, 0.5 | 0.684 → 0.687 | 0.733 → 0.735 |
| HGB A | lr 0.05, 7 leaves, 100 trees → lr 0.1, 3 leaves, 50 trees | 0.693 → 0.694 | 0.721 → 0.733 |
| NN A | (16) → (8), wd 1e-3, 158 epochs | 0.702 → 0.706 | 0.716 → 0.738 |
| LR B | unchanged (`C` 0.1) | 0.690 → 0.690 | 0.721 → 0.721 |
| RF B | → depth 4, leaf 100, 0.5 | 0.684 → 0.689 | 0.731 → 0.730 |
| HGB B | → lr 0.1, 3 leaves, leaf 160, 100 trees | 0.692 → 0.690 | 0.727 → 0.729 |
| NN B | (16), wd 1e-4 → (8), wd 1e-3, 119 epochs | 0.704 → 0.701 | 0.731 → 0.731 |

**Outcome:**
- Changes went to simpler settings that tie the best.
- The primary model stays logistic regression, model A (the network tied it, and logistic regression is simpler).
- Fingerprint `2ebc3457…`, recorded **before** the test (`outputs/frozen_models_v2.json`, `outputs/model_artifacts_v2/`).

---

## Run 6: Section 7, the one-time test evaluation (2022–2024, frozen v2 models)

No training happened in this run: the frozen models only made predictions.

| Model | Test PR-AUC | ROC-AUC | Brier | Precision | Recall | F1 (threshold) | Top-3 F1 |
|---|---|---|---|---|---|---|---|
| Rule "grid ≤ 3" | – | – | – | 0.650 | 0.647 | 0.649 | – |
| Grid-only LR | 0.647 | 0.914 | 0.075 | 0.583 | 0.775 | 0.665 | 0.652 |
| **LR A (primary)** | **0.705** | 0.937 | 0.070 | 0.616 | 0.730 | 0.668 | 0.667 |
| RF A | 0.719 | 0.939 | 0.069 | 0.614 | 0.755 | 0.677 | 0.642 |
| HGB A | 0.718 | 0.938 | 0.068 | 0.597 | 0.799 | 0.683 | 0.642 |
| NN A | 0.698 | 0.935 | 0.072 | 0.621 | 0.716 | 0.665 | 0.627 |
| LR B | 0.712 | 0.937 | 0.070 | 0.585 | 0.809 | 0.679 | 0.657 |
| RF B | 0.725 | 0.941 | 0.070 | 0.641 | 0.735 | 0.685 | 0.667 |
| HGB B | 0.718 | 0.939 | 0.069 | 0.625 | 0.735 | 0.676 | 0.652 |
| NN B | 0.704 | 0.937 | 0.070 | 0.551 | 0.819 | 0.659 | 0.627 |

**Comparisons:**
- **Against the grid-only baseline:** every model-A family is clearly better (+0.05 to +0.07 PR-AUC).
- **Against the primary model:** no family is clearly different.
- **Model B vs model A:** no clear difference.

**The v1 models on the test seasons** (comparison only, no decision; PR-AUC v1 → v2):

| Model | A | B |
|---|---|---|
| Logistic regression | identical | identical |
| Random forest | 0.713 → 0.719 | 0.716 → 0.725 |
| Gradient boosting | 0.710 → 0.718 | 0.706 → 0.718 |
| Neural network | 0.700 → 0.698 | 0.692 → 0.704 |

**Official output (7.10): the 3 most likely podium finishers per race**, from the primary model.

| Period | Pick | Top-3 F1 | Correct drivers per race (of 3) | All 3 right |
|---|---|---|---|---|
| Validation | top 3 on the starting grid | 0.717 | 2.15 | 20 / 60 |
| Validation | official (LR A) | 0.700 | 2.10 | 21 / 60 |
| Test | top 3 on the starting grid | 0.652 | 1.96 | 11 / 68 |
| Test | **official (LR A)** | **0.667** | **2.00** | **16 / 68** |

On the test, official minus grid is +0.044 correct drivers per race, interval [−0.088, +0.176], so it isn't clear. As a podium pick, the model matches the front of the grid; its clear advantage is in ranking and probabilities.

---

## Run 7: Section 8 exploratory round (after the test; historical CV folds only)

**Status:** validation and test seasons were not used, and no frozen model changed.

**Setup:**
- Baseline: logistic regression, `C` 1.
- 8 configurations × 4 folds = 32 fits.
- Decision rule fixed in advance: top-3 F1 gain > 0, gain in at least 3 of 4 folds, PR-AUC not lower, and an interval excluding 0.

| Configuration | Extra inputs | CV PR-AUC | CV top-3 F1 | Correct / race | Exact-3 rate | Top-3 gain | Folds improved (top-3 / PR-AUC) | Rule |
|---|---|---|---|---|---|---|---|---|
| C0 baseline | – (21 inputs) | 0.703 | 0.666 | 2.00 | 0.265 | – | – | – |
| C1 | `log(grid)` | 0.716 | 0.668 | 2.00 | 0.252 | +0.002 | 2 / **4** | no |
| C2 | grid spline (4 knots) | 0.713 | 0.668 | 2.00 | 0.246 | +0.002 | 2 / 3 | no |
| C3 | team rank within the race | 0.705 | 0.660 | 1.98 | 0.259 | −0.006 | 0 / 3 | no |
| C4 | qualifying-pace z-score within the race | 0.698 | 0.670 | 2.01 | 0.264 | +0.004 | 2 / 2 | no |
| C5 | driver-form rank within the race | 0.703 | 0.658 | 1.97 | 0.240 | −0.008 | 0 / 1 | no |
| C6 | C1 + C3 + C4 + C5 | 0.714 | 0.674 | 2.02 | 0.252 | +0.008 | 3 / 3 | no |
| C7 | gradient boosting (frozen v2 setting) | 0.694 | 0.659 | 1.98 | 0.233 | −0.007 | 2 / 2 | no |

**Outcome:**
- No configuration met the rule, so the frozen model stays.
- `log(grid)` was the only consistent change: PR-AUC improved in all 4 folds, but not the podium pick.
- It is a candidate only for a future version tested on new seasons (2025+).

---

## Run 8: Four proposed enhancements (exploratory, after the test; historical CV folds only)

**Status:** run **outside the notebook** during development (scratch scripts that reuse the notebook's sections 4–5 and 8 code). Validation and test seasons were not used, and no frozen model changed. Same baseline, metrics and race-resampling intervals as Run 7.

| Proposal | Configuration | CV PR-AUC | CV top-3 F1 | Correct / race | Exact-3 rate | Top-3 gain [95% interval] | Folds improved (top-3 / PR-AUC) |
|---|---|---|---|---|---|---|---|
| – | C0 baseline (LR `C` 1, 21 inputs) | 0.703 | 0.666 | 2.00 | 0.265 | – | – |
| P1 race-aware ranking | pairwise ranker (LR on within-race podium vs non-podium pairs) | 0.683 | 0.655 | 1.97 | 0.239 | **−0.011 [−0.020, −0.002], clearly worse** | 0 / 0 |
| P1 | listwise ranker (softmax over each race, linear score) | 0.686 | 0.662 | 1.98 | 0.240 | −0.004 [−0.019, +0.011] | 0 / 0 |
| P1 | listwise ranker + `log(grid)` | 0.688 | 0.662 | 1.98 | 0.240 | −0.004 (same picks as above) | 0 / 0 |
| P2 grid shape | LR + `log(grid)` | **0.716** | 0.668 | 2.00 | 0.252 | +0.002 [−0.013, +0.015] | 2 / **4** |
| P3 better form | LR + exponentially weighted form (half-life 3: driver podiums, driver finish, team podiums) | 0.703 | 0.668 | 2.00 | 0.252 | +0.002 [−0.011, +0.015] | 1 / 1 |
| P3 | LR + teammate-relative Q1 pace (previous 5 races) | 0.703 | 0.666 | 2.00 | 0.265 | 0.000 (identical picks) | 0 / 2 |
| P3 | LR + both | 0.703 | 0.668 | 2.00 | 0.252 | +0.002 [−0.011, +0.015] | 2 / 1 |

**P4: selecting settings by a race-level metric.** This was nested over time: at folds 2–4, the setting was chosen using only earlier folds, by PR-AUC or by top-3 F1, then scored on the next fold.

| Family | Chosen by | Next-fold top-3 F1 | Correct / race | Next-fold PR-AUC |
|---|---|---|---|---|
| Logistic regression (8 settings) | PR-AUC | 0.674 | 2.02 | 0.718 |
| Logistic regression | top-3 F1 | 0.677 | 2.03 | 0.718 |
| Gradient boosting (16 settings) | PR-AUC | **0.699** | **2.10** | 0.700 |
| Gradient boosting | top-3 F1 | 0.688 | 2.06 | 0.694 |

**Outcome:**
- No enhancement improves the podium pick; the pairwise ranker makes it clearly worse.
- `log(grid)` again improves PR-AUC only.
- Choosing settings by top-3 F1 gives no consistent benefit and is worse for gradient boosting, because the metric is noisier.
- The frozen model is kept.

---

## Run 9: New information from our own data (exploratory, after the test; historical CV folds only)

**Status:** run **outside the notebook**, like Run 8. Validation and test seasons were not used, and no frozen model changed.

**New features**, each using only earlier races, or this race's own qualifying (300 rows recomputed by hand):
- **Recent team qualifying form:** the team's mean Q1 gap and mean qualifying position ÷ field size over its previous 5 races, within team-history segments.
- **This weekend's trend:** the current Q1 gap minus that recent mean.
- **Driver's circuit history:** the driver's earlier podium rate at the circuit (smoothed with 2 pseudo-starts at 14%) and number of earlier starts there.
- **Circuit × grid-band history:** the earlier podium rate from the same grid band (1–3 / 4–6 / 7–10 / 11+) at that circuit, smoothed towards the band's overall earlier rate (10 pseudo-starts).

| Configuration | CV PR-AUC | CV top-3 F1 | Correct / race | Exact-3 rate | Top-3 gain [95% interval] | PR-AUC gain [95% interval] | Folds improved (top-3 / PR-AUC) |
|---|---|---|---|---|---|---|---|
| C0 baseline (LR) | 0.703 | 0.666 | 2.00 | 0.265 | – | – | – |
| LR + recent team qualifying form | 0.705 | 0.662 | 1.98 | 0.252 | −0.004 [−0.013, +0.002] | +0.002 [−0.001, +0.005] | 0 / 4 |
| LR + team qualifying form + trend | 0.705 | 0.662 | 1.98 | 0.252 | −0.004 [−0.013, +0.002] | +0.002 [−0.002, +0.004] | 0 / 3 |
| LR + driver history at the circuit | 0.701 | 0.659 | 1.98 | 0.258 | −0.006 [−0.013, +0.000] | −0.003 [−0.008, +0.002] | 0 / 1 |
| LR + circuit × grid-band history | 0.699 | 0.664 | 1.99 | 0.259 | −0.002 [−0.019, +0.013] | −0.004 [−0.013, +0.003] | 2 / 1 |
| LR + both circuit histories | 0.697 | 0.666 | 2.00 | 0.265 | −0.000 [−0.017, +0.017] | −0.007 [−0.015, +0.004] | 2 / 1 |
| LR + all new features | 0.697 | 0.659 | 1.98 | 0.259 | −0.007 [−0.021, +0.008] | −0.006 [−0.012, +0.006] | 1 / 1 |
| Gradient boosting (frozen v2 setting) | 0.694 | 0.659 | 1.98 | 0.233 | – (reference for the next row) | – | – |
| Gradient boosting + all new features | 0.694 | 0.666 | 2.00 | 0.246 | +0.006 [−0.004, +0.019] vs gradient boosting | +0.000 [−0.006, +0.006] | 2 / 2 |

**Outcome:**
- Recent team qualifying form adds nothing beyond this weekend's qualifying.
- Circuit history is too sparse (a few starts per driver per circuit) and slightly hurts.
- With every new feature, gradient boosting only matches the logistic-regression baseline.
- **Not tested:** practice pace, weather forecasts and tyre plans aren't in the dataset. Practice pace (e.g. via the FastF1 library, about 2018 onwards) is the most promising source for a future version.

---

## Run 10: Section 9, feature-group ablation (historical CV folds only)

**Setup:**
- **Groups removed, one at a time:** the five groups, each with its missing-value flags. Grid (2 inputs), qualifying (6), standings (4), form (6), context (3).
- **Refits:** each model is genuinely refitted inside each of the four historical folds, with preprocessing re-fitted on the fold's training seasons and the frozen v2 settings (no re-tuning). Groups are removed *after* preprocessing, because the imputer needs all 18 raw columns.
- **Fits:** 48 in total (2 families × 6 variants × 4 folds), about 4 s.
- **Data:** validation and test seasons not used; frozen models unchanged.
- **Check:** the "full" variants reproduce the Run 7 baselines exactly.
- **"Drop"** = full model − model without the group. Intervals come from resampling the 158 fold races.

| Model | Variant | CV PR-AUC | ROC-AUC | Top-3 F1 | Correct / race | Exact-3 | PR-AUC drop [95%] | Correct-picks drop [95%] | Folds with lower PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| LR | full model | 0.703 | 0.938 | 0.666 | 2.00 | 0.265 | – | – | – |
| LR | without grid | 0.645 | 0.926 | 0.611 | 1.83 | 0.132 | **+0.059 [+0.037, +0.083]** | **+0.165 [+0.079, +0.260]** | 4 / 4 |
| LR | without qualifying | 0.699 | 0.936 | 0.676 | 2.03 | 0.253 | +0.004 [+0.001, +0.010] | −0.032 [−0.070, +0.000] | 3 / 4 |
| LR | without standings | 0.701 | 0.936 | 0.657 | 1.97 | 0.239 | +0.002 [−0.011, +0.010] | +0.025 [−0.013, +0.070] | 2 / 4 |
| LR | without form | 0.687 | 0.934 | 0.642 | 1.93 | 0.207 | **+0.017 [+0.002, +0.033]** | **+0.070 [+0.006, +0.136]** | 4 / 4 |
| LR | without context | 0.704 | 0.937 | 0.653 | 1.96 | 0.227 | −0.001 [−0.016, +0.006] | +0.038 [−0.022, +0.101] | 2 / 4 |
| HGB | full model | 0.694 | 0.940 | 0.659 | 1.98 | 0.233 | – | – | – |
| HGB | without grid | 0.617 | 0.922 | 0.596 | 1.79 | 0.126 | **+0.077 [+0.046, +0.101]** | **+0.190 [+0.076, +0.307]** | 4 / 4 |
| HGB | without qualifying | 0.692 | 0.941 | 0.664 | 1.99 | 0.240 | +0.002 [−0.010, +0.012] | −0.013 [−0.044, +0.013] | 2 / 4 |
| HGB | without standings | 0.696 | 0.939 | 0.663 | 1.99 | 0.240 | −0.002 [−0.012, +0.005] | −0.013 [−0.044, +0.019] | 2 / 4 |
| HGB | without form | 0.680 | 0.937 | 0.644 | 1.93 | 0.213 | +0.013 [−0.004, +0.024] | +0.044 [−0.010, +0.095] | 4 / 4 |
| HGB | without context | 0.694 | 0.941 | 0.659 | 1.98 | 0.233 | −0.000 [−0.010, +0.003] | 0.000 [−0.019, +0.013] | 2 / 4 |

**Outcome:**
- **Grid** is by far the most important group, and its loss is clear for both models.
- **Form** is the second clear group for logistic regression; for gradient boosting it has the same size, but the interval includes 0.
- **Qualifying, standings and context** add little on top of the others, because they overlap with grid and form. This measures each group's *unique* contribution.

---

## Section 10: Explainability (no training; frozen primary model, validation 2019–2021)

No model was trained or changed here; the frozen logistic regression A (v2) was explained as saved.

**Group importance under each method** (rank 1 = most important):

| Method | Grid | Form | Standings | Qualifying | Context |
|---|---|---|---|---|---|
| Ablation, LR, PR-AUC drop | 1 | 2 | 4 | 3 | 5 |
| Ablation, LR, correct-picks drop | 1 | 2 | 4 | 5 | 3 |
| Ablation, gradient boosting, PR-AUC drop | 1 | 2 | 5 | 3 | 4 |
| Grouped permutation, PR-AUC (shuffled across rows) | 1 (0.417) | 2 (0.074) | 3 (0.051) | 4 (0.047) | 5 (0.005) |
| Grouped permutation, top-3 F1 (shuffled within race) | 1 (0.375) | 2 (0.038) | 3 (0.031) | 4 (0.024) | 5 (0.001) |
| SHAP, LR: mean of the group's total contribution (log-odds) | 1 (1.62) | 2 (0.51) | 3 (0.42) | 4 (0.40) | 5 (0.26) |
| Tree SHAP, gradient boosting: same measure | 1 (0.85) | 2 (0.53) | 3 (0.36) | 4 (0.10) | 5 (0.06) |
| Sum of the absolute LR weights | 2 | 3 | 4 | 1 | 5 |

**Main points:**
- Grid is ranked 1st and form 2nd by every method except the raw weights.
- Qualifying's large weights belong to flags that rarely apply or that offset each other.
- Age and experience cancel each other, which is why context looks moderate in SHAP measure (A) but negligible elsewhere.
- LIME's local fits are weak (R² 0.14–0.55) and agree only partly with SHAP (1–3 of the top 5 inputs), so SHAP is the reference.
- Details are in README 7.7.

---

## Summary: what changed from run to run, and why

| Step | Change | Why | Effect |
|---|---|---|---|
| Run 1 trials | Added feature groups one at a time; dropped `home_race` | Evidence-based feature selection | Validation 0.677 (grid) → 0.738 (18 features) |
| Run 1 review | Fixed `reached_q3` and stale team form | Wrong values for 114 / 141 rows | Scores almost unchanged; features now correct |
| Run 2 | Final preprocessing (flags, meaningful fills, gap cap, scaling) | Correctness and no leakage | 0.7384 → 0.7350, within noise |
| Run 3 (dev) | Tie rule made paired and monotone; random forest on 1 core; parallel CV | Selection-logic errors; reproducibility; runtime | Sensible selections; identical reruns |
| Run 4 (v1) | 4 families × 2 training windows, tuned by CV | Brief requires ≥ 2 statistical models + a network | All about 0.72–0.74 on validation; logistic regression primary |
| Run 5 (v2) | One-step grid extension where v1 sat on a grid edge | Check that the optimum isn't just outside the grid | Simpler tree / network settings; CV ±0.006; same primary |
| Run 6 (test) | One-time evaluation of the v2 freeze | Brief requires test results | Primary 0.705 PR-AUC; every family clearly beats the grid baseline |
| 7.10 | Top-3 per race made the official output | Every race has exactly 3 podiums | Test: 2.00 correct per race, 16 / 68 races fully right |
| Run 7 (exploratory) | Grid shape + race-relative features | Try to improve the podium pick cheaply | No consistent improvement; frozen model kept |
| Run 8 (exploratory, outside the notebook) | Race-aware rankers, `log(grid)`, better form features, selecting by top-3 F1 | Proposed enhancements aimed at the podium pick | None improves the podium pick; the pairwise ranker is clearly worse; `log(grid)` helps PR-AUC only |
| Run 9 (exploratory, outside the notebook) | Recent team qualifying form, circuit history (driver and circuit × grid band) | New information available in our own data | No improvement; circuit history slightly hurts; frozen model kept |
| Run 10 (Section 9 ablation) | Each feature group removed in turn and the model refitted (LR and HGB, frozen settings) | Brief requires a feature-group ablation | Grid clearly the most important (−0.16 to −0.19 correct picks per race without it); form second; qualifying / standings / context largely overlap with them |
| Section 10 (explainability) | No training: weights, SHAP, LIME, permutation and race explanations of the frozen primary model | Brief requires XAI | Grid and form dominate every method; SHAP is exact for this model; LIME is unreliable here |
