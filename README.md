# ACL-2: F1 Podium Prediction (Milestone 1)

Predict whether a Formula 1 driver finishes on the **podium** (`positionOrder <= 3`), using only information available **after qualifying and before the race starts**.
Data: Kaggle / Ergast *Formula 1 World Championship (1950–2024)*, 14 CSVs.
Grain: **one row = one driver in one race**, i.e. a unique (`raceId`, `driverId`).

This README is the project log. It records **what was done, why, and what was tried and changed**, so a new working session (or a teammate) can continue without re-deriving anything.

---

## 1. Repository layout

| Path | What it is |
|---|---|
| `f1_podium_prediction.ipynb` | **The deliverable.** One run-all notebook (the brief requires a single notebook). The committed copy is the **executed** run of sections 1–10 (commit `f5660ee`); the generator now also contains sections 11–12, and the executed notebook with them will replace this copy once a verified full run is saved (see 7.8 "Status"). |
| `tools/build_notebook.py` | **Generator** that writes every markdown and code cell of the notebook. The notebook was built from this script. |
| `tools/run_notebook.py` | Executes the notebook top to bottom with `jupyter_client` and stores the outputs (no `nbconvert` needed). |
| `outputs/anomalies/` | CSVs written by notebook section 1.12: one per raw table, plus `_summary.csv`. |
| `outputs/clean_data/` | **The cleaned dataset**, written by notebook section 2.13: 15 CSVs (`model_base`, `results_full`, every cleaned table, the `*_prerace` standings), `schema.json` (column types), `manifest.csv` (rows + SHA-256) and its own `README.md`. Load with `load_clean("<table>")` to get the exact types back. Rewritten on every run; never edit by hand. |
| `outputs/final_features.csv` | **The final feature table**, written by notebook section 4.7: every row of `feat` (24,645, all seasons 1950–2024) × 23 columns = `raceId`, `driverId`, `year`, `split`, target `podium` + the 18 `FINAL_FEATURES` in order. Missing values are empty cells (no filling or scaling: that is section 5, fitted on training seasons only). Rows sorted by race, grid, `driverId`. Read with `pd.read_csv(path, keep_default_na=False, na_values=[""])`. Rewritten on every run (on Kaggle under `/kaggle/working/outputs/`). |
| `outputs/frozen_models.json` | **The frozen section-6 models**, written by notebook 6.11: for each of the 8 final models (4 families × model A and B) and the 2 baselines, the chosen settings, pre-processing, threshold, training seasons, seed / epochs, CV and validation PR-AUC, and a SHA-256 fingerprint of the whole record. No data and no test metric. Rewritten on every run (identical as long as nothing changes). |
| `outputs/model_artifacts/` | **The fitted frozen models as files**, written by notebook 6.12 on every run: one `cloudpickle` file per scikit-learn model (fitted pre-processing + model), `*_preprocessor.pkl` + `*_weights.pt` (PyTorch `state_dict`) per network, the grid-only baseline, `reference_validation_predictions.npz`, `manifest.json` (files + SHA-256, full-precision thresholds incl. exact hex, architecture/epochs, input columns, frozen fingerprint, library versions) and `load_artifacts.py`. ~6 MB (mostly the two random forests). See README 7.3 "Model artifacts". |
| `outputs/frozen_models_v2.json`, `outputs/model_artifacts_v2/` | **The second freeze (v2, section 7)**: same structure as v1 (record + fingerprint; fitted models, manifest, reference validation predictions, loader). The v2 record also stores the v1 fingerprint and a `changed from v1` flag. **The test evaluation (7.5) uses these files.** The v1 files stay as the record of the first freeze. |
| `outputs/podium_inference.py` | **The inference interface** (`PodiumPredictor`), written by notebook section 11 and usable outside the notebook: `PodiumPredictor('outputs/model_artifacts_v2').predict_proba(records)` / `.predict_race(entries)`. Loads only the frozen v2 primary model (checksums + fingerprint). |
| `.gitignore` | Ignores `__pycache__/` (bytecode caches created when the notebook loads `load_artifacts.py` / `podium_inference.py`). |
| `../Dataset.zip` | Raw data (one folder above the repo, **not** in git). Read directly from the zip, never extracted. |
| `../GUC_..._2026-09-30....pdf` | The milestone brief (one folder above the repo). |

### How to rebuild and run

```bash
cd ACL-2
python tools/build_notebook.py f1_podium_prediction.ipynb   # regenerate cells from the generator
python tools/run_notebook.py  f1_podium_prediction.ipynb    # execute top to bottom, save outputs
```

- **Important:** running the generator **overwrites** the notebook. Pick one workflow:
  - keep editing `tools/build_notebook.py` and regenerate (that's how sections 1–3 were made), **or**
  - edit the notebook directly from now on and stop using the generator.
- The data path is auto-detected: env var `F1_DATA` → Kaggle mount `/kaggle/input/formula-1-world-championship-1950-2020` → `../Dataset.zip` → `Dataset.zip` → `../Dataset` / `Dataset` / `data/raw`. It works on Kaggle (submission requirement) and locally.
- Environment: the notebook prints its versions in 6.1. Last full run: Python 3.14, numpy 2.4, pandas 2.3, scikit-learn 1.8, PyTorch 2.10 (CPU), joblib (parallel CV jobs). **Do not install or uninstall packages while the notebook runs:** on 18 Oct torch was uninstalled mid-run, so a new parallel worker could not import it (`BrokenProcessPool`); torch 2.10.0+cpu was reinstalled (all versions again equal the ones recorded with the v2 freeze). The CV runner now also falls back to running the jobs in the main process if a worker dies (`F1_N_JOBS=1` forces serial runs). Earlier sections were first built with Python 3.10 / numpy 2.2. SHAP and LIME are installed for the XAI section; XGBoost/LightGBM/Keras are not used.
- On Windows, set `PYTHONIOENCODING=utf-8` when printing names with accents from a script (the terminal codepage otherwise fails on e.g. *Räikkönen*).

### Git
- Work happens on branch **`amr`** (repo `Farah-Essam/ACL-2`; earlier work on `youssef`, merged into `amr`).
- Commits so far: audit + cleaning + questions (`d7fda8a`); cleaning review fixes, Indianapolis 500 removal, cleaned-dataset export and the approved section-4 plan (`45d54b0`); section 4 feature selection & engineering (16 Oct 2026); section-4 review fixes + feature export (`bce49fd`); sections 5–6 (`a3ac4f7`); sections 7–8 (`a8c1dcc`); sections 9–10 (`f5660ee`); final compliance round: sections 11–12, accuracy and metric choice, season pin, worker fallback, docs (18 Oct 2026). `Project.code-workspace` is a local editor file and is not committed.
- Commit or push only when the user asks.

---

## 2. Requirements from the brief (summary)

- **Cleaning & audit:** EDA before and after cleaning (with plots); sentinels (`\N`) handled; semantic typing (dates, durations → seconds); grain fixed to unique (`raceId`, `driverId`) with a documented duplicate rule; PKs unique and non-null; FKs reconcile; **raw CSVs immutable** (clean copies only).
- **3 data-engineering questions** (multi-table joins, with a plot, answer, and a justification of grain, missingness and cutoff):
  1. Which circuits convert a front-row start into a podium most often, and does the ranking change after 2014? Distinguish qualifying position from grid.
  2. Home-country podium effect, **controlling for starting position**.
  3. Constructors with the highest **mechanical**-retirement rate, 2014–2021 vs 2022–2024 (crash, driver and regulation causes excluded).
- **Features:** justified by EDA; **leakage-safe form feature (past races only)**; proper train/val/test split.
- **Models:** ≥ 2 statistical models + 1 shallow NN on the same table; ROC-AUC, PR-AUC, F1 on validation **and** test; thresholds and hyperparameters chosen without the test set; baseline required; ≥ 3 modeling attempts.
- **Experiments:** feature-group ablation; ≥ 2 preprocessing-order experiments.
- **XAI:** permutation importance + SHAP (global); LIME and/or local SHAP; explicitly **not causal**.
- **Deliverables:** run-all notebook, inference function (complete + partially-missing record), PDF report, Kaggle notebook link, GitHub repo link. **Deadline: 18 Oct 2026, 23:59 in the brief; extended by the team to Mon 19 Oct 2026, 23:59 (confirm against the official course announcement).** Submission form: https://forms.gle/esDxHsvfzffYHVAZ9 (Kaggle link, GitHub link, PDF report).
- Evaluation: every team member can be asked to justify any step. "An AI suggested it" isn't a justification, so every choice needs evidence in the notebook.

---

## 3. Notebook contents and status

| Section | Status | Content |
|---|---|---|
| **1. Data Audit** (1.0–1.13) | done | Read-only audit, anomaly CSV export, raw-file fingerprint check |
| **2. Data Cleaning** (2.1–2.13) | **closed** (review fixes applied 15 Oct 2026; 2.13 exports the cleaned dataset to `outputs/clean_data/`) | All cleaning applied in memory on copies. 2.9.1 status mapping with row counts; 2.10 re-validation (all 21 FKs); 2.10.2 proof that standings are post-race; 2.11 six before/after plot pairs |
| **3. Data-Engineering Questions** (3.1–3.3) | done | Q1 (+ two robustness sections), Q2, Q3, each with plots and plain-language answers |
| **4. Feature Selection & Engineering** (4.1–4.7) | done (16 Oct 2026; review fixes the same day) | split, 22 candidate features with leakage asserts, 4.2.1 checks of the two corrected features (`reached_q3`, team form), evidence plots (train only), redundancy, trials on validation, final 18-feature table, 4.7 export to `outputs/final_features.csv`. Results in README 7.1 |
| **5. Pre-processing** (5.1–5.8) | done (17 Oct 2026) | missing-value audit, `F1Imputer` (indicators + fills + `gap_q1_pct` cap), scaling, order experiments E1/E2 (+ checks), final pipelines for models A and B, leakage checks. Results in README 7.2 |
| **6. Predictive Modeling** (6.1–6.11) | done (17 Oct 2026); **frozen**, test not evaluated | forward-chaining CV (4 folds), 4 families × models A and B, OOF thresholds, validation once, A vs B + training-window experiments, freeze. Results in README 7.3 |
| **7. Final optimisation, second freeze, test evaluation** (7.1–7.10) | done (17 Oct 2026) | grid-edge extension (rule-based), re-tuning with the same CV and rule, v2 freeze (`frozen_models_v2.json`, `model_artifacts_v2/`), **one-time test evaluation 2022–2024**, interpretation, 7.10 official race-level output (top 3 per race). Results in README 7.4 |
| **8. Exploratory (post-test)** | done (17 Oct 2026) | 8 configurations on the historical folds only (grid shape, race-relative features, GB reference); no change to the frozen models. Results in README 7.5 |
| **9. Feature-group ablation** | done (17 Oct 2026) | leave-one-group-out refits of LR and HGB (frozen v2 settings) on the historical folds. Results in README 7.7 |
| **10. Explainable AI** | done (17 Oct 2026) | LR weights / odds ratios, global + local SHAP, complete-race top-3 explanations, LIME (supplementary), individual + grouped permutation importance, tree SHAP for HGB, method comparison. Results in README 7.7 |
| **11. Final inference** (11.1–11.5) | done (18 Oct 2026) | `PodiumPredictor` (written to `outputs/podium_inference.py`): frozen primary model, schema checks, single records (complete / partially missing) and complete races (top 3); reproduces every frozen validation/test probability and every race's official picks exactly. Results in README 7.8 |
| **12. Known limitations and conclusions** | done (18 Oct 2026) | consolidated, evidence-based limitations (data/era, models, explanations, reproducibility) and conclusions |
| PDF analytical report | see README 7.8 | built from the executed notebook's outputs |

Variables available after running sections 1–4:
- `RAW`: dict of raw tables, **all strings** (`dtype=str`, `keep_default_na=False`).
- `NAIVE`: default `pd.read_csv` (comparison only).
- `CLEAN`: cleaned tables (dict).
- `results_full`: all cleaned driver-races 1950–2024 (24,645 rows, **without the Indianapolis 500**), **with** post-race columns; used for the questions.
- `INDY500_RESULTS`: the 381 cleaned Indianapolis 500 rows set aside in 2.6.1, used only by the Q2 robustness column.
- `model_base`: 2003+, pre-race columns + target only; the start of the modeling table.
- Helpers: `is_missing`, `duration_to_seconds`, `log_step`, `STATUS_GROUPS`, `STATUS_CATEGORIES`, `NON_START_STATUSES`, `RULE_B_OVERRIDES`, `UNUSED_TABLES`, `DEFERRED_TABLES`, `NATIONALITY_TO_COUNTRY`, `season_order`, `CLEAN_DATA_DIR`, `load_clean(name)` (reads an exported table back with its exact types).
- Section 4: `feat` (all 24,645 rows 1950–2024: keys, `year`, `split`, `podium`, every candidate feature), `season_split(year)`, `FEATURE_GROUPS`, `ALL_CANDIDATES`, `FINAL_FEATURES` (the 18 chosen), `TRAIN_A` / `VAL` (model-A training rows, validation rows), `TRIALS` (every trial's scores), `form_features(n)`, `run_trial`, `gain_interval`, `TEAM_RACES` / `TEAM_GAP_DAYS` (constructor history segments for team form), `ELIMINATION_FORMAT` (the two 2016 races with 8 Q3 places), `FINAL_FEATURES_PATH` / `final_features_df` (the 4.7 export).
- Section 5: `data` (the CSV read back), `TRAIN_A_ROWS` / `TRAIN_B_ROWS` / `VAL_ROWS`, `CONTINUOUS_FEATURES` / `BINARY_FEATURES` / `INDICATORS`, `F1Imputer`, `make_preprocessor(scaled=...)`, `fit_on_training(pipe, rows, label)` (refuses non-training rows), **`PREPROCESSORS`** (4 fitted pipelines), **`MODEL_INPUTS`**, `EXPERIMENTS` / `experiments_df`, `evaluate_preprocessing`, `pr_auc_difference`.
- Section 6: `ARTIFACT_DIR`, `load_model_artifacts(folder)`, `LOADER_SOURCE` (6.12), `FOLDS` / `fold_rows(model, k)`, `fit6` (training-rows guard + log), `GRIDS`, `SEARCH` (every setting's fold scores), `CHOSEN`, `OOF` (out-of-fold predictions), `THRESHOLDS`, `FINAL6` (fitted final models: `{'pre', 'model'}`), `VAL_PRED`, `ShallowNet` (PyTorch), `PRIMARY_MODEL`, **`FROZEN`** / `FROZEN_MODELS` / `FROZEN_FINGERPRINT`, **`predict_frozen(key, rows)`** (probabilities + threshold rule + top-3 rule; for the test evaluation), `FIT6_LOG` / `PREDICT6_LOG`.
- Section 7: `GRIDS_V2`, `SEARCH_V2`, `CHOSEN_V2`, `FINAL_V2`, `THRESHOLDS_V2`, `VAL_PRED_V2`, `PRIMARY_MODEL_V2`, `FROZEN_V2` / `FROZEN_V2_FINGERPRINT`, `ARTIFACT_DIR_V2`, `save_model_artifacts` / `verify_model_artifacts` (shared with 6.12), `TEST_ROWS`, `TEST_PRED`, `TEST_THRESHOLDS`, `TEST_BOOT`, `test_table`, `four` (train/CV/val/test), `comparisons`, `season_table`.

---

## 4. Audit findings (section 1): the evidence behind every decision

- **Keys:** every surrogate `...Id` is unique and non-null; **all 23 FKs reconcile (0 orphans)**. The only failing key is `results (raceId, driverId)`, our grain.
- **Sentinels:** `\N` mostly means "doesn't exist", not "lost" (e.g. Q3 time missing = eliminated earlier; `results.time` only for lead-lap finishers). **`qualifying.q2/q3` also use empty strings (68 cells)**, so an inconsistent sentinel. One padded value: `drivers.nationality = 'Argentinian '`.
- **Typing:** dates are clean `YYYY-MM-DD`. Lap and qualifying times are `m:ss.sss`, some `h:mm:ss.sss`; pit durations over 60 s are `m:ss.sss`. The parser `duration_to_seconds` handles 1–3 parts and reproduces the official `milliseconds` exactly (0 ms difference on 589k laps). **`results.time` mixes the winner's total time with `+gap` values for others: never parse it.**
- **Duplicates:** 85 duplicated (raceId, driverId) pairs / 176 rows (0.66%), 1950–1978, mostly shared drives. Races could show 4–7 "podium" rows. Also **cross-driver shared drives** (two different drivers credited with one car's result, e.g. Indy 500 1951/1955): 166 rows up to 1964. Those aren't grain duplicates.
- **Qualifying coverage:** 0% before 1994; 1994–95 near complete; **1996–2002 patchy** (whole races missing, as low as 6% in 2001); **~100% from 2003**. 2003-2004 and most of 2005 used a single timed lap (q1 only). The first 6 races of 2005 used two sessions whose times were added together (q1 + q2). From 2006, qualifying uses the Q1/Q2/Q3 knockout.
- **Grid vs qualifying:** they match only ~76% of the time (27% differ since 2003, because of penalties). **`grid = 0` means no grid slot**: before ~1994 mostly DNQ/DNPQ/Withdrew (never started); in the modern era mostly **pit-lane starts** (they raced).
- **Labels:** driver nationalities are demonyms (`British`), circuit countries are names (`UK`, plus both `USA` **and** `United States`). **No nationality string equalled a country string.**
- **Encoding:** suspected "web-scraping encoding" issue checked: **files are valid UTF-8 with 0 mojibake**. The `Aut�dromo` seen once was only the Windows terminal display.
- **Leakage map (1.10):** post-race columns are `position*`, `points`, `laps`, `status`, `time`, `milliseconds`, `fastestLap*`, `rank`, all of `lap_times` / `pit_stops`, and **standings at the current `raceId`** (they already include that race's points).
- **Anomaly CSVs (1.12):** `outputs/anomalies/<table>_anomalies.csv`, one line per (raw row, anomaly) with `csv_line`, `anomaly_type`, `anomaly_detail` and the original columns. Notable extras: standings `positionText` letter codes (`D` = disqualified, `E` = excluded); one Q1 lap of 1002.6 s (Japan 1995); 626 laps ≥ 10 min and 517 pit stops ≥ 60 s (red flags).

---

## 5. Cleaning decisions (section 2): what and why

The user wrote the base plan. Claude's review added items; the user approved "your plan + my recommendations".

| # | Step | Decision | Why / evidence |
|---|---|---|---|
| 1 | Working copies | `CLEAN` built from `RAW`; raw verified unchanged by SHA-256 at the end of sections 1 and 2 | brief: raw immutable |
| 2 | Unused tables | not carried: `seasons` (years already in `races.year`), `lap_times`, `pit_stops` (recorded during the race). Cleaned but not used: `constructor_results` (team points already in results/standings, status 99.9% empty); `sprint_results` deferred (optional feature: sprint finish + `has_sprint`) | 1.10, 2.1 |
| 3 | Sentinels | `\N` **and** empty/whitespace → `NaN`; strip text | 160,076 `\N` cells + 68 empty + 1 padded |
| 4 | Encoding | **nothing to fix**; documented as a check instead | 0 mojibake (see trial log) |
| 5 | Dropped columns | circuits `circuitRef, lat, lng, alt, url`; constructors `url`; drivers `number, code, url`; races `url`, fp1–3, `quali_*`, `sprint_*` date/time; qualifying `constructorId` (10 rows disagree with results); results/sprint `position, positionText, time`; standings `positionText`; constructor_results `status` | links, >88% missing, duplicates, or inconsistent |
| 6 | Types | numeric-only columns → `Int64` (nullable) / `float64`; `sprint_results.points` forced to float like `results.points`; `races.date`, `drivers.dob` → datetime; `races.start_utc` = date + time; `q1..q3`, `fastestLapTime` → `*_sec` floats | 1.3 |
| 7 | Labels | `United States` → `USA`; `Argentinian` → `Argentine`; new `drivers.home_country` via `NATIONALITY_TO_COUNTRY` (dual nationalities → first listed; East German → Germany) | needed for Q2 |
| 8 | Status taxonomy | every status → one of **six**: finished / mechanical / accident / regulation / driver / **unclear**. Mechanical = parts that fail on their own (+ Technical, Seat). Unclear = Retired, Out of fuel, Withdrew, Wheel, Wheel nut, Front/Rear wing, Undertray, Vibrations, Puncture, Damage, ... DNQ / DNPQ / 107% → regulation. The non-starter rule uses its own `NON_START_STATUSES` list | Q3 definition; asserted complete and non-overlapping; full mapping with row counts in 2.9.1. Q3 reported strict (mechanical) and broad (mechanical + unclear) |
| 9 | Target | `podium = positionOrder <= 3`, created **before** removing anything | cleaning can't change outcomes |
| 10 | Non-starters | remove `(status ∈ {DNQ, DNPQ, Withdrew, 107% Rule} AND laps == 0) OR (grid == 0 AND laps == 0)` → **1,645 rows**; keeps the **72 pit-lane starters** | they never took the start = easy fake negatives |
| 11 | Duplicates | **Rule B**: keep the row with the **lowest `resultId`** as a proxy for the car the driver started in, plus 3 verified overrides (Ascari 1951 French GP, González 1954 Belgian GP, Moss 1961 British GP) → 88 rows removed | chosen without looking at the result. Grid can't decide (both rows grid > 0 in all 82 pairs). Cost: 17 drivers (1950–1960) lose shared-drive podium credit. The proxy may be wrong for a few other 1950s pairs (affects only the all-years model and Q2, negligible) |
| 11b | Indianapolis 500 | **removed** from the cleaned results (381 rows, 11 races 1950–1960); `races` rows kept so FKs hold; set aside in `INDY500_RESULTS` for the Q2 robustness check | separate American race inside the championship: 380 of 381 starts by Americans, 103 of 107 drivers never started another championship race, ~33-car oval field. It made up 16% of Q2's home starts with no away drivers in the same race, and would distort an all-years model (2.6.1) |
| 12 | `grid = 0` | → that **race's** max grid + 1, plus a `pit_lane_start` flag (also in sprint_results) | 0 would look better than pole |
| 13 | Qualifying outlier | lap times > 300 s → NaN (only the 1002.6 s Q1) | impossible flying lap |
| 14 | Leakage-safe standings | `driver_standings_prerace` / `constructor_standings_prerace` = standings after the **previous round of the same season**, keyed by the race being predicted | same-race standings leak (2021 finale: tied on 369.5 before; the table already shows the final) |
| 15 | Modeling window | **2003+** → `model_base` = 8,934 rows, 428 races, podium rate 14.4% | qualifying ~100% from 2003; older F1 is a different sport (reliability, DNQs). Model version B (all years, training 1950–2018) will be built from `results_full` in section 4 (see 7.1), so pre-2003 rows are cleaned too |
| 16 | Two outputs | `results_full` (all years except the Indianapolis 500, with post-race cols, for the questions) and `model_base` (pre-race only) | the questions need status/position; the model mustn't see them |

Validation (2.10): 15 automated checks all pass (unique keys, **all 21 FKs** with a per-FK table, no `\N`/empty left, grid ≥ 1, model-ready dtypes, no post-race columns, **exactly 3 podiums per race in 2003+**). 2.10.1 has further sanity checks. 2.10.2 rebuilds cumulative points: they match `driver_standings` in 96.1% of rows with the current race included vs 74.7% without (92.6% vs 0.1% on rows where the driver scored), and every mismatch is explained (689 sprint points, 452 Indianapolis 500 points still in the championship standings, 84 shared drives, 124 pre-1991 best-results rule; half points cause none). 2.11 has six before/after plot pairs (podium rows per race, missing values per key column, grid, Q1 times, rows per table, podium rate per season).

**Known data facts (not errors), documented in 2.10.1:**
- Smallest fields are real: 2005 US GP has 6 starters, 2015 Australian GP has 15.
- 4 pairs of drivers share a grid slot in the source data (e.g. Nasr/Sainz, Italy 2015). Left as is (8 rows).
- 1 race with > 3 podium rows remains: the 1955 Argentine GP (cross-driver shared drive, outside the window). There were 4: the 1951 French GP was caused by the Rule B proxy (fixed by the Ascari override), and the 1951/1955 Indianapolis 500 were removed with the rest of the Indy rows.
- **Formation-lap limitation:** a car that failed on the formation lap (never started) can't be told from a lap-1 retirement (both 0 laps + failure status). About 49 rows in 2003+ (0.5%) may be non-starters recorded as starters. Kept, because dropping all 0-lap rows would remove real lap-1 retirements and select rows by race outcome.
- **Standings gap:** the standings table only lists drivers who already have an entry (2003 round 1: 11 standings rows for 20 starters). In `model_base`, 597 driver rows have no pre-race standing: 455 are round 1, **142 later**; 31 later for constructors. It means "no points yet" or "didn't race the previous round". **Not fixed yet**: it's a feature-engineering step (see section 7).

---

## 6. Data-engineering answers (section 3)

**Q1: front row → podium (main answer: front row = grid 1–2, 2003–2013 vs 2014–2024, ≥ 8 front-row starts per era per circuit):**
- Overall conversion ~7/10, **68% → 76%** in the hybrid era.
- Hybrid top: **Suzuka (89%), Yas Marina (86%), Monaco (85%)**. 2003–2013 top: Monaco, Monza, Silverstone.
- **Ranking changed a lot** (Spearman 0.21).
- Qualifying-based front row gives 71% vs 72% overall; 36 front-row qualifiers started further back (penalties).
- **3.1.1 robustness, grid only, 1950–2013 vs 2014+:** conversion depends on the decade (46–54% in the 1950s–80s, 74–75% since 2010), so an all-years "pre-2014" ranks circuits by **age**. Monaco/Monza/Silverstone/Spa sink to ~47–56%; the two "pre-2014" rankings correlate −0.14. **Decision: keep 2003–2013 as main**; report all-years as a limitation. The hybrid-era top barely changes (Zandvoort enters thanks to its old races).
- **3.1.2 why qualifying is kept (not just grid):**
  - Qualified top 3 but started 6th+ (penalties): **32% podium** vs **3.4%** for others from the same slots.
  - Same grid slot, faster half by gap to pole: e.g. grid 5 **27% vs 15%** (grid 3 shows no difference).

**Q2: home effect, controlled by grid bands 1–3 / 4–6 / 7–10 / 11+, all seasons, Indianapolis 500 excluded:**
- Raw: home 11.6% vs away 13.7% (misleading: home avg grid 12.7 vs 11.7). 1,934 home starts.
- Within bands, home ≈ away in every band (1–3: 55.8% vs 55.4%; 7–10: 7.2% vs 9.3%).
- Grid-adjusted difference **−0.60 pts, 95% bootstrap CI [−1.76, +0.56]**. **No home advantage.** The 2003+ subset agrees (−1.55, CI includes 0), and so does the robustness version with Indy put back (−0.59, CI [−1.73, +0.59]).

**Q3: mechanical retirements per start (status_category == "mechanical"), teams with ≥ 40 starts:**
- 2014–2021: **Lotus F1 17.3%, Toro Rosso 16.1%, McLaren 13.5%**, Renault 13.5%.
- 2022–2024: **Alfa Romeo 12.5%, Alpine 11.0%**, AlphaTauri 6.8%.
- Field average 9.1% → 5.5% (it was 9.5% → 6.0% before the status review moved Wheel, Rear wing, Undertray and Vibrations to unclear).
- Reported strict (mechanical) and broad (mechanical + unclear, up to ~7 pts higher): top two unchanged in both periods.
- Renamed teams are separate entries (e.g. Toro Rosso → AlphaTauri → RB).

---

## 7. Section-by-section decisions and results (sections 4–10), and next steps

### 7.1 Section 4: Feature selection & engineering. Plan approved 15 Oct 2026, BUILT 16 Oct 2026

Notebook layout:
4.1 split + checks · 4.2 build candidate features (+ leakage asserts) · 4.3 evidence per group · 4.4 redundancy · 4.5 feature trials on validation · 4.6 final feature table.

**Split (by season, never random; a race is never split):**

| Set | Model A (2003+) | Model B (all years) |
|---|---|---|
| Train | 2003–2018 | 1950–2018 (missing qualifying = NaN + a "qualifying missing" flag) |
| Validation | 2019–2021 | 2019–2021 |
| Test | 2022–2024 | 2022–2024 |

- Justify in the notebook: random split leaks the future (form/standings look back in time) and splits races (exactly 3 podiums per race); 2022 new car rules = honest unseen test. Show rows and podium rate per split.
- **Test seasons (2022–2024) are never touched** in section 4: not for plots, not for trials, not for decisions.

**Rules for evidence:**
- **All evidence plots, correlations and feature decisions use TRAINING seasons only.**
- Every section gets a short plain-language markdown note: what it shows and why.

**Feature groups (also the later ablation units):**

| Group | Features | Decisions |
|---|---|---|
| Grid | `grid`, `pit_lane_start` | |
| Qualifying | qualifying position, **gap to pole % from Q1 only**, `grid − quali_pos` (penalty indicator), reached Q3 | Gap from Q1 because almost every driver sets a Q1 time and, from 2006, Q1 is run without race fuel (not perfectly equal conditions: 2003–2005 `q1` is the race-fuel single lap, and grip/weather change within a session; documented in 4.5 trial 2b). **Written trial** comparing it with best-of-Q1/Q2/Q3, noting the 2006–2009 race-fuel Q3 and the early-2005 aggregate (q1 + q2) format. Reached Q3 only exists from 2006: **add a flag for earlier years** (format had no Q3) |
| Standings | **previous-race standings only** (`*_prerace`): championship **position** and **share of the leader's points**, wins; driver and constructor; `no_standing_yet` flag (round 1 and gaps) | **Never raw points** (scale changed in 2010: win 10 → 25) |
| Form | driver: podium rate / average finish / retirement rate over the last N races; constructor: podium rate over the last N races | Computed with **`shift(1)` in race order**. **Asserts:** the current race is excluded, and a driver's first race has no form value |
| Context | number of starters, driver experience (career starts), age; **home race as one trial** | Home race kept or dropped on **validation PR-AUC**, with the result written down (Q2 found no effect) |

**Excluded (decided):** team (constructor) and circuit as features; year / era feature; sprint finish (out for now; `sprint_results` stays cleaned for a later optional trial).

**Redundancy:** for highly correlated pairs (grid vs qualifying position; championship position vs points share), state in the notebook which one is kept and why.

**Trials (4.5):** logistic regression, **grid only** first, then **add one group at a time**, scored by **validation PR-AUC**. **Every trial in its own cell**, all kept in the notebook. This is feature selection; the full ablation (all models) is in the experiments section.

**Results (built 16 Oct 2026):**
- Split sizes: model A train 6,381 rows / 300 races, validation 1,199 / 60, test 1,354 / 68 (test podium rate deliberately not inspected). Model B train 22,092 rows / 986 races.
- Leakage asserts pass: 686 first driver races and 334 first constructor races (first race ever, or first race after a long absence) have no form; 500 random rows match a hand recomputation from earlier races only.
- **Keep rule (changed with the user, 16 Oct):** keep a group if (1) its 4.3 plots show a clear link with podiums and (2) it doesn't lower validation PR-AUC. My first rule (gain ≥ 0.005) was dropped: validation has only 60 races, PR-AUC moves by about ±0.047 between race resamples, so every decision now also shows a 95% race-resampling interval for the gain.
- Redundancy: `grid` kept, `quali_pos` dropped (Spearman 0.97; info kept through `grid_minus_quali` + `gap_q1_pct`). Driver: championship **position** kept over points share (−0.95; 0.854 vs 0.852 AUC alone). Constructor: **points share** kept over position (−0.93; 0.856 vs 0.826).
- Trials (logistic regression, validation PR-AUC, after the review fixes below): grid 0.6765 → + qualifying 0.7210 (+0.045, interval +0.019 to +0.071) → + standings 0.7343 (+0.013) → + form 0.7382 (+0.004) → + context 0.7384 (+0.0002) → + home race 0.7351 (**dropped**, −0.003, interval entirely below 0). Before the fixes: 0.7208 / 0.7339 / 0.7362 / 0.7376 / 0.7343.
- **How strong the evidence is:** grid and qualifying are the only groups with clear evidence (qualifying is the only gain whose interval is entirely above 0). The standings, form and context gains are small and their intervals include 0; they are kept because of their 4.3 training-season plots and because they don't clearly lower the score, not because the trials prove they help. These are logistic-regression results only; the feature-group ablation in the model-experiments section (after pre-processing) re-tests every group with every model.
- Written trials: gap from Q1 0.7210 vs best-of-sessions 0.7099 (difference within noise, interval −0.008 to +0.033; 2006–2009: pole-sitter not fastest by best time in 9–11 races per season). Form window: 5 races 0.7382, 3 races 0.7351, 10 races 0.7302; both differences have intervals that include 0, so **5 is kept as a reasonable middle value chosen in advance, not because the trial shows it is better**.
- **Review fixes (16 Oct 2026, section 4.2.1):**
  - `reached_q3` was "has a Q3 time", which marked drivers who advanced to Q3 but set no time as eliminated. Now: a Q3 time, or one of the Q3 places left over, filled by the fastest Q2 times among drivers without a Q3 time (10 places; 8 in the 2016 Australian and Bahrain GP elimination format). Asserted: all 375 knockout races have exactly their number of Q3 places. **114** starter rows changed 0 → 1 (79 train, 16 validation, 19 test). Podium rate (2006–2018 train): reached Q3 29% vs 1% (was 30% vs 1%).
  - Team form followed `constructorId` with no time limit, so returning IDs used decades-old results (Mercedes 2010 ← 1955, Honda 2006 ← 1968, Alfa Romeo 2019 ← 1985, Aston Martin 2021 ← 1960). Now team form only looks back within a history segment; a gap of more than **550 days** starts a new one (normal breaks ≤ 217 days in 2003+; every 2003+ reset is a gap of 4+ years, so any threshold from 1 to 2 years gives the same model-A features). 7 resets in 2003+ (also Renault 2016, Sauber 2010 and 2024), 39 in all seasons; **141** rows changed (135 train, 4 validation, 2 test), all within the first 5 races after a reset; the comeback race itself now has no team form. Renamed teams (Toro Rosso → AlphaTauri, Force India → Racing Point → Aston Martin, ...) keep separate IDs: documented limitation.
  - Notebook text corrected: form window and Q1 comparability (above), the ablation reference (model-experiments section, not section 5), and how strong the selection evidence is.
- Fix found while plotting: in round 1 nobody has standings, so "behind everyone ranked" had become position 1 for everyone. Now round-1 rows are set to the back (number of starters), with the `no_standing_yet` flag.
- **Final 18 features (`FINAL_FEATURES`):** grid `grid`, `pit_lane_start` · qualifying `gap_q1_pct`, `grid_minus_quali`, `reached_q3`, `no_q3_format`, `quali_missing` · standings `drv_champ_pos`, `cons_points_share`, `drv_no_standing_yet`, `cons_no_standing_yet` · form `drv_podium_rate_5`, `drv_avg_finish_5`, `drv_dnf_rate_5`, `cons_podium_rate_5` · context `n_starters`, `drv_experience`, `drv_age`. Model B uses the same 18.

### 7.2 Section 5: Pre-processing. Plan approved 16 Oct 2026 (with adjustments), BUILT 17 Oct 2026

Notebook layout: 5.1 load + feature types · 5.2 missing-value audit · 5.3 imputation step (`F1Imputer`) · 5.4 `gap_q1_pct` cap · 5.5 scaling · 5.6 order experiments (+ consistency check, era diagnostic, invalid leakage demo, transformation check) · 5.7 final pipelines · 5.8 leakage and behaviour checks.

- **Input:** `outputs/final_features.csv` (read, never changed; its SHA-256 is asserted unchanged). 12 continuous + 6 binary features. `raceId`, `driverId`, `year`, `split`, `podium` are never inputs (asserted).
- **Missing values (train A / train B / validation):** `gap_q1_pct` 1.8% / 64.8% / 1.7%, `grid_minus_quali` 0.3 / 64.2 / 0.3, `reached_q3` 16.6 / 75.9 / 0.3, driver form (3 features) 1.1 / 3.0 / 0.8, team form 0.6 / 1.4 / 0.8; nothing else. Causes asserted exactly: `grid_minus_quali` missing ⇔ `quali_missing`; `reached_q3` missing ⇔ `no_q3_format` or `quali_missing`; driver form missing together ⇔ debut (`drv_experience = 0`).
- **Indicators (built from the raw values, before filling):** `q1_time_missing` (qualifying record but no Q1 time; never overlaps `quali_missing`), `drv_form_missing`, `cons_form_missing` → **21 model inputs**. The CSV keeps the 18 features.
- **Fills (all fitted on training rows of the model concerned):**

  | Gap | Fill | Model A value | Model B value |
  |---|---|---|---|
  | `gap_q1_pct`, qualifying record but no Q1 time (A 94 rows, podium 4.3%; B 125, 5.6%) | training 95th percentile (slow lap) | 5.185 | 5.697 |
  | `gap_q1_pct`, no qualifying record (A 19 rows; B 14,189) | training median (neutral), `quali_missing` kept | 1.359 | 1.533 |
  | `grid_minus_quali` | 0 (no known penalty) | 0 | 0 |
  | `reached_q3` | 0 (column stays 0/1; flags explain the gap) | 0 | 0 |
  | driver form / team form | training medians (+ indicators) | 0 / 11.4 / 0.2 / 0 | 0 / 12.0 / 0.4 / 0 |
  | anything else (partial new record only) | training median (continuous), 0 (flags); missing `quali_missing` derived from the qualifying features | | |

- **Outliers:** only `gap_q1_pct` is capped, at the training 99th percentile **of observed laps** (A 7.856%, 63 rows; B 9.015%, 78 rows; the largest value was 33.3%). All rows kept. A few capped drivers did reach the podium (4 in A, 3 in B); they stay "among the slowest" after the cap. Not capped: `grid_minus_quali` (real penalty starts), model B field-size values (`drv_champ_pos` up to 104, `n_starters` up to 34).
- **Scaling:** `StandardScaler` on the 12 continuous features for logistic regression / NN; the 6 flags + 3 indicators stay 0/1; tree pipeline without scaling. Pandas output keeps column names (for SHAP).
- **Final pipelines:** `PREPROCESSORS[("A" | "B", "scaled" | "tree")]`, each fitted only on its own training rows (A 2003–2018, B 1950–2018); `MODEL_INPUTS["scaled" | "tree"]` (21 names; same set, different order). Use `PREPROCESSORS[...].transform(rows[FINAL_FEATURES])`.
- **Order experiments (model A rows unless stated; logistic regression; validation PR-AUC with 95% race-resampling interval of the difference):**
  - Consistency check, not an experiment: median impute → scale vs scale → median impute: 0.7331 vs 0.7330, largest probability difference 0.0026 (only the scaler statistics move).
  - **E1, KNN impute → scale vs scale → KNN impute:** filled values genuinely differ (e.g. `gap_q1_pct` mean fill 3.24 vs 3.45, all 113 cells differ; `drv_avg_finish_5` mean |difference| 2.1 places), but PR-AUC 0.7368 vs 0.7370, difference +0.0002 [−0.0016, +0.0021]. Order matters for the data, not measurably for this model.
  - **E2, cap fitted on observed laps → impute vs impute → cap fitted on filled column:** checked first, as asked. Model A: nearly equivalent (cap 7.856 vs 7.764, 64 rows change by ≤ 0.09; PR-AUC 0.7350 vs 0.7348). **Model B: genuinely different** (cap 9.015 vs 6.542, because 64% of the column is median-filled; 221 training laps change by up to 2.47 points), PR-AUC 0.7236 vs 0.7245, difference +0.0009 [−0.0017, +0.0052]. Reported on model B. Kept: cap fitted on observed laps (principle: the cap describes real laps and must not depend on how much data is missing; the score doesn't decide).
  - Supporting: era filter (model A pre-processing fitted on 1950–2018 vs 2003–2018): scaler means move (DNF rate 0.42 vs 0.23, experience 61 vs 92), PR-AUC identical to 4 decimals (0.7350).
  - **INVALID demonstration** (fitted on training + validation): PR-AUC unchanged to 4 decimals, largest probability change 0.0043. Shown only to explain the `fit_on_training` guard.
  - Transformation check (not an order experiment): `log(grid)` 0.7350 (tie), `1/grid` 0.7139, `log(1+experience)` 0.7286, `log(1+gap)` 0.7336, RobustScaler 0.7345, vs final 0.7350; every interval includes 0 → no extra transformation.
- **Conclusion:** no pre-processing alternative changes validation PR-AUC beyond noise; choices rest on correctness (no leakage, meaningful fills, undistorted cap). Note: Section 4's temporary pipeline (median + 7 duplicate indicators + scaling of flags) scored 0.7384; the final 21-input pipeline gives 0.7350 with the same logistic regression, a difference well inside the noise (bootstrap SD of one score ≈ 0.045).
- **Checks (5.8), all pass:** no missing values in any output; inputs exactly the 18 features; every learned value equals a hand recomputation from training rows; all fits in section 5 logged as training rows only; 5 partial-record cases filled as designed; A and B give the same 21 names; `final_features.csv` unchanged.
- **Limitations:** `pit_lane_start` has 7 model-A training rows (pit-lane starts are only recorded as `grid = 0` from 2015); `no_q3_format` is always 0 in validation/test; model B's `quali_missing` mostly marks old eras, and one coefficient per feature can't adapt to different eras; validation reuse (60 races) makes every small difference here noise.

### 7.3 Section 6: Predictive modeling. Plan approved 17 Oct 2026 (with adjustments), BUILT and FROZEN 17 Oct 2026

Notebook layout: 6.1 set-up, guards, metrics · 6.2 forward-chaining CV + selection rules + grids · 6.3 baselines · 6.4 logistic regression · 6.5 random forest · 6.6 gradient boosting · 6.7 PyTorch network (+ final fits, 3-seed check, LR weights) · 6.8 thresholds · 6.9 validation, overfitting, calibration · 6.10 model A vs B + training-window experiments · 6.11 primary model + freeze + guards.

- **Cross-validation (tuning):** 4 forward-chaining folds scoring 2011–12, 13–14, 15–16, 17–18 (158 races); fold training = all earlier training seasons (A from 2003, B from 1950; identical scoring rows asserted). Pre-processing re-fitted inside every fold. **Validation (2019–2021) used once per final model, never for a choice.**
- **Selection rule:** best mean fold PR-AUC; ties = settings not clearly worse by a **paired** one-standard-error rule (mean per-fold shortfall vs the best ≤ standard error of those shortfalls), closed upwards (any setting scoring at least as high as a tie is a tie); among ties the simplest/most regularised by a fixed per-family order. (See the trial log for the two corrections to this rule.)
- **Network:** PyTorch, Adam lr 0.001, batch 256, BCE loss; early stopping on an inner subset = the last 2 seasons of each fold's training period (pre-processing fitted on the inner training seasons), patience 15, max 200 epochs; then retrained on the whole fold training period for that epoch count and scored on the fold; final fit trains for the median of the 4 folds' best epochs. Deterministic mode, 1 thread per job; (setting, fold) jobs run in parallel processes (joblib), results identical to sequential.
- **Chosen settings:**

  | Family | Model A (2003–2018) | Model B (1950–2018) |
  |---|---|---|
  | logistic regression | `C` 1, no class weight (4 of 8 tied) | `C` 0.1, no class weight |
  | random forest (500 trees) | depth 6, min leaf 50, `sqrt` (14 of 18 tied) | depth 6, min leaf 50, `sqrt` |
  | gradient boosting | lr 0.05, 7 leaves, min leaf 80, 100 trees (3 of 16 tied) | same |
  | network | 1 layer × 16, dropout 0, weight decay 1e-3, 144 epochs (8 of 12 tied) | 1 × 16, dropout 0, wd 1e-4, 112 epochs |

- **Results, model A** (mean CV PR-AUC → validation PR-AUC / ROC-AUC / Brier; threshold from out-of-fold predictions → validation precision / recall / F1; F1 top-3 per race):

  | Model | CV PR-AUC | val PR-AUC | val ROC-AUC | Brier | threshold | precision | recall | F1 | F1 top-3 |
  |---|---|---|---|---|---|---|---|---|---|
  | rule grid ≤ 3 | (CV F1 0.651) | – | – | – | – | 0.717 | 0.717 | 0.717 | – |
  | grid-only LR (baseline) | 0.646 | 0.676 | 0.917 | 0.071 | 0.363 | 0.617 | 0.822 | 0.705 | 0.717 |
  | **logistic regression (primary)** | **0.703** | **0.735** | 0.937 | 0.065 | 0.381 | 0.664 | 0.778 | 0.716 | 0.700 |
  | random forest | 0.684 | 0.733 | 0.936 | 0.065 | 0.299 | 0.635 | 0.811 | 0.712 | 0.700 |
  | gradient boosting | 0.693 | 0.721 | 0.938 | 0.064 | 0.387 | 0.675 | 0.783 | 0.725 | 0.717 |
  | neural network | 0.702 | 0.716 | 0.936 | 0.065 | 0.377 | 0.656 | 0.794 | 0.719 | 0.700 |

- **Paired race-resampling comparisons (validation PR-AUC):** vs grid-only LR: LR +0.059 [+0.011, +0.107], RF +0.057 [+0.012, +0.104], HGB +0.044 [+0.011, +0.085] (all clear); NN +0.039 [−0.002, +0.092] (not clear). vs LR: RF −0.002, HGB −0.014, NN −0.020, **all intervals include 0**. → Features beyond grid help; **no family is clearly best**.
- **F1 caveat:** the yes/no F1 of every model (0.71–0.72) is about the same as the plain "grid ≤ 3" rule (0.717). The models' gain is in ranking and probabilities (PR-AUC, Brier), not in many more correct yes/no calls.
- **Overfitting:** PR-AUC train / CV / validation: LR 0.688 / 0.703 / 0.735, RF 0.705 / 0.684 / 0.733, HGB 0.736 / 0.693 / 0.721, NN 0.706 / 0.702 / 0.716. Small gaps (trees regularised). Validation > CV for all: 2019–2021 easier (dominant teams); fold scores range 0.64–0.79 by season.
- **Network seeds (chosen setting):** CV 0.700 ± 0.003, validation 0.721 ± 0.010 over seeds 42/43/44 (frozen: seed 42).
- **Calibration:** Brier 0.064–0.065 (baseline 0.071); predicted probabilities sum to 3.0–3.3 per race (3 real podiums). No re-calibration (by decision).
- **Model A vs B (validation, B − A):** LR −0.014 [−0.038, +0.013], RF −0.002, HGB +0.006, NN +0.016 [−0.006, +0.030]: all intervals include 0. CV within 0.014. **More (older) data does not help.**
- **Training window (descriptive, validation):** LR: every start before 2003 ≈ −0.011 vs 2003–2018; start 2010 −0.003. HGB: older starts +0.001 to +0.006. **Same size (6,381 rows) from all eras** (5 race samples): LR 0.716 ± 0.009 and HGB 0.712 ± 0.010, below model A's 0.735 / 0.721 → at equal size, modern seasons are more useful; old data only partly makes up through quantity.
- **Primary model (rule fixed in advance: highest mean CV PR-AUC among model-A families, simpler family if tied by the paired rule):** **logistic regression, model A** (CV 0.703; the network 0.702 tied; LR is simpler).
- **Freeze:** 8 final models + 2 baselines, fitted objects in `FROZEN_MODELS`, record + fingerprint in `outputs/frozen_models.json`, `predict_frozen()` verified to reproduce the validation predictions exactly. Guards: 697 fits, all on training rows; all 664 CV fits used only seasons before their scoring block (asserted at fit time + from the log); 486 predictions, none on test rows; `final_features.csv` unchanged.
- **Limitations:** 60 validation races (differences of ~0.02–0.04 are within noise); thresholds from CV folds whose models trained on fewer seasons (they still predicted 209–230 podiums for 180 real ones); 2022 regulation change may lower test scores; `pit_lane_start` / `no_q3_format` carry little usable signal; model B mixes eras with one coefficient per feature; the tie rule was corrected twice after the first runs (trial log).

#### Model artifacts (6.12)

- **What is saved:** the 8 frozen final models (4 families × model A and B) and the grid-only baseline, each with its **fitted pre-processing** from 5.7, in `outputs/model_artifacts/`. Nothing is retrained: 6.12 serialises the objects frozen in 6.11.
- **Formats:** scikit-learn pipelines and models with **`cloudpickle`** (the pre-processing contains `F1Imputer`, a notebook-defined class; plain `pickle`/`joblib` would store only a reference to it and fail outside the notebook). Networks: PyTorch **`state_dict`** (`torch.save`, loaded with `weights_only=True`) + architecture in the manifest; their pre-processing with `cloudpickle`.
- **Thresholds:** stored at full precision (JSON float, round-trips exactly) and as an exact hexadecimal float (`float.hex`); the loader uses the hex value. (`outputs/frozen_models.json` keeps 6-decimal thresholds for reading only.)
- **Loading (outside the notebook):**
  ```python
  import sys; sys.path.insert(0, "outputs/model_artifacts")
  from load_artifacts import load_model_artifacts
  models, manifest = load_model_artifacts("outputs/model_artifacts")
  m = models["A_logistic_regression"]                 # primary model; others: A_/B_ + random_forest, gradient_boosting, neural_network; A_grid_only_LR
  p = m["predict"](rows)                              # rows: DataFrame with the 18 FINAL_FEATURES columns (missing values allowed)
  yes = p >= m["threshold"]                           # frozen threshold rule
  ```
  Every file is checked against its SHA-256 in the manifest when loading.
- **Verification (17 Oct 2026):** in the notebook (6.12) and in a separate Python process with no notebook code, all 9 re-loaded models reproduce the frozen validation predictions **bit for bit** (largest difference 0.0; notebook tolerance 1e-9), with identical thresholds and identical threshold / top-3 decisions. A record with missing qualifying and form values is scored without error.
- **Versions that wrote them:** Python 3.14.0, numpy 2.4.1, pandas 2.3.3, scikit-learn 1.8.0, torch 2.10.0+cpu, cloudpickle 3.1.2 (also in `manifest.json`).
- **Portability limits:** load with the **same versions**. `cloudpickle` files contain Python code by value and need the same Python minor version; scikit-learn objects may not load (or may warn) under another scikit-learn version; PyTorch weights are the most portable part. The networks compute in 32-bit floats: with a different number of CPU threads their probabilities can differ by ~1e-8 (checked: up to 3e-8 with 8 threads, identical decisions); call `torch.set_num_threads(1)` first for bit-identical results (the notebook does). On Kaggle (other versions) *Run All* simply re-creates the files; predictions are identical between runs, but the bytes/SHA-256 of the `cloudpickle` files can differ between runs, so the manifest always describes the files of its own run.

### 7.4 Section 7: Final optimisation, second freeze, test evaluation. Authorised 17 Oct 2026, BUILT 17 Oct 2026

Order: Phase 1 re-tuning → Phase 2 freeze v2 → Phase 3 one-time test → Phase 4 interpretation. Section 6 (v1) is untouched.

- **Phase 1 rule (fixed before running):** for each setting dimension where the best or chosen setting of model A or B used the smallest/largest value tried, add one value beyond that edge (fixed list in 7.1); run the new combinations only; choose among old + new with the **same** folds, paired one-SE rule (closed upwards) and simplicity order (random forest order gains depth 4). Validation not used for any choice; training windows, features and pre-processing unchanged.
- **Extensions added:** LR `C` 100 (2 new settings); RF depth 4, min leaf 100 (14 new); HGB learning rate 0.02, 3 leaves, min leaf 10 and 160, 50 and 600 trees (128 new); network hidden (8,), dropout 0.4, weight decay 1e-5 and 1e-2 (36 new). Same for A and B. Run time of the 7.2 search ≈ 26 min (the full notebook now ≈ 45 min).
- **v2 settings (CV PR-AUC v1 → v2):** LR A unchanged `C` 1 (0.703); LR B unchanged `C` 0.1 (0.690); RF A and B → depth 4, min leaf 100, `max_features` 0.5 (A 0.684 → 0.687, B 0.684 → 0.689); HGB A → lr 0.1, 3 leaves, min leaf 80, 50 trees (0.693 → 0.694); HGB B → lr 0.1, 3 leaves, min leaf 160, 100 trees (0.692 → 0.690); NN A → (8,), dropout 0, wd 1e-3, 158 epochs (0.702 → 0.706); NN B → (8,), dropout 0, wd 1e-3, 119 epochs (0.704 → 0.701). Changes are to simpler tied settings; the best overall CV scores moved by ≤ +0.006.
- **v2 primary model (same rule):** best mean CV = network A (0.706); logistic regression A is tied (paired rule) and simpler → **primary stays logistic regression, model A**.
- **v2 thresholds (out-of-fold F1):** LR A 0.3806, RF A 0.3315, HGB A 0.3090, NN A 0.4441, LR B 0.2986, RF B 0.3660, HGB B 0.3686, NN B 0.2922 (full precision in the manifest).
- **Validation (reported once):** v2 − v1: RF A +0.002, HGB A +0.012, NN A +0.023 [+0.001, +0.038], RF B −0.001, HGB B +0.002, NN B −0.001; LR identical.
- **Freeze v2:** fingerprint **`2ebc3457f47525afaf17b679d98bffc83bd3481b8261f4c5b3da25e108abafaa`**, recorded 17 Oct 2026 17:46 **before any test cell existed**; checked before the test by `verify_v2_freeze` (7.5, see 7.7 "Freeze verification"). Artifacts in `outputs/model_artifacts_v2/` re-load bit for bit (tolerance 1e-9).
- **Test 2022–2024 (68 races, 1,354 rows, 204 podiums), frozen v2 models loaded from disk:**

  | Model | PR-AUC | ROC-AUC | Brier | precision | recall | F1 | F1 top-3 |
  |---|---|---|---|---|---|---|---|
  | rule grid ≤ 3 | – | – | – | 0.650 | 0.647 | 0.649 | – |
  | grid-only LR | 0.647 | 0.914 | 0.075 | 0.583 | 0.775 | 0.665 | 0.652 |
  | **LR A (primary)** | **0.705** | 0.937 | 0.070 | 0.616 | 0.730 | 0.668 | 0.667 |
  | RF A | 0.719 | 0.939 | 0.069 | 0.614 | 0.755 | 0.677 | 0.642 |
  | HGB A | 0.718 | 0.938 | 0.068 | 0.597 | 0.799 | 0.683 | 0.642 |
  | NN A | 0.698 | 0.935 | 0.072 | 0.621 | 0.716 | 0.665 | 0.627 |
  | LR B | 0.712 | 0.937 | 0.070 | 0.585 | 0.809 | 0.679 | 0.657 |
  | RF B | 0.725 | 0.941 | 0.070 | 0.641 | 0.735 | 0.685 | 0.667 |
  | HGB B | 0.718 | 0.939 | 0.069 | 0.625 | 0.735 | 0.676 | 0.652 |
  | NN B | 0.704 | 0.937 | 0.070 | 0.551 | 0.819 | 0.659 | 0.627 |

- **Paired race-resampling intervals (test PR-AUC):** vs grid-only: LR A +0.057 [+0.009, +0.104], RF A +0.072 [+0.036, +0.108], HGB A +0.071 [+0.042, +0.098], NN A +0.051 [+0.006, +0.095] → **all clear**. vs primary: RF A +0.015, HGB A +0.013, NN A −0.007 → **none clear**. B − A: LR +0.008, RF +0.006, HGB −0.001, NN +0.006 → **none clear**.
- **Train / CV / validation / test PR-AUC (model A):** LR 0.688 / 0.703 / 0.735 / 0.705; RF 0.699 / 0.687 / 0.735 / 0.719; HGB 0.705 / 0.694 / 0.733 / 0.718; NN 0.703 / 0.706 / 0.738 / 0.698. Model B training PR-AUC 0.56–0.58 (old seasons are harder, not overfitting).
- **Regulation change:** test − validation for model A −0.015 to −0.040, every interval includes 0 (≈ ±0.10). Primary model per season: 2019 0.737, 2020 0.726, 2021 0.742, 2022 0.696, 2023 0.783, 2024 0.645 (grid-only similar shape: 2023 0.678, 2024 0.657) → season predictability (team dominance) matters more than the rule change; no clear evidence of a 2022 effect.
- **Calibration (test):** Brier 0.068–0.072 (baseline 0.075); probabilities sum to 3.05–3.31 per race.
- **Conclusions:** the four families are statistically indistinguishable on the test seasons; the primary logistic regression generalises (test ≈ CV, ROC-AUC stable) and, like every model, clearly beats the grid-only baseline; old data neither helps nor hurts; no clear overfitting; no clear regulation-change effect. On test, every model's F1 (0.659–0.685) is above the "grid ≤ 3" rule (0.649), unlike on validation (descriptive).
- **Official prediction output (7.10, decided after the test; presentation only, no model change):**
  - **Complete race → the predicted podium = the 3 drivers with the highest podium probability** from the frozen primary model (logistic regression, model A, v2). They are the most likely podium finishers, **not** a forecast of 1st/2nd/3rd. This rule was part of the frozen plan (6.8) and scored about the same as the threshold rule on the test (F1 0.667 vs 0.668), so it was not chosen for a better score but because every race has exactly 3 podium places.
  - **Single driver → a podium probability** (no claim about the whole podium; the top-3 rule needs the whole race).
  - **Threshold rule (≥ frozen threshold) → secondary, academic result** (any number of "yes" per race; precision/recall/F1 kept in 6.9, 7.5, 7.6).
  - **Race-level results of the frozen predictions:**

    | Period | Pick | top-3 F1 | correct podium drivers per race (of 3) | all 3 right | at least 2 right |
    |---|---|---|---|---|---|
    | validation 2019–2021 (60 races) | top 3 on the starting grid | 0.717 | 2.15 | 20 (33.3%) | 85.0% |
    | validation 2019–2021 | **official (LR A)** | 0.700 | 2.10 | 21 (35.0%) | 80.0% |
    | **test 2022–2024 (68 races)** | top 3 on the starting grid | 0.652 | 1.96 | 11 (16.2%) | 80.9% |
    | **test 2022–2024** | **official (LR A)** | **0.667** | **2.00** | **16 (23.5%)** | 77.9% |

    Official minus grid pick on the test seasons: +0.044 correct drivers per race, 95% race-resampling interval −0.088 to +0.176 → **not clear** (validation: −0.050, also not clear). As a podium pick the model is about as good as the front three on the grid; its clear advantage is in ranking/probabilities (test PR-AUC 0.705 vs 0.647). Other model-A families pick 1.88–1.93 correct per race on the test.
  - **Deterministic:** no exact tie between the 3rd and 4th probability in any validation or test race (ties would be broken by grid, then `driverId`); the official test picks have fingerprint `5de40cb915d35a8d…` (printed in 7.10).
- **v1 on test (comparison only, no decision):** LR identical; RF A 0.713 → v2 0.719, HGB A 0.710 → 0.718, NN A 0.700 → 0.698, RF B 0.716 → 0.725, HGB B 0.706 → 0.718, NN B 0.692 → 0.704.
- **Integrity:** test rows first loaded in 7.5; the v2 fingerprint was recorded before; no setting, threshold, model or primary-model choice was changed after the test evaluation. The test cell re-executes deterministically on every *Run All* (same numbers); that is not a re-evaluation for decisions. The 7.9 interpretation cell (markdown only) was added after the run; the notebook's code cells and outputs are from that single run.
- **Post-test exploratory ideas (not applied):** normalise per-race probabilities to sum to 3; a "dominance of the leading team" feature; evaluate on 2025 as a second genuine test.

### 7.5 Section 8: Exploratory round after the test (17 Oct 2026)

> **Exploratory, separate from the frozen final evaluation (section 7).** Nothing in section 8 changes a frozen model, threshold, fingerprint, the primary model, the official output or any section-7 result; its numbers come from the historical folds only and are not test results.

- **Status:** exploratory, after the test results were seen. Judged on the four historical forward-chaining folds (scoring 2011–2018, model A) only; **validation and test seasons not used**; frozen v1/v2 models, fingerprints, primary model and section-7 results unchanged. Budget 8 configurations × 4 folds = 32 fits (4 s); no further tuning.
- **Configurations:** C0 baseline LR (`C` 1, 21 inputs); C1 + `log(grid)`; C2 + grid spline (4 knots, cubic); C3 + team rank within the race (`cons_points_share`, `cons_podium_rate_5`); C4 + `gap_q1_pct` z-score within the race; C5 + rank within the race by `drv_avg_finish_5`; C6 = C1 + C3 + C4 + C5; C7 = gradient boosting with the frozen v2 setting. Race-relative features use only that race's pre-race features; new columns imputed/scaled inside each fold.
- **Decision rule (fixed in advance):** mean top-3 F1 gain > 0, gain in ≥ 3 of 4 folds, mean PR-AUC not lower, and 95% paired race-resampling interval (158 races) of the top-3 F1 gain excluding 0.

  | Configuration | CV PR-AUC | CV top-3 F1 | correct picks / race | exact-3 rate | top-3 F1 gain | folds improved (top-3 / PR-AUC) | interval (top-3 gain) | rule |
  |---|---|---|---|---|---|---|---|---|
  | C0 baseline | 0.703 | 0.666 | 2.00 | 0.265 | – | – | – | – |
  | C1 + log(grid) | 0.716 | 0.668 | 2.00 | 0.252 | +0.002 | 2 / **4** | [−0.013, +0.015] | no |
  | C2 + grid spline | 0.713 | 0.668 | 2.00 | 0.246 | +0.002 | 2 / 3 | [−0.012, +0.018] | no |
  | C3 + relative team strength | 0.705 | 0.660 | 1.98 | 0.259 | −0.006 | 0 / 3 | [−0.015, +0.000] | no |
  | C4 + relative qualifying pace | 0.698 | 0.670 | 2.01 | 0.264 | +0.004 | 2 / 2 | [−0.006, +0.015] | no |
  | C5 + relative driver form | 0.703 | 0.658 | 1.97 | 0.240 | −0.008 | 0 / 1 | [−0.019, +0.000] | no |
  | C6 + log(grid) + all relative | 0.714 | 0.674 | 2.02 | 0.252 | +0.008 | 3 / 3 | [−0.006, +0.023] | no |
  | C7 GB (frozen v2 setting) | 0.694 | 0.659 | 1.98 | 0.233 | −0.007 | 2 / 2 | [−0.027, +0.015] | no |

- **Findings:** no configuration meets the rule; podium-pick changes are within about ±4 correct picks out of 474. `log(grid)` is the only consistent change, but in **ranking** (PR-AUC +0.013, all 4 folds), not in the podium pick. Picking all three podium drivers right happens in about 1 race in 4; about 2 of 3 are right per race. Rescaling each race's probabilities (e.g. to sum to 3) keeps their order, so it cannot change the top-3 pick and was not tested as an improvement.
- **Recommendation:** keep the frozen v2 logistic regression (model A) as the official model. Only `log(grid)` is worth carrying into a future version, to be evaluated on new seasons (2025+), not on 2019–2024.

### 7.6 Further exploratory rounds outside the notebook (17 Oct 2026; details in `TRAINING_RUNS.md`, Runs 8–9)

> **Exploratory, after the test, and not part of the notebook.** Run as development scripts that reuse the notebook's section 4–5 and 8 code. Historical forward-chaining folds only (scoring 2011–2018, model A); validation and test seasons not used; frozen models, fingerprints, primary model, official output and all section-7 results unchanged. Baseline: LR `C` 1, 21 inputs (CV PR-AUC 0.703, top-3 F1 0.666, 2.00 correct picks per race).

- **Run 8: four proposed enhancements.**
  - **Race-aware rankers:** the pairwise ranker scores top-3 F1 0.655, −0.011 [−0.020, −0.002], **clearly worse**. The listwise ranker scores 0.662, −0.004 [−0.019, +0.011]. Both are worse on PR-AUC (0.683 and 0.686).
  - **`log(grid)`:** PR-AUC 0.716, better in 4 of 4 folds; top-3 F1 +0.002 [−0.013, +0.015].
  - **Exponentially weighted form / teammate-relative pace:** top-3 F1 +0.002 / 0.000 (identical picks), PR-AUC unchanged.
  - **Selecting settings by top-3 F1 instead of PR-AUC** (nested over time, folds 2–4): logistic regression 2.03 vs 2.02 correct picks per race; gradient boosting **2.06 vs 2.10**, worse.
- **Run 9: new information from our own data.**
  - **Recent team qualifying form** (previous 5 races) **± this weekend's trend:** PR-AUC 0.705, +0.002 [−0.001, +0.005]; top-3 F1 −0.004, 0 of 4 folds improved.
  - **Driver history at the circuit:** top-3 F1 −0.006.
  - **Circuit × grid-band history:** −0.002.
  - **All new features together:** −0.007.
  - **Gradient boosting + all new features:** 0.666, only matching the LR baseline.
- **Conclusion:** none meets the section-8 decision rule; every change moves the podium pick by fewer than about 4 correct picks out of 474. The frozen v2 logistic regression (model A) stays the official model.
- **Future work** (needs new data and new seasons to test): practice-session long-run pace (e.g. FastF1, about 2018 onwards), a race-aware objective with richer inputs, and `log(grid)` for better probabilities.

### 7.7 Sections 9–10: Feature-group ablation and explainable AI (17 Oct 2026)

**Common rules:**
- Both sections use only the **saved v2 model files** (`outputs/model_artifacts_v2/`), the model-A training rows (2003–2018) and the validation rows (2019–2021). **No test rows.**
- On loading, they check that the files' fingerprint equals the v2 record and that the frozen validation predictions are reproduced exactly.
- The frozen models, fingerprints, settings and all section-7 results are unchanged.
- The Sections 6–7 searches are not repeated; Sections 9–10 run in about 1 minute.

**Section 9: feature-group ablation** (details: `TRAINING_RUNS.md` Run 10)
- **Groups**, at the 21-input level, each missing-value flag kept with its features:
  - grid: `grid`, `pit_lane_start`;
  - qualifying: `gap_q1_pct`, `grid_minus_quali`, `reached_q3`, `no_q3_format`, `quali_missing`, `q1_time_missing`;
  - standings: the 4 standings inputs;
  - form: the 4 form inputs + `drv_form_missing`, `cons_form_missing`;
  - context: `n_starters`, `drv_experience`, `drv_age`.
- **Method:** a genuine refit for each "without group" variant in each historical fold, with preprocessing re-fitted inside the fold. The group is removed after preprocessing, through a column-selection step, because the imputer needs all 18 raw columns. LR and HGB keep their frozen v2 settings. 48 fits; the full variants reproduce the section-8 baselines exactly.
- **Results** (drop = full − without group; 95% intervals over the 158 fold races):
  - **grid:** LR −0.059 PR-AUC [+0.037, +0.083] and −0.165 correct picks per race [+0.079, +0.260]; HGB −0.077 and −0.190. Clear, 4 of 4 folds.
  - **form:** LR −0.017 [+0.002, +0.033] and −0.070 picks [+0.006, +0.136]; HGB −0.013 (interval includes 0).
  - **qualifying, standings, context:** ≤ 0.004 PR-AUC, not clear. Qualifying overlaps with grid; standings with form and grid; context's age and experience cancel.

**Section 10: explainable AI** (frozen primary model: logistic regression A, v2; validation 2019–2021; examples chosen by fixed rules)
- **Scale:** contributions are in **log-odds** (they add up: base value + contributions = the model's log-odds); probabilities are shown separately.
- **10.2 weights:**
  - grid −2.0 per SD (each grid place multiplies the odds by 0.73);
  - `q1_time_missing` +1.9 (offsets the back-of-grid start and slow-lap fill of penalised fast cars);
  - `drv_no_standing_yet` +0.7 (offsets the "behind everyone" fill);
  - `reached_q3` −0.6 and `drv_dnf_rate_5` +0.4 (correlated with grid / average finish);
  - `drv_experience` +0.4 vs `drv_age` −0.4 (correlation 0.81, they cancel).
- **10.3 global SHAP** (exact: weight × (value − training mean); the shap library's LinearExplainer gives identical values):
  - base value −3.64 log-odds (probability 0.026);
  - mean |SHAP|: grid 1.62, then `drv_avg_finish_5` 0.44, `drv_age` 0.43, `drv_experience` 0.41, `cons_points_share` 0.34.
  - **Group importance, two definitions:**
    - (A) = sum over the group's inputs of mean |SHAP|;
    - (B) = mean over rows of |sum of the group's SHAP values|.
  - (B): grid 1.62, form 0.51, standings 0.42, qualifying 0.40, context 0.26. (A) is higher where inputs cancel (context 0.87, form 0.95, qualifying 0.62).
- **10.4 local SHAP (4 rule-chosen drivers):**
  - Verstappen, pole, Abu Dhabi 2021 (0.909): every input agrees.
  - Kvyat, grid 14, Germany 2019 (0.007, finished 3rd in the rain): a genuine surprise.
  - Verstappen, grid 7 after a penalty, Qatar 2021 (0.796): `grid_minus_quali` = 5 adds +1.0.
  - Verstappen, grid 20 with no Q1 time, Russia 2021 (0.135, finished 2nd): the flag gives back +1.9 of the −2.8 grid cost.
- **10.5 complete races:**
  - **Singapore 2019:** the 3rd vs 4th gap is 0.0005 (Verstappen vs Bottas), decided by standings −0.62, context +0.51, grid +0.32, form −0.25. 2 of 3 picks right; winner Vettel was 5th most likely.
  - **Australia 2019:** the model picks Verstappen (grid 4) over grid-2 Bottas on form +1.00 vs grid −0.64. Model and grid both 2 of 3.
- **10.6 LIME (supplementary):**
  - Run on the 21 filled inputs in original units. A wrapper on the 18 raw features is impractical: LIME can't sample missing values, and the missing-value flags could never switch on.
  - Weak local fits (R² 0.14–0.55); LIME's local predictions are far from the model's (e.g. 0.38 vs 0.91).
  - Stable across seeds (3–4 of the top 5 shared), but only 1–3 of the top 5 overlap with SHAP.
  - Impossible synthetic rows: 1.5% "no Q1 time but a real gap", 1.4% "debut with earlier starts", 2.0% "grid beyond the field".
  - **Not an exact explanation;** SHAP is the reference.
- **10.7 permutation, 50 repeats on validation:**
  - **across rows → PR-AUC drop:** grid 0.417, form 0.074, standings 0.051, qualifying 0.047, context 0.005;
  - **within race → top-3 F1 drop:** grid 0.375, form 0.038, standings 0.031, qualifying 0.024, context 0.001;
  - **grouped permutation** uses one shared row order per group;
  - **individual inputs:** `drv_experience` alone 0.057, but the whole context group only 0.005. This is the correlation effect (age and experience offset each other).
- **10.8 tree SHAP, HGB A** (shap's TreeExplainer; additivity checked): group (B) grid 0.85, form 0.53, standings 0.36, qualifying 0.10, context 0.06.
- **10.9 agreement:** grid is 1st and form 2nd under every method except the raw weights. Qualifying is 1st by the sum of |weights| only (flags that rarely apply or offset each other). Standings and qualifying swap places between methods.
- **Not causal:** these explanations describe what the model uses, not what causes podiums.

**Interpretation caveats** (documented in Section 10's introduction):
- **`pit_lane_start`:** only recorded from 2015 (7 model-A training rows, none on the podium).
- **`no_q3_format`:** constant 0 in validation.
- **`*_no_standing_yet`:** means "0 points yet" in 2003–09, but mostly "first race / debut" later.
- **`grid_minus_quali`:** on 2021–22 sprint weekends it reflects the sprint result as well as penalties.
- **Correlated groups:** grid, qualifying, standings and form all describe car and driver pace.

**Freeze verification before the test (7.5; replaces the earlier plain fingerprint assertion):**
- It compares the current v2 freeze with the record frozen on 17 Oct 2026, embedded in the notebook.
- **Strict, any difference stops the notebook:** the 10 entries; every model's family, settings, training seasons, preprocessing, seed and flags; the 18 input columns and both 21-input lists; every fitted model's preprocessing steps and model type.
- **Numbers** (thresholds, CV / validation PR-AUC, epochs):
  - identical fingerprint → exact match;
  - different numbers under the **same** library versions → stop (unexplained);
  - under **other** versions (e.g. Kaggle) → warning table, stopping only beyond tolerance (threshold 0.05, CV PR-AUC 0.02, validation PR-AUC 0.03).
- Tested in all five scenarios: exact; small numeric with other versions; small numeric with the same versions; a structural change; beyond tolerance.

**Kaggle and dependencies:**
- **SHAP** is pre-installed on Kaggle. If missing, the notebook tries `pip install`; without shap, the linear SHAP values are still computed exactly by formula and plotted with matplotlib, and tree SHAP is reported as missing.
- **LIME** may need `pip install lime` (internet). If that fails, 10.6 prints a clear limitation message instead of silently skipping.
- **Runtime:** Sections 9–10 take about 1 minute; on Kaggle (4 CPUs) the whole notebook is estimated at 1.5–2 hours, mostly Sections 6–7.

### 7.8 Final compliance round (18 Oct 2026): sections 11–12, metrics, season pin

Changes after the compliance audit against the original brief. **No model, feature, threshold, split or frozen result changed**; the v1/v2 fingerprints are verified as before.
- **Accuracy and the metric choice (6.1, 6.9, 7.5):** `scores()` also reports **accuracy**, so the validation and test tables show it for every model, next to a new reference row **"always 'no podium'"** (accuracy ≈ 0.85 with precision/recall/F1 = 0), which shows why accuracy is misleading at a 15% podium rate. 6.1 now states the **primary metric (PR-AUC)** and the **secondary metrics** (ROC-AUC, binary F1, precision, recall, accuracy, race-level top-3 F1, Brier) and why accuracy is not used for any decision.
- **Season pin (1.0) and split check (4.1):** raw tables are filtered in memory to seasons ≤ 2024 right after loading (nothing dropped with our files; a newer Kaggle download could contain 2025+ races, which would otherwise enter the test set). 4.1 asserts the established split: model A 300 / 60 / 68 races and 6,381 / 1,199 / 1,354 rows; model B 986 races / 22,092 training rows.
- **Section 11, final inference (brief deliverable 7):** `PodiumPredictor` in `outputs/podium_inference.py` (written by the notebook, usable outside it). It loads only the frozen primary model (checksums + v2 fingerprint), validates the schema (18 features present, numeric, finite, flags 0/1, ranges; `grid` must be known; missing values allowed elsewhere and filled by the frozen pre-processing), returns probabilities with identifiers and the list of unknown inputs, and for a complete race (rows = `n_starters`, one `raceId`, unique drivers) the 3 most likely podium finishers (ties by grid, then `driverId`, as in 7.10); incomplete races are refused unless `allow_incomplete=True` (warning). Checks: identical probabilities on all 1,199 validation and 1,354 test rows and identical picks in all 128 races. Demos: complete record (Verstappen, 2021 Abu Dhabi, 0.909), the same record with qualifying details and form unknown (0.807), the full 2024 Abu Dhabi GP (Norris, Sainz, Verstappen; 2 of 3 correct), and 11 invalid inputs.
- **Section 12:** consolidated limitations (qualifying-coverage drift, era sensitivity, season predictability, recording quirks, identity choices, information ceiling, top-3 vs grid not clearly different, calibration, sample size, post-test work, non-causal explanations, reproducibility) and conclusions.
- **Post-test exploratory checks (outside the notebook, historical folds only, nothing changed):** Run 11, a driver-form reset after breaks of > 550 days (+ comeback flag): identical logistic-regression podium picks, PR-AUC ±0.004; Run 12, XGBoost (24 settings + nested check): no clear gain over logistic regression. Details in `TRAINING_RUNS.md`.
- **Notebook map** in the title cell (brief section → notebook section); `.gitignore` for `__pycache__/` (the loaders' bytecode cache).
- **Parallel-worker fallback (6.2, `run_jobs`):** if a joblib worker dies or cannot start (`BrokenProcessPool` / `TerminatedWorkerError`), the jobs are re-run in the main process. The jobs are seeded and independent, so results are identical; nothing from the failed attempt is kept. `F1_N_JOBS` (environment variable) sets the number of workers (`1` = serial).
- **Status (18 Oct 2026, 22:10):**
  - **Verified:** the inference interface against the frozen files, 16/16 checks (identical probabilities on all 1,199 validation and 1,354 test rows, identical top-3 picks in all 128 races, complete / partial / full-race cases, every invalid-input case refused). A full VS Code run (21:34–21:59) reproduced the frozen state: v1 `53625c66…` and v2 `2ebc3457…` fingerprints unchanged, all 46 protected files (raw data, cleaned data, anomaly reports, `final_features.csv`, frozen records) byte-identical, all 18 models with identical predictions and thresholds; it passed the 4.1 split check and the 7.5 freeze check and reached section 11.
  - **Not yet done:** that run's outputs were not saved into the notebook, so an **executed notebook with sections 11–12** (and the new accuracy columns) still has to be produced and saved; then the **PDF report** (builder ready: every figure and number is taken from the executed notebook), the **Kaggle** notebook and the submission form.
- **Remaining, outside the notebook:** the PDF report, the Kaggle notebook (attach the dataset `rohanrao/formula-1-world-championship-1950-2020`, enable Internet for `pip install lime`), pushing `amr` to GitHub, and the submission form.

---

## 8. Trial-and-error log (what changed and why)

| What was tried / assumed | What happened | Resolution |
|---|---|---|
| User plan: "fix web-scraping encoding" in constructors | Checked every file: valid UTF-8, 0 mojibake, all constructor names ASCII. The earlier `Aut�dromo` was a terminal display artefact | Step replaced by a documented encoding check |
| User plan: remove non-starters = "Withdrew with 0 laps" | Would leave 1,024 DNQ + 331 DNPQ rows. Also `laps == 0` alone is wrong: 353 Collision + 251 Accident rows have 0 laps (lap-1 crashes, they started) | Use non-start **statuses** |
| Claude's first rule: non-start status **AND grid == 0** | Missed the **2005 US GP**: 14 Michelin drivers withdrew while holding grid slots (Trulli P1, Räikkönen P2, Button P3), inside the 2003+ window, which would teach "pole → no podium" | Final rule: status with **laps == 0**, OR grid == 0 with laps == 0. Keeps the 72 pit-lane starters |
| `grid = 0` recode as one global "max + 1" | Grid sizes range ~20–34 by era | Per-race max grid + 1, applied **after** removing non-starters |
| Standings joined as-is | Same-race standings contain the result (leak) | `*_prerace` tables from the previous round |
| Assumed standings list every driver each round | 142 non-round-1 gaps found in the post-cleaning sanity checks | Documented. Fill rule set for feature engineering (not applied yet) |
| Modeling window: all years + "has qualifying" flag vs 2003+ | The flag would encode the era; 1996–2002 is patchy | **2003+** (8,934 rows is plenty) |
| User asked: why not grid only (all years) instead of qualifying? | Q1 all-years ranks circuits by age (corr −0.14); qualifying recovers penalised fast cars (32% vs 3.4%) and pace within a grid slot | Grid stays primary. Qualifying kept. Grid-only all-years becomes **modeling Attempt A** |
| Mechanical retirements: borderline statuses | Puncture/Damage/Debris/Front wing usually come from contact; Retired gives no cause | "unclear" category, reported as a sensitivity check in Q3 |
| Rule B for duplicates vs best-result rule (A) | Rule A picks a row *because of* its outcome (subtle leakage) | Rule B. Options A–D documented in 1.6 |
| `pd.Series.str.isascii` | Doesn't exist in pandas | `.map(str.isascii)` |
| `fillna(False)` on a merge indicator | pandas FutureWarning (downcasting) | `merge(..., indicator=True)` + `== "both"` |
| Review (15 Oct 2026): Rule B described as "grid > 0 first, then lowest resultId" | Both rows had grid > 0 in all 82 pairs, so only the resultId decided; the proxy kept the wrong car for Ascari 1951 (gave the 1951 French GP 4 podium rows), González 1954 and Moss 1961 | Reworded to "lowest resultId as a proxy" + `RULE_B_OVERRIDES` (3 verified pairs). 1951 French GP now has 3 podium rows |
| Review: status taxonomy had Wheel / Rear wing / Undertray / Vibrations as mechanical but Wheel nut / Front wing as unclear, plus a 7th group "did not start" | Inconsistent; Q3 counts depend on it | Six groups only, consistent assignment (README §5 row 8); Q3 numbers updated |
| Q2 included the Indianapolis 500 (1950–1960) | 380 of its 381 starts were American drivers at a US circuit (all "home"), with no away drivers in the same race: 16% of all home starts | Indy removed in cleaning (2.6.1); Q2 answer unchanged (−0.59 → −0.60 pts), Indy version kept as a robustness column |
| A few hard-coded claims in the markdown ("pre-1965", "few dozen rows", "~135 starts", "1–6 points") | Didn't match the computed numbers exactly | Replaced with computed values or corrected text. **Rule: every number in the markdown must match the output** |
| Section 4 review: `reached_q3 = q3 time present` | 114 starters had advanced to Q3 without a Q3 time (penalties, crashes, tyre saving, 2015 US GP Q3 cancelled). A plain `quali_pos <= 10` would also be wrong: 2016 Australia/Bahrain had 8 Q3 places, some positions include penalties (2007 Belgian GP), and a plain Q2 ranking lets an 11th driver in where the recorded Q2 time isn't the lap that counted (2008 Turkish GP) | Q3 time, or the leftover Q3 places by fastest Q2 among drivers without a Q3 time; asserted per race (4.2.1). 114 starter rows changed |
| Section 4 review: team form grouped by `constructorId` only | Returning IDs reused results from 1955 (Mercedes), 1968 (Honda), 1985 (Alfa Romeo), 1960 (Aston Martin) | History segments split at gaps > 550 days; first race of a segment has no form (4.2.1). 141 rows changed |
| Section 4 review: notebook and generator differed in 10 section-3 cells (the notebook read `outputs/clean_data/`, the generator didn't) | Regenerating would have silently undone the section-3 change | The notebook's cells were copied into `tools/build_notebook.py`; regenerating now reproduces the notebook exactly |
| Section 4 text: "5-race form is best", "everyone runs Q1 in the same conditions", "ablation in section 5" | Form-window differences are within noise; 2003–2005 `q1` was a race-fuel lap; section 5 is pre-processing | Text corrected (README 7.1, notebook 4.5/4.6) |
| Section 5 plan: E2 = capping before vs after imputation | Checked first: for model A the two orders are nearly the same (cap 7.856 vs 7.764), for model B genuinely different (9.015 vs 6.542: the median-filled 64% of the column pulls the percentile down) | E2 reported on model B, model A shown as the near-equivalent case. Median-impute/scale order kept only as a consistency check (scaler ignores NaN when fitted, so it is nearly equivalent) |
| Section 5 text: "no podium finisher among the capped gap rows" (first draft) | 4 capped model-A rows (3 in B) are podium finishes | Text corrected before running; they stay "among the slowest" after the cap |
| Section 6 selection rule, v1: one-standard-error rule with the best setting's fold SD ÷ √4 (unpaired) | First development run: every setting tied in every family (8/8, 18/18, 12/12) because the fold SD (~0.06) mostly measures how hard each season is, which is shared by all settings; the rule always picked the simplest setting even when clearly worse (LR `C` 0.01: 0.674 vs best 0.704) | **Paired** rule: per-fold shortfall vs the best setting, tie if its mean ≤ its standard error (all settings are scored on the same folds). A statistical correction, documented here because it was made after seeing the first selections |
| Section 6 selection rule, v2: paired rule as above | Second run: not monotone: LR `C` 1 + balanced (0.7015, noisy shortfall) counted as a tie while `C` 1 without weights (0.7032, small but consistent shortfall) did not | Ties closed upwards: any setting with a mean at least as high as a tie's is a tie. Final rule (v3). Effect: LR model A `C` 1 without class weight (probabilities sum to ~3.2 per race instead of ~6.3 with balanced weights) |
| Section 6: random forest with `n_jobs=-1` | Predictions differed in the last floating-point digits between calls, so the frozen-model reproduction check failed | `n_jobs=1` everywhere: bit-for-bit reproducible (parallelism moved to the CV jobs) |
| Section 6: CV jobs run one after another | Model-B searches alone took ~25 min (network: ~130 ms per epoch on 20k rows, Python overhead) | (setting, fold) jobs in parallel processes with joblib; each job seeded, so results are identical (best CV scores reproduced exactly); section 6 now ~8–10 min |
| Section 7 (authorised by the user after v1 was frozen): re-tuning | Several v1 best/chosen settings sat on grid edges | Rule-based one-step grid extension, same CV and selection rule; v2 freeze recorded before the test; v1 kept unchanged for comparison. The full notebook run time grew to ≈ 45 min |

---

## 9. Working conventions with this user

- The user is learning and will be **questioned on every choice**. Explain in plain language, with evidence (numbers or plots), and keep explanations short. The user asked for "easy" explanations.
- Work **step by step**. Review the user's plans critically (say what's wrong and why) before applying them.
- Raw CSVs are never modified. Cleaned data stays in memory unless the user asks to save it.
- Each notebook section ends with "What this means / Why it matters" markdown. Each question states the joins, grain, missingness and cutoff.
- Don't commit or push unless asked. Branch `youssef`.
