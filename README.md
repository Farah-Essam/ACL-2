# ACL-2: F1 Podium Prediction (Milestone 1)

Predict whether a Formula 1 driver finishes on the **podium** (`positionOrder <= 3`), using only information available **after qualifying and before the race starts**.
Data: Kaggle / Ergast *Formula 1 World Championship (1950–2024)*, 14 CSVs.
Grain: **one row = one driver in one race**, i.e. a unique (`raceId`, `driverId`).

This README is the project log. It records **what was done, why, and what was tried and changed**, so a new working session (or a teammate) can continue without re-deriving anything.

---

## 1. Repository layout

| Path | What it is |
|---|---|
| `f1_podium_prediction.ipynb` | **The deliverable.** One run-all notebook (the brief requires a single notebook). Currently sections 1–3. Saved with outputs. |
| `tools/build_notebook.py` | **Generator** that writes every markdown and code cell of the notebook. The notebook was built from this script. |
| `tools/run_notebook.py` | Executes the notebook top to bottom with `jupyter_client` and stores the outputs (no `nbconvert` needed). |
| `outputs/anomalies/` | CSVs written by notebook section 1.12: one per raw table, plus `_summary.csv`. |
| `outputs/clean_data/` | **The cleaned dataset**, written by notebook section 2.13: 15 CSVs (`model_base`, `results_full`, every cleaned table, the `*_prerace` standings), `schema.json` (column types), `manifest.csv` (rows + SHA-256) and its own `README.md`. Load with `load_clean("<table>")` to get the exact types back. Rewritten on every run; never edit by hand. |
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
- Environment used: Python 3.10, pandas 2.3, matplotlib 3.10, numpy 2.2, ipykernel. No scipy or statsmodels needed so far.
- On Windows, set `PYTHONIOENCODING=utf-8` when printing names with accents from a script (the terminal codepage otherwise fails on e.g. *Räikkönen*).

### Git
- Work happens on branch **`youssef`** (pushed to `origin`, repo `Farah-Essam/ACL-2`).
- Commits so far: audit + cleaning + questions (`d7fda8a`), then the cleaning review fixes, Indianapolis 500 removal, cleaned-dataset export and the approved section-4 plan. `Project.code-workspace` is a local editor file and is not committed.
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
- **Deliverables:** run-all notebook, inference function (complete + partially-missing record), PDF report, Kaggle notebook link, GitHub repo link. **Deadline: 18 Oct 2026, 23:59.**
- Evaluation: every team member can be asked to justify any step. "An AI suggested it" isn't a justification, so every choice needs evidence in the notebook.

---

## 3. Notebook contents and status

| Section | Status | Content |
|---|---|---|
| **1. Data Audit** (1.0–1.13) | done | Read-only audit, anomaly CSV export, raw-file fingerprint check |
| **2. Data Cleaning** (2.1–2.13) | **closed** (review fixes applied 15 Oct 2026; 2.13 exports the cleaned dataset to `outputs/clean_data/`) | All cleaning applied in memory on copies. 2.9.1 status mapping with row counts; 2.10 re-validation (all 21 FKs); 2.10.2 proof that standings are post-race; 2.11 six before/after plot pairs |
| **3. Data-Engineering Questions** (3.1–3.3) | done | Q1 (+ two robustness sections), Q2, Q3, each with plots and plain-language answers |
| 4. Feature engineering | **next** | see section 7 of this README |
| 5+. Split, models, experiments, XAI, inference | to do | |

Variables available after running sections 1–3:
- `RAW`: dict of raw tables, **all strings** (`dtype=str`, `keep_default_na=False`).
- `NAIVE`: default `pd.read_csv` (comparison only).
- `CLEAN`: cleaned tables (dict).
- `results_full`: all cleaned driver-races 1950–2024 (24,645 rows, **without the Indianapolis 500**), **with** post-race columns; used for the questions.
- `INDY500_RESULTS`: the 381 cleaned Indianapolis 500 rows set aside in 2.6.1, used only by the Q2 robustness column.
- `model_base`: 2003+, pre-race columns + target only; the start of the modeling table.
- Helpers: `is_missing`, `duration_to_seconds`, `log_step`, `STATUS_GROUPS`, `STATUS_CATEGORIES`, `NON_START_STATUSES`, `RULE_B_OVERRIDES`, `UNUSED_TABLES`, `DEFERRED_TABLES`, `NATIONALITY_TO_COUNTRY`, `season_order`, `CLEAN_DATA_DIR`, `load_clean(name)` (reads an exported table back with its exact types).

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

## 7. Next steps (agreed but not done)

### 7.1 Section 4: Feature selection & engineering. APPROVED PLAN (15 Oct 2026), not started

Nothing is built yet. Start only when the user says so. Notebook layout:
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
| Qualifying | qualifying position, **gap to pole % from Q1 only**, `grid − quali_pos` (penalty indicator), reached Q3 | Gap from Q1 because everyone runs Q1 in the same conditions. **Written trial** comparing it with best-of-Q1/Q2/Q3, noting the 2006–2009 race-fuel Q3 and the early-2005 aggregate (q1 + q2) format. Reached Q3 only exists from 2006: **add a flag for earlier years** (format had no Q3) |
| Standings | **previous-race standings only** (`*_prerace`): championship **position** and **share of the leader's points**, wins; driver and constructor; `no_standing_yet` flag (round 1 and gaps) | **Never raw points** (scale changed in 2010: win 10 → 25) |
| Form | driver: podium rate / average finish / retirement rate over the last N races; constructor: podium rate over the last N races | Computed with **`shift(1)` in race order**. **Asserts:** the current race is excluded, and a driver's first race has no form value |
| Context | number of starters, driver experience (career starts), age; **home race as one trial** | Home race kept or dropped on **validation PR-AUC**, with the result written down (Q2 found no effect) |

**Excluded (decided):** team (constructor) and circuit as features; year / era feature; sprint finish (out for now; `sprint_results` stays cleaned for a later optional trial).

**Redundancy:** for highly correlated pairs (grid vs qualifying position; championship position vs points share), state in the notebook which one is kept and why.

**Trials (4.5):** logistic regression, **grid only** first, then **add one group at a time**, scored by **validation PR-AUC**. **Every trial in its own cell**, all kept in the notebook. This is feature selection; the full ablation (all models) is in the experiments section.

### 7.2 Later steps

1. **Models:** baseline (e.g. logistic regression on grid) + ≥ 2 statistical models + shallow NN; ROC-AUC, PR-AUC, F1 on val/test; threshold chosen on validation. Podium rate ~14%, so PR-AUC matters.
2. **Two model versions on the same validation/test seasons:** **A** = 2003+ (full qualifying); **B** = all years 1950–2018 for training, missing qualifying left as NaN + flag. (Replaces the earlier "grid only, all seasons" idea.)
3. Ablation by feature group (grid, qualifying, standings, form); ≥ 2 preprocessing-order experiments; XAI; inference function; report.

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

---

## 9. Working conventions with this user

- The user is learning and will be **questioned on every choice**. Explain in plain language, with evidence (numbers or plots), and keep explanations short. The user asked for "easy" explanations.
- Work **step by step**. Review the user's plans critically (say what's wrong and why) before applying them.
- Raw CSVs are never modified. Cleaned data stays in memory unless the user asks to save it.
- Each notebook section ends with "What this means / Why it matters" markdown. Each question states the joins, grain, missingness and cutoff.
- Don't commit or push unless asked. Branch `youssef`.
