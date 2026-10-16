# Cleaned F1 dataset

Written by section 2.13 of `f1_podium_prediction.ipynb` (do not edit by hand; re-run the notebook instead).
Raw source: Kaggle / Ergast *Formula 1 World Championship (1950-2024)*. The raw CSVs are unchanged.

Load a table with its original types (nullable integers, dates):

```python
df = load_clean("model_base")   # defined in section 2.13 of the notebook
```
or with plain pandas: `pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip")`, then apply the types in `schema.json`.

## Files

| file | rows | key | what it is |
|---|---|---|---|
| `model_base.csv` | 8,934 | raceId + driverId | Modeling base, 2003+: one row per driver per race, pre-race columns + target `podium` only. |
| `results_full.csv` | 24,645 | raceId + driverId | All cleaned driver-races 1950-2024 (no Indianapolis 500), with race info and post-race columns. Used by the data-engineering questions. Post-race columns are never features. |
| `results.csv` | 24,645 | resultId (and raceId + driverId) | Cleaned results: target, non-starters removed, rule B, pit-lane grid recode, status + category. |
| `qualifying.csv` | 10,494 | qualifyId (and raceId + driverId) | q1/q2/q3 as seconds (`*_sec`); empty strings and the 1002.6 s outlier are NaN. Missing q2/q3 = knocked out. |
| `races.csv` | 1,125 | raceId | date as a date; start_utc where the start time is recorded (2005+). |
| `drivers.csv` | 861 | driverId | dob as a date; nationality fixed; home_country for Q2. |
| `constructors.csv` | 212 | constructorId | Teams. |
| `circuits.csv` | 77 | circuitId | 'United States' merged into 'USA'. |
| `status.csv` | 139 | statusId | status_category: finished / mechanical / accident / regulation / driver / unclear. |
| `driver_standings.csv` | 34,863 | driverStandingsId (and raceId + driverId) | Standings AFTER each race (post-race, leakage!). Use driver_standings_prerace for features. |
| `constructor_standings.csv` | 13,391 | constructorStandingsId (and raceId + constructorId) | Standings AFTER each race (post-race, leakage!). Use constructor_standings_prerace for features. |
| `driver_standings_prerace.csv` | 31,671 | raceId + driverId | Leakage-safe: standings after the previous round, keyed by the race being predicted. Round 1 has no row. |
| `constructor_standings_prerace.csv` | 12,471 | raceId + constructorId | Leakage-safe: standings after the previous round, keyed by the race being predicted. Round 1 has no row. |
| `sprint_results.csv` | 360 | resultId (and raceId + driverId) | Cleaned but not used yet (optional feature later). |
| `constructor_results.csv` | 12,625 | constructorResultsId | Cleaned but not used. |

## What cleaning did (details in notebook section 2)

- `\N` and empty strings -> missing values; text stripped; columns typed (IDs/counts integer, points float, dates as dates, lap times as seconds).
- `results`: target `podium = positionOrder <= 3` created first; 1,645 non-starters removed; one row per (raceId, driverId) by rule B (lowest resultId as a proxy for the starting car, 3 verified overrides); `grid = 0` pit-lane starts -> last slot + 1 with `pit_lane_start`; the Indianapolis 500 (1950-1960, 381 rows) removed.
- `qualifying`: times in seconds; the 1002.6 s Q1 lap set to missing.
- Labels: 'Argentinian ' -> 'Argentine', 'United States' -> 'USA', `drivers.home_country` added; every status has a category.
- `*_prerace` standings: leakage-safe standings from the previous round.
- Not exported: `seasons`, `lap_times`, `pit_stops` (not used).

**Leakage reminder:** post-race columns (`positionOrder`, `points`, `laps`, `status*`, `milliseconds`, `fastestLap*`, `rank`) and the non-prerace standings must never be used as features.
