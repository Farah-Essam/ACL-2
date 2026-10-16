"""Generates f1_podium_prediction.ipynb (all markdown + code cells). Usage: python tools/build_notebook.py f1_podium_prediction.ipynb"""
import json, sys

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


# ---------------------------------------------------------------- title
md(r"""
# F1 Podium Prediction

**Task:** predict whether a driver finishes on the podium (`positionOrder <= 3`) using only information available **after qualifying and before the race starts**.
**Grain (one prediction = one row):** one driver entering one race → unique (`raceId`, `driverId`).
**Data:** Kaggle / Ergast *Formula 1 World Championship (1950–2024)*, 14 relational CSVs.
""")

md(r"""
# 1. Data Audit

This section **only looks** at the data. Nothing is cleaned or modified. The only files written are the anomaly reports in 1.12 (new files in `outputs/anomalies/`).

Ground rules:
- **Raw files stay immutable.** The CSVs are read straight from `Dataset.zip` (or the Kaggle input folder); nothing is extracted, and nothing is written next to the raw data.
  We fingerprint every file (SHA-256) now and re-check it at the end of the section (1.13).
- **Everything is loaded as text first** (`dtype=str`, `keep_default_na=False`), so pandas cannot silently convert placeholders
  like `\N`, `""` or `"NA"` into `NaN`, or guess types for us. We want to *see* every raw value before deciding what it means.
""")

# ---------------------------------------------------------------- 1.0 setup
md(r"""
## 1.0 Setup and raw loading
""")

code(r'''
import hashlib
import io
import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

SENTINEL = "\\N"  # the literal two characters backslash + N used by Ergast for "missing"

TABLES = [
    "circuits", "constructors", "constructor_results", "constructor_standings", "drivers",
    "driver_standings", "races", "results", "qualifying", "lap_times", "pit_stops",
    "sprint_results", "seasons", "status",
]

# Where the raw data lives: an env var override, the Kaggle mount, or the local zip / folder.
CANDIDATE_SOURCES = [
    os.environ.get("F1_DATA"),
    "/kaggle/input/formula-1-world-championship-1950-2020",
    "../Dataset.zip", "Dataset.zip", "../Dataset", "Dataset", "data/raw",
]


def find_data_source():
    for cand in filter(None, CANDIDATE_SOURCES):
        p = Path(cand)
        if p.is_file() and zipfile.is_zipfile(p):
            return p
        if p.is_dir() and (p / "results.csv").exists():
            return p
    for p in Path("/kaggle/input").glob("*/results.csv"):  # any Kaggle dataset slug
        return p.parent
    raise FileNotFoundError("Raw F1 data not found; set the F1_DATA environment variable.")


DATA_SOURCE = find_data_source()


def read_raw_bytes(table):
    """Return the untouched bytes of one raw CSV (read-only, never extracted to disk)."""
    if DATA_SOURCE.is_file():
        with zipfile.ZipFile(DATA_SOURCE) as zf:
            return zf.read(f"{table}.csv")
    return (DATA_SOURCE / f"{table}.csv").read_bytes()


print("Raw data source:", DATA_SOURCE.resolve())
''')

code(r'''
RAW_BYTES = {t: read_raw_bytes(t) for t in TABLES}
RAW_FINGERPRINT = {t: hashlib.sha256(b).hexdigest() for t, b in RAW_BYTES.items()}

# RAW: every value kept exactly as written in the file (all strings, nothing becomes NaN).
RAW = {t: pd.read_csv(io.BytesIO(b), dtype=str, keep_default_na=False, na_filter=False)
       for t, b in RAW_BYTES.items()}

# NAIVE: what plain pd.read_csv() would give us; only used to show what pandas guesses.
NAIVE = {t: pd.read_csv(io.BytesIO(b), low_memory=False) for t, b in RAW_BYTES.items()}

print(f"Loaded {len(RAW)} tables; fingerprints recorded.")
''')

md(r"""
**Why load twice?** `RAW` is our source of truth: every cell is the exact string from the file.
`NAIVE` is only there for comparison: it shows what pandas would *silently* do by default. By default pandas turns `""`, `"NA"`, `"null"`, etc. into `NaN`,
but it does **not** recognise `\N`, so a numeric column containing `\N` becomes text (`object`). Either way we'd lose track of which placeholder was used.
""")

# ---------------------------------------------------------------- 1.1 overview
md(r"""
## 1.1 Table overview: shape, columns, dtypes, samples
""")

code(r'''
overview = pd.DataFrame({
    "rows": {t: d.shape[0] for t, d in RAW.items()},
    "columns": {t: d.shape[1] for t, d in RAW.items()},
    "file size (MB)": {t: round(len(RAW_BYTES[t]) / 1e6, 2) for t in TABLES},
}).sort_values("rows", ascending=False)
display(overview)
''')

code(r'''
def describe_table(name):
    raw, naive = RAW[name], NAIVE[name]
    summary = pd.DataFrame({
        "raw dtype": raw.dtypes.astype(str),
        "naive pandas dtype": naive.dtypes.astype(str),
        "n_unique": raw.nunique(),
        "first value": raw.iloc[0],
    })
    display(Markdown(f"### `{name}` - {raw.shape[0]:,} rows x {raw.shape[1]} columns"))
    display(summary)
    display(raw.sample(3, random_state=42))


for t in TABLES:
    describe_table(t)
''')

md(r"""
**What this shows.** All 14 tables loaded. `lap_times` (~589k rows) is by far the largest. `results` (~26.8k rows) is the table closest to our grain.
`raw dtype` is always `object` because we forced text. The interesting column is **`naive pandas dtype`**:
- Columns that look numeric but come out as `object` (e.g. `results.position`, `results.milliseconds`, `results.fastestLap`, `drivers.number`) contain a non-numeric placeholder. That placeholder is `\N` (see 1.2).
- Columns that really are dates or durations (`races.date`, `qualifying.q1`, `lap_times.time`, ...) are `object` too. pandas doesn't parse them by default, so they need **semantic typing** (1.3).

**Why it matters.** If we trusted the default load, numeric features would arrive as strings. A model would then either crash or treat them as categories.
""")

# ---------------------------------------------------------------- 1.2 sentinels
md(r"""
## 1.2 Sentinels and placeholder values

For every column we count:
- `\N`: the Ergast missing-value sentinel.
- empty strings `""` and whitespace-only strings.
- values with leading/trailing spaces (they break joins and group-bys silently).
- other common "missing" tokens (`NA`, `N/A`, `null`, `None`, `nan`, `-`, `?`).
""")

code(r'''
NA_TOKENS = {"NA", "N/A", "n/a", "na", "null", "NULL", "None", "none", "nan", "NaN", "-", "--", "?"}

rows = []
for t, df in RAW.items():
    for col in df.columns:
        s = df[col]
        stripped = s.str.strip()
        counts = {
            "\\N": int((s == SENTINEL).sum()),
            "empty": int((s == "").sum()),
            "whitespace-only": int(((stripped == "") & (s != "")).sum()),
            "padded (leading/trailing space)": int(((s != stripped) & (stripped != "")).sum()),
            "other NA token": int(s.isin(NA_TOKENS).sum()),
        }
        if any(counts.values()):
            rows.append({"table": t, "column": col, "rows": len(s), **counts})

placeholder_report = pd.DataFrame(rows)
placeholder_report["% \\N"] = (100 * placeholder_report["\\N"] / placeholder_report["rows"]).round(1)
placeholder_report = placeholder_report.sort_values(["table", "% \\N"], ascending=[True, False]).reset_index(drop=True)
display(placeholder_report)

clean_tables = sorted(set(TABLES) - set(placeholder_report["table"]))
print("Tables with no placeholder of any kind:", clean_tables)
''')

code(r'''
plot_df = placeholder_report[placeholder_report["\\N"] > 0].copy()
plot_df["label"] = plot_df["table"] + "." + plot_df["column"]
plot_df = plot_df.sort_values("% \\N")

fig, ax = plt.subplots(figsize=(9, 0.32 * len(plot_df) + 1))
ax.barh(plot_df["label"], plot_df["% \\N"], color="#4C72B0")
for y, v in enumerate(plot_df["% \\N"]):
    ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=8)
ax.set_xlabel(r"% of rows equal to '\N'")
ax.set_title(r"Share of '\N' sentinel per column (columns with at least one)")
ax.set_xlim(0, 110)
plt.tight_layout()
plt.show()
''')

md(r"""
### 1.2.1 Zeros that may be placeholders

`\N` is not the only way "unknown" is encoded. Some numeric columns use `0` where 0 is not a real value (there is no grid slot 0, no position 0).
""")

code(r'''
zero_checks = [
    ("results", "grid"), ("sprint_results", "grid"), ("qualifying", "position"),
    ("results", "positionOrder"), ("results", "number"), ("qualifying", "number"),
    ("results", "laps"), ("circuits", "alt"),
]
zero_report = pd.DataFrame([
    {"table": t, "column": c, "n_zero": int((RAW[t][c] == "0").sum()),
     "% zero": round(100 * (RAW[t][c] == "0").mean(), 2)}
    for t, c in zero_checks
])
display(zero_report)
''')

md(r"""
### 1.2.2 Inconsistent labels (matter for joins on text)

Question 2 (home-country podiums) needs to match a **driver nationality** to a **circuit country**. Both are free text, so we check how they're spelled.
""")

code(r'''
nat = RAW["drivers"]["nationality"]
country = RAW["circuits"]["country"]

print("Driver nationalities with leading/trailing spaces:", sorted(nat[nat != nat.str.strip()].unique()))
print("\nCircuit countries:", sorted(country.unique()))
print("\nDriver nationalities:", sorted(nat.str.strip().unique()))

direct_matches = sorted(set(nat.str.strip()) & set(country))
print("\nNationality strings that literally equal a circuit country:", direct_matches)
''')

md(r"""
**What this means.**
- `\N` appears in exactly the columns we'd expect to be "not applicable" or "not recorded". Examples: `results.position` (~41%, drivers who weren't classified), `results.time/milliseconds` (~71%, only finishers on the lead lap get a time), `fastestLap*` (~69%, only recorded since 2004), `qualifying.q2/q3` (only drivers who advanced to Q2/Q3), `races.fp*/quali_*/sprint_*` (session schedule only recorded recently), `drivers.number/code` (permanent numbers/codes are a modern thing).
  So **`\N` is mostly "doesn't exist" rather than "lost"**. That changes how we treat it: a missing Q3 time means "eliminated earlier", which is information, not noise.
- **Inconsistent sentinel:** `qualifying.q2/q3` also contain **empty strings** (not `\N`) in some recent rows. A cleaner that only replaces `\N` would miss them, so both must map to missing.
- **`0` as placeholder:** `results.grid = 0` (~6% of rows) is not a real slot (explained in 1.8).
- **Text labels don't line up:** `'Argentinian '` has a trailing space (and `Argentine` also exists). Nationalities are demonyms (`British`, `American`) while countries are names (`UK`, `USA` **and** `United States`). Almost no nationality equals a country string literally, so Q2 needs an explicit nationality → country mapping.

**Why it matters.** Mis-handled placeholders become fake values (e.g. `0` grid read as "pole position", which is better than P1!) or silently drop rows from joins.
""")

# ---------------------------------------------------------------- 1.3 semantic typing
md(r"""
## 1.3 Semantic typing: dates, times, durations

For each temporal column we show the **value shapes** (every digit replaced by `9`, so `1:27.452` → `9:99.999`), example raw values, and the target type.
Then we **test-parse** each column in memory (not saved) and count values that fail to parse. That way the conversion rule is proven, not assumed.
""")

code(r'''
def is_missing(s):
    return (s == SENTINEL) | (s.str.strip() == "")


def value_shapes(s, top=4):
    shapes = s[~is_missing(s)].str.replace(r"\d", "9", regex=True).value_counts()
    return ", ".join(f"{k} ({v:,})" for k, v in shapes.head(top).items()) + (" ..." if len(shapes) > top else "")


def duration_to_seconds(v):
    """'h:mm:ss.sss' | 'm:ss.sss' | 'ss.sss' -> float seconds (NaN if not parseable)."""
    try:
        secs = 0.0
        for part in v.split(":"):
            secs = secs * 60 + float(part)
        return secs
    except (ValueError, AttributeError):
        return np.nan


def parse_failures(s, kind):
    present = s[~is_missing(s)]
    if kind == "date":
        parsed = pd.to_datetime(present, format="%Y-%m-%d", errors="coerce")
    elif kind == "clock":
        parsed = pd.to_datetime(present, format="%H:%M:%S", errors="coerce")
    elif kind == "duration":
        parsed = present.map(duration_to_seconds)
    elif kind == "number":
        parsed = pd.to_numeric(present, errors="coerce")
    else:
        return np.nan
    return int(parsed.isna().sum())


SEMANTIC = [
    # table, column, parse kind, target type, note
    ("races", "date", "date", "datetime64 (date)", "race day"),
    ("races", "time", "clock", "time of day -> combine with date (UTC)", "start time; only recorded from ~2005"),
    ("races", "quali_date", "date", "datetime64 (date)", "session schedule, recent seasons only"),
    ("races", "quali_time", "clock", "time of day", "session schedule, recent seasons only"),
    ("races", "fp1_date", "date", "datetime64 (date)", "same pattern for fp2/fp3/sprint"),
    ("drivers", "dob", "date", "datetime64 (date)", "for driver age at race"),
    ("qualifying", "q1", "duration", "float seconds", "lap time m:ss.sss"),
    ("qualifying", "q2", "duration", "float seconds", "missing = eliminated in Q1 / format"),
    ("qualifying", "q3", "duration", "float seconds", "missing = eliminated in Q2 / format"),
    ("results", "time", "mixed", "DO NOT PARSE -> use milliseconds", "winner: total time; others: '+gap'"),
    ("results", "milliseconds", "number", "Int64 milliseconds", "total race time, finishers only"),
    ("results", "fastestLapTime", "duration", "float seconds", "post-race info"),
    ("results", "fastestLapSpeed", "number", "float km/h", "post-race info"),
    ("sprint_results", "time", "mixed", "DO NOT PARSE -> use milliseconds", "same format as results.time"),
    ("sprint_results", "fastestLapTime", "duration", "float seconds", ""),
    ("lap_times", "time", "duration", "float seconds", "also has milliseconds column"),
    ("pit_stops", "time", "clock", "time of day (local)", "wall-clock time of the stop"),
    ("pit_stops", "duration", "duration", "float seconds", "some > 60 s written as m:ss.sss"),
]

typing_report = pd.DataFrame([
    {
        "table.column": f"{t}.{c}",
        "non-missing": int((~is_missing(RAW[t][c])).sum()),
        "value shapes (count)": value_shapes(RAW[t][c]),
        "examples": " | ".join(RAW[t][c][~is_missing(RAW[t][c])].drop_duplicates().head(3)),
        "target type": target,
        "parse failures": parse_failures(RAW[t][c], kind),
        "note": note,
    }
    for t, c, kind, target, note in SEMANTIC
])
display(typing_report)
''')

md(r"""
**Cross-check:** `lap_times`, `pit_stops` and `results` carry a `milliseconds` column next to the text time. If our parser is right, both must agree.
""")

code(r'''
checks = []
for t, text_col in [("lap_times", "time"), ("pit_stops", "duration")]:
    d = RAW[t]
    ok = ~is_missing(d[text_col]) & ~is_missing(d["milliseconds"])
    parsed_ms = d.loc[ok, text_col].map(duration_to_seconds) * 1000
    diff = (parsed_ms - pd.to_numeric(d.loc[ok, "milliseconds"])).abs()
    checks.append({"column": f"{t}.{text_col}", "rows compared": int(ok.sum()),
                   "max |parsed - milliseconds| (ms)": round(diff.max(), 3),
                   "rows differing by > 1 ms": int((diff > 1).sum())})

# results.time only equals milliseconds for the winner (absolute time); everyone else has a '+gap'.
r = RAW["results"]
has_time = ~is_missing(r["time"])
checks.append({"column": "results.time", "rows compared": int(has_time.sum()),
               "max |parsed - milliseconds| (ms)": np.nan,
               "rows differing by > 1 ms": int(r.loc[has_time, "time"].str.startswith("+").sum())})
display(pd.DataFrame(checks))
print("results.time rows starting with '+' are gaps to the winner, not race times.")
''')

md(r"""
### 1.3.1 Numeric columns stored as text

Every column where **all non-missing values parse as numbers** should become numeric. If it has missing values, it should use a nullable type (`Int64` / `float`), so the missing ones stay missing instead of becoming `0`.
""")

code(r'''
num_rows = []
for t, df in RAW.items():
    for col in df.columns:
        s = df[col]
        present = s[~is_missing(s)]
        if present.empty:
            continue
        as_num = pd.to_numeric(present, errors="coerce")
        if as_num.notna().all():
            is_int = bool((as_num == np.floor(as_num)).all())
            n_missing = int(is_missing(s).sum())
            num_rows.append({"table": t, "column": col, "naive dtype": str(NAIVE[t][col].dtype),
                             "missing": n_missing,
                             "proposed": ("Int64" if is_int else "float64") + (" (nullable)" if n_missing else "")})

numeric_report = pd.DataFrame(num_rows)
print(f"{len(numeric_report)} columns are purely numeric apart from placeholders.")
display(numeric_report[numeric_report["naive dtype"] == "object"]
        .rename(columns={"naive dtype": "naive dtype (wrong)"}))
''')

md(r"""
**What this means.**
- **Dates** (`races.date`, `drivers.dob`, session dates) are clean `YYYY-MM-DD`, with 0 parse failures. They become real dates, so we can sort races chronologically (essential for "past races only" features) and compute driver age.
- **Lap and qualifying times** (`m:ss.sss`) become **seconds as float**, which makes them comparable and usable as numbers. Some `pit_stops.duration` values over a minute are written `m:ss.sss` (red-flag stoppages), and a few lap times use `h:mm:ss.sss`. That's why the parser handles 1–3 parts instead of assuming one format. The milliseconds cross-check shows the parser reproduces the official milliseconds exactly.
- **`results.time` must not be parsed as a duration.** For the winner it's the total race time, for everyone else it's a gap (`+5.478`). The same column mixes two meanings. `milliseconds` is the consistent version, and it's post-race anyway.
- The numeric list shows which ID/count columns are text only because of `\N`.

**Why it matters.** Correct types are the precondition for everything else: chronological ordering for leakage-safe features, numeric features for the models, and correct joins on ID columns.
""")

# ---------------------------------------------------------------- 1.4 PK
md(r"""
## 1.4 Primary-key check

For each table we test the **surrogate key** (the `...Id` column) and the **natural key** (what the row *means*, e.g. "one driver in one race").
A valid key is **unique** and **non-null** (no `\N` / empty).
""")

code(r'''
KEYS = [
    # table, key columns, key kind
    ("circuits", ["circuitId"], "surrogate"), ("circuits", ["circuitRef"], "natural"),
    ("constructors", ["constructorId"], "surrogate"), ("constructors", ["constructorRef"], "natural"),
    ("drivers", ["driverId"], "surrogate"), ("drivers", ["driverRef"], "natural"),
    ("seasons", ["year"], "natural"),
    ("status", ["statusId"], "surrogate"),
    ("races", ["raceId"], "surrogate"), ("races", ["year", "round"], "natural"),
    ("results", ["resultId"], "surrogate"), ("results", ["raceId", "driverId"], "natural (our grain)"),
    ("qualifying", ["qualifyId"], "surrogate"), ("qualifying", ["raceId", "driverId"], "natural"),
    ("sprint_results", ["resultId"], "surrogate"), ("sprint_results", ["raceId", "driverId"], "natural"),
    ("driver_standings", ["driverStandingsId"], "surrogate"), ("driver_standings", ["raceId", "driverId"], "natural"),
    ("constructor_standings", ["constructorStandingsId"], "surrogate"), ("constructor_standings", ["raceId", "constructorId"], "natural"),
    ("constructor_results", ["constructorResultsId"], "surrogate"), ("constructor_results", ["raceId", "constructorId"], "natural"),
    ("lap_times", ["raceId", "driverId", "lap"], "natural (no surrogate)"),
    ("pit_stops", ["raceId", "driverId", "stop"], "natural (no surrogate)"),
]

pk_rows = []
for t, key, kind in KEYS:
    d = RAW[t]
    n_null = int(d[key].apply(is_missing).any(axis=1).sum())
    n_dup_rows = int(d.duplicated(key, keep=False).sum())
    pk_rows.append({"table": t, "key": " + ".join(key), "kind": kind, "rows": len(d),
                    "null keys": n_null, "rows in duplicated keys": n_dup_rows,
                    "valid PK": n_null == 0 and n_dup_rows == 0})

pk_report = pd.DataFrame(pk_rows)
display(pk_report.style.apply(
    lambda row: ["background-color: #f8d7da" if not row["valid PK"] else "" for _ in row], axis=1))
''')

md(r"""
**What this means.** Every surrogate `...Id` key is unique and non-null, so the tables are internally consistent.
The **only key that fails is `results (raceId, driverId)`**: the same driver appears more than once in the same race. That's exactly our modeling grain, so it must be fixed (see 1.6).
`qualifying` and `driver_standings` *are* unique per (raceId, driverId), so joining them onto `results` won't multiply rows.

**Why it matters.** If a join key isn't unique on the "one" side, a join silently duplicates rows. Duplicated rows inflate the dataset, bias the model, and can leak the same race into train and test.
""")

# ---------------------------------------------------------------- 1.5 FK
md(r"""
## 1.5 Foreign-key check

For every foreign key we count child values that **don't exist** in the parent table (orphans). We also check the reverse direction where it matters (parents with no children).
""")

code(r'''
FKS = [
    # child table, child column, parent table, parent column
    ("races", "circuitId", "circuits", "circuitId"),
    ("races", "year", "seasons", "year"),
    ("results", "raceId", "races", "raceId"),
    ("results", "driverId", "drivers", "driverId"),
    ("results", "constructorId", "constructors", "constructorId"),
    ("results", "statusId", "status", "statusId"),
    ("qualifying", "raceId", "races", "raceId"),
    ("qualifying", "driverId", "drivers", "driverId"),
    ("qualifying", "constructorId", "constructors", "constructorId"),
    ("sprint_results", "raceId", "races", "raceId"),
    ("sprint_results", "driverId", "drivers", "driverId"),
    ("sprint_results", "constructorId", "constructors", "constructorId"),
    ("sprint_results", "statusId", "status", "statusId"),
    ("driver_standings", "raceId", "races", "raceId"),
    ("driver_standings", "driverId", "drivers", "driverId"),
    ("constructor_standings", "raceId", "races", "raceId"),
    ("constructor_standings", "constructorId", "constructors", "constructorId"),
    ("constructor_results", "raceId", "races", "raceId"),
    ("constructor_results", "constructorId", "constructors", "constructorId"),
    ("lap_times", "raceId", "races", "raceId"),
    ("lap_times", "driverId", "drivers", "driverId"),
    ("pit_stops", "raceId", "races", "raceId"),
    ("pit_stops", "driverId", "drivers", "driverId"),
]

fk_rows = []
for child, ccol, parent, pcol in FKS:
    vals = RAW[child][ccol]
    missing = is_missing(vals)
    orphan = ~missing & ~vals.isin(RAW[parent][pcol])
    fk_rows.append({"FK": f"{child}.{ccol} -> {parent}.{pcol}", "rows": len(vals),
                    "null FK": int(missing.sum()), "orphans": int(orphan.sum()),
                    "orphan examples": ", ".join(vals[orphan].unique()[:5])})
fk_report = pd.DataFrame(fk_rows)
display(fk_report)
print("All foreign keys reconcile:", bool((fk_report["orphans"] == 0).all()))
''')

code(r'''
# Reverse direction + cross-table consistency checks.
res, quali, races = RAW["results"], RAW["qualifying"], RAW["races"]

races_without_results = races[~races["raceId"].isin(res["raceId"])]
q_keys = quali.set_index(["raceId", "driverId"]).index
r_keys = res.set_index(["raceId", "driverId"]).index
quali_without_result = int((~q_keys.isin(r_keys)).sum())

# Same driver, same race: does qualifying list the same constructor as results?
both = res.merge(quali, on=["raceId", "driverId"], suffixes=("_res", "_q"))
team_mismatch = both[both["constructorId_res"] != both["constructorId_q"]]

print(f"Races with no results rows:           {len(races_without_results)}")
print(f"Qualifying rows with no results row:  {quali_without_result}")
print(f"Drivers never appearing in results:   {int((~RAW['drivers']['driverId'].isin(res['driverId'])).sum())}")
print(f"results vs qualifying constructorId mismatches: {len(team_mismatch)}")
display(team_mismatch.merge(races[["raceId", "year", "name"]], on="raceId")
        [["raceId", "year", "name", "driverId", "constructorId_res", "constructorId_q", "position_q", "grid"]])
''')

md(r"""
**What this means.** Every foreign key resolves (0 orphans). Every race has results and every qualifying row has a matching result, so a **left join from `results`** loses nothing.
The one soft inconsistency is a handful of rows where `qualifying` and `results` list a *different constructor* for the same driver in the same race (data-entry quirks). When we need the constructor, we'll take it from `results`, the table that defines the grain.

**Why it matters.** Orphan keys would make inner joins silently drop rows, or left joins produce unexplained NaNs. Here we can join safely, and any NaN after a join means "not recorded", not "broken link".
""")

# ---------------------------------------------------------------- 1.6 duplicates
md(r"""
## 1.6 Duplicate (raceId, driverId) pairs in `results`
""")

code(r'''
drivers = RAW["drivers"].assign(driver=lambda d: d["forename"] + " " + d["surname"])[["driverId", "driver"]]
dup_mask = res.duplicated(["raceId", "driverId"], keep=False)
dups = (res[dup_mask]
        .merge(races[["raceId", "year", "name"]], on="raceId")
        .merge(drivers, on="driverId")
        .merge(RAW["status"], on="statusId")
        .sort_values(["raceId", "driverId", "resultId"]))

pairs = dups.groupby(["raceId", "driverId"])
print(f"Duplicated rows: {len(dups)}  |  distinct (raceId, driverId) pairs: {pairs.ngroups}  "
      f"|  share of results: {100 * len(dups) / len(res):.2f}%")
print("Rows per duplicated pair:", pairs.size().value_counts().sort_index().to_dict())
print("Constructors per duplicated pair:", pairs["constructorId"].nunique().value_counts().sort_index().to_dict())
print("Years:", f"{dups['year'].min()}-{dups['year'].max()}")
''')

code(r'''
per_year = dups.groupby("year").apply(lambda g: g.groupby(["raceId", "driverId"]).ngroups, include_groups=False)
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(per_year.index.astype(int), per_year.values, color="#C44E52")
ax.set_title("Duplicated (raceId, driverId) pairs per season")
ax.set_xlabel("season")
ax.set_ylabel("pairs")
plt.tight_layout()
plt.show()

display(dups.groupby(["year", "name"]).agg(pairs=("driverId", lambda s: s.nunique()), rows=("resultId", "size"))
        .reset_index().sort_values("year"))
''')

md(r"""
A few complete examples (all original columns), chosen to show the different kinds of duplicate:
""")

code(r'''
def show_pair(race_id, driver_id):
    d = dups[(dups["raceId"] == race_id) & (dups["driverId"] == driver_id)]
    display(Markdown(f"**{d['year'].iat[0]} {d['name'].iat[0]} - {d['driver'].iat[0]}**"))
    display(d[list(res.columns) + ["status"]])


# Pick one example of each kind found in the data.
kinds = pairs.agg(n_constructors=("constructorId", "nunique"),
                  any_podium=("positionOrder", lambda s: (s.astype(int) <= 3).any()),
                  all_no_start=("grid", lambda s: (s == "0").all())).reset_index()
examples = [
    kinds[(kinds["n_constructors"] == 1) & kinds["any_podium"]].iloc[0],    # shared drive that reached the podium
    kinds[(kinds["n_constructors"] == 1) & ~kinds["any_podium"]].iloc[0],   # shared drive, no podium
    kinds[kinds["n_constructors"] > 1].iloc[0],                              # two different teams' cars
    kinds[kinds["all_no_start"]].iloc[-1],                                   # entries that never started
]
for ex in examples:
    show_pair(ex["raceId"], ex["driverId"])
''')

md(r"""
**Effect on the target:** if duplicates are left in, a race can have more than 3 "podium" rows.
""")

code(r'''
po = res.assign(positionOrder=res["positionOrder"].astype(int))
podium_rows = po[po["positionOrder"] <= 3].groupby("raceId").size()
odd = podium_rows[podium_rows != 3]
print("Races by number of rows with positionOrder <= 3:", podium_rows.value_counts().sort_index().to_dict())
dup_races = set(dups["raceId"])
print(f"Races with != 3 podium rows: {len(odd)}, of which contain duplicated pairs: {sum(r in dup_races for r in odd.index)}")
print("positionOrder values repeated within a race:", int(po.duplicated(["raceId", "positionOrder"]).sum()))
''')

md(r"""
The 2 races with too many podium rows but **no** duplicated pair point to a second kind of shared drive. **Different drivers** share one car, so each driver appears only once,
but both are credited with the same `positionOrder`. Checking for repeated (`raceId`, `positionOrder`) finds these:
""")

code(r'''
shared_pos = (po[po.duplicated(["raceId", "positionOrder"], keep=False)]
              .merge(races[["raceId", "year", "name"]], on="raceId")
              .merge(drivers, on="driverId"))
shared_pos["same driver twice"] = shared_pos.duplicated(["raceId", "driverId"], keep=False)
print(f"Rows sharing a positionOrder with another row in the same race: {len(shared_pos)} "
      f"(years {shared_pos['year'].min()}-{shared_pos['year'].max()})")
print("  of which are the same driver twice (already covered above):", int(shared_pos["same driver twice"].sum()))
print("  of which are different drivers sharing one result:", int((~shared_pos["same driver twice"]).sum()))

example_races = odd.index[~odd.index.isin(list(dup_races))]
display(shared_pos[shared_pos["raceId"].isin(example_races) & (shared_pos["positionOrder"] <= 3)]
        [["year", "name", "driver", "constructorId", "number", "grid", "positionOrder", "points", "laps"]]
        .sort_values(["year", "positionOrder"]))
''')

md(r"""
**What this means.**
- There are **85 duplicated pairs (176 rows, ~0.7% of results)**. Most pairs have 2 rows and a few have 3. They concentrate in **1950–1958**, with a small tail into the 1960s and one case in 1978. There are none after 1978.
- Most are **shared drives**: the same constructor, with two drivers credited for one car. That's why a race can show 4–7 rows with `positionOrder <= 3` and why `positionOrder` repeats within a race: the shared result is credited to every driver of that car.
- A minority are **different constructors**: a driver entered in two cars, or took over a team-mate's car. Some are entries that never started at all (`grid = 0`, e.g. *Did not qualify* / *Withdrew* twice).
- There is also a **second kind of shared drive** that the (raceId, driverId) check can't see: two *different* drivers credited with one car's result (e.g. the 1951 and 1955 Indianapolis 500). These rows aren't duplicates for our grain, since each driver appears once. They do mean a race can legitimately have **more than 3 podium-labelled drivers** even after de-duplication. That matches the official record (both drivers are credited) and only affects 1950-1964.

**Options for which row to keep (not chosen yet):**

| Rule | How | Pros | Cons |
|---|---|---|---|
| A. Best result | keep the row with the lowest `positionOrder` | Target = "did this driver get podium credit at all", matches the official record | The kept row's `grid`/car may be one the driver **didn't start in**. Its pre-race features then describe another car (subtle leakage: the row was chosen *because* of its outcome) |
| B. Starting entry | keep the row the driver actually **started** in. In practice: the lowest `resultId`, as a proxy for the starting car, plus a short list of verified overrides (2.6) | Selection uses only pre-race info, so features and target describe the same car. Consistent with the cutoff | May label a driver "no podium" even though he shared a podium car later in the race. The proxy can be wrong for a few pairs |
| C. Hybrid | features from the starting entry, target = podium in **any** of his rows | Keeps the official credit and pre-race-consistent features | Mixes two rows into one, which is harder to explain and test |
| D. Drop all duplicated pairs | remove these 176 rows | Simplest, no arbitrary choice | Loses data (small), and removes some genuine podiums from 1950s races |

Practical note: these rows are all from 1950-1978, so they never reach a 2003+ model. They do reach an all-years model and the data-engineering questions that use all seasons, and the brief requires a documented rule.
""")

# ---------------------------------------------------------------- 1.7 qualifying coverage
md(r"""
## 1.7 Qualifying coverage by season

For each season: what % of `results` rows have a matching `qualifying` row (same raceId + driverId)?
We also show the % of races with *any* qualifying data (to separate "whole race missing" from "some drivers missing"), and the % of rows with a usable `grid` (> 0) for comparison.
""")

code(r'''
rq = (res[["raceId", "driverId", "grid"]]
      .merge(quali[["raceId", "driverId", "position"]].rename(columns={"position": "quali_pos"}),
             on=["raceId", "driverId"], how="left")
      .merge(races[["raceId", "year"]], on="raceId"))
rq["year"] = rq["year"].astype(int)
rq["has_quali"] = rq["quali_pos"].notna()

coverage = pd.DataFrame({
    "% results rows with qualifying": 100 * rq.groupby("year")["has_quali"].mean(),
    "% races with any qualifying": 100 * rq.groupby(["year", "raceId"])["has_quali"].any().groupby("year").mean(),
    "% rows with grid > 0": 100 * (rq["grid"] != "0").groupby(rq["year"]).mean(),
}).round(1)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(coverage.index, coverage.iloc[:, 0], marker="o", ms=3, label=coverage.columns[0], color="#4C72B0")
ax.plot(coverage.index, coverage.iloc[:, 1], ls="--", label=coverage.columns[1], color="#DD8452")
ax.plot(coverage.index, coverage.iloc[:, 2], ls=":", label=coverage.columns[2], color="#55A868")
ax.axvspan(1993.5, 2002.5, color="grey", alpha=0.12, label="partial-coverage years")
ax.set_ylim(-3, 105)
ax.set_ylabel("%")
ax.set_xlabel("season")
ax.set_title("Qualifying coverage of results rows by season")
ax.legend(loc="center left", fontsize=8)
plt.tight_layout()
plt.show()

display(coverage.loc[1990:2006])
first_full = coverage.index[(coverage.iloc[:, 0] >= 98) & (coverage.index > 2002)].min()
print("First season of continuous ~full coverage:", first_full)
''')

md(r"""
**What this means.**
- **1950–1993: 0% coverage.** The `qualifying` table simply doesn't exist for those seasons.
- **1994–2002: patchy.** 1994–95 are nearly complete, but 1996–2002 only cover some races (whole races missing, the dashed line), dropping as low as ~6% of rows in 2001.
- **2003 onward: ~100%** (tiny gaps are drivers with no qualifying entry).
- Meanwhile `grid` (dotted line) is available in almost every season.

**Why it matters.** Qualifying features (position, gap to pole, Q1/Q2/Q3 times) are only usable from 2003 (or 1994 if we accept gaps). Our options are: (a) restrict the modeling table to ≥ 2003; (b) use all years but rely on `grid` and treat qualifying as missing (with a "has qualifying" flag); (c) impute. A model trained across eras would also learn "qualifying missing ⇒ old era", which is a known limitation the brief explicitly asks us to discuss (qualifying-coverage drift).
""")

# ---------------------------------------------------------------- 1.8 quali vs grid
md(r"""
## 1.8 Qualifying position vs. `grid`

`qualifying.position` = where the driver qualified. `results.grid` = where they actually **started**. They can differ (penalties, pit-lane starts, other drivers' penalties). Compared on rows that have both.
""")

code(r'''
both = rq[rq["has_quali"]].copy()
both["grid_n"] = both["grid"].astype(int)
both["quali_n"] = both["quali_pos"].astype(int)
both["grid_minus_quali"] = both["grid_n"] - both["quali_n"]
started_from_grid = both[both["grid_n"] > 0]

print(f"Rows with both values:         {len(both):,}")
print(f"grid == qualifying position:   {100 * (both['grid_minus_quali'] == 0).mean():.1f}%")
print(f"grid == 0 (among these rows):  {int((both['grid_n'] == 0).sum())}")
print(f"Among grid > 0: started further back {100 * (started_from_grid['grid_minus_quali'] > 0).mean():.1f}%, "
      f"further forward {100 * (started_from_grid['grid_minus_quali'] < 0).mean():.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
diff_by_year = 100 * (both["grid_minus_quali"] != 0).groupby(both["year"]).mean()
axes[0].bar(diff_by_year.index, diff_by_year.values, color="#8172B2")
axes[0].set_title("% of rows where grid != qualifying position")
axes[0].set_xlabel("season")
axes[0].set_ylabel("%")

clipped = started_from_grid["grid_minus_quali"].clip(-10, 10)
axes[1].hist(clipped, bins=np.arange(-10.5, 11.5, 1), color="#64B5CD", edgecolor="white")
axes[1].set_yscale("log")
axes[1].set_title("grid - qualifying position (grid > 0, clipped at +/-10)")
axes[1].set_xlabel("positions (positive = started further back)")
plt.tight_layout()
plt.show()
''')

code(r'''
# What does grid = 0 mean? Look at the race status of those rows, split by whether they have a qualifying row.
g0 = (res[res["grid"] == "0"]
      .merge(RAW["status"], on="statusId")
      .merge(races[["raceId", "year"]], on="raceId")
      .merge(quali[["raceId", "driverId"]], on=["raceId", "driverId"], how="left", indicator=True))
g0["in_qualifying"] = g0.pop("_merge") == "both"
g0["era"] = np.where(g0["year"].astype(int) < 1994, "1950-1993", "1994-2024")

print(f"Total results rows with grid = 0: {len(g0):,}")
display(g0.groupby("era").size().rename("rows").to_frame())
for era, d in g0.groupby("era"):
    display(Markdown(f"**{era}: top statuses for grid = 0**"))
    display(d["status"].value_counts().head(8).to_frame("rows"))
display(Markdown("**grid = 0 rows that DO have a qualifying row: did they race?**"))
display(g0[g0["in_qualifying"]]["status"].value_counts().head(8).to_frame("rows"))
''')

md(r"""
**What this means.**
- On rows with both values, about **3 in 4 match**. The rest differ, and the difference is common in the modern era (often 20–45% of rows per season) because of grid penalties (engine/gearbox changes, impeding, etc.) and their knock-on effect on everyone else. **Qualifying position is not the starting position.**
- **`grid = 0` means "no grid slot"**, and it has two different meanings:
  - **Before ~1994:** mostly *Did not qualify / Did not prequalify / Withdrew*. These drivers **never started the race**, but they still have a `results` row (with a `positionOrder`!).
  - **Modern era (with a qualifying row):** mostly *Finished / +N Laps*. The driver **started from the pit lane** (e.g. after car changes under parc fermé). They raced, just from behind the last slot.
- So `grid = 0` must **never** be read as a number "better than pole".

**Why it matters.**
- For Q1 (front-row conversion) we must decide whether "front row" means **qualified** P1–P2 or **started** P1–P2. The brief asks us to distinguish them.
- For the model, `grid` is valid pre-race info (it's known at lights out) and is likely our strongest feature. It needs recoding: pit-lane start → a value behind the last grid slot plus a flag, not `0`.
- Rows of drivers who **never started** (DNQ/DNPQ/withdrew) arguably aren't "a driver entering a race" in the prediction sense. Keeping them adds easy negatives that inflate scores. This is a decision for the cleaning plan.
""")

# ---------------------------------------------------------------- 1.9 joins
md(r"""
## 1.9 Join map: which tables, which keys

**Modeling table (grain = one row per raceId + driverId)**, built from `results` as the spine (left joins, so no driver-race is dropped):

| Join | Key(s) | What it brings | Unique on the right side? (1.4) |
|---|---|---|---|
| `results` ← `races` | `raceId` | year, round, date, circuitId (chronological ordering) | yes |
| `races` ← `circuits` | `circuitId` | circuit identity, country | yes |
| `results` ← `qualifying` | `raceId`, `driverId` | qualifying position, Q1/Q2/Q3 times | yes |
| `results` ← `drivers` | `driverId` | dob (age), nationality | yes |
| `results` ← `constructors` | `constructorId` | team identity | yes |
| `results` ← `driver_standings` of the **previous race** | `driverId` + previous `raceId` of the same season | points/position/wins *before* this race | yes |
| `results` ← `constructor_standings` of the **previous race** | `constructorId` + previous `raceId` | team form before this race | yes |
| `results` ← `sprint_results` (optional) | `raceId`, `driverId` | sprint finish (sprint runs *before* the GP, 2021+) | yes |
| past `results` (self-join on earlier races) | `driverId` / `constructorId`, date < race date | rolling form: recent podium rate, average finish | built by us |

**Data-engineering questions**

| Question | Tables | Keys |
|---|---|---|
| Q1 Front-row → podium by circuit, pre/post 2014 | `results` + `qualifying` + `races` + `circuits` | `raceId`+`driverId` (quali), `raceId` (races), `circuitId` (circuits) |
| Q2 Home-country podium effect, controlling for grid | `results` + `races` + `circuits` + `drivers` (+ a nationality→country mapping we create) | `raceId`, `circuitId`, `driverId` |
| Q3 Constructor mechanical-retirement rates 2014–21 vs 2022–24 | `results` + `status` + `races` + `constructors` | `statusId`, `raceId` (year), `constructorId` |

For Q3 we'll need to classify `status` values into mechanical / accident / driver / regulation / finished. Here is what actually occurs in 2014–2024:
""")

code(r'''
status_recent = (res.merge(races[["raceId", "year"]], on="raceId")
                 .loc[lambda d: d["year"].astype(int) >= 2014]
                 .merge(RAW["status"], on="statusId")["status"].value_counts())
print(f"{len(status_recent)} distinct status values in 2014-2024:")
display(status_recent.to_frame("rows").T)
''')

md(r"""
**What this means.** The modeling table only needs **one-to-one left joins on keys we've verified are unique**, so the grain can't blow up. The time-dependent tables (standings, past results) must be joined to the **previous** race, not the current one (see 1.10).
The status list shows many specific failure causes (`Engine`, `Gearbox`, `Power Unit`, `Hydraulics`, ...) mixed with crash (`Accident`, `Collision`), driver and regulation (`Disqualified`) labels. Q3's answer depends on an explicit, documented mapping.
""")

# ---------------------------------------------------------------- 1.10 leakage
md(r"""
## 1.10 Leakage and irrelevant columns

The test for every candidate feature is: **would we know this value after qualifying, before lights out?**
""")

code(r'''
leakage = pd.DataFrame([
    ("results", "position, positionText, positionOrder", "LEAK (this is the target)", "finishing result of the race we predict"),
    ("results", "points", "LEAK", "awarded after the race"),
    ("results", "laps", "LEAK", "laps completed: known only at the end (low laps = DNF)"),
    ("results", "statusId / status", "LEAK", "Finished / Engine / Accident: outcome of the race"),
    ("results", "time, milliseconds", "LEAK", "race duration / gap to winner"),
    ("results", "fastestLap, rank, fastestLapTime, fastestLapSpeed", "LEAK", "measured during the race"),
    ("lap_times", "all (current race)", "LEAK", "recorded during the race"),
    ("pit_stops", "all (current race)", "LEAK", "recorded during the race"),
    ("driver_standings", "points, position, wins @ current raceId", "LEAK", "standings AFTER this race -> must use the previous race's row"),
    ("constructor_standings", "points, position, wins @ current raceId", "LEAK", "same: use previous race"),
    ("constructor_results", "points, status @ current raceId", "LEAK", "constructor points from this race"),
    ("results", "grid", "OK", "starting slot, known at lights out (needs grid=0 recoding)"),
    ("qualifying", "position, q1, q2, q3", "OK", "qualifying happens before the race"),
    ("sprint_results", "position, grid, points (2021+)", "OK with care", "sprint is held before the GP; only exists for ~18 weekends"),
    ("races", "year, round, circuitId, date", "OK", "schedule is known in advance"),
    ("drivers / constructors / circuits", "ids, dob, nationality, country", "OK", "static attributes"),
    ("past results / standings", "rolling form over EARLIER races", "OK if strictly lagged", "must exclude the current race (shift by one race)"),
    ("all", "url, circuitRef, driverRef, constructorRef, name strings", "IRRELEVANT", "identifiers/links, no predictive signal beyond the Id"),
    ("races", "fp1-3 / quali / sprint date & time", "IRRELEVANT", ">90% missing, schedule metadata only"),
    ("drivers", "number, code", "IRRELEVANT", "mostly missing, just labels"),
    ("circuits", "lat, lng, alt", "LOW VALUE", "static location, circuit Id already captures it"),
    ("seasons", "url", "IRRELEVANT", "only a link"),
], columns=["table", "columns", "verdict", "reason"])
display(leakage)
''')

md(r"""
**Why it matters.** Leakage is the biggest risk in this project: `laps`, `status` or same-race `points` would give near-perfect scores that are useless in practice.
The subtle one is **standings**. `driver_standings` for a `raceId` includes the points *from that race*, so it looks like "season context" but contains the answer. Any form/standings feature must come from **strictly earlier races**.
""")

# ---------------------------------------------------------------- 1.11 plan
md(r"""
## 1.11 Cleaning plan (approved, applied in section 2)

Ordered by priority: grain and leakage first (they decide correctness), then typing, then convenience. Every step works on **copies**; the raw files stay untouched.

| # | Step | Evidence | Decision you need to make |
|---|---|---|---|
| 1 | **Fix the grain**: one row per (`raceId`, `driverId`) in `results` | 1.4, 1.6: 85 duplicated pairs, races with > 3 "podiums" | Choose rule A/B/C/D (1.6), and whether to accept > 3 podium drivers per race from cross-driver shared drives (1950-1964) |
| 2 | **Define who is "entering a race"**: decide whether to drop rows that never started (grid = 0 with DNQ / DNPQ / Withdrew) | 1.8 | Keep or drop non-starters |
| 3 | **Sentinels → missing**: replace `\N` **and** empty/whitespace strings with `NaN` everywhere; strip padded text (`'Argentinian '`) | 1.2 | – |
| 4 | **Numeric typing**: IDs/counts → `Int64` (nullable), points/speeds → float | 1.3.1 | – |
| 5 | **Semantic typing**: dates → `datetime64`; q1/q2/q3, lap times, pit durations, fastest laps → float seconds; race date + time → timestamp; **don't parse `results.time`** (use `milliseconds`) | 1.3 + ms cross-check | – |
| 6 | **Recode `grid = 0`**: pit-lane start → a value behind the last grid slot + `pit_lane_start` flag; never keep `0` as a number | 1.8 | Exact recoding value |
| 7 | **Choose the modeling window**: 2003+ (full qualifying), 1994+ with gaps, or all years with missing-qualifying handling | 1.7 | Window |
| 8 | **Leakage guard**: drop or quarantine every post-race column (1.10); build standings and form features from the **previous** race only | 1.10 | – |
| 9 | **Label harmonisation for Q2**: nationality → country mapping (`British`→`UK`, `American`→`USA`/`United States`, ...), unify `USA`/`United States` | 1.2.2 | Treatment of dual nationalities (`American-Italian`, `East German`) |
| 10 | **Status taxonomy for Q3**: map each status to mechanical / accident / driver / regulation / finished | 1.9 | Borderline labels (`Retired`, `Puncture`, `Damage`, `Debris`) |
| 11 | **Re-run the audit checks on the cleaned copies** (PK unique, FK reconcile, no `\N`, types correct) plus a before/after EDA, as the brief requires | – | – |
""")

# ---------------------------------------------------------------- 1.12 anomaly export
md(r"""
## 1.12 Export: duplicates and anomalies per table

Every row-level problem found above is collected and written to **one CSV per raw table** in `outputs/anomalies/`, so it can be inspected outside the notebook (Excel, a reviewer, the report).
These are new files; the raw CSVs are not touched (verified again in 1.13).

Each output row is **one raw row + one anomaly** (a row with two problems appears twice):
- `csv_line`: line number in the original CSV (header = line 1), to find the row in the raw file.
- `anomaly_type` / `anomaly_detail`: what is wrong and why it was flagged.
- then all original columns, exactly as written in the raw file.

`_summary.csv` lists the count per table and anomaly type, including tables with nothing flagged.

Checks applied (all derived from sections 1.2–1.8):

| Anomaly type | Tables | Source |
|---|---|---|
| duplicate key | any table whose natural key repeats | 1.4 / 1.6 |
| shared result (different drivers) | results | 1.6 |
| race has > 3 podium rows | results | 1.6 |
| grid = 0 (no grid slot) | results, sprint_results | 1.2.1 / 1.8 |
| constructor differs from results / qualifying | results, qualifying | 1.5 |
| empty-string sentinel (not `\N`) | any | 1.2 |
| padded text | any | 1.2 |
| label synonym | circuits.country, drivers.nationality | 1.2.2 |
| duration outlier (pit stop ≥ 60 s, lap ≥ 10 min, qualifying lap ≥ 5 min) | pit_stops, lap_times, qualifying | 1.3 |
| unusual status value | constructor_results | 1.2 |
| non-numeric positionText | driver/constructor standings | sanity check |
| missing car number | results | 1.2 |
""")

code(r'''
ANOMALY_DIR = Path("outputs/anomalies")
ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
for old in ANOMALY_DIR.glob("*.csv"):  # only our own generated reports live here
    old.unlink()

found = {t: [] for t in TABLES}


def flag(table, mask, kind, detail):
    """Record the raw rows selected by `mask` under one anomaly type."""
    rows = RAW[table].loc[mask].copy()
    if rows.empty:
        return
    if isinstance(detail, pd.Series):
        detail = detail.loc[rows.index]
    rows.insert(0, "anomaly_detail", detail)
    rows.insert(0, "anomaly_type", kind)
    rows.insert(0, "csv_line", rows.index + 2)  # header is line 1
    found[table].append(rows)


status_name = RAW["results"]["statusId"].map(dict(zip(RAW["status"]["statusId"], RAW["status"]["status"])))

# --- generic checks on every table --------------------------------------------------------
for t, df in RAW.items():
    for col in df.columns:
        s, stripped = df[col], df[col].str.strip()
        flag(t, stripped == "", "empty-string sentinel", f"{col} is empty instead of \\N")
        flag(t, (s != stripped) & (stripped != ""), "padded text", f"{col} has leading/trailing spaces: '" + s + "'")

for t, key, kind in KEYS:
    d = RAW[t]
    n = d.groupby(key)[key[0]].transform("size")
    flag(t, n > 1, "duplicate key", f"{' + '.join(key)} shared by " + n.astype(str) + " rows")

# --- results / sprint_results ---------------------------------------------------------------
r = RAW["results"]
same_driver_twice = r.duplicated(["raceId", "driverId"], keep=False)
shared_pos = r.duplicated(["raceId", "positionOrder"], keep=False) & ~same_driver_twice
flag("results", shared_pos, "shared result (different drivers)",
     "positionOrder " + r["positionOrder"] + " also credited to another driver in this race")

po_int = r["positionOrder"].astype(int)
podiums_per_race = (po_int <= 3).groupby(r["raceId"]).transform("sum")
flag("results", (po_int <= 3) & (podiums_per_race > 3), "race has > 3 podium rows",
     "race has " + podiums_per_race.astype(str) + " rows with positionOrder <= 3")

flag("results", r["grid"] == "0", "grid = 0 (no grid slot)", "status: " + status_name)
flag("sprint_results", RAW["sprint_results"]["grid"] == "0", "grid = 0 (no grid slot)", "pit-lane / no-grid sprint start")
flag("results", r["number"] == SENTINEL, "missing car number", "number is \\N")

# results vs qualifying: same driver, same race, different constructor (flag the row in both tables)
pair = (r.reset_index().merge(RAW["qualifying"].reset_index(), on=["raceId", "driverId"], suffixes=("_r", "_q"))
        .loc[lambda d: d["constructorId_r"] != d["constructorId_q"]])
flag("results", r.index.isin(pair["index_r"]), "constructor differs from qualifying",
     pd.Series("qualifying constructorId " + pair["constructorId_q"].values, index=pair["index_r"].values))
flag("qualifying", RAW["qualifying"].index.isin(pair["index_q"]), "constructor differs from results",
     pd.Series("results constructorId " + pair["constructorId_r"].values, index=pair["index_q"].values))

# --- duration outliers ----------------------------------------------------------------------
pit_s = RAW["pit_stops"]["duration"].map(duration_to_seconds)
flag("pit_stops", pit_s >= 60, "duration outlier", "pit stop of " + pit_s.round(1).astype(str) + " s (likely red flag / repair)")
lap_s = pd.to_numeric(RAW["lap_times"]["milliseconds"]) / 1000
flag("lap_times", lap_s >= 600, "duration outlier", "lap of " + (lap_s / 60).round(1).astype(str) + " min (likely red flag / safety car)")
for q in ["q1", "q2", "q3"]:
    q_s = RAW["qualifying"][q].map(duration_to_seconds)
    flag("qualifying", q_s >= 300, "duration outlier", f"{q} lap of " + q_s.round(1).astype(str) + " s")

# --- categorical oddities --------------------------------------------------------------------
cr = RAW["constructor_results"]
flag("constructor_results", cr["status"] != SENTINEL, "unusual status value",
     "status = '" + cr["status"] + "' (all other rows are \\N)")
for t in ["driver_standings", "constructor_standings"]:
    pt = RAW[t]["positionText"]
    flag(t, ~pt.str.isdigit(), "non-numeric positionText", "positionText = '" + pt + "'")

LABEL_SYNONYMS = {("circuits", "country"): [{"USA", "United States"}],
                  ("drivers", "nationality"): [{"Argentine", "Argentinian"}]}
for (t, col), groups in LABEL_SYNONYMS.items():
    values = RAW[t][col].str.strip()
    for group in groups:
        flag(t, values.isin(group), "label synonym", f"{col} '" + values + "' has a synonym in " + str(sorted(group)))

# --- write one CSV per table + a summary ------------------------------------------------------
summary_rows = []
for t in TABLES:
    if found[t]:
        out = pd.concat(found[t]).sort_values(["csv_line", "anomaly_type"])
        out.to_csv(ANOMALY_DIR / f"{t}_anomalies.csv", index=False)
        summary_rows += [{"table": t, "anomaly_type": k, "rows": v}
                         for k, v in out["anomaly_type"].value_counts().items()]
    else:
        summary_rows.append({"table": t, "anomaly_type": "(none found)", "rows": 0})

anomaly_summary = pd.DataFrame(summary_rows)
anomaly_summary.to_csv(ANOMALY_DIR / "_summary.csv", index=False)

print("Written to", ANOMALY_DIR.resolve())
for f in sorted(ANOMALY_DIR.glob("*.csv")):
    print(f"  {f.name:40s} {sum(1 for _ in open(f, encoding='utf-8')) - 1:>6,} rows")
display(anomaly_summary)
''')

md(r"""
**What this gives you.** Each file is a concrete, row-level list of what the cleaning step (1.11) has to deal with, with the exact line in the raw CSV. For example:
- `results_anomalies.csv` holds the 176 duplicate rows (1.6), the cross-driver shared drives, the races with too many podium rows and all `grid = 0` rows, each with its race status.
- `qualifying_anomalies.csv` holds the empty-string sentinels and the constructor mismatches.

Some of these are **real data quirks to fix** (duplicates, empty strings, padded text, synonyms). Others are **legitimate but unusual values to handle deliberately** (long pit stops under red flags, pit-lane starts). The anomaly type says which is which, and the reports make it easy to show the evaluator exactly which rows each cleaning rule touches.
""")

# ---------------------------------------------------------------- 1.13 immutability
md(r"""
## 1.13 Raw-data immutability check
""")

code(r'''
after = {t: hashlib.sha256(read_raw_bytes(t)).hexdigest() for t in TABLES}
unchanged = all(after[t] == RAW_FINGERPRINT[t] for t in TABLES)
assert unchanged, "A raw file changed during the audit!"
print(f"All {len(TABLES)} raw files are byte-identical to when they were loaded (SHA-256).")
''')

# ================================================================ 2. CLEANING
md(r"""
# 2. Data Cleaning

Everything here works on **copies** (`CLEAN`) of the raw tables. `RAW` and the files on disk are never changed (re-verified in 2.12).

Decisions taken (evidence in section 1):

| Decision | Choice | Why |
|---|---|---|
| Duplicate (raceId, driverId) | **Rule B**: keep the row with the **lowest `resultId`**, as a proxy for the car the driver started in, plus 3 verified overrides where the proxy is wrong | the row is chosen without looking at the result, so features and target describe the same car (1.6, 2.6) |
| Non-starters | **removed**: non-start status (DNQ, DNPQ, Withdrew, 107%) with 0 laps, or no grid slot and 0 laps | they never took the start, so they aren't a "driver entering the race" and would be easy fake negatives (1.8) |
| Indianapolis 500 (1950–1960) | **removed** from the cleaned results (kept aside only for a Q2 robustness check) | a separate American race inside the championship: its own drivers and cars, a 33-car oval field; it distorts Q2 and an all-years model (2.6.1) |
| `grid = 0` (pit-lane start) | that race's last grid slot + 1, plus a `pit_lane_start` flag | 0 would look *better* than pole (1.8) |
| Modeling window | `model_base` = **2003+**; every pre-2003 row is cleaned too (`results_full`) | qualifying coverage is ~100% only from 2003 (1.7). A later all-years model version (missing qualifying left as NaN + a flag) needs the older rows clean as well |
| Standings | use the standings **after the previous round**, never the current race | same-race standings already contain the result (1.10, proven in 2.10.2) |
| Unused / deferred tables | `seasons`, `lap_times`, `pit_stops`, `constructor_results` not used; `sprint_results` cleaned but deferred | one-line reason per table in 2.1 |
""")

# ---------------------------------------------------------------- 2.1
md(r"""
## 2.1 Working copies, sentinels and text
""")

code(r'''
# Tables not carried into CLEAN at all.
UNUSED_TABLES = {
    "seasons": "not used: it only lists the years already in races.year (plus a Wikipedia url)",
    "lap_times": "not used as features: recorded during the race (post-race information, 1.10)",
    "pit_stops": "not used as features: recorded during the race (post-race information, 1.10)",
}
# Tables cleaned (so their keys are re-validated in 2.10) but not used by the questions or the current models.
DEFERRED_TABLES = {
    "constructor_results": "not used: per-race team points are already in results and constructor_standings; "
                           "its status column is 99.9% empty",
    "sprint_results": "cleaned but not used in the current models; to be tested later as an optional feature "
                      "(sprint finish + has_sprint flag)",
}
CLEAN = {t: RAW[t].copy() for t in TABLES if t not in UNUSED_TABLES}
cleaning_log = []


def log_step(step, table, before, after, note=""):
    cleaning_log.append({"step": step, "table": table, "rows before": before, "rows after": after,
                         "rows removed": before - after, "note": note})


counts = {"\\N -> NaN": 0, "empty / whitespace-only -> NaN": 0, "leading/trailing spaces stripped": 0}
for t, df in CLEAN.items():
    stripped = df.apply(lambda s: s.str.strip())
    counts["\\N -> NaN"] += int((stripped == SENTINEL).sum().sum())
    counts["empty / whitespace-only -> NaN"] += int((stripped == "").sum().sum())
    counts["leading/trailing spaces stripped"] += int(((df != stripped) & (stripped != "")).sum().sum())
    CLEAN[t] = stripped.mask(stripped.isin([SENTINEL, ""]))  # both placeholder spellings become real missing values

display(pd.DataFrame([(t, "not carried", why) for t, why in UNUSED_TABLES.items()]
                     + [(t, "cleaned, not used", why) for t, why in DEFERRED_TABLES.items()],
                     columns=["table", "status", "reason"]))
display(pd.Series(counts, name="cells").to_frame())
''')

code(r'''
# Encoding check: is there any mis-decoded text ("web-scraping" mojibake) to repair?
MOJIBAKE = ["Ã", "Â", "â€", "�"]
hits = {f"{t}.{c}": int(df[c].dropna().str.contains("|".join(MOJIBAKE)).sum())
        for t, df in CLEAN.items() for c in df.select_dtypes("object")}
print("Cells with mojibake markers:", sum(hits.values()))
accented = CLEAN["drivers"].loc[~CLEAN["drivers"]["surname"].map(str.isascii), ["forename", "surname"]].head(5)
print("Accented names decode correctly, e.g.:")
display(accented)
''')

md(r"""
**What this did.** `\N` **and** empty strings are now both real missing values (`NaN`), so later steps can't confuse a placeholder with data. Stray spaces are gone (`'Argentinian '` → `'Argentinian'`).
**Unused and deferred tables (one line each, see the table above):**
- `seasons`: not used; it only lists the years already in `races.year`.
- `lap_times`, `pit_stops`: not used as features; they are recorded during the race.
- `constructor_results`: not used; its per-race team points are already in `results` and `constructor_standings`, and its `status` column is 99.9% empty. It is still cleaned so its keys are checked in 2.10.
- `sprint_results`: cleaned but not used in the current models; to be tested later as an optional feature (sprint finish + `has_sprint` flag).

**Encoding:** there is nothing to fix. No cell contains mis-decoded characters, and accented names (e.g. *Räikkönen*) read correctly because the files are valid UTF-8. We document the check instead of "repairing" text that isn't broken.
""")

# ---------------------------------------------------------------- 2.2
md(r"""
## 2.2 Columns removed
""")

code(r'''
DROP_COLUMNS = [
    ("circuits", ["circuitRef", "lat", "lng", "alt", "url"], "identifier / link / static geography: circuitId + country carry this"),
    ("constructors", ["url"], "link only"),
    ("drivers", ["number", "code", "url"], "number/code 88-93% missing and only labels; url is a link"),
    ("races", ["url", "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
               "quali_date", "quali_time", "sprint_date", "sprint_time"], "session schedule >90% missing; link"),
    ("qualifying", ["constructorId"], "10 rows disagree with results; results is the source of truth for the team (1.5)"),
    ("results", ["position", "positionText", "time"], "position/positionText repeat positionOrder; time mixes total time and gaps (1.3)"),
    ("sprint_results", ["position", "positionText", "time"], "same as results"),
    ("driver_standings", ["positionText"], "text copy of position, with letter codes ('D')"),
    ("constructor_standings", ["positionText"], "text copy of position, with letter codes ('E')"),
    ("constructor_results", ["status"], "99.9% missing ('D' in 17 rows)"),
]
for t, cols, _ in DROP_COLUMNS:
    CLEAN[t] = CLEAN[t].drop(columns=cols)

display(pd.DataFrame([(t, ", ".join(c), why) for t, c, why in DROP_COLUMNS],
                     columns=["table", "dropped columns", "reason"]))
''')

md(r"""
**Why these and not more.** Each dropped column is either a link/identifier, mostly empty, a duplicate of another column, or inconsistent with the table that owns the information.
Post-race columns of `results` (`points`, `laps`, `statusId`, ...) are **not** dropped here: the data-engineering questions need them (Q3 uses status). They are kept out of the **modeling** table instead (2.9).
""")

# ---------------------------------------------------------------- 2.3
md(r"""
## 2.3 Types: numbers, dates, durations
""")

code(r'''
def to_numeric_if_possible(s):
    """Text column whose non-missing values are all numbers -> Int64 (nullable) or float64."""
    if s.dtype != object or s.dropna().empty:
        return s
    num = pd.to_numeric(s.dropna(), errors="coerce")
    if num.isna().any():
        return s
    full = pd.to_numeric(s)
    return full.astype("Int64") if (num == np.floor(num)).all() else full.astype("float64")


for t in CLEAN:
    CLEAN[t] = CLEAN[t].apply(to_numeric_if_possible)

# Points are always float, even where every value happens to be whole (sprint points: 8, 7, ... 1).
CLEAN["sprint_results"]["points"] = CLEAN["sprint_results"]["points"].astype("float64")

# Dates (and race start as a timestamp where the start time is recorded)
races_c = CLEAN["races"]
races_c["start_utc"] = pd.to_datetime(races_c["date"] + " " + races_c["time"], format="%Y-%m-%d %H:%M:%S")
races_c["date"] = pd.to_datetime(races_c["date"], format="%Y-%m-%d")
CLEAN["races"] = races_c.drop(columns="time")
CLEAN["drivers"]["dob"] = pd.to_datetime(CLEAN["drivers"]["dob"], format="%Y-%m-%d")

# Lap-time strings -> seconds (parser validated against milliseconds in 1.3)
DURATIONS = {"qualifying": ["q1", "q2", "q3"], "results": ["fastestLapTime"], "sprint_results": ["fastestLapTime"]}
for t, cols in DURATIONS.items():
    for c in cols:
        CLEAN[t][f"{c}_sec"] = CLEAN[t][c].map(duration_to_seconds)
    CLEAN[t] = CLEAN[t].drop(columns=cols)

dtype_summary = pd.DataFrame([{"table": t, "dtype": str(d), "column": c}
                              for t, df in CLEAN.items() for c, d in df.dtypes.items()])
display(dtype_summary.groupby(["table", "dtype"])["column"].apply(", ".join).unstack(fill_value=""))
''')

md(r"""
**What this did.** Every column that only holds numbers is now numeric. Columns with gaps use the **nullable** `Int64` type, so a missing value stays missing instead of becoming 0 or forcing floats.
Dates are real dates (needed to order races in time), and lap times are seconds (`q1_sec`, ...), so they can be compared and averaged.
The only text columns left are genuine labels (names, nationality, country, status).
`sprint_results.points` holds only whole numbers, so the automatic typing made it an integer. It is set to float like `results.points`, so the two can be added or compared without a type mismatch.
""")

# ---------------------------------------------------------------- 2.4
md(r"""
## 2.4 Labels: countries and nationalities (for Q2)
""")

code(r'''
CLEAN["circuits"]["country"] = CLEAN["circuits"]["country"].replace({"United States": "USA"})
CLEAN["drivers"]["nationality"] = CLEAN["drivers"]["nationality"].replace({"Argentinian": "Argentine"})

# Demonym -> the country spelling used in circuits.country.
# Dual nationalities take the first one listed; East German -> Germany (circuits are listed under Germany).
NATIONALITY_TO_COUNTRY = {
    "American": "USA", "American-Italian": "USA", "Argentine": "Argentina", "Argentine-Italian": "Argentina",
    "Australian": "Australia", "Austrian": "Austria", "Belgian": "Belgium", "Brazilian": "Brazil",
    "British": "UK", "Canadian": "Canada", "Chilean": "Chile", "Chinese": "China", "Colombian": "Colombia",
    "Czech": "Czech Republic", "Danish": "Denmark", "Dutch": "Netherlands", "East German": "Germany",
    "Finnish": "Finland", "French": "France", "German": "Germany", "Hungarian": "Hungary", "Indian": "India",
    "Indonesian": "Indonesia", "Irish": "Ireland", "Italian": "Italy", "Japanese": "Japan",
    "Liechtensteiner": "Liechtenstein", "Malaysian": "Malaysia", "Mexican": "Mexico", "Monegasque": "Monaco",
    "New Zealander": "New Zealand", "Polish": "Poland", "Portuguese": "Portugal", "Rhodesian": "Rhodesia",
    "Russian": "Russia", "South African": "South Africa", "Spanish": "Spain", "Swedish": "Sweden",
    "Swiss": "Switzerland", "Thai": "Thailand", "Uruguayan": "Uruguay", "Venezuelan": "Venezuela",
}
CLEAN["drivers"]["home_country"] = CLEAN["drivers"]["nationality"].map(NATIONALITY_TO_COUNTRY)
unmapped = sorted(CLEAN["drivers"].loc[CLEAN["drivers"]["home_country"].isna(), "nationality"].unique())
assert not unmapped, f"unmapped nationalities: {unmapped}"

circuit_countries = set(CLEAN["circuits"]["country"])
home_countries = set(CLEAN["drivers"]["home_country"])
print("Circuit countries:", len(circuit_countries))
print("Circuit countries with at least one home driver:", len(circuit_countries & home_countries))
print("Circuit countries with no driver from there:", sorted(circuit_countries - home_countries))
''')

md(r"""
**What this did.** Before, **no** nationality string matched a circuit country (1.2.2). Now every driver has a `home_country` in the same vocabulary as `circuits.country`, so Q2 becomes a simple equality test (`home_country == circuit country`).
Choices to defend: dual nationalities use the first one listed (the nationality the driver raced under), and *East German* maps to *Germany*.
""")

# ---------------------------------------------------------------- 2.5
md(r"""
## 2.5 Status taxonomy (for Q3 and the non-starter rule)

Every status is mapped to exactly one of **six categories**: finished, mechanical, accident, regulation, driver, unclear. The brief defines a **mechanical retirement** as a car component failing on its own, excluding crashes, the driver, and regulation decisions.
Statuses whose cause can't be told from the label go to **unclear** instead of being guessed.

The **non-starter rule** in 2.6 doesn't use these categories. It has its own list of statuses that mean "never took the start" (`NON_START_STATUSES`).
""")

code(r'''
import re

# Statuses that mean the car never took the start (used by the non-starter rule in 2.6, together with laps == 0).
NON_START_STATUSES = {"Did not qualify", "Did not prequalify", "Withdrew", "107% Rule"}

STATUS_GROUPS = {
    # failed to qualify, or excluded / disqualified by the rules
    "regulation": {"Disqualified", "Excluded", "Underweight", "Not classified",
                   "Did not qualify", "Did not prequalify", "107% Rule"},
    "accident": {"Accident", "Collision", "Collision damage", "Spun off", "Fatal accident"},
    "driver": {"Injury", "Injured", "Illness", "Driver unwell", "Physical", "Eye injury"},
    # cause can't be told from the label: often contact or a pit error, a symptom rather than a part, or no cause given
    "unclear": {"Retired", "Withdrew", "Out of fuel", "Damage", "Debris", "Puncture", "Tyre puncture", "Tyre",
                "Not restarted", "Stalled", "Safety concerns", "Safety", "Fuel rig", "Refuelling",
                "Wheel", "Wheel nut", "Front wing", "Rear wing", "Broken wing", "Undertray", "Vibrations"},
    # a part of the car that fails on its own
    "mechanical": {
        "Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics", "Electrical", "Radiator", "Suspension", "Brakes",
        "Differential", "Overheating", "Mechanical", "Driver Seat", "Driveshaft", "Fuel pressure", "Water pressure",
        "Throttle", "Steering", "Technical", "Electronics", "Heat shield fire", "Exhaust", "Oil leak",
        "Wheel rim", "Water leak", "Fuel pump", "Track rod", "Oil pressure", "Engine fire", "Engine misfire",
        "Pneumatics", "Handling", "Fire", "Wheel bearing", "Fuel system", "Oil line", "Launch control",
        "Fuel", "Power loss", "Drivetrain", "Ignition", "Chassis", "Battery", "Halfshaft", "Crankshaft",
        "Alternator", "Safety belt", "Oil pump", "Fuel leak", "Injection", "Distributor", "Turbo", "CV joint",
        "Water pump", "Spark plugs", "Fuel pipe", "Oil pipe", "Axle", "Water pipe", "Magneto", "Supercharger",
        "Power Unit", "ERS", "Brake duct", "Seat", "Cooling system",
    },
}
STATUS_CATEGORIES = ["finished", "mechanical", "accident", "regulation", "driver", "unclear"]
all_listed = [s for names in STATUS_GROUPS.values() for s in names]
assert len(all_listed) == len(set(all_listed)), "a status is listed in two categories"


def status_category(name):
    if name == "Finished" or re.fullmatch(r"\+\d+ Laps?", name):
        return "finished"
    return next((cat for cat, names in STATUS_GROUPS.items() if name in names), None)


CLEAN["status"]["status_category"] = CLEAN["status"]["status"].map(status_category)
unmapped = CLEAN["status"].loc[CLEAN["status"]["status_category"].isna(), "status"].tolist()
assert not unmapped, f"unmapped statuses: {unmapped}"
assert set(CLEAN["status"]["status_category"]) == set(STATUS_CATEGORIES)

display(CLEAN["status"].groupby("status_category")["status"]
        .agg(n_statuses="size", statuses=lambda s: ", ".join(sorted(s)[:12]) + (" ..." if len(s) > 12 else ""))
        .loc[STATUS_CATEGORIES])
''')

md(r"""
**What this did.** All 139 statuses have exactly one of the six categories, and the code asserts that none is missed or listed twice. `+N Laps` counts as *finished* (the car was running at the end, just lapped).

**What changed and why (after review):** the assignment is now consistent.
- **mechanical** holds only parts that fail on their own (engine, gearbox, hydraulics, brakes, suspension, electrical, ...), plus `Technical` and `Seat`.
- **unclear** now also holds `Wheel`, `Rear wing`, `Undertray` and `Vibrations`. They join `Wheel nut` and `Front wing`, which were already there: these are often caused by contact or a pit-stop error, or are a symptom rather than a failed part. `Retired`, `Out of fuel` and `Withdrew` (during the race) give no cause, so they are *unclear* too.
- The old extra group *did not start* is gone. `Did not qualify`, `Did not prequalify` and `107% Rule` are **regulation** outcomes; `Withdrew` is *unclear*. The non-starter rule keeps its own status list, so it is unaffected.

**Note for Q3:** the constructor ranking is reported twice, **strict** (mechanical only) and **broad** (mechanical + unclear). A team that ranks high in both is unreliable whichever way the borderline statuses are counted.
""")

# ---------------------------------------------------------------- 2.6
md(r"""
## 2.6 `results`: target, non-starters, duplicates, grid

Order matters. The **target is created first** from the official classification, before any row is removed. Then come non-starters, then duplicates, then the grid recoding.
""")

code(r'''
res_c = CLEAN["results"].merge(CLEAN["status"], on="statusId", how="left", validate="many_to_one")
res_c["podium"] = (res_c["positionOrder"] <= 3).astype("int8")   # target: positionOrder <= 3

# --- Non-starters --------------------------------------------------------------------------------
by_status = res_c["status"].isin(NON_START_STATUSES) & (res_c["laps"] == 0)
no_slot = (res_c["grid"] == 0) & (res_c["laps"] == 0)
non_starter = by_status | no_slot

reason = np.select([by_status, no_slot], ["non-start status, 0 laps", "no grid slot, 0 laps (other status)"], "kept")
display(pd.crosstab(reason, np.where(res_c["grid"] == 0, "grid = 0", "grid > 0"), margins=True))

print("Why the rule uses laps = 0 and not grid = 0: drivers who withdrew while holding a grid slot, e.g. the 2005 US GP:")
usgp05 = CLEAN["races"].query("year == 2005 and name == 'United States Grand Prix'")["raceId"]
display(res_c[res_c["raceId"].isin(usgp05) & by_status]
        .merge(CLEAN["drivers"][["driverId", "surname"]], on="driverId")[["surname", "grid", "laps", "status"]].head(6))
print("Pit-lane starters kept (grid = 0, laps > 0):", int(((res_c["grid"] == 0) & (res_c["laps"] > 0)).sum()))

n = len(res_c)
res_c = res_c[~non_starter].copy()
log_step("remove non-starters", "results", n, len(res_c), "non-start status with 0 laps, or grid = 0 with 0 laps")
''')

code(r'''
# --- Duplicates: rule B ----------------------------------------------------------------------------
# Keep the row with the lowest resultId, as a proxy for the car the driver started in.
# Verified overrides: pairs where the proxy is known to keep the car the driver took over mid-race.
# In each case the proxy's row is an exact copy (car, grid, laps, finish) of another driver's only row,
# i.e. that car was the other driver's start.
RULE_B_OVERRIDES = pd.DataFrame([
    (1951, "French Grand Prix", "Ascari", 19935,
     "started car 12 from grid 3 (gearbox, lap 10), then took over González's car 14; the car-14 row copies González's row 20187"),
    (1954, "Belgian Grand Prix", "González", 19276,
     "started car 6 from grid 2 (engine, lap 1), then took over Hawthorn's car 10; the car-10 row copies Hawthorn's row 20238"),
    (1961, "British Grand Prix", "Moss", 17964,
     "started car 28 from grid 5 (brakes, lap 44), then took over Fairman's car 26; the car-26 row copies Fairman's row 20299"),
], columns=["year", "race", "driver", "keep resultId", "why"])

n = len(res_c)
dups = res_c[res_c.duplicated(["raceId", "driverId"], keep=False)]
dup_pairs = int(res_c.duplicated(["raceId", "driverId"]).sum())
both_started = dups.groupby(["raceId", "driverId"])["grid"].apply(lambda g: (g > 0).all())
print(f"Duplicated pairs after removing non-starters: {len(both_started)}; "
      f"pairs where both rows have grid > 0: {int(both_started.sum())} -> grid can't tell the rows apart")

# Each override must be one of the duplicated rows, for the named driver and race.
ov = (RULE_B_OVERRIDES.merge(dups[["resultId", "raceId", "driverId"]], left_on="keep resultId", right_on="resultId", how="left")
      .merge(CLEAN["races"][["raceId", "year", "name"]], on="raceId", how="left", suffixes=("", "_race"))
      .merge(CLEAN["drivers"][["driverId", "surname"]], on="driverId", how="left"))
assert ov["resultId"].notna().all() and (ov["year"] == ov["year_race"]).all() and (ov["race"] == ov["name"]).all() \
    and (ov["driver"] == ov["surname"]).all(), "an override does not match a duplicated pair"

proxy_choice = dups.sort_values("resultId").drop_duplicates(["raceId", "driverId"]).set_index(["raceId", "driverId"])["resultId"]
RULE_B_OVERRIDES["proxy would keep"] = [proxy_choice.loc[(r, d)] for r, d in zip(ov["raceId"], ov["driverId"])]
display(RULE_B_OVERRIDES[["year", "race", "driver", "proxy would keep", "keep resultId", "why"]])

podium_any = res_c.groupby(["raceId", "driverId"])["podium"].transform("max")
res_c = (res_c.assign(_not_override=~res_c["resultId"].isin(RULE_B_OVERRIDES["keep resultId"]))
         .sort_values(["raceId", "driverId", "_not_override", "resultId"])   # override first, then lowest resultId
         .drop_duplicates(["raceId", "driverId"], keep="first")
         .drop(columns="_not_override"))
lost = res_c[res_c["podium"] < podium_any.loc[res_c.index]].merge(CLEAN["races"][["raceId", "year"]], on="raceId")
log_step("rule B: one row per driver per race", "results", n, len(res_c), f"{dup_pairs} extra rows")
print(f"Duplicate rows removed: {n - len(res_c)}")
print(f"Drivers whose shared-drive podium is not credited under rule B: {len(lost)} "
      f"(seasons {lost['year'].min()}-{lost['year'].max()}, outside the 2003+ window)")

french51 = CLEAN["races"].query("year == 1951 and name == 'French Grand Prix'")["raceId"]
print("1951 French GP, podium rows after rule B + override:")
display(res_c[res_c["raceId"].isin(french51) & (res_c["podium"] == 1)]
        .merge(CLEAN["drivers"][["driverId", "surname"]], on="driverId")
        .sort_values("positionOrder")[["surname", "resultId", "number", "grid", "positionOrder", "status"]])

# --- grid = 0 (pit-lane start) -> that race's last slot + 1, with a flag -------------------------
def recode_pit_lane(df):
    df = df.copy()
    df["pit_lane_start"] = df["grid"] == 0
    last_slot = df.groupby("raceId")["grid"].transform("max")
    df.loc[df["pit_lane_start"], "grid"] = last_slot[df["pit_lane_start"]] + 1
    return df


res_c = recode_pit_lane(res_c)
CLEAN["sprint_results"] = recode_pit_lane(CLEAN["sprint_results"])
print("Pit-lane starts recoded:", int(res_c["pit_lane_start"].sum()), "in results,",
      int(CLEAN["sprint_results"]["pit_lane_start"].sum()), "in sprint_results")
print("Smallest grid value now:", int(res_c["grid"].min()))
CLEAN["results"] = res_c
''')

md(r"""
**What this did.**
- **Target first:** `podium` is computed from `positionOrder` before anything is dropped, so cleaning can't change who finished on the podium.
- **Non-starters:** the rule tests **laps = 0**, not `grid = 0`, for the non-start statuses. The 2005 US GP shows why: 14 drivers withdrew before the start *while holding grid slots* (several near the front). With a `grid = 0` test they would stay in the 2003+ data as "started from P3, no podium", which is false. Rows with no grid slot *and* no laps are also removed (crashed in warm-up, injured, excluded before the start).
  Pit-lane starters (`grid = 0` but laps > 0) **raced**, so they stay.
- **Rule B (reworded after review):** keep the row with the **lowest `resultId`**, as a proxy for the car the driver started in. In this dataset the extra (takeover) rows were mostly added later: in 70 of the 85 raw pairs, one row has a `resultId` below 20,000 and the other one above it. `grid` can't decide: in every remaining pair **both** rows have grid > 0, because the second row carries the grid slot of the car's original driver. The rule never looks at the result, so features and target describe the same car. The cost is that a few 1950s drivers lose podium credit from a car they took over mid-race.
- **Verified overrides:** in 3 pairs the proxy is known to be wrong. Its row is an exact copy of another driver's only row (same car, grid, laps and finish), so that car was the other driver's start. For those pairs we keep the driver's own starting car (table above). The most visible case is **Ascari, 1951 French GP**: the proxy kept González's car (2nd place), which gave that race a 4th podium row. With the override, the race has exactly 3 podium rows: Fagioli 1st, González 2nd, Villoresi 3rd.
- **Limitation:** the proxy may still be wrong for a few other 1950s pairs that can't be verified from the data. All of them are before 1979, so they affect only the all-years model version and Q2 (which uses all seasons). Even then the effect is negligible: 82 pairs among ~25,000 rows, and most of them aren't podium rows.
- **Grid:** a pit-lane start becomes "behind the last car" (that race's highest grid + 1), with `pit_lane_start` keeping the information that it happened.

**Limitation: formation-lap failures (documented, no code change).** The data cannot distinguish a car that failed on the formation lap (never started) from one that retired on lap 1; both show 0 laps with a failure status. About 49 rows in 2003+ (0.5%) may be non-starters recorded as starters. We keep them, because removing all 0-lap rows would also remove genuine first-lap retirements and would select rows using race outcomes.
""")

md(r"""
### 2.6.1 Indianapolis 500 (1950–1960)

From 1950 to 1960 the Indianapolis 500 counted as a round of the World Championship, but it was a separate American race: its own drivers and cars, a 33-car field on an oval, and the Grand Prix teams almost never took part. The checks below measure how different it is from the other races.
""")

code(r'''
indy_races = CLEAN["races"].loc[CLEAN["races"]["name"] == "Indianapolis 500", ["raceId", "year"]]
assert len(indy_races) == 11 and indy_races["year"].between(1950, 1960).all()

res_c = CLEAN["results"]
is_indy = res_c["raceId"].isin(indy_races["raceId"])
indy = res_c[is_indy].merge(CLEAN["drivers"][["driverId", "nationality"]], on="driverId")
indy_drivers = indy["driverId"].unique()
other_races = res_c.loc[~is_indy, "driverId"].unique()
print(f"Indianapolis 500: {len(indy_races)} races ({indy_races['year'].min()}-{indy_races['year'].max()}), "
      f"{len(indy)} cleaned rows, {len(indy_drivers)} drivers")
print(f"American drivers: {int((indy['nationality'] == 'American').sum())} of {len(indy)} rows")
print(f"Drivers who never started any other championship race: {int((~pd.Series(indy_drivers).isin(other_races)).sum())} "
      f"of {len(indy_drivers)}")
same_years = CLEAN["races"].loc[CLEAN["races"]["year"] <= 1960, "raceId"]
other_fields = res_c[~is_indy & res_c["raceId"].isin(same_years)].groupby("raceId").size()
print(f"Starters per race: Indy {indy.groupby('raceId').size().median():.0f} (median), "
      f"other 1950-1960 races {other_fields.median():.0f} (median)")

# Set aside (fully cleaned: non-starters, rule B and grid already applied), used only by the Q2 robustness check.
INDY500_RESULTS = res_c[is_indy].copy()
n = len(res_c)
CLEAN["results"] = res_c[~is_indy].copy()
log_step("remove Indianapolis 500 (1950-1960)", "results", n, len(CLEAN["results"]),
         "separate US race in the championship: own drivers, 33-car oval field")
print(f"Rows removed: {n - len(CLEAN['results'])}")
''')

md(r"""
**What this did and why.** The Indianapolis 500 rows are removed from the cleaned results:
- **Not the same competition:** 380 of its 381 starts are American drivers, and 103 of its 107 drivers never started another championship race. It doesn't describe the F1 Grands Prix we want to predict.
- **It breaks Q2 (home advantage):** every Indy start is a "home" start for an American driver at a US circuit (the only exception is Ascari in 1952). There is no away driver in the same race to compare with, and the 33-car field pushes most of these rows into the back grid bands. Indy alone was 16% of all home starts.
- **It would distort an all-years model:** 381 rows from another series, with no qualifying data.

The `races` rows stay (the races did happen, and `driver_standings` still refers to them), so no foreign key breaks. The removed rows are kept aside in `INDY500_RESULTS`, only so that Q2 can show the result *with* Indy as a robustness check.
""")

# ---------------------------------------------------------------- 2.7
md(r"""
## 2.7 Qualifying times: outliers
""")

code(r'''
Q_COLS = ["q1_sec", "q2_sec", "q3_sec"]
q_c = CLEAN["qualifying"]
too_slow = q_c[Q_COLS] > 300   # no modern F1 qualifying lap takes 5 minutes
display(q_c[too_slow.any(axis=1)].merge(CLEAN["races"][["raceId", "year", "name"]], on="raceId"))
q_c[Q_COLS] = q_c[Q_COLS].mask(too_slow)
CLEAN["qualifying"] = q_c

ratio = q_c["q1_sec"] / q_c.groupby("raceId")["q1_sec"].transform("min")
print("Q1 time / fastest Q1 in that race, after the fix:")
display(ratio.describe().round(3).to_frame("ratio").T)
''')

md(r"""
**What this did.** One Q1 time of ~1002 s (about 16 minutes) is impossible as a flying lap, so it becomes missing; the qualifying **position** of that row is kept. After the fix, the slowest Q1 time is a plausible multiple of the race's fastest lap.
Missing Q2/Q3 times are **not** filled in: they mean "eliminated earlier", which is handled as a feature in section 3 (e.g. gap to pole from the best session time).
""")

# ---------------------------------------------------------------- 2.8
md(r"""
## 2.8 Leakage-safe standings (pre-race)

`driver_standings` at `raceId = R` are the standings **after** race R. Before race R we only know the standings after the **previous round of the same season**.
We build `*_prerace` tables keyed by the race being predicted. Round 1 has no previous round, so it gets no row (handled as "start of season" in feature engineering).
""")

code(r'''
season_order = CLEAN["races"].sort_values(["year", "round"])[["raceId", "year", "round"]].copy()
season_order["prev_raceId"] = season_order.groupby("year")["raceId"].shift(1).astype("Int64")


def prerace_standings(table, key):
    after_prev = CLEAN[table][["raceId", key, "points", "position", "wins"]].rename(columns={
        "raceId": "prev_raceId", "points": "pre_points", "position": "pre_position", "wins": "pre_wins"})
    out = season_order[["raceId", "prev_raceId"]].dropna().merge(after_prev, on="prev_raceId", how="inner")
    return out.drop(columns="prev_raceId")


CLEAN["driver_standings_prerace"] = prerace_standings("driver_standings", "driverId")
CLEAN["constructor_standings_prerace"] = prerace_standings("constructor_standings", "constructorId")

# Demonstration on the 2021 season finale: same-race standings already contain that race's result.
finale = season_order[season_order["year"] == 2021].iloc[-1]["raceId"]
names = CLEAN["drivers"].assign(driver=CLEAN["drivers"]["forename"] + " " + CLEAN["drivers"]["surname"])[["driverId", "driver"]]
after = CLEAN["driver_standings"].query("raceId == @finale")[["driverId", "points", "position"]]
before = CLEAN["driver_standings_prerace"].query("raceId == @finale")[["driverId", "pre_points", "pre_position"]]
display(before.merge(after, on="driverId").merge(names, on="driverId")
        .rename(columns={"points": "points AFTER race (leak)", "position": "position AFTER race (leak)"})
        .sort_values("pre_position").head(3)[["driver", "pre_points", "pre_position", "points AFTER race (leak)", "position AFTER race (leak)"]])
''')

md(r"""
**What this shows.** Before the 2021 finale, the top two drivers were tied on points. The same-race standings already show who won the title, i.e. the race result.
Using `driver_standings` at the current `raceId` as a feature would leak the answer. The `*_prerace` tables only contain what was known before lights out.
""")

# ---------------------------------------------------------------- 2.9
md(r"""
## 2.9 Output tables: full results (for the questions) and the modeling base (2003+)
""")

code(r'''
MODEL_START_YEAR = 2003

race_info = CLEAN["races"][["raceId", "year", "round", "date", "circuitId", "name"]].rename(columns={"name": "race_name"})
results_full = CLEAN["results"].merge(race_info, on="raceId", how="left", validate="many_to_one")

POST_RACE = ["positionOrder", "points", "laps", "milliseconds", "fastestLap", "rank", "fastestLapTime_sec",
             "fastestLapSpeed", "statusId", "status", "status_category"]
PRE_RACE = ["raceId", "driverId", "constructorId", "year", "round", "date", "circuitId", "grid", "pit_lane_start"]

model_base = (results_full.loc[results_full["year"] >= MODEL_START_YEAR, PRE_RACE + ["podium"]]
              .sort_values(["date", "raceId", "grid"]).reset_index(drop=True))
assert not set(POST_RACE) & set(model_base.columns), "post-race column in the modeling table"
log_step(f"modeling window {MODEL_START_YEAR}+", "model_base", len(results_full), len(model_base), "qualifying coverage ~100% from 2003")

print(f"results_full: {len(results_full):,} rows ({results_full['year'].min()}-{results_full['year'].max()}), "
      f"used for the data-engineering questions")
print(f"model_base:   {len(model_base):,} rows, {model_base['raceId'].nunique()} races, "
      f"podium rate {100 * model_base['podium'].mean():.1f}%")
display(model_base.head())
''')

md(r"""
**Two tables, two purposes.**
- `results_full`: every cleaned driver-race (all years, without the Indianapolis 500) **with** post-race columns. Q1–Q3 need the finishing order and status.
- `model_base`: the modeling grain (one row per driver per race, 2003+) with **only pre-race columns** plus the target. Feature engineering (section 3) adds qualifying, form and standings features to it.
  The code asserts that no post-race column is in it. The podium rate (~14%) shows the class imbalance the metrics must handle (hence PR-AUC).
""")

md(r"""
### 2.9.1 Final status mapping, with row counts

The full status → category table from 2.5, with how many cleaned race starts carry each status: all seasons, and the Q3 window (2014–2024).
""")

code(r'''
status_counts = (results_full.assign(q3_window=results_full["year"] >= 2014)
                 .groupby("statusId").agg(rows_all_years=("resultId", "size"), rows_2014_2024=("q3_window", "sum")))
status_map = (CLEAN["status"].merge(status_counts, on="statusId", how="left")
              .fillna({"rows_all_years": 0, "rows_2014_2024": 0})
              .astype({"rows_all_years": int, "rows_2014_2024": int}))
status_map["status_category"] = pd.Categorical(status_map["status_category"], STATUS_CATEGORIES, ordered=True)
status_map = status_map.sort_values(["status_category", "rows_all_years"], ascending=[True, False])

display(status_map.groupby("status_category", observed=True)[["rows_all_years", "rows_2014_2024"]].sum()
        .assign(n_statuses=status_map.groupby("status_category", observed=True).size()))
with pd.option_context("display.max_rows", 200):
    display(status_map[["status_category", "status", "rows_all_years", "rows_2014_2024"]].reset_index(drop=True))
''')

md(r"""
**What this shows.** Every status and its category, for review. Six statuses have 0 rows: `107% Rule`, `Did not prequalify`, `Underweight` and `Safety concerns` only occurred on removed non-starter rows, and `+38 Laps` / `+49 Laps` exist in the status table but never in `results`. In the Q3 window, `Retired` is the largest *unclear* status, which is why Q3 also reports the broad version (mechanical + unclear).
""")

# ---------------------------------------------------------------- 2.10
md(r"""
## 2.10 Re-validation after cleaning

The section-1 checks re-run on the cleaned tables. They must all pass.
""")

code(r'''
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "passed": bool(ok), "detail": detail})


for name, df, key in [("results_full", results_full, ["raceId", "driverId"]),
                      ("model_base", model_base, ["raceId", "driverId"]),
                      ("qualifying", CLEAN["qualifying"], ["raceId", "driverId"]),
                      ("driver_standings_prerace", CLEAN["driver_standings_prerace"], ["raceId", "driverId"]),
                      ("constructor_standings_prerace", CLEAN["constructor_standings_prerace"], ["raceId", "constructorId"])]:
    n_dup, n_null = int(df.duplicated(key).sum()), int(df[key].isna().any(axis=1).sum())
    check(f"{name}: {' + '.join(key)} unique and non-null", n_dup == 0 and n_null == 0, f"{n_dup} duplicates, {n_null} null")

# Every foreign key between the cleaned tables (child.column -> parent.column).
FKS_CLEAN = [
    ("races", "circuitId", "circuits"),
    ("results", "raceId", "races"), ("results", "driverId", "drivers"),
    ("results", "constructorId", "constructors"), ("results", "statusId", "status"),
    ("qualifying", "raceId", "races"), ("qualifying", "driverId", "drivers"),
    ("sprint_results", "raceId", "races"), ("sprint_results", "driverId", "drivers"),
    ("sprint_results", "constructorId", "constructors"), ("sprint_results", "statusId", "status"),
    ("driver_standings", "raceId", "races"), ("driver_standings", "driverId", "drivers"),
    ("constructor_standings", "raceId", "races"), ("constructor_standings", "constructorId", "constructors"),
    ("constructor_results", "raceId", "races"), ("constructor_results", "constructorId", "constructors"),
    ("driver_standings_prerace", "raceId", "races"), ("driver_standings_prerace", "driverId", "drivers"),
    ("constructor_standings_prerace", "raceId", "races"), ("constructor_standings_prerace", "constructorId", "constructors"),
]
fk_report = pd.DataFrame([{
    "foreign key": f"{child}.{col} -> {parent}.{col}",
    "child rows": len(CLEAN[child]),
    "broken references": int((~CLEAN[child][col].dropna().isin(CLEAN[parent][col])).sum()),
} for child, col, parent in FKS_CLEAN])
display(fk_report)
check(f"all {len(FKS_CLEAN)} foreign keys reconcile", fk_report["broken references"].sum() == 0,
      f"{int(fk_report['broken references'].sum())} broken references")
for col, parent in [("raceId", "races"), ("driverId", "drivers"), ("constructorId", "constructors"), ("circuitId", "circuits")]:
    broken = int((~model_base[col].isin(CLEAN[parent][col])).sum())
    check(f"FK model_base.{col} -> {parent}", broken == 0, f"{broken} broken")

leftover = sum(int(df[c].isin([SENTINEL, ""]).sum()) for df in CLEAN.values() for c in df.select_dtypes("object"))
check("no '\\N' or empty strings left in any table", leftover == 0, f"{leftover} cells")
check("results_full: grid >= 1 everywhere", results_full["grid"].min() >= 1, f"min grid = {results_full['grid'].min()}")
check("model_base: no text columns (model-ready types)", model_base.select_dtypes("object").empty,
      str(model_base.dtypes.astype(str).to_dict()))
per_race = model_base.groupby("raceId")["podium"].sum()
check("model_base: exactly 3 podium rows per race", (per_race == 3).all(), str(per_race.value_counts().to_dict()))
check("model_base: no post-race columns", not set(POST_RACE) & set(model_base.columns))

check_report = pd.DataFrame(checks)
display(check_report)
assert check_report["passed"].all(), "a post-cleaning check failed"
''')

md(r"""
**What this means.** The grain is fixed (unique `raceId + driverId`), no placeholder text is left, `grid` has no fake zeros, and every 2003+ race has exactly 3 podium rows (the shared-drive anomaly only existed before 1965).
**Foreign keys (extended after review):** the check now covers **all 21** foreign keys between the cleaned tables (it covered 8 before), including the deferred `sprint_results` and `constructor_results` and the new `*_prerace` tables. None has a broken reference: every row still points to an existing race, driver, team, circuit or status. (Only `results` lost rows, and no table references a `results` row.)
""")

md(r"""
### 2.10.1 Further sanity checks on the cleaned data

These don't pass/fail. They look for anything suspicious the cleaning might have caused or left behind.
""")

code(r'''
race_names = CLEAN["races"][["raceId", "year", "round", "name"]]
surnames = CLEAN["drivers"][["driverId", "surname"]]

print("Missing values in model_base:", int(model_base.isna().sum().sum()))
print("Ranges: grid", model_base["grid"].min(), "-", model_base["grid"].max(),
      "| dates", model_base["date"].min().date(), "-", model_base["date"].max().date())
age = (model_base.merge(CLEAN["drivers"][["driverId", "dob"]], on="driverId")
       .pipe(lambda d: (d["date"] - d["dob"]).dt.days / 365.25))
print(f"Driver age at race: {age.min():.1f} - {age.max():.1f} years")
print("Qualifying times (s):", {c: (round(CLEAN['qualifying'][c].min(), 1), round(CLEAN['qualifying'][c].max(), 1)) for c in Q_COLS})

display(Markdown("**Smallest fields (should be explainable races, not cleaning errors)**"))
display(model_base.groupby("raceId").size().nsmallest(2).rename("starters").to_frame().join(race_names.set_index("raceId")))

display(Markdown("**Two starters on the same grid slot (excluding pit-lane starts)**"))
same_slot = model_base[~model_base["pit_lane_start"]]
same_slot = same_slot[same_slot.duplicated(["raceId", "grid"], keep=False)]
display(same_slot.merge(race_names, on=["raceId", "year", "round"]).merge(surnames, on="driverId")[["year", "name", "surname", "grid"]])

display(Markdown("**Races with more than 3 podium rows in results_full**"))
pods = results_full.groupby("raceId")["podium"].sum()
display(race_names[race_names["raceId"].isin(pods[pods > 3].index)])

display(Markdown("**Pre-race standings coverage for model_base rows**"))
for table, key in [("driver_standings_prerace", "driverId"), ("constructor_standings_prerace", "constructorId")]:
    joined = model_base.merge(CLEAN[table], on=["raceId", key], how="left")
    missing = joined["pre_points"].isna()
    print(f"{table}: {int(missing.sum())} rows without a standing, "
          f"{int((missing & (joined['round'] == 1)).sum())} of them in round 1, "
          f"{int((missing & (joined['round'] > 1)).sum())} later in the season")

early = CLEAN["races"].query("year == 2003").sort_values("round")["raceId"].head(3)
print("2003 rounds 1-3 -> standings rows:", CLEAN["driver_standings"][CLEAN["driver_standings"]["raceId"].isin(early)]
      .groupby("raceId").size().tolist(), "vs starters:", results_full[results_full["raceId"].isin(early)].groupby("raceId").size().tolist())
''')

md(r"""
**What these show.**
- **No missing values** in `model_base`, and ranges are plausible (grid 1–24, driver ages ~17–44, qualifying laps ~53–154 s).
- **Small fields are real:** the 2005 US GP had 6 starters (14 withdrew), and the 2015 Australian GP had 15 (several cars did not start). They aren't cleaning errors.
- **4 pairs of drivers share a grid slot** (e.g. 2015 Italian GP). This comes from the source data around grid penalties, not from cleaning. It's 8 rows out of 8,934. Grid is used as an ordinal feature, so a tie is harmless and the rows are left as they are.
- **1 race with > 3 podium rows** is left: the 1955 Argentine GP, a cross-driver shared drive (1.6) where two different drivers are credited with one car's 3rd place. It's expected and outside the 2003+ window. (There were 4: the Rule B override fixed the 1951 French GP, and the 1951 and 1955 Indianapolis 500 were removed in 2.6.1.)
- **Gaps in the pre-race standings:** `driver_standings` early in a season only lists drivers who already have an entry (2003 round 1 has 11 standings rows for 20 starters). Missing values after round 1 therefore mean **"no points yet"** (or the driver didn't race the previous round), not a broken join. **Rule for feature engineering (section 3):** fill `pre_points` and `pre_wins` with 0, and give `pre_position` a "behind everyone ranked" value plus a `no_standing_yet` flag. Round 1 always gets this treatment.
""")

md(r"""
### 2.10.2 Evidence that standings are post-race

`driver_standings.points` at race R should be the driver's season total **after** race R. To check this, we rebuild each driver's cumulative season points race by race from the cleaned `results` and compare them to the standings in two ways:
- **current race included:** points from round 1 up to and including R;
- **current race not included:** points from round 1 up to R − 1.

If standings are post-race, the first version should match almost every row and the second should not.
""")

code(r'''
rounds = CLEAN["races"][["raceId", "year", "round"]]


def per_race_points(df, name):
    return df.groupby(["raceId", "driverId"])["points"].sum().rename(name).reset_index()


# Points on raw rows that cleaning removed: the Indianapolis 500 (2.6.1), and shared-drive rows dropped by rule B.
raw_pts = RAW["results"][["resultId", "raceId", "driverId", "points"]].astype(
    {"resultId": int, "raceId": int, "driverId": int, "points": float})
removed = raw_pts[~raw_pts["resultId"].isin(CLEAN["results"]["resultId"])]
indy_pts = per_race_points(removed[removed["raceId"].isin(indy_races["raceId"])], "indy_pts")
removed_pts = per_race_points(removed[~removed["raceId"].isin(indy_races["raceId"])], "removed_pts")

# One row per driver, per round of every season in which the driver appears (0 points when he didn't score or race).
driver_seasons = (pd.concat([CLEAN["results"][["raceId", "driverId"]], CLEAN["driver_standings"][["raceId", "driverId"]]])
                  .merge(rounds, on="raceId")[["driverId", "year"]].drop_duplicates())
season = driver_seasons.merge(rounds, on="year")
for extra in [per_race_points(CLEAN["results"], "race_pts"), per_race_points(CLEAN["sprint_results"], "sprint_pts"),
              indy_pts, removed_pts]:
    season = season.merge(extra, on=["raceId", "driverId"], how="left")
season = (season.fillna({"race_pts": 0.0, "sprint_pts": 0.0, "indy_pts": 0.0, "removed_pts": 0.0})
          .sort_values(["driverId", "year", "round"]))
by_season = season.groupby(["driverId", "year"])
season["cum_incl"] = by_season["race_pts"].cumsum()                  # rounds 1..R
season["cum_excl"] = season["cum_incl"] - season["race_pts"]          # rounds 1..R-1
season["sprint_cum"] = by_season["sprint_pts"].cumsum()
season["indy_cum"] = by_season["indy_pts"].cumsum()
season["removed_cum"] = by_season["removed_pts"].cumsum()

st = CLEAN["driver_standings"][["raceId", "driverId", "points"]].merge(season, on=["raceId", "driverId"], how="left",
                                                                         validate="one_to_one")
st["match_incl"] = np.isclose(st["points"], st["cum_incl"])
st["match_excl"] = np.isclose(st["points"], st["cum_excl"])
scored = st["race_pts"] > 0
display(pd.DataFrame({
    "all standings rows": [st["match_incl"].mean(), st["match_excl"].mean()],
    "rows where the driver scored in that race": [st.loc[scored, "match_incl"].mean(), st.loc[scored, "match_excl"].mean()],
}, index=["current race INCLUDED", "current race NOT included"]).mul(100).round(1).astype(str).add(" %"))

# Why do the remaining rows not match (current race included)?
mis = st[~st["match_incl"]].copy()
all_removed = mis["cum_incl"] + mis["indy_cum"] + mis["removed_cum"]
half = (mis["points"] % 1 != 0) | (mis["cum_incl"] % 1 != 0)
CAUSES = ["sprint points (standings add the sprint, results don't)",
          "Indianapolis 500 points (rows removed in 2.6.1, still in the standings)",
          "shared drive (points of a row removed by rule B)",
          "best-results rule before 1991 (only the best N scores counted)",
          "half points"]
mis["cause"] = np.select(
    [np.isclose(mis["points"], mis["cum_incl"] + mis["sprint_cum"]),
     np.isclose(mis["points"], all_removed) & (mis["indy_cum"] > 0),
     np.isclose(mis["points"], all_removed) & (mis["removed_cum"] > 0),
     (mis["year"] < 1991) & (mis["points"] < all_removed),
     half],
    CAUSES, "other")
causes = (mis.groupby("cause").agg(rows=("points", "size"), seasons=("year", lambda y: f"{y.min()}-{y.max()}"))
          .reindex(CAUSES + ["other"], fill_value=0))
display(causes)
names_ = CLEAN["drivers"].assign(driver=CLEAN["drivers"]["forename"] + " " + CLEAN["drivers"]["surname"])[["driverId", "driver"]]
display(mis.merge(names_, on="driverId").sort_values(["cause", "year", "round"]).groupby("cause").head(2)
        [["cause", "year", "round", "driver", "points", "cum_incl", "cum_excl", "sprint_cum", "indy_cum", "removed_cum"]]
        .rename(columns={"points": "standings points", "cum_incl": "results incl. race", "cum_excl": "results excl. race"}))
''')

md(r"""
**What this shows.** Standings are **post-race**.
- With the current race **included**, the rebuilt totals match `driver_standings.points` in **96.1%** of rows; **not included**, only **74.7%** (those are mostly rows where the driver scored 0 in that race, so both totals are the same).
- The clearest test is the rows where the driver **scored** in that race: **92.6%** match with the race included vs **0.1%** without. The standings row for race R already contains race R's points, so it must be lagged by one race (2.8).
- **Every remaining mismatch has a known cause:** 689 rows are **sprint points** (2021–2024: the standings add the sprint, `results` doesn't). 452 are **Indianapolis 500 points** (1950–1960: the championship counted them, but the Indy rows were removed in 2.6.1). 84 are **shared drives** (1950–1957: the standings count the points of a row that rule B removed). 124 are the **best-results rule** before 1991 (only a driver's best N scores counted, so the standings are lower than the plain sum). **Half points** cause no mismatch, because `results.points` already holds the halves.
""")

# ---------------------------------------------------------------- 2.11
md(r"""
## 2.11 Before/after cleaning

The brief asks for EDA **before and after** cleaning, to show what the cleaning changed. Each figure puts the raw data on the **left** and the cleaned data on the **right**.
""")

code(r'''
raw_res = RAW["results"].assign(grid=lambda d: d["grid"].astype(int), positionOrder=lambda d: d["positionOrder"].astype(int))
raw_res["podium"] = (raw_res["positionOrder"] <= 3).astype(int)
raw_res = raw_res.merge(RAW["races"][["raceId", "year"]], on="raceId").assign(year=lambda d: d["year"].astype(int))
BEFORE_COLOR, AFTER_COLOR = "#8C8C8C", "#4C72B0"


def before_after(title, figsize=(13, 4), sharey=True):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=sharey)
    axes[0].set_title(f"Before: {title}")
    axes[1].set_title(f"After: {title}")
    return fig, axes


fig, axes = before_after("number of podium rows per race")
for ax, df, color in [(axes[0], raw_res, BEFORE_COLOR), (axes[1], results_full, AFTER_COLOR)]:
    counts = df.groupby("raceId")["podium"].sum().value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color=color)
    ax.bar_label(bars, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("podium rows in the race")
    ax.set_ylabel("races (log scale)")
plt.tight_layout()
plt.show()
''')

md(r"""
**Podium rows per race:** before cleaning, 18 races had 4, 5 or 7 "podium" rows (shared drives counted the same driver twice); after the non-starter and duplicate rules and removing the Indianapolis 500, 1,113 of the 1,114 races have exactly 3. The one left with 4 is the 1955 Argentine GP, where two *different* drivers share one car's 3rd place.
""")

code(r'''
KEY_COLUMNS = [  # (table, raw column, cleaned column; None = dropped in 2.2)
    ("results", "position", None), ("results", "time", None), ("results", "milliseconds", "milliseconds"),
    ("results", "fastestLapTime", "fastestLapTime_sec"), ("results", "rank", "rank"), ("results", "number", "number"),
    ("qualifying", "q1", "q1_sec"), ("qualifying", "q2", "q2_sec"), ("qualifying", "q3", "q3_sec"),
    ("races", "time", "start_utc"), ("drivers", "number", None), ("drivers", "code", None),
]
rows = []
for t, raw_col, clean_col in KEY_COLUMNS:
    s = RAW[t][raw_col].str.strip()
    rows.append({"column": f"{t}.{raw_col}",
                 "raw '\\N'": int((s == SENTINEL).sum()), "raw empty string": int((s == "").sum()),
                 "raw rows": len(s),
                 "clean column": clean_col or "dropped",
                 "clean NaN": np.nan if clean_col is None else int(CLEAN[t][clean_col].isna().sum()),
                 "clean rows": len(CLEAN[t])})
missing_tbl = pd.DataFrame(rows)
missing_tbl["raw missing %"] = 100 * (missing_tbl["raw '\\N'"] + missing_tbl["raw empty string"]) / missing_tbl["raw rows"]
missing_tbl["clean NaN %"] = 100 * missing_tbl["clean NaN"] / missing_tbl["clean rows"]

fig, axes = before_after("missing values per key column (% of rows)", figsize=(14, 5))
y = np.arange(len(missing_tbl))
sent_pct = 100 * missing_tbl["raw '\\N'"] / missing_tbl["raw rows"]
empty_pct = 100 * missing_tbl["raw empty string"] / missing_tbl["raw rows"]
axes[0].barh(y, sent_pct, color=BEFORE_COLOR, label="'\\N' text")
axes[0].barh(y, empty_pct, left=sent_pct, color="#DD8452", label="empty string")
axes[0].legend(loc="lower right", fontsize=8)
axes[1].barh(y, missing_tbl["clean NaN %"].fillna(0), color=AFTER_COLOR, label="NaN")
for yy, (pct, col) in enumerate(zip(missing_tbl["clean NaN %"], missing_tbl["clean column"])):
    axes[1].text(1 if np.isnan(pct) else pct + 1, yy, "dropped (2.2)" if col == "dropped" else f"{pct:.0f}%  ({col})",
                 va="center", fontsize=8)
axes[0].set_yticks(y, missing_tbl["column"])
axes[0].invert_yaxis()
for ax in axes:
    ax.set_xlim(0, 125)
    ax.set_xlabel("% of rows")
plt.tight_layout()
plt.show()
display(missing_tbl[["column", "raw '\\N'", "raw empty string", "raw missing %", "clean column", "clean NaN", "clean NaN %"]].round(1))
''')

md(r"""
**Missing values:** before cleaning, pandas sees **none** of these gaps because they are the text `\N` (or an empty string in 68 `q2`/`q3` cells); after cleaning they are real `NaN` (for `qualifying` and `races` the counts are identical, plus the 1002.6 s Q1 lap set to NaN), sparse columns like `drivers.number/code` and `results.position/time` are dropped, and the `results` percentages fall slightly because non-starters, who had no times, were removed.
""")

code(r'''
fig, axes = before_after("grid position distribution")
bins = np.arange(-0.5, 36.5, 1)
axes[0].hist(raw_res["grid"], bins=bins, color=BEFORE_COLOR)
axes[1].hist(results_full["grid"].astype(int), bins=bins, color=AFTER_COLOR)
n_zero = int((raw_res["grid"] == 0).sum())
axes[0].annotate(f"grid = 0: {n_zero:,} rows", xy=(0, n_zero), xytext=(8, n_zero * 0.9), arrowprops={"arrowstyle": "->"})
for ax in axes:
    ax.set_xlabel("grid position")
    ax.set_ylabel("rows")
plt.tight_layout()
plt.show()
print(f"grid = 0 rows: {n_zero:,} before, {int((results_full['grid'] == 0).sum())} after; "
      f"pit-lane starters recoded and flagged: {int(results_full['pit_lane_start'].sum())}")
''')

md(r"""
**Grid:** the spike of 1,638 rows at `grid = 0` is gone: 1,566 of them were non-starters and were removed, and the 72 pit-lane starters now sit behind the last car of their race (with `pit_lane_start = True`), so `grid` is a clean ordinal feature where 1 is always the best.
""")

code(r'''
q1_raw = RAW["qualifying"]["q1"].map(duration_to_seconds)   # '\N' and '' parse to NaN
q1_clean = CLEAN["qualifying"]["q1_sec"]
outlier = RAW["qualifying"].loc[q1_raw.idxmax()]
fig, axes = before_after("qualifying Q1 lap time", sharey=False)
axes[0].hist(q1_raw.dropna(), bins=np.arange(40, 1060, 10), color=BEFORE_COLOR)
axes[0].set_yscale("log")
axes[0].annotate(f"'{outlier['q1']}' = {q1_raw.max():.0f} s", xy=(q1_raw.max(), 1), xytext=(q1_raw.max() - 450, 30),
                 arrowprops={"arrowstyle": "->"})
axes[1].hist(q1_clean.dropna(), bins=np.arange(40, 165, 2), color=AFTER_COLOR)
axes[0].set_ylabel("rows (log scale)")
axes[1].set_ylabel("rows")
for ax in axes:
    ax.set_xlabel("Q1 time (seconds)")
plt.tight_layout()
plt.show()
print(f"Q1 range before: {q1_raw.min():.1f}-{q1_raw.max():.1f} s; after: {q1_clean.min():.1f}-{q1_clean.max():.1f} s")
''')

md(r"""
**Q1 times:** before cleaning, one Q1 time of `16:42.640` (1002.6 s, Japan 1995) stretches the range to over 16 minutes; after setting it to NaN (2.7) every Q1 time lies between 53.9 and 153.9 s, a plausible range for a flying lap.
""")

code(r'''
rows_tbl = pd.DataFrame({"raw rows": {t: len(RAW[t]) for t in TABLES},
                         "clean rows": {t: len(CLEAN[t]) if t in CLEAN else 0 for t in TABLES}})
rows_tbl["removed"] = rows_tbl["raw rows"] - rows_tbl["clean rows"]
rows_tbl["note"] = ["not carried (2.1)" if t in UNUSED_TABLES else ("cleaned, not used (2.1)" if t in DEFERRED_TABLES else "")
                    for t in rows_tbl.index]

fig, axes = before_after("rows per table (log scale)", figsize=(14, 5))
y = np.arange(len(rows_tbl))
axes[0].barh(y, rows_tbl["raw rows"], color=BEFORE_COLOR)
axes[1].barh(y, rows_tbl["clean rows"].replace(0, np.nan), color=AFTER_COLOR)
for yy, (t, r) in enumerate(rows_tbl.iterrows()):
    axes[0].text(r["raw rows"] * 1.1, yy, f"{r['raw rows']:,}", va="center", fontsize=8)
    label = r["note"] if t in UNUSED_TABLES else f"{r['clean rows']:,}" + (f"  (-{r['removed']:,})" if r["removed"] else "")
    axes[1].text(max(r["clean rows"], 1) * 1.1, yy, label, va="center", fontsize=8)
axes[0].set_yticks(y, rows_tbl.index)
axes[0].invert_yaxis()
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1, 5e6)
    ax.set_xlabel("rows")
plt.tight_layout()
plt.show()
display(rows_tbl)
display(pd.DataFrame(cleaning_log))
''')

md(r"""
**Rows per table:** only `results` loses rows (26,759 → 24,645: 1,645 non-starters, 88 duplicate rows and 381 Indianapolis 500 rows, see the step table); `seasons`, `lap_times` and `pit_stops` are not carried, and every other table keeps all its rows (only values and types changed).
""")

code(r'''
fig, axes = before_after("podium rate per season")
for ax, df, color in [(axes[0], raw_res, BEFORE_COLOR), (axes[1], results_full, AFTER_COLOR)]:
    rate = 100 * df.groupby("year")["podium"].mean()
    ax.plot(rate.index, rate.values, color=color)
    ax.axvspan(MODEL_START_YEAR, 2024, color="#55A868", alpha=0.1, label="2003+ window")
    ax.set_xlabel("season")
    ax.set_ylabel("% of rows with a podium")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
''')

md(r"""
**Podium rate per season:** before cleaning, seasons with many non-qualifiers (late 1980s, as low as ~8%) look artificially weak because cars that never started sit in the denominator; after cleaning the rate follows 3 ÷ number of starters (about 11.5–19%), and inside the 2003+ window it barely changes, so the modeling data was mostly clean already.
""")

# ---------------------------------------------------------------- 2.12
md(r"""
## 2.12 Raw data still untouched
""")

code(r'''
after = {t: hashlib.sha256(read_raw_bytes(t)).hexdigest() for t in TABLES}
assert all(after[t] == RAW_FINGERPRINT[t] for t in TABLES), "A raw file changed during cleaning!"
print(f"All {len(TABLES)} raw files are still byte-identical after cleaning (SHA-256).")
''')

# ---------------------------------------------------------------- 2.13
md(r"""
## 2.13 Export: the cleaned dataset

The cleaned tables are written to `outputs/clean_data/`, so the next steps (and teammates) can start from them without re-running the cleaning. The raw files are not touched: these are new files.
- **One CSV per table**, missing values written as empty cells.
- **`schema.json`**: the type of every column. CSV loses types (e.g. nullable integers, dates), so `load_clean()` below uses it to restore them exactly.
- **`manifest.csv`**: rows, columns and SHA-256 of every file, so anyone can check they have the same data.
- **`README.md`**: what each file is, its key, and what cleaning did to it.

The code reads every file back and checks it is **identical** to the in-memory table (values and types).
""")

code(r'''
import json

CLEAN_DATA_DIR = Path("outputs/clean_data")
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# file -> (table, key, description)
EXPORT = {
    "model_base": (model_base, "raceId + driverId",
                   "Modeling base, 2003+: one row per driver per race, pre-race columns + target `podium` only."),
    "results_full": (results_full, "raceId + driverId",
                     "All cleaned driver-races 1950-2024 (no Indianapolis 500), with race info and post-race columns. "
                     "Used by the data-engineering questions. Post-race columns are never features."),
    "results": (CLEAN["results"], "resultId (and raceId + driverId)",
                "Cleaned results: target, non-starters removed, rule B, pit-lane grid recode, status + category."),
    "qualifying": (CLEAN["qualifying"], "qualifyId (and raceId + driverId)",
                   "q1/q2/q3 as seconds (`*_sec`); empty strings and the 1002.6 s outlier are NaN. Missing q2/q3 = knocked out."),
    "races": (CLEAN["races"], "raceId", "date as a date; start_utc where the start time is recorded (2005+)."),
    "drivers": (CLEAN["drivers"], "driverId", "dob as a date; nationality fixed; home_country for Q2."),
    "constructors": (CLEAN["constructors"], "constructorId", "Teams."),
    "circuits": (CLEAN["circuits"], "circuitId", "'United States' merged into 'USA'."),
    "status": (CLEAN["status"], "statusId", "status_category: finished / mechanical / accident / regulation / driver / unclear."),
    "driver_standings": (CLEAN["driver_standings"], "driverStandingsId (and raceId + driverId)",
                         "Standings AFTER each race (post-race, leakage!). Use driver_standings_prerace for features."),
    "constructor_standings": (CLEAN["constructor_standings"], "constructorStandingsId (and raceId + constructorId)",
                              "Standings AFTER each race (post-race, leakage!). Use constructor_standings_prerace for features."),
    "driver_standings_prerace": (CLEAN["driver_standings_prerace"], "raceId + driverId",
                                 "Leakage-safe: standings after the previous round, keyed by the race being predicted. Round 1 has no row."),
    "constructor_standings_prerace": (CLEAN["constructor_standings_prerace"], "raceId + constructorId",
                                      "Leakage-safe: standings after the previous round, keyed by the race being predicted. Round 1 has no row."),
    "sprint_results": (CLEAN["sprint_results"], "resultId (and raceId + driverId)",
                       "Cleaned but not used yet (optional feature later)."),
    "constructor_results": (CLEAN["constructor_results"], "constructorResultsId", "Cleaned but not used."),
}

schema, manifest = {}, []
for name, (df, key, desc) in EXPORT.items():
    path = CLEAN_DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)   # dates without a time are written as YYYY-MM-DD
    schema[name] = {c: str(d) for c, d in df.dtypes.items()}
    manifest.append({"file": path.name, "rows": len(df), "columns": df.shape[1], "key": key, "description": desc,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(CLEAN_DATA_DIR / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
manifest = pd.DataFrame(manifest)
manifest.to_csv(CLEAN_DATA_DIR / "manifest.csv", index=False)


def load_clean(name, folder=CLEAN_DATA_DIR):
    """Read one exported table back with its original column types."""
    types = json.loads((Path(folder) / "schema.json").read_text(encoding="utf-8"))[name]
    dates = [c for c, t in types.items() if t.startswith("datetime")]
    df = pd.read_csv(Path(folder) / f"{name}.csv", dtype={c: t for c, t in types.items() if c not in dates},
                     parse_dates=dates, keep_default_na=False, na_values=[""],
                     float_precision="round_trip")   # the default fast parser can be off in the last digit
    for c in dates:
        df[c] = df[c].astype(types[c])
    return df


# Round trip: every file must read back identical to the in-memory table.
for name, (df, _, _) in EXPORT.items():
    pd.testing.assert_frame_equal(load_clean(name), df.reset_index(drop=True), check_exact=True)
print(f"{len(EXPORT)} tables written to {CLEAN_DATA_DIR}/ and read back identical (values and types).")
display(manifest[["file", "rows", "columns", "key"]])
''')

code(r'''
lines = ["# Cleaned F1 dataset", "",
         "Written by section 2.13 of `f1_podium_prediction.ipynb` (do not edit by hand; re-run the notebook instead).",
         "Raw source: Kaggle / Ergast *Formula 1 World Championship (1950-2024)*. The raw CSVs are unchanged.", "",
         "Load a table with its original types (nullable integers, dates):", "",
         "```python", "df = load_clean(\"model_base\")   # defined in section 2.13 of the notebook", "```",
         "or with plain pandas: `pd.read_csv(path, keep_default_na=False, na_values=[\"\"], float_precision=\"round_trip\")`, "
         "then apply the types in `schema.json`.", "",
         "## Files", "", "| file | rows | key | what it is |", "|---|---|---|---|"]
lines += [f"| `{r.file}` | {r.rows:,} | {r.key} | {r.description} |" for r in manifest.itertuples()]
lines += ["", "## What cleaning did (details in notebook section 2)", "",
          "- `\\N` and empty strings -> missing values; text stripped; columns typed (IDs/counts integer, points float, dates as dates, lap times as seconds).",
          "- `results`: target `podium = positionOrder <= 3` created first; 1,645 non-starters removed; one row per (raceId, driverId) by rule B "
          "(lowest resultId as a proxy for the starting car, 3 verified overrides); `grid = 0` pit-lane starts -> last slot + 1 with `pit_lane_start`; "
          "the Indianapolis 500 (1950-1960, 381 rows) removed.",
          "- `qualifying`: times in seconds; the 1002.6 s Q1 lap set to missing.",
          "- Labels: 'Argentinian ' -> 'Argentine', 'United States' -> 'USA', `drivers.home_country` added; every status has a category.",
          "- `*_prerace` standings: leakage-safe standings from the previous round.",
          "- Not exported: `seasons`, `lap_times`, `pit_stops` (not used).", "",
          "**Leakage reminder:** post-race columns (`positionOrder`, `points`, `laps`, `status*`, `milliseconds`, `fastestLap*`, `rank`) "
          "and the non-prerace standings must never be used as features."]
(CLEAN_DATA_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("README.md written:", len(lines), "lines")
''')

md(r"""
**What this did.** The cleaned dataset now exists as files in `outputs/clean_data/`: 15 CSVs, plus `schema.json`, `manifest.csv` and a `README.md` describing each file. Reading every file back gives exactly the in-memory tables, so later sections (or a teammate) can start from `load_clean("model_base")` instead of re-running the cleaning. The files are rewritten on every run, so they always match the notebook.
""")

# ================================================================ 3. DATA-ENGINEERING QUESTIONS
md(r"""
# 3. Data-Engineering Questions

Each question is answered from the **cleaned dataset files** in `outputs/clean_data/` (written in 2.13), read back with `load_clean()`, using multi-table joins. The questions don't use the in-memory tables, so they show that the exported files alone are enough. For every question we state:
- **Tables & joins:** which tables are combined, and on which keys.
- **Grain:** what one row represents.
- **Missingness:** what is missing and how it is handled.
- **Cutoff:** whether the pre-race cutoff matters.

Then come the result, a plot, and a plain-language answer.
""")

code(r'''
# Section 3 reads the cleaned dataset from its files (written in 2.13), not from the in-memory tables.
CLEAN_DIR_CANDIDATES = [os.environ.get("F1_CLEAN"), str(CLEAN_DATA_DIR),
                        *map(str, Path("/kaggle/input").glob("*/clean_data"))]
QA_CLEAN_DIR = next(Path(c) for c in filter(None, CLEAN_DIR_CANDIDATES)
                    if (Path(c) / "results_full.csv").exists())
print("Cleaned data source:", QA_CLEAN_DIR.resolve())

results_full_c = load_clean("results_full", QA_CLEAN_DIR)
races_c = load_clean("races", QA_CLEAN_DIR)
qualifying_c = load_clean("qualifying", QA_CLEAN_DIR)
drivers_c = load_clean("drivers", QA_CLEAN_DIR)
circuits_c = load_clean("circuits", QA_CLEAN_DIR)[["circuitId", "name", "country"]].rename(columns={"name": "circuit"})
constructors_c = (load_clean("constructors", QA_CLEAN_DIR)[["constructorId", "name"]]
                  .rename(columns={"name": "constructor"}))
race_info_c = races_c[["raceId", "year", "round", "date", "circuitId", "name"]].rename(columns={"name": "race_name"})
quali_pos = qualifying_c[["raceId", "driverId", "position"]].rename(columns={"position": "quali_pos"})

# Checks on what we loaded: keys unique and non-null, and every foreign key found.
for name, df, key in [("results_full", results_full_c, ["raceId", "driverId"]), ("races", races_c, ["raceId"]),
                      ("qualifying", qualifying_c, ["raceId", "driverId"]), ("drivers", drivers_c, ["driverId"]),
                      ("circuits", circuits_c, ["circuitId"]), ("constructors", constructors_c, ["constructorId"])]:
    assert df[key].notna().all().all() and not df.duplicated(key).any(), name
for col, parent in [("circuitId", circuits_c), ("driverId", drivers_c), ("constructorId", constructors_c)]:
    assert results_full_c[col].isin(parent[col]).all(), col
assert results_full_c["raceId"].isin(races_c["raceId"]).all()
assert results_full_c["status_category"].notna().all()

display(pd.DataFrame({"rows": [len(results_full_c), len(races_c), len(qualifying_c), len(drivers_c),
                               len(circuits_c), len(constructors_c)]},
                     index=["results_full", "races", "qualifying", "drivers", "circuits", "constructors"]))
''')

# ---------------------------------------------------------------- Q1
md(r"""
## 3.1 Q1: Which circuits convert a front-row start into a podium most often, and did that change in the hybrid era (2014+)?

**In simple words:** if a driver starts in the first two places, how often does he finish in the top three, and at which tracks is that most likely?

- **Tables & joins:** `clean_data/results_full.csv` (grid, podium, circuit, year) ⟶ `clean_data/qualifying.csv` on `raceId + driverId` (qualifying position) ⟶ `clean_data/circuits.csv` on `circuitId` (circuit name).
- **Grain:** one row = one driver in one race.
- **Two definitions of "front row"**, because they differ (penalties):
  - **started** on the front row = `grid` 1 or 2 (the main answer: it's where the car actually lined up);
  - **qualified** on the front row = qualifying position 1 or 2 (shown for comparison).
- **Missingness:** qualifying is complete only from 2003 (1.7), so the comparison is **2003–2013 (pre-hybrid) vs 2014–2024 (hybrid)**. Circuits need **≥ 8 front-row starts in each era** (about 4 races) so a single race doesn't decide the ranking.
- **Cutoff:** grid and qualifying are known before the race, so both are valid "starting" information.
""")

code(r'''
ERAS = ["2003-2013 (pre-hybrid)", "2014-2024 (hybrid)"]
q1 = (results_full_c[results_full_c["year"] >= 2003]
      .merge(quali_pos, on=["raceId", "driverId"], how="left", validate="one_to_one")
      .merge(circuits_c, on="circuitId", how="left", validate="many_to_one"))
q1["era"] = np.where(q1["year"] >= 2014, ERAS[1], ERAS[0])

started_fr = q1[q1["grid"] <= 2]
qualified_fr = q1[q1["quali_pos"] <= 2]
print(f"Front-row qualifiers who did NOT start on the front row (penalties): {int((qualified_fr['grid'] > 2).sum())}")
print(f"Front-row starters who did NOT qualify on the front row (moved up):  {int((started_fr['quali_pos'] > 2).sum())}")

overall = pd.DataFrame({
    "front-row starts": started_fr.groupby("era").size(),
    "podium % (started front row)": (100 * started_fr.groupby("era")["podium"].mean()).round(1),
    "podium % (qualified front row)": (100 * qualified_fr.groupby("era")["podium"].mean()).round(1),
})
display(overall)

MIN_STARTS = 8
by_start = started_fr.groupby(["circuit", "era"])["podium"].agg(["size", "mean"]).unstack("era")
by_quali = qualified_fr.groupby(["circuit", "era"])["podium"].mean().unstack("era")
keep = (by_start["size"].reindex(columns=ERAS) >= MIN_STARTS).all(axis=1)

rates = (100 * by_start["mean"].loc[keep, ERAS]).round(1)
q1_table = pd.concat({
    "podium % (started FR)": rates,
    "rank": rates.rank(ascending=False, method="min").astype(int),
    "podium % (qualified FR)": (100 * by_quali.loc[keep, ERAS]).round(1),
    "front-row starts": by_start["size"].loc[keep, ERAS].astype(int),
}, axis=1).sort_values(("podium % (started FR)", ERAS[1]), ascending=False)
display(q1_table)

rank_corr = rates[ERAS[0]].corr(rates[ERAS[1]], method="spearman")
print(f"{keep.sum()} circuits compared. Rank correlation between eras (Spearman): {rank_corr:.2f}  (1 = same order, 0 = unrelated)")
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2.2]})

x = np.arange(len(ERAS))
axes[0].bar(x - 0.2, overall["podium % (started front row)"], 0.4, label="started front row (grid)", color="#4C72B0")
axes[0].bar(x + 0.2, overall["podium % (qualified front row)"], 0.4, label="qualified front row", color="#DD8452")
for i, (a, b) in enumerate(zip(overall["podium % (started front row)"], overall["podium % (qualified front row)"])):
    axes[0].text(i - 0.2, a + 1, f"{a:.0f}%", ha="center", fontsize=9)
    axes[0].text(i + 0.2, b + 1, f"{b:.0f}%", ha="center", fontsize=9)
axes[0].set_xticks(x, [e.replace(" (", "\n(") for e in ERAS])
axes[0].set_ylim(0, 100)
axes[0].set_ylabel("% of front-row cars finishing on the podium")
axes[0].set_title("All circuits: front row -> podium")
axes[0].legend(fontsize=8, loc="lower right")

plot_rates = rates.sort_values(ERAS[1])
y = np.arange(len(plot_rates))
axes[1].hlines(y, plot_rates[ERAS[0]], plot_rates[ERAS[1]], color="#BBBBBB", zorder=1)
axes[1].scatter(plot_rates[ERAS[0]], y, color="#8C8C8C", label=ERAS[0], zorder=2)
axes[1].scatter(plot_rates[ERAS[1]], y, color="#4C72B0", label=ERAS[1], zorder=3)
axes[1].set_yticks(y, plot_rates.index)
axes[1].set_xlabel("% of front-row STARTERS finishing on the podium")
axes[1].set_title(f"Per circuit (>= {MIN_STARTS} front-row starts in each era), sorted by hybrid era")
axes[1].legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()
''')

md(r"""
**Answer to Q1.**
- **Overall:** a front-row start turns into a podium about **7 times out of 10**. It became **more reliable in the hybrid era** (≈ 68% → 76%): with dominant teams and more reliable cars, the drivers at the front usually stay there.
- **Most "converting" circuits in the hybrid era:** **Suzuka, Yas Marina (Abu Dhabi) and Monaco** (≈ 85–89%). Monaco makes sense because overtaking there is almost impossible.
- **Before 2014** the top three were **Monaco, Monza and Silverstone**.
- **The ranking did change a lot:** the rank correlation between the two eras is only about **0.2**, so a circuit's position in the list before 2014 says little about its position after. Monza and Hockenheim dropped, and Yas Marina and Interlagos rose sharply. Monaco stays near the top in both.
- **Qualifying position vs grid:** 36 front-row qualifiers started further back (grid penalties), and 36 others were promoted onto the front row. Counting by qualifying instead of grid changes the overall rate only slightly (71% vs 72%), but it changes some individual circuits, which is why both columns are shown.

*Caution:* each circuit has only ~10–24 front-row starts per era, so one race can move a circuit's rate by 5–10 points. Treat close rankings as ties.
""")

md(r"""
### 3.1.1 Robustness check: starting position only, all seasons (1950–2024)

`grid` exists for every season, so Q1 could be answered from grid alone, with "before 2014" meaning **1950–2013** instead of 2003–2013.
Does that give a different (or better) answer?
""")

code(r'''
ERAS_ALL = ["1950-2013 (all pre-hybrid)", "2014-2024 (hybrid)"]
q1_all = results_full_c.merge(circuits_c, on="circuitId", how="left", validate="many_to_one")
q1_all["era"] = np.where(q1_all["year"] >= 2014, ERAS_ALL[1], ERAS_ALL[0])
fr_all = q1_all[q1_all["grid"] <= 2]

by_decade = fr_all.assign(decade=fr_all["year"] // 10 * 10).groupby("decade")["podium"].agg(["size", "mean"])
by_all = fr_all.groupby(["circuit", "era"])["podium"].agg(["size", "mean"]).unstack("era")
keep_all = (by_all["size"].reindex(columns=ERAS_ALL) >= MIN_STARTS).all(axis=1)
rates_all = (100 * by_all["mean"].loc[keep_all, ERAS_ALL]).round(1)

pre_compare = pd.concat({"pre-2014 = 1950-2013": rates_all[ERAS_ALL[0]],
                         "pre-2014 = 2003-2013": rates[ERAS[0]]}, axis=1).dropna()
print(f"Front-row conversion 1950-2013: {100 * fr_all.loc[fr_all['era'] == ERAS_ALL[0], 'podium'].mean():.1f}%  "
      f"vs 2003-2013: {overall.loc[ERAS[0], 'podium % (started front row)']}%")
print(f"Rank correlation, 1950-2013 vs hybrid era: {rates_all[ERAS_ALL[0]].corr(rates_all[ERAS_ALL[1]], method='spearman'):.2f}")
print(f"Rank correlation between the two 'pre-2014' versions ({len(pre_compare)} common circuits): "
      f"{pre_compare.iloc[:, 0].corr(pre_compare.iloc[:, 1], method='spearman'):.2f}")

display(Markdown("**Top 5 circuits before 2014, under each definition**"))
display(pd.DataFrame({
    "1950-2013 (grid only, all years)": rates_all[ERAS_ALL[0]].sort_values(ascending=False).head(5).index,
    "2003-2013 (main answer)": rates[ERAS[0]].sort_values(ascending=False).head(5).index,
}))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
axes[0].bar(by_decade.index.astype(str) + "s", 100 * by_decade["mean"], color="#8C8C8C")
for i, (v, n) in enumerate(zip(by_decade["mean"], by_decade["size"])):
    axes[0].text(i, 100 * v + 1, f"{100 * v:.0f}%\n(n={n})", ha="center", fontsize=8)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel("% of front-row starters on the podium")
axes[0].set_title("Front row -> podium by decade (grid, all seasons)")

axes[1].scatter(pre_compare.iloc[:, 1], pre_compare.iloc[:, 0], color="#4C72B0")
for name, (v_all, v_modern) in pre_compare.iterrows():
    short = name.replace("Circuit de ", "").replace("Circuit ", "").replace("Autodromo ", "").replace("Autódromo ", "")[:18]
    axes[1].annotate(short, (v_modern, v_all), fontsize=7, xytext=(3, 2), textcoords="offset points")
lims = [45, 82]
axes[1].plot(lims, lims, ls="--", color="grey", lw=0.8)
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel("pre-2014 rate using 2003-2013 (%)")
axes[1].set_ylabel("pre-2014 rate using 1950-2013 (%)")
axes[1].set_title("Same circuit, two definitions of 'before 2014'")
plt.tight_layout()
plt.show()
''')

md(r"""
**What the robustness check shows.**
- **The conversion rate depends strongly on the decade:** about **46–54%** from the 1950s to the 1980s, rising to **74–75%** since 2010. Older cars broke down far more often, so even a front-row car often didn't finish.
- Because of this, **"before 2014" = 1950–2013 mostly measures how old a circuit is**:
  - Classic circuits (Monza, Silverstone, Monaco) carry decades of unreliable-era races, which pulls their rate down.
  - Newer circuits (Bahrain, Shanghai) only have modern races, so they look better.

  That's why the two "before 2014" rankings disagree (rank correlation about **−0.1**). The all-years version would also exaggerate the hybrid-era change (56% → 76% instead of 68% → 76%).
- **The hybrid-era (2014–2024) answer barely depends on this choice:** Suzuka, Yas Marina and Monaco stay at the top. Zandvoort also appears, because its 1950s–80s races let it pass the minimum-sample rule.
- **Decision:** the main answer keeps **2003–2013 vs 2014–2024**. Both periods are modern, comparable, and have qualifying data, which the question explicitly asks us to distinguish from grid. The all-years version is reported as a limitation: it compares eras more than circuits.
""")

md(r"""
### 3.1.2 Why the qualifying table is still needed (beyond the brief asking for it)

Grid is the starting position, so it's the primary measure. The question is whether **qualifying adds information that grid hides**.
""")

code(r'''
q_best = qualifying_c[["raceId", "driverId", "position", "q1_sec", "q2_sec", "q3_sec"]].copy()
q_best["best_sec"] = q_best[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)
q_best["gap_to_pole_pct"] = 100 * (q_best["best_sec"] / q_best.groupby("raceId")["best_sec"].transform("min") - 1)
qg = results_full_c[results_full_c["year"] >= 2003].merge(q_best[["raceId", "driverId", "position", "gap_to_pole_pct"]],
                                                      on=["raceId", "driverId"], how="left", validate="one_to_one")
print(f"Since 2003, grid differs from qualifying position in {100 * (qg['grid'] != qg['position'])[qg['position'].notna()].mean():.0f}% of rows.")

# (1) Penalised fast cars: qualified top 3, started 6th or worse
penalised = qg[(qg["position"] <= 3) & (qg["grid"] >= 6)]
same_slots = qg[qg["grid"].isin(penalised["grid"].unique()) & (qg["position"] > 3)]
penalty_view = pd.Series({
    f"qualified top 3, started 6th+ (n={len(penalised)})": 100 * penalised["podium"].mean(),
    f"others starting from the same slots (n={len(same_slots)})": 100 * same_slots["podium"].mean(),
}).round(1)

# (2) Same grid slot, faster vs slower half by gap to pole
gap_rows = []
for slot in [2, 3, 4, 5]:
    s = qg[(qg["grid"] == slot) & qg["gap_to_pole_pct"].notna()]
    fast = s["gap_to_pole_pct"] <= s["gap_to_pole_pct"].median()
    gap_rows.append({"grid": slot, "faster half": 100 * s.loc[fast, "podium"].mean(),
                     "slower half": 100 * s.loc[~fast, "podium"].mean(), "n": len(s)})
gap_view = pd.DataFrame(gap_rows).set_index("grid").round(1)
display(penalty_view.to_frame("podium %"))
display(gap_view)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].barh(penalty_view.index, penalty_view.values, color=["#C44E52", "#8C8C8C"])
for i, v in enumerate(penalty_view.values):
    axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center")
axes[0].set_xlabel("podium %")
axes[0].set_title("Same starting slots, different qualifying result")
x = np.arange(len(gap_view))
axes[1].bar(x - 0.2, gap_view["faster half"], 0.4, label="faster half (smaller gap to pole)", color="#4C72B0")
axes[1].bar(x + 0.2, gap_view["slower half"], 0.4, label="slower half", color="#8C8C8C")
axes[1].set_xticks(x, [f"grid {g}" for g in gap_view.index])
axes[1].set_ylabel("podium %")
axes[1].set_title("Same grid slot, split by qualifying pace")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
''')

md(r"""
**What this shows.**
- **Penalties hide fast cars.** Drivers who qualified in the **top 3** but started **6th or worse** (grid penalties) still reached the podium about **32%** of the time. Other drivers starting from the same slots managed only about **3%**. Using grid alone, those fast cars look like midfield cars.
- **Qualifying shows how fast a car is, not just its order.** Among drivers starting from the same slot, the faster half (smaller gap to pole) reaches the podium more often from grid 2, 4 and 5 (e.g. **27% vs 15%** from grid 5). Grid 3 is the exception, with no difference. Grid is only a rank, while qualifying times measure the gap.
- **So we use both:** `grid` as the starting position (the main measure, all seasons), and qualifying for the information grid loses. Since 2003, grid and qualifying position differ in about a quarter of rows.
- **Planned modeling experiment:** *Attempt A* = grid only, all seasons vs *Attempt B* = grid + qualifying, 2003+. Both are evaluated on the same recent test seasons. If A is as good as B, the simpler data choice wins. This comparison will be reported in the modeling section.
""")

# ---------------------------------------------------------------- Q2
md(r"""
## 3.2 Q2: Do drivers finish on the podium more often at home, once we account for where they start?

**In simple words:** a British driver at Silverstone, a Dutch driver at Zandvoort... do they do better than usual? To be fair, we must compare drivers who **started from similar places**. Otherwise "home advantage" could just mean "the home driver qualified well".

- **Tables & joins:** `clean_data/results_full.csv` ⟶ `clean_data/drivers.csv` on `driverId` (`home_country`, built in 2.4) ⟶ `clean_data/circuits.csv` on `circuitId` (`country`). **Home race** = `home_country == circuit country`.
- **Grain:** one row = one driver starting one race (all seasons; non-starters were removed in 2.6, and the Indianapolis 500 in 2.6.1).
- **Why no Indianapolis 500:** in 1950–1960 it was a separate American race. Almost every starter was an American at a US circuit (a "home" start), with no away drivers in the same race to compare with. It would have made up 16% of all home starts. The result with Indy put back is shown as a robustness column. Those rows are removed by cleaning, so they are not in the clean files: this one column takes them from memory (`INDY500_RESULTS`, 2.6.1).
- **Controlling for starting position:** drivers are grouped into **grid bands** (1–3, 4–6, 7–10, 11+). Home and away drivers are compared **inside the same band**. The band differences are then averaged, weighted by how many home starts fall in each band ("grid-adjusted difference").
- **Uncertainty:** a bootstrap (resampling the rows 500 times) gives a 95% interval for the adjusted difference.
- **Missingness:** none. Every driver has a `home_country`, and every cleaned row has a valid grid.
- **Cutoff:** grid is pre-race information, so it's a valid control.
""")

code(r'''
nat = drivers_c[["driverId", "home_country"]]
GRID_BANDS = ["1-3", "4-6", "7-10", "11+"]


def home_table(res):
    df = (res.merge(nat, on="driverId", how="left", validate="many_to_one")
          .merge(circuits_c, on="circuitId", how="left", validate="many_to_one"))
    df["home"] = df["home_country"] == df["country"]
    df["grid_band"] = pd.cut(df["grid"].astype(float), [0, 3, 6, 10, 99], labels=GRID_BANDS)
    return df


q2 = home_table(results_full_c)
# Robustness only: put back the Indianapolis 500 rows removed in 2.6.1.
q2_with_indy = home_table(pd.concat([results_full_c, INDY500_RESULTS.merge(race_info_c, on="raceId", how="left")]))


def adjusted_home_diff(df):
    """Home minus away podium rate inside each grid band, averaged with weights = home starts per band."""
    t = df.groupby(["grid_band", "home"], observed=True)["podium"].agg(["mean", "size"]).unstack("home")
    diff = t[("mean", True)] - t[("mean", False)]
    weights = t[("size", True)]
    return (diff * weights).sum() / weights.sum()


def summarize(df):
    rng = np.random.default_rng(42)
    boot = [adjusted_home_diff(df.sample(frac=1, replace=True, random_state=rng)) for _ in range(500)]
    return {
        "home starts": int(df["home"].sum()),
        "podium % home": round(100 * df.loc[df["home"], "podium"].mean(), 1),
        "podium % away": round(100 * df.loc[~df["home"], "podium"].mean(), 1),
        "avg grid home": round(df.loc[df["home"], "grid"].mean(), 1),
        "avg grid away": round(df.loc[~df["home"], "grid"].mean(), 1),
        "raw difference (pts)": round(100 * (df.loc[df["home"], "podium"].mean() - df.loc[~df["home"], "podium"].mean()), 2),
        "grid-adjusted difference (pts)": round(100 * adjusted_home_diff(df), 2),
        "95% interval (pts)": f"[{100 * np.percentile(boot, 2.5):.2f}, {100 * np.percentile(boot, 97.5):.2f}]",
    }


q2_summary = pd.DataFrame({"all seasons (1950-2024)": summarize(q2),
                           "2003-2024 only": summarize(q2[q2["year"] >= 2003]),
                           "robustness: + Indianapolis 500": summarize(q2_with_indy)})
display(q2_summary)

band_table = (q2.groupby(["grid_band", "home"], observed=True)["podium"].agg(["mean", "size"]).unstack("home"))
band_table = pd.DataFrame({
    "podium % away": (100 * band_table[("mean", False)]).round(1),
    "podium % home": (100 * band_table[("mean", True)]).round(1),
    "away starts": band_table[("size", False)],
    "home starts": band_table[("size", True)],
})
display(band_table)
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2, 1]})
x = np.arange(len(GRID_BANDS))
axes[0].bar(x - 0.2, band_table["podium % away"], 0.4, label="away", color="#8C8C8C")
axes[0].bar(x + 0.2, band_table["podium % home"], 0.4, label="home race", color="#C44E52")
for i, (a, h) in enumerate(zip(band_table["podium % away"], band_table["podium % home"])):
    axes[0].text(i - 0.2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
    axes[0].text(i + 0.2, h + 1, f"{h:.1f}", ha="center", fontsize=8)
axes[0].set_xticks(x, [f"grid {b}" for b in GRID_BANDS])
axes[0].set_ylabel("podium %")
axes[0].set_title("Podium rate by starting position: home vs away (all seasons)")
axes[0].legend()

s = q2_summary["all seasons (1950-2024)"]
vals = [s["raw difference (pts)"], s["grid-adjusted difference (pts)"]]
axes[1].bar(["raw\n(ignores grid)", "grid-adjusted"], vals, color=["#DD8452", "#4C72B0"])
axes[1].axhline(0, color="black", lw=0.8)
for i, v in enumerate(vals):
    axes[1].text(i, v / 2, f"{v:+.2f}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
axes[1].set_ylabel("home minus away (percentage points)")
axes[1].set_title("Home effect")
plt.tight_layout()
plt.show()
''')

md(r"""
**Answer to Q2: No, there is no home advantage once starting position is taken into account.**
- **Without controlling for grid**, home drivers even look *worse* (≈ 11.6% vs 13.7% podiums). That's misleading, because home drivers start further back on average (grid ≈ 12.7 vs 11.7). In older seasons many local drivers entered only their home race in weaker cars.
- **Comparing drivers who started from the same grid band**, the home rate is almost identical to the away rate in **every** band: slightly higher at the front (grid 1–3: ≈ 55.8% home vs 55.4% away; grid 4–6: 26.5% vs 26.0%) and slightly lower further back (grid 7–10: 7.2% vs 9.3%).
- The **grid-adjusted difference is close to zero** (about −0.6 points), and the 95% interval includes 0, so the small gap is consistent with chance. The modern era (2003+) points the same way, and so does the robustness version with the Indianapolis 500 put back (−0.59 points, interval also includes 0).
- **Takeaway:** what looks like a "home effect" is really a **starting-position effect**. Where you start matters a lot; whether you're at home doesn't. For the model this suggests a "home race" feature will add little.

*Note:* controlling with grid bands is a simple, transparent method. Within a band, small grid differences remain, but the per-band rates are so close that a finer control wouldn't change the conclusion.
""")

# ---------------------------------------------------------------- Q3
md(r"""
## 3.3 Q3: Which constructors had the highest mechanical-retirement rate in 2014–2021 vs 2022–2024?

**In simple words:** out of all the races a team's cars started, how often did one stop because something on the car broke?

- **Data source:** this question reads the **cleaned CSVs directly from the clean-data folder** (`outputs/clean_data/`, written in 2.13, loaded at the start of section 3), not the raw files and not the in-memory tables. `results_full.csv` there already has the cleaning applied: non-starters, duplicate rows and the Indianapolis 500 removed, and every status mapped to a `status_category` (2.5–2.6).
- **Tables & joins:** `clean_data/results_full.csv` ⟶ `clean_data/races.csv` on `raceId` (for `year`) ⟶ `clean_data/constructors.csv` on `constructorId` (team name).
- **Grain:** one row = one car starting one race. **Rate = mechanical retirements ÷ race starts** for that team in that period.
- **Mechanical retirement:** `status_category == "mechanical"` (engine, gearbox, hydraulics, power unit, ...; 2.5). Crashes, driver problems and regulation decisions are excluded, as the brief defines. Statuses whose cause is unclear (`Retired`, `Wheel`, `Puncture`, `Damage`, ...) are excluded from the main (**strict**) rate and added in the **broad** rate (mechanical + unclear, an upper bound).
- **Missingness:** none. Every status has a category, and non-starters were removed, so the denominator counts real starts only (checked below).
- **Small teams:** only teams with **≥ 40 starts** in a period are ranked. Team names follow the dataset: renamed teams (e.g. Toro Rosso → AlphaTauri → RB, Force India → Racing Point → Aston Martin) appear as separate entries.
- **Cutoff:** not relevant. This is a historical post-race analysis, not a feature.
""")

code(r'''
# Same clean-data folder as the rest of section 3.
q3_results = results_full_c[["raceId", "driverId", "constructorId", "status", "status_category"]]
q3_races = races_c[["raceId", "year"]]
q3_teams = constructors_c

# Checks on what we loaded: unique grain, no missing categories, every key found.
assert not q3_results.duplicated(["raceId", "driverId"]).any()
assert q3_results["status_category"].notna().all()
assert q3_results["raceId"].isin(q3_races["raceId"]).all()
assert q3_results["constructorId"].isin(q3_teams["constructorId"]).all()

PERIODS = ["2014-2021", "2022-2024"]
q3 = (q3_results
      .merge(q3_races, on="raceId", how="left", validate="many_to_one")
      .merge(q3_teams, on="constructorId", how="left", validate="many_to_one"))
q3 = q3[q3["year"] >= 2014].copy()
q3["period"] = np.where(q3["year"] <= 2021, PERIODS[0], PERIODS[1])
q3["mechanical"] = q3["status_category"] == "mechanical"
q3["mechanical_or_unclear"] = q3["status_category"].isin(["mechanical", "unclear"])
print(f"Race starts 2014-2024: {len(q3):,}")

MIN_TEAM_STARTS = 40
q3_rates = (q3.groupby(["period", "constructor"])
            .agg(starts=("mechanical", "size"), mechanical_DNFs=("mechanical", "sum"),
                 mech_rate=("mechanical", "mean"), upper_bound=("mechanical_or_unclear", "mean"))
            .reset_index())
q3_rates = q3_rates[q3_rates["starts"] >= MIN_TEAM_STARTS]
q3_rates[["mech_rate", "upper_bound"]] = (100 * q3_rates[["mech_rate", "upper_bound"]]).round(1)
q3_rates = q3_rates.rename(columns={"mech_rate": "mechanical %", "upper_bound": "mechanical + unclear %"})

print("Field-wide mechanical-retirement rate:",
      {p: f"{100 * q3.loc[q3['period'] == p, 'mechanical'].mean():.1f}%" for p in PERIODS})
for p in PERIODS:
    display(Markdown(f"**{p}: top 5 constructors**"))
    display(q3_rates[q3_rates["period"] == p].sort_values("mechanical %", ascending=False)
            .head(5).drop(columns="period").reset_index(drop=True))
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, p in zip(axes, PERIODS):
    d = q3_rates[q3_rates["period"] == p].sort_values("mechanical %")
    ax.barh(d["constructor"], d["mechanical + unclear %"], color="#F2C6A0", label="broad: + unclear causes (upper bound)")
    ax.barh(d["constructor"], d["mechanical %"], color="#C44E52", label="strict: mechanical")
    for yy, (v, ub, n) in enumerate(zip(d["mechanical %"], d["mechanical + unclear %"], d["starts"])):
        ax.text(max(v, ub) + 0.3, yy, f"{v:.1f}%  (n={n})", va="center", fontsize=8)
    field = 100 * q3.loc[q3["period"] == p, "mechanical"].mean()
    ax.axvline(field, color="black", ls="--", lw=0.8, label=f"field average {field:.1f}%")
    ax.set_title(f"Mechanical retirements per start, {p}")
    ax.set_xlabel("% of race starts")
    ax.set_xlim(0, 27)
    ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()
''')

md(r"""
**Answer to Q3.**
- **2014–2021 (first hybrid engines):** the least reliable teams were **Lotus F1 (≈ 17% of starts ended with a car failure), Toro Rosso (≈ 16%) and McLaren (≈ 13.5%, tied with Renault)**. McLaren's years with the early Honda engine (2015–2017) are a well-known cause.
- **2022–2024 (new car rules):** the highest rates belong to **Alfa Romeo (≈ 12.5%) and Alpine (≈ 11%)**, then AlphaTauri (≈ 7%).
- **Big picture:** cars became **much more reliable**. The field-wide mechanical-retirement rate fell from about **9% to 5.5%** of starts, and the top teams in 2022–2024 (Red Bull, McLaren, Mercedes) retired from mechanical failures in only about 4% of starts.
- **Strict vs broad (2.5):** the strict ranking counts only *mechanical* statuses (dark bars). The broad ranking also counts *unclear* ones (`Retired`, `Wheel`, `Puncture`, ...; light bars). That raises some rates by up to ~7 points but doesn't change the top two in either period.

*Caution:* in 2022–2024 each team has at most ~136 starts, so one extra failure moves the rate by ~0.7 points. The differences between the middle teams are small.
""")

# ================================================================ 4. FEATURE SELECTION & ENGINEERING
md(r"""
# 4. Feature Selection & Engineering

**Goal:** choose the inputs the models will use, with evidence for each choice. Every feature must be **known after qualifying and before the race starts**.

How this section works:
1. **4.1 Split** the seasons into train / validation / test first, so every later decision can be made **without looking at the test seasons**.
2. **4.2 Build** the candidate features in five groups (grid, qualifying, standings, form, context), with checks that none of them uses the current race's result.
3. **4.3 Evidence:** one plot per group, on the **training seasons only**.
4. **4.4 Redundancy:** features that say almost the same thing; which one we keep and why.
5. **4.5 Trials:** a simple logistic regression, starting from grid only and adding one group at a time, scored on the **validation** seasons. Every trial is kept.
6. **4.6 Final feature set**, with the reason each feature is allowed.

The **test seasons (2022–2024) are not used** for any plot, correlation or decision in this section.
""")

# ---------------------------------------------------------------- 4.1
md(r"""
## 4.1 Train / validation / test split

We split **by season**, never randomly:

| Set | Model A (2003+) | Model B (all years) |
|---|---|---|
| Train | 2003–2018 | 1950–2018 (missing qualifying kept as NaN + a flag) |
| Validation | 2019–2021 | 2019–2021 |
| Test | 2022–2024 | 2022–2024 |

**Why by season:**
- **No peeking at the future.** Form and standings features look back in time. With a random split, the model would train on 2023 races and be tested on 2015 races, learning from the future.
- **A race is never split.** In each race exactly 3 drivers get a podium, so the rows of one race depend on each other. Splitting by season keeps every race whole inside one set.
- **An honest test.** 2022 brought completely new car rules, so 2022–2024 tests whether the model still works after a big change, as it would in real use.
- **Validation** (2019–2021) is the most recent period before the test. It's used to compare feature sets now, and to tune models and thresholds later.
""")

code(r'''
SPLIT_YEARS = {"train": (2003, 2018), "validation": (2019, 2021), "test": (2022, 2024)}
MODEL_B_TRAIN_START = 1950


def season_split(year):
    year = np.asarray(year, dtype=int)
    return np.select([year <= 2018, year <= 2021], ["train", "validation"], "test")


split_summary = []
for model, start in [("A (2003+)", MODEL_START_YEAR), ("B (all years)", MODEL_B_TRAIN_START)]:
    rows = results_full[results_full["year"] >= start]
    sp = season_split(rows["year"])
    for s in ["train", "validation", "test"]:
        part = rows[sp == s]
        split_summary.append({"model": model, "set": s, "seasons": f"{part['year'].min()}-{part['year'].max()}",
                              "races": part["raceId"].nunique(), "rows": len(part),
                              # the test podium rate is deliberately not looked at
                              "podium rate %": round(100 * part["podium"].mean(), 1) if s != "test" else "not inspected"})
split_summary = pd.DataFrame(split_summary)
display(split_summary)

# A race can only belong to one set (a race lies inside one season).
race_sets = results_full.assign(set=season_split(results_full["year"])).groupby("raceId")["set"].nunique()
assert (race_sets == 1).all(), "a race is split across sets"
print("Every race belongs to exactly one set.")
''')

md(r"""
**What this shows.** Model A trains on 16 seasons (6,381 rows) and model B on 69 (22,092 rows); both are validated on the same 3 seasons and will be tested on the same 3 seasons, so their scores can be compared directly. The training podium rate of model A is close to the validation rate, so the class balance is similar in both. The test podium rate is intentionally not printed.
""")

# ---------------------------------------------------------------- 4.2
md(r"""
## 4.2 Building the candidate features

All features are computed for **every season** (1950–2024), because model B needs the old seasons and form features need each driver's earlier races. Building a feature for the test seasons is fine: it uses only information from before each race. Choosing features is what must not look at the test seasons.

| Group | Features | Known before the race because... |
|---|---|---|
| **Grid** | `grid`, `pit_lane_start` | the starting order is fixed before the start |
| **Qualifying** | `quali_pos`, `gap_q1_pct` (gap to the fastest Q1 time, %), `grid_minus_quali` (penalty indicator), `reached_q3` (advanced to Q3: a Q3 time, **or** a Q2 time ranked inside that race's Q3 places; + `no_q3_format` flag before 2006), `quali_missing` flag | qualifying is finished |
| **Standings** | driver and constructor championship **position** and **share of the leader's points** (from the **previous** race), `no_standing_yet` flags. No standing yet = behind everyone ranked with 0% of the leader's points; in round 1 nobody is ranked, so everyone is set to the back (number of starters) | the previous race's standings are known; never the current race's (2.10.2) |
| **Form** | driver podium rate, average finish and retirement rate over the **last 5 races**; constructor podium rate over its last 5 races, **restarted after a long absence** (see the checks below) | computed with `shift(1)`, so only earlier races are used |
| **Context** | `n_starters`, `drv_experience` (career starts so far), `drv_age`, `home_race` | known from the entry list, the calendar and the driver's birth date |
""")

code(r'''
# Race order over all seasons: (year, round) -> 0, 1, 2, ...
race_seq = CLEAN["races"].sort_values(["year", "round"])[["raceId"]].reset_index(drop=True)
race_seq["race_seq"] = np.arange(len(race_seq))

feat = (results_full[["raceId", "driverId", "constructorId", "circuitId", "year", "round", "date", "grid",
                      "pit_lane_start", "podium"]]
        .merge(race_seq, on="raceId", how="left", validate="many_to_one"))
feat["split"] = season_split(feat["year"])

# --- Grid ------------------------------------------------------------------------------------------
feat["grid"] = feat["grid"].astype(float)
feat["pit_lane_start"] = feat["pit_lane_start"].astype(float)

# --- Qualifying --------------------------------------------------------------------------------------
q = CLEAN["qualifying"][["raceId", "driverId", "position", "q1_sec", "q2_sec", "q3_sec"]].copy()
q["gap_q1_pct"] = 100 * (q["q1_sec"] / q.groupby("raceId")["q1_sec"].transform("min") - 1)
q["best_sec"] = q[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)          # only for the trial in 4.5
q["gap_best_pct"] = 100 * (q["best_sec"] / q.groupby("raceId")["best_sec"].transform("min") - 1)

# reached_q3 = the driver advanced to Q3. A Q3 lap time is not the only evidence: drivers who advanced but set no
# Q3 time (engine-change penalty, crash, tyre saving, Q3 cancelled) still advanced.
# Rule: every driver with a Q3 time reached Q3; the Q3 places left over go to the fastest Q2 times among the
# drivers WITHOUT a Q3 time (ties broken by qualifying position). Anchoring on Q3 times matters: in a few races the
# recorded Q2 time isn't the lap that counted, so a plain Q2 ranking would let an extra driver in.
# Qualifying position alone isn't used, because it sometimes already includes a penalty (2007 Belgian/Japanese GP).
# Q3 places: 10 in the knockout format (2006+), except the 2016 Australian and Bahrain GP (elimination format: 8).
ELIMINATION_FORMAT = CLEAN["races"].query("year == 2016 and name in ['Australian Grand Prix', 'Bahrain Grand Prix']")["raceId"]
assert len(ELIMINATION_FORMAT) == 2
q["q3_places"] = np.where(q["raceId"].isin(ELIMINATION_FORMAT), 8, 10)
q["q3_times"] = q.groupby("raceId")["q3_sec"].transform("count")
no_q3_time = q[q["q3_sec"].isna() & q["q2_sec"].notna()].sort_values(["raceId", "q2_sec", "position"])
q["q2_order_no_q3"] = no_q3_time.groupby("raceId").cumcount() + 1     # 1 = fastest Q2 among drivers without a Q3 time
q["reached_q3_time_only"] = q["q3_sec"].notna().astype(float)          # the old definition, kept only for the check in 4.2.1
q["reached_q3"] = (q["q3_sec"].notna() | (q["q2_order_no_q3"] <= q["q3_places"] - q["q3_times"])).astype(float)
feat = feat.merge(q[["raceId", "driverId", "position", "gap_q1_pct", "gap_best_pct", "reached_q3"]]
                  .rename(columns={"position": "quali_pos"}), on=["raceId", "driverId"], how="left", validate="one_to_one")
feat["quali_pos"] = feat["quali_pos"].astype(float)
feat["quali_missing"] = feat["quali_pos"].isna().astype(float)
feat["grid_minus_quali"] = feat["grid"] - feat["quali_pos"]
feat["no_q3_format"] = (feat["year"] < 2006).astype(float)          # the Q1/Q2/Q3 knockout started in 2006
feat.loc[feat["no_q3_format"] == 1, "reached_q3"] = np.nan          # not defined before 2006

# --- Standings (previous race only) ---------------------------------------------------------------------


def prerace_features(table, key, prefix):
    s = CLEAN[table][["raceId", key, "pre_points", "pre_position"]].copy()
    leader = s.groupby("raceId")["pre_points"].transform("max")
    s[f"{prefix}_points_share"] = np.where(leader > 0, s["pre_points"] / leader, 0.0)
    s[f"{prefix}_champ_pos"] = s["pre_position"].astype(float)
    out = feat[["raceId", key]].merge(s[["raceId", key, f"{prefix}_points_share", f"{prefix}_champ_pos"]],
                                      on=["raceId", key], how="left", validate="many_to_one")
    ranked = s.groupby("raceId")["pre_position"].max().astype(float)
    out[f"{prefix}_no_standing_yet"] = out[f"{prefix}_champ_pos"].isna().astype(float)
    # No standing yet: behind everyone ranked, 0% of the leader's points. In round 1 nobody is ranked yet,
    # so every driver is set to the back (= number of starters), not to position 1.
    starters = feat.groupby("raceId")["driverId"].transform("size").to_numpy()
    behind = (out["raceId"].map(ranked) + 1).fillna(pd.Series(starters, index=out.index).astype(float))
    out[f"{prefix}_champ_pos"] = out[f"{prefix}_champ_pos"].fillna(behind)
    out[f"{prefix}_points_share"] = out[f"{prefix}_points_share"].fillna(0.0)
    return out.drop(columns=["raceId", key])


feat = pd.concat([feat, prerace_features("driver_standings_prerace", "driverId", "drv"),
                  prerace_features("constructor_standings_prerace", "constructorId", "cons")], axis=1)

# --- Form (earlier races only) -------------------------------------------------------------------------
FORM_WINDOW = 5
hist = (results_full[["raceId", "driverId", "constructorId", "positionOrder", "podium", "status_category"]]
        .merge(race_seq, on="raceId"))
hist["finish"] = hist["positionOrder"].astype(float)
hist["dnf"] = (hist["status_category"] != "finished").astype(float)
hist["podium_f"] = hist["podium"].astype(float)


def past_mean(values, keys, n):
    """Mean of the previous n values within each key, excluding the current row (shift(1))."""
    return values.groupby(keys).transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())


# Constructor history segments. A constructorId can come back after decades (Mercedes 1955 -> 2010), so its
# "previous races" would describe another era's team. A gap of more than TEAM_GAP_DAYS between two races of the
# same constructor starts a new segment, and team form only looks back inside the current segment.
TEAM_GAP_DAYS = 550
TEAM_RACES = (hist.groupby(["constructorId", "raceId", "race_seq"])["podium_f"].agg(["sum", "size"])
              .reset_index().sort_values(["constructorId", "race_seq"]))
TEAM_RACES["date"] = TEAM_RACES["raceId"].map(CLEAN["races"].set_index("raceId")["date"])
TEAM_RACES["gap_days"] = TEAM_RACES.groupby("constructorId")["date"].diff().dt.days
TEAM_RACES["team_segment"] = (TEAM_RACES["gap_days"] > TEAM_GAP_DAYS).groupby(TEAM_RACES["constructorId"]).cumsum()


def form_features(n, reset_gap=True):
    h = hist.sort_values(["driverId", "race_seq"])
    out = pd.DataFrame({"raceId": h["raceId"], "driverId": h["driverId"]})
    out[f"drv_podium_rate_{n}"] = past_mean(h["podium_f"], h["driverId"], n)
    out[f"drv_avg_finish_{n}"] = past_mean(h["finish"], h["driverId"], n)
    out[f"drv_dnf_rate_{n}"] = past_mean(h["dnf"], h["driverId"], n)
    # Constructor: podium finishes / car-starts over its previous n races (both cars of a race count),
    # within the current history segment (reset_gap=False is the old version, kept only for the check in 4.2).
    team = TEAM_RACES.copy()
    keys = ["constructorId", "team_segment"] if reset_gap else ["constructorId"]
    roll = lambda col: team.groupby(keys)[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    team[f"cons_podium_rate_{n}"] = roll("sum") / roll("size")
    out = out.merge(feat[["raceId", "driverId", "constructorId"]], on=["raceId", "driverId"])
    out = out.merge(team[["constructorId", "raceId", f"cons_podium_rate_{n}"]], on=["constructorId", "raceId"], how="left")
    return out.drop(columns="constructorId")


form5 = form_features(FORM_WINDOW)
feat = feat.merge(form5, on=["raceId", "driverId"], how="left", validate="one_to_one")

# --- Context ----------------------------------------------------------------------------------------------
feat["n_starters"] = feat.groupby("raceId")["driverId"].transform("size").astype(float)
exp = hist.sort_values(["driverId", "race_seq"])
exp["drv_experience"] = exp.groupby("driverId").cumcount().astype(float)          # starts BEFORE this race
feat = feat.merge(exp[["raceId", "driverId", "drv_experience"]], on=["raceId", "driverId"], validate="one_to_one")
feat = feat.merge(CLEAN["drivers"][["driverId", "dob", "home_country"]], on="driverId", how="left")
feat["drv_age"] = (feat["date"] - feat["dob"]).dt.days / 365.25
feat = feat.merge(CLEAN["circuits"][["circuitId", "country"]], on="circuitId", how="left")
feat["home_race"] = (feat["home_country"] == feat["country"]).astype(float)
feat = feat.drop(columns=["dob", "home_country", "country"]).sort_values(["race_seq", "grid"]).reset_index(drop=True)

FEATURE_GROUPS = {
    "grid": ["grid", "pit_lane_start"],
    "qualifying": ["quali_pos", "gap_q1_pct", "grid_minus_quali", "reached_q3", "no_q3_format", "quali_missing"],
    "standings": ["drv_champ_pos", "drv_points_share", "drv_no_standing_yet",
                  "cons_champ_pos", "cons_points_share", "cons_no_standing_yet"],
    "form": [f"drv_podium_rate_{FORM_WINDOW}", f"drv_avg_finish_{FORM_WINDOW}", f"drv_dnf_rate_{FORM_WINDOW}",
             f"cons_podium_rate_{FORM_WINDOW}"],
    "context": ["n_starters", "drv_experience", "drv_age", "home_race"],
}
ALL_CANDIDATES = [c for cols in FEATURE_GROUPS.values() for c in cols]
print(f"Feature table: {len(feat):,} rows (1950-2024), {len(ALL_CANDIDATES)} candidate features in {len(FEATURE_GROUPS)} groups")
display(feat[ALL_CANDIDATES].describe().T.round(2))
''')

md(r"""
**Leakage checks.** The form features are the riskiest, because they are built from race results. The checks below prove they only use **earlier** races:
- every driver's **first** race has no form value, and so does every constructor's first race of a history segment (its first race ever, or its first race after a long absence, see 4.2.1);
- for 500 random rows, the form value is recomputed by hand from the races **strictly before** that race (for the team: inside the same history segment), and it must match exactly;
- no post-race column (finishing position, points, status, ...) is a candidate feature.
""")

code(r'''
form_cols = FEATURE_GROUPS["form"]
by_driver = feat.sort_values(["driverId", "race_seq"])
first_drv = by_driver.groupby("driverId").head(1)
assert first_drv[[c for c in form_cols if c.startswith("drv_")]].isna().all().all(), "a driver's first race has a form value"
# First race of each constructor history segment (its first race ever, or its first race after a long absence).
seg_start = TEAM_RACES.loc[TEAM_RACES.groupby(["constructorId", "team_segment"]).cumcount() == 0, ["constructorId", "raceId"]]
first_cons = feat.merge(seg_start, on=["constructorId", "raceId"])
assert first_cons[f"cons_podium_rate_{FORM_WINDOW}"].isna().all(), "a constructor's first race (of a segment) has a form value"
segment_of = TEAM_RACES.set_index(["constructorId", "raceId"])["team_segment"]

rng = np.random.default_rng(0)
sample = feat.iloc[rng.choice(len(feat), 500, replace=False)]
for r in sample.itertuples():
    past = hist[(hist["driverId"] == r.driverId) & (hist["race_seq"] < r.race_seq)].sort_values("race_seq").tail(FORM_WINDOW)
    expected = past["podium_f"].mean() if len(past) else np.nan
    got = getattr(r, f"drv_podium_rate_{FORM_WINDOW}")
    assert (np.isnan(expected) and np.isnan(got)) or np.isclose(expected, got), f"form leak at raceId {r.raceId}"
    same_segment = TEAM_RACES[(TEAM_RACES["constructorId"] == r.constructorId)
                              & (TEAM_RACES["team_segment"] == segment_of.loc[(r.constructorId, r.raceId)])]
    team_past = hist[(hist["constructorId"] == r.constructorId) & (hist["race_seq"] < r.race_seq)
                     & hist["raceId"].isin(same_segment["raceId"])]
    last_races = team_past["race_seq"].drop_duplicates().nlargest(FORM_WINDOW)
    tp = team_past[team_past["race_seq"].isin(last_races)]
    expected_c = tp["podium_f"].mean() if len(tp) else np.nan
    got_c = getattr(r, f"cons_podium_rate_{FORM_WINDOW}")
    assert (np.isnan(expected_c) and np.isnan(got_c)) or np.isclose(expected_c, got_c), f"team form leak at raceId {r.raceId}"

assert not set(ALL_CANDIDATES) & set(POST_RACE), "a post-race column is a candidate feature"
print(f"Leakage checks passed: {len(first_drv)} first driver races and {len(first_cons)} first constructor races "
      f"(including restarts after a long absence) have no form; "
      f"500 random rows match a hand recomputation from earlier races only; no post-race column is a feature.")
''')

md(r"""
### 4.2.1 Checks for two corrected features: `reached_q3` and team form

A review of the first version found two features that were computed wrongly for a small number of rows. Both are fixed above; the cells below show what changed and prove the fixes.

**`reached_q3`.** The first version used "has a Q3 lap time". But some drivers **advanced to Q3 without setting a Q3 time** (an engine-change penalty made a lap pointless, a crash, saving tyres, or Q3 was cancelled), so they were wrongly marked as eliminated. The corrected rule: every driver **with a Q3 time** reached Q3, and the **Q3 places left over** go to the fastest Q2 times among the drivers without a Q3 time. Format exceptions and data quirks:
- **Q3 places = 10** in the knockout format (2006 onwards), except the **2016 Australian and Bahrain GP**: the short-lived "elimination" format sent only **8** drivers to Q3, so a plain "qualified in the top 10" rule would be wrong there.
- **Not the qualifying position:** in a few races the recorded position already includes a penalty (e.g. 2007 Belgian GP: Kubica is listed 14th although he ran in Q3), so it doesn't show who advanced.
- **Anchored on Q3 times, not a plain Q2 ranking:** in a few races the recorded Q2 time isn't the lap that counted (e.g. Heidfeld, 2008 Turkish GP, listed 14th in Q2 but ran in Q3), so ranking Q2 times alone would let an 11th driver into Q3. Equal Q2 times at the cut-off (2011 Indian GP) are separated by qualifying position.
- **2015 US GP:** Q3 was cancelled by rain (no Q3 times at all); the 10 drivers who advanced from Q2 count as having reached Q3.
- Before 2006 there was no Q3: `reached_q3` stays missing and `no_q3_format = 1`, as before.
- **Check:** in every one of the 375 knockout races, exactly 10 drivers (8 in the two elimination-format races) reached Q3.

**Team form.** A `constructorId` can return after decades: Mercedes raced in 1954–1955 and came back in 2010 with the same ID, so its first 2010 races were using 1955 results as "recent form". Team form now only looks back inside a **history segment**: a new segment starts when the team's previous race was more than **550 days** (about 18 months) earlier.
- **Why 550 days:** a normal winter break is 3–7 months. Small 1950s–60s teams that entered only a few races a season can go about a year between races while still racing every season. More than 18 months means the team missed at least a whole season. In 2003+ **every** reset is a gap of 4 years or more, so any threshold between 1 and 2 years gives exactly the same model-A features (the sensitivity table below shows it); the choice only affects a few pre-2003 rows used by model B.
- The first race of a new segment has **no** team form (missing, like a new team); from its second race the form uses the races available in the segment, up to 5. The formula is unchanged: podium finishes divided by car-starts over those races.
- **Limitation (documented, not changed):** renamed teams have different IDs (Toro Rosso → AlphaTauri → RB, Force India → Racing Point → Aston Martin, Renault → Alpine, ...). Their form starts again after a rename even though the factory is the same. We don't merge IDs: which renames are "the same team" is a judgement call, and these rows are few.
""")

code(r'''
nm = (CLEAN["drivers"].assign(driver=CLEAN["drivers"]["forename"] + " " + CLEAN["drivers"]["surname"])[["driverId", "driver"]])
rc = CLEAN["races"][["raceId", "name"]].rename(columns={"name": "race"})

# --- reached_q3 ---------------------------------------------------------------------------------------------
# In every knockout race (qualifying table, all entrants), exactly the number of Q3 places reached Q3.
knockout = q[q["raceId"].isin(CLEAN["races"].loc[CLEAN["races"]["year"] >= 2006, "raceId"])]
per_race = knockout.groupby("raceId").agg(reached=("reached_q3", "sum"), places=("q3_places", "first"))
assert (per_race["reached"] == per_race["places"]).all(), "a race has the wrong number of drivers in Q3"
print(f"All {len(per_race)} knockout races (2006+) have exactly their number of Q3 places marked as reached Q3.")

# Rows of the feature table (starters, 2006+): old definition (Q3 time only) vs corrected.
q3c = (feat.loc[feat["no_q3_format"] == 0, ["raceId", "driverId", "year", "split", "quali_pos", "grid", "reached_q3"]]
       .merge(q[["raceId", "driverId", "q2_sec", "q3_sec", "q3_places", "reached_q3_time_only"]], on=["raceId", "driverId"], how="left")
       .merge(nm, on="driverId").merge(rc, on="raceId"))
q3c = q3c[q3c["reached_q3"].notna()]
q3_changed = q3c[q3c["reached_q3"] != q3c["reached_q3_time_only"]]
print(f"reached_q3 ({len(q3c):,} starters with qualifying, 2006+): {len(q3_changed)} values changed, "
      f"all from 0 to 1: {bool((q3_changed['reached_q3'] == 1).all())}")
display(q3_changed.groupby("split").size().rename("changed rows").to_frame().T)

show = ["year", "race", "driver", "quali_pos", "grid", "q2_sec", "q3_sec", "q3_places", "reached_q3_time_only", "reached_q3"]
print("Examples: advanced to Q3 without setting a Q3 time (0 before, 1 now)")
display(q3_changed[show].head(6))
print("Format exceptions: 2016 elimination format (8 places), 2007 position that includes a penalty, "
      "2008 recorded Q2 time not the one that counted, 2015 US GP (Q3 cancelled)")
special = q3c[(q3c["raceId"].isin(ELIMINATION_FORMAT) & q3c["quali_pos"].between(8, 11))
              | ((q3c["year"] == 2007) & (q3c["race"] == "Belgian Grand Prix") & q3c["quali_pos"].between(9, 14))
              | ((q3c["year"] == 2008) & (q3c["race"] == "Turkish Grand Prix") & q3c["quali_pos"].between(9, 11))
              | ((q3c["year"] == 2015) & (q3c["race"] == "United States Grand Prix") & q3c["quali_pos"].between(9, 11))]
display(special[show].sort_values(["year", "race", "quali_pos"]).reset_index(drop=True))
''')

code(r'''
# --- Team form: threshold evidence -----------------------------------------------------------------------------
gaps = TEAM_RACES["gap_days"].dropna()
gap_bins = pd.cut(gaps, [0, 120, 250, 365, 550, 730, 1095, 100000],
                  labels=["<=4 months", "4-8 months", "8-12 months", "12-18 months", "18-24 months", "2-3 years", "3+ years"])
print("Days between two consecutive races of the same constructor (all seasons):")
display(gap_bins.value_counts().sort_index().rename("gaps").to_frame().T)
modern = TEAM_RACES["date"].dt.year >= MODEL_START_YEAR
print(f"Longest gap in {MODEL_START_YEAR}+ that is NOT a reset: {int(TEAM_RACES.loc[modern & (TEAM_RACES['gap_days'] <= TEAM_GAP_DAYS), 'gap_days'].max())} days")
display(pd.DataFrame([{"threshold (days)": t, "resets, all seasons": int((gaps > t).sum()),
                       f"resets, {MODEL_START_YEAR}+": int((modern & (TEAM_RACES["gap_days"] > t)).sum())}
                      for t in [365, 450, 550, 730]]).set_index("threshold (days)").T)

# --- Team form: old (no reset) vs corrected ----------------------------------------------------------------------
col = f"cons_podium_rate_{FORM_WINDOW}"
old_team = form_features(FORM_WINDOW, reset_gap=False)[["raceId", "driverId", col]].rename(columns={col: "old (no reset)"})
tc = feat[["raceId", "driverId", "constructorId", "year", "split", col]].merge(old_team, on=["raceId", "driverId"], validate="one_to_one")
same = (tc[col].isna() & tc["old (no reset)"].isna()) | np.isclose(tc[col], tc["old (no reset)"])
team_changed = tc[~same]

# Only the first FORM_WINDOW races of a restarted segment may change; every other window must be identical.
pos_in_segment = TEAM_RACES.groupby(["constructorId", "team_segment"]).cumcount()
may_change = TEAM_RACES.loc[(TEAM_RACES["team_segment"] > 0) & (pos_in_segment < FORM_WINDOW), ["constructorId", "raceId"]]
assert team_changed.merge(may_change, on=["constructorId", "raceId"]).shape[0] == len(team_changed), \
    "team form changed outside the first races of a restarted segment"
restarts = TEAM_RACES[TEAM_RACES["gap_days"] > TEAM_GAP_DAYS][["constructorId", "raceId"]]
assert feat.merge(restarts, on=["constructorId", "raceId"])[col].isna().all(), "the first race after a reset has a form value"
print(f"Team form: {len(team_changed)} rows changed ({team_changed.groupby('split').size().to_dict()}); "
      f"all inside the first {FORM_WINDOW} races after a reset. All other rows are unchanged.")

modern_resets = (TEAM_RACES[modern & (TEAM_RACES["gap_days"] > TEAM_GAP_DAYS)]
                 .merge(CLEAN["constructors"][["constructorId", "name"]].rename(columns={"name": "team"}), on="constructorId")
                 .merge(rc, on="raceId"))
modern_resets["previous race"] = modern_resets["date"] - pd.to_timedelta(modern_resets["gap_days"], unit="D")
first_rows =tc.merge(modern_resets[["constructorId", "raceId"]], on=["constructorId", "raceId"]).groupby(["constructorId", "raceId"]).first()
modern_resets = modern_resets.join(first_rows[["old (no reset)", col]], on=["constructorId", "raceId"])
print(f"Resets in {MODEL_START_YEAR}+ (the team's form at its comeback race, before and after the fix):")
display(modern_resets[["team", "race", "date", "previous race", "gap_days", "old (no reset)", col]]
        .rename(columns={"gap_days": "gap (days)", col: "corrected"}).sort_values("date").reset_index(drop=True))
''')

md(r"""
**Missing values in the candidates (training seasons of model A).** Some gaps are expected and meaningful: a driver's first race has no form, a new team has no team form, `reached_q3` doesn't exist before 2006. How to fill them is a **pre-processing** decision for the next section; for the trials in 4.5 a simple provisional fill is used (median + a "was missing" indicator).
""")

code(r'''
TRAIN_A = feat[(feat["split"] == "train") & (feat["year"] >= MODEL_START_YEAR)]
VAL = feat[feat["split"] == "validation"]
missing = pd.DataFrame({"train A missing %": 100 * TRAIN_A[ALL_CANDIDATES].isna().mean(),
                        "validation missing %": 100 * VAL[ALL_CANDIDATES].isna().mean()}).round(1)
display(missing[missing.max(axis=1) > 0])
''')

# ---------------------------------------------------------------- 4.3
md(r"""
## 4.3 Evidence for each feature group (training seasons only)

Each plot shows the **podium rate** for groups of training rows (2003–2018) with similar values of a feature. A feature is useful if the podium rate **changes clearly** across its values. The number of rows behind each bar is shown, so small groups aren't over-read.
""")

code(r'''
def rate_plot(ax, df, col, bins=None, labels=None, title=None, xlabel=None):
    x = df[col] if bins is None else pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    t = df.groupby(x, observed=True)["podium"].agg(["mean", "size"])
    ax.bar(t.index.astype(str), 100 * t["mean"], color="#4C72B0")
    for i, (m, n) in enumerate(zip(t["mean"], t["size"])):
        ax.text(i, 100 * m + 1, f"n={n}", ha="center", fontsize=7, rotation=90 if len(t) > 12 else 0)
    ax.set_title(title or col, fontsize=10)
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel("podium %")
    ax.tick_params(axis="x", labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
rate_plot(axes[0], TRAIN_A[TRAIN_A["grid"] <= 24].assign(grid=lambda d: d["grid"].astype(int)), "grid",
          title="Grid: podium rate by starting position")
rate_plot(axes[1], TRAIN_A, "pit_lane_start", title="Pit-lane start (1 = yes)")
plt.tight_layout()
plt.show()
''')

md(r"""
**Grid:** the strongest single signal. From pole, 79% of drivers finish on the podium; from 2nd 62%, 3rd 50%, 5th 22%, and from 8th or lower under 5%. The drop is steep, not a straight line, which matters for the linear model (it can only draw a gradual curve). `pit_lane_start` has only 7 training rows (none on the podium), so it can't show a pattern on its own; it is kept with `grid` because it tells the model that the grid number of those rows is a recode (2.6), not a real slot.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_plot(axes[0], TRAIN_A, "gap_q1_pct", bins=[0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 100],
          labels=["0-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5", "5+"],
          title="Qualifying: gap to fastest Q1 time", xlabel="gap to fastest Q1 (%)")
rate_plot(axes[1], TRAIN_A, "grid_minus_quali", bins=[-30, -1, 0, 2, 5, 30],
          labels=["moved up", "same", "+1-2", "+3-5", "+6 or more"],
          title="grid - qualifying position (penalties)", xlabel="started this many places behind qualifying")
rate_plot(axes[2], TRAIN_A[TRAIN_A["no_q3_format"] == 0], "reached_q3", title="Reached Q3 (2006-2018)")
plt.tight_layout()
plt.show()
''')

md(r"""
**Qualifying:** pace matters on top of grid position. Drivers within 0.25% of the fastest Q1 time reach the podium 54% of the time; at 0.5–1% it is 20%, and beyond 2% almost never. Reaching Q3 (2006–2018): 29% vs 1%. The middle plot (`grid − qualifying position`) mixes all grid positions, so on its own it looks odd. The fair comparison is inside the same grid slots (3.1.2): drivers who qualified in the top 3 but started 6th or lower because of a penalty still reached the podium **32%** of the time vs **3.4%** for others starting there. That is the information this feature carries.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
ranked_drv = TRAIN_A[TRAIN_A["drv_no_standing_yet"] == 0]
ranked_cons = TRAIN_A[TRAIN_A["cons_no_standing_yet"] == 0]
rate_plot(axes[0], ranked_drv, "drv_champ_pos", bins=[0, 1, 2, 3, 5, 10, 200], labels=["1", "2", "3", "4-5", "6-10", "11+"],
          title="Driver championship position (before the race)", xlabel="position")
rate_plot(axes[1], ranked_drv, "drv_points_share",
          bins=[-0.01, 0, 0.1, 0.25, 0.5, 0.75, 0.99, 1], labels=["0", "0-10%", "10-25%", "25-50%", "50-75%", "75-99%", "leader"],
          title="Driver: share of the leader's points", xlabel="share")
rate_plot(axes[2], ranked_cons, "cons_champ_pos", bins=[0, 1, 2, 3, 5, 100], labels=["1", "2", "3", "4-5", "6+"],
          title="Constructor championship position", xlabel="position")
plt.tight_layout()
plt.show()
''')

md(r"""
**Standings (ranked drivers and teams only):** the championship leader before the race reaches the podium 67% of the time, 2nd 57%, 3rd 42%, and from 11th down about 1%. Share of the leader's points tells the same story (leader 67%, under 10% of the leader's points about 1–2%). Constructor position is even sharper at the bottom (6th or lower: 2%). Rows with no standing yet (round 1, or no points so far) are left out of these plots; the `no_standing_yet` flags carry them.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_plot(axes[0], TRAIN_A, f"drv_podium_rate_{FORM_WINDOW}", bins=[-0.01, 0, 0.2, 0.4, 0.6, 0.8, 1],
          labels=["0", "1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"],
          title="Driver podiums in the last 5 races", xlabel="podium rate (last 5)")
rate_plot(axes[1], TRAIN_A, f"cons_podium_rate_{FORM_WINDOW}", bins=[-0.01, 0, 0.1, 0.25, 0.5, 0.75, 1],
          labels=["0", "0-10%", "10-25%", "25-50%", "50-75%", "75%+"],
          title="Team podium rate, last 5 races", xlabel="share of the team's starts on the podium")
rate_plot(axes[2], TRAIN_A, f"drv_dnf_rate_{FORM_WINDOW}", bins=[-0.01, 0, 0.2, 0.4, 1],
          labels=["0", "1 of 5", "2 of 5", "3+ of 5"], title="Driver retirements in the last 5 races", xlabel="DNF rate (last 5)")
plt.tight_layout()
plt.show()
''')

md(r"""
**Form (last 5 races, current race excluded):** a driver with no podium in the last 5 races gets one only 3% of the time; with 2 of 5 it is 39%, with 4–5 of 5 about 65–69%. The team's recent podium rate shows the same climb (2% → 70%). Recent retirements lower the chance too (no DNF in 5 races: 21%; 3 or more: 4%).
""")

code(r'''
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
rate_plot(axes[0], TRAIN_A, "n_starters", bins=[0, 18, 20, 22, 24, 30], labels=["<=18", "19-20", "21-22", "23-24", "25+"],
          title="Number of starters", xlabel="starters in the race")
rate_plot(axes[1], TRAIN_A, "drv_experience", bins=[-1, 0, 10, 30, 60, 100, 400],
          labels=["debut", "1-10", "11-30", "31-60", "61-100", "100+"], title="Driver experience", xlabel="career starts before the race")
rate_plot(axes[2], TRAIN_A, "drv_age", bins=[0, 23, 26, 29, 32, 35, 50], labels=["<23", "23-26", "26-29", "29-32", "32-35", "35+"],
          title="Driver age", xlabel="age (years)")
rate_plot(axes[3], TRAIN_A, "home_race", title="Home race (1 = yes)")
plt.tight_layout()
plt.show()
''')

md(r"""
**Context:** weaker effects, but real ones. Fewer starters means a higher podium rate (19% with 18 or fewer cars vs 12.5% with 23–24), simply because 3 podium places are shared by fewer drivers. Experience rises steadily (debut 3%, 100+ starts 23%). Age peaks at 29–32 (21%) and is lower for young (<23: 7%) and older (35+: 12.5%) drivers. **Home race** shows 12.4% vs 14.2%, the same misleading raw gap as Q2 (home drivers start further back). Q2 found no effect once grid is accounted for.
""")

code(r'''
# One number per feature: how well does it rank podium finishers on its own? (training seasons of model A)
from sklearn.metrics import roc_auc_score

single = []
for group, cols in FEATURE_GROUPS.items():
    for c in cols:
        d = TRAIN_A[[c, "podium"]].dropna()
        if d[c].nunique() < 2:
            continue
        auc = roc_auc_score(d["podium"], d[c])
        single.append({"group": group, "feature": c, "rows": len(d),
                       "AUC alone": round(max(auc, 1 - auc), 3), "direction": "higher -> more podiums" if auc >= 0.5 else "lower -> more podiums"})
single = pd.DataFrame(single).sort_values("AUC alone", ascending=False).reset_index(drop=True)
display(single)
''')

md(r"""
**Ranking the features one at a time.** "AUC alone" is the chance that a random podium row has a "better" value than a random non-podium row (0.5 = useless, 1 = perfect). Starting position (grid 0.907, qualifying position 0.909) is the best single signal, followed by team form (0.872), the driver's recent average finish (0.862) and the standings (0.83–0.86). The flags (`no_standing_yet`, `quali_missing`, `no_q3_format`, `pit_lane_start`) score about 0.5 on purpose: they don't predict podiums on their own; they tell the model when another feature is missing or means something different.
""")

# ---------------------------------------------------------------- 4.4
md(r"""
## 4.4 Redundancy: features that say the same thing

Two features that move together carry almost the same information. Keeping both makes a linear model's coefficients unstable and its explanations harder to read, without adding much. We look at the **Spearman correlation** (rank-based, so it works for positions) on the training seasons.
""")

code(r'''
corr = TRAIN_A[ALL_CANDIDATES].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        if i != j and abs(corr.iloc[i, j]) >= 0.7:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
ax.set_title("Correlation between candidate features (training seasons 2003-2018)")
plt.tight_layout()
plt.show()

pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("spearman").reset_index()
         .rename(columns={"level_0": "feature 1", "level_1": "feature 2"}))
display(pairs[pairs["spearman"].abs() >= 0.7].sort_values("spearman", key=abs, ascending=False).round(2).reset_index(drop=True))
''')

code(r'''
auc_alone = single.set_index("feature")["AUC alone"]
DECISIONS = []

# 1. grid vs qualifying position: keep grid (where the car actually starts) and express qualifying as
#    the difference grid - quali_pos, so the penalty information stays without a near-duplicate column.
DECISIONS.append(("grid vs quali_pos", "keep grid + grid_minus_quali, drop quali_pos",
                  f"Spearman {corr.loc['grid', 'quali_pos']:.2f}; AUC alone grid {auc_alone['grid']:.3f} vs quali_pos {auc_alone['quali_pos']:.3f}"))

# 2. championship position vs points share (driver and constructor): keep the one that ranks podiums better on its own.
STANDINGS_KEEP = []
for who in ["drv", "cons"]:
    pos, share = f"{who}_champ_pos", f"{who}_points_share"
    keep = share if auc_alone[share] >= auc_alone[pos] else pos
    STANDINGS_KEEP.append(keep)
    DECISIONS.append((f"{pos} vs {share}", f"keep {keep}",
                      f"Spearman {corr.loc[pos, share]:.2f}; AUC alone {pos} {auc_alone[pos]:.3f} vs {share} {auc_alone[share]:.3f}"))
display(pd.DataFrame(DECISIONS, columns=["pair", "decision", "evidence (training seasons)"]))

FEATURE_GROUPS["qualifying"] = ["gap_q1_pct", "grid_minus_quali", "reached_q3", "no_q3_format", "quali_missing"]
FEATURE_GROUPS["standings"] = STANDINGS_KEEP + ["drv_no_standing_yet", "cons_no_standing_yet"]
print("Qualifying group now:", FEATURE_GROUPS["qualifying"])
print("Standings group now: ", FEATURE_GROUPS["standings"])
''')

md(r"""
**Decisions.**
- **grid vs qualifying position (correlation 0.97): keep `grid`**, drop `quali_pos`. Grid is where the car actually starts the race. The qualifying information isn't lost: `grid_minus_quali` keeps the penalty signal and `gap_q1_pct` keeps the pace. (For a linear model, `grid` + `quali_pos` and `grid` + `grid − quali_pos` hold exactly the same information, so this costs nothing and avoids two near-identical columns.)
- **Driver: championship position vs share of the leader's points (−0.95): keep position.** They rank podiums almost equally well on their own (0.854 vs 0.852); position is the simpler one to explain.
- **Constructor: position vs points share (−0.93): keep points share**, which ranks podiums clearly better (0.856 vs 0.826). It also shows *how far* ahead the top team is, not just its rank.
- **Other strongly linked pairs are kept on purpose:** driver vs team standings (0.89–0.93), form vs standings (~0.8), experience vs age (0.81). Each pair measures something different (the driver vs the car; recent races vs the whole season; years of racing vs age). The trials in 4.5 check whether they add anything.
""")

# ---------------------------------------------------------------- 4.5
md(r"""
## 4.5 Feature trials on the validation seasons

**Method:** a **logistic regression** (the simplest reasonable model) is trained on the training seasons (2003–2018) and scored on the validation seasons (2019–2021).
- We start with **grid only**, then add **one group at a time**.
- **Score: PR-AUC** (average precision). Podiums are only ~15% of rows, and PR-AUC focuses on how well the true podium finishers are ranked at the top. ROC-AUC is shown too.
- **Keep rule:** a group is kept if **(1)** its plots in 4.3 show a clear link with podiums, and **(2)** adding it **does not lower** validation PR-AUC.
- **Noise:** the validation period has only 60 races, so the score moves by luck. For every decision we also show a **95% interval for the gain**, from resampling the validation races 500 times. An interval that includes 0 means "too small to call either way". The groups then stand on their 4.3 evidence, and the feature-group ablation in the model-experiments section (after pre-processing) measures their contribution for every model.
- **Provisional pre-processing** (the final choice comes in the next section): missing values are filled with the training median plus a "was missing" indicator, then features are standardised. Both are fitted on the training seasons only.

Every trial has its own cell and is collected in `TRIALS`.
""")

code(r'''
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRIALS = []
PREDS = {}                                   # validation predictions of each trial (for the gain intervals)
val_y = VAL["podium"].to_numpy()
val_race = VAL["raceId"].to_numpy()
rng_boot = np.random.default_rng(42)
BOOT_RACES = [rng_boot.choice(np.unique(val_race), len(np.unique(val_race))) for _ in range(500)]
BOOT_IDX = [np.concatenate([np.flatnonzero(val_race == r) for r in races]) for races in BOOT_RACES]


def run_trial(name, cols, note=""):
    tr, va = TRAIN_A, VAL
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(max_iter=5000))
    model.fit(tr[cols].astype(float), tr["podium"])
    p_tr = model.predict_proba(tr[cols].astype(float))[:, 1]
    p_va = model.predict_proba(va[cols].astype(float))[:, 1]
    PREDS[name] = p_va
    res = {"trial": name, "n features": len(cols),
           "train PR-AUC": round(average_precision_score(tr["podium"], p_tr), 4),
           "val PR-AUC": round(average_precision_score(va["podium"], p_va), 4),
           "val ROC-AUC": round(roc_auc_score(va["podium"], p_va), 4), "note": note}
    TRIALS.append(res)
    display(pd.DataFrame([res]))
    return name


def gain_interval(before, after):
    """95% interval of the PR-AUC gain, resampling the validation races (same resamples for both)."""
    pa, pb = PREDS[before], PREDS[after]
    gains = [average_precision_score(val_y[i], pb[i]) - average_precision_score(val_y[i], pa[i]) for i in BOOT_IDX]
    return np.percentile(gains, [2.5, 97.5])


def decide(group, before, after):
    s_before = next(t["val PR-AUC"] for t in TRIALS if t["trial"] == before)
    s_after = next(t["val PR-AUC"] for t in TRIALS if t["trial"] == after)
    gain = s_after - s_before
    lo, hi = gain_interval(before, after)
    kept = gain >= 0
    print(f"{group}: validation PR-AUC {s_before:.4f} -> {s_after:.4f}, gain {gain:+.4f} "
          f"(95% interval {lo:+.4f} to {hi:+.4f}) -> {'KEEP' if kept else 'DROP'}")
    TRIALS[-1].update({"gain": round(gain, 4), "gain 95% interval": f"[{lo:+.3f}, {hi:+.3f}]",
                       "note": "keep" if kept else "drop"})
    return kept


print(f"Training rows: {len(TRAIN_A):,} | validation rows: {len(VAL):,} ({int(VAL['podium'].sum())} podiums)")
''')

md(r"""
### Trial 1: grid only (baseline)
""")

code(r'''
current = list(FEATURE_GROUPS["grid"])
best = run_trial("1. grid only", current, "baseline")
''')

md(r"""
### Trial 2: + qualifying
""")

code(r'''
s = run_trial("2. + qualifying (gap from Q1)", current + FEATURE_GROUPS["qualifying"])
if decide("qualifying", best, s):
    current, best = current + FEATURE_GROUPS["qualifying"], s
''')

md(r"""
### Trial 2b: gap to pole from Q1 only vs from the best session (written trial)

The gap to pole could also use each driver's **best time of Q1/Q2/Q3**. We use **Q1 only**, because:
- **2006–2009:** Q3 was run with the **race fuel load** on board, so Q3 times were slower than Q2 times and depended on each team's fuel strategy, not just pace.
- **Early 2005** (first 6 races): two sessions whose times were **added together**; the best single session isn't what decided the grid.
- **Almost every driver sets a Q1 time**, so the Q1 gap exists for nearly the whole field, while Q2/Q3 times exist only for the faster drivers.

**Limits of the Q1 gap (it is not a perfectly equal comparison):**
- **2003–2005:** there was no knockout. `q1` is the single qualifying lap that set the grid, and from 2003 to most of 2005 it was run **with race fuel**, so these gaps also mix in fuel strategy (the same problem as the 2006–2009 Q3). In the first 6 races of 2005, `q1` is only the first of two sessions whose times were added together.
- **Within one session,** conditions still differ: track grip improves, rain can arrive mid-session, and top teams sometimes do just enough in Q1 to get through. The Q1 gap is therefore a **noisy** pace measure, not an exact one.
- These limits affect every gap definition. The choice of Q1 rests on it being the most widely available and least strategy-dependent time, plus the check below.

The table shows how often the pole-sitter was **not** the fastest driver by best-of-sessions time, per season (training seasons only), followed by the trial.
""")

code(r'''
qq = CLEAN["qualifying"].merge(CLEAN["races"][["raceId", "year"]], on="raceId")
qq = qq[qq["year"].between(2003, 2018)]
qq["best_sec"] = qq[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)
pole = qq[qq["position"] == 1].set_index("raceId")["best_sec"]
fastest = qq.groupby("raceId")["best_sec"].min()
not_fastest = (pole > fastest.loc[pole.index]).groupby(qq.drop_duplicates("raceId").set_index("raceId").loc[pole.index, "year"]).sum()
display(not_fastest.rename("races where the pole-sitter isn't fastest by best-of-sessions").to_frame().T)

swap = [c if c != "gap_q1_pct" else "gap_best_pct" for c in current] if "gap_q1_pct" in current \
    else current + [c if c != "gap_q1_pct" else "gap_best_pct" for c in FEATURE_GROUPS["qualifying"]]
s_best = run_trial("2b. qualifying with gap from best-of-Q1/Q2/Q3", swap, "written trial: alternative gap definition")
s_q1 = run_trial("2b. qualifying with gap from Q1 (same set)", [c for c in swap if c != "gap_best_pct"] + ["gap_q1_pct"],
                 "written trial: chosen gap definition")
score_of = {t["trial"]: t["val PR-AUC"] for t in TRIALS}
print(f"Gap from Q1: {score_of[s_q1]:.4f} | gap from best session: {score_of[s_best]:.4f}")
lo, hi = gain_interval(s_best, s_q1)
print(f"Q1 minus best-of-sessions: {score_of[s_q1] - score_of[s_best]:+.4f} (95% interval {lo:+.4f} to {hi:+.4f})")
''')

md(r"""
**Result of trial 2b:** the gap from **Q1** scores a little better than the gap from the best session (0.7210 vs 0.7099). The difference is within the noise of 60 races (95% interval −0.0082 to +0.0325), so the trial doesn't prove Q1 is better; the decision rests mainly on the reasoning above, and the data supports that reasoning: in 2006–2009 the pole-sitter was **not** the fastest driver by best-of-sessions time in 9–11 races every season (race-fuel Q3), against 0–3 races in other seasons. **We keep the Q1 gap**, with the limits listed above.
""")

md(r"""
### Trial 3: + standings
""")

code(r'''
s = run_trial("3. + standings", current + FEATURE_GROUPS["standings"])
if decide("standings", best, s):
    current, best = current + FEATURE_GROUPS["standings"], s
''')

md(r"""
### Trial 4: + form (last 5 races)
""")

code(r'''
s = run_trial(f"4. + form (last {FORM_WINDOW})", current + FEATURE_GROUPS["form"])
if decide("form", best, s):
    current, best = current + FEATURE_GROUPS["form"], s
''')

md(r"""
### Trial 4b: how many past races for form? (3 vs 5 vs 10)
""")

code(r'''
base_wo_form = [c for c in current if c not in FEATURE_GROUPS["form"]]
window_scores = {}
for n in [3, 5, 10]:
    f_n = form_features(n)
    cols_n = [c for c in f_n.columns if c not in ("raceId", "driverId")]
    tmp = feat.drop(columns=[c for c in feat.columns if c in cols_n]).merge(f_n, on=["raceId", "driverId"], how="left")
    TRAIN_A_n = tmp[(tmp["split"] == "train") & (tmp["year"] >= MODEL_START_YEAR)]
    VAL_n = tmp[tmp["split"] == "validation"]
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(TRAIN_A_n[base_wo_form + cols_n].astype(float), TRAIN_A_n["podium"])
    assert (VAL_n["raceId"].to_numpy() == val_race).all()          # same validation rows, same order
    PREDS[f"4b. window {n}"] = model.predict_proba(VAL_n[base_wo_form + cols_n].astype(float))[:, 1]
    window_scores[n] = average_precision_score(VAL_n["podium"], PREDS[f"4b. window {n}"])
    TRIALS.append({"trial": f"4b. form window = last {n} races", "n features": len(base_wo_form + cols_n),
                   "val PR-AUC": round(window_scores[n], 4), "note": "written trial: form window"})
display(pd.Series(window_scores, name="validation PR-AUC").rename_axis("form window (races)").round(4).to_frame().T)
for other in [3, 10]:
    lo, hi = gain_interval(f"4b. window {other}", f"4b. window {FORM_WINDOW}")
    print(f"window {FORM_WINDOW} minus window {other}: {window_scores[FORM_WINDOW] - window_scores[other]:+.4f} "
          f"(95% interval {lo:+.4f} to {hi:+.4f})")
''')

md(r"""
**Result of trial 4b:** the last **5** races scores highest (0.7382), ahead of 3 races (0.7351) and 10 races (0.7302), but the differences are small and **both 95% intervals include 0**, so this trial does **not** show that 5 is better. **We keep 5 as a reasonable middle value**, chosen before the trial: 3 races can swing on one lucky or unlucky weekend, and 10 races reach back about half a season, to a car that may have changed. The trial only confirms that 5 isn't worse than the alternatives.
""")

md(r"""
### Trial 5: + context (starters, experience, age)
""")

code(r'''
context_core = ["n_starters", "drv_experience", "drv_age"]
s = run_trial("5. + context", current + context_core)
if decide("context", best, s):
    current, best = current + context_core, s
''')

md(r"""
### Trial 6: + home race

Q2 found no home advantage once starting position is accounted for, so we expect no gain. This trial checks it on the validation seasons.
""")

code(r'''
s = run_trial("6. + home race", current + ["home_race"])
if decide("home race", best, s):
    current, best = current + ["home_race"], s
''')

md(r"""
### Summary of all trials
""")

code(r'''
trials_df = pd.DataFrame(TRIALS)
display(trials_df)

main = trials_df[trials_df["trial"].str.match(r"^\d\. ")]
fig, ax = plt.subplots(figsize=(10, 3.8))
colors = ["#55A868" if ("keep" in str(n) or n == "baseline") else "#C44E52" for n in main["note"]]
ax.barh(main["trial"], main["val PR-AUC"], color=colors)
for y, v in enumerate(main["val PR-AUC"]):
    ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, max(main["val PR-AUC"]) * 1.15)
ax.set_xlabel("validation PR-AUC (2019-2021)")
ax.set_title("Adding one feature group at a time (green = kept, red = dropped)")
plt.tight_layout()
plt.show()
print("Final feature set:", current)
''')

md(r"""
**What the trials show.**
- **Grid alone** already gives validation PR-AUC **0.676**.
- **Qualifying** adds the most (+0.045 to **0.721**). Together with grid, it is the clearest evidence of predictive value: it is the only gain whose 95% interval is entirely above 0 (+0.019 to +0.071).
- **Standings** (+0.013), **form** (+0.004) and **context** (+0.0002) each raise the score only a little, and **all three intervals include 0**. So this trial does **not** show that they help; for this linear model, most of what they say is already in grid and qualifying.
- They are **kept** for two reasons, not because of these gains: (1) their 4.3 plots on the training seasons show a clear link with podiums, and (2) adding them doesn't clearly lower the validation score. Form is also required by the brief.
- These trials use **logistic regression only**. A feature that adds little to a linear model can still help a tree model or a neural network (or the reverse), so the feature-group ablation in the **model-experiments section** (after pre-processing) re-tests each group with every model.
- **Home race** lowers the score (−0.003, interval entirely below 0) and is **dropped**. That drop is small too; the main reason is Q2, which found no home advantage once the starting position is known.
- The model doesn't overfit: training PR-AUC (0.69) is close to validation (0.74). Validation being slightly higher is most likely because a few teams dominated 2019–2021, which makes podiums easier to predict.
- **Final set: 18 features** in four groups (grid, qualifying, standings, form) plus context. Final validation PR-AUC **0.738** for this simple model, up from 0.676 with grid only.
""")

# ---------------------------------------------------------------- 4.6
md(r"""
## 4.6 Final feature set

The features below are used by all models in the next sections (model A, and model B with the same features). For each one: the group, why it is known **before the race**, and the evidence behind it.
""")

code(r'''
WHY_PRE_RACE = {
    "grid": "starting order, fixed before the start", "pit_lane_start": "known from the starting order",
    "gap_q1_pct": "Q1 lap times, set before the race", "grid_minus_quali": "grid and qualifying, both known before the start",
    "reached_q3": "qualifying result", "no_q3_format": "qualifying format of the season", "quali_missing": "whether qualifying data exists",
    "quali_pos": "qualifying result",
    "drv_champ_pos": "standings after the PREVIOUS race", "drv_points_share": "standings after the PREVIOUS race",
    "drv_no_standing_yet": "standings after the PREVIOUS race", "cons_champ_pos": "standings after the PREVIOUS race",
    "cons_points_share": "standings after the PREVIOUS race", "cons_no_standing_yet": "standings after the PREVIOUS race",
    f"drv_podium_rate_{FORM_WINDOW}": "earlier races only (shift(1), asserted)", f"drv_avg_finish_{FORM_WINDOW}": "earlier races only (shift(1), asserted)",
    f"drv_dnf_rate_{FORM_WINDOW}": "earlier races only (shift(1), asserted)", f"cons_podium_rate_{FORM_WINDOW}": "earlier races only (shift(1), asserted)",
    "n_starters": "entry list / starting grid", "drv_experience": "count of earlier starts", "drv_age": "birth date and race date",
    "home_race": "nationality and circuit country",
}
group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
TRIAL_OF_GROUP = {"grid": 1, "qualifying": 2, "standings": 3, "form": 4, "context": 5}
FINAL_FEATURES = list(current)
final_table = pd.DataFrame({
    "feature": FINAL_FEATURES,
    "group": [group_of.get(c, "context") for c in FINAL_FEATURES],
    "known before the race because": [WHY_PRE_RACE[c] for c in FINAL_FEATURES],
    "AUC alone (train)": [auc_alone.get(c, np.nan) for c in FINAL_FEATURES],
    "evidence": [f"4.3 plot + trial {TRIAL_OF_GROUP[group_of.get(c, 'context')]} (4.5)" for c in FINAL_FEATURES],
})
display(final_table)

dropped = [c for c in ALL_CANDIDATES + ["quali_pos", "drv_champ_pos", "drv_points_share", "cons_champ_pos", "cons_points_share"]
           if c not in FINAL_FEATURES]
print("Candidates not used:", sorted(set(dropped)))
assert not set(FINAL_FEATURES) & set(POST_RACE)
''')

md(r"""
**Summary.** The models will use **18 features**: grid (2), qualifying (5, including the format flags), standings (4), form (4) and context (3). Every one is known before the race starts. Standings come from the previous race, and form was proven by the asserts in 4.2 to use only earlier races. Not used: `quali_pos` and `drv_points_share` / `cons_champ_pos` (near-duplicates, 4.4), and `home_race` (lowered the score, 4.5). Model B uses the same 18 features; its `quali_missing` flag marks the seasons without qualifying data. Missing values (a driver's first race, a team's first race or first race after a long absence, `reached_q3` before 2006) are handled in the pre-processing section.

**How strong is the evidence?** Grid and qualifying clearly help (4.5). Standings, form and context are kept because of their training-season plots (4.3) and because they don't clearly hurt the validation score; their gains in the logistic-regression trials are small and uncertain. Whether they help the other models is tested in the feature-group ablation of the model-experiments section.
""")

# ---------------------------------------------------------------- 4.7
md(r"""
## 4.7 Export: the final feature table

The engineered features are saved to **`outputs/final_features.csv`**, next to the cleaned dataset (`outputs/clean_data/`), so the next sections (and teammates) can start from them. The file is rewritten on every run.
- **Rows:** every row of `feat`, i.e. all seasons 1950–2024 with their `split` label, so both model A (2003+) and model B (all years) can use the same file. Test rows are saved, but nothing in this notebook uses their outcomes to make a decision.
- **Columns (23):** `raceId`, `driverId`, `year`, `split`, the target `podium`, and the **18 final features** in the order of `FINAL_FEATURES`. No rejected candidates and no post-race columns.
- **Missing values stay missing** (empty cells). No filling or scaling here: those steps are fitted on the training seasons in the pre-processing section.
- Rows are sorted by race (in calendar order), then grid position, then `driverId`, so the file is identical on every run.
""")

code(r'''
FINAL_FEATURES_PATH = CLEAN_DATA_DIR.parent / "final_features.csv"   # outputs/ (on Kaggle: /kaggle/working/outputs/)
FINAL_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

EXPORT_COLUMNS = ["raceId", "driverId", "year", "split", "podium"] + list(FINAL_FEATURES)
assert len(FINAL_FEATURES) == 18 and len(EXPORT_COLUMNS) == 23 and len(set(EXPORT_COLUMNS)) == 23
assert not set(FINAL_FEATURES) & set(POST_RACE)

final_features_df = (feat.sort_values(["race_seq", "grid", "driverId"])[EXPORT_COLUMNS]
                     .reset_index(drop=True))
assert not final_features_df.duplicated(["raceId", "driverId"]).any()
assert len(final_features_df) == len(feat)
final_features_df.to_csv(FINAL_FEATURES_PATH, index=False)

# Read the file back: same rows, same values, missing values still missing.
back = pd.read_csv(FINAL_FEATURES_PATH, keep_default_na=False, na_values=[""], float_precision="round_trip")
assert list(back.columns) == EXPORT_COLUMNS
pd.testing.assert_frame_equal(back, final_features_df, check_dtype=False, check_exact=True)
print(f"Saved {FINAL_FEATURES_PATH}: {len(back):,} rows x {back.shape[1]} columns, read back identical.")

print("Missing values per feature (%), only features with gaps; 'train' here is all training seasons 1950-2018:")
display(pd.DataFrame({f"{s} {lo}-{hi}": back.loc[back["split"] == s, FINAL_FEATURES].isna().mean().mul(100).round(1)
                      for s, (lo, hi) in [("train", (back["year"].min(), SPLIT_YEARS["train"][1])),
                                          ("validation", SPLIT_YEARS["validation"]), ("test", SPLIT_YEARS["test"])]})
        .loc[lambda d: d.max(axis=1) > 0])
display(back["split"].value_counts().rename("rows").to_frame().T)
''')


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")