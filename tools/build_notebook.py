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
| B. Starting entry | keep the row the driver actually **started** in (grid > 0, then lowest `resultId`) | Selection uses only pre-race info, so features and target describe the same car. Consistent with the cutoff | May label a driver "no podium" even though he shared a podium car later in the race |
| C. Hybrid | features from the starting entry, target = podium in **any** of his rows | Keeps the official credit and pre-race-consistent features | Mixes two rows into one, which is harder to explain and test |
| D. Drop all duplicated pairs | remove these 176 rows | Simplest, no arbitrary choice | Loses data (small), and removes some genuine podiums from 1950s races |

Practical note: if the modeling window starts where qualifying data exists (1994+, see 1.7), **none of these rows reach the model**. The rule still matters for the data-engineering questions if they use all years, and the brief requires a documented rule.
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
| Duplicate (raceId, driverId) | **Rule B**: keep the entry the driver **started** in (grid > 0 first, then lowest `resultId`) | the row is chosen with pre-race information only, so features and target describe the same car (1.6) |
| Non-starters | **removed**: non-start status (DNQ, DNPQ, Withdrew, 107%) with 0 laps, or no grid slot and 0 laps | they never took the start, so they aren't a "driver entering the race" and would be easy fake negatives (1.8) |
| `grid = 0` (pit-lane start) | that race's last grid slot + 1, plus a `pit_lane_start` flag | 0 would look *better* than pole (1.8) |
| Modeling window | **2003+** | qualifying coverage is ~100% only from 2003 (1.7) |
| Standings | use the standings **after the previous round**, never the current race | same-race standings already contain the result (1.10) |
| Unused tables | `seasons`, `lap_times`, `pit_stops` | only a link / in-race data (1.10) |
""")

# ---------------------------------------------------------------- 2.1
md(r"""
## 2.1 Working copies, sentinels and text
""")

code(r'''
UNUSED_TABLES = {
    "seasons": "only year + Wikipedia url; races.year already holds the season (FK verified in 1.5)",
    "lap_times": "recorded during the race = post-race information for the race we predict (1.10)",
    "pit_stops": "recorded during the race = post-race information for the race we predict (1.10)",
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

display(pd.Series(UNUSED_TABLES, name="reason not used").to_frame())
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
`seasons`, `lap_times` and `pit_stops` aren't carried forward.

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

Every status is mapped to one category. The brief defines a **mechanical retirement** as a car component failing, excluding crashes, the driver, and regulation decisions.
Statuses whose cause can't be told from the label go to **unclear** instead of being guessed. Q3 can then report the result with and without them.
""")

code(r'''
import re

STATUS_GROUPS = {
    "did not start": {"Did not qualify", "Did not prequalify", "Withdrew", "107% Rule"},
    "regulation": {"Disqualified", "Excluded", "Underweight", "Not classified"},
    "accident": {"Accident", "Collision", "Collision damage", "Spun off", "Fatal accident"},
    "driver": {"Injury", "Injured", "Illness", "Driver unwell", "Physical", "Eye injury"},
    "unclear": {"Retired", "Damage", "Debris", "Puncture", "Tyre puncture", "Tyre", "Out of fuel", "Not restarted",
                "Stalled", "Safety concerns", "Safety", "Fuel rig", "Refuelling", "Wheel nut", "Front wing", "Broken wing"},
    "mechanical": {
        "Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics", "Electrical", "Radiator", "Suspension", "Brakes",
        "Differential", "Overheating", "Mechanical", "Driver Seat", "Driveshaft", "Fuel pressure", "Water pressure",
        "Wheel", "Throttle", "Steering", "Technical", "Electronics", "Heat shield fire", "Exhaust", "Oil leak",
        "Wheel rim", "Water leak", "Fuel pump", "Track rod", "Oil pressure", "Engine fire", "Engine misfire",
        "Pneumatics", "Handling", "Rear wing", "Fire", "Wheel bearing", "Fuel system", "Oil line", "Launch control",
        "Fuel", "Power loss", "Vibrations", "Drivetrain", "Ignition", "Chassis", "Battery", "Halfshaft", "Crankshaft",
        "Alternator", "Safety belt", "Oil pump", "Fuel leak", "Injection", "Distributor", "Turbo", "CV joint",
        "Water pump", "Spark plugs", "Fuel pipe", "Oil pipe", "Axle", "Water pipe", "Magneto", "Supercharger",
        "Power Unit", "ERS", "Brake duct", "Seat", "Undertray", "Cooling system",
    },
}


def status_category(name):
    if name == "Finished" or re.fullmatch(r"\+\d+ Laps?", name):
        return "finished"
    return next((cat for cat, names in STATUS_GROUPS.items() if name in names), None)


CLEAN["status"]["status_category"] = CLEAN["status"]["status"].map(status_category)
unmapped = CLEAN["status"].loc[CLEAN["status"]["status_category"].isna(), "status"].tolist()
assert not unmapped, f"unmapped statuses: {unmapped}"

display(CLEAN["status"].groupby("status_category")["status"]
        .agg(n_statuses="size", statuses=lambda s: ", ".join(sorted(s)[:12]) + (" ..." if len(s) > 12 else "")))
''')

md(r"""
**What this did.** All 139 statuses have a category, and the code asserts none is missed. `+N Laps` counts as *finished* (the car was running at the end, just lapped).
Borderline calls to defend: `Puncture`, `Damage`, `Debris` and `Front wing` are usually caused by contact, and `Retired` gives no cause, so they are *unclear*, not mechanical. `Not classified` (running but too few laps) is a *regulation* outcome.
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
NON_START = STATUS_GROUPS["did not start"]
by_status = res_c["status"].isin(NON_START) & (res_c["laps"] == 0)
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
# --- Duplicates: rule B (keep the entry the driver started in) ------------------------------------
n = len(res_c)
dup_pairs = int(res_c.duplicated(["raceId", "driverId"]).sum())
podium_any = res_c.groupby(["raceId", "driverId"])["podium"].transform("max")
res_c = (res_c.assign(_started=res_c["grid"] > 0)
         .sort_values(["raceId", "driverId", "_started", "resultId"], ascending=[True, True, False, True])
         .drop_duplicates(["raceId", "driverId"], keep="first")
         .drop(columns="_started"))
lost = res_c[res_c["podium"] < podium_any.loc[res_c.index]].merge(CLEAN["races"][["raceId", "year"]], on="raceId")
log_step("rule B: keep starting entry", "results", n, len(res_c), f"{dup_pairs} extra rows")
print(f"Duplicate rows removed: {n - len(res_c)}")
print(f"Drivers whose shared-drive podium is not credited under rule B: {len(lost)} "
      f"(seasons {lost['year'].min()}-{lost['year'].max()}, outside the 2003+ window)")

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
- **Rule B:** the kept row is chosen only with pre-race information (did the driver start in that car?). The cost is that a few 1950s drivers lose podium credit from a car they took over mid-race. That's documented and outside the modeling window.
- **Grid:** a pit-lane start becomes "behind the last car" (that race's highest grid + 1), with `pit_lane_start` keeping the information that it happened.
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
- `results_full`: every cleaned driver-race (all years) **with** post-race columns. Q1–Q3 need the finishing order and status.
- `model_base`: the modeling grain (one row per driver per race, 2003+) with **only pre-race columns** plus the target. Feature engineering (section 3) adds qualifying, form and standings features to it.
  The code asserts that no post-race column is in it. The podium rate (~14%) shows the class imbalance the metrics must handle (hence PR-AUC).
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

for child_name, child, col, parent, pcol in [
        ("results_full", results_full, "raceId", CLEAN["races"], "raceId"),
        ("results_full", results_full, "driverId", CLEAN["drivers"], "driverId"),
        ("results_full", results_full, "constructorId", CLEAN["constructors"], "constructorId"),
        ("results_full", results_full, "statusId", CLEAN["status"], "statusId"),
        ("model_base", model_base, "circuitId", CLEAN["circuits"], "circuitId"),
        ("qualifying", CLEAN["qualifying"], "raceId", CLEAN["races"], "raceId"),
        ("driver_standings_prerace", CLEAN["driver_standings_prerace"], "driverId", CLEAN["drivers"], "driverId"),
        ("constructor_standings_prerace", CLEAN["constructor_standings_prerace"], "constructorId", CLEAN["constructors"], "constructorId")]:
    orphans = int((~child[col].dropna().isin(parent[pcol])).sum())
    check(f"FK {child_name}.{col} -> {pcol}", orphans == 0, f"{orphans} orphans")

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
**What this means.** The grain is fixed (unique `raceId + driverId`), every key still resolves, no placeholder text is left, `grid` has no fake zeros, and every 2003+ race has exactly 3 podium rows (the shared-drive anomaly only existed before 1965).
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
- **4 races with > 3 podium rows** are all 1951/1955 cross-driver shared drives (1.6). They're expected and outside the modeling window.
- **Gaps in the pre-race standings:** `driver_standings` early in a season only lists drivers who already have an entry (2003 round 1 has 11 standings rows for 20 starters). Missing values after round 1 therefore mean **"no points yet"** (or the driver didn't race the previous round), not a broken join. **Rule for feature engineering (section 3):** fill `pre_points` and `pre_wins` with 0, and give `pre_position` a "behind everyone ranked" value plus a `no_standing_yet` flag. Round 1 always gets this treatment.
""")

# ---------------------------------------------------------------- 2.11
md(r"""
## 2.11 EDA before vs. after cleaning
""")

code(r'''
display(pd.DataFrame(cleaning_log))

raw_res = RAW["results"].assign(grid=lambda d: d["grid"].astype(int), podium=lambda d: (d["positionOrder"].astype(int) <= 3))
raw_res = raw_res.merge(RAW["races"][["raceId", "year"]], on="raceId").assign(year=lambda d: d["year"].astype(int))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

log_df = pd.DataFrame(cleaning_log)
axes[0].barh(log_df["step"], log_df["rows removed"], color="#C44E52")
axes[0].set_title("Rows removed per cleaning step")
axes[0].set_xlabel("rows")
for y, v in enumerate(log_df["rows removed"]):
    axes[0].text(v, y, f" {v:,}", va="center", fontsize=8)

bins = np.arange(-0.5, 36.5, 1)
axes[1].hist(raw_res["grid"], bins=bins, alpha=0.5, label="raw", color="#8C8C8C")
axes[1].hist(results_full["grid"], bins=bins, alpha=0.6, label="clean", color="#4C72B0")
axes[1].set_title("grid distribution: raw vs clean")
axes[1].set_xlabel("grid (raw 0 = no slot)")
axes[1].legend()

rate_raw = 100 * raw_res.groupby("year")["podium"].mean()
rate_clean = 100 * results_full.groupby("year")["podium"].mean()
axes[2].plot(rate_raw.index, rate_raw.values, label="raw results", color="#8C8C8C")
axes[2].plot(rate_clean.index, rate_clean.values, label="clean results", color="#4C72B0")
axes[2].axvspan(MODEL_START_YEAR, rate_clean.index.max(), color="#55A868", alpha=0.1, label="modeling window")
axes[2].set_title("Podium rate per season")
axes[2].set_ylabel("% of rows with podium")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

display(pd.DataFrame({
    "raw rows": {t: len(RAW[t]) for t in CLEAN if t in RAW},
    "clean rows": {t: len(CLEAN[t]) for t in CLEAN if t in RAW},
}).assign(removed=lambda d: d["raw rows"] - d["clean rows"]))
''')

md(r"""
**What the comparison shows.**
- **Rows removed:** most removals are non-starters, concentrated before the 1990s, when many entrants failed to (pre)qualify. Rule B removes 88 rows.
- **Grid:** the raw spike at 0 is gone. Pit-lane starters now sit behind the last grid slot, so `grid` is a clean ordinal feature.
- **Podium rate:** in the raw data, seasons with many non-qualifiers have an artificially *low* podium rate (the denominator includes cars that never started). After cleaning, the rate is stable at roughly 3 / starters per race. Inside the 2003+ window the change is small, which is good: the modeling data was mostly clean already.
- Only `results` lost rows; every other table keeps all its rows (only values and types changed).
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

# ================================================================ 3. DATA-ENGINEERING QUESTIONS
md(r"""
# 3. Data-Engineering Questions

Each question is answered from the **cleaned** tables (section 2) using multi-table joins. For every question we state:
- **Tables & joins:** which tables are combined, and on which keys.
- **Grain:** what one row represents.
- **Missingness:** what is missing and how it is handled.
- **Cutoff:** whether the pre-race cutoff matters.

Then come the result, a plot, and a plain-language answer.
""")

code(r'''
circuits_c = CLEAN["circuits"][["circuitId", "name", "country"]].rename(columns={"name": "circuit"})
constructors_c = CLEAN["constructors"][["constructorId", "name"]].rename(columns={"name": "constructor"})
quali_pos = CLEAN["qualifying"][["raceId", "driverId", "position"]].rename(columns={"position": "quali_pos"})
''')

# ---------------------------------------------------------------- Q1
md(r"""
## 3.1 Q1: Which circuits convert a front-row start into a podium most often, and did that change in the hybrid era (2014+)?

**In simple words:** if a driver starts in the first two places, how often does he finish in the top three, and at which tracks is that most likely?

- **Tables & joins:** `results_full` (grid, podium, circuit, year) ⟶ `qualifying` on `raceId + driverId` (qualifying position) ⟶ `circuits` on `circuitId` (circuit name).
- **Grain:** one row = one driver in one race.
- **Two definitions of "front row"**, because they differ (penalties):
  - **started** on the front row = `grid` 1 or 2 (the main answer: it's where the car actually lined up);
  - **qualified** on the front row = qualifying position 1 or 2 (shown for comparison).
- **Missingness:** qualifying is complete only from 2003 (1.7), so the comparison is **2003–2013 (pre-hybrid) vs 2014–2024 (hybrid)**. Circuits need **≥ 8 front-row starts in each era** (about 4 races) so a single race doesn't decide the ranking.
- **Cutoff:** grid and qualifying are known before the race, so both are valid "starting" information.
""")

code(r'''
ERAS = ["2003-2013 (pre-hybrid)", "2014-2024 (hybrid)"]
q1 = (results_full[results_full["year"] >= 2003]
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
q1_all = results_full.merge(circuits_c, on="circuitId", how="left", validate="many_to_one")
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
- **The conversion rate depends strongly on the decade:** about **46–52%** from the 1950s to the 1980s, rising to **74–75%** since 2010. Older cars broke down far more often, so even a front-row car often didn't finish.
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
q_best = CLEAN["qualifying"][["raceId", "driverId", "position", "q1_sec", "q2_sec", "q3_sec"]].copy()
q_best["best_sec"] = q_best[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)
q_best["gap_to_pole_pct"] = 100 * (q_best["best_sec"] / q_best.groupby("raceId")["best_sec"].transform("min") - 1)
qg = results_full[results_full["year"] >= 2003].merge(q_best[["raceId", "driverId", "position", "gap_to_pole_pct"]],
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

- **Tables & joins:** `results_full` ⟶ `drivers` on `driverId` (`home_country`, built in 2.4) ⟶ `circuits` on `circuitId` (`country`). **Home race** = `home_country == circuit country`.
- **Grain:** one row = one driver starting one race (all seasons; non-starters were removed in 2.6).
- **Controlling for starting position:** drivers are grouped into **grid bands** (1–3, 4–6, 7–10, 11+). Home and away drivers are compared **inside the same band**. The band differences are then averaged, weighted by how many home starts fall in each band ("grid-adjusted difference").
- **Uncertainty:** a bootstrap (resampling the rows 500 times) gives a 95% interval for the adjusted difference.
- **Missingness:** none. Every driver has a `home_country`, and every cleaned row has a valid grid.
- **Cutoff:** grid is pre-race information, so it's a valid control.
""")

code(r'''
nat = CLEAN["drivers"][["driverId", "home_country"]]
q2 = (results_full.merge(nat, on="driverId", how="left", validate="many_to_one")
      .merge(circuits_c, on="circuitId", how="left", validate="many_to_one"))
q2["home"] = q2["home_country"] == q2["country"]
GRID_BANDS = ["1-3", "4-6", "7-10", "11+"]
q2["grid_band"] = pd.cut(q2["grid"].astype(float), [0, 3, 6, 10, 99], labels=GRID_BANDS)


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
                           "2003-2024 only": summarize(q2[q2["year"] >= 2003])})
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
- **Without controlling for grid**, home drivers even look *worse* (≈ 11.2% vs 13.7% podiums). That's misleading, because home drivers start further back on average (grid ≈ 13.4 vs 11.7). In older seasons many local drivers entered only their home race in weaker cars.
- **Comparing drivers who started from the same grid band**, the home rate is almost identical to the away rate in **every** band, just slightly lower (e.g. grid 1–3: ≈ 54% home vs 56% away).
- The **grid-adjusted difference is close to zero** (about −0.6 points), and the 95% interval includes 0, so the small gap is consistent with chance. The modern era (2003+) points the same way.
- **Takeaway:** what looks like a "home effect" is really a **starting-position effect**. Where you start matters a lot; whether you're at home doesn't. For the model this suggests a "home race" feature will add little.

*Note:* controlling with grid bands is a simple, transparent method. Within a band, small grid differences remain, but the per-band rates are so close that a finer control wouldn't change the conclusion.
""")

# ---------------------------------------------------------------- Q3
md(r"""
## 3.3 Q3: Which constructors had the highest mechanical-retirement rate in 2014–2021 vs 2022–2024?

**In simple words:** out of all the races a team's cars started, how often did one stop because something on the car broke?

- **Tables & joins:** `results_full` (already joined to `status` and its category in 2.5–2.6) ⟶ `constructors` on `constructorId` (team name), filtered by `year` (from `races`).
- **Grain:** one row = one car starting one race. **Rate = mechanical retirements ÷ race starts** for that team in that period.
- **Mechanical retirement:** `status_category == "mechanical"` (engine, gearbox, hydraulics, power unit, ...; 2.5). Crashes, driver problems and regulation decisions are excluded, as the brief defines. Statuses whose cause is unclear (`Retired`, `Puncture`, `Damage`, ...) are excluded from the main rate and shown as a **sensitivity check** ("upper bound").
- **Missingness:** none. Every status has a category, and non-starters were removed, so the denominator counts real starts only.
- **Small teams:** only teams with **≥ 40 starts** in a period are ranked. Team names follow the dataset: renamed teams (e.g. Toro Rosso → AlphaTauri → RB, Force India → Racing Point → Aston Martin) appear as separate entries.
- **Cutoff:** not relevant. This is a historical post-race analysis, not a feature.
""")

code(r'''
PERIODS = ["2014-2021", "2022-2024"]
q3 = (results_full[results_full["year"] >= 2014]
      .merge(constructors_c, on="constructorId", how="left", validate="many_to_one"))
q3["period"] = np.where(q3["year"] <= 2021, PERIODS[0], PERIODS[1])
q3["mechanical"] = q3["status_category"] == "mechanical"
q3["mechanical_or_unclear"] = q3["status_category"].isin(["mechanical", "unclear"])

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
    ax.barh(d["constructor"], d["mechanical + unclear %"], color="#F2C6A0", label="+ unclear causes (upper bound)")
    ax.barh(d["constructor"], d["mechanical %"], color="#C44E52", label="mechanical")
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
- **2014–2021 (first hybrid engines):** the least reliable teams were **Lotus F1 (≈ 19% of starts ended with a car failure), Toro Rosso (≈ 16%) and McLaren (≈ 14%)**, followed by Renault. McLaren's years with the early Honda engine (2015–2017) are a well-known cause.
- **2022–2024 (new car rules):** the highest rates belong to **Alfa Romeo (≈ 12.5%) and Alpine (≈ 11%)**, then AlphaTauri (≈ 8%).
- **Big picture:** cars became **much more reliable**. The field-wide mechanical-retirement rate fell from about **9.5% to 6%** of starts, and the top teams in 2022–2024 (Red Bull, McLaren, Mercedes) retired from mechanical failures in only about 4% of starts.
- **Robustness:** counting the "unclear" statuses as mechanical (light bars) raises some rates by up to ~6 points but doesn't change the top two in either period.

*Caution:* in 2022–2024 each team has at most ~136 starts, so one extra failure moves the rate by ~0.7 points. The differences between the middle teams are small.
""")

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
