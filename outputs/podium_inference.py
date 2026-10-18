# podium_inference.py: written by f1_podium_prediction.ipynb, section 11. Usage outside the notebook:
#   from podium_inference import PodiumPredictor
#   predictor = PodiumPredictor("outputs/model_artifacts_v2")
#   predictor.predict_proba(records)         # one row per driver: podium probability
#   predictor.predict_race(race_entries)     # every starter of one race: the 3 most likely podium finishers
# The model estimates "podium or not". The three picks of a race are NOT a prediction of 1st, 2nd and 3rd place.
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["grid", "pit_lane_start", "gap_q1_pct", "grid_minus_quali", "reached_q3", "no_q3_format", "quali_missing",
            "drv_champ_pos", "cons_points_share", "drv_no_standing_yet", "cons_no_standing_yet", "drv_podium_rate_5",
            "drv_avg_finish_5", "drv_dnf_rate_5", "cons_podium_rate_5", "n_starters", "drv_experience", "drv_age"]
FLAGS = ["pit_lane_start", "reached_q3", "no_q3_format", "quali_missing", "drv_no_standing_yet", "cons_no_standing_yet"]
# Allowed range of a known value (None = no bound).
RANGES = {"grid": (1, None), "gap_q1_pct": (0, None), "drv_champ_pos": (1, None), "cons_points_share": (0, 1),
          "drv_podium_rate_5": (0, 1), "drv_avg_finish_5": (1, None), "drv_dnf_rate_5": (0, 1), "cons_podium_rate_5": (0, 1),
          "n_starters": (1, None), "drv_experience": (0, None), "drv_age": (15, 70)}
MUST_BE_KNOWN = ["grid"]            # after qualifying the starting position is always known
RACE_KEYS = ["raceId", "driverId"]
PRIMARY = "A_logistic_regression"


class PodiumInputError(ValueError):
    pass


class PodiumPredictor:
    def __init__(self, artifact_dir, expected_fingerprint=None):
        folder = Path(artifact_dir)
        spec = importlib.util.spec_from_file_location("frozen_v2_loader", folder / "load_artifacts.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        models, manifest = loader.load_model_artifacts(folder)        # checks every file against its SHA-256
        if expected_fingerprint is not None and manifest["frozen fingerprint"] != expected_fingerprint:
            raise RuntimeError("the model files are not the expected frozen version")
        primary = [name for name, m in manifest["models"].items() if m.get("primary")]
        if primary != [PRIMARY] or manifest["input columns"] != FEATURES:
            raise RuntimeError(f"unexpected model files: primary {primary}, input columns {manifest['input columns']}")
        self.model_name = PRIMARY
        self.fingerprint = manifest["frozen fingerprint"]
        self._predict = models[PRIMARY]["predict"]

    @staticmethod
    def _frame(records):
        if isinstance(records, dict):
            records = [records]
        if isinstance(records, list):
            if not records or not all(isinstance(r, dict) for r in records):
                raise PodiumInputError("records must be a dict, a non-empty list of dicts, or a DataFrame")
            records = pd.DataFrame(records)
        if not isinstance(records, pd.DataFrame):
            raise PodiumInputError("records must be a dict, a non-empty list of dicts, or a DataFrame")
        if records.empty:
            raise PodiumInputError("no records given")
        return records.reset_index(drop=True)

    def validate(self, records):
        # Returns (the records, their 18 features as floats); raises PodiumInputError on any schema problem.
        df = self._frame(records)
        absent = [c for c in FEATURES if c not in df.columns]
        if absent:
            raise PodiumInputError(f"missing feature column(s) {absent}: all {len(FEATURES)} features must be present "
                                   "(use None / NaN for a value that is unknown)")
        X = pd.DataFrame(index=df.index)
        for c in FEATURES:
            num = pd.to_numeric(df[c], errors="coerce")
            bad = df[c].notna() & num.isna()
            if bad.any():
                raise PodiumInputError(f"non-numeric value(s) in {c}: {df.loc[bad, c].tolist()[:5]}")
            X[c] = num.astype(float)
        if np.isinf(X.to_numpy()).any():
            raise PodiumInputError("infinite values are not allowed")
        for c in MUST_BE_KNOWN:
            if X[c].isna().any():
                raise PodiumInputError(f"{c} must be known (rows {X.index[X[c].isna()].tolist()})")
        for c in FLAGS:
            bad = X[c].notna() & ~X[c].isin([0, 1])
            if bad.any():
                raise PodiumInputError(f"{c} must be 0 or 1 (or unknown); got {X.loc[bad, c].tolist()[:5]}")
        for c, (lo, hi) in RANGES.items():
            v = X[c]
            bad = v.notna() & (((v < lo) if lo is not None else False) | ((v > hi) if hi is not None else False))
            if bad.any():
                raise PodiumInputError(f"{c} must lie in [{lo}, {hi}]; got {v[bad].tolist()[:5]}")
        return df, X

    def predict_proba(self, records):
        df, X = self.validate(records)
        p = np.asarray(self._predict(X), dtype=float)
        if p.shape != (len(X),) or not np.all((p >= 0) & (p <= 1)):
            raise RuntimeError("the model returned invalid probabilities")
        out = df[[c for c in df.columns if c not in FEATURES]].copy()     # identifiers and extra columns, unchanged
        out["podium_probability"] = p
        out["missing_inputs"] = [", ".join(X.columns[row]) or "-" for row in X.isna().to_numpy()]
        return out

    def predict_race(self, race_entries, allow_incomplete=False):
        df = self._frame(race_entries)
        absent = [c for c in RACE_KEYS if c not in df.columns]
        if absent:
            raise PodiumInputError(f"a race prediction needs the identifier columns {RACE_KEYS}; missing {absent}")
        if df[RACE_KEYS].isna().any().any():
            raise PodiumInputError("raceId and driverId must be known for every driver")
        if df["raceId"].nunique() != 1:
            raise PodiumInputError(f"one race per call; got raceIds {sorted(df['raceId'].unique().tolist())}")
        if df["driverId"].duplicated().any():
            raise PodiumInputError(f"duplicate driverId(s): {df.loc[df['driverId'].duplicated(), 'driverId'].tolist()}")
        _, X = self.validate(df)
        if X["n_starters"].isna().any() or X["n_starters"].nunique() != 1:
            raise PodiumInputError("n_starters must be known and the same for every driver of the race")
        n_starters = int(X["n_starters"].iloc[0])
        complete = len(df) == n_starters
        if not complete:
            msg = f"incomplete race entry: {len(df)} drivers supplied, but n_starters = {n_starters}"
            if not allow_incomplete:
                raise PodiumInputError(msg + " (pass allow_incomplete=True to rank only the supplied drivers)")
            warnings.warn(msg + "; the three picks are only among the supplied drivers")
        if len(df) < 3:
            raise PodiumInputError("a race needs at least 3 drivers")
        # Same order as the evaluation: ties in probability are broken by grid position, then driverId.
        order = df.assign(_grid=X["grid"]).sort_values(["_grid", "driverId"], kind="mergesort").index
        df = df.loc[order].reset_index(drop=True)
        out = self.predict_proba(df)
        out["grid"] = pd.to_numeric(df["grid"]).to_numpy()
        out["probability_rank"] = out["podium_probability"].rank(method="first", ascending=False).astype(int)
        out["predicted_podium"] = out["probability_rank"] <= 3
        out["complete_race"] = complete
        return out.sort_values("probability_rank").reset_index(drop=True)
