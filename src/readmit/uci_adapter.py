"""Optional real-data benchmark on the UCI "Diabetes 130-US hospitals (1999-2008)" dataset.

Why it exists
-------------
The dataset (Strack et al., 2014; CC BY 4.0; 101,766 encounters) has a documented 30-day
readmission field, so a *label-valid* benchmark is possible. It does NOT contain admission or
discharge dates, facility identifiers, laboratory time series or dated medication records, so
it cannot support this project's point-in-time features, temporal validation, facility
monitoring or day-level capacity simulation. It is therefore used only as an external
sanity benchmark of the modelling code, with a patient-grouped (not temporal) split.

STATUS: the adapter is unit-tested on a small schema-faithful fixture generated inside the
tests. It has NOT been run on the real file in the build environment (network egress to the
UCI archive was blocked), so no real-data results are reported anywhere in this repository.

Usage
-----
    python -m readmit.uci_adapter --csv /path/to/diabetic_data.csv
    python -m readmit.uci_adapter --download          # only works if your network allows it
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
# discharge_disposition_id values for death / hospice in the dataset's IDS_mapping.csv
# (11 Expired, 13 Hospice/home, 14 Hospice/medical facility, 19-21 Expired variants).
# Verify against the IDS_mapping.csv that ships with the download.
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
           "number_emergency", "number_inpatient", "number_diagnoses", "age_mid"]
CATEGORICAL = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "payer_code",
               "medical_specialty", "dx1_group", "max_glu_serum", "A1Cresult", "change", "diabetesMed"]
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"]


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:  # noqa: S310 (fixed https URL)
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    z.extractall(dest)
    return dest / "diabetic_data.csv"


def load_uci(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["?", "Unknown/Invalid"], low_memory=False)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Label, cohort exclusions and a compact feature table."""
    d = df.copy()
    required = {"patient_nbr", "readmitted", "discharge_disposition_id", "time_in_hospital"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"not a UCI Diabetes-130 file; missing columns {sorted(missing)}")
    d["readmit_30d"] = (d["readmitted"] == "<30").astype(int)  # '>30' and 'NO' are both negatives
    d = d[~d["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)].copy()
    # age arrives as a bracket like "[70-80)"
    lo = d["age"].astype(str).str.extract(r"\[(\d+)-(\d+)\)").astype(float)
    d["age_mid"] = (lo[0] + lo[1]) / 2
    d["dx1_group"] = d["diag_1"].astype(str).str[:3].where(d["diag_1"].notna())
    top = d["dx1_group"].value_counts().head(30).index
    d["dx1_group"] = d["dx1_group"].where(d["dx1_group"].isin(top), "other")
    spec = d["medical_specialty"].value_counts().head(15).index
    d["medical_specialty"] = d["medical_specialty"].where(d["medical_specialty"].isin(spec), "other")
    meds = [c for c in MED_COLS if c in d.columns]
    d["n_med_changes"] = d[meds].isin(["Up", "Down"]).sum(axis=1)
    d["n_meds_taken"] = d[meds].isin(["Steady", "Up", "Down"]).sum(axis=1)
    for c in CATEGORICAL:
        if c in d.columns:
            d[c] = d[c].astype(str)
    return d


def grouped_split(d: pd.DataFrame, seed: int = 0, test: float = 0.2, val: float = 0.1):
    """Patient-grouped split (the dataset has no dates, so a temporal split is impossible)."""
    rng = np.random.default_rng(seed)
    pats = np.sort(d["patient_nbr"].unique())
    rng.shuffle(pats)
    n = len(pats)
    te, va = set(pats[: int(n * test)]), set(pats[int(n * test): int(n * (test + val))])
    g = d["patient_nbr"]
    return (~g.isin(te | va)).to_numpy(), g.isin(va).to_numpy(), g.isin(te).to_numpy()


def benchmark(d: pd.DataFrame, seed: int = 0, n_boot: int = 100) -> dict:
    import lightgbm as lgb
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from .models import evaluate as ev

    num = [c for c in NUMERIC + ["n_med_changes", "n_meds_taken"] if c in d.columns]
    cat = [c for c in CATEGORICAL if c in d.columns]
    tr, va, te = grouped_split(d, seed)
    y = d["readmit_30d"].to_numpy()
    X = d[num + cat].copy()
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]), num),
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), cat)])
    lr = Pipeline([("p", pre), ("m", LogisticRegression(C=0.05, max_iter=3000))]).fit(X[tr], y[tr])
    Xl = X.copy()
    for c in cat:
        Xl[c] = Xl[c].astype("category")
    gb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=60, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=seed)
    gb.fit(Xl[tr], y[tr], eval_set=[(Xl[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
    preds = {"logistic_regression": lr.predict_proba(X[te])[:, 1], "lightgbm": gb.predict_proba(Xl[te])[:, 1]}
    groups = d.loc[te, "patient_nbr"].to_numpy()
    boot = ev.cluster_bootstrap_metrics(y[te], preds, groups, n_boot=n_boot, seed=seed)
    return {
        "dataset": "UCI Diabetes 130-US hospitals 1999-2008 (CC BY 4.0)", "split": "patient-grouped (no dates available)",
        "n_encounters": int(len(d)), "n_patients": int(d["patient_nbr"].nunique()), "readmit_30d_rate": float(y.mean()),
        "test_n": int(te.sum()),
        "models": {k: ev.core_metrics(y[te], p) for k, p in preds.items()}, "bootstrap": boot,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("artifacts/uci_benchmark.json"))
    a = ap.parse_args(argv)
    if a.download:
        try:
            a.csv = download(Path("data/uci"))
        except Exception as e:  # network blocked, etc.
            print(f"Download failed ({e}). Download the zip manually from the UCI archive and pass --csv.", file=sys.stderr)
            return 2
    if not a.csv or not a.csv.exists():
        print("Provide --csv /path/to/diabetic_data.csv (or --download).", file=sys.stderr)
        return 2
    res = benchmark(prepare(load_uci(a.csv)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    for k, v in res["models"].items():
        print(f"{k}: AUROC {v['auroc']:.3f}  PR-AUC {v['pr_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
