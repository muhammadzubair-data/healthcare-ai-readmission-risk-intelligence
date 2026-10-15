"""Model training: splits, model zoo, calibration and uncertainty ensemble.

Validation design
-----------------
* Temporal: train <= `train_end`; validation (early stopping, C / leaf selection, isotonic
  calibration) up to `val_end`; the test set is everything after `val_end` with complete
  30-day follow-up. The validation window doubles as a gap so that no training label's
  follow-up period overlaps the test period.
* Patient-level: a random fraction of patients is removed from training and validation
  entirely. Their future-period encounters form `test_unseen` (new-patient generalisation);
  the remaining future encounters form `test` (the deployment-like primary test).
* Grouped K-fold CV by patient inside the training period quantifies sampling variability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ..clinical import lace_style_score
from ..config import Config
from ..features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from . import uncertainty

MODEL_NAMES = ["lace_style", "logistic_regression", "random_forest", "lightgbm", "lightgbm_calibrated"]
MODEL_LABELS = {
    "lace_style": "LACE-style heuristic (Platt-scaled)",
    "logistic_regression": "Logistic regression (L2)",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "lightgbm_calibrated": "LightGBM + isotonic calibration",
}


# ------------------------------------------------------------------------------ splits
def make_splits(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.model.seed)
    pats = np.sort(df["patient_id"].unique())
    holdout = set(rng.choice(pats, size=int(len(pats) * cfg.split.patient_holdout_frac), replace=False))
    is_hold = df["patient_id"].isin(holdout).to_numpy()
    ts = df["discharge_ts"]
    tr_end = pd.Timestamp(cfg.split.train_end) + pd.Timedelta(days=1)
    va_end = pd.Timestamp(cfg.split.val_end) + pd.Timedelta(days=1)
    return {
        "train": ((ts < tr_end).to_numpy()) & ~is_hold,
        "val": ((ts >= tr_end) & (ts < va_end)).to_numpy() & ~is_hold,
        "test": (ts >= va_end).to_numpy() & ~is_hold,
        "test_unseen": (ts >= va_end).to_numpy() & is_hold,
        "holdout_patient_flag": is_hold,
    }


# ------------------------------------------------------------------------------ preprocessing
def sklearn_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def rf_preprocessor() -> ColumnTransformer:
    cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def category_levels(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}


def prep_frame(df: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Feature frame with categorical dtypes pinned to the training levels (unseen -> NaN)."""
    X = df[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    for c in NUMERIC_FEATURES:
        X[c] = X[c].astype(float)
    return X


def _sk_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    return X


# ------------------------------------------------------------------------------ bundle
@dataclass
class Bundle:
    levels: dict
    lace_platt: LogisticRegression
    logreg: Pipeline
    logreg_C: float
    rf: Pipeline
    lgbm: lgb.Booster
    lgbm_params: dict
    best_iteration: int
    isotonic: IsotonicRegression
    tier_cutoffs: tuple[float, float]
    ensemble: list = field(default_factory=list)
    cv: dict = field(default_factory=dict)
    selection_log: list = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = prep_frame(df, self.levels)
        lace = lace_style_score(df["los_days"], df["acute_admission"], df["charlson_score"], df["ed_visits_prior_6m"])
        raw = self.lgbm.predict(X, num_iteration=self.best_iteration)
        return {
            "lace_style": self.lace_platt.predict_proba(lace[:, None])[:, 1],
            "logistic_regression": self.logreg.predict_proba(_sk_frame(X))[:, 1],
            "random_forest": self.rf.predict_proba(_sk_frame(X))[:, 1],
            "lightgbm": raw,
            "lightgbm_calibrated": self.isotonic.predict(raw),
        }

    def predict_final(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict(df)["lightgbm_calibrated"]

    def shap_contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Exact TreeSHAP values in log-odds for the (uncalibrated) LightGBM model, plus the base value."""
        X = prep_frame(df, self.levels)
        c = self.lgbm.predict(X, num_iteration=self.best_iteration, pred_contrib=True)
        return c[:, :-1], c[:, -1]


# ------------------------------------------------------------------------------ training
def _fit_lgbm(Xtr, ytr, Xva, yva, params, max_rounds, es):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=CATEGORICAL_FEATURES, reference=dtr, free_raw_data=False)
    p = dict(params, metric="binary_logloss")
    booster = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(es, verbose=False)])
    return booster


def train_all(df: pd.DataFrame, splits: dict[str, np.ndarray], cfg: Config, log=print) -> Bundle:
    levels = category_levels(df)
    tr, va = splits["train"], splits["val"]
    dtr, dva = df[tr], df[va]
    ytr, yva = dtr["readmit_30d"].to_numpy(), dva["readmit_30d"].to_numpy()
    Xtr, Xva = prep_frame(dtr, levels), prep_frame(dva, levels)
    sel: list = []

    # --- LACE-style baseline (Platt-scaled on training scores)
    lace_tr = lace_style_score(dtr["los_days"], dtr["acute_admission"], dtr["charlson_score"], dtr["ed_visits_prior_6m"])
    lace_platt = LogisticRegression(C=1e6, max_iter=500).fit(lace_tr[:, None], ytr)

    # --- logistic regression: choose C on validation log-loss
    best = None
    for C in (0.01, 0.05, 0.2, 1.0):
        lr = Pipeline([("pre", sklearn_preprocessor()), ("clf", LogisticRegression(C=C, max_iter=4000))])
        lr.fit(_sk_frame(Xtr), ytr)
        ll = log_loss(yva, lr.predict_proba(_sk_frame(Xva))[:, 1])
        sel.append(dict(model="logistic_regression", C=C, val_log_loss=float(ll)))
        if best is None or ll < best[0]:
            best = (ll, C, lr)
    logreg, logreg_C = best[2], best[1]
    log(f"  logreg C={logreg_C} val logloss={best[0]:.4f}")

    # --- random forest (bagging comparator)
    rf = Pipeline([("pre", rf_preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=60, max_features=0.35,
                                                  n_jobs=2, random_state=cfg.model.seed))])
    rf.fit(_sk_frame(Xtr), ytr)
    log(f"  random forest val AUROC={roc_auc_score(yva, rf.predict_proba(_sk_frame(Xva))[:, 1]):.4f}")

    # --- LightGBM: small search over tree size, early stopping on validation log-loss
    best = None
    for leaves, mcs in ((15, 100), (31, 80), (63, 120)):
        params = dict(cfg.model.lgbm_params, num_leaves=leaves, min_child_samples=mcs, seed=cfg.model.seed)
        b = _fit_lgbm(Xtr, ytr, Xva, yva, params, cfg.model.max_rounds, cfg.model.early_stopping)
        ll = b.best_score["valid_0"]["binary_logloss"]
        sel.append(dict(model="lightgbm", num_leaves=leaves, min_child_samples=mcs, best_iteration=int(b.best_iteration), val_log_loss=float(ll)))
        log(f"  lightgbm leaves={leaves} best_iter={b.best_iteration} val logloss={ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, params, b)
    lgbm_params, lgbm = best[1], best[2]
    best_iter = int(lgbm.best_iteration)

    # --- isotonic calibration fitted on the validation set only
    raw_va = lgbm.predict(Xva, num_iteration=best_iter)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, yva)
    cal_va = iso.predict(raw_va)
    cutoffs = (float(np.quantile(cal_va, cfg.model.tier_quantiles[0])), float(np.quantile(cal_va, cfg.model.tier_quantiles[1])))

    # --- grouped CV on the training period (patient-level folds)
    cv = {"logistic_regression": [], "lightgbm": []}
    gkf = GroupKFold(n_splits=cfg.split.cv_folds)
    for k, (a, b) in enumerate(gkf.split(Xtr, ytr, groups=dtr["patient_id"].to_numpy())):
        lr_k = Pipeline([("pre", sklearn_preprocessor()), ("clf", LogisticRegression(C=logreg_C, max_iter=4000))])
        lr_k.fit(_sk_frame(Xtr.iloc[a]), ytr[a])
        cv["logistic_regression"].append(float(roc_auc_score(ytr[b], lr_k.predict_proba(_sk_frame(Xtr.iloc[b]))[:, 1])))
        gb = lgb.train(dict(lgbm_params, seed=cfg.model.seed + k), lgb.Dataset(Xtr.iloc[a], label=ytr[a], categorical_feature=CATEGORICAL_FEATURES),
                       num_boost_round=best_iter)
        cv["lightgbm"].append(float(roc_auc_score(ytr[b], gb.predict(Xtr.iloc[b]))))
    log(f"  grouped CV AUROC  LR={np.mean(cv['logistic_regression']):.4f}  LGBM={np.mean(cv['lightgbm']):.4f}")

    # --- bootstrap ensemble for uncertainty bands
    ens = uncertainty.train_bootstrap_ensemble(
        Xtr, ytr, dtr["patient_id"].to_numpy(), dict(lgbm_params, metric="binary_logloss"), best_iter,
        cfg.model.ensemble_size, cfg.model.seed, CATEGORICAL_FEATURES,
    )
    return Bundle(levels, lace_platt, logreg, logreg_C, rf, lgbm, lgbm_params, best_iter, iso, cutoffs, ens, cv, sel)


# ------------------------------------------------------------------------------ persistence
def save_bundle(b: Bundle, directory: Path, commit_ensemble: bool = False) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    b.lgbm.save_model(str(directory / "lightgbm.txt"), num_iteration=b.best_iteration)
    joblib.dump(b.logreg, directory / "logreg.joblib", compress=3)
    joblib.dump(b.rf, directory / "random_forest.joblib", compress=5)
    joblib.dump(b.lace_platt, directory / "lace_platt.joblib")
    joblib.dump(b.isotonic, directory / "isotonic.joblib")
    if b.ensemble:
        ed = directory / "ensemble"
        ed.mkdir(exist_ok=True)
        for i, m in enumerate(b.ensemble):
            m.save_model(str(ed / f"member_{i:02d}.txt"))
    meta = dict(levels=b.levels, features=FEATURE_COLUMNS, numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES,
                best_iteration=b.best_iteration, tier_cutoffs=list(b.tier_cutoffs), lgbm_params=b.lgbm_params,
                logreg_C=b.logreg_C, cv_auroc=b.cv, selection_log=b.selection_log)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_lightgbm_for_app(directory: Path) -> tuple[lgb.Booster, dict]:
    directory = Path(directory)
    return lgb.Booster(model_file=str(directory / "lightgbm.txt")), json.loads((directory / "meta.json").read_text())
