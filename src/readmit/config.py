"""Central configuration for the readmission platform.

All tunable design targets live here so that the pipeline, tests and the app
agree on a single source of truth. Values are *design parameters*, not results.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    artifacts: Path = ROOT / "artifacts"
    models: Path = ROOT / "artifacts" / "models"
    docs: Path = ROOT / "docs"

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.artifacts, self.models, self.docs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 42
    n_patients: int = 90_000
    n_facilities: int = 10
    window_start: str = "2021-01-01"
    window_end: str = "2025-12-31"
    # Intercept of the simulator's readmission hazard, tuned so the recorded 30-day
    # readmission rate is roughly 14-16% (a plausible all-cause range, not a sourced figure).
    base_logit: float = -2.80
    # Global shrinkage of all simulated effect sizes and sd of hidden (unobservable) encounter-level noise.
    # Chosen so that discrimination lands in a realistic range (see docs/DATA_CARD.md).
    effect_scale: float = 0.62
    hidden_noise_sd: float = 0.55
    # Fraction of true readmissions that happen outside the network and are
    # therefore invisible in the data (label noise that real systems also have).
    out_of_network_rate: float = 0.06
    # Documented, injected data-quality defects (exercise the DQ monitor).
    dq_duplicate_rate: float = 0.002
    dq_impossible_lab_rate: float = 0.003
    dq_bad_dates_rate: float = 0.0005
    dq_unit_change_facility: str = "F04"
    dq_unit_change_date: str = "2023-06-01"
    # Documented, injected facility signal: used to check that monitoring detects it.
    drift_facility: str = "F07"
    drift_start: str = "2024-07-01"
    drift_logit_shift: float = 0.35


@dataclass(frozen=True)
class CohortConfig:
    followup_days: int = 30
    # CMS-style index-admission exclusions (documented in docs/DATA_CARD.md).
    excluded_dispositions: tuple[str, ...] = ("EXPIRED", "HOSPICE", "TRANSFER_ACUTE")
    min_age: int = 18


@dataclass(frozen=True)
class SplitConfig:
    train_end: str = "2023-12-31"
    val_end: str = "2024-06-30"
    # Test period runs from val_end to the last date with a complete 30-day follow-up.
    patient_holdout_frac: float = 0.15
    cv_folds: int = 5


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    ensemble_size: int = 10
    lgbm_params: dict = field(
        default_factory=lambda: dict(
            objective="binary",
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=80,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            reg_lambda=5.0,
            verbosity=-1,
            n_jobs=2,
        )
    )
    max_rounds: int = 1500
    early_stopping: int = 60
    capacity_fracs: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    n_bootstrap: int = 200
    tier_quantiles: tuple[float, float] = (0.70, 0.90)  # medium >= q70, high >= q90 (of train risk)


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    synth: SynthConfig = field(default_factory=SynthConfig)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


DEFAULT = Config()

DISCLAIMER = (
    "DEMONSTRATION ONLY. All patient records in this project are synthetic. Model outputs are "
    "predictive associations, not clinical causes, and must never be used for real clinical, "
    "coverage, or staffing decisions."
)
