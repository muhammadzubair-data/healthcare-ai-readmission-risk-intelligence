import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readmit.config import Config, ModelConfig, SynthConfig  # noqa: E402
from readmit.data.ingest import clean  # noqa: E402
from readmit.features.build import build_features  # noqa: E402
from readmit.synth.generator import inject_data_quality_issues, simulate  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config(synth=SynthConfig(n_patients=8000, seed=7), model=ModelConfig(ensemble_size=2, n_bootstrap=20, max_rounds=120, seed=7))


@pytest.fixture(scope="session")
def sim(cfg):
    tables = simulate(cfg.synth)
    truth = tables.pop("_truth")
    return tables, truth


@pytest.fixture(scope="session")
def defects(sim, cfg):
    raw, ledger = inject_data_quality_issues(sim[0], cfg.synth)
    return raw, ledger


@pytest.fixture(scope="session")
def cleaned(defects, cfg):
    return clean(defects[0], cfg)


@pytest.fixture(scope="session")
def window(cfg):
    return pd.Timestamp(cfg.synth.window_start), pd.Timestamp(cfg.synth.window_end) + pd.Timedelta(days=1)


@pytest.fixture(scope="session")
def feat(cleaned, cfg, window):
    return build_features(cleaned[0], cfg.cohort, *window)


@pytest.fixture(scope="session")
def feat_raw_ids(sim, cfg, window):
    """Features built on the defect-free simulated tables (raw ids) for label-vs-truth checks."""
    return build_features(sim[0], cfg.cohort, *window)
