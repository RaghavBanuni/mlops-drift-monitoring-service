"""Shared fixtures.  Small on purpose: the suite has to stay fast enough to run on every push."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.baseline import fit_baseline
from driftwatch.simulate import FEATURES, ScenarioConfig, build_scenario

SMALL = ScenarioConfig(n_reference=1_200, n_window=400, n_windows=8, change_at=4, seed=11)


@pytest.fixture(scope="session")
def config() -> ScenarioConfig:
    return SMALL


@pytest.fixture(scope="session")
def stable():
    return build_scenario("stable", SMALL)


@pytest.fixture(scope="session")
def shifted():
    return build_scenario("sudden_covariate", SMALL)


@pytest.fixture(scope="session")
def concept():
    return build_scenario("concept", SMALL)


@pytest.fixture(scope="session")
def baseline(stable):
    return fit_baseline(
        stable.reference,
        features=list(FEATURES),
        prediction_column="prediction",
        target_column="label",
        model_version="test-1",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
