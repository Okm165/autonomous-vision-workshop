"""Shared helpers for the workshop test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixed random data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def points_3d(rng) -> np.ndarray:
    """200 points in the camera frustum: X, Y in [-2, 2], Z in [1, 10]."""
    return np.column_stack(
        [
            rng.uniform(-2, 2, 200),
            rng.uniform(-2, 2, 200),
            rng.uniform(1, 10, 200),
        ]
    )
