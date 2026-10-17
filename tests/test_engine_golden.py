"""The engine must keep producing exactly the pictures recorded in tests/data/engine_golden.npz."""

import numpy as np
import pytest

from core.engine import OrnamentEngine, load_tile
from tests.engine_cases import CASES, GOLDEN, run_case


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name, golden):
    result = run_case(name, OrnamentEngine, load_tile)
    for key, value in result.items():
        expected = golden[f"{name}_{key}"]
        assert value.shape == expected.shape, key
        diff = np.abs(value.astype(int) - expected.astype(int))
        assert diff.max() == 0, f"case {name} {key}: {np.count_nonzero(diff)} values differ, max {diff.max()}"
