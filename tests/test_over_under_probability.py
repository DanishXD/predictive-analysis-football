"""Tests untuk over_under_probability di corner_model & discipline_model."""

import numpy as np
import penaltyblog as pb
import pytest

from corner_model import over_under_probability as corner_over_under
from discipline_model import over_under_probability as yellow_over_under

OVER_UNDER_IMPLEMENTATIONS = [corner_over_under, yellow_over_under]


@pytest.fixture(scope="module")
def grid():
    return pb.models.create_dixon_coles_grid(1.8, 1.2, rho=0.0, max_goals=15)


@pytest.mark.parametrize("over_under", OVER_UNDER_IMPLEMENTATIONS)
def test_half_threshold_partitions_probability_mass(over_under, grid):
    result = over_under(grid, threshold=9.5)
    dist = grid.total_goals_distribution()
    assert result["push"] == pytest.approx(0.0)
    assert result["over"] == pytest.approx(float(np.sum(dist[10:])))
    assert result["under"] == pytest.approx(float(np.sum(dist[:10])))
    assert result["over"] + result["under"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("over_under", OVER_UNDER_IMPLEMENTATIONS)
def test_whole_threshold_push_not_double_counted_as_under(over_under, grid):
    """Threshold bulat: massa probabilitas tepat di garis adalah push,
    tidak boleh ikut terhitung sebagai under."""
    result = over_under(grid, threshold=9.0)
    dist = grid.total_goals_distribution()
    assert result["push"] == pytest.approx(float(dist[9]))
    assert result["over"] + result["under"] + result["push"] == (
        pytest.approx(1.0, abs=1e-9)
    )


@pytest.mark.parametrize("over_under", OVER_UNDER_IMPLEMENTATIONS)
def test_threshold_reported_back(over_under, grid):
    assert over_under(grid, threshold=4.5)["threshold"] == 4.5
