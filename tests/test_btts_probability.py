"""Tests untuk btts_probability dari statistical_models."""

import numpy as np
import penaltyblog as pb
import pytest

from statistical_models import btts_probability


def test_btts_matches_manual_grid_sum():
    grid = pb.models.create_dixon_coles_grid(1.6, 1.4, rho=0.0, max_goals=12)
    expected = float(np.sum(grid.grid[1:, 1:]))
    result = btts_probability(grid)
    assert result == pytest.approx(expected, abs=1e-9)


def test_btts_within_valid_range():
    grid = pb.models.create_dixon_coles_grid(1.8, 1.2, rho=0.0, max_goals=12)
    result = btts_probability(grid)
    assert 0.0 < result < 1.0


def test_btts_increases_with_goal_expectations():
    low_scoring = pb.models.create_dixon_coles_grid(0.8, 0.7, rho=0.0, max_goals=12)
    high_scoring = pb.models.create_dixon_coles_grid(2.2, 2.0, rho=0.0, max_goals=12)
    assert btts_probability(high_scoring) > btts_probability(low_scoring)
