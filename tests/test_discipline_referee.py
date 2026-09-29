"""Tests untuk fitur referee di discipline_model (tanpa jaringan)."""

import numpy as np
import pandas as pd
import pytest

from config import REFEREE_MIN_MATCHES, REFEREE_OTHER_LABEL
from discipline_model import (
    apply_referee_factor,
    build_referee_card_factors,
    expected_cards,
    referee_factor_for,
)


def _train(referees, yellow_home, yellow_away):
    n = len(referees)
    return pd.DataFrame(
        {
            "referee": referees,
            "yellow_cards_home": yellow_home,
            "yellow_cards_away": yellow_away,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )


def test_factor_is_one_for_league_average_referee():
    n = REFEREE_MIN_MATCHES * 3
    train = _train(
        ["A Ref"] * n,
        [2] * (n // 2) + [2] * (n - n // 2),
        [2] * n,
    )
    factors = build_referee_card_factors(train, "yellow_cards_home", "yellow_cards_away", "yellow")
    assert referee_factor_for("A Ref", factors) == pytest.approx(1.0, abs=1e-9)


def test_high_card_referee_gets_factor_above_one():
    """Wasit dengan kartu lebih tinggi dari rata-rata liga dapat faktor > 1."""
    n = REFEREE_MIN_MATCHES * 4
    train = _train(
        ["A Ref"] * n + ["B Ref"] * n,
        [5] * n + [2] * n,
        [5] * n + [2] * n,
    )
    factors = build_referee_card_factors(train, "yellow_cards_home", "yellow_cards_away", "yellow")
    assert referee_factor_for("A Ref", factors) > 1.0
    assert referee_factor_for("B Ref", factors) < 1.0


def test_rare_referee_is_grouped_to_other_label():
    train = _train(
        ["Common"] * (REFEREE_MIN_MATCHES * 2) + ["Rookie"],
        [2] * (REFEREE_MIN_MATCHES * 2) + [9],
        [2] * (REFEREE_MIN_MATCHES * 2) + [9],
    )
    factors = build_referee_card_factors(train, "yellow_cards_home", "yellow_cards_away", "yellow")
    rookie = factors.loc[factors["referee"] == "Rookie"].iloc[0]
    assert rookie["category"] == REFEREE_OTHER_LABEL
    # notwithstanding rate tinggi, faktor diturunkan ke 1.0
    assert rookie["factor"] == pytest.approx(1.0, abs=1e-9)
    assert rookie["training_matches"] < REFEREE_MIN_MATCHES


def test_frequent_referee_keeps_own_category():
    n = REFEREE_MIN_MATCHES + 5
    train = _train(["Veteran"] * n, [3] * n, [3] * n)
    factors = build_referee_card_factors(train, "yellow_cards_home", "yellow_cards_away", "yellow")
    row = factors.loc[factors["referee"] == "Veteran"].iloc[0]
    assert row["category"] == "Veteran"


def test_unknown_referee_factor_defaults_to_one():
    train = _train(["A Ref"] * 40, [2] * 40, [2] * 40)
    factors = build_referee_card_factors(train, "yellow_cards_home", "yellow_cards_away", "yellow")
    assert referee_factor_for("Z Stranger", factors) == pytest.approx(1.0)
    assert referee_factor_for(None, factors) == pytest.approx(1.0)
    assert referee_factor_for("A Ref", pd.DataFrame()) == pytest.approx(1.0)


def test_shrinkage_pulls_rare_but_above_threshold_toward_league():
    """Wasit nyaris ambang: rate ekstrem harus di-shrink, tidak dipakai mentah."""
    n = REFEREE_MIN_MATCHES + 1  # di atas ambang, tapi sample kecil
    train = _train(
        ["Fresh"] * n + ["Steady"] * (n * 20),
        [10] * n + [3] * (n * 20),
        [10] * n + [3] * (n * 20),
    )
    factors = build_referee_card_factors(train, "yellow_cards_home", "yellow_cards_away", "yellow")
    fresh = factors.loc[factors["referee"] == "Fresh"].iloc[0]
    assert fresh["raw_rate"] > fresh["shrunk_rate"]  # menyusut karena shrinkage
    assert fresh["factor"] < fresh["raw_rate"] / fresh["league_rate"]


def test_apply_referee_factor_scales_expectations():
    import penaltyblog as pb

    grid = pb.models.create_dixon_coles_grid(2.0, 2.0, rho=0.0, max_goals=15)
    home, away = expected_cards(grid)
    scaled = apply_referee_factor(grid, 1.2)
    s_home, s_away = expected_cards(scaled)
    assert s_home == pytest.approx(home * 1.2)
    assert s_away == pytest.approx(away * 1.2)
    # faktor 1.0 -> grid tidak berubah
    assert apply_referee_factor(grid, 1.0) is grid


def test_factors_preserve_probability_mass():
    import penaltyblog as pb

    grid = pb.models.create_dixon_coles_grid(2.0, 2.0, rho=0.0, max_goals=15)
    scaled = apply_referee_factor(grid, 1.3)
    assert scaled.grid.sum() == pytest.approx(1.0, abs=1e-9)
