"""Tests untuk fitur is_empty_stadium di feature_engineering."""

import pandas as pd
import pytest

from config import EMPTY_STADIUM_SEASONS
from feature_engineering import FEATURE_COLUMNS, generate_features


def _round_robin(seasons, teams=("A", "B", "C", "D")):
    """Bangun tabel pertandingan sintetis untuk beberapa musim."""
    rows = []
    day = pd.Timestamp("2020-08-01")
    for season in seasons:
        pairs = [(teams[i], teams[(i + 1) % len(teams)]) for i in range(len(teams))]
        for idx, (home, away) in enumerate(pairs):
            rows.append(
                {
                    "match_id": f"{season}-{idx}",
                    "competition": "ENG Premier League",
                    "season": season,
                    "datetime": day + pd.Timedelta(days=idx),
                    "date": day + pd.Timedelta(days=idx),
                    "team_home": home,
                    "team_away": away,
                    "goals_home": 2,
                    "goals_away": 1,
                    "result": "H",
                }
            )
        day = day + pd.Timedelta(days=20)
    return pd.DataFrame(rows)


def test_is_empty_stadium_is_declared_as_feature():
    assert "is_empty_stadium" in FEATURE_COLUMNS


def test_is_empty_stadium_marks_only_configured_seasons():
    seasons = ["2019-2020", "2020-2021", "2021-2022"]
    features = generate_features(_round_robin(seasons))
    by_season = features.groupby("season")["is_empty_stadium"].first().to_dict()
    for season in seasons:
        expected = 1 if season in EMPTY_STADIUM_SEASONS else 0
        assert by_season[season] == expected, f"season {season}"


def test_is_empty_stadium_is_binary_everywhere():
    features = generate_features(_round_robin(["2020-2021", "2024-2025"]))
    assert set(features["is_empty_stadium"].unique()) <= {0, 1}


def test_is_empty_stadium_does_not_leak_future_seasons():
    """Nilai fitur hanya bergantung pada season match itu sendiri."""
    full = generate_features(_round_robin(["2020-2021", "2021-2022"]))
    partial = generate_features(_round_robin(["2020-2021"]))
    only_2020_full = full.loc[full["season"] == "2020-2021"]
    assert len(only_2020_full) == len(partial)
    assert (only_2020_full["is_empty_stadium"].to_numpy()
            == partial["is_empty_stadium"].to_numpy()).all()


def test_is_empty_stadium_has_no_missing_values():
    features = generate_features(_round_robin(["2020-2021"]))
    assert features["is_empty_stadium"].notna().all()
