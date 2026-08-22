"""Tests untuk normalize_columns & clean_matches di data_collection."""

import pandas as pd
import pytest

from data_collection import COMPETITION, clean_matches, normalize_columns

BASE_ROW = {
    "Date": "13/08/2016",
    "Time": "15:00",
    "HomeTeam": "Man United",
    "AwayTeam": "Bournemouth",
    "FTHG": 3,
    "FTAG": 1,
    "FTR": "H",
}


def _season_frame(rows, season="2016-2017"):
    """Meniru alur main(): normalize lalu beri label season."""
    frame = normalize_columns(pd.DataFrame(rows))
    frame["season"] = season
    return frame


def test_normalize_columns_maps_football_data_names():
    normalized = normalize_columns(pd.DataFrame([BASE_ROW]))
    for column in ("date", "time", "team_home", "team_away", "goals_home", "goals_away", "result"):
        assert column in normalized.columns


def test_normalize_columns_maps_penaltyblog_aliases():
    normalized = normalize_columns(pd.DataFrame([{"fthg": 2, "ftag": 0, "hc": 5}]))
    assert normalized.loc[0, "goals_home"] == 2
    assert normalized.loc[0, "goals_away"] == 0
    assert normalized.loc[0, "corners_home"] == 5


def test_clean_matches_standardizes_team_names_and_whitespace():
    rows = [
        BASE_ROW,
        {**BASE_ROW, "Date": "20/08/2016", "HomeTeam": " Man City ", "FTHG": 1, "FTAG": 1},
        {**BASE_ROW, "Date": "27/08/2016", "HomeTeam": "Liverpool", "FTHG": 0, "FTAG": 2},
    ]
    clean = clean_matches([_season_frame(rows)])
    assert list(clean["team_home"].head(3)) == [
        "Manchester United",
        "Manchester City",
        "Liverpool",
    ]


def test_clean_matches_derives_result_from_scores_not_source_column():
    rows = [
        BASE_ROW,
        {**BASE_ROW, "Date": "20/08/2016", "FTHG": 1, "FTAG": 1, "FTR": "H"},
        {**BASE_ROW, "Date": "27/08/2016", "HomeTeam": "Liverpool", "FTHG": 0, "FTAG": 2},
    ]
    clean = clean_matches([_season_frame(rows)])
    assert list(clean["result"]) == ["H", "D", "A"]


def test_clean_matches_drops_incomplete_rows_silently_counted_by_caller():
    rows = [
        BASE_ROW,
        {**BASE_ROW, "Date": "20/08/2016", "AwayTeam": "Southampton", "FTAG": None},
        {**BASE_ROW, "Date": "31/02/2017", "HomeTeam": "Burnley"},
        {**BASE_ROW, "Date": "10/09/2016", "HomeTeam": "Watford"},
    ]
    clean = clean_matches([_season_frame(rows)])
    assert len(clean) == 2
    assert set(clean["team_home"]) == {"Manchester United", "Watford"}


def test_clean_matches_builds_match_id_and_competition():
    clean = clean_matches([_season_frame([BASE_ROW])])
    row = clean.iloc[0]
    assert row["match_id"] == "20160813-manchester-united-bournemouth"
    assert row["competition"] == COMPETITION


def test_clean_matches_falls_back_to_date_when_time_missing():
    rows = [{**BASE_ROW, "Time": None}]
    clean = clean_matches([_season_frame(rows)])
    assert clean.iloc[0]["datetime"] is not pd.NaT
