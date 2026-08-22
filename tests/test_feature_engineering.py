import pandas as pd

import config
from feature_engineering import IDENTIFIER_COLUMNS, generate_features


def make_matches(rows):
    matches = pd.DataFrame(rows)
    for column in IDENTIFIER_COLUMNS:
        assert column in matches.columns
    return matches


def epl_row(
    match_id, season, day, home, away, goals_home, goals_away,
    home_league_position=None, away_league_position=None,
):
    if goals_home > goals_away:
        result = "H"
    elif goals_home < goals_away:
        result = "A"
    else:
        result = "D"
    return {
        "match_id": match_id,
        "competition": "ENG Premier League",
        "season": season,
        "datetime": f"{day} 15:00:00",
        "date": day,
        "team_home": home,
        "team_away": away,
        "goals_home": goals_home,
        "goals_away": goals_away,
        "result": result,
    }


def round_robin_season(season, teams, start_day, outcomes_by_pair):
    rows = []
    day = pd.Timestamp(start_day)
    match_id = f"{season}-M"
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i == j:
                continue
            home, away = teams[i], teams[j]
            gh, ga = outcomes_by_pair.get((home, away), (1, 1))
            rows.append(epl_row(
                f"{match_id}{len(rows)}", season, day.strftime("%Y-%m-%d"),
                home, away, gh, ga,
            ))
            day += pd.Timedelta(days=1)
    return rows


def test_missing_required_columns_rejected():
    rows = [epl_row("m1", "2020-2021", "2020-08-15", "Arsenal", "Chelsea", 2, 1)]
    matches = pd.DataFrame(rows).drop(columns=["goals_home"])
    try:
        generate_features(matches)
        assert False, "seharusnya memicu ValueError"
    except ValueError as exc:
        assert "tidak lengkap" in str(exc)


def test_same_day_matches_do_not_leak():
    rows = [
        epl_row("m1", "2020-2021", "2020-08-15", "Arsenal", "Chelsea", 2, 1),
        epl_row("m2", "2020-2021", "2020-08-15", "Tottenham", "Everton", 0, 1),
        epl_row("m3", "2020-2021", "2020-08-15", "Leeds", "Brighton", 3, 0),
    ]
    features = generate_features(make_matches(rows))
    assert len(features) == 3
    for col in ("home_form5_played", "away_form5_played",
                "home_league_played", "away_league_played", "h2h_played"):
        assert (features[col] == 0).all(), f"{col} harus 0 di match pertama tanggal ini"


def test_subsequent_match_sees_only_prior_results():
    rows = [
        epl_row("m1", "2020-2021", "2020-08-15", "Arsenal", "Chelsea", 2, 1),
        epl_row("m2", "2020-2021", "2020-08-16", "Arsenal", "Chelsea", 1, 1),
    ]
    features = generate_features(make_matches(rows))
    second = features.loc[features["match_id"] == "m2"].iloc[0]
    assert second["home_form5_played"] == 1
    assert second["home_form5_wins"] == 1
    assert second["home_league_played"] == 1
    assert second["away_form5_played"] == 1
    assert second["away_form5_losses"] == 1
    assert second["h2h_played"] == 1
    assert second["h2h_home_wins"] == 1


def test_form_window_caps_at_rolling_window():
    rows = []
    day = pd.Timestamp("2020-08-15")
    for k in range(config.ROLLING_WINDOW + 3):
        rows.append(epl_row(
            f"m{k}", "2020-2021", day.strftime("%Y-%m-%d"),
            "Arsenal", "Chelsea", 2, 1,
        ))
        day += pd.Timedelta(days=1)
    features = generate_features(make_matches(rows))
    assert (features["home_form5_played"] <= config.ROLLING_WINDOW).all()
    assert (features["away_form5_played"] <= config.ROLLING_WINDOW).all()


def test_league_position_is_max_at_played_count():
    rows = [
        epl_row("m1", "2020-2021", "2020-08-15", "Arsenal", "Chelsea", 2, 1),
        epl_row("m2", "2020-2021", "2020-08-16", "Tottenham", "Everton", 3, 0),
    ]
    features = generate_features(make_matches(rows))
    second = features.loc[features["match_id"] == "m2"].iloc[0]
    assert second["home_league_position"] <= 4
    assert second["home_league_position"] >= 1
