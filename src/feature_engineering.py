"""Generate leakage-safe, point-in-time features from cleaned EPL matches."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from config import ROLLING_WINDOW


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

IDENTIFIER_COLUMNS = [
    "match_id",
    "competition",
    "season",
    "datetime",
    "date",
    "team_home",
    "team_away",
]

FEATURE_COLUMNS = [
    "home_form5_played",
    "home_form5_wins",
    "home_form5_draws",
    "home_form5_losses",
    "home_form5_goal_difference",
    "away_form5_played",
    "away_form5_wins",
    "away_form5_draws",
    "away_form5_losses",
    "away_form5_goal_difference",
    "h2h_played",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "home_home_played",
    "home_home_wins",
    "home_home_draws",
    "home_home_losses",
    "home_home_points_per_game",
    "home_home_goal_difference_per_game",
    "away_away_played",
    "away_away_wins",
    "away_away_draws",
    "away_away_losses",
    "away_away_points_per_game",
    "away_away_goal_difference_per_game",
    "home_rest_days",
    "away_rest_days",
    "home_league_position",
    "home_league_points",
    "home_league_played",
    "away_league_position",
    "away_league_points",
    "away_league_played",
]


def empty_stats() -> dict[str, int]:
    """Create an empty W/D/L and goal record."""
    return {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
    }


def summarize_form(history: deque[tuple[str, int]]) -> dict[str, int]:
    """Summarize a team's last five results from the team's perspective."""
    outcomes = [outcome for outcome, _ in history]
    return {
        "played": len(history),
        "wins": outcomes.count("W"),
        "draws": outcomes.count("D"),
        "losses": outcomes.count("L"),
        "goal_difference": sum(goal_difference for _, goal_difference in history),
    }


def summarize_split(stats: dict[str, int]) -> dict[str, int | float]:
    """Summarize expanding season-to-date home or away performance."""
    played = stats["played"]
    if played == 0:
        points_per_game = float("nan")
        goal_difference_per_game = float("nan")
    else:
        points_per_game = stats["points"] / played
        goal_difference_per_game = (
            stats["goals_for"] - stats["goals_against"]
        ) / played

    return {
        "played": played,
        "wins": stats["wins"],
        "draws": stats["draws"],
        "losses": stats["losses"],
        "points_per_game": points_per_game,
        "goal_difference_per_game": goal_difference_per_game,
    }


def get_table_positions(
    season: str,
    season_teams: dict[str, list[str]],
    table: dict[tuple[str, str], dict[str, int]],
) -> dict[str, int]:
    """Build positions from points, goal difference, and goals scored."""
    ranking = []
    for team in season_teams[season]:
        stats = table[(season, team)]
        ranking.append(
            (
                team,
                stats["points"],
                stats["goals_for"] - stats["goals_against"],
                stats["goals_for"],
            )
        )

    ranking.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))

    positions = {}
    previous_record = None
    current_position = 1
    for index, (team, points, goal_difference, goals_for) in enumerate(
        ranking, start=1
    ):
        record = (points, goal_difference, goals_for)
        if previous_record is not None and record != previous_record:
            current_position = index
        positions[team] = current_position
        previous_record = record
    return positions


def add_prefixed(target: dict, prefix: str, values: dict) -> None:
    """Add a feature summary to a row with a home/away prefix."""
    for name, value in values.items():
        target[f"{prefix}_{name}"] = value


def update_stats(
    stats: dict[str, int],
    outcome: str,
    goals_for: int,
    goals_against: int,
) -> None:
    """Update W/D/L, points, and goals after a match has finished."""
    stats["played"] += 1
    stats["goals_for"] += goals_for
    stats["goals_against"] += goals_against

    if outcome == "W":
        stats["wins"] += 1
        stats["points"] += 3
    elif outcome == "D":
        stats["draws"] += 1
        stats["points"] += 1
    else:
        stats["losses"] += 1


def generate_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Generate features using only matches strictly before each match date."""
    required = set(IDENTIFIER_COLUMNS + ["goals_home", "goals_away", "result"])
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Kolom input tidak lengkap: {missing}")

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise").dt.normalize()
    matches["datetime"] = pd.to_datetime(matches["datetime"], errors="raise")
    matches = matches.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)

    season_teams = {}
    for season, season_matches in matches.groupby("season", sort=False):
        teams = sorted(
            set(season_matches["team_home"]) | set(season_matches["team_away"])
        )
        season_teams[season] = teams

    form_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    h2h_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    split_stats = defaultdict(empty_stats)
    table = defaultdict(empty_stats)
    last_match_date: dict[str, pd.Timestamp] = {}
    feature_rows = []

    # Every match on a date is snapshotted before any result from that date is
    # applied. This enforces a strict "before match date" cutoff.
    for match_date, daily_matches in matches.groupby("date", sort=True):
        positions_by_season = {
            season: get_table_positions(season, season_teams, table)
            for season in daily_matches["season"].unique()
        }

        for match in daily_matches.itertuples(index=False):
            home = match.team_home
            away = match.team_away
            season = match.season
            row = {column: getattr(match, column) for column in IDENTIFIER_COLUMNS}

            add_prefixed(row, "home_form5", summarize_form(form_history[home]))
            add_prefixed(row, "away_form5", summarize_form(form_history[away]))

            pair = tuple(sorted((home, away)))
            prior_h2h = h2h_history[pair]
            row["h2h_played"] = len(prior_h2h)
            row["h2h_home_wins"] = sum(winner == home for winner in prior_h2h)
            row["h2h_draws"] = sum(winner is None for winner in prior_h2h)
            row["h2h_away_wins"] = sum(winner == away for winner in prior_h2h)

            add_prefixed(
                row,
                "home_home",
                summarize_split(split_stats[(season, home, "home")]),
            )
            add_prefixed(
                row,
                "away_away",
                summarize_split(split_stats[(season, away, "away")]),
            )

            row["home_rest_days"] = (
                (match_date - last_match_date[home]).days
                if home in last_match_date
                else float("nan")
            )
            row["away_rest_days"] = (
                (match_date - last_match_date[away]).days
                if away in last_match_date
                else float("nan")
            )

            for side, team in (("home", home), ("away", away)):
                team_table = table[(season, team)]
                row[f"{side}_league_position"] = positions_by_season[season][team]
                row[f"{side}_league_points"] = team_table["points"]
                row[f"{side}_league_played"] = team_table["played"]

            row["result"] = match.result
            feature_rows.append(row)

        # Update histories only after all snapshots for this date are complete.
        for match in daily_matches.itertuples(index=False):
            home = match.team_home
            away = match.team_away
            season = match.season
            goals_home = int(match.goals_home)
            goals_away = int(match.goals_away)

            if match.result == "H":
                home_outcome, away_outcome, winner = "W", "L", home
            elif match.result == "A":
                home_outcome, away_outcome, winner = "L", "W", away
            else:
                home_outcome, away_outcome, winner = "D", "D", None

            form_history[home].append((home_outcome, goals_home - goals_away))
            form_history[away].append((away_outcome, goals_away - goals_home))
            h2h_history[tuple(sorted((home, away)))].append(winner)

            update_stats(
                split_stats[(season, home, "home")],
                home_outcome,
                goals_home,
                goals_away,
            )
            update_stats(
                split_stats[(season, away, "away")],
                away_outcome,
                goals_away,
                goals_home,
            )
            update_stats(
                table[(season, home)], home_outcome, goals_home, goals_away
            )
            update_stats(
                table[(season, away)], away_outcome, goals_away, goals_home
            )
            last_match_date[home] = match_date
            last_match_date[away] = match_date

    return pd.DataFrame(feature_rows)[IDENTIFIER_COLUMNS + FEATURE_COLUMNS + ["result"]]


def validate_features(features: pd.DataFrame, expected_rows: int) -> None:
    """Validate feature ranges and key point-in-time invariants."""
    errors = []

    if len(features) != expected_rows:
        errors.append(f"jumlah baris {len(features)}, seharusnya {expected_rows}")
    if features["match_id"].duplicated().any():
        errors.append("match_id duplikat ditemukan")
    if features[["match_id", "date", "team_home", "team_away", "result"]].isna().any().any():
        errors.append("missing value pada identifier atau target")
    if features[["home_form5_played", "away_form5_played", "h2h_played"]].max().max() > ROLLING_WINDOW:
        errors.append("rolling window melebihi lima pertandingan")
    if features[["home_league_played", "away_league_played"]].max().max() > 37:
        errors.append("jumlah laga klasemen pre-match melebihi 37")

    first_dates = features.groupby("season")["date"].transform("min")
    season_openers = features.loc[features["date"] == first_dates]
    opener_columns = [
        "home_home_played",
        "away_away_played",
        "home_league_points",
        "away_league_points",
        "home_league_played",
        "away_league_played",
    ]
    if season_openers[opener_columns].ne(0).any().any():
        errors.append("fitur season-to-date pada tanggal pembuka tidak nol")

    rest_days = features[["home_rest_days", "away_rest_days"]].stack()
    if (rest_days <= 0).any():
        errors.append("rest days nol atau negatif ditemukan")

    if errors:
        raise ValueError("; ".join(errors))


def main() -> None:
    """Load clean matches, generate point-in-time features, and save them."""
    matches = pd.read_csv(INPUT_PATH, parse_dates=["datetime", "date"])
    features = generate_features(matches)
    validate_features(features, expected_rows=len(matches))
    features.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")

    missing = features[FEATURE_COLUMNS].isna().sum()
    missing = missing.loc[missing > 0].to_dict()
    print(f"Tersimpan: {OUTPUT_PATH}")
    print(f"Total: {len(features)} pertandingan, {len(FEATURE_COLUMNS)} fitur")
    print(f"Missing karena histori belum tersedia: {missing}")
    print("Validasi point-in-time: lolos")


if __name__ == "__main__":
    main()
