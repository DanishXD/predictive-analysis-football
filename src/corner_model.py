"""Poisson corner model: fit corner counts and predict expected corners + over/under."""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from config import (
    CORNER_OVER_UNDER_DEFAULT,
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_SEASON,
    TIME_DECAY_XI,
)

# Imported for documentation — data already uses canonical names from matches_clean.csv
from team_mapping import TEAM_NAME_MAPPING

INPUT_PATH = PROCESSED_DIR / "matches_clean.csv"
CORNER_MODEL_PATH = MODELS_DIR / "corner_poisson_model.pkl"


def fit_corner_model(train: pd.DataFrame):
    """Fit a Poisson model on corner counts with time-decay weights."""
    weights = pb.models.dixon_coles_weights(
        train["date"],
        xi=TIME_DECAY_XI,
        base_date=train["date"].max(),
    )
    model = pb.models.PoissonGoalsModel(
        goals_home=train["corners_home"],
        goals_away=train["corners_away"],
        teams_home=train["team_home"],
        teams_away=train["team_away"],
        weights=weights,
    )
    model.fit()
    return model


def predict_corner_fixture(model, home: str, away: str, max_corners: int = 25):
    """Predict corner outcome, using league-average for unseen teams."""
    known_teams = set(model.teams)
    cold_start = home not in known_teams or away not in known_teams

    if not cold_start:
        return model.predict(home, away, max_goals=max_corners), False

    params = model.params_array
    attacks = params[: model.n_teams]
    defenses = params[model.n_teams : 2 * model.n_teams]
    home_adv = float(params[-1])

    def corner_strength(team):
        if team in known_teams:
            idx = model.team_to_idx[team]
            return (float(params[idx]), float(params[idx + model.n_teams]))
        return (float(attacks.mean()), float(defenses.mean()))

    home_attack, home_defense = corner_strength(home)
    away_attack, away_defense = corner_strength(away)

    expected_home = np.exp(home_adv + home_attack + away_defense)
    expected_away = np.exp(away_attack + home_defense)

    grid = pb.models.create_dixon_coles_grid(
        expected_home, expected_away, rho=0.0, max_goals=max_corners
    )
    return grid, True


def expected_corners(grid):
    """Extract expected home and away corners from a probability grid."""
    return float(grid.home_goal_expectation), float(grid.away_goal_expectation)


def over_under_probability(grid, threshold: float = CORNER_OVER_UNDER_DEFAULT):
    """Compute P(over threshold) and P(under threshold) for total corners."""
    total_dist = grid.total_goals_distribution()
    min_total = int(np.floor(threshold))
    over = float(sum(total_dist[min_total + 1 :]))
    under = float(sum(total_dist[: min_total + 1]))
    push = total_dist[min_total] if threshold == min_total else 0.0
    return {"over": over, "under": under, "push": push, "threshold": threshold}


def get_corner_strengths(model, all_teams: list[str]) -> pd.DataFrame:
    """Return corner-attack and corner-defense parameters per team."""
    rows = []
    for team in all_teams:
        if team in model.teams:
            idx = model.team_to_idx[team]
            attack = float(model.params_array[idx])
            defense = float(model.params_array[idx + model.n_teams])
            source = "fitted"
        else:
            attacks = model.params_array[: model.n_teams]
            defenses = model.params_array[model.n_teams : 2 * model.n_teams]
            attack = float(attacks.mean())
            defense = float(defenses.mean())
            source = "cold_start_league_average"
        rows.append(
            {"team": team, "corner_attack": attack, "corner_defense": defense, "source": source}
        )
    return pd.DataFrame(rows).sort_values("corner_attack", ascending=False, ignore_index=True)


def main() -> None:
    """Fit corner model, predict test set, and save outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(INPUT_PATH, parse_dates=["datetime", "date"])
    train = matches.loc[matches["season"] != TEST_SEASON].copy()
    test = matches.loc[matches["season"] == TEST_SEASON].copy()

    print(f"Fit corner Poisson pada {len(train)} pertandingan...")
    model = fit_corner_model(train)
    model.save(str(CORNER_MODEL_PATH))

    # Save training metadata
    import json
    metadata = {
        "training_matches": len(train),
        "training_seasons": sorted(train["season"].unique().tolist()),
        "training_period_start": train["season"].min(),
        "training_period_end": train["season"].max(),
        "test_season": TEST_SEASON,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model": "corner_poisson",
    }
    (CORNER_MODEL_PATH.parent / "corner_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    rows = []
    for match in test.sort_values(["datetime", "match_id"]).itertuples(index=False):
        grid, cold = predict_corner_fixture(model, match.team_home, match.team_away)
        exp_home, exp_away = expected_corners(grid)
        ou = over_under_probability(grid)
        rows.append(
            {
                "match_id": match.match_id,
                "season": match.season,
                "datetime": match.datetime,
                "team_home": match.team_home,
                "team_away": match.team_away,
                "actual_corners_home": match.corners_home,
                "actual_corners_away": match.corners_away,
                "expected_corners_home": exp_home,
                "expected_corners_away": exp_away,
                "expected_total_corners": exp_home + exp_away,
                "total_corners_over_under_threshold": ou["threshold"],
                "prob_over": ou["over"],
                "prob_under": ou["under"],
                "cold_start": cold,
            }
        )

    test_predictions = pd.DataFrame(rows)
    test_predictions.to_csv(
        PROCESSED_DIR / "corner_test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    all_teams = sorted(set(matches["team_home"]) | set(matches["team_away"]))
    strengths = get_corner_strengths(model, all_teams)
    strengths.to_csv(PROCESSED_DIR / "corner_team_strengths.csv", index=False)

    # Best and worst corner teams
    best_attack = strengths.iloc[0]["team"]
    best_defense = strengths.sort_values("corner_defense", ascending=True).iloc[0]["team"]

    print(f"\nCorner model tersimpan: {CORNER_MODEL_PATH}")
    print(f"Test predictions: {len(test_predictions)} match")
    print(f"Cold-start: {int(test_predictions['cold_start'].sum())}")
    print(f"Expected total corner range: {test_predictions['expected_total_corners'].min():.2f} - {test_predictions['expected_total_corners'].max():.2f}")
    print(f"\nCorner-attack terkuat: {best_attack}")
    print(f"Corner-defense terkuat: {best_defense}")

    print("\nModel corner: arsitektur Poisson sama seperti model gol.")
    print("Akurasi model corner biasanya LEBIH RENDAH dari model gol karena corner lebih noisy/random")
    print("(dipengaruhi taktik, defensive block, gaya main, dan faktor situasional lainnya).")
    print("Gunakan prediksi corner dengan hati-hati, jangan overconfident.")


if __name__ == "__main__":
    main()
