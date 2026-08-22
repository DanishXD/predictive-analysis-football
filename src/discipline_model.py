"""Poisson discipline model: yellow cards, plus optional red cards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from config import (
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_SEASON,
    TIME_DECAY_XI,
    YELLOW_OVER_UNDER_DEFAULT,
)
from team_mapping import TEAM_NAME_MAPPING

INPUT_PATH = PROCESSED_DIR / "matches_clean.csv"
YELLOW_MODEL_PATH = MODELS_DIR / "yellow_card_model.pkl"
RED_MODEL_PATH = MODELS_DIR / "red_card_model.pkl"


def fit_discipline_model(
    train: pd.DataFrame, home_col: str, away_col: str, label: str
):
    """Fit a Poisson model on card counts with time-decay weights."""
    weights = pb.models.dixon_coles_weights(
        train["date"],
        xi=TIME_DECAY_XI,
        base_date=train["date"].max(),
    )
    model = pb.models.PoissonGoalsModel(
        goals_home=train[home_col],
        goals_away=train[away_col],
        teams_home=train["team_home"],
        teams_away=train["team_away"],
        weights=weights,
    )
    model.fit()
    return model


def predict_discipline_fixture(model, home: str, away: str, max_cards: int = 15):
    """Predict card outcome, using league-average for unseen teams."""
    known_teams = set(model.teams)
    cold_start = home not in known_teams or away not in known_teams

    if not cold_start:
        return model.predict(home, away, max_goals=max_cards), False

    params = model.params_array
    attacks = params[: model.n_teams]
    defenses = params[model.n_teams : 2 * model.n_teams]
    home_adv = float(params[-1])

    def strength(team):
        if team in known_teams:
            idx = model.team_to_idx[team]
            return (float(params[idx]), float(params[idx + model.n_teams]))
        return (float(attacks.mean()), float(defenses.mean()))

    home_attack, home_defense = strength(home)
    away_attack, away_defense = strength(away)

    expected_home = np.exp(home_adv + home_attack + away_defense)
    expected_away = np.exp(away_attack + home_defense)

    grid = pb.models.create_dixon_coles_grid(
        expected_home, expected_away, rho=0.0, max_goals=max_cards
    )
    return grid, True


def expected_cards(grid):
    """Extract expected home and away card counts from a probability grid."""
    return float(grid.home_goal_expectation), float(grid.away_goal_expectation)


def over_under_probability(grid, threshold: float):
    """Compute P(over threshold) and P(under threshold) for total cards."""
    total_dist = grid.total_goals_distribution()
    min_total = int(np.floor(threshold))
    over = float(sum(total_dist[min_total + 1 :]))
    under = float(sum(total_dist[: min_total + 1]))
    push = total_dist[min_total] if threshold == min_total else 0.0
    return {"over": over, "under": under, "push": push, "threshold": threshold}


def first_card_probability(grid) -> dict:
    """Probability which team gets more cards (proxy for 'first card')."""
    home_more = float(np.sum(grid.grid[np.triu_indices_from(grid.grid, k=1)]))
    away_more = float(np.sum(grid.grid[np.tril_indices_from(grid.grid, k=-1)]))
    equal = float(np.sum(np.diag(grid.grid)))
    return {
        "home_more": home_more,
        "away_more": away_more,
        "equal": equal,
        "home_label": "home team gets more cards",
        "away_label": "away team gets more cards",
    }


def get_discipline_strengths(model, all_teams: list[str], label: str) -> pd.DataFrame:
    """Return discipline-attack and discipline-defense parameters per team."""
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
            {
                "team": team,
                f"{label}_attack": attack,
                f"{label}_defense": defense,
                "source": source,
            }
        )
    return pd.DataFrame(rows).sort_values(
        f"{label}_attack", ascending=False, ignore_index=True
    )


def main() -> None:
    """Fit yellow card model, test set predictions, save outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(INPUT_PATH, parse_dates=["datetime", "date"])
    train = matches.loc[matches["season"] != TEST_SEASON].copy()
    test = matches.loc[matches["season"] == TEST_SEASON].copy()
    all_teams = sorted(set(matches["team_home"]) | set(matches["team_away"]))

    # Save training metadata
    import json
    metadata = {
        "training_matches": len(train),
        "training_seasons": sorted(train["season"].unique().tolist()),
        "training_period_start": train["season"].min(),
        "training_period_end": train["season"].max(),
        "test_season": TEST_SEASON,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": ["yellow_card", "red_card"],
    }
    (YELLOW_MODEL_PATH.parent / "discipline_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # --- Yellow cards ---
    print(f"Fit yellow card Poisson pada {len(train)} pertandingan...")
    yellow_model = fit_discipline_model(
        train, "yellow_cards_home", "yellow_cards_away", "yellow"
    )
    yellow_model.save(str(YELLOW_MODEL_PATH))

    yellow_rows = []
    for match in test.sort_values(["datetime", "match_id"]).itertuples(index=False):
        grid, cold = predict_discipline_fixture(
            yellow_model, match.team_home, match.team_away
        )
        exp_home, exp_away = expected_cards(grid)
        ou = over_under_probability(grid, YELLOW_OVER_UNDER_DEFAULT)
        fc = first_card_probability(grid)
        yellow_rows.append(
            {
                "match_id": match.match_id,
                "season": match.season,
                "datetime": match.datetime,
                "team_home": match.team_home,
                "team_away": match.team_away,
                "actual_yellow_home": match.yellow_cards_home,
                "actual_yellow_away": match.yellow_cards_away,
                "expected_yellow_home": exp_home,
                "expected_yellow_away": exp_away,
                "expected_total_yellow": exp_home + exp_away,
                "yellow_over_under_threshold": ou["threshold"],
                "prob_yellow_over": ou["over"],
                "prob_yellow_under": ou["under"],
                "prob_home_more_yellow": fc["home_more"],
                "prob_away_more_yellow": fc["away_more"],
                "prob_equal_yellow": fc["equal"],
                "cold_start": cold,
            }
        )

    yellow_test = pd.DataFrame(yellow_rows)
    yellow_test.to_csv(
        PROCESSED_DIR / "yellow_card_test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    yellow_strengths = get_discipline_strengths(yellow_model, all_teams, "yellow")
    yellow_strengths.to_csv(
        PROCESSED_DIR / "yellow_card_team_strengths.csv", index=False
    )

    best_yellow_attack = yellow_strengths.iloc[0]["team"]
    best_yellow_defense = yellow_strengths.sort_values(
        "yellow_defense", ascending=True
    ).iloc[0]["team"]

    print(
        f"Yellow card test predictions: {len(yellow_test)} match, "
        f"cold-start: {int(yellow_test['cold_start'].sum())}"
    )
    print(
        f"Expected yellow range: "
        f"{yellow_test['expected_total_yellow'].min():.2f} - "
        f"{yellow_test['expected_total_yellow'].max():.2f}"
    )
    print(f"Yellow-attack terkuat: {best_yellow_attack}")
    print(f"Yellow-defense terkuat: {best_yellow_defense}")
    print(f"Yellow model tersimpan: {YELLOW_MODEL_PATH}")

    # --- Red cards (opsional, sparse data) ---
    red_matches_with_cards = ((train["red_cards_home"] > 0) | (train["red_cards_away"] > 0)).sum()
    print(
        f"\nRed cards: {red_matches_with_cards}/{len(train)} train matches have red cards "
        f"({red_matches_with_cards / len(train) * 100:.1f}%)"
    )

    try:
        red_model = fit_discipline_model(
            train, "red_cards_home", "red_cards_away", "red"
        )
        red_model.save(str(RED_MODEL_PATH))
        red_strengths = get_discipline_strengths(red_model, all_teams, "red")
        red_strengths.to_csv(
            PROCESSED_DIR / "red_card_team_strengths.csv", index=False
        )
        print(f"Red model tersimpan: {RED_MODEL_PATH}")
        print(
            "PERINGATAN: data kartu merah sangat jarang (~10% pertandingan), "
            "model Poisson kurang stabil. Gunakan prediksi merah dengan hati-hati."
        )
    except Exception as exc:
        print(f"Red model gagal di-fit: {exc}")
        print("Red card model dilewati karena data terlalu sparse.")

    print("\nModel disiplin: arsitektur Poisson sama seperti model gol dan corner.")
    print("Akurasi model kartu kuning lebih rendah dari model gol karena kartu")
    print("dipengaruhi faktor situasional (wasit, tensi pertandingan, dll).")
    print("Model kartu merah sangat tidak stabil karena data sparse.")


if __name__ == "__main__":
    main()
