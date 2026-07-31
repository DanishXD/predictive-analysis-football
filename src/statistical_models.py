"""Fit leakage-safe statistical models and Elo ratings for EPL matches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import penaltyblog as pb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_SEASON = "2025-2026"
TIME_DECAY_XI = 0.0018
MAX_GOALS = 15
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0


def fit_goal_models(train: pd.DataFrame):
    """Fit Poisson and Dixon-Coles models with identical time-decay weights."""
    weights = pb.models.dixon_coles_weights(
        train["date"],
        xi=TIME_DECAY_XI,
        base_date=train["date"].max(),
    )
    model_inputs = (
        train["goals_home"],
        train["goals_away"],
        train["team_home"],
        train["team_away"],
    )

    poisson = pb.models.PoissonGoalsModel(*model_inputs, weights=weights)
    poisson.fit()

    dixon_coles = pb.models.DixonColesGoalModel(*model_inputs, weights=weights)
    dixon_coles.fit()
    return poisson, dixon_coles


def get_team_parameters(model, team: str) -> tuple[float, float]:
    """Return attack and defense parameters for a fitted team."""
    index = model.team_to_idx[team]
    return (
        float(model.params_array[index]),
        float(model.params_array[index + model.n_teams]),
    )


def predict_fixture(model, home: str, away: str, dixon_coles: bool):
    """Predict a fixture, using league-average strength for an unseen team."""
    known_teams = set(model.teams)
    cold_start = home not in known_teams or away not in known_teams
    if not cold_start:
        return model.predict(home, away, max_goals=MAX_GOALS), False

    parameters = model.params_array
    attacks = parameters[: model.n_teams]
    defenses = parameters[model.n_teams : 2 * model.n_teams]

    home_attack = (
        get_team_parameters(model, home)[0]
        if home in known_teams
        else float(attacks.mean())
    )
    home_defense = (
        get_team_parameters(model, home)[1]
        if home in known_teams
        else float(defenses.mean())
    )
    away_attack = (
        get_team_parameters(model, away)[0]
        if away in known_teams
        else float(attacks.mean())
    )
    away_defense = (
        get_team_parameters(model, away)[1]
        if away in known_teams
        else float(defenses.mean())
    )

    home_advantage = float(parameters[-2] if dixon_coles else parameters[-1])
    rho = float(parameters[-1]) if dixon_coles else 0.0
    expected_home = np.exp(home_advantage + home_attack + away_defense)
    expected_away = np.exp(away_attack + home_defense)
    grid = pb.models.create_dixon_coles_grid(
        expected_home,
        expected_away,
        rho=rho,
        max_goals=MAX_GOALS,
    )
    return grid, True


def build_team_strengths(
    poisson,
    dixon_coles,
    all_teams: list[str],
) -> pd.DataFrame:
    """Create attack/defense tables, including neutral cold-start priors."""
    rows = []
    for model_name, model in (("poisson", poisson), ("dixon_coles", dixon_coles)):
        attacks = model.params_array[: model.n_teams]
        defenses = model.params_array[model.n_teams : 2 * model.n_teams]
        known_teams = set(model.teams)

        for team in all_teams:
            if team in known_teams:
                attack, defense = get_team_parameters(model, team)
                source = "fitted"
            else:
                attack = float(attacks.mean())
                defense = float(defenses.mean())
                source = "cold_start_league_average"
            rows.append(
                {
                    "model": model_name,
                    "team": team,
                    "attack": attack,
                    "defense": defense,
                    "parameter_source": source,
                }
            )

    strengths = pd.DataFrame(rows)
    strengths["attack_rank"] = strengths.groupby("model")["attack"].rank(
        method="min", ascending=False
    ).astype(int)
    # A lower defense parameter means fewer expected goals conceded.
    strengths["defense_rank"] = strengths.groupby("model")["defense"].rank(
        method="min", ascending=True
    ).astype(int)
    return strengths.sort_values(["model", "attack_rank", "team"]).reset_index(
        drop=True
    )


def build_test_probabilities(
    test: pd.DataFrame,
    poisson,
    dixon_coles,
) -> pd.DataFrame:
    """Generate static holdout probabilities from train-only fitted models."""
    rows = []
    for match in test.sort_values(["datetime", "match_id"]).itertuples(index=False):
        poisson_grid, poisson_cold_start = predict_fixture(
            poisson, match.team_home, match.team_away, dixon_coles=False
        )
        dc_grid, dc_cold_start = predict_fixture(
            dixon_coles, match.team_home, match.team_away, dixon_coles=True
        )
        rows.append(
            {
                "match_id": match.match_id,
                "season": match.season,
                "datetime": match.datetime,
                "date": match.date,
                "team_home": match.team_home,
                "team_away": match.team_away,
                "result": match.result,
                "cold_start": poisson_cold_start or dc_cold_start,
                "poisson_expected_goals_home": poisson_grid.home_goal_expectation,
                "poisson_expected_goals_away": poisson_grid.away_goal_expectation,
                "poisson_prob_home": poisson_grid.home_win,
                "poisson_prob_draw": poisson_grid.draw,
                "poisson_prob_away": poisson_grid.away_win,
                "dixon_coles_expected_goals_home": dc_grid.home_goal_expectation,
                "dixon_coles_expected_goals_away": dc_grid.away_goal_expectation,
                "dixon_coles_prob_home": dc_grid.home_win,
                "dixon_coles_prob_draw": dc_grid.draw,
                "dixon_coles_prob_away": dc_grid.away_win,
            }
        )
    return pd.DataFrame(rows)


def build_elo_history(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pre- and post-match Elo ratings in chronological order."""
    elo = pb.ratings.Elo(k=ELO_K, home_field_advantage=ELO_HOME_ADVANTAGE)
    history_rows = []
    result_codes = {"H": 0, "D": 1, "A": 2}

    ordered = matches.sort_values(["datetime", "match_id"])
    for match in ordered.itertuples(index=False):
        home_pre = elo.get_team_rating(match.team_home)
        away_pre = elo.get_team_rating(match.team_away)
        elo.update_ratings(
            match.team_home,
            match.team_away,
            result_codes[match.result],
        )
        home_post = elo.get_team_rating(match.team_home)
        away_post = elo.get_team_rating(match.team_away)

        history_rows.extend(
            [
                {
                    "match_id": match.match_id,
                    "datetime": match.datetime,
                    "date": match.date,
                    "season": match.season,
                    "team": match.team_home,
                    "opponent": match.team_away,
                    "venue": "home",
                    "result": match.result,
                    "elo_pre": home_pre,
                    "opponent_elo_pre": away_pre,
                    "elo_gap_pre": home_pre - away_pre,
                    "elo_post": home_post,
                },
                {
                    "match_id": match.match_id,
                    "datetime": match.datetime,
                    "date": match.date,
                    "season": match.season,
                    "team": match.team_away,
                    "opponent": match.team_home,
                    "venue": "away",
                    "result": match.result,
                    "elo_pre": away_pre,
                    "opponent_elo_pre": home_pre,
                    "elo_gap_pre": away_pre - home_pre,
                    "elo_post": away_post,
                },
            ]
        )

    history = pd.DataFrame(history_rows)
    final_ratings = pd.DataFrame(
        [{"team": team, "elo_rating": rating} for team, rating in elo.ratings.items()]
    ).sort_values("elo_rating", ascending=False, ignore_index=True)
    final_ratings["elo_rank"] = np.arange(1, len(final_ratings) + 1)
    final_ratings["as_of"] = ordered["datetime"].max()
    return history, final_ratings


def build_model_metrics(poisson, dixon_coles, train: pd.DataFrame) -> pd.DataFrame:
    """Build a directly comparable weighted likelihood/AIC summary."""
    metrics = pd.DataFrame(
        [
            {
                "model": "poisson",
                "train_matches": len(train),
                "time_decay_xi": TIME_DECAY_XI,
                "log_likelihood": poisson.loglikelihood,
                "n_parameters": poisson.n_params,
                "aic": poisson.aic,
                "rho": np.nan,
            },
            {
                "model": "dixon_coles",
                "train_matches": len(train),
                "time_decay_xi": TIME_DECAY_XI,
                "log_likelihood": dixon_coles.loglikelihood,
                "n_parameters": dixon_coles.n_params,
                "aic": dixon_coles.aic,
                "rho": dixon_coles.params["rho"],
            },
        ]
    )
    poisson_ll = float(metrics.loc[metrics["model"] == "poisson", "log_likelihood"].iloc[0])
    poisson_aic = float(metrics.loc[metrics["model"] == "poisson", "aic"].iloc[0])
    metrics["log_likelihood_gain_vs_poisson"] = metrics["log_likelihood"] - poisson_ll
    metrics["aic_improvement_vs_poisson"] = poisson_aic - metrics["aic"]
    return metrics


def validate_outputs(
    matches: pd.DataFrame,
    test: pd.DataFrame,
    strengths: pd.DataFrame,
    probabilities: pd.DataFrame,
    elo_history: pd.DataFrame,
) -> None:
    """Validate probability, coverage, and point-in-time output invariants."""
    errors = []
    if len(probabilities) != len(test):
        errors.append("probabilitas tidak mencakup seluruh test set")
    if probabilities["match_id"].duplicated().any():
        errors.append("match_id probabilitas duplikat")
    if probabilities.isna().any().any():
        errors.append("missing value pada probabilitas test")

    for prefix in ("poisson", "dixon_coles"):
        probability_sum = probabilities[
            [f"{prefix}_prob_home", f"{prefix}_prob_draw", f"{prefix}_prob_away"]
        ].sum(axis=1)
        if not np.allclose(probability_sum, 1.0, atol=1e-9):
            errors.append(f"probabilitas {prefix} tidak berjumlah satu")

    all_teams = set(matches["team_home"]) | set(matches["team_away"])
    for model_name in ("poisson", "dixon_coles"):
        model_teams = set(strengths.loc[strengths["model"] == model_name, "team"])
        if model_teams != all_teams:
            errors.append(f"strength {model_name} tidak mencakup semua tim")

    if len(elo_history) != len(matches) * 2:
        errors.append("histori Elo harus memiliki dua baris per pertandingan")
    if elo_history[["elo_pre", "opponent_elo_pre", "elo_post"]].isna().any().any():
        errors.append("missing value pada histori Elo")

    if errors:
        raise ValueError("; ".join(errors))


def main() -> None:
    """Run all Phase 4 statistical models and persist their outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(INPUT_PATH, parse_dates=["datetime", "date"])
    train = matches.loc[matches["season"] != TEST_SEASON].copy()
    test = matches.loc[matches["season"] == TEST_SEASON].copy()
    if train["date"].max() >= test["date"].min():
        raise ValueError("Split waktu tidak valid: train overlap dengan test")

    print(f"Fit model pada {len(train)} pertandingan sebelum {TEST_SEASON}...")
    poisson, dixon_coles = fit_goal_models(train)

    all_teams = sorted(set(matches["team_home"]) | set(matches["team_away"]))
    strengths = build_team_strengths(poisson, dixon_coles, all_teams)
    probabilities = build_test_probabilities(test, poisson, dixon_coles)
    elo_history, final_elo = build_elo_history(matches)
    metrics = build_model_metrics(poisson, dixon_coles, train)
    validate_outputs(matches, test, strengths, probabilities, elo_history)

    strengths.to_csv(PROCESSED_DIR / "statistical_team_strengths.csv", index=False)
    probabilities.to_csv(
        PROCESSED_DIR / "test_match_probabilities.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    elo_history.to_csv(
        PROCESSED_DIR / "elo_history.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    final_elo.to_csv(
        PROCESSED_DIR / "elo_current_ratings.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    metrics.to_csv(PROCESSED_DIR / "statistical_model_metrics.csv", index=False)
    poisson.save(str(MODELS_DIR / "poisson_goal_model.pkl"))
    dixon_coles.save(str(MODELS_DIR / "dixon_coles_goal_model.pkl"))

    strongest_attack = strengths.loc[
        (strengths["model"] == "dixon_coles") & (strengths["attack_rank"] == 1),
        "team",
    ].iloc[0]
    strongest_defense = strengths.loc[
        (strengths["model"] == "dixon_coles") & (strengths["defense_rank"] == 1),
        "team",
    ].iloc[0]
    dc_metrics = metrics.loc[metrics["model"] == "dixon_coles"].iloc[0]

    print(f"Attack terkuat (Dixon-Coles): {strongest_attack}")
    print(f"Defense terkuat (Dixon-Coles): {strongest_defense}")
    print(
        "Dixon-Coles vs Poisson: "
        f"LL gain {dc_metrics['log_likelihood_gain_vs_poisson']:.4f}, "
        f"AIC improvement {dc_metrics['aic_improvement_vs_poisson']:.4f}"
    )
    print(f"Probabilitas test: {len(probabilities)} match")
    print(f"Match cold-start: {int(probabilities['cold_start'].sum())}")
    print("Validasi output: lolos")


def btts_probability(grid):
    """
    Compute Both Teams to Score probability from a FootballProbabilityGrid.

    Parameters
    ----------
    grid : FootballProbabilityGrid
        The probability grid from a fitted Poisson/Dixon-Coles model.

    Returns
    -------
    float
        Probability that both home and away teams score at least one goal.
    """
    return grid.btts_yes


def btts_probabilities_for_test(
    test_matches: pd.DataFrame,
    poisson_model,
    dixon_coles_model,
) -> pd.DataFrame:
    """Generate BTTS probabilities for all matches in a test set."""
    rows = []
    for match in test_matches.itertuples(index=False):
        poisson_grid, poisson_cold = predict_fixture(
            poisson_model, match.team_home, match.team_away, dixon_coles=False
        )
        dc_grid, dc_cold = predict_fixture(
            dixon_coles_model, match.team_home, match.team_away, dixon_coles=True
        )
        rows.append(
            {
                "match_id": match.match_id,
                "season": match.season,
                "datetime": match.datetime,
                "team_home": match.team_home,
                "team_away": match.team_away,
                "poisson_btts": btts_probability(poisson_grid),
                "dixon_coles_btts": btts_probability(dc_grid),
                "actual_btts": int(
                    match.goals_home >= 1 and match.goals_away >= 1
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
