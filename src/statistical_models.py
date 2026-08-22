"""Fit leakage-safe statistical models and Elo ratings for EPL matches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from config import (
    ELO_DEFAULT_RATING,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    MAX_GOALS,
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_SEASON,
    TIME_DECAY_XI,
)

try:
    import soccerdata as sd
except ImportError:
    sd = None

INPUT_PATH = PROCESSED_DIR / "matches_clean.csv"


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


def get_clubelo_rating(team: str, as_of_date: pd.Timestamp) -> float | None:
    """
    Query ClubElo API for a team's rating as of a specific date.
    
    Returns None if ClubElo unavailable or team not found.
    """
    if sd is None:
        return None
    
    try:
        # ClubElo uses specific date format
        date_str = as_of_date.strftime("%Y-%m-%d")
        clubelo = sd.ClubElo()
        
        # ClubElo team name mapping (EPL canonical -> ClubElo format)
        # Most names match, but some need mapping
        team_mapping = {
            "Brighton & Hove Albion": "Brighton",
            "Manchester City": "Man City",
            "Manchester United": "Man United",
            "Newcastle United": "Newcastle",
            "Nottingham Forest": "Nott'm Forest",
            "Sheffield United": "Sheffield Utd",
            "Tottenham Hotspur": "Tottenham",
            "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves",
        }
        clubelo_name = team_mapping.get(team, team)
        
        # Get rating history up to the date
        ratings = clubelo.read_by_date(date_str)
        
        # Filter for the team
        team_rating = ratings[ratings.index.get_level_values('team') == clubelo_name]
        
        if not team_rating.empty:
            # Get the most recent rating before or on the date
            return float(team_rating['elo'].iloc[-1])
        
        return None
        
    except Exception:
        # ClubElo API down, network error, or team not found
        return None


def get_bottom_three_average_elo(
    elo_ratings: dict[str, float],
    matches: pd.DataFrame,
    current_season: str,
) -> float:
    """
    Fallback: compute average Elo of bottom 3 teams from previous season's final table.
    
    If no prior season data, returns default rating.
    """
    # Find previous season
    seasons = sorted(matches['season'].unique())
    try:
        prev_season_idx = seasons.index(current_season) - 1
        if prev_season_idx < 0:
            return ELO_DEFAULT_RATING
        prev_season = seasons[prev_season_idx]
    except (ValueError, IndexError):
        return ELO_DEFAULT_RATING
    
    # Get final table of previous season (points, GD, GF)
    prev_matches = matches[matches['season'] == prev_season].copy()
    if prev_matches.empty:
        return ELO_DEFAULT_RATING
    
    # Build final standings
    standings = {}
    for _, match in prev_matches.iterrows():
        for team, goals_for, goals_against, is_home in [
            (match['team_home'], match['goals_home'], match['goals_away'], True),
            (match['team_away'], match['goals_away'], match['goals_home'], False),
        ]:
            if team not in standings:
                standings[team] = {'points': 0, 'gf': 0, 'ga': 0}
            
            standings[team]['gf'] += goals_for
            standings[team]['ga'] += goals_against
            
            if goals_for > goals_against:
                standings[team]['points'] += 3
            elif goals_for == goals_against:
                standings[team]['points'] += 1
    
    # Sort by points, GD, GF
    sorted_teams = sorted(
        standings.items(),
        key=lambda x: (x[1]['points'], x[1]['gf'] - x[1]['ga'], x[1]['gf']),
        reverse=False  # ascending = bottom teams first
    )
    
    # Get bottom 3 teams' current Elo (from elo_ratings dict)
    bottom_three_elos = []
    for team_name, _ in sorted_teams[:3]:
        if team_name in elo_ratings:
            bottom_three_elos.append(elo_ratings[team_name])
    
    if bottom_three_elos:
        return float(np.mean(bottom_three_elos))
    
    return ELO_DEFAULT_RATING


def identify_promoted_teams(matches: pd.DataFrame, season: str) -> set[str]:
    """Identify teams appearing in season but not in previous season."""
    seasons = sorted(matches['season'].unique())
    try:
        season_idx = seasons.index(season)
        if season_idx == 0:
            return set()
        
        prev_season = seasons[season_idx - 1]
        prev_teams = set(matches[matches['season'] == prev_season]['team_home']) | \
                      set(matches[matches['season'] == prev_season]['team_away'])
        curr_teams = set(matches[matches['season'] == season]['team_home']) | \
                      set(matches[matches['season'] == season]['team_away'])
        
        return curr_teams - prev_teams
    
    except (ValueError, IndexError):
        return set()


def build_elo_history(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pre- and post-match Elo ratings with ClubElo cold-start for promoted teams."""
    elo = pb.ratings.Elo(k=ELO_K, home_field_advantage=ELO_HOME_ADVANTAGE)
    history_rows = []
    result_codes = {"H": 0, "D": 1, "A": 2}
    
    # Track which teams got ClubElo vs fallback
    cold_start_log = []
    
    ordered = matches.sort_values(["datetime", "match_id"])
    
    # Process season by season to handle promoted teams
    for season in sorted(matches['season'].unique()):
        season_matches = ordered[ordered['season'] == season]
        promoted = identify_promoted_teams(matches, season)
        
        if promoted:
            # Get first match date of the season for ClubElo query
            first_match_date = season_matches['date'].min()
            
            for team in promoted:
                # Try ClubElo first
                clubelo_rating = get_clubelo_rating(team, first_match_date)
                
                if clubelo_rating is not None:
                    elo.ratings[team] = clubelo_rating
                    cold_start_log.append({
                        'season': season,
                        'team': team,
                        'elo_rating': clubelo_rating,
                        'source': 'ClubElo',
                    })
                else:
                    # Fallback: average of bottom 3 from previous season
                    fallback_rating = get_bottom_three_average_elo(
                        elo.ratings, matches, season
                    )
                    elo.ratings[team] = fallback_rating
                    cold_start_log.append({
                        'season': season,
                        'team': team,
                        'elo_rating': fallback_rating,
                        'source': 'bottom_3_average_fallback',
                    })
        
        # Process matches for this season
        for match in season_matches.itertuples(index=False):
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
    
    # Log cold-start info
    if cold_start_log:
        print("\nElo cold-start untuk tim promosi:")
        for entry in cold_start_log:
            print(f"  {entry['season']} - {entry['team']}: {entry['elo_rating']:.1f} ({entry['source']})")
    
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

    # Save training metadata for consistency checking
    import json
    metadata = {
        "training_matches": len(train),
        "training_seasons": sorted(train["season"].unique().tolist()),
        "training_period_start": train["season"].min(),
        "training_period_end": train["season"].max(),
        "test_season": TEST_SEASON,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": ["poisson", "dixon_coles", "elo"],
    }
    (MODELS_DIR / "statistical_models_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

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
