"""Compare model probabilities with manual bookmaker odds for learning only."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import penaltyblog as pb

from config import (
    COMPETITION,
    ELO_DEFAULT_RATING,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from feature_engineering import generate_features
from statistical_models import predict_fixture
from team_mapping import TEAM_NAME_MAPPING


MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
ELO_RATINGS_PATH = PROCESSED_DIR / "elo_current_ratings.csv"
ML_MODEL_PATH = MODELS_DIR / "best_ml_model.pkl"
ML_METADATA_PATH = MODELS_DIR / "best_ml_model_metadata.json"
POISSON_MODEL_PATH = MODELS_DIR / "poisson_goal_model.pkl"
DIXON_COLES_MODEL_PATH = MODELS_DIR / "dixon_coles_goal_model.pkl"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "upcoming_fixtures_example.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "value_betting_output.csv"

REQUIRED_INPUT_COLUMNS = [
    "fixture_id",
    "season",
    "datetime",
    "team_home",
    "team_away",
    "odds_home",
    "odds_draw",
    "odds_away",
]
OUTCOMES = [
    ("H", "Home Win", "odds_home", "prob_home"),
    ("D", "Draw", "odds_draw", "prob_draw"),
    ("A", "Away Win", "odds_away", "prob_away"),
]
DISCLAIMER = (
    "EDUCATIONAL ONLY: not a betting recommendation. High EV may be noise, "
    "not a real edge."
)
DATA_WARNING = (
    "Only about 3,800 EPL matches were used; market odds beat every model on "
    "RPS and log loss in Phase 7."
)


def parse_args() -> argparse.Namespace:
    """Parse optional manual input and output paths."""
    parser = argparse.ArgumentParser(
        description="Educational model-vs-market EV comparison for EPL fixtures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV containing future fixtures and decimal 1X2 odds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV for the sorted EV table.",
    )
    return parser.parse_args()


def load_fixtures(path: Path, historical_matches: pd.DataFrame) -> pd.DataFrame:
    """Load and validate manually supplied future fixtures and decimal odds."""
    fixtures = pd.read_csv(path)
    missing = sorted(set(REQUIRED_INPUT_COLUMNS) - set(fixtures.columns))
    if missing:
        raise ValueError(f"Kolom input fixture tidak lengkap: {missing}")

    fixtures = fixtures[REQUIRED_INPUT_COLUMNS].copy()
    fixtures["datetime"] = pd.to_datetime(fixtures["datetime"], errors="raise")
    for column in ("team_home", "team_away"):
        fixtures[column] = (
            fixtures[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace(TEAM_NAME_MAPPING)
        )
    for column in ("odds_home", "odds_draw", "odds_away"):
        fixtures[column] = pd.to_numeric(fixtures[column], errors="raise")
        if (fixtures[column] <= 1.0).any():
            raise ValueError(f"{column} harus berupa decimal odds di atas 1.0")

    if fixtures["fixture_id"].isna().any() or fixtures["fixture_id"].duplicated().any():
        raise ValueError("fixture_id wajib terisi dan unik")
    if (fixtures["team_home"] == fixtures["team_away"]).any():
        raise ValueError("Tim kandang dan tandang tidak boleh sama")
    if fixtures["season"].isna().any():
        raise ValueError("season wajib terisi")

    latest_history = historical_matches["datetime"].max()
    if (fixtures["datetime"] <= latest_history).any():
        raise ValueError(
            f"Semua fixture harus setelah data historis terakhir ({latest_history})"
        )
    return fixtures.sort_values(["datetime", "fixture_id"]).reset_index(drop=True)


def build_ml_feature_rows(
    fixtures: pd.DataFrame,
    historical_matches: pd.DataFrame,
    elo_ratings: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Reconstruct each future fixture's pre-match features independently."""
    rows = []
    for fixture in fixtures.itertuples(index=False):
        # The synthetic score is never used for this fixture's features because
        # generate_features snapshots a date before applying that date's result.
        synthetic = pd.DataFrame(
            [
                {
                    "match_id": fixture.fixture_id,
                    "competition": COMPETITION,
                    "season": fixture.season,
                    "datetime": fixture.datetime,
                    "date": fixture.datetime.normalize(),
                    "team_home": fixture.team_home,
                    "team_away": fixture.team_away,
                    "goals_home": 0,
                    "goals_away": 0,
                    "result": "D",
                }
            ]
        )
        combined = pd.concat(
            [
                historical_matches[
                    [
                        "match_id",
                        "competition",
                        "season",
                        "datetime",
                        "date",
                        "team_home",
                        "team_away",
                        "goals_home",
                        "goals_away",
                        "result",
                    ]
                ],
                synthetic,
            ],
            ignore_index=True,
        )
        generated = generate_features(combined)
        feature_row = generated.loc[generated["match_id"] == fixture.fixture_id].iloc[0]
        home_elo = float(elo_ratings.get(fixture.team_home, ELO_DEFAULT_RATING))
        away_elo = float(elo_ratings.get(fixture.team_away, ELO_DEFAULT_RATING))

        row = {"fixture_id": fixture.fixture_id}
        for column in feature_columns:
            if column == "home_elo_pre":
                row[column] = home_elo
            elif column == "away_elo_pre":
                row[column] = away_elo
            elif column == "elo_gap_pre":
                row[column] = home_elo - away_elo
            else:
                row[column] = feature_row[column]
        rows.append(row)
    return pd.DataFrame(rows)


def decode_market_probabilities(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Remove bookmaker overround with penaltyblog's multiplicative method."""
    rows = []
    for fixture in fixtures.itertuples(index=False):
        decoded = pb.implied.calculate_implied(
            [fixture.odds_home, fixture.odds_draw, fixture.odds_away],
            method="multiplicative",
            odds_format="decimal",
            market_names=["Home Win", "Draw", "Away Win"],
        )
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "overround": decoded.margin,
                "market_prob_home": decoded.probabilities[0],
                "market_prob_draw": decoded.probabilities[1],
                "market_prob_away": decoded.probabilities[2],
                "decode_method": decoded.method.value,
            }
        )
    return pd.DataFrame(rows)


def predict_all_models(
    fixtures: pd.DataFrame,
    ml_features: pd.DataFrame,
    feature_columns: list[str],
    known_teams: set[str],
    ml_model,
    poisson,
    dixon_coles,
) -> pd.DataFrame:
    """Generate H/D/A probabilities from the best ML model and Track A models."""
    rows = []
    ml_input = ml_features.set_index("fixture_id").loc[
        fixtures["fixture_id"], feature_columns
    ]
    ml_probabilities = ml_model.predict_proba(ml_input)

    for index, fixture in enumerate(fixtures.itertuples(index=False)):
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "model": "random_forest_phase5",
                "model_cold_start": (
                    fixture.team_home not in known_teams
                    or fixture.team_away not in known_teams
                ),
                "prob_home": ml_probabilities[index, 0],
                "prob_draw": ml_probabilities[index, 1],
                "prob_away": ml_probabilities[index, 2],
            }
        )

        poisson_grid, poisson_cold = predict_fixture(
            poisson, fixture.team_home, fixture.team_away, dixon_coles=False
        )
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "model": "poisson",
                "model_cold_start": poisson_cold,
                "prob_home": poisson_grid.home_win,
                "prob_draw": poisson_grid.draw,
                "prob_away": poisson_grid.away_win,
            }
        )

        dc_grid, dc_cold = predict_fixture(
            dixon_coles, fixture.team_home, fixture.team_away, dixon_coles=True
        )
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "model": "dixon_coles",
                "model_cold_start": dc_cold,
                "prob_home": dc_grid.home_win,
                "prob_draw": dc_grid.draw,
                "prob_away": dc_grid.away_win,
            }
        )

    predictions = pd.DataFrame(rows)
    probability_columns = ["prob_home", "prob_draw", "prob_away"]
    probabilities = predictions[probability_columns].to_numpy()
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions[probability_columns] = probabilities
    return predictions


def build_ev_table(
    fixtures: pd.DataFrame,
    market: pd.DataFrame,
    predictions: pd.DataFrame,
    historical_data_cutoff: pd.Timestamp,
    model_training_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Create one EV row per fixture, model, and possible outcome."""
    fixture_market = fixtures.merge(
        market, on="fixture_id", how="left", validate="one_to_one"
    )
    merged = fixture_market.merge(
        predictions, on="fixture_id", how="left", validate="one_to_many"
    )
    rows = []
    for item in merged.itertuples(index=False):
        for outcome_code, outcome_name, odds_column, probability_column in OUTCOMES:
            odds = float(getattr(item, odds_column))
            model_probability = float(getattr(item, probability_column))
            market_probability = float(
                getattr(item, f"market_{probability_column}")
            )
            expected_value = model_probability * odds - 1.0
            rows.append(
                {
                    "fixture_id": item.fixture_id,
                    "season": item.season,
                    "datetime": item.datetime,
                    "fixture": f"{item.team_home} vs {item.team_away}",
                    "team_home": item.team_home,
                    "team_away": item.team_away,
                    "model": item.model,
                    "model_cold_start": item.model_cold_start,
                    "outcome_code": outcome_code,
                    "outcome": outcome_name,
                    "model_probability": model_probability,
                    "market_probability": market_probability,
                    "probability_edge": model_probability - market_probability,
                    "decimal_odds": odds,
                    "overround": item.overround,
                    "decode_method": item.decode_method,
                    "expected_value": expected_value,
                    "expected_value_percent": expected_value * 100,
                    "historical_data_cutoff": historical_data_cutoff,
                    "model_training_cutoff": model_training_cutoff,
                    "analysis_only": True,
                    "disclaimer": DISCLAIMER,
                    "data_warning": DATA_WARNING,
                }
            )

    output = pd.DataFrame(rows).sort_values(
        ["expected_value", "model_probability"],
        ascending=[False, False],
        ignore_index=True,
    )
    output.insert(0, "ev_rank", np.arange(1, len(output) + 1))
    return output


def validate_output(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    """Validate model coverage, probability sums, and EV arithmetic."""
    errors = []
    expected_models = {"random_forest_phase5", "poisson", "dixon_coles"}
    if set(predictions["model"]) != expected_models:
        errors.append("prediksi tidak mencakup semua model")
    if len(predictions) != len(fixtures) * len(expected_models):
        errors.append("jumlah prediksi model tidak lengkap")
    if len(output) != len(fixtures) * len(expected_models) * 3:
        errors.append("jumlah baris EV tidak lengkap")
    if output.isna().any().any():
        errors.append("missing value pada output EV")
    probability_sum = predictions[["prob_home", "prob_draw", "prob_away"]].sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-9):
        errors.append("probabilitas model tidak berjumlah satu")
    recomputed_ev = output["model_probability"] * output["decimal_odds"] - 1.0
    if not np.allclose(output["expected_value"], recomputed_ev, atol=1e-12):
        errors.append("perhitungan EV tidak konsisten")
    if not output["expected_value"].is_monotonic_decreasing:
        errors.append("output belum diurutkan dari EV tertinggi")

    if errors:
        raise ValueError("; ".join(errors))


def main() -> None:
    """Run the educational manual-fixture EV comparison."""
    args = parse_args()
    historical_matches = pd.read_csv(
        MATCHES_PATH, parse_dates=["datetime", "date"]
    )
    fixtures = load_fixtures(args.input, historical_matches)

    metadata = json.loads(ML_METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    ml_model = joblib.load(ML_MODEL_PATH)
    poisson = pb.models.PoissonGoalsModel.load(str(POISSON_MODEL_PATH))
    dixon_coles = pb.models.DixonColesGoalModel.load(str(DIXON_COLES_MODEL_PATH))
    elo_ratings_frame = pd.read_csv(ELO_RATINGS_PATH)
    elo_ratings = dict(
        zip(elo_ratings_frame["team"], elo_ratings_frame["elo_rating"])
    )

    ml_features = build_ml_feature_rows(
        fixtures,
        historical_matches,
        elo_ratings,
        feature_columns,
    )
    market = decode_market_probabilities(fixtures)
    predictions = predict_all_models(
        fixtures,
        ml_features,
        feature_columns,
        set(historical_matches["team_home"]) | set(historical_matches["team_away"]),
        ml_model,
        poisson,
        dixon_coles,
    )
    model_training_cutoff = historical_matches.loc[
        historical_matches["season"] != metadata["test_season"], "datetime"
    ].max()
    output = build_ev_table(
        fixtures,
        market,
        predictions,
        historical_data_cutoff=historical_matches["datetime"].max(),
        model_training_cutoff=model_training_cutoff,
    )
    validate_output(fixtures, predictions, output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 78)
    print(DISCLAIMER)
    print(DATA_WARNING)
    print("Positive EV is a model-market disagreement, NOT proof of profit.")
    print("=" * 78)
    print(
        output[
            [
                "ev_rank",
                "fixture",
                "model",
                "outcome",
                "model_probability",
                "market_probability",
                "decimal_odds",
                "expected_value_percent",
            ]
        ].head(15).to_string(index=False)
    )
    print(f"\nOutput lengkap: {args.output}")
    print("Validasi output: lolos")


if __name__ == "__main__":
    main()
