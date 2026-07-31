"""Test leakage-safe stacking of statistical expected goals into the best ML model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss

from statistical_models import fit_goal_models, predict_fixture
from train_model import (
    RANDOM_STATE,
    TARGET_MAPPING,
    TARGET_NAMES,
    TEST_SEASON,
    build_models,
    date_based_time_splits,
    load_modeling_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
PHASE4_PROBABILITIES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_match_probabilities.csv"
)
PHASE5_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ml_model_metrics.csv"
STACKING_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stacking_expected_goals.csv"
)
STACKING_METRICS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stacking_model_metrics.csv"
)
STACKING_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stacking_test_predictions.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "stacked_random_forest.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "stacked_random_forest_metadata.json"

STACKING_COLUMNS = [
    "poisson_expected_goals_home",
    "poisson_expected_goals_away",
    "dixon_coles_expected_goals_home",
    "dixon_coles_expected_goals_away",
]
BOOTSTRAP_SAMPLES = 5000


def expected_goals_for_matches(
    matches: pd.DataFrame,
    poisson,
    dixon_coles,
    source: str,
    fold: int | None,
) -> list[dict]:
    """Predict expected goals for matches unseen by the fitted goal models."""
    rows = []
    for match in matches.itertuples(index=False):
        poisson_grid, poisson_cold = predict_fixture(
            poisson, match.team_home, match.team_away, dixon_coles=False
        )
        dc_grid, dc_cold = predict_fixture(
            dixon_coles, match.team_home, match.team_away, dixon_coles=True
        )
        rows.append(
            {
                "match_id": match.match_id,
                "date": match.date,
                "source": source,
                "fold": fold,
                "cold_start": poisson_cold or dc_cold,
                "poisson_expected_goals_home": poisson_grid.home_goal_expectation,
                "poisson_expected_goals_away": poisson_grid.away_goal_expectation,
                "dixon_coles_expected_goals_home": dc_grid.home_goal_expectation,
                "dixon_coles_expected_goals_away": dc_grid.away_goal_expectation,
            }
        )
    return rows


def generate_oof_expected_goals(train: pd.DataFrame) -> pd.DataFrame:
    """Generate expanding-window expected goals for validation rows only."""
    rows = []
    for fold, train_idx, validation_idx in date_based_time_splits(train):
        fold_train = train.iloc[train_idx]
        fold_validation = train.iloc[validation_idx]
        print(
            f"Expected goals fold {fold}: fit {len(fold_train)}, "
            f"predict {len(fold_validation)}"
        )
        poisson, dixon_coles = fit_goal_models(fold_train)
        rows.extend(
            expected_goals_for_matches(
                fold_validation,
                poisson,
                dixon_coles,
                source="train_oof",
                fold=fold,
            )
        )
    return pd.DataFrame(rows)


def load_test_expected_goals() -> pd.DataFrame:
    """Load train-only Phase 4 expected goals for the final-season test set."""
    columns = ["match_id", "date", "cold_start"] + STACKING_COLUMNS
    expected_goals = pd.read_csv(
        PHASE4_PROBABILITIES_PATH,
        usecols=columns,
        parse_dates=["date"],
    )
    expected_goals["source"] = "test_train_only"
    expected_goals["fold"] = pd.NA
    return expected_goals[
        ["match_id", "date", "source", "fold", "cold_start"] + STACKING_COLUMNS
    ]


def evaluate_cv(model, data: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Return forward-validation metrics for one fixed model specification."""
    fold_metrics = []
    X = data[feature_columns]
    y = data["target"]

    for _fold, train_idx, validation_idx in date_based_time_splits(data):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities = fitted.predict_proba(X.iloc[validation_idx])
        predictions = fitted.predict(X.iloc[validation_idx])
        fold_metrics.append(
            {
                "accuracy": accuracy_score(y.iloc[validation_idx], predictions),
                "macro_f1": f1_score(
                    y.iloc[validation_idx], predictions, average="macro"
                ),
                "log_loss": log_loss(
                    y.iloc[validation_idx], probabilities, labels=[0, 1, 2]
                ),
            }
        )

    metrics = pd.DataFrame(fold_metrics)
    return {
        "cv_accuracy_mean": metrics["accuracy"].mean(),
        "cv_accuracy_std": metrics["accuracy"].std(),
        "cv_macro_f1_mean": metrics["macro_f1"].mean(),
        "cv_macro_f1_std": metrics["macro_f1"].std(),
        "cv_log_loss_mean": metrics["log_loss"].mean(),
        "cv_log_loss_std": metrics["log_loss"].std(),
    }


def fit_and_evaluate(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
):
    """Fit a model and return holdout metrics, probabilities, and predictions."""
    fitted = clone(model).fit(train[feature_columns], train["target"])
    probabilities = fitted.predict_proba(test[feature_columns])
    predictions = fitted.predict(test[feature_columns])
    metrics = {
        "test_accuracy": accuracy_score(test["target"], predictions),
        "test_macro_f1": f1_score(test["target"], predictions, average="macro"),
        "test_log_loss": log_loss(
            test["target"], probabilities, labels=[0, 1, 2]
        ),
    }
    return fitted, metrics, probabilities, predictions


def paired_bootstrap_log_loss(
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    stacked_probabilities: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate the paired test log-loss improvement and its 95% bootstrap CI."""
    row_index = np.arange(len(y_true))
    baseline_losses = -np.log(
        np.clip(baseline_probabilities[row_index, y_true], 1e-15, 1.0)
    )
    stacked_losses = -np.log(
        np.clip(stacked_probabilities[row_index, y_true], 1e-15, 1.0)
    )
    per_match_improvement = baseline_losses - stacked_losses

    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.integers(
        0, len(y_true), size=(BOOTSTRAP_SAMPLES, len(y_true))
    )
    bootstrap_means = per_match_improvement[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(per_match_improvement.mean()), float(lower), float(upper)


def main() -> None:
    """Run the controlled Phase 6 stacking experiment."""
    data, baseline_columns = load_modeling_data()
    data["target"] = data["result"].map(TARGET_MAPPING)
    train = data.loc[data["season"] != TEST_SEASON].copy()
    test = data.loc[data["season"] == TEST_SEASON].copy()

    match_goals = pd.read_csv(
        MATCHES_PATH,
        usecols=["match_id", "goals_home", "goals_away"],
    )
    train_for_statistics = train.merge(
        match_goals,
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if train_for_statistics[["goals_home", "goals_away"]].isna().any().any():
        raise ValueError("Skor historis untuk fitting model statistik tidak lengkap")

    oof_expected_goals = generate_oof_expected_goals(train_for_statistics)
    test_expected_goals = load_test_expected_goals()
    stacking_features = pd.concat(
        [oof_expected_goals, test_expected_goals], ignore_index=True
    )
    if stacking_features["match_id"].duplicated().any():
        raise ValueError("Expected goals stacking memiliki match_id duplikat")
    stacking_features.to_csv(
        STACKING_FEATURES_PATH,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    merge_columns = ["match_id"] + STACKING_COLUMNS
    controlled_train = train.merge(
        oof_expected_goals[merge_columns],
        on="match_id",
        how="inner",
        validate="one_to_one",
    ).sort_values(["date", "datetime", "match_id"], ignore_index=True)
    stacked_test = test.merge(
        test_expected_goals[merge_columns],
        on="match_id",
        how="left",
        validate="one_to_one",
    ).sort_values(["date", "datetime", "match_id"], ignore_index=True)
    if stacked_test[STACKING_COLUMNS].isna().any().any():
        raise ValueError("Expected goals test tidak lengkap")

    random_forest = build_models()["random_forest"]
    stacked_columns = baseline_columns + STACKING_COLUMNS

    print(
        f"Controlled train: {len(controlled_train)} | Test: {len(stacked_test)} | "
        f"Baseline features: {len(baseline_columns)} | Stacked: {len(stacked_columns)}"
    )
    baseline_cv = evaluate_cv(random_forest, controlled_train, baseline_columns)
    stacked_cv = evaluate_cv(random_forest, controlled_train, stacked_columns)
    baseline_model, baseline_test, baseline_probabilities, baseline_predictions = (
        fit_and_evaluate(
            random_forest, controlled_train, stacked_test, baseline_columns
        )
    )
    stacked_model, stacked_test_metrics, stacked_probabilities, stacked_predictions = (
        fit_and_evaluate(
            random_forest, controlled_train, stacked_test, stacked_columns
        )
    )

    improvement, ci_lower, ci_upper = paired_bootstrap_log_loss(
        stacked_test["target"].to_numpy(),
        baseline_probabilities,
        stacked_probabilities,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "random_forest_controlled_baseline",
                "train_rows": len(controlled_train),
                "feature_count": len(baseline_columns),
                **baseline_cv,
                **baseline_test,
            },
            {
                "model": "random_forest_stacked_xg",
                "train_rows": len(controlled_train),
                "feature_count": len(stacked_columns),
                **stacked_cv,
                **stacked_test_metrics,
            },
        ]
    )
    metrics["test_log_loss_improvement_vs_control"] = [0.0, improvement]
    metrics["bootstrap_95_ci_lower"] = [np.nan, ci_lower]
    metrics["bootstrap_95_ci_upper"] = [np.nan, ci_upper]

    phase5_metrics = pd.read_csv(PHASE5_METRICS_PATH)
    phase5_rf = phase5_metrics.loc[phase5_metrics["model"] == "random_forest"].iloc[0]
    metrics["phase5_rf_test_log_loss_reference"] = phase5_rf["test_log_loss"]
    metrics["phase5_rf_test_accuracy_reference"] = phase5_rf["test_accuracy"]
    metrics.to_csv(STACKING_METRICS_PATH, index=False)

    prediction_output = stacked_test[
        ["match_id", "season", "datetime", "team_home", "team_away", "result"]
    ].copy()
    prediction_output["baseline_prediction"] = [
        TARGET_NAMES[int(value)] for value in baseline_predictions
    ]
    prediction_output["stacked_prediction"] = [
        TARGET_NAMES[int(value)] for value in stacked_predictions
    ]
    for index, label in enumerate(("home", "draw", "away")):
        prediction_output[f"baseline_prob_{label}"] = baseline_probabilities[:, index]
        prediction_output[f"stacked_prob_{label}"] = stacked_probabilities[:, index]
    prediction_output.to_csv(
        STACKING_PREDICTIONS_PATH,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    joblib.dump(stacked_model, MODEL_PATH)
    metadata = {
        "model_name": "random_forest_stacked_xg",
        "experiment": "Phase 6 statistical expected goals stacking",
        "train_rows": len(controlled_train),
        "test_rows": len(stacked_test),
        "baseline_feature_columns": baseline_columns,
        "stacking_feature_columns": STACKING_COLUMNS,
        "all_feature_columns": stacked_columns,
        "target_mapping": TARGET_MAPPING,
        "selection_warning": (
            "This experiment model does not replace the Phase 5 best model automatically."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    probabilities_valid = np.allclose(
        stacked_probabilities.sum(axis=1), 1.0, atol=1e-7
    ) and np.allclose(baseline_probabilities.sum(axis=1), 1.0, atol=1e-7)
    if not probabilities_valid or len(prediction_output) != len(test):
        raise ValueError("Validasi output stacking gagal")

    print("\nControlled comparison:")
    print(
        metrics[
            [
                "model",
                "cv_log_loss_mean",
                "cv_accuracy_mean",
                "test_log_loss",
                "test_accuracy",
                "test_macro_f1",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nTest log-loss improvement: {improvement:+.6f} "
        f"(95% bootstrap CI {ci_lower:+.6f} to {ci_upper:+.6f})"
    )
    print(f"Stacked model tersimpan: {MODEL_PATH}")
    print("Validasi output: lolos")


if __name__ == "__main__":
    main()
