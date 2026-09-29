"""Train and compare leakage-safe EPL classification models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    MODELS_DIR,
    N_SPLITS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_MAPPING,
    TARGET_NAMES,
    TEST_SEASON,
)

FEATURES_PATH = PROCESSED_DIR / "features.csv"
ELO_PATH = PROCESSED_DIR / "elo_history.csv"
METRICS_PATH = PROCESSED_DIR / "ml_model_metrics.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "ml_test_predictions.csv"
MODEL_PATH = MODELS_DIR / "best_ml_model.pkl"
METADATA_PATH = MODELS_DIR / "best_ml_model_metadata.json"

NON_FEATURE_COLUMNS = {
    "match_id",
    "competition",
    "season",
    "datetime",
    "date",
    "team_home",
    "team_away",
    "result",
}


def load_modeling_data() -> tuple[pd.DataFrame, list[str]]:
    """Merge Phase 3 features with leakage-safe pre-match Elo ratings."""
    features = pd.read_csv(FEATURES_PATH, parse_dates=["datetime", "date"])
    elo = pd.read_csv(ELO_PATH, parse_dates=["datetime", "date"])
    home_elo = (
        elo.loc[
            elo["venue"] == "home",
            ["match_id", "team", "opponent", "elo_pre", "opponent_elo_pre"],
        ]
        .rename(
            columns={
                "team": "elo_team_home",
                "opponent": "elo_team_away",
                "elo_pre": "home_elo_pre",
                "opponent_elo_pre": "away_elo_pre",
            }
        )
        .copy()
    )
    if home_elo["match_id"].duplicated().any():
        raise ValueError("Histori Elo home memiliki match_id duplikat")

    data = features.merge(home_elo, on="match_id", how="left", validate="one_to_one")
    if data[["home_elo_pre", "away_elo_pre"]].isna().any().any():
        raise ValueError("Ada pertandingan yang tidak memiliki Elo pre-match")
    if not (
        data["team_home"].eq(data["elo_team_home"])
        & data["team_away"].eq(data["elo_team_away"])
    ).all():
        raise ValueError("Identitas tim antara features dan histori Elo tidak cocok")

    data["elo_gap_pre"] = data["home_elo_pre"] - data["away_elo_pre"]
    data = data.drop(columns=["elo_team_home", "elo_team_away"])
    data = data.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)
    feature_columns = [
        column for column in data.columns if column not in NON_FEATURE_COLUMNS
    ]

    non_numeric = data[feature_columns].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Fitur non-numerik ditemukan: {non_numeric}")
    return data, feature_columns


def build_models() -> dict[str, Pipeline]:
    """Create three progressively more flexible model pipelines."""
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=8,
                        min_samples_leaf=6,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=3,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def date_based_time_splits(data: pd.DataFrame, n_splits: int = N_SPLITS):
    """Apply TimeSeriesSplit to unique dates so a date never spans two folds.

    `n_splits` opsional supaya adapter cv (mis. untuk GridSearchCV di
    model_tuning.py) bisa meminta jumlah fold lain tanpa menduplikasi logika.
    Default tetap N_SPLITS supaya perilaku Fase 5 tidak berubah.
    """
    unique_dates = np.array(sorted(data["date"].unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_date_idx, validation_date_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        train_dates = unique_dates[train_date_idx]
        validation_dates = unique_dates[validation_date_idx]
        train_idx = np.flatnonzero(data["date"].isin(train_dates).to_numpy())
        validation_idx = np.flatnonzero(
            data["date"].isin(validation_dates).to_numpy()
        )
        if data.iloc[train_idx]["date"].max() >= data.iloc[validation_idx]["date"].min():
            raise ValueError(f"Temporal overlap pada fold {fold}")
        yield fold, train_idx, validation_idx


def cross_validate_models(
    models: dict[str, Pipeline],
    train: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Evaluate each model on the same forward-only validation folds."""
    X = train[feature_columns]
    y = train["target"]
    rows = []

    splits = list(date_based_time_splits(train))
    for model_name, model in models.items():
        print(f"TimeSeriesSplit: {model_name}")
        for fold, train_idx, validation_idx in splits:
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probabilities = fold_model.predict_proba(X.iloc[validation_idx])
            predictions = fold_model.predict(X.iloc[validation_idx])
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_end": train.iloc[train_idx]["date"].max(),
                    "validation_start": train.iloc[validation_idx]["date"].min(),
                    "validation_end": train.iloc[validation_idx]["date"].max(),
                    "train_rows": len(train_idx),
                    "validation_rows": len(validation_idx),
                    "accuracy": accuracy_score(y.iloc[validation_idx], predictions),
                    "macro_f1": f1_score(
                        y.iloc[validation_idx], predictions, average="macro"
                    ),
                    "log_loss": log_loss(
                        y.iloc[validation_idx], probabilities, labels=[0, 1, 2]
                    ),
                }
            )
    return pd.DataFrame(rows)


def fit_and_evaluate_holdout(
    models: dict[str, Pipeline],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    """Fit on all pre-test data and evaluate once on the final-season holdout."""
    fitted_models = {}
    metric_rows = []
    prediction_rows = []

    X_train = train[feature_columns]
    y_train = train["target"]
    X_test = test[feature_columns]
    y_test = test["target"]

    for model_name, model in models.items():
        fitted = clone(model).fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_test)
        predictions = fitted.predict(X_test)
        fitted_models[model_name] = fitted
        metric_rows.append(
            {
                "model": model_name,
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_macro_f1": f1_score(y_test, predictions, average="macro"),
                "test_log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
            }
        )

        for index, match in enumerate(test.itertuples(index=False)):
            prediction_rows.append(
                {
                    "match_id": match.match_id,
                    "season": match.season,
                    "datetime": match.datetime,
                    "team_home": match.team_home,
                    "team_away": match.team_away,
                    "actual_result": match.result,
                    "model": model_name,
                    "predicted_result": TARGET_NAMES[int(predictions[index])],
                    "prob_home": probabilities[index, TARGET_MAPPING["H"]],
                    "prob_draw": probabilities[index, TARGET_MAPPING["D"]],
                    "prob_away": probabilities[index, TARGET_MAPPING["A"]],
                }
            )

    return fitted_models, pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_metrics(
    cv_results: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine cross-validation summaries with untouched test performance."""
    cv_summary = (
        cv_results.groupby("model")
        .agg(
            cv_accuracy_mean=("accuracy", "mean"),
            cv_accuracy_std=("accuracy", "std"),
            cv_macro_f1_mean=("macro_f1", "mean"),
            cv_macro_f1_std=("macro_f1", "std"),
            cv_log_loss_mean=("log_loss", "mean"),
            cv_log_loss_std=("log_loss", "std"),
        )
        .reset_index()
    )
    return cv_summary.merge(holdout_metrics, on="model", validate="one_to_one").sort_values(
        "cv_log_loss_mean", ignore_index=True
    )


def validate_outputs(
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Check temporal split, coverage, and probability invariants."""
    errors = []
    if train["date"].max() >= test["date"].min():
        errors.append("train dan test overlap secara waktu")
    if set(metrics["model"]) != {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }:
        errors.append("metrics tidak mencakup ketiga model")
    if len(predictions) != len(test) * 3:
        errors.append("prediksi test tidak lengkap")
    probability_sum = predictions[["prob_home", "prob_draw", "prob_away"]].sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-7):
        errors.append("probabilitas test tidak berjumlah satu")
    if predictions.isna().any().any():
        errors.append("missing value pada prediksi test")
    if len(data) != len(train) + len(test):
        errors.append("baris hilang saat split")

    if errors:
        raise ValueError("; ".join(errors))


def main() -> None:
    """Run Phase 5 training, comparison, and model persistence."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data, feature_columns = load_modeling_data()
    data["target"] = data["result"].map(TARGET_MAPPING)
    if data["target"].isna().any():
        raise ValueError("Target di luar kelas H/D/A")

    train = data.loc[data["season"] != TEST_SEASON].copy()
    test = data.loc[data["season"] == TEST_SEASON].copy()
    if train.empty or test.empty:
        raise ValueError("Train atau test set kosong")

    models = build_models()
    print(
        f"Train: {len(train)} match | Test {TEST_SEASON}: {len(test)} match | "
        f"Fitur: {len(feature_columns)}"
    )
    cv_results = cross_validate_models(models, train, feature_columns)
    fitted_models, holdout_metrics, predictions = fit_and_evaluate_holdout(
        models, train, test, feature_columns
    )
    metrics = summarize_metrics(cv_results, holdout_metrics)
    validate_outputs(data, train, test, metrics, predictions)

    # Select by forward-validation log loss, never by the holdout test result.
    best_model_name = metrics.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "model_name": best_model_name,
        "selection_metric": "mean TimeSeriesSplit validation log loss",
        "test_season": TEST_SEASON,
        "train_rows": len(train),
        "test_rows": len(test),
        "training_matches": len(train),
        "training_seasons": sorted(train["season"].unique().tolist()),
        "training_period_start": train["season"].min(),
        "training_period_end": train["season"].max(),
        "timestamp": pd.Timestamp.now().isoformat(),
        "feature_columns": feature_columns,
        "target_mapping": TARGET_MAPPING,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(
        PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )

    print("\nPerbandingan model:")
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
    print(f"\nModel terbaik berdasarkan CV log loss: {best_model_name}")
    print(f"Tersimpan: {MODEL_PATH}")
    print("Validasi output: lolos")


if __name__ == "__main__":
    main()
