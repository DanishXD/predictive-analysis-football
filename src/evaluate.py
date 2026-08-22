"""Evaluate every statistical and ML model against decoded bookmaker odds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import penaltyblog as pb
import seaborn as sns
from config import TEST_SEASON
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
STATISTICAL_PATH = PROCESSED_DIR / "test_match_probabilities.csv"
ELO_PATH = PROCESSED_DIR / "elo_history.csv"
ML_PATH = PROCESSED_DIR / "ml_test_predictions.csv"
STACKING_PATH = PROCESSED_DIR / "stacking_test_predictions.csv"

SUMMARY_PATH = PROCESSED_DIR / "evaluation_summary.csv"
CLASS_METRICS_PATH = PROCESSED_DIR / "evaluation_class_metrics.csv"
CONFUSION_PATH = PROCESSED_DIR / "evaluation_confusion_matrices.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "evaluation_predictions.csv"
BOOKMAKER_PATH = PROCESSED_DIR / "bookmaker_probabilities.csv"

CLASS_CODES = ["H", "D", "A"]
CLASS_NAMES = ["Home Win", "Draw", "Away Win"]
TARGET_MAPPING = {"H": 0, "D": 1, "A": 2}
MODEL_ORDER = [
    "poisson",
    "dixon_coles",
    "elo",
    "logistic_regression",
    "random_forest",
    "xgboost",
    "random_forest_stacked_xg",
    "bookmaker_avg_odds",
]
MODEL_TRACKS = {
    "poisson": "Track A",
    "dixon_coles": "Track A",
    "elo": "Track A",
    "logistic_regression": "Track B",
    "random_forest": "Track B",
    "xgboost": "Track B",
    "random_forest_stacked_xg": "Track B stacking",
    "bookmaker_avg_odds": "Benchmark",
}


def base_test_matches() -> pd.DataFrame:
    """Load the canonical test rows and bookmaker odds."""
    columns = [
        "match_id",
        "season",
        "datetime",
        "date",
        "team_home",
        "team_away",
        "result",
        "odds_avg_home",
        "odds_avg_draw",
        "odds_avg_away",
    ]
    matches = pd.read_csv(
        MATCHES_PATH,
        usecols=columns,
        parse_dates=["datetime", "date"],
    )
    matches = matches.loc[matches["season"] == TEST_SEASON].copy()
    return matches.sort_values(["datetime", "match_id"]).reset_index(drop=True)


def decode_bookmaker_odds(matches: pd.DataFrame) -> pd.DataFrame:
    """Remove average-market overround with penaltyblog's multiplicative method."""
    rows = []
    for match in matches.itertuples(index=False):
        odds = [match.odds_avg_home, match.odds_avg_draw, match.odds_avg_away]
        if any(pd.isna(odds)) or any(value <= 1 for value in odds):
            raise ValueError(f"Odds tidak valid untuk {match.match_id}: {odds}")
        decoded = pb.implied.calculate_implied(
            odds,
            method="multiplicative",
            odds_format="decimal",
            market_names=CLASS_NAMES,
        )
        rows.append(
            {
                "match_id": match.match_id,
                "odds_avg_home": odds[0],
                "odds_avg_draw": odds[1],
                "odds_avg_away": odds[2],
                "overround": decoded.margin,
                "prob_home": decoded.probabilities[0],
                "prob_draw": decoded.probabilities[1],
                "prob_away": decoded.probabilities[2],
                "decode_method": decoded.method.value,
            }
        )
    return pd.DataFrame(rows)


def build_elo_probabilities(base: pd.DataFrame) -> pd.DataFrame:
    """Recreate 1X2 Elo probabilities from the saved pre-match ratings."""
    elo_history = pd.read_csv(ELO_PATH)
    home_rows = elo_history.loc[
        (elo_history["season"] == TEST_SEASON) & (elo_history["venue"] == "home"),
        ["match_id", "team", "opponent", "elo_pre", "opponent_elo_pre"],
    ]
    merged = base[["match_id", "team_home", "team_away"]].merge(
        home_rows,
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["elo_pre", "opponent_elo_pre"]].isna().any().any():
        raise ValueError("Elo pre-match test tidak lengkap")
    if not (
        merged["team_home"].eq(merged["team"])
        & merged["team_away"].eq(merged["opponent"])
    ).all():
        raise ValueError("Tim pada histori Elo tidak cocok dengan test set")

    rows = []
    for match in merged.itertuples(index=False):
        elo = pb.ratings.Elo(k=20.0, home_field_advantage=100.0)
        elo.ratings = {
            match.team_home: float(match.elo_pre),
            match.team_away: float(match.opponent_elo_pre),
        }
        probabilities = elo.calculate_match_probabilities(
            match.team_home, match.team_away
        )
        rows.append(
            {
                "match_id": match.match_id,
                "prob_home": probabilities["home_win"],
                "prob_draw": probabilities["draw"],
                "prob_away": probabilities["away_win"],
            }
        )
    return pd.DataFrame(rows)


def standardize_predictions(base: pd.DataFrame) -> pd.DataFrame:
    """Combine all model predictions into one long-format probability table."""
    identity = base[
        [
            "match_id",
            "season",
            "datetime",
            "team_home",
            "team_away",
            "result",
        ]
    ]
    model_frames = []

    statistical = pd.read_csv(STATISTICAL_PATH)
    for model, prefix in (("poisson", "poisson"), ("dixon_coles", "dixon_coles")):
        frame = statistical[
            [
                "match_id",
                f"{prefix}_prob_home",
                f"{prefix}_prob_draw",
                f"{prefix}_prob_away",
            ]
        ].rename(
            columns={
                f"{prefix}_prob_home": "prob_home",
                f"{prefix}_prob_draw": "prob_draw",
                f"{prefix}_prob_away": "prob_away",
            }
        )
        frame["model"] = model
        model_frames.append(frame)

    elo = build_elo_probabilities(base)
    elo["model"] = "elo"
    model_frames.append(elo)

    ml = pd.read_csv(ML_PATH)
    for model in ("logistic_regression", "random_forest", "xgboost"):
        frame = ml.loc[
            ml["model"] == model,
            ["match_id", "prob_home", "prob_draw", "prob_away"],
        ].copy()
        frame["model"] = model
        model_frames.append(frame)

    # Stacking model is optional (may not exist if stacking.py not run)
    if STACKING_PATH.exists():
        stacking = pd.read_csv(STACKING_PATH)
        frame = stacking[
            [
                "match_id",
                "stacked_prob_home",
                "stacked_prob_draw",
                "stacked_prob_away",
            ]
        ].rename(
            columns={
                "stacked_prob_home": "prob_home",
                "stacked_prob_draw": "prob_draw",
                "stacked_prob_away": "prob_away",
            }
        )
        frame["model"] = "random_forest_stacked_xg"
        model_frames.append(frame)
    else:
        print("Stacking model not found, skipping (stacking.py not run)")

    bookmaker = decode_bookmaker_odds(base)
    bookmaker.to_csv(BOOKMAKER_PATH, index=False)
    frame = bookmaker[["match_id", "prob_home", "prob_draw", "prob_away"]].copy()
    frame["model"] = "bookmaker_avg_odds"
    model_frames.append(frame)

    predictions = pd.concat(model_frames, ignore_index=True)
    predictions = identity.merge(
        predictions,
        on="match_id",
        how="left",
        validate="one_to_many",
    )
    predictions["track"] = predictions["model"].map(MODEL_TRACKS)
    probability_columns = ["prob_home", "prob_draw", "prob_away"]
    probability_array = predictions[probability_columns].to_numpy()
    probability_array = probability_array / probability_array.sum(axis=1, keepdims=True)
    predictions[probability_columns] = probability_array
    predictions["predicted_result"] = np.array(CLASS_CODES)[
        probability_array.argmax(axis=1)
    ]
    return predictions.sort_values(
        ["model", "datetime", "match_id"], ignore_index=True
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate mean one-vs-rest ECE across Home/Draw/Away."""
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    class_errors = []
    for class_index in range(3):
        predicted = probabilities[:, class_index]
        observed = (outcomes == class_index).astype(float)
        class_error = 0.0
        for bin_index in range(bins):
            lower = bin_edges[bin_index]
            upper = bin_edges[bin_index + 1]
            if bin_index == bins - 1:
                mask = (predicted >= lower) & (predicted <= upper)
            else:
                mask = (predicted >= lower) & (predicted < upper)
            if mask.any():
                class_error += mask.mean() * abs(
                    predicted[mask].mean() - observed[mask].mean()
                )
        class_errors.append(class_error)
    return float(np.mean(class_errors))


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate overall, per-class, and confusion-matrix metrics."""
    summary_rows = []
    class_rows = []
    confusion_rows = []

    # Only evaluate models that actually have predictions
    available_models = predictions["model"].unique()
    models_to_evaluate = [m for m in MODEL_ORDER if m in available_models]

    for model in models_to_evaluate:
        model_data = predictions.loc[predictions["model"] == model]
        
        # Skip if no data for this model
        if len(model_data) == 0:
            continue
            
        y_true = model_data["result"].map(TARGET_MAPPING).to_numpy()
        y_pred = model_data["predicted_result"].map(TARGET_MAPPING).to_numpy()
        probabilities = model_data[["prob_home", "prob_draw", "prob_away"]].to_numpy()

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        summary_rows.append(
            {
                "model": model,
                "track": MODEL_TRACKS[model],
                "accuracy": accuracy_score(y_true, y_pred),
                "log_loss": log_loss(y_true, probabilities, labels=[0, 1, 2]),
                "brier_score": pb.metrics.multiclass_brier_score(
                    probabilities, y_true
                ),
                "rps": pb.metrics.rps_average(probabilities, y_true),
                "mean_ece": expected_calibration_error(probabilities, y_true),
                "draw_predictions": int((y_pred == TARGET_MAPPING["D"]).sum()),
                "draw_recall": recall[TARGET_MAPPING["D"]],
            }
        )

        predicted_counts = np.bincount(y_pred, minlength=3)
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_rows.append(
                {
                    "model": model,
                    "track": MODEL_TRACKS[model],
                    "class": class_name,
                    "precision": precision[class_index],
                    "recall": recall[class_index],
                    "f1": f1[class_index],
                    "support": int(support[class_index]),
                    "predicted_count": int(predicted_counts[class_index]),
                }
            )
            for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                confusion_rows.append(
                    {
                        "model": model,
                        "actual_class": class_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[class_index, predicted_index]),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values("rps", ignore_index=True)
    return summary, pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def plot_calibration(predictions: pd.DataFrame) -> None:
    """Plot one-vs-rest calibration curves for every model and outcome."""
    sns.set_theme(style="whitegrid", context="notebook")
    
    # Only plot models that have predictions
    available_models = predictions["model"].unique()
    models_to_plot = [m for m in MODEL_ORDER if m in available_models]
    
    colors = dict(zip(MODEL_ORDER, sns.color_palette("tab10", len(MODEL_ORDER))))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)

    for class_index, (class_code, class_name) in enumerate(
        zip(CLASS_CODES, CLASS_NAMES)
    ):
        axis = axes[class_index]
        for model in models_to_plot:
            model_data = predictions.loc[predictions["model"] == model]
            
            # Skip if no data
            if len(model_data) == 0:
                continue
                
            y_binary = (model_data["result"] == class_code).astype(int)
            predicted = model_data[
                ["prob_home", "prob_draw", "prob_away"][class_index]
            ]
            observed_rate, predicted_mean = calibration_curve(
                y_binary,
                predicted,
                n_bins=8,
                strategy="quantile",
            )
            line_width = 2.8 if model == "bookmaker_avg_odds" else 1.5
            axis.plot(
                predicted_mean,
                observed_rate,
                marker="o",
                markersize=3.5,
                linewidth=line_width,
                color=colors[model],
                label=model,
            )
        axis.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1)
        axis.set(
            title=class_name,
            xlabel="Predicted probability",
            ylabel="Actual frequency" if class_index == 0 else "",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        sns.despine(ax=axis)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Calibration Check: EPL 2025/26 Holdout", fontsize=15)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(FIGURES_DIR / "calibration_all_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(confusion: pd.DataFrame) -> None:
    """Plot raw confusion matrices for all models and the odds benchmark."""
    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for axis, model in zip(axes.flat, MODEL_ORDER):
        model_data = confusion.loc[confusion["model"] == model]
        matrix = model_data.pivot(
            index="actual_class", columns="predicted_class", values="count"
        ).reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=axis,
        )
        axis.set(
            title=model,
            xlabel="Predicted",
            ylabel="Actual",
        )
    fig.suptitle("Confusion Matrices: EPL 2025/26 Holdout", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES_DIR / "confusion_matrices_all_models.png", dpi=180)
    plt.close(fig)


def validate_outputs(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    class_metrics: pd.DataFrame,
) -> None:
    """Validate complete model coverage and probability invariants."""
    errors = []
    
    # Dynamic model count (only models actually present)
    actual_models = predictions["model"].unique()
    expected_rows = len(base) * len(actual_models)
    
    if len(predictions) != expected_rows:
        errors.append(f"prediksi {len(predictions)}, seharusnya {expected_rows}")
    if predictions.duplicated(["match_id", "model"]).any():
        errors.append("kombinasi match_id/model duplikat")
    if predictions.isna().any().any():
        errors.append("missing value pada tabel prediksi")
    probability_sum = predictions[["prob_home", "prob_draw", "prob_away"]].sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-7):
        errors.append("probabilitas tidak berjumlah satu")
    if set(summary["model"]) != set(actual_models):
        errors.append(f"summary tidak mencakup semua model (expected {len(actual_models)}, got {len(summary)})")
    if len(class_metrics) != len(actual_models) * 3:
        errors.append(f"class metrics tidak lengkap (expected {len(actual_models)*3}, got {len(class_metrics)})")
    if not summary[["accuracy", "brier_score", "rps", "mean_ece"]].ge(0).all().all():
        errors.append("metrik negatif ditemukan")

    if errors:
        raise ValueError("; ".join(errors))


def check_training_consistency() -> None:
    """Warn if models were trained on different data periods (tolerant to missing metadata)."""
    import json
    
    metadata_files = {
        "ml_models": MODELS_DIR / "best_ml_model_metadata.json",
        "statistical_models": MODELS_DIR / "statistical_models_metadata.json",
        "corner_model": MODELS_DIR / "corner_model_metadata.json",
        "discipline_model": MODELS_DIR / "discipline_model_metadata.json",
    }
    
    training_periods = {}
    missing_metadata = []
    
    for model_name, meta_path in metadata_files.items():
        if meta_path.exists():
            try:
                meta = json.load(open(meta_path))
                training_periods[model_name] = {
                    "matches": meta.get("training_matches"),
                    "seasons": meta.get("training_seasons", []),
                    "period": (meta.get("training_period_start"), meta.get("training_period_end")),
                    "timestamp": meta.get("timestamp"),
                }
            except Exception:
                missing_metadata.append(f"{model_name} (corrupt metadata)")
        else:
            missing_metadata.append(f"{model_name} (file not found)")
    
    # Tolerant: skip check if metadata not available
    if missing_metadata:
        print("\n" + "="*80)
        print("Training consistency check: metadata not available for:")
        for item in missing_metadata:
            print(f"  - {item}")
        print("Skipping consistency check (will be available after next training run)")
        print("="*80 + "\n")
        return
    
    # Check if all models have same training period
    if not training_periods:
        return
    
    unique_periods = set(tp["period"] for tp in training_periods.values())
    unique_matches = set(tp["matches"] for tp in training_periods.values())
    
    if len(unique_periods) > 1 or len(unique_matches) > 1:
        print("\n" + "="*80)
        print("WARNING: Models trained on DIFFERENT data periods")
        print("="*80)
        for model, info in sorted(training_periods.items()):
            seasons_str = f"{info['period'][0]} to {info['period'][1]}" if info['period'][0] else "N/A"
            timestamp_str = info['timestamp'][:19] if info['timestamp'] else "N/A"
            print(f"  {model:20s}: {seasons_str:20s} | {info['matches']:4d} matches | trained: {timestamp_str}")
        print("\nComparison may not be fair. Consider re-running full cascade to ensure")
        print("all models trained on the same data period.")
        print("="*80 + "\n")
    else:
        # All consistent - optionally print confirmation
        sample = list(training_periods.values())[0]
        print(f"\nTraining consistency check: All models trained on same period")
        print(f"  Period: {sample['period'][0]} to {sample['period'][1]} ({sample['matches']} matches)\n")


def main() -> None:
    """Run the complete Phase 7 evaluation."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check training consistency before evaluation
    check_training_consistency()
    
    base = base_test_matches()
    predictions = standardize_predictions(base)
    summary, class_metrics, confusion = evaluate_predictions(predictions)
    validate_outputs(base, predictions, summary, class_metrics)

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    class_metrics.to_csv(CLASS_METRICS_PATH, index=False)
    confusion.to_csv(CONFUSION_PATH, index=False)
    plot_calibration(predictions)
    plot_confusion_matrices(confusion)

    benchmark = summary.loc[summary["model"] == "bookmaker_avg_odds"].iloc[0]
    actual_models = summary.loc[summary["track"] != "Benchmark"]
    best_model = actual_models.sort_values(["rps", "log_loss"]).iloc[0]
    worst_draw = actual_models.sort_values(
        ["draw_recall", "draw_predictions", "log_loss"],
        ascending=[True, True, False],
    ).iloc[0]
    beat_rps = actual_models.loc[actual_models["rps"] < benchmark["rps"], "model"].tolist()
    beat_log_loss = actual_models.loc[
        actual_models["log_loss"] < benchmark["log_loss"], "model"
    ].tolist()

    print("Evaluasi semua model:")
    print(
        summary[
            [
                "model",
                "accuracy",
                "log_loss",
                "brier_score",
                "rps",
                "mean_ece",
                "draw_recall",
                "draw_predictions",
            ]
        ].to_string(index=False)
    )
    print(f"\nModel terbaik (RPS, lalu log loss): {best_model['model']}")
    print(
        f"Benchmark odds: RPS {benchmark['rps']:.6f}, "
        f"log loss {benchmark['log_loss']:.6f}"
    )
    print(f"Mengalahkan odds pada RPS: {beat_rps or 'tidak ada'}")
    print(f"Mengalahkan odds pada log loss: {beat_log_loss or 'tidak ada'}")
    print(
        f"Draw terburuk: {worst_draw['model']} "
        f"(recall {worst_draw['draw_recall']:.3f}, "
        f"prediksi Draw {int(worst_draw['draw_predictions'])})"
    )
    print(f"Calibration plot: {FIGURES_DIR / 'calibration_all_models.png'}")
    print("Validasi output: lolos")


if __name__ == "__main__":
    main()
