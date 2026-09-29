"""Evaluate every statistical and ML model against decoded bookmaker odds."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import penaltyblog as pb
import seaborn as sns
import config
from config import (
    BOOTSTRAP_SAMPLES,
    MODELS_DIR,
    N_SPLITS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET_MAPPING,
    TEST_SEASON,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
STATISTICAL_PATH = PROCESSED_DIR / "test_match_probabilities.csv"
ELO_PATH = PROCESSED_DIR / "elo_history.csv"
ML_PATH = PROCESSED_DIR / "ml_test_predictions.csv"
STACKING_PATH = PROCESSED_DIR / "stacking_test_predictions.csv"
ML_METRICS_PATH = PROCESSED_DIR / "ml_model_metrics.csv"
STATISTICAL_METRICS_PATH = PROCESSED_DIR / "statistical_model_metrics.csv"
POISSON_MODEL_PATH = MODELS_DIR / "poisson_goal_model.pkl"
DIXON_COLES_MODEL_PATH = MODELS_DIR / "dixon_coles_goal_model.pkl"

# Tabel model selection. Berisi metrik LATIHAN (bukan test) supaya pemilihan
# model tidak tercemar performa test set. predict_match.py membaca file ini.
CV_SELECTION_PATH = PROCESSED_DIR / "cv_model_selection.csv"

SUMMARY_PATH = PROCESSED_DIR / "evaluation_summary.csv"
CLASS_METRICS_PATH = PROCESSED_DIR / "evaluation_class_metrics.csv"
CONFUSION_PATH = PROCESSED_DIR / "evaluation_confusion_matrices.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "evaluation_predictions.csv"
BOOKMAKER_PATH = PROCESSED_DIR / "bookmaker_probabilities.csv"
# Bootstrap CI outputs — file baru, tidak menggantikan SUMMARY_PATH
CI_SUMMARY_PATH = PROCESSED_DIR / "evaluation_summary_with_ci.csv"
COMPARISON_PATH = PROCESSED_DIR / "model_comparison_bootstrap.csv"

CLASS_CODES = ["H", "D", "A"]
CLASS_NAMES = ["Home Win", "Draw", "Away Win"]
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


def _new_elo():
    """Buat objek Elo dengan parameter yang dibaca dari ``config`` saat dipanggil.

    Nilai diambil lewat ``config.X`` (bukan ``from config import X``) supaya
    monkeypatch di test benar-benar mengubah perilaku. Guard AST di
    ``tests/test_config.py`` hanya menangkap assignment module-level, jadi
    keyword argument yang ditulis literal sebelumnya lolos dari sana.
    """
    return pb.ratings.Elo(
        k=config.ELO_K, home_field_advantage=config.ELO_HOME_ADVANTAGE
    )


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
        elo = _new_elo()
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


def _rps_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Wrapper RPS untuk dipakai dalam bootstrap."""
    import penaltyblog as pb
    return pb.metrics.rps_average(probabilities, y_true)


def _brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Wrapper Brier untuk dipakai dalam bootstrap."""
    import penaltyblog as pb
    return pb.metrics.multiclass_brier_score(probabilities, y_true)


def bootstrap_metric_ci(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Hitung bootstrap 95% CI untuk RPS, log_loss, accuracy, dan brier.

    Resample dilakukan pada level match (bukan observation) — paired bootstrap
    sehingga setiap baris prediksi + label tetap pasangan yang sama.

    Returns dict: {metric_name: {"mean": ..., "ci_low": ..., "ci_high": ...}}
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    alpha = 1.0 - confidence

    boot_rps = np.empty(n_samples)
    boot_ll = np.empty(n_samples)
    boot_acc = np.empty(n_samples)
    boot_brier = np.empty(n_samples)

    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        p_boot = probabilities[idx]
        y_boot = y_true[idx]
        # normalize agar sum tetap 1 setelah resampling (seharusnya sudah, tapi jaga-jaga)
        p_boot = p_boot / p_boot.sum(axis=1, keepdims=True)
        boot_rps[i] = _rps_score(p_boot, y_boot)
        boot_ll[i] = log_loss(y_boot, p_boot, labels=[0, 1, 2])
        boot_acc[i] = accuracy_score(y_boot, p_boot.argmax(axis=1))
        boot_brier[i] = _brier_score(p_boot, y_boot)

    def _ci(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }

    return {
        "rps": _ci(boot_rps),
        "log_loss": _ci(boot_ll),
        "accuracy": _ci(boot_acc),
        "brier_score": _ci(boot_brier),
    }


def bootstrap_model_comparison(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_samples: int = BOOTSTRAP_SAMPLES,
    random_state: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test: apakah model_a lebih baik dari model_b pada RPS?

    Paired = resample match yang sama untuk kedua model secara bersamaan.
    Mengembalikan:
    - delta_rps_mean: mean(RPS_a - RPS_b) — negatif = model_a lebih baik
    - p_value: fraksi bootstrap samples di mana model_a >= model_b (H0: tidak ada beda)
    - ci_low, ci_high: 95% CI untuk delta RPS
    """
    import penaltyblog as pb

    match_ids = predictions["match_id"].unique()
    data_a = predictions.loc[predictions["model"] == model_a].set_index("match_id")
    data_b = predictions.loc[predictions["model"] == model_b].set_index("match_id")

    # Hanya match yang ada di kedua model
    common = sorted(set(data_a.index) & set(data_b.index))
    data_a = data_a.loc[common]
    data_b = data_b.loc[common]

    prob_a = data_a[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    prob_b = data_b[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    y_true = data_a["result"].map(TARGET_MAPPING).to_numpy()

    n = len(common)
    rng = np.random.default_rng(random_state)
    boot_delta = np.empty(n_samples)

    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        rps_a = pb.metrics.rps_average(prob_a[idx], y_true[idx])
        rps_b = pb.metrics.rps_average(prob_b[idx], y_true[idx])
        boot_delta[i] = rps_a - rps_b

    # Observed delta
    obs_delta = _rps_score(prob_a, y_true) - _rps_score(prob_b, y_true)
    # p-value: fraksi bootstrap delta >= 0 (H0: model_a tidak lebih baik)
    p_value = float((boot_delta >= 0).mean())

    return {
        "model_a": model_a,
        "model_b": model_b,
        "obs_rps_a": float(_rps_score(prob_a, y_true)),
        "obs_rps_b": float(_rps_score(prob_b, y_true)),
        "delta_rps_mean": float(boot_delta.mean()),
        "delta_rps_ci_low": float(np.percentile(boot_delta, 2.5)),
        "delta_rps_ci_high": float(np.percentile(boot_delta, 97.5)),
        "p_value_a_not_better": p_value,
        "significant_95": bool(p_value < 0.05),
        "n_matches": n,
        "n_bootstrap": n_samples,
    }


def run_bootstrap_evaluation(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate CI summary dan pairwise model comparison.

    Mengembalikan (ci_summary_df, comparison_df).
    ci_summary_df: summary + kolom CI tambahan
    comparison_df: pairwise bootstrap comparison
    """
    print(f"\nMenghitung bootstrap CIs ({BOOTSTRAP_SAMPLES} samples)...")
    ci_rows = []
    available_models = [m for m in MODEL_ORDER if m in summary["model"].values]

    for model in available_models:
        model_data = predictions.loc[predictions["model"] == model]
        y_true = model_data["result"].map(TARGET_MAPPING).to_numpy()
        probs = model_data[["prob_home", "prob_draw", "prob_away"]].to_numpy()

        print(f"  Bootstrap: {model}...", end=" ", flush=True)
        cis = bootstrap_metric_ci(probs, y_true)
        print("selesai")

        base_row = summary.loc[summary["model"] == model].iloc[0].to_dict()
        for metric, stats in cis.items():
            base_row[f"{metric}_ci_low"] = stats["ci_low"]
            base_row[f"{metric}_ci_high"] = stats["ci_high"]
        ci_rows.append(base_row)

    ci_summary = pd.DataFrame(ci_rows)

    # Pairwise comparison: semua Track B models vs bookmaker, dan antar sesama
    print("\nMenghitung paired bootstrap comparison...")
    comp_pairs = []
    track_b_models = ["logistic_regression", "random_forest", "xgboost"]
    actual_models = [m for m in track_b_models if m in summary["model"].values]

    # Setiap model vs bookmaker
    for model in actual_models:
        if "bookmaker_avg_odds" in summary["model"].values:
            result = bootstrap_model_comparison(predictions, model, "bookmaker_avg_odds")
            comp_pairs.append(result)
            print(
                f"  {model} vs bookmaker: delta={result['delta_rps_mean']:+.4f} "
                f"p={result['p_value_a_not_better']:.3f} "
                f"({'signifikan' if result['significant_95'] else 'tidak signifikan'})"
            )

    # RF vs LR vs XGB pairwise
    for i, m_a in enumerate(actual_models):
        for m_b in actual_models[i + 1:]:
            result = bootstrap_model_comparison(predictions, m_a, m_b)
            comp_pairs.append(result)
            print(
                f"  {m_a} vs {m_b}: delta={result['delta_rps_mean']:+.4f} "
                f"p={result['p_value_a_not_better']:.3f} "
                f"({'signifikan' if result['significant_95'] else 'tidak signifikan'})"
            )

    comparison = pd.DataFrame(comp_pairs)
    return ci_summary, comparison


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


def _train_rows() -> pd.DataFrame:
    """Baris training (semua season selain TEST_SEASON) untuk metrik Track A."""
    columns = ["match_id", "season", "datetime", "date", "team_home", "team_away", "result"]
    matches = pd.read_csv(MATCHES_PATH, usecols=columns, parse_dates=["datetime", "date"])
    return matches.loc[matches["season"] != TEST_SEASON].sort_values(
        ["datetime", "match_id"]
    ).reset_index(drop=True)


def _log_loss_and_accuracy(
    probabilities: np.ndarray, y_true: np.ndarray
) -> tuple[float, float]:
    return (
        float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        float(accuracy_score(y_true, probabilities.argmax(axis=1))),
    )


def _track_a_selection_metrics() -> pd.DataFrame:
    """Hitung log loss & accuracy Track A pada data TRAINING saja.

    Model Poisson/Dixon-Coles tidak punya cross-validation, jadi dipakai
    single-split di atas data training. Test metrics sengaja tidak dipakai:
    memakai test set untuk memilih model justru kontaminasi yang dihindari
    dalam model selection. Label selection_basis menyatakan basisnya dengan jujur.
    """
    train = _train_rows()
    y_true = train["result"].map(TARGET_MAPPING).to_numpy()
    rows = []

    goal_models = [
        ("poisson", POISSON_MODEL_PATH, pb.models.PoissonGoalsModel),
        ("dixon_coles", DIXON_COLES_MODEL_PATH, pb.models.DixonColesGoalModel),
    ]
    for name, path, model_class in goal_models:
        if not path.exists():
            rows.append(
                {
                    "model": name,
                    "track": "Track A",
                    "cv_log_loss_mean": np.nan,
                    "cv_log_loss_std": np.nan,
                    "cv_accuracy_mean": np.nan,
                    "cv_accuracy_std": np.nan,
                    "selection_basis": "tidak tersedia (model belum disimpan)",
                    "draw_recall": np.nan,
                }
            )
            continue
        model = model_class.load(str(path))
        probabilities = _score_grid_probabilities(model, train)
        loss, accuracy = _log_loss_and_accuracy(probabilities, y_true)
        rows.append(
            {
                "model": name,
                "track": "Track A",
                "cv_log_loss_mean": loss,
                "cv_log_loss_std": np.nan,
                "cv_accuracy_mean": accuracy,
                "cv_accuracy_std": np.nan,
                "selection_basis": "train-only single split (no CV)",
                "draw_recall": np.nan,
            }
        )

    # Elo: likewise train-only, rebuilt from the saved pre-match ratings.
    elo_path = ELO_PATH
    if elo_path.exists():
        elo_history = pd.read_csv(elo_path)
        train_seasons = sorted(train["season"].unique())
        home_rows = elo_history.loc[
            elo_history["season"].isin(train_seasons) & (elo_history["venue"] == "home"),
            ["match_id", "team", "opponent", "elo_pre", "opponent_elo_pre"],
        ].set_index("match_id")
        if not home_rows.empty and home_rows.index.isin(set(train["match_id"])).all():
            probabilities = np.array(
                [
                    _elo_row_probabilities(home_rows.loc[mid])
                    for mid in train["match_id"]
                ]
            )
            loss, accuracy = _log_loss_and_accuracy(probabilities, y_true)
            rows.append(
                {
                    "model": "elo",
                    "track": "Track A",
                    "cv_log_loss_mean": loss,
                    "cv_log_loss_std": np.nan,
                    "cv_accuracy_mean": accuracy,
                    "cv_accuracy_std": np.nan,
                    "selection_basis": "train-only single split (no CV)",
                    "draw_recall": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _score_grid_probabilities(model, train: pd.DataFrame) -> np.ndarray:
    """Konversi score grid 1X2 tiap match training menjadi vektor probabilitas."""
    rows = []
    for match in train.itertuples(index=False):
        grid = model.predict(
            match.team_home, match.team_away, max_goals=config.MAX_GOALS
        )
        home, draw, away = grid.home_draw_away
        rows.append([float(home), float(draw), float(away)])
    return np.array(rows)


def _elo_row_probabilities(row) -> list[float]:
    elo = _new_elo()
    elo.ratings = {
        row["team"]: float(row["elo_pre"]),
        row["opponent"]: float(row["opponent_elo_pre"]),
    }
    probabilities = elo.calculate_match_probabilities(row["team"], row["opponent"])
    return [
        probabilities["home_win"],
        probabilities["draw"],
        probabilities["away_win"],
    ]


def export_cv_model_selection(draw_recall_by_model: dict[str, float]) -> pd.DataFrame:
    """Tulis cv_model_selection.csv: metrik selection berbasis data training.

    Track B memakai CV metrics asli dari Fase 5. Track A memakai single split
    train-only karena modelnya tidak punya CV. Test metrics tidak ikut supaya
    pemilihan model tidak tercemar test set. Dipanggil setiap run evaluate.py
    supaya file ini tidak pernah basi.
    """
    rows = []

    if ML_METRICS_PATH.exists():
        ml_metrics = pd.read_csv(ML_METRICS_PATH)
        for row in ml_metrics.itertuples(index=False):
            rows.append(
                {
                    "model": row.model,
                    "track": "Track B",
                    "cv_log_loss_mean": row.cv_log_loss_mean,
                    "cv_log_loss_std": row.cv_log_loss_std,
                    "cv_accuracy_mean": row.cv_accuracy_mean,
                    "cv_accuracy_std": row.cv_accuracy_std,
                    "selection_basis": f"TimeSeriesSplit {N_SPLITS}-fold CV",
                    "draw_recall": draw_recall_by_model.get(row.model, np.nan),
                }
            )
    else:
        print(f"PERINGATAN: {ML_METRICS_PATH.name} tidak ada, Track A saja")

    for row in _track_a_selection_metrics().to_dict("records"):
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(CV_SELECTION_PATH, index=False)
    return table


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

    # Bootstrap CIs — file terpisah, tidak menggantikan evaluation_summary.csv
    ci_summary, comparison = run_bootstrap_evaluation(predictions, summary)
    ci_summary.to_csv(CI_SUMMARY_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    print(f"\nBootstrap CI disimpan ke: {CI_SUMMARY_PATH}")
    print(f"Model comparison disimpan ke: {COMPARISON_PATH}")

    # Model selection table — ditulis tiap run supaya tidak pernah basi.
    # draw_recall diambil dari summary (test metrics) karena hanya dipakai untuk
    # peringatan di CLI, BUKAN untuk memilih model.
    draw_recall_by_model = dict(
        zip(summary["model"], summary["draw_recall"])
    )
    cv_table = export_cv_model_selection(draw_recall_by_model)
    print(f"\nModel selection table disimpan ke: {CV_SELECTION_PATH}")
    print(
        cv_table[["model", "track", "cv_log_loss_mean", "cv_accuracy_mean", "selection_basis"]]
        .to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )

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
