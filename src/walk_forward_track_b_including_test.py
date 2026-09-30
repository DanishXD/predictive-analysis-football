"""Walk-forward (expanding window) validation Track B — 6 fold, INCLUDING test season.

Module ini evaluating robustness model KLASIFIKASI Track B (Logistic
Regression, Random Forest, XGBoost) di 6 periode out-of-sample yang
independen. Setiap fold menambah satu musim ke training set (expanding
window).

  Fold design:
    Fold 1: Train 2016-17 -> 2019-20, Validasi 2020-21
    Fold 2: Train 2016-17 -> 2020-21, Validasi 2021-22
    Fold 3: Train 2016-17 -> 2021-22, Validasi 2022-23
    Fold 4: Train 2016-17 -> 2022-23, Validasi 2023-24
    Fold 5: Train 2016-17 -> 2023-24, Validasi 2024-25
    Fold 6: Train 2016-17 -> 2024-25, Validasi 2025-26  <-- TEST_SEASON

NAMA FILE MENYERUPAI MODUL LAIN, BEDAKAN DENGAN SERIUS:
    src/track_a_cv.py
        5 fold, model Track A (Poisson/Dixon-Coles/Elo), season training
        saja. INI YANG DIPAKAI untuk model selection di evaluate.py.
    modul ini
        6 fold, model Track B, dan fold 6 memakai season yang sama dengan
        TEST_SEASON. Hanya untuk PELAPORAN variasi antar musim, bukan untuk
        memilih model: memakai fold 6 sama dengan melihat test set.
        Angka fold 1-5 pun tidak sebanding langsung dengan TimeSeriesSplit
        5-fold di Track B, karena training window dan batasannya beda.

Output:
  data/processed/walk_forward_track_b_including_test_results.csv
  data/processed/walk_forward_track_b_including_test_summary.csv

Dulu modul ini bernama ``walk_forward.py`` dan hanya berisi fungsi
``run_walk_forward()`` di ``track_a_cv.py`` dan ``exposure_poisson.py``.
Nama lamanya tidak membedakan keduanya, padahal perilakunya berlawanan soal
test season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import brier_score_loss

from config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    SEASONS,
    TARGET_MAPPING,
    TARGET_NAMES,
)
from train_model import build_models, load_modeling_data

RESULTS_PATH = PROCESSED_DIR / "walk_forward_track_b_including_test_results.csv"
SUMMARY_PATH = PROCESSED_DIR / "walk_forward_track_b_including_test_summary.csv"

# 6 fold; fold terakhir sengaja memakai TEST_SEASON, jadi output modul ini
# hanya untuk pelaporan dan tidak boleh dipakai untuk model selection.
# Bandingkan dengan WALK_FORWARD_FOLDS di src/track_a_cv.py (5 fold,
# season training saja) yang justru dipakai evaluate.py.
WALK_FORWARD_FOLDS = [
    # (last_train_season, test_season)
    ("2019-2020", "2020-2021"),
    ("2020-2021", "2021-2022"),
    ("2021-2022", "2022-2023"),
    ("2022-2023", "2023-2024"),
    ("2023-2024", "2024-2025"),
    ("2024-2025", "2025-2026"),
]


def ranked_probability_score(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute mean RPS untuk klasifikasi 3-kelas (H/D/A).

    RPS = rata-rata sum dari (CDF prediksi - CDF aktual)^2.
    Lebih rendah = lebih baik.
    """
    n_classes = 3
    total = 0.0
    for prob_row, target in zip(probabilities, targets):
        # CDF prediksi dan aktual (kumulatif dari kiri)
        pred_cdf = np.cumsum(prob_row)[: n_classes - 1]
        actual_onehot = np.zeros(n_classes)
        actual_onehot[int(target)] = 1.0
        actual_cdf = np.cumsum(actual_onehot)[: n_classes - 1]
        total += np.sum((pred_cdf - actual_cdf) ** 2) / (n_classes - 1)
    return total / len(targets)


def run_walk_forward() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan 6-fold walk-forward validation dan kembalikan (results, summary)."""
    data, feature_columns = load_modeling_data()

    # Tambahkan kolom target numerik (sama seperti train_model.py)
    if "target" not in data.columns:
        data["target"] = data["result"].map(TARGET_MAPPING)

    models = build_models()
    all_rows = []

    for fold_num, (last_train_season, test_season) in enumerate(
        WALK_FORWARD_FOLDS, start=1
    ):
        # Validasi fold definition konsisten dengan SEASONS
        if last_train_season not in SEASONS:
            raise ValueError(
                f"Fold {fold_num}: last_train_season '{last_train_season}' "
                "tidak ada di config.SEASONS"
            )
        if test_season not in SEASONS:
            raise ValueError(
                f"Fold {fold_num}: test_season '{test_season}' "
                "tidak ada di config.SEASONS"
            )

        # Expanding window: semua season s/d last_train_season = train
        train_seasons = [s for s in SEASONS if s <= last_train_season]
        train = data.loc[data["season"].isin(train_seasons)].copy()
        test = data.loc[data["season"] == test_season].copy()

        # Temporal integrity check
        if len(train) == 0:
            raise ValueError(f"Fold {fold_num}: training set kosong")
        if len(test) == 0:
            raise ValueError(
                f"Fold {fold_num}: test set kosong untuk season {test_season}"
            )
        if train["date"].max() >= test["date"].min():
            raise ValueError(
                f"Fold {fold_num}: temporal overlap antara train "
                f"(max {train['date'].max().date()}) dan test "
                f"(min {test['date'].min().date()})"
            )

        X_train = train[feature_columns]
        y_train = train["target"]
        X_test = test[feature_columns]
        y_test = test["target"]

        print(
            f"Fold {fold_num}/6 | Test: {test_season} | "
            f"Train: {len(train)} matches ({len(train_seasons)} seasons) | "
            f"Test: {len(test)} matches"
        )

        for model_name, model_pipeline in models.items():
            fitted = clone(model_pipeline).fit(X_train, y_train)
            probabilities = fitted.predict_proba(X_test)
            predictions = fitted.predict(X_test)

            rps = ranked_probability_score(probabilities, y_test.to_numpy())
            ll = log_loss(y_test, probabilities, labels=[0, 1, 2])
            acc = accuracy_score(y_test, predictions)

            # Brier score: rata-rata atas 3 kelas (one-vs-rest)
            brier_total = 0.0
            for cls_idx in range(3):
                actual_binary = (y_test.to_numpy() == cls_idx).astype(float)
                brier_total += brier_score_loss(actual_binary, probabilities[:, cls_idx])
            brier = brier_total / 3.0

            all_rows.append(
                {
                    "fold": fold_num,
                    "last_train_season": last_train_season,
                    "test_season": test_season,
                    "train_matches": len(train),
                    "train_seasons": len(train_seasons),
                    "test_matches": len(test),
                    "model": model_name,
                    "rps": rps,
                    "log_loss": ll,
                    "accuracy": acc,
                    "brier_score": brier,
                }
            )
            print(
                f"  {model_name:<22} | RPS: {rps:.4f} | "
                f"log_loss: {ll:.4f} | acc: {acc:.4f}"
            )

    results = pd.DataFrame(all_rows)

    # Aggregate: mean ± std per model
    summary_rows = []
    for model_name, group in results.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "rps_mean": group["rps"].mean(),
                "rps_std": group["rps"].std(),
                "log_loss_mean": group["log_loss"].mean(),
                "log_loss_std": group["log_loss"].std(),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(),
                "brier_mean": group["brier_score"].mean(),
                "brier_std": group["brier_score"].std(),
                "n_folds": len(group),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("rps_mean").reset_index(drop=True)

    return results, summary


def print_summary(summary: pd.DataFrame) -> None:
    """Print aggregated walk-forward summary table ke konsol."""
    width = 78
    print("\n" + "=" * width)
    print("WALK-FORWARD VALIDATION — RINGKASAN (6 FOLDS)")
    print("=" * width)
    print(
        f"{'Model':<24} {'RPS mean':>9} {'RPS std':>8} "
        f"{'LogLoss mean':>13} {'LogLoss std':>12} {'Acc mean':>9}"
    )
    print("-" * width)
    for row in summary.itertuples(index=False):
        print(
            f"{row.model:<24} {row.rps_mean:>9.4f} {row.rps_std:>8.4f} "
            f"{row.log_loss_mean:>13.4f} {row.log_loss_std:>12.4f} "
            f"{row.accuracy_mean:>9.4f}"
        )
    print("=" * width)
    print(
        "Catatan: Lebih rendah lebih baik untuk RPS dan Log Loss. "
        "Hasil ini adalah out-of-sample\npada 6 musim independen "
        "(bukan test set yang dipakai untuk pelaporan Fase 7)."
    )
    print("=" * width)


def main() -> None:
    """Entry point: jalankan walk-forward dan simpan output."""
    print("Memulai walk-forward validation (6 expanding folds)...")
    print("Ini akan melatih ulang semua model per fold — harap tunggu.\n")

    results, summary = run_walk_forward()

    results.to_csv(RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nHasil per fold disimpan ke: {RESULTS_PATH}")
    print(f"Ringkasan disimpan ke: {SUMMARY_PATH}")

    print_summary(summary)


if __name__ == "__main__":
    main()
