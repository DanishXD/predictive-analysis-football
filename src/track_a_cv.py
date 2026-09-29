"""Walk-forward cross-validation untuk Track A (Poisson / Dixon-Coles / Elo).

Kenapa modul ini ada
-------------------
Angka Track A di ``data/processed/cv_model_selection.csv`` selama ini
dihitung di dalam sample: ``evaluate._track_a_selection_metrics()`` memuat
model yang di-fit di data training, lalu menilainya di data training yang
sama. Angka Track B di tabel yang sama berasal dari TimeSeriesSplit
out-of-fold, jadi satu kolom ``cv_log_loss_mean`` mencampur dua basis yang
berbeda. Label ``selection_basis`` sudah jujur soal ini, tapi tabelnya tetap
menyesatkan kalau dibaca sebagai ranking CV tunggal.

Modul ini menghasilkan angka Track A yang benar-benar out-of-sample memakai
walk-forward musiman (expanding window), supaya bisa dibandingkan langsung
dengan angka Track B.

Aturan anti-leakage
-------------------
- Hanya season training yang boleh jadi validation. ``TEST_SEASON`` tidak
  pernah jadi fold, dan fungsi ``validate_folds`` gagal keras kalau ada
  yang memaksanya masuk.
- Training sebuah fold hanya berisi season sebelum season validasi, dan
  batas waktunya diperiksa eksplisit (``train.date.max() < test.date.min()``).
- Bobot time-decay dihitung dengan ``base_date`` di akhir data training fold
  itu, bukan di akhir dataset, supaya bobot tidak melihat masa depan.

Batasan penting
----------------
Modul ini murni eksperimen. TIDAK menulis model produksi, TIDAK menyentuh
``evaluate.py`` atau ``predict_match.py``, dan tidak mengubah
``cv_model_selection.csv``. Angka di sini belum disambungkan ke pemilihan
model sampai direview manual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.metrics import accuracy_score, log_loss

from config import (
    DATA_DIR,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    MAX_GOALS,
    PROCESSED_DIR,
    SEASONS,
    TARGET_MAPPING,
    TEST_SEASON,
    TIME_DECAY_XI,
)
from statistical_models import predict_fixture

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
RESULTS_PATH = PROCESSED_DIR / "track_a_cv_results.csv"
SUMMARY_PATH = PROCESSED_DIR / "track_a_cv_summary.csv"
COMPARISON_PATH = PROCESSED_DIR / "track_a_cv_comparison.csv"
METADATA_PATH = DATA_DIR / "metadata" / "track_a_cv_summary.json"

# Minimal empat musim sebelum fold pertama supaya model punya cukup history
# untuk attack/defense tiap tim, dan supaya tim promosi di season validasi
# sudah punya profil goal dari sekurang-kurangnya satu musim.
MIN_TRAIN_SEASONS = 4


@dataclass(frozen=True)
class Fold:
    """Satu pasangan (season training, season validasi)."""

    number: int
    last_train_season: str
    validation_season: str


def build_folds() -> list[Fold]:
    """Bangun expanding-window folds hanya dari season training.

    Season validasi diambil berurutan dari season training ke-5 sampai
    season training terakhir. Season terakhir dalam SEASONS adalah
    TEST_SEASON dan karena itu tidak pernah dipakai sebagai validasi.
    """
    train_seasons = [season for season in SEASONS if season != TEST_SEASON]
    if len(train_seasons) <= MIN_TRAIN_SEASONS:
        raise ValueError(
            f"Season training terlalu sedikit ({len(train_seasons)}) untuk "
            f"MIN_TRAIN_SEASONS={MIN_TRAIN_SEASONS}"
        )
    folds = []
    for index in range(MIN_TRAIN_SEASONS, len(train_seasons)):
        folds.append(
            Fold(
                number=len(folds) + 1,
                last_train_season=train_seasons[index - 1],
                validation_season=train_seasons[index],
            )
        )
    return folds


def validate_folds(folds: list[Fold]) -> None:
    """Gagal keras kalau ada fold yang melanggar aturan anti-leakage."""
    seen_validation: set[str] = set()
    for fold in folds:
        if fold.last_train_season not in SEASONS or fold.validation_season not in SEASONS:
            raise ValueError(f"Fold {fold.number}: season tidak dikenal di config.SEASONS")
        if fold.last_train_season >= fold.validation_season:
            raise ValueError(
                f"Fold {fold.number}: training season {fold.last_train_season} "
                f"tidak boleh >= validation season {fold.validation_season}"
            )
        if fold.validation_season == TEST_SEASON:
            raise ValueError(
                f"Fold {fold.number}: TEST_SEASON tidak boleh dipakai sebagai "
                "validation di modul ini. Semua angka penentu harus dari "
                "training/CV, bukan test set."
            )
        if SEASONS.index(fold.last_train_season) < MIN_TRAIN_SEASONS - 1:
            raise ValueError(
                f"Fold {fold.number}: training terlalu pendek, di bawah "
                f"MIN_TRAIN_SEASONS={MIN_TRAIN_SEASONS}"
            )
        if fold.validation_season in seen_validation:
            raise ValueError(
                f"Fold {fold.number}: season {fold.validation_season} dipakai "
                "lebih dari sekali sebagai validation"
            )
        seen_validation.add(fold.validation_season)


def fit_fold_models(train: pd.DataFrame):
    """Fit Poisson dan Dixon-Coles dengan time-decay dihitung di dalam fold.

    ``base_date`` sengaja memakai tanggal akhir data training fold, bukan
    akhir dataset, supaya bobot time-decay tidak merujuk masa depan.
    """
    weights = pb.models.dixon_coles_weights(
        train["date"],
        xi=TIME_DECAY_XI,
        base_date=train["date"].max(),
    )
    inputs = (
        train["goals_home"],
        train["goals_away"],
        train["team_home"],
        train["team_away"],
    )
    poisson = pb.models.PoissonGoalsModel(*inputs, weights=weights)
    poisson.fit()
    dixon_coles = pb.models.DixonColesGoalModel(*inputs, weights=weights)
    dixon_coles.fit()
    return poisson, dixon_coles


def score_fold(model, validation: pd.DataFrame, dixon_coles: bool) -> np.ndarray:
    """Hasilkan matriks probabilitas 1X2 out-of-sample untuk satu fold."""
    rows = []
    for match in validation.sort_values(["date", "match_id"]).itertuples(index=False):
        grid, cold_start = predict_fixture(
            model, match.team_home, match.team_away, dixon_coles=dixon_coles
        )
        home, draw, away = grid.home_draw_away
        rows.append(
            {
                "prob_home": float(home),
                "prob_draw": float(draw),
                "prob_away": float(away),
                "cold_start": bool(cold_start),
            }
        )
    return pd.DataFrame(rows)


def elo_probabilities_for(
    train: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    """Rekam probabilitas Elo pre-match untuk season validasi.

    Semua tim mulai dari ``ELO_DEFAULT_RATING`` (1500) lalu rating di-update
    kronologis.     Pertandingan validasi diproses satu per satu: probabilitas diambil dari
    rating SEBELUM pertandingan itu, baru rating di-update dengan
    hasilnya. Jadi tidak ada kebocoran dari pertandingan yang belum terjadi.

    Bedanya dengan Elo produksi: tidak ada cold-start ClubElo untuk tim
    promosi, jadi tim promosi mulai dari 1500. Itu keputusan sadar supaya
    modul ini deterministik dan tidak memanggil API luar yang bisa
    rate-limit antar-run (jebakan 6.6 di HANDOFF), dan supaya antar-fold
    benar-benar apples-to-apples.
    """
    elo = pb.ratings.Elo(k=ELO_K, home_field_advantage=ELO_HOME_ADVANTAGE)
    result_codes = {"H": 0, "D": 1, "A": 2}

    ordered_train = train.sort_values(["date", "match_id"])
    for match in ordered_train.itertuples(index=False):
        elo.update_ratings(
            match.team_home, match.team_away, result_codes[match.result]
        )

    rows = []
    for match in validation.sort_values(["date", "match_id"]).itertuples(index=False):
        probabilities = elo.calculate_match_probabilities(
            match.team_home, match.team_away
        )
        rows.append(
            {
                "prob_home": float(probabilities["home_win"]),
                "prob_draw": float(probabilities["draw"]),
                "prob_away": float(probabilities["away_win"]),
                "cold_start": False,
            }
        )
        elo.update_ratings(
            match.team_home, match.team_away, result_codes[match.result]
        )
    return pd.DataFrame(rows)


def evaluate_fold(
    fold: Fold, matches: pd.DataFrame, y_true: np.ndarray, validation: pd.DataFrame
) -> list[dict]:
    """Nilai semua model Track A pada satu fold out-of-sample."""
    train_seasons = [season for season in SEASONS if season <= fold.last_train_season]
    train = matches.loc[matches["season"].isin(train_seasons)].copy()

    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.number}: training atau validation kosong")
    if train["date"].max() >= validation["date"].min():
        raise ValueError(
            f"Fold {fold.number}: overlap waktu antara training "
            f"({train['date'].max().date()}) dan validation "
            f"({validation['date'].min().date()})"
        )
    if (train["season"] == TEST_SEASON).any():
        raise ValueError(f"Fold {fold.number}: TEST_SEASON bocor ke training")

    poisson, dixon_coles = fit_fold_models(train)

    rows = []
    for name, model, dixon_coles_flag in (
        ("poisson", poisson, False),
        ("dixon_coles", dixon_coles, True),
    ):
        probabilities = score_fold(model, validation, dixon_coles_flag)
        rows.append(_metrics_row(fold, name, probabilities, y_true))

    elo_frame = elo_probabilities_for(train, validation)
    rows.append(_metrics_row(fold, "elo", elo_frame, y_true))
    return rows


def _metrics_row(
    fold: Fold, model_name: str, probabilities: pd.DataFrame, y_true: np.ndarray
) -> dict:
    matrix = probabilities[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    predicted = matrix.argmax(axis=1)
    draw_codes = predicted == TARGET_MAPPING["D"]
    true_codes = y_true == TARGET_MAPPING["D"]
    return {
        "fold": fold.number,
        "last_train_season": fold.last_train_season,
        "validation_season": fold.validation_season,
        "model": model_name,
        "n_matches": len(matrix),
        "log_loss": float(log_loss(y_true, matrix, labels=[0, 1, 2])),
        "rps": float(pb.metrics.rps_average(matrix, y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "draw_predictions": int(draw_codes.sum()),
        "draw_recall": float((draw_codes & true_codes).sum() / max(true_codes.sum(), 1)),
        "cold_start_matches": int(probabilities["cold_start"].sum()),
        "selection_basis": "walk-forward musiman, season training saja",
    }


def run_walk_forward() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan seluruh fold dan kembalikan (hasil per fold, ringkasan)."""
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["datetime", "date"])
    folds = build_folds()
    validate_folds(folds)

    all_rows: list[dict] = []
    for fold in folds:
        validation = matches.loc[matches["season"] == fold.validation_season].copy()
        y_true = validation["result"].map(TARGET_MAPPING).to_numpy()
        train_count = int(
            (matches["season"] <= fold.last_train_season).sum()
        )
        print(
            f"Fold {fold.number}/{len(folds)} | validasi: {fold.validation_season} | "
            f"train: {train_count} match"
        )
        all_rows.extend(evaluate_fold(fold, matches, y_true, validation))

    results = pd.DataFrame(all_rows)
    summary = (
        results.groupby("model")
        .agg(
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            rps_mean=("rps", "mean"),
            rps_std=("rps", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            draw_recall_mean=("draw_recall", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values("log_loss_mean")
        .reset_index()
    )
    return results, summary


def compare_models(results: pd.DataFrame, baseline: str = "poisson") -> pd.DataFrame:
    """Bandingkan tiap model dengan baseline, paired per fold.

    Selisih dihitung dalam satu fold yang sama supaya perbandingan tidak
    dipengaruhi variasi musim. Nilai negatif berarti model yang diuji lebih
    baik dari baseline.
    """
    rows = []
    base = results.loc[results["model"] == baseline].set_index("fold")
    for model_name, group in results.groupby("model"):
        if model_name == baseline:
            continue
        other = group.set_index("fold")
        shared = base.index.intersection(other.index)
        delta_log_loss = other.loc[shared, "log_loss"] - base.loc[shared, "log_loss"]
        delta_rps = other.loc[shared, "rps"] - base.loc[shared, "rps"]
        rows.append(
            {
                "model": model_name,
                "baseline": baseline,
                "n_folds": len(shared),
                "delta_log_loss_mean": float(delta_log_loss.mean()),
                "delta_log_loss_std": float(delta_log_loss.std()),
                "folds_better": int((delta_log_loss < 0).sum()),
                "delta_rps_mean": float(delta_rps.mean()),
                "delta_rps_std": float(delta_rps.std()),
                "rps_folds_better": int((delta_rps < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Cetak ringkasan walk-forward Track A ke konsol."""
    width = 78
    print("\n" + "=" * width)
    print("WALK-FORWARD CV — TRACK A (season training saja)")
    print("=" * width)
    print(
        f"{'Model':<14}{'LogLoss':>11}{'RPS':>10}{'Acc':>9}{'DrawRec':>9}{'Fold':>6}"
    )
    print("-" * width)
    for row in summary.itertuples(index=False):
        print(
            f"{row.model:<14}{row.log_loss_mean:>11.6f}{row.rps_mean:>10.6f}"
            f"{row.accuracy_mean:>9.4f}{row.draw_recall_mean:>9.4f}{row.n_folds:>6.0f}"
        )
    print("=" * width)
    print("LogLoss & RPS: mean out-of-sample per model, lebih kecil lebih baik.")

    print("\nRincian per fold:")
    print(
        results[["fold", "validation_season", "model", "log_loss", "rps", "accuracy"]]
        .to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )

    print("\nPerbandingan paired per fold (terhadap poisson):")
    comparison = compare_models(results)
    print(
        comparison[
            [
                "model",
                "n_folds",
                "delta_log_loss_mean",
                "folds_better",
                "delta_rps_mean",
                "rps_folds_better",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:+.6f}")
    )

    print(
        "\nCatatan: angka ini OUT-OF-SAMPLE di season training, bukan test "
        "season.\nTEST_SEASON tidak pernah dipakai sebagai fold, jadi "
        "seleksi model tidak\ntercemar test set. Model produksi dan "
        "cv_model_selection.csv TIDAK diubah\nmodul ini."
    )
    return comparison


def main() -> None:
    print("=" * 78)
    print("Walk-forward CV Track A (Poisson / Dixon-Coles / Elo)")
    print("=" * 78)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    results, summary = run_walk_forward()
    results.to_csv(RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    comparison = print_summary(results, summary)
    comparison.to_csv(COMPARISON_PATH, index=False)

    payload = {
        "purpose": (
            "Angka Track A out-of-sample. Eksperimen: evaluate.py dan "
            "predict_match.py sengaja tidak diubah sampai angka ini direview."
        ),
        "cv_strategy": (
            f"Walk-forward musiman (expanding window), minimal {MIN_TRAIN_SEASONS} "
            "season training per fold, time-decay per fold"
        ),
        "test_season_excluded": True,
        "test_season": TEST_SEASON,
        "max_goals": MAX_GOALS,
        "time_decay_xi": TIME_DECAY_XI,
        "folds": [
            {
                "fold": fold.number,
                "last_train_season": fold.last_train_season,
                "validation_season": fold.validation_season,
            }
            for fold in build_folds()
        ],
        "summary": summary.to_dict("records"),
        "comparison_vs_poisson": compare_models(results).to_dict("records"),
    }
    METADATA_PATH.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nTersimpan: {RESULTS_PATH}")
    print(f"Tersimpan: {SUMMARY_PATH}")
    print(f"Tersimpan: {COMPARISON_PATH}")
    print(f"Tersimpan: {METADATA_PATH}")


if __name__ == "__main__":
    main()
