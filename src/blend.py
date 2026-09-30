"""Blend sistematis seluruh model: meta-learner di atas OOF predictions.

Kenapa modul ini ada
--------------------
Fase 6 lama (``src/stacking.py``) hanya menumpuk expected goals dari Poisson
dan Dixon-Coles ke dalam Random Forest. Itu satu feature tambahan untuk satu
model, bukan blend. Modul ini collect probabilitas 1X2 dari SELURUH model
lalu membuat meta-learner di atasnya.

Model yang diblend (7 base model)
---------------------------------
``poisson``, ``dixon_coles``, ``elo``          (Track A, penaltyblog)
``logistic_regression``, ``random_forest``, ``xgboost``  (Track B, features.csv)
``exposure_sot``                              (eksperimen Stage 3, BUKAN xG)

Catatan penting soal dua di antaranya:

* ``exposure_sot`` berasal dari Stage 3. Namanya "exposure-based Poisson"
  dan itu BUKAN xG: tidak ada shot location atau shot quality, hanya volume
  dan efisiensi agregat per tim per match. Di blend ini ia diperlakukan
  sebagai base model biasa.
* Tidak ada base model "terkalibrasi" maupun RF "ter-tuning". Kalibrasi
  post-hoc adalah hasil negatif (HANDOFF 5.2) dan tuning RF juga tidak
  significant (5.6); Random Forest masih memakai setting Phase 5. Yang
  ter-tuning hanya XGBoost.

Disiplin anti-leakage
--------------------
Ini bagian yang paling mudah salah, jadi dijaga berlapis:

1. **OOF, bukan in-sample.** Meta-learner dilatih HANYA pada prediksi
   out-of-fold. Untuk setiap fold, semua base model di-fit ulang pada data
   fold-training, lalu memprediksi fold-validation. Prediksi pada data yang
   sama dengan training base model TIDAK PERNAH dipakai sebagai input
   meta-learner.
2. **Meta-learner juga di-CV.** Implementasi naif ("fit meta di OOF, skor di
   test") menyiratkan pemilihan model berdasarkan test. Di sini meta-learner
   dinilai dengan walk-forward sendiri di level kedua: untuk setiap fold,
   meta-learner dilatih pada OOF fold-fold sebelumnya dan dievaluasi pada
   fold berjalan. Jadi angka "apakah blend menang" berasal dari training.
3. **Test season hanya dibaca sekali**, di akhir, untuk pelaporan. Tidak
   pernah jadi fold, tidak pernah jadi sumber pemilihan.
4. Guard eksplisit: nilai OOF harus berasal dari fold yang tidak melihat baris
   itu, dan test season tidak boleh masuk ke training meta-learner sama sekali.

Batas eksperimen
----------------
Modul ini tidak menulis model produksi dan tidak menyentuh ``evaluate.py``,
``predict_match.py``, atau ``cv_model_selection.csv``. Angka di sini untuk
melihat apakah blend layak jadi kandidat produksi; kalau iya, itu keputusan
terpisah.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.special import gammaln
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import exposure_poisson
from config import (
    BOOTSTRAP_SAMPLES,
    DATA_DIR,
    EXPOSURE_XI_GRID,
    MAX_GOALS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_MAPPING,
    TEST_SEASON,
    TIME_DECAY_XI,
)
from statistical_models import fit_goal_models, predict_fixture
from track_a_cv import elo_probabilities_for
from exposure_poisson import ExposureModel, fit_exposure_model
from train_model import build_models, date_based_time_splits, load_modeling_data

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
BOOKMAKER_PATH = PROCESSED_DIR / "bookmaker_probabilities.csv"
OOF_PATH = PROCESSED_DIR / "blend_oof_predictions.csv"
TEST_PREDICTIONS_PATH = PROCESSED_DIR / "blend_test_predictions.csv"
FOLD_RESULTS_PATH = PROCESSED_DIR / "blend_fold_results.csv"
SUMMARY_PATH = PROCESSED_DIR / "blend_summary.csv"
GATEWAY_PATH = PROCESSED_DIR / "blend_gateways.csv"
COMPARISON_PATH = PROCESSED_DIR / "blend_comparison.csv"
GAP_PATH = PROCESSED_DIR / "blend_gap_to_bookmaker.csv"
METADATA_PATH = DATA_DIR / "metadata" / "blend_summary.json"

# Angka test season SEBELUM empat minggu perbaikan dimulai, dari capture
# pre-upgrade di research/evaluations/baseline.md. File itu milik sesi
# paralel jadi hanya dibaca, tidak ditulis. Dipakai HANYA untuk menjawab
# "berapa jarak ke bandar sekarang dibanding dulu", bukan untuk-training
# maupun pemilihan model.
PRE_IMPROVEMENT = {
    "bookmaker_avg_odds": {"log_loss": 1.015, "rps": 0.205},
    "best_model": {"name": "random_forest", "log_loss": 1.036, "rps": 0.211},
}

TRACK_A_MODELS = ("poisson", "dixon_coles")
TRACK_B_MODELS = ("logistic_regression", "random_forest", "xgboost")
BASE_MODELS = TRACK_A_MODELS + ("elo",) + TRACK_B_MODELS + ("exposure_sot",)

PROBABILITY_LABELS = ("home", "draw", "away")
BOOKMAKER_NAME = "bookmaker_avg_odds"
# xi terbaik untuk exposure_sot dari eksperimen Stage 3 (training season only).
EXPOSURE_XI = 0.0025


def feature_names() -> list[str]:
    """Nama kolom fitur meta-learner: 3 probabilitas per base model."""
    return [
        f"{model}_prob_{label}"
        for model in BASE_MODELS
        for label in PROBABILITY_LABELS
    ]


# --------------------------------------------------------------------------
# Prediksi base model untuk satu subset data
# --------------------------------------------------------------------------


def _poisson_pmf(rate: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    log_rates = (
        -rate + np.arange(max_goals) * np.log(rate) - gammaln(np.arange(max_goals) + 1)
    )
    return np.exp(log_rates)


def _grid_to_1x2(expected_home: float, expected_away: float) -> np.ndarray:
    joint = np.outer(_poisson_pmf(expected_home), _poisson_pmf(expected_away))
    index = np.arange(MAX_GOALS)[:, None]
    probabilities = np.array(
        [
            joint[index > index.T].sum(),
            joint[index == index.T].sum(),
            joint[index < index.T].sum(),
        ]
    )
    return probabilities / probabilities.sum()


def _cold_start_expected_goals(model, home: str, away: str, dixon_coles: bool):
    """Expected goals dengan fallback rata-rata liga, sama seperti predict_fixture."""
    parameters = model.params_array
    n_teams = model.n_teams
    attacks = parameters[:n_teams]
    defenses = parameters[n_teams : 2 * n_teams]
    known = set(model.teams)
    if home not in known or away not in known:
        home_attack = (
            float(attacks.mean()) if home not in known else float(attacks[model.team_to_idx[home]])
        )
        home_defense = (
            float(defenses.mean()) if home not in known else float(defenses[model.team_to_idx[home]])
        )
        away_attack = (
            float(attacks.mean()) if away not in known else float(attacks[model.team_to_idx[away]])
        )
        away_defense = (
            float(defenses.mean()) if away not in known else float(defenses[model.team_to_idx[away]])
        )
        home_advantage = float(parameters[-2] if dixon_coles else parameters[-1])
        rho = float(parameters[-1]) if dixon_coles else 0.0
        return (
            float(np.exp(home_advantage + home_attack + away_defense)),
            float(np.exp(away_attack + home_defense)),
            rho,
        )
    grid, _ = predict_fixture(model, home, away, dixon_coles=dixon_coles)
    return float(grid.home_goal_expectation), float(grid.away_goal_expectation), 0.0


def _sot_expected_goals(model: ExposureModel, home: str, away: str) -> tuple[float, float]:
    return model.expected_goals(home, away)


def predict_base_models(
    matches: pd.DataFrame,
    track_a_train: pd.DataFrame,
    track_b_train: pd.DataFrame | None,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    """Hitung probabilitas 1X2 dari semua base model untuk satu subset.

    ``track_a_train`` dipakai untuk fit model Track A dan exposure. Track B
    butuh baris dengan fitur lengkap, jadi diberi terpisah supaya pemanggil
    bisa membedakan subset yang punya fitur dan yang tidak.
    """
    out: dict[str, np.ndarray] = {}

    poisson, dixon_coles = fit_goal_models(track_a_train)
    exposure = fit_exposure_model(track_a_train, "sot", EXPOSURE_XI)

    poisson_rows, dc_rows, sot_rows = [], [], []
    for match in matches.itertuples(index=False):
        eh, ea, _ = _cold_start_expected_goals(poisson, match.team_home, match.team_away, False)
        poisson_rows.append(_grid_to_1x2(eh, ea))
        eh_dc, ea_dc, rho = _cold_start_expected_goals(
            dixon_coles, match.team_home, match.team_away, True
        )
        dc_rows.append(_dixon_coles_1x2(eh_dc, ea_dc, rho))
        sh, sa = _sot_expected_goals(exposure, match.team_home, match.team_away)
        sot_rows.append(_grid_to_1x2(sh, sa))

    out["poisson"] = np.array(poisson_rows)
    out["dixon_coles"] = np.array(dc_rows)
    out["exposure_sot"] = np.array(sot_rows)

    # elo_probabilities_for sudah kronologis: rating di-update setelah setiap
    # match, dan probabilitas diambil dari rating SEBELUM match itu.
    elo_frame = elo_probabilities_for(track_a_train, matches)
    if len(elo_frame) == len(matches):
        out["elo"] = elo_frame[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    else:
        out["elo"] = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (len(matches), 1))

    if track_b_train is not None:
        models = build_models()
        X_train = track_b_train[feature_columns]
        y_train = track_b_train["target"]
        X_target = matches[feature_columns]
        for name in TRACK_B_MODELS:
            fitted = clone(models[name]).fit(X_train, y_train)
            out[name] = fitted.predict_proba(X_target)
    return out


def _dixon_coles_1x2(expected_home: float, expected_away: float, rho: float) -> np.ndarray:
    grid = pb.models.create_dixon_coles_grid(
        expected_home, expected_away, rho=rho, max_goals=MAX_GOALS
    )
    home, draw, away = grid.home_draw_away
    total = float(home) + float(draw) + float(away)
    return np.array([float(home) / total, float(draw) / total, float(away) / total])


# --------------------------------------------------------------------------
# Meta-learner
# --------------------------------------------------------------------------


def build_meta_learner() -> Pipeline:
    """Logistic regression dengan scaling, untuk menggabungkan probabilitas."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def _assemble(features: dict[str, np.ndarray]) -> np.ndarray:
    """Susun matriks fitur meta-learner dari dict probabilitas base model.

    Keluaran adalah matriks (n_rows, 3 * n_base_model): setiap base model
    menyumbang tiga kolom H/D/A-nya sendiri.
    """
    columns = []
    for model in BASE_MODELS:
        if model not in features:
            raise ValueError(f"Base model tidak ada di features: {model}")
        matrix = features[model]
        if matrix.ndim != 2 or matrix.shape[1] != 3:
            raise ValueError(
                f"{model} harus matriks (n_rows, 3), dapat {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{model} punya nilai non-finite")
        # Validasi jumlah-1 berlaku per base model, BUKAN pada matriks gabungan.
        # Menjumlahkan seluruh kolom gabungan akan menjumlahkan semua base model
        # sekaligus, sehingga selalu menghasilkan ~7, bukan 1.
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(
                f"Baris probabilitas {model} tidak berjumlah 1 "
                f"(deviasi maks {np.abs(matrix.sum(axis=1) - 1).max():.2e})"
            )
        columns.append(matrix)
    return np.hstack(columns)


def simple_average(features: dict[str, np.ndarray]) -> np.ndarray:
    """Rata-rata aritmetik semua base model, tanpa bobot.

    Ada sebagai pembanding. Kalau meta-learner tidak mengalahkan ini, maka
    nilai tambah meta-learner hanya datang dari pemBelajaran bobot, bukan
    dari menggabungkan informasi yang benar-benar berbeda.
    """
    stacked = _assemble(features)
    averaged = stacked.reshape(len(stacked), len(BASE_MODELS), 3).mean(axis=1)
    return averaged / averaged.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Walk-forward twofold: OOF base models, lalu meta-learner
# --------------------------------------------------------------------------


def build_oof_predictions(
    data: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Bangun prediksi out-of-fold untuk seluruh baris yang tercakup.

    Hanya baris yang benar-benar ada di fold validation yang menghasilkan
    output. Baris paling awal tidak pernah masuk validation (TimeSeriesSplit
    butuh history), jadi tabel hasilnya lebih pendek dari train — itu
    normal, bukan bug.
    """
    train = data.loc[data["season"] != TEST_SEASON].copy()
    train = train.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)
    track_a_train = _with_goals(train)

    rows = []
    for fold, train_idx, validation_idx in date_based_time_splits(train):
        fold_train = track_a_train.iloc[train_idx]
        fold_validation = track_a_train.iloc[validation_idx]
        features_train = train.iloc[train_idx]
        features_validation = train.iloc[validation_idx]

        # Jaga agar baris validasi tidak pernah ikut training base model.
        if fold_train["match_id"].isin(fold_validation["match_id"]).any():
            raise ValueError(f"Fold {fold}: baris validasi bocor ke training")

        predictions = predict_base_models(
            fold_validation, fold_train, features_train, feature_columns
        )
        for position, index in enumerate(validation_idx):
            row = {
                "match_id": train.iloc[index]["match_id"],
                "date": train.iloc[index]["date"],
                "datetime": train.iloc[index]["datetime"],
                "season": train.iloc[index]["season"],
                "fold": fold,
                "target": train.iloc[index]["target"],
            }
            for model in BASE_MODELS:
                for label_index, label in enumerate(PROBABILITY_LABELS):
                    row[f"{model}_prob_{label}"] = predictions[model][
                        position, label_index
                    ]
            rows.append(row)
    return pd.DataFrame(rows)


def _with_goals(frame: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan kolom skor/tembakan yang dibutuhkan model Track A."""
    matches = pd.read_csv(
        MATCHES_PATH,
        usecols=[
            "match_id",
            "goals_home",
            "goals_away",
            "shots_home",
            "shots_away",
            "shots_on_target_home",
            "shots_on_target_away",
        ],
    )
    merged = frame.merge(matches, on="match_id", how="left", validate="one_to_one")
    needed = [
        "goals_home",
        "goals_away",
        "shots_home",
        "shots_away",
        "shots_on_target_home",
        "shots_on_target_away",
    ]
    if merged[needed].isna().any().any():
        raise ValueError("Kolom statistik tidak lengkap untuk beberapa match")
    return merged


def evaluate_blend(
    oof: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nilai meta-learner dengan walk-forward di level kedua.

    Untuk setiap fold, meta-learner dilatih pada OOF fold-fold SEBELUMNYA,
    lalu dievaluasi pada fold berjalan. Ini yang membuat angka "apakah blend
    menang" berasal dari training season, bukan dari test set.
    """
    features = feature_columns
    folds = sorted(oof["fold"].unique())
    rows = []
    for index, fold in enumerate(folds):
        meta_train = oof.loc[oof["fold"] < fold]
        meta_validation = oof.loc[oof["fold"] == fold]
        if meta_train.empty:
            continue
        y_train = meta_train["target"].to_numpy()
        y_validation = meta_validation["target"].to_numpy()

        X_train = _assemble(
            {model: meta_train[[f"{model}_prob_{label}" for label in PROBABILITY_LABELS]].to_numpy()
             for model in BASE_MODELS}
        )
        X_validation = _assemble(
            {model: meta_validation[[f"{model}_prob_{label}" for label in PROBABILITY_LABELS]].to_numpy()
             for model in BASE_MODELS}
        )

        fitted = clone(build_meta_learner()).fit(X_train, y_train)
        blend = fitted.predict_proba(X_validation)
        average = simple_average(
            {model: meta_validation[[f"{model}_prob_{label}" for label in PROBABILITY_LABELS]].to_numpy()
             for model in BASE_MODELS}
        )

        for name, matrix in (("blend_meta", blend), ("blend_simple_average", average)):
            rows.append(
                {
                    "fold": int(fold),
                    "model": name,
                    "n_matches": len(y_validation),
                    "log_loss": float(
                        log_loss(y_validation, matrix, labels=[0, 1, 2])
                    ),
                    "rps": float(pb.metrics.rps_average(matrix, y_validation)),
                    "accuracy": float(accuracy_score(y_validation, matrix.argmax(axis=1))),
                }
            )

        # Base model pada fold yang sama, sebagai pembanding paired.
        for model in BASE_MODELS:
            matrix = meta_validation[
                [f"{model}_prob_{label}" for label in PROBABILITY_LABELS]
            ].to_numpy()
            rows.append(
                {
                    "fold": int(fold),
                    "model": model,
                    "n_matches": len(y_validation),
                    "log_loss": float(log_loss(y_validation, matrix, labels=[0, 1, 2])),
                    "rps": float(pb.metrics.rps_average(matrix, y_validation)),
                    "accuracy": float(accuracy_score(y_validation, matrix.argmax(axis=1))),
                }
            )
    return pd.DataFrame(rows), _meta_learner_report(oof)


def _meta_learner_report(oof: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan koefisien meta-learner dari fit penuh di OOF.

    LogisticRegression 3 kelas punya ``coef_`` berbentuk (3, n_features):
    satu baris per kelas H/D/A. Melihat per-kelas berguna untuk melihat
    apakah suatu base model berperan berbeda di tiap kelas, tapi untuk
    "siapa yang paling berpengaruh" yang relevan adalah besar absolut
    rata-rata antar kelas. Keduanya dilaporkan supaya tidak ada yang
    disembunyikan.
    """
    X = _assemble(
        {
            model: oof[[f"{model}_prob_{label}" for label in PROBABILITY_LABELS]].to_numpy()
            for model in BASE_MODELS
        }
    )
    y = oof["target"].to_numpy()
    fitted = clone(build_meta_learner()).fit(X, y)
    coefficients = np.asarray(fitted.named_steps["classifier"].coef_)

    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    if coefficients.shape[1] != len(feature_names()):
        raise ValueError(
            f"Jumlah koefisien ({coefficients.shape[1]}) tidak cocok dengan "
            f"jumlah fitur ({len(feature_names())})"
        )

    table = pd.DataFrame(
        {
            "feature": feature_names(),
            "coef_home": coefficients[0],
            "coef_draw": coefficients[1],
            "coef_away": coefficients[2],
        }
    )
    table["abs_coefficient"] = table[
        ["coef_home", "coef_draw", "coef_away"]
    ].abs().mean(axis=1)
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Test season: satu kali baca, hanya untuk pelaporan
# --------------------------------------------------------------------------


def evaluate_test_set(
    data: pd.DataFrame, feature_columns: list[str], oof: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nilai blend di test season, hanya untuk pelaporan akhir."""
    train = data.loc[data["season"] != TEST_SEASON].copy()
    test = data.loc[data["season"] == TEST_SEASON].copy()
    train = train.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)
    test = test.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)

    if (train["season"] == TEST_SEASON).any():
        raise ValueError("TEST_SEASON bocor ke training meta-learner")

    track_a_train = _with_goals(train)
    test_features = predict_base_models(
        _with_goals(test), track_a_train, train, feature_columns
    )

    # Meta-learner dilatih HANYA di OOF training season.
    X_oof = _assemble(
        {
            model: oof[[f"{model}_prob_{label}" for label in PROBABILITY_LABELS]].to_numpy()
            for model in BASE_MODELS
        }
    )
    y_oof = oof["target"].to_numpy()
    meta = clone(build_meta_learner()).fit(X_oof, y_oof)

    X_test = _assemble(test_features)
    blend = meta.predict_proba(X_test)
    average = simple_average(test_features)

    y_test = test["target"].to_numpy()
    rows = []
    for name, matrix in (("blend_meta", blend), ("blend_simple_average", average)):
        rows.append(
            {
                "model": name,
                "n_matches": len(y_test),
                "log_loss": float(log_loss(y_test, matrix, labels=[0, 1, 2])),
                "rps": float(pb.metrics.rps_average(matrix, y_test)),
                "accuracy": float(accuracy_score(y_test, matrix.argmax(axis=1))),
            }
        )
    for model in BASE_MODELS:
        matrix = test_features[model]
        rows.append(
            {
                "model": model,
                "n_matches": len(y_test),
                "log_loss": float(log_loss(y_test, matrix, labels=[0, 1, 2])),
                "rps": float(pb.metrics.rps_average(matrix, y_test)),
                "accuracy": float(accuracy_score(y_test, matrix.argmax(axis=1))),
            }
        )

    bookmaker = _load_bookmaker(test)
    if bookmaker is not None:
        rows.append(
            {
                "model": BOOKMAKER_NAME,
                "n_matches": len(y_test),
                "log_loss": float(log_loss(y_test, bookmaker, labels=[0, 1, 2])),
                "rps": float(pb.metrics.rps_average(bookmaker, y_test)),
                "accuracy": float(accuracy_score(y_test, bookmaker.argmax(axis=1))),
            }
        )

    predictions = test[
        ["match_id", "season", "datetime", "team_home", "team_away", "target"]
    ].copy()
    for index, label in enumerate(PROBABILITY_LABELS):
        predictions[f"blend_meta_prob_{label}"] = blend[:, index]
        predictions[f"blend_simple_average_prob_{label}"] = average[:, index]
    for model in BASE_MODELS:
        for index, label in enumerate(PROBABILITY_LABELS):
            predictions[f"{model}_prob_{label}"] = test_features[model][:, index]
    if bookmaker is not None:
        for index, label in enumerate(PROBABILITY_LABELS):
            predictions[f"{BOOKMAKER_NAME}_prob_{label}"] = bookmaker[:, index]

    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True), predictions


def _load_bookmaker(test: pd.DataFrame) -> np.ndarray | None:
    """Muat probabilitas odds bandar untuk baris test, atau None kalau tidak ada."""
    if not BOOKMAKER_PATH.exists():
        return None
    bookmaker = pd.read_csv(BOOKMAKER_PATH)
    merged = test[["match_id"]].merge(bookmaker, on="match_id", how="inner")
    if len(merged) != len(test):
        return None
    return merged[["prob_home", "prob_draw", "prob_away"]].to_numpy()


# --------------------------------------------------------------------------
# Perbandingan & laporan
# --------------------------------------------------------------------------


def _per_match_loss(probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    target = probabilities[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(target, 1e-15, 1.0))


def paired_bootstrap(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """CI bootstrap untuk selisih log loss kandidat minus referensi."""
    delta = _per_match_loss(candidate, y_true) - _per_match_loss(reference, y_true)
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = delta[rng.integers(0, n, (BOOTSTRAP_SAMPLES, n))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(delta.mean()), float(low), float(high)


def compare_blends(
    test_summary: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Bandingkan blend vs tiap base model dan vs odds bandar di test season."""
    y_true = predictions["target"].to_numpy()
    best_single_name = None
    base_names = [m for m in BASE_MODELS]
    base_rows = test_summary.loc[test_summary["model"].isin(base_names)]
    if not base_rows.empty:
        best_single_name = str(base_rows.sort_values("log_loss").iloc[0]["model"])

    rows = []
    for blend_name in ("blend_meta", "blend_simple_average"):
        candidate = predictions[
            [f"{blend_name}_prob_{label}" for label in PROBABILITY_LABELS]
        ].to_numpy()
        references = [(best_single_name, "best_single_model")] if best_single_name else []
        references += [
            (m, "base_model") for m in base_names
        ]
        if BOOKMAKER_NAME in set(test_summary["model"]):
            references.append((BOOKMAKER_NAME, "bookmaker"))
        for reference_name, kind in references:
            if reference_name == blend_name:
                continue
            reference = predictions[
                [f"{reference_name}_prob_{label}" for label in PROBABILITY_LABELS]
            ].to_numpy()
            mean, low, high = paired_bootstrap(
                y_true, candidate, reference, RANDOM_STATE
            )
            rows.append(
                {
                    "blend": blend_name,
                    "reference": reference_name,
                    "reference_kind": kind,
                    "delta_log_loss_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "significant": bool(low * high > 0),
                    "blend_better": bool(mean < 0),
                }
            )
    return pd.DataFrame(rows)


def gap_to_bookmaker(
    test_summary: pd.DataFrame, comparison: pd.DataFrame
) -> pd.DataFrame:
    """Hitung jarak tiap kandidat ke odds bandar, lalu bandingkan dengan dulu."""
    indexed = test_summary.set_index("model")
    if BOOKMAKER_NAME not in indexed.index:
        return pd.DataFrame()

    bookmaker = indexed.loc[BOOKMAKER_NAME]
    base_rows = test_summary.loc[test_summary["model"].isin(BASE_MODELS)]
    best_base = base_rows.sort_values("log_loss").iloc[0]

    candidates = {
        "blend_meta": float(
            indexed.loc["blend_meta", "log_loss"] - bookmaker["log_loss"]
        ),
        "blend_simple_average": float(
            indexed.loc["blend_simple_average", "log_loss"] - bookmaker["log_loss"]
        ),
        f"best_single_model({best_base['model']})": float(
            best_base["log_loss"] - bookmaker["log_loss"]
        ),
    }
    previous = PRE_IMPROVEMENT["best_model"]
    old_gap = previous["log_loss"] - PRE_IMPROVEMENT["bookmaker_avg_odds"]["log_loss"]
    old_rps_gap = previous["rps"] - PRE_IMPROVEMENT["bookmaker_avg_odds"]["rps"]

    rps_gaps = {
        "blend_meta": float(
            indexed.loc["blend_meta", "rps"] - bookmaker["rps"]
        ),
        "blend_simple_average": float(
            indexed.loc["blend_simple_average", "rps"] - bookmaker["rps"]
        ),
        f"best_single_model({best_base['model']})": float(
            best_base["rps"] - bookmaker["rps"]
        ),
    }

    rows = []
    for name, gap in candidates.items():
        rows.append(
            {
                "candidate": name,
                "log_loss_gap_vs_bookmaker": gap,
                "rps_gap_vs_bookmaker": rps_gaps[name],
                "log_loss_gap_before_4_weeks": old_gap,
                "rps_gap_before_4_weeks": old_rps_gap,
                "log_loss_gap_change": gap - old_gap,
                "rps_gap_change": rps_gaps[name] - old_rps_gap,
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss_gap_vs_bookmaker").reset_index(
        drop=True
    )


def print_report(
    fold_results: pd.DataFrame,
    test_summary: pd.DataFrame,
    gateways: pd.DataFrame,
    comparison: pd.DataFrame,
    gap: pd.DataFrame,
) -> None:
    width = 78
    print("\n" + "=" * width)
    print("BLEND SISTEMATIS — META-LEARNER DI ATAS OOF DARI 7 BASE MODEL")
    print("=" * width)

    print("\nPenting: base model yang tersedia (hasil koreksi premis):")
    print("  - Tidak ada RF terkalibrasi: kalibrasi = hasil negatif (HANDOFF 5.2)")
    print("  - Tidak ada RF ter-tuning: tuning RF tidak significant (5.6)")
    print("  - Tidak ada model xG: yang Stage 3 hasil = exposure-based, bukan xG")

    print("\n[1] Level-2 walk-forward (training season) — inilah angka penentu:")
    cv_blend = fold_results.loc[fold_results["model"].str.startswith("blend")]
    cv_pivot = cv_blend.pivot_table(index="fold", columns="model", values="log_loss")
    print(cv_pivot.to_string(float_format=lambda v: f"{v:.6f}"))
    print("\nRata-rata per model (level-2 CV):")
    print(
        fold_results.groupby("model")[["log_loss", "rps"]]
        .mean()
        .sort_values("log_loss")
        .to_string(float_format=lambda v: f"{v:.6f}")
    )

    print("\n[2] Test season 2025-2026 (hanya pelaporan, tidak dipakai memilih):")
    print(
        test_summary[
            ["model", "log_loss", "rps", "accuracy"]
        ].to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )

    print("\n[3] Koefisien meta-learner (top 10, |coef| rata-rata terbesar):")
    print(
        gateways.head(10)[
            ["feature", "coef_home", "coef_draw", "coef_away", "abs_coefficient"]
        ].to_string(index=False, float_format=lambda v: f"{v:+.4f}")
    )

    print("\n[4] Perbandingan paired di test season (negatif = blend lebih baik):")
    top = comparison.loc[comparison["reference_kind"].isin(["best_single_model", "bookmaker"])]
    print(
        top[["blend", "reference", "delta_log_loss_mean", "ci_low", "ci_high", "significant"]]
        .to_string(index=False, float_format=lambda v: f"{v:+.6f}")
    )

    if not gap.empty:
        print("\n[5] Jarak ke odds bandar, sekarang vs sebelum 4 minggu perbaikan:")
        print(
            gap[
                [
                    "candidate",
                    "log_loss_gap_vs_bookmaker",
                    "rps_gap_vs_bookmaker",
                    "log_loss_gap_before_4_weeks",
                    "log_loss_gap_change",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:+.6f}")
        )
        print(
            f"\n  Jarak SEBELUM: log loss {PRE_IMPROVEMENT['best_model']['log_loss'] - PRE_IMPROVEMENT['bookmaker_avg_odds']['log_loss']:+.6f}, "
            f"RPS {PRE_IMPROVEMENT['best_model']['rps'] - PRE_IMPROVEMENT['bookmaker_avg_odds']['rps']:+.6f} "
            "(random_forest, capture pre-upgrade)"
        )
    print("=" * width)


def main() -> None:
    print("=" * 78)
    print("Eksperimen: blend sistematis semua model (meta-learner di atas OOF)")
    print("=" * 78)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    data, feature_columns = load_modeling_data()
    data["target"] = data["result"].map(TARGET_MAPPING)

    print(f"\nMembangun OOF untuk {len(BASE_MODELS)} base model: {', '.join(BASE_MODELS)}")
    oof = build_oof_predictions(data, feature_columns)
    oof.to_csv(OOF_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"OOF rows: {len(oof)} dari {int((data['season'] != TEST_SEASON).sum())} training")

    fold_results, gateways = evaluate_blend(oof, feature_columns)
    fold_results.to_csv(FOLD_RESULTS_PATH, index=False)

    test_summary, predictions = evaluate_test_set(data, feature_columns, oof)
    test_summary.to_csv(SUMMARY_PATH, index=False)
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    gateways.to_csv(GATEWAY_PATH, index=False)

    comparison = compare_blends(test_summary, predictions)
    comparison.to_csv(COMPARISON_PATH, index=False)

    gap = gap_to_bookmaker(test_summary, comparison)
    if not gap.empty:
        gap.to_csv(GAP_PATH, index=False)

    print_report(fold_results, test_summary, gateways, comparison, gap)

    payload = {
        "purpose": (
            "Blend seluruh base model. evaluate.py, predict_match.py, dan "
            "cv_model_selection.csv sengaja tidak diubah."
        ),
        "base_models": list(BASE_MODELS),
        "premise_corrections": {
            "no_calibrated_rf": "Kalibrasi post-hoc = hasil negatif (HANDOFF 5.2)",
            "no_tuned_rf": "Tuning RF tidak significant (HANDOFF 5.6); XGBoost yang ter-tuning",
            "no_xg_model": (
                "Stage 3 menghasilkan exposure-based Poisson, BUKAN xG. Tidak ada "
                "sumber xG EPL gratis berlisensi jelas."
            ),
        },
        "meta_learner": "LogisticRegression(StandardScaler) di atas 21 fitur probabilitas",
        "leakage_controls": [
            "Base models di-fit ulang per fold; meta dilatih hanya di OOF",
            "Meta-learner di-evaluasi walk-forward di level kedua (bukan di test)",
            "Test season hanya dibaca untuk pelaporan, tidak pernah jadi fold",
        ],
        "exposure_xi_used": EXPOSURE_XI,
        "exposure_xi_source": "eksperimen Stage 3, training season only",
        "test_season": TEST_SEASON,
        "cv_summary": fold_results.groupby("model")[["log_loss", "rps", "accuracy"]]
        .mean()
        .to_dict("index"),
        "test_summary": test_summary.to_dict("records"),
        "top_coefficients": gateways.head(10).to_dict("records"),
        "comparison": comparison.to_dict("records"),
        "gap_to_bookmaker": gap.to_dict("records"),
        "gap_before_4_weeks": {
            "source": "research/evaluations/baseline.md (pre-upgrade capture)",
            "log_loss_gap": PRE_IMPROVEMENT["best_model"]["log_loss"]
            - PRE_IMPROVEMENT["bookmaker_avg_odds"]["log_loss"],
            "rps_gap": PRE_IMPROVEMENT["best_model"]["rps"]
            - PRE_IMPROVEMENT["bookmaker_avg_odds"]["rps"],
        },
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nTersimpan: {OOF_PATH}")
    print(f"Tersimpan: {FOLD_RESULTS_PATH}")
    print(f"Tersimpan: {SUMMARY_PATH}")
    print(f"Tersimpan: {GATEWAY_PATH}")
    print(f"Tersimpan: {COMPARISON_PATH}")
    if not gap.empty:
        print(f"Tersimpan: {GAP_PATH}")
    print(f"Tersimpan: {METADATA_PATH}")


if __name__ == "__main__":
    main()
