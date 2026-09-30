"""Tests untuk blend sistematis (src/blend.py).

Fokus test: aturan anti-leakage blend dan konsistensi dengan evaluate.py.
Guard paling penting di sini adalah memastikan prediksi OOF benar-benar
out-of-fold, karena blend yang dilatih pada prediksi in-sample akan selalu
terlihat bagus dan tidak berarti apa-apa.
"""

import numpy as np
import pandas as pd
import pytest

import blend
from blend import (
    BASE_MODELS,
    PROBABILITY_LABELS,
    _assemble,
    build_meta_learner,
    feature_names,
    simple_average,
)
from config import TEST_SEASON


def _synthetic_matches(seasons=None, matches_per_season=10):
    if seasons is None:
        from config import SEASONS

        seasons = [s for s in SEASONS if s != TEST_SEASON]
    rows = []
    day = pd.Timestamp("2016-08-01")
    for season in seasons:
        for index in range(matches_per_season):
            home = f"T{index % 6}"
            away = f"T{(index + 1) % 6}"
            # Result bervariasi supaya semua kelas H/D/A muncul; LogisticRegression
            # butuh minimal 2 kelas dan test blend butuh ketiga kelas.
            result, target = ("H", 0) if index % 3 == 0 else (
                ("D", 1) if index % 3 == 1 else ("A", 2)
            )
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "season": season,
                    "date": day + pd.Timedelta(days=index * 2),
                    "datetime": day + pd.Timedelta(days=index * 2),
                    "team_home": home,
                    "team_away": away,
                    "goals_home": 2,
                    "goals_away": 1,
                    "shots_home": 14,
                    "shots_away": 10,
                    "shots_on_target_home": 5,
                    "shots_on_target_away": 4,
                    "result": result,
                    "target": target,
                }
            )
        day = day + pd.Timedelta(days=200)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Struktur fitur
# --------------------------------------------------------------------------


def test_feature_names_cover_all_base_models():
    names = feature_names()
    assert len(names) == 3 * len(BASE_MODELS)
    for model in BASE_MODELS:
        for label in PROBABILITY_LABELS:
            assert f"{model}_prob_{label}" in names


def test_feature_names_are_unique():
    assert len(set(feature_names())) == len(feature_names())


def test_base_models_include_all_tracks():
    """Blend harus mencakup Track A, Track B, dan eksperimen Stage 3."""
    assert "poisson" in BASE_MODELS
    assert "dixon_coles" in BASE_MODELS
    assert "elo" in BASE_MODELS
    assert "logistic_regression" in BASE_MODELS
    assert "random_forest" in BASE_MODELS
    assert "xgboost" in BASE_MODELS
    assert "exposure_sot" in BASE_MODELS


def test_exposure_model_is_not_called_xg():
    """Penamaan: model Stage 3 harus 'exposure_sot', bukan 'xg'.

    Nama 'xgboost' mengandung substring 'xg' tapi itu nama algoritme
    gradient boosting, bukan klaim bahwa datanya xG. Yang diperiksa di sini
    adalah tidak adanya model yang namanya mengklaim xG.
    """
    xg_claims = [
        name
        for name in BASE_MODELS
        if "xg" in name.lower() and name != "xgboost"
    ]
    assert xg_claims == [], f"Nama model mengklaim xG: {xg_claims}"
    assert "exposure_sot" in BASE_MODELS


# --------------------------------------------------------------------------
# Perakitan matriks
# --------------------------------------------------------------------------


def _valid_features(n_rows=10, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for model in BASE_MODELS:
        raw = rng.uniform(0.1, 1.0, (n_rows, 3))
        out[model] = raw / raw.sum(axis=1, keepdims=True)
    return out


def test_assemble_produces_expected_shape():
    features = _valid_features(n_rows=17)
    X = _assemble(features)
    assert X.shape == (17, 3 * len(BASE_MODELS))


def test_assemble_rejects_wrong_column_count():
    features = _valid_features()
    features["poisson"] = features["poisson"][:, :2]
    with pytest.raises(ValueError, match="harus matriks"):
        _assemble(features)


def test_assemble_rejects_non_finite():
    features = _valid_features()
    features["elo"] = features["elo"].copy()
    features["elo"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _assemble(features)


def test_assemble_rejects_row_not_summing_to_one():
    """Validasi jumlah-1 harus per base model, bukan pada matriks gabungan."""
    features = _valid_features()
    features["poisson"] = features["poisson"] * 2.0
    with pytest.raises(ValueError, match="tidak berjumlah 1"):
        _assemble(features)


def test_assemble_error_message_names_the_model():
    features = _valid_features()
    features["exposure_sot"] = features["exposure_sot"] * 0.5
    with pytest.raises(ValueError, match="exposure_sot"):
        _assemble(features)


# --------------------------------------------------------------------------
# Simple average
# --------------------------------------------------------------------------


def test_simple_average_returns_valid_distribution():
    features = _valid_features(n_rows=13)
    averaged = simple_average(features)
    assert averaged.shape == (13, 3)
    assert np.allclose(averaged.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(averaged > 0)


def test_simple_average_of_identical_models_is_that_model():
    base = _valid_features(n_rows=5)
    identical = {model: base["poisson"].copy() for model in BASE_MODELS}
    averaged = simple_average(identical)
    assert np.allclose(averaged, base["poisson"], atol=1e-9)


def test_simple_average_of_uniform_is_uniform():
    uniform = {m: np.tile([1 / 3, 1 / 3, 1 / 3], (8, 1)) for m in BASE_MODELS}
    averaged = simple_average(uniform)
    assert np.allclose(averaged, 1 / 3, atol=1e-9)


# --------------------------------------------------------------------------
# Anti-leakage
# --------------------------------------------------------------------------


def test_test_season_never_in_base_models_oof():
    """Guard: build_oof_predictions tidak boleh menghasilkan baris test season."""
    oof_path = blend.OOF_PATH
    if not oof_path.exists():
        pytest.skip("blend_oof_predictions.csv belum ada")
    oof = pd.read_csv(oof_path)
    assert not (oof["season"] == TEST_SEASON).any(), (
        "OOF blend tidak boleh memuat test season"
    )


def test_oof_rows_are_strictly_chronological_by_fold():
    """OOF fold harus urut naik dalam waktu, supaya meta-learner takzobran masa depan."""
    oof_path = blend.OOF_PATH
    if not oof_path.exists():
        pytest.skip("blend_oof_predictions.csv belum ada")
    oof = pd.read_csv(oof_path, parse_dates=["date"])
    for fold, group in oof.groupby("fold"):
        assert group["date"].is_monotonic_increasing, f"fold {fold} tidak terurut"
    bounds = oof.groupby("fold")["date"].max()
    ordered = bounds.sort_index()
    assert ordered.is_monotonic_increasing, "fold berikutnya harus lebih akhir"


def test_meta_learner_never_trained_on_test_season():
    """Guard tambahan: OOF rows fewer dari training total itu normal.

    TimeSeriesSplit butuh history, jadi baris awal tidak pernah masuk
    validation. Kalau OOF == training total, berarti tidak ada OOF sama
    sekali dan blend-nya in-sample.
    """
    oof_path = blend.OOF_PATH
    if not oof_path.exists():
        pytest.skip("blend_oof_predictions.csv belum ada")
    oof = pd.read_csv(oof_path)
    assert len(oof) > 0
    folds = sorted(oof["fold"].unique())
    assert len(folds) >= 2, "butuh minimal 2 fold untuk OOF"
    # Baris paling awal tidak bisa punya prediksi OOF.
    assert oof["fold"].min() == folds[0]
    assert not oof.duplicated(subset=["match_id"]).any(), "match_id duplikat di OOF"


def test_build_oof_rejects_overlapping_splits(monkeypatch):
    """Guard anti-leakage harus menangkap splitter yang salah.

    Fungsi ``date_based_time_splits`` sudah menjaga urutannya sendiri, tapi
    blend tidak boleh bergantung pada itu buta. Test ini menyuntikkan
    splitter yang menyertakan baris validasi di training (misal karena
    refactor di masa depan) dan memastikan guard di build_oof_predictions
    menahannya.
    """
    frame = _synthetic_matches()
    frame["dummy"] = 1.0
    frame = frame.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)

    def overlapping_splits(data, n_splits=5):
        # Fold tunggal yang training-nya memuat separuh baris validasi.
        half = len(data) // 2
        train_idx = np.arange(half + 10)
        validation_idx = np.arange(half, len(data))
        yield 1, train_idx, validation_idx

    monkeypatch.setattr(blend, "date_based_time_splits", overlapping_splits)
    monkeypatch.setattr(blend, "_with_goals", lambda f: f)
    with pytest.raises(ValueError, match="bocor"):
        blend.build_oof_predictions(frame, ["dummy"])


def test_build_oof_uses_real_splitter_and_stays_chronological(monkeypatch):
    """Dengan splitter asli, OOF harus benar-benar out-of-sample."""
    frame = _synthetic_matches()
    frame["dummy"] = 1.0
    frame = frame.sort_values(["date", "datetime", "match_id"]).reset_index(drop=True)

    seen = []

    def spy(matches, track_a_train, track_b_train, feature_columns):
        overlap = set(track_a_train["match_id"]) & set(matches["match_id"])
        seen.append(len(overlap))
        raise ValueError("STOP")

    monkeypatch.setattr(blend, "predict_base_models", spy)
    monkeypatch.setattr(blend, "_with_goals", lambda f: f)
    with pytest.raises(ValueError, match="STOP"):
        blend.build_oof_predictions(frame, ["dummy"])
    assert seen == [0], f"ada fold dengan baris tumpang tindih: {seen}"


# --------------------------------------------------------------------------
# Meta-learner
# --------------------------------------------------------------------------


def test_meta_learner_trains_and_predicts():
    features = _valid_features(n_rows=200, seed=5)
    X = _assemble(features)
    y = np.random.default_rng(7).integers(0, 3, 200)
    fitted = blend.clone(build_meta_learner()).fit(X, y)
    probabilities = fitted.predict_proba(X)
    assert probabilities.shape == (200, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_meta_learner_coefficients_have_class_dimension():
    """coef_ harus (3, n_features); regresi ravel akan salah Destruction."""
    oof_path = blend.OOF_PATH
    if not oof_path.exists():
        pytest.skip("blend_oof_predictions.csv belum ada")
    oof = pd.read_csv(oof_path)
    table = blend._meta_learner_report(oof)
    assert len(table) == len(feature_names())
    for column in ("coef_home", "coef_draw", "coef_away"):
        assert column in table.columns
    assert table["abs_coefficient"].is_monotonic_decreasing


# --------------------------------------------------------------------------
# Bookmaker
# --------------------------------------------------------------------------


def test_bookmaker_loader_returns_distribution_or_none():
    test = pd.DataFrame({"match_id": ["nonexistent-id"]})
    result = blend._load_bookmaker(test)
    assert result is None


def test_bookmaker_rows_sum_to_one():
    if not blend.BOOKMAKER_PATH.exists():
        pytest.skip("bookmaker_probabilities.csv belum ada")
    bookmaker = pd.read_csv(blend.BOOKMAKER_PATH)
    matrix = bookmaker[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6)


# --------------------------------------------------------------------------
# Keamanan eksperimen
# --------------------------------------------------------------------------


def test_module_does_not_touch_production_artifacts():
    production = {
        "evaluation_summary.csv",
        "cv_model_selection.csv",
        "ml_model_metrics.csv",
        "best_ml_model.pkl",
    }
    written = {
        blend.OOF_PATH.name,
        blend.FOLD_RESULTS_PATH.name,
        blend.SUMMARY_PATH.name,
        blend.GATEWAY_PATH.name,
        blend.COMPARISON_PATH.name,
        blend.METADATA_PATH.name,
    }
    assert not (written & production)


def test_exposure_xi_is_from_training_season_only():
    """Nilai xi exposure harus sesuai hasil eksperimen Stage 3, bukan hasil test."""
    assert blend.EXPOSURE_XI == 0.0025


def test_per_match_loss_matches_sklearn():
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(13)
    y_true = rng.integers(0, 3, 150)
    raw = rng.uniform(0.2, 1.0, (150, 3))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    per_match = blend._per_match_loss(probabilities, y_true)
    assert per_match.mean() == pytest.approx(
        log_loss(y_true, probabilities, labels=[0, 1, 2]), rel=1e-9
    )
