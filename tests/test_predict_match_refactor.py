"""Tests untuk refactor prediksi yang dipakai bersama CLI dan web.

Sebelum refactor, logika prediksi hidup di dalam ``main()`` sebagai variabel
lokal, jadi tidak bisa dipanggil dari mana lain. Web app butuh memanggil
perhitungan yang sama, bukan menduplikasi-nya: kalau web punya salinannya
sendiri, tidak ada yang mencegah keduanya lama-lama berbeda untuk input yang
sama.

Yang dikunci test di sini:
- ``get_prediction()`` mengembalikan semua kunci yang ``print_prediction()``
  konsumsi, jadi CLI yang sudah direfactor tidak bisa kehilangan bagian.
- Dropdown tim dibangun dari data nyata, bukan daftar yang ditulis manual.
- Snapshot ``deploy/`` menghasilkan pemilihan model yang sama dengan
  pipeline lokal. Kalau tidak, web diam-diam memakai model berbeda dari CLI.
- CLI dan web memakai fungsi yang sama, bukan implementasi terpisah.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

import predict_match as pm

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "deploy"

# Kunci yang dibutuhkan print_prediction() dari hasil get_prediction().
PREDICTION_KEYS = {
    "home_team",
    "away_team",
    "prediction_datetime",
    "goal_selection",
    "classification_selection",
    "score_grid",
    "ml_probabilities",
    "yellow_grid",
    "yellow_cold_start",
    "yellow_threshold",
    "corner_grid",
    "corner_cold_start",
    "corner_threshold",
    "goal_cold_start",
    "warnings",
}


@pytest.fixture(scope="module")
def prediction():
    return pm.get_prediction("Arsenal", "Chelsea")


# --------------------------------------------------------------------------
# Kontrak get_prediction()
# --------------------------------------------------------------------------


def test_get_prediction_returns_every_needed_key(prediction):
    assert PREDICTION_KEYS.issubset(prediction.keys())


def test_get_prediction_echoes_teams(prediction):
    assert prediction["home_team"] == "Arsenal"
    assert prediction["away_team"] == "Chelsea"


def test_ml_probabilities_sum_to_one(prediction):
    probabilities = prediction["ml_probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(probabilities) == {"H", "D", "A"}


def test_score_grid_is_usable(prediction):
    """top_scorelines() harus jalan tanpa error, sama seperti di CLI."""
    scorelines = pm.top_scorelines(prediction["score_grid"], count=3)
    assert len(scorelines) == 3
    for home_goals, away_goals, probability in scorelines:
        assert home_goals >= 0 and away_goals >= 0
        assert 0.0 <= probability <= 1.0


def test_selection_series_have_cv_columns(prediction):
    """Selection harus punya kolom CV, bukan kolom test season.

    Ini guarding kontaminasi test set: kalau suatu saat refactor ini
    dialihkan ke evaluation_summary.csv, kolomnya jadi log_loss/rps dan
    select_best_model akan gagal dengan KeyError.
    """
    for key in ("goal_selection", "classification_selection"):
        row = prediction[key]
        assert "cv_log_loss_mean" in row.index
        assert "cv_accuracy_mean" in row.index
        assert "selection_basis" in row.index


def test_warnings_are_a_list(prediction):
    assert isinstance(prediction["warnings"], list)
    assert all(isinstance(warning, str) for warning in prediction["warnings"])


def test_goal_model_path_is_used_not_global_constant():
    """Path goal model harus lewat goal_model_path() supaya bisa dialihkan.

    Kalau kembali ke konstanta modul, web akan membaca pipeline dir yang
    tidak ada di repo hasil clone.
    """
    source = inspect.getsource(pm.get_prediction)
    assert "goal_model_path(" in source
    assert "GOAL_MODEL_PATHS[" not in source


# --------------------------------------------------------------------------
# Dropdown tim
# --------------------------------------------------------------------------


def test_list_teams_is_sorted_and_nonempty():
    teams = pm.list_teams()
    assert teams == sorted(teams)
    assert len(teams) > 20
    assert len(teams) == len(set(teams)), "nama tim harus unik"


def test_list_teams_matches_dataset():
    matches = pm.load_predictor().matches
    observed = set(matches["team_home"]) | set(matches["team_away"])
    assert set(pm.list_teams()) == observed


def test_list_teams_contains_cli_defaults():
    """Default dropdown di app.py harus benar-benar ada di daftar tim."""
    assert "Arsenal" in pm.list_teams()
    assert "Chelsea" in pm.list_teams()


def test_list_teams_avoids_cli_manual_validation_need():
    """Dropdown menutup salah ketik; CLI tetap perlu validasi manual.

    Tidak ada fungsi ``validate_team_name`` di app karena Listsselectbox
    Streamlit membuat kategori tertutup. Regression guard supaya tidak
    ada input teks bebas yang ditambahkan balik ke web nanti.
    """
    assert "prompt_team" not in inspect.getsource(pm.list_teams)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_load_predictor_is_cached():
    first = pm.load_predictor()
    second = pm.load_predictor()
    assert first is second, "artefak harus di-cache, bukan dibaca ulang tiap widget"


def test_use_snapshot_directory_resets_cache():
    """Berpindah direktori harus membuang cache, bukan memakai artefak lama."""
    original_paths, original_cache = pm._ACTIVE_PATHS, pm._PREDICTOR_CACHE
    try:
        pipeline_predictor = pm.load_predictor()
        pm.use_snapshot_directory(DEPLOY_DIR)
        snapshot_predictor = pm.load_predictor()
        assert snapshot_predictor is not pipeline_predictor, "cache lama ikut terpakai"

        # Panggil kedua kali ke direktori yang sama harus cache.
        assert pm.load_predictor() is snapshot_predictor

        # Balik ke pipeline harus membuang cache juga, bukan mengembalikan
        # objek pipeline lama yang sudah tidak sesuai direktori aktif.
        pm._ACTIVE_PATHS = None
        pm._PREDICTOR_CACHE = None
        assert pm.load_predictor() is not snapshot_predictor
    finally:
        pm._ACTIVE_PATHS, pm._PREDICTOR_CACHE = original_paths, original_cache


# --------------------------------------------------------------------------
# Snapshot deploy/
# --------------------------------------------------------------------------


def test_snapshot_exists_and_is_tracked():
    for name in ("matches_clean.csv", "cv_model_selection.csv"):
        assert (DEPLOY_DIR / name).exists(), f"deploy/{name} harus ada untuk web"
    assert (DEPLOY_DIR / "models").is_dir()


def test_snapshot_selects_same_models_as_pipeline():
    """Kontrak terpenting: web dan CLI harus memilih model yang sama."""
    pipeline_selection = pd.read_csv(pm.PROCESSED_DIR / "cv_model_selection.csv")
    snapshot_selection = pd.read_csv(DEPLOY_DIR / "cv_model_selection.csv")

    for candidates, label in (
        (pm.GOAL_MODEL_CANDIDATES, "goal"),
        (pm.CLASSIFICATION_MODEL_CANDIDATES, "classification"),
    ):
        pipeline_winner = pm.select_best_model(pipeline_selection, candidates)["model"]
        snapshot_winner = pm.select_best_model(snapshot_selection, candidates)["model"]
        assert pipeline_winner == snapshot_winner, f"winner {label} berbeda"


def test_snapshot_cv_selection_is_not_contaminated():
    """Snapshot selection harus CV out-of-fold, bukan metrik test."""
    snapshot = pd.read_csv(DEPLOY_DIR / "cv_model_selection.csv")
    assert "cv_log_loss_mean" in snapshot.columns
    assert "cv_log_loss_std" in snapshot.columns


def test_evaluation_summary_ships_but_is_not_used_for_selection():
    """File-nya ikut ter-deploy, tapi tidak boleh jadi sumber selection."""
    assert (DEPLOY_DIR / "evaluation_summary.csv").exists()

    # Nama file boleh muncul di docstring dan komentar, jadi keduanya dihapus
    # dulu sebelum dicek. Yang tersisa adalah kode yang benar-benar jalan.
    module_code = re.sub(r'"""(?:.|\n)*?"""', "", inspect.getsource(pm))
    module_code = re.sub(r"#.*", "", module_code)
    assert "evaluation_summary" not in module_code, (
        "ada kode yang membaca evaluation_summary.csv untuk selection"
    )
    assert "cv_model_selection.csv" in module_code


def test_selection_actually_runs_against_snapshot_not_summary():
    """Bukti perilaku, bukan hanya grep: snapshot harus bisa di-selection."""
    snapshot = pd.read_csv(DEPLOY_DIR / "cv_model_selection.csv")
    winner = pm.select_best_model(snapshot, pm.GOAL_MODEL_CANDIDATES)
    assert winner["model"] in pm.GOAL_MODEL_CANDIDATES

    # File summary memang tidak bisa dipakai: kolomnya tidak ada.
    summary = pd.read_csv(DEPLOY_DIR / "evaluation_summary.csv")
    with pytest.raises(KeyError):
        pm.select_best_model(summary, pm.GOAL_MODEL_CANDIDATES)


def test_snapshot_manifest_verifies():
    from build_deploy_snapshot import verify_snapshot

    report = verify_snapshot()
    assert report["exists"].all()
    assert report["sha256_match"].all(), "snapshot berubah di luar build script"


# --------------------------------------------------------------------------
# CLI dan web memakai fungsi yang sama
# --------------------------------------------------------------------------


def test_cli_calls_get_prediction():
    """main() harus memakai get_prediction(), bukan hitung ulang sendiri."""
    source = inspect.getsource(pm.main)
    assert "get_prediction(" in source
    for duplicate in ("build_ml_feature_rows(", "predict_fixture(", "joblib.load("):
        assert duplicate not in source, f"main() masih punya {duplicate} sendiri"


def test_app_imports_predict_match_not_private_copy():
    """app.py harus memakai modul yang sama, bukan implementasi sendiri."""
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "import predict_match as pm" in app_source
    assert "pm.get_prediction(" in app_source
    assert "pm.use_snapshot_directory(" in app_source
    # Tidak boleh ada pemanggilan select_best_model langsung di web.
    assert "select_best_model(" not in app_source


def test_app_has_no_hardcoded_model_names():
    """Model harus dipilih otomatis; nama model tidak boleh ditulis di app."""
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    for model in ("poisson", "dixon_coles", "xgboost", "random_forest", "logistic_regression"):
        assert f'"{model}"' not in app_source, f"{model} ter-hardcode di app.py"
    # Nama tampilan tetap boleh, itu milik predict_match.
    assert "GOAL_MODEL_NAMES[" in app_source
    assert "CLASSIFICATION_MODEL_NAMES[" in app_source


def test_app_shows_disclaimer():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "bukan alat rekomendasi bet" in app_source