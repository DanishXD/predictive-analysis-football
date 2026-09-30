"""Tests untuk sambungan walk-forward Track A ke model selection.

Sebelum September 2026, ``evaluate._track_a_selection_metrics()`` me-load
model Poisson/Dixon-Coles/Elo yang **di-fit di data training**, lalu
menilainya di **data training yang sama**. Angka Track B di
``cv_model_selection.csv`` berasal dari TimeSeriesSplit out-of-fold. Jadi
kolom ``cv_log_loss_mean`` mencampur dua basis berbeda dalam satu tabel, dan
selisih 0.000002 antara Dixon-Coles dan Poisson yang menentukan goal model
sama sekali bukan bukti.

``src/track_a_cv.py`` sudah menghasilkan walk-forward CV per fold untuk Track
A. Test di sini mengunci bahwa angka itu yang dipakai, bahwa agregasinya
benar, dan terutama bahwa test season tidak pernah bocor ke sana.
"""

import numpy as np
import pandas as pd
import pytest

import evaluate
import track_a_cv
from evaluate import _assert_track_a_folds_current, _load_track_a_walk_forward

LAST_TRAIN = [
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
]

FOLD_RESULTS = pd.DataFrame(
    {
        "model": ["poisson"] * 5 + ["dixon_coles"] * 5 + ["elo"] * 5,
        "fold": list(range(1, 6)) * 3,
        "last_train_season": LAST_TRAIN * 3,
        "validation_season": [
            "2020-2021",
            "2021-2022",
            "2022-2023",
            "2023-2024",
            "2024-2025",
        ]
        * 3,
        "log_loss": [
            1.10,
            1.12,
            1.09,
            1.11,
            1.08,
            1.14,
            1.10,
            1.13,
            1.09,
            1.12,
            1.20,
            1.18,
            1.22,
            1.19,
            1.21,
        ],
        "accuracy": [
            0.44,
            0.46,
            0.43,
            0.45,
            0.47,
            0.43,
            0.45,
            0.42,
            0.44,
            0.46,
            0.46,
            0.44,
            0.47,
            0.45,
            0.46,
        ],
    }
)


@pytest.fixture
def results_file(tmp_path, monkeypatch):
    path = tmp_path / "track_a_cv_results.csv"
    FOLD_RESULTS.to_csv(path, index=False)
    monkeypatch.setattr(evaluate, "TRACK_A_CV_RESULTS_PATH", path)
    return path


# --------------------------------------------------------------------------
# Guard kontaminasi test season
# --------------------------------------------------------------------------


def test_test_season_in_fold_raises(results_file):
    contaminated = FOLD_RESULTS.copy()
    contaminated.loc[0, "validation_season"] = evaluate.TEST_SEASON
    contaminated.to_csv(results_file, index=False)
    with pytest.raises(ValueError, match="TEST_SEASON"):
        _load_track_a_walk_forward()


def test_any_model_contaminated_is_caught(results_file):
    contaminated = FOLD_RESULTS.copy()
    contaminated.loc[contaminated["model"] == "elo", "validation_season"] = (
        evaluate.TEST_SEASON
    )
    contaminated.to_csv(results_file, index=False)
    with pytest.raises(ValueError, match="elo"):
        _load_track_a_walk_forward()


def test_real_results_file_has_no_test_season():
    """Guard pada data hasil run_track_a_cv.py yang sebenarnya.

    Kalau file belum ada (track_a_cv.py belum dijalankan) test di-skip,
    bukan dianggap lulus.
    """
    if not evaluate.TRACK_A_CV_RESULTS_PATH.exists():
        pytest.skip("track_a_cv_results.csv belum ada; jalankan src/track_a_cv.py")
    results = pd.read_csv(evaluate.TRACK_A_CV_RESULTS_PATH)
    assert evaluate.TEST_SEASON not in set(results["validation_season"])


def test_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate, "TRACK_A_CV_RESULTS_PATH", tmp_path / "tidak_ada.csv"
    )
    assert _load_track_a_walk_forward() == {}


def test_missing_columns_returns_empty(results_file):
    FOLD_RESULTS.drop(columns=["log_loss"]).to_csv(results_file, index=False)
    assert _load_track_a_walk_forward() == {}


# --------------------------------------------------------------------------
# Agregasi
# --------------------------------------------------------------------------


def test_aggregates_match_pandas(results_file):
    rows = _load_track_a_walk_forward()
    assert set(rows) == {"poisson", "dixon_coles", "elo"}
    for model, group in FOLD_RESULTS.groupby("model"):
        row = rows[model]
        assert row["cv_log_loss_mean"] == pytest.approx(group["log_loss"].mean())
        assert row["cv_log_loss_std"] == pytest.approx(group["log_loss"].std())
        assert row["cv_accuracy_mean"] == pytest.approx(group["accuracy"].mean())
        assert row["cv_accuracy_std"] == pytest.approx(group["accuracy"].std())


def test_rows_have_expected_schema(results_file):
    rows = _load_track_a_walk_forward()
    for row in rows.values():
        assert row["track"] == "Track A"
        assert row["model"] in {"poisson", "dixon_coles", "elo"}
        assert np.isnan(row["draw_recall"])


def test_selection_basis_declares_out_of_sample(results_file):
    for row in _load_track_a_walk_forward().values():
        basis = row["selection_basis"]
        assert "walk-forward" in basis
        assert "out-of-sample" in basis
        assert "5 fold" in basis
        assert "in-sample" not in basis.replace("out-of-sample", "")


def test_std_is_populated(results_file):
    """In-sample lama selalu NaN di kolom std; walk-forward punya std."""
    rows = _load_track_a_walk_forward()
    for row in rows.values():
        assert not np.isnan(row["cv_log_loss_std"])
        assert not np.isnan(row["cv_accuracy_std"])


# --------------------------------------------------------------------------
# Konsekuensi pada pemilihan goal model
# --------------------------------------------------------------------------


def test_goal_model_flip_is_visible_in_cv_numbers():
    """Angka CV resmi: Poisson mengalahkan Dixon-Coles.

    Ini flip yang dilaporkan ke user. Di angka in-sample lama urutannya
    terbalik (Dixon-Coles 0.983277 vs Poisson 0.983279, selisih 0.000002).
    Test ini mengunci angka walk-forward yang sekarang jadi dasar selection
    supaya flip tidak hilang tanpa jejak.
    """
    results = pd.read_csv(evaluate.TRACK_A_CV_RESULTS_PATH)
    means = results.groupby("model")["log_loss"].mean()
    assert means["poisson"] < means["dixon_coles"]
    assert means["poisson"] < means["elo"]
    gap = means["dixon_coles"] - means["poisson"]
    assert gap > 0, "selisih harus positif: Dixon-Coles lebih buruk"
    # Jauh lebih besar dari noise in-sample yang pernah menentukan winner.
    assert gap > 0.001


def test_full_track_a_ranking_is_locked():
    """Urutan lengkap Track A harus dijaga: Poisson, Dixon-Coles, Elo."""
    results = pd.read_csv(evaluate.TRACK_A_CV_RESULTS_PATH)
    means = results.groupby("model")["log_loss"].mean().sort_values()
    assert list(means.index) == ["poisson", "dixon_coles", "elo"]


# --------------------------------------------------------------------------
# Guard staleness
# --------------------------------------------------------------------------


def _expected_folds() -> set[tuple[str, str]]:
    return {
        (fold.last_train_season, fold.validation_season)
        for fold in track_a_cv.build_folds()
    }


def test_current_file_matches_config_folds():
    results = pd.read_csv(evaluate.TRACK_A_CV_RESULTS_PATH)
    assert _assert_track_a_folds_current(results) is None


def test_stale_fold_count_is_rejected(results_file):
    """Kasus utama: TEST_SEASON bergeser, jumlah fold berubah."""
    results_file.write_text("", encoding="utf-8")  # pastikan ada file
    stale = FOLD_RESULTS[FOLD_RESULTS["validation_season"] != "2024-2025"]
    stale.to_csv(results_file, index=False)
    with pytest.raises(ValueError, match="tidak cocok dengan config"):
        _assert_track_a_folds_current(stale)


def test_extra_fold_is_rejected(results_file):
    """Fold nyasar di file tapi tidak lagi diharapkan ikut ditolak."""
    extra = pd.concat(
        [
            FOLD_RESULTS,
            pd.DataFrame(
                {
                    "model": ["poisson"],
                    "fold": [6],
                    "last_train_season": ["2024-2025"],
                    "validation_season": ["2025-2026"],
                    "log_loss": [1.05],
                    "accuracy": [0.50],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="tidak cocok dengan config"):
        _assert_track_a_folds_current(extra)


def test_staleness_guard_fires_when_old_guard_would_not(monkeypatch):
    """Guard baru harus menutup gap yang guard test-season tidak trembus.

    Kalau hanya ``config.SEASONS`` berubah (musim ditambah atau dipotong)
    tanpa mengubah ``TEST_SEASON``, tidak ada baris dengan
    ``validation_season == TEST_SEASON`` sehingga guard pertama diam. Tapi
    angka di file jadi dihitung dari training period yang salah.
    """
    stale = FOLD_RESULTS.copy()
    original = list(track_a_cv.SEASONS)
    try:
        track_a_cv.SEASONS = sorted(set(original) - {"2016-2017"})
        assert not (stale["validation_season"] == track_a_cv.TEST_SEASON).any()
        with pytest.raises(ValueError, match="tidak cocok dengan config"):
            _assert_track_a_folds_current(stale)
    finally:
        track_a_cv.SEASONS = original


def test_error_message_names_the_fix():
    stale = FOLD_RESULTS[FOLD_RESULTS["validation_season"] != "2024-2025"]
    with pytest.raises(ValueError) as excinfo:
        _assert_track_a_folds_current(stale)
    message = str(excinfo.value)
    assert "src\\track_a_cv.py" in message
    assert "TEST_SEASON=" in message
    # Pesan harus menyebut fold mana yang stale dan mana yang hilang.
    assert "2024-2025" in message