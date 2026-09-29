"""Tests untuk eksperimen seasonal HFA (src/seasonal_hfa.py).

Fokus test: HFA memang berbeda secara statistik antar musim, tapi itu
tidak otomatis berarti worth implementing. Test ini mengunci kedua hal
supaya hasilnya tidak salah dibaca nanti.
"""

import numpy as np
import pandas as pd
import pytest

import seasonal_hfa
from config import EMPTY_STADIUM_SEASONS, SEASONS, TEST_SEASON
from track_a_cv import build_folds

TEST_FOLDS = [fold for fold in build_folds()]


def _synthetic_matches(seasons=None, teams=("A", "B", "C", "D")):
    """Buat tabel pertandingan deterministik untuk beberapa musim."""
    if seasons is None:
        seasons = [s for s in SEASONS if s != TEST_SEASON]
    rows = []
    day = pd.Timestamp("2016-08-01")
    for season in seasons:
        for index in range(len(teams)):
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "season": season,
                    "date": day + pd.Timedelta(days=index),
                    "datetime": day + pd.Timedelta(days=index),
                    "team_home": teams[index],
                    "team_away": teams[(index + 1) % len(teams)],
                    "goals_home": 2,
                    "goals_away": 1,
                    "result": "H",
                }
            )
        day = day + pd.Timedelta(days=300)
    return pd.DataFrame(rows)


def test_empty_stadium_seasons_do_not_include_test_season():
    """Pondasi seluruh eksperimen: test season harusnya bukan musim no-fans."""
    assert TEST_SEASON not in EMPTY_STADIUM_SEASONS


def test_model_with_hfa_does_not_mutate_original():
    matches = _synthetic_matches()
    model, hfa_normal, _ = seasonal_hfa.fit_with_seasonal_hfa(matches)
    before = float(model._params[-1])
    shifted = seasonal_hfa._model_with_hfa(model, hfa_normal + 0.5)
    assert float(model._params[-1]) == pytest.approx(before)
    assert float(shifted._params[-1]) == pytest.approx(hfa_normal + 0.5)
    assert shifted._params is not model._params


def test_model_with_hfa_actually_changes_probabilities():
    matches = _synthetic_matches()
    model, hfa_normal, _ = seasonal_hfa.fit_with_seasonal_hfa(matches)
    base = model.predict("A", "B", max_goals=15).home_draw_away
    shifted = seasonal_hfa._model_with_hfa(model, hfa_normal + 0.5)
    altered = shifted.predict("A", "B", max_goals=15).home_draw_away
    assert base[0] != pytest.approx(altered[0], abs=1e-6)


def test_refit_home_advantage_recovers_global_hfa():
    """Profile-HFA di seluruh data harus mendekati HFA dari fit penuh."""
    matches = _synthetic_matches()
    model, _, _ = seasonal_hfa.fit_with_seasonal_hfa(matches)
    weights = np.ones(len(matches), dtype=float)
    refit = seasonal_hfa.refit_home_advantage(model, matches, weights)
    assert refit == pytest.approx(float(model._params[-1]), abs=1e-3)


def test_evaluate_fold_skips_fold_without_no_fans_training():
    """Fold sebelum musim no-fans masuk training harus dilewati, bukan dipaksa."""
    matches = _synthetic_matches()
    validation = matches.loc[
        matches["season"] == TEST_FOLDS[0].validation_season
    ].copy()
    y_true = validation["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    row = seasonal_hfa.evaluate_fold(TEST_FOLDS[0], matches, validation, y_true)
    assert row["applicable"] is False
    assert "tidak ada musim no-fans" in row["skip_reason"]


def test_evaluate_fold_never_uses_test_season():
    matches = _synthetic_matches()
    matches = matches.loc[matches["season"] != TEST_SEASON]
    for fold in TEST_FOLDS[1:]:
        validation = matches.loc[
            matches["season"] == fold.validation_season
        ].copy()
        y_true = validation["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        row = seasonal_hfa.evaluate_fold(fold, matches, validation, y_true)
        assert row["validation_season"] != TEST_SEASON


def test_only_fold_validating_no_fans_has_no_no_fans_training():
    """Batasan struktural: satu-satunya fold no-fans tidak punya training no-fans.

    Season 2020/2021 jadi validation di fold 1, tapi season itu sendiri baru
    masuk training mulai fold 2. Jadi tidak ada satu pun fold yang bisa
    dipakai untuk menguji varian B (HFA khusus no-fans) secara out-of-sample.
    """
    sensitive = [
        fold for fold in TEST_FOLDS if fold.validation_season in EMPTY_STADIUM_SEASONS
    ]
    for fold in sensitive:
        training = [s for s in SEASONS if s <= fold.last_train_season]
        assert not set(training) & set(EMPTY_STADIUM_SEASONS), (
            f"Fold {fold.number} punya musim no-fans di training, jadi docstring "
            "dan batasan struktural eksperimen ini perlu diperbarui."
        )


def test_hfa_log_likelihood_peaks_at_mle():
    """Dengan rate away tetap, MLE HFA adalah log(rata-rata gol home)."""
    goals = np.array([2.0, 1.0, 3.0, 0.0])
    expected = np.ones(4)
    weights = np.ones(4)
    expected_mle = float(np.log(goals.mean()))
    grid = np.linspace(-1.0, 1.0, 401)
    values = [
        seasonal_hfa._hfa_log_likelihood(hfa, expected, expected, goals, goals, weights)
        for hfa in grid
    ]
    assert grid[int(np.argmax(values))] == pytest.approx(expected_mle, abs=0.01)


def test_hfa_log_likelihood_is_unimodal():
    expected = np.ones(20)
    goals = np.full(20, 2.0)
    weights = np.ones(20)
    grid = np.linspace(-0.8, 0.8, 81)
    values = np.array(
        [
            seasonal_hfa._hfa_log_likelihood(
                hfa, expected, expected, goals, goals, weights
            )
            for hfa in grid
        ]
    )
    peak = int(np.argmax(values))
    assert np.all(np.diff(values[: peak + 1]) >= 0)
    assert np.all(np.diff(values[peak:]) <= 0)


def test_probabilities_helper_selects_right_columns():
    frame = pd.DataFrame(
        {
            "prob_home": [0.5, 0.6],
            "prob_draw": [0.3, 0.2],
            "prob_away": [0.2, 0.2],
            "cold_start": [False, True],
        }
    )
    matrix = seasonal_hfa._probabilities(frame)
    assert matrix.shape == (2, 3)
    assert matrix[0].tolist() == [0.5, 0.3, 0.2]


def test_module_does_not_touch_production_artifacts():
    production = {
        "evaluation_summary.csv",
        "cv_model_selection.csv",
        "poisson_goal_model.pkl",
        "dixon_coles_goal_model.pkl",
    }
    written = {
        seasonal_hfa.DETAIL_PATH.name,
        seasonal_hfa.COMPARISON_PATH.name,
        seasonal_hfa.HFA_ESTIMATE_PATH.name,
        seasonal_hfa.METADATA_PATH.name,
    }
    assert not (written & production)
    assert seasonal_hfa.METADATA_PATH.name == "seasonal_hfa_summary.json"
