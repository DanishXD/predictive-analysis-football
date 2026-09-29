"""Tests untuk walk-forward CV Track A (src/track_a_cv.py).

Fokus utama test ini adalah aturan anti-leakage, karena angka Track A
sebelumnya in-sample dan gap itu justru yang ingin ditutup modul ini.
"""

import numpy as np
import pandas as pd
import pytest

import track_a_cv
from config import SEASONS, TEST_SEASON
from track_a_cv import build_folds, compare_models, validate_folds


def test_folds_never_use_test_season():
    for fold in build_folds():
        assert fold.validation_season != TEST_SEASON
        assert fold.last_train_season != TEST_SEASON


def test_training_always_precedes_validation():
    for fold in build_folds():
        assert fold.last_train_season < fold.validation_season


def test_folds_are_in_ascending_season_order():
    folds = build_folds()
    positions = [SEASONS.index(fold.validation_season) for fold in folds]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions), "season validasi harus unik"


def test_folds_cover_all_trainable_seasons_after_minimum():
    folds = build_folds()
    train_seasons = [season for season in SEASONS if season != TEST_SEASON]
    expected = train_seasons[track_a_cv.MIN_TRAIN_SEASONS :]
    assert [fold.validation_season for fold in folds] == expected


def test_validate_folds_accepts_generated_folds():
    validate_folds(build_folds())


def test_validate_folds_rejects_test_season():
    folds = build_folds()
    folds[0] = track_a_cv.Fold(
        number=99,
        last_train_season="2024-2025",
        validation_season=TEST_SEASON,
    )
    with pytest.raises(ValueError, match="TEST_SEASON"):
        validate_folds(folds)


def test_validate_folds_rejects_inverted_split():
    folds = build_folds()
    folds[0] = track_a_cv.Fold(
        number=99,
        last_train_season="2023-2024",
        validation_season="2016-2017",
    )
    with pytest.raises(ValueError, match="tidak boleh >="):
        validate_folds(folds)


def test_validate_folds_rejects_duplicate_validation_season():
    folds = build_folds()
    duplicate = folds[1]
    folds[0] = track_a_cv.Fold(
        number=99,
        last_train_season="2020-2021",
        validation_season=duplicate.validation_season,
    )
    with pytest.raises(ValueError, match="lebih dari sekali"):
        validate_folds(folds)


def test_validate_folds_rejects_short_training():
    folds = build_folds()
    folds[0] = track_a_cv.Fold(
        number=99, last_train_season="2016-2017", validation_season="2017-2018"
    )
    with pytest.raises(ValueError, match="terlalu pendek"):
        validate_folds(folds)


def _synthetic_matches():
    rows = []
    day = pd.Timestamp("2016-08-01")
    teams = ("A", "B", "C", "D")
    for season in [s for s in SEASONS if s != TEST_SEASON]:
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


def test_evaluate_fold_uses_no_future_data():
    """Nilai fold tidak boleh berubah kalau data masa depan dihapus."""
    matches = _synthetic_matches()
    fold = build_folds()[-1]

    truncated = matches.loc[
        matches["season"].isin(
            [s for s in SEASONS if s <= fold.last_train_season]
            + [fold.validation_season]
        )
    ].copy()

    rows_full = track_a_cv.evaluate_fold(
        fold, matches, _codes(matches, fold.validation_season), _validation(matches, fold)
    )
    rows_truncated = track_a_cv.evaluate_fold(
        fold, truncated, _codes(truncated, fold.validation_season),
        _validation(truncated, fold),
    )
    for full, cut in zip(rows_full, rows_truncated):
        assert full["model"] == cut["model"]
        assert full["log_loss"] == pytest.approx(cut["log_loss"])
        assert full["rps"] == pytest.approx(cut["rps"])


def _validation(matches, fold):
    return matches.loc[matches["season"] == fold.validation_season].copy()


def _codes(matches, season):
    from config import TARGET_MAPPING

    return matches.loc[matches["season"] == season, "result"].map(TARGET_MAPPING).to_numpy()


def test_elo_probabilities_sum_to_one():
    matches = _synthetic_matches()
    fold = build_folds()[-1]
    train = matches.loc[matches["season"] <= fold.last_train_season]
    validation = _validation(matches, fold)
    frame = track_a_cv.elo_probabilities_for(train, validation)
    total = frame[["prob_home", "prob_draw", "prob_away"]].sum(axis=1).to_numpy()
    assert np.allclose(total, 1.0, atol=1e-9)


def test_elo_probabilities_are_proper_distributions():
    """Probabilitas harus strictly antara 0 dan 1, tidak degenerate."""
    matches = _synthetic_matches()
    fold = build_folds()[-1]
    train = matches.loc[matches["season"] <= fold.last_train_season]
    validation = _validation(matches, fold)
    frame = track_a_cv.elo_probabilities_for(train, validation)
    assert (frame["prob_home"] > 0).all()
    assert (frame["prob_home"] < 1).all()


def test_elo_is_deterministic_across_runs():
    matches = _synthetic_matches()
    fold = build_folds()[-1]
    train = matches.loc[matches["season"] <= fold.last_train_season]
    validation = _validation(matches, fold)
    first = track_a_cv.elo_probabilities_for(train, validation)
    second = track_a_cv.elo_probabilities_for(train, validation)
    pd.testing.assert_frame_equal(first, second)


def test_compare_models_is_paired_per_fold():
    results = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "model": ["poisson", "dixon_coles", "poisson", "dixon_coles"],
            "log_loss": [1.0, 1.1, 1.2, 1.15],
            "rps": [0.2, 0.21, 0.22, 0.215],
        }
    )
    comparison = compare_models(results)
    row = comparison.iloc[0]
    assert row["model"] == "dixon_coles"
    assert row["n_folds"] == 2
    assert row["delta_log_loss_mean"] == pytest.approx(0.025)
    assert row["folds_better"] == 1


def test_compare_models_excludes_baseline_row():
    results = pd.DataFrame(
        {
            "fold": [1, 1],
            "model": ["poisson", "dixon_coles"],
            "log_loss": [1.0, 1.1],
            "rps": [0.2, 0.21],
        }
    )
    assert "poisson" not in set(compare_models(results)["model"])


def test_module_does_not_touch_production_artifacts():
    """Modul eksperimen tidak boleh menulis path produksi."""
    production_paths = {
        "evaluation_summary.csv",
        "cv_model_selection.csv",
        "ml_model_metrics.csv",
        "test_match_probabilities.csv",
    }
    written = {
        track_a_cv.RESULTS_PATH.name,
        track_a_cv.SUMMARY_PATH.name,
        track_a_cv.COMPARISON_PATH.name,
        track_a_cv.METADATA_PATH.name,
    }
    assert not (written & production_paths)
    assert track_a_cv.METADATA_PATH.name == "track_a_cv_summary.json"
