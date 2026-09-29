"""Tests untuk src/model_tuning.py (adapter cv & custom RPS scorer)."""

import numpy as np
import pandas as pd
import pytest

from model_tuning import (
    DateTimeSeriesSplit,
    make_rps_scorer,
    rps_scorer,
    single_threaded,
)
from train_model import build_models


def test_rps_scorer_matches_penaltyblog_average():
    """Custom scorer harus identik dengan pb.metrics.rps_average."""
    import penaltyblog as pb

    probs = np.array([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    codes = np.array([0, 2, 1])
    assert rps_scorer(codes, probs) == pytest.approx(
        float(pb.metrics.rps_average(probs, codes))
    )


def test_rps_scorer_equals_mean_of_per_sample_rps():
    import penaltyblog as pb

    rng = np.random.default_rng(7)
    probs = rng.dirichlet([1, 1, 1], size=25)
    codes = rng.integers(0, 3, size=25)
    per_sample = [
        float(pb.metrics.rps_average(probs[i : i + 1], codes[i : i + 1]))
        for i in range(25)
    ]
    assert rps_scorer(codes, probs) == pytest.approx(float(np.mean(per_sample)))


def test_make_rps_scorer_is_lower_is_better():
    scorer = make_rps_scorer()
    assert scorer._sign == -1


def test_date_time_series_split_is_forward_only():
    dates = pd.date_range("2023-01-01", periods=40, freq="D").to_numpy()
    splitter = DateTimeSeriesSplit(dates, n_splits=4)
    folds = list(splitter.split())
    assert len(folds) == 4
    for train_idx, validation_idx in folds:
        assert dates[train_idx].max() < dates[validation_idx].min()


def test_date_time_series_split_never_splits_one_date():
    """Semua baris pada tanggal yang sama harus satu sisi."""
    dates = pd.to_datetime(
        ["2023-01-01"] * 3 + ["2023-01-02"] * 3 + [f"2023-01-{d:02d}" for d in range(3, 25)]
    ).to_numpy()
    splitter = DateTimeSeriesSplit(dates, n_splits=3)
    for train_idx, validation_idx in splitter.split():
        train_dates = set(dates[train_idx])
        validation_dates = set(dates[validation_idx])
        assert not (train_dates & validation_dates)


def test_date_time_series_split_get_n_splits():
    dates = pd.date_range("2023-01-01", periods=30, freq="D").to_numpy()
    assert DateTimeSeriesSplit(dates, n_splits=5).get_n_splits() == 5


def test_single_threaded_disables_nested_parallelism():
    """build_models() pakai n_jobs=-1; harus dimatikan saat GridSearchCV paralel."""
    for name in ("random_forest", "xgboost"):
        model = single_threaded(build_models()[name])
        assert model.named_steps["classifier"].n_jobs == 1


def test_single_threaded_returns_a_clone():
    original = build_models()["random_forest"]
    tuned = single_threaded(original)
    assert tuned is not original
    assert original.named_steps["classifier"].n_jobs == -1
