"""Tests untuk date_based_time_splits di train_model."""

import pandas as pd

from config import N_SPLITS
from train_model import date_based_time_splits


def _frame(n_days=30, rows_per_day=2):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frames = []
    for day_number, date in enumerate(dates):
        for row in range(rows_per_day):
            frames.append({"date": date, "value": day_number * 10 + row})
    return pd.DataFrame(frames)


def test_yields_expected_folds_in_order():
    data = _frame()
    folds = list(date_based_time_splits(data))
    assert [fold for fold, _, _ in folds] == list(range(1, N_SPLITS + 1))


def test_train_dates_strictly_before_validation_dates():
    data = _frame()
    for _, train_idx, validation_idx in date_based_time_splits(data):
        train_max = data.iloc[train_idx]["date"].max()
        validation_min = data.iloc[validation_idx]["date"].min()
        assert train_max < validation_min


def test_same_date_rows_stay_on_one_side_of_the_fold():
    """Baris dengan tanggal sama tidak boleh terbelah antara train & validation."""
    data = _frame(n_days=24, rows_per_day=3)
    dates = set(data["date"])
    for _, train_idx, validation_idx in date_based_time_splits(data):
        assert not (dates & set(data.iloc[train_idx]["date"])) & set(
            data.iloc[validation_idx]["date"]
        )


def test_union_of_folds_covers_all_rows():
    data = _frame(n_days=20, rows_per_day=2)
    seen: set[int] = set()
    for _, train_idx, validation_idx in date_based_time_splits(data):
        seen.update(data.iloc[train_idx].index.tolist())
        seen.update(data.iloc[validation_idx].index.tolist())
    assert seen == set(data.index)
