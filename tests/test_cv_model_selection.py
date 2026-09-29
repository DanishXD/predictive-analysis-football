"""Tests untuk ekspor cv_model_selection.csv di evaluate.py.

Regression guard: predict_match.py membaca file ini untuk model selection.
Dulu file itu dibuat manual sekali sehingga diam-diam basi setiap pipeline
re-run. Test ini memastikan evaluate.py menghasilkannya dari data training.
"""

import numpy as np
import pandas as pd
import pytest

import evaluate
from config import TEST_SEASON
from evaluate import export_cv_model_selection

COLUMNS = [
    "model",
    "track",
    "cv_log_loss_mean",
    "cv_log_loss_std",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "selection_basis",
    "draw_recall",
]


def test_export_writes_expected_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CV_SELECTION_PATH", tmp_path / "cv.csv")
    table = export_cv_model_selection({})
    assert list(table.columns) == COLUMNS
    assert (tmp_path / "cv.csv").exists()


def test_track_b_comes_from_ml_metrics(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "cv_log_loss_mean": 0.98,
                "cv_log_loss_std": 0.02,
                "cv_accuracy_mean": 0.54,
                "cv_accuracy_std": 0.01,
            },
            {
                "model": "xgboost",
                "cv_log_loss_mean": 1.00,
                "cv_log_loss_std": 0.02,
                "cv_accuracy_mean": 0.53,
                "cv_accuracy_std": 0.01,
            },
        ]
    )
    metrics.to_csv(tmp_path / "ml_model_metrics.csv", index=False)
    monkeypatch.setattr(evaluate, "ML_METRICS_PATH", tmp_path / "ml_model_metrics.csv")
    monkeypatch.setattr(evaluate, "CV_SELECTION_PATH", tmp_path / "cv.csv")
    monkeypatch.setattr(evaluate, "_track_a_selection_metrics", lambda: pd.DataFrame(
        [{"model": "poisson", "track": "Track A", "cv_log_loss_mean": np.nan,
          "cv_log_loss_std": np.nan, "cv_accuracy_mean": np.nan,
          "cv_accuracy_std": np.nan, "selection_basis": "na", "draw_recall": np.nan}]
    ))

    table = export_cv_model_selection({"random_forest": 0.0})
    rf = table.loc[table["model"] == "random_forest"].iloc[0]
    assert rf["track"] == "Track B"
    assert rf["cv_log_loss_mean"] == pytest.approx(0.98)
    assert "CV" in rf["selection_basis"]
    assert rf["draw_recall"] == pytest.approx(0.0)


def test_missing_ml_metrics_does_not_crash(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "ML_METRICS_PATH", tmp_path / "tidak_ada.csv")
    monkeypatch.setattr(evaluate, "CV_SELECTION_PATH", tmp_path / "cv.csv")
    monkeypatch.setattr(evaluate, "_track_a_selection_metrics", lambda: pd.DataFrame(
        [{"model": "poisson", "track": "Track A", "cv_log_loss_mean": 1.0,
          "cv_log_loss_std": np.nan, "cv_accuracy_mean": 0.5,
          "cv_accuracy_std": np.nan, "selection_basis": "na", "draw_recall": np.nan}]
    ))
    table = export_cv_model_selection({})
    assert list(table["model"]) == ["poisson"]
    assert "tidak_ada.csv" in capsys.readouterr().out


def test_track_a_uses_train_only_not_test_metrics(tmp_path, monkeypatch):
    """Pastikan metrik Track A tidak diambil dari test set."""
    train = evaluate._train_rows()
    assert (train["season"] != TEST_SEASON).all(), "sumber metrik Track A tidak boleh test"
    assert len(train) > 0
    # kolom yang dibutuhkan untuk menghitung log loss harus ada
    assert train["result"].isin(["H", "D", "A"]).all()


def test_elo_probability_helper_returns_valid_distribution():
    row = pd.Series(
        {"team": "Arsenal", "opponent": "Chelsea",
         "elo_pre": 1600.0, "opponent_elo_pre": 1500.0}
    )
    home, draw, away = evaluate._elo_row_probabilities(row)
    assert home + draw + away == pytest.approx(1.0, abs=1e-9)
    assert home > away, "kandang harus lebih unggul saat rating lebih tinggi"
