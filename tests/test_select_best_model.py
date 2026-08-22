"""Tests untuk select_best_model di predict_match."""

import pandas as pd
import pytest

from predict_match import CLASSIFICATION_MODEL_CANDIDATES, select_best_model


def _frame(rows):
    return pd.DataFrame(rows, columns=["model", "rps", "log_loss"])


def test_selects_lowest_rps():
    evaluation = _frame(
        [
            ("random_forest", 0.220, 1.040),
            ("logistic_regression", 0.210, 1.050),
            ("xgboost", 0.230, 1.030),
        ]
    )
    selected = select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)
    assert selected["model"] == "logistic_regression"


def test_log_loss_breaks_rps_tie():
    evaluation = _frame(
        [
            ("random_forest", 0.210, 1.020),
            ("logistic_regression", 0.210, 1.010),
            ("xgboost", 0.250, 1.000),
        ]
    )
    selected = select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)
    assert selected["model"] == "logistic_regression"


def test_missing_candidate_raises_with_names():
    evaluation = _frame(
        [
            ("random_forest", 0.210, 1.020),
            ("xgboost", 0.250, 1.000),
        ]
    )
    with pytest.raises(ValueError, match="belum lengkap"):
        select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)


def test_nan_metric_raises():
    evaluation = _frame(
        [
            ("random_forest", 0.210, 1.020),
            ("logistic_regression", float("nan"), 1.050),
            ("xgboost", 0.250, 1.000),
        ]
    )
    with pytest.raises(ValueError, match="tidak lengkap"):
        select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)


def test_extra_models_outside_candidates_are_ignored():
    evaluation = _frame(
        [
            ("elo", 0.100, 0.900),
            ("random_forest", 0.220, 1.040),
            ("logistic_regression", 0.210, 1.050),
            ("xgboost", 0.230, 1.030),
        ]
    )
    selected = select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)
    assert selected["model"] == "logistic_regression"
