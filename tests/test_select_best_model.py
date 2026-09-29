"""Tests untuk select_best_model di predict_match.

Setelah fix P0 (TASK-0002), select_best_model menggunakan cv_log_loss_mean
(CV metrics) sebagai primary selection metric — bukan test set RPS/log loss.
"""

import pandas as pd
import pytest

from predict_match import CLASSIFICATION_MODEL_CANDIDATES, select_best_model


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["model", "cv_log_loss_mean", "cv_accuracy_mean"],
    )


def test_selects_lowest_cv_log_loss():
    evaluation = _frame(
        [
            ("random_forest", 0.990, 0.540),
            ("logistic_regression", 0.980, 0.530),
            ("xgboost", 1.010, 0.535),
        ]
    )
    selected = select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)
    assert selected["model"] == "logistic_regression"


def test_accuracy_breaks_log_loss_tie():
    evaluation = _frame(
        [
            ("random_forest", 0.983, 0.542),
            ("logistic_regression", 0.983, 0.521),
            ("xgboost", 1.002, 0.529),
        ]
    )
    selected = select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)
    # Sama cv_log_loss_mean, tapi random_forest punya cv_accuracy_mean lebih tinggi
    assert selected["model"] == "random_forest"


def test_missing_candidate_raises_with_names():
    evaluation = _frame(
        [
            ("random_forest", 0.983, 0.542),
            ("xgboost", 1.002, 0.529),
        ]
    )
    with pytest.raises(ValueError, match="belum lengkap"):
        select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)


def test_nan_metric_raises():
    evaluation = _frame(
        [
            ("random_forest", 0.983, 0.542),
            ("logistic_regression", float("nan"), 0.521),
            ("xgboost", 1.002, 0.529),
        ]
    )
    with pytest.raises(ValueError, match="tidak lengkap"):
        select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)


def test_extra_models_outside_candidates_are_ignored():
    evaluation = _frame(
        [
            ("poisson", 0.900, 0.470),
            ("random_forest", 0.983, 0.542),
            ("logistic_regression", 0.990, 0.521),
            ("xgboost", 1.002, 0.529),
        ]
    )
    selected = select_best_model(evaluation, CLASSIFICATION_MODEL_CANDIDATES)
    assert selected["model"] == "random_forest"
