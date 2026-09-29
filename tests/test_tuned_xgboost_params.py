"""Tests untuk penerapan hyperparameter XGBoost hasil tuning di Fase 5."""

import pytest

from config import XGBOOST_TUNED_PARAMS
from train_model import build_models


def test_build_models_uses_tuned_xgboost_params():
    """Fase 5 harus memakai setting dari config, bukan nilai lama yang hardcode."""
    classifier = build_models()["xgboost"].named_steps["classifier"]
    assert classifier.n_estimators == XGBOOST_TUNED_PARAMS["n_estimators"]
    assert classifier.learning_rate == pytest.approx(XGBOOST_TUNED_PARAMS["learning_rate"])
    assert classifier.max_depth == XGBOOST_TUNED_PARAMS["max_depth"]


def test_tuned_params_are_not_the_old_hardcoded_values():
    """Nilai lama (300/0.03/3) harus benar-benar berubah; kalau tidak ada
    perubahan berarti tuning tidak benar-benar diterapkan."""
    assert XGBOOST_TUNED_PARAMS["n_estimators"] != 300
    assert XGBOOST_TUNED_PARAMS["learning_rate"] != 0.03
    assert XGBOOST_TUNED_PARAMS["max_depth"] != 3


def test_tuned_params_come_from_phase19_search_space():
    """Nilai tuned harus berada di dalam search space Phase 19."""
    from model_tuning import XGBOOST_GRID

    assert XGBOOST_TUNED_PARAMS["n_estimators"] in XGBOOST_GRID["classifier__n_estimators"]
    assert XGBOOST_TUNED_PARAMS["learning_rate"] in XGBOOST_GRID["classifier__learning_rate"]
    assert XGBOOST_TUNED_PARAMS["max_depth"] in XGBOOST_GRID["classifier__max_depth"]


def test_other_models_untouched_by_tuning():
    """Tuning XGBoost tidak boleh mengubah model lain."""
    rf = build_models()["random_forest"].named_steps["classifier"]
    assert rf.n_estimators == 350
    assert rf.max_depth == 8
    assert rf.min_samples_leaf == 6


def test_build_models_returns_fresh_instances():
    """Setiap pemanggilan harus memberi objek baru (tidak boleh berbagi instance)."""
    a = build_models()["xgboost"].named_steps["classifier"]
    b = build_models()["xgboost"].named_steps["classifier"]
    assert a is not b
