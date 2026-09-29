"""Tests untuk src/model_calibration.py (tanpa jaringan, data sintetis)."""

import numpy as np
import pandas as pd
import pytest

from config import TARGET_MAPPING
from model_calibration import (
    DRAW_PRIOR_GRID,
    apply_draw_prior,
    draw_recall,
    expected_calibration_error,
    normalize_probabilities,
    select_draw_prior,
    summarise,
)


def _probs(rows):
    return np.array(rows, dtype=float)


def test_normalize_makes_rows_sum_to_one():
    raw = _probs([[0.6, 0.25, 0.15], [0.2, 0.2, 0.2]])
    normalized = normalize_probabilities(raw)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert normalized[0, 0] == pytest.approx(0.6)  # baris ini sudah 1.0


def test_normalize_rejects_all_zero_row():
    with pytest.raises(ValueError, match="nol total"):
        normalize_probabilities(_probs([[0.0, 0.0, 0.0]]))


def test_apply_draw_prior_scales_only_draw_and_renormalizes():
    raw = _probs([[0.5, 0.2, 0.3]])
    adjusted = apply_draw_prior(raw, 2.0)
    assert adjusted.sum() == pytest.approx(1.0)
    # P(draw) naik...
    assert adjusted[0, 1] > raw[0, 1]
    # ...sementara P(home) & P(away) turun oleh faktor yang sama (renormalisasi)
    assert adjusted[0, 0] / raw[0, 0] == pytest.approx(adjusted[0, 2] / raw[0, 2])
    # dan tidak ada kelas lain yang bertambah
    assert adjusted[0, 0] < raw[0, 0]
    assert adjusted[0, 2] < raw[0, 2]


def test_apply_draw_prior_factor_one_is_noop():
    raw = _probs([[0.5, 0.2, 0.3]])
    assert np.allclose(apply_draw_prior(raw, 1.0), raw)


def test_draw_recall_counts_correct_draw_predictions():
    # y_true: 2 draw, 2 home ; pred: tepat 1 draw
    probs = _probs([
        [0.2, 0.5, 0.3],   # pred draw, true draw
        [0.2, 0.5, 0.3],   # pred draw, true home
        [0.6, 0.2, 0.2],   # pred home, true draw
        [0.6, 0.2, 0.2],   # pred home, true home
    ])
    codes = [TARGET_MAPPING["D"], 0, TARGET_MAPPING["D"], 0]
    assert draw_recall(probs, np.array(codes)) == pytest.approx(0.5)


def test_draw_recall_is_nan_without_actual_draws():
    probs = _probs([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]])
    assert np.isnan(draw_recall(probs, np.array([0, 0])))


def test_expected_calibration_error_perfect_and_worst():
    perfect = _probs([[1.0, 0.0, 0.0]] * 5)
    assert expected_calibration_error(perfect, np.zeros(5, dtype=int)) == pytest.approx(0.0)
    # seluruh probabilitas seragam tapi semua benar -> terlaluunder-confident
    flat = _probs([[1 / 3, 1 / 3, 1 / 3]] * 5)
    assert expected_calibration_error(flat, np.zeros(5, dtype=int)) > 0.0


def test_summarise_returns_all_required_keys():
    probs = _probs([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]])
    result = summarise(probs, np.array([0, 2]))
    for key in ("rps", "log_loss", "accuracy", "draw_recall",
                "draw_predictions", "mean_ece", "mean_pred_draw"):
        assert key in result
    assert result["draw_predictions"] == 0
    assert 0.0 <= result["rps"] <= 1.0


def test_select_draw_prior_picks_minimum_rps_on_given_oof():
    """Faktor dipilih dari data OOF, bukan dari test."""
    # Bangun prediksi yang Draw-nya terlalu rendah -> RPS membaik saat dinaikkan.
    rng = np.random.default_rng(0)
    n = 400
    codes = rng.choice([0, 1, 2], size=n, p=[0.45, 0.30, 0.25])
    probs = np.zeros((n, 3))
    probs[np.arange(n), codes] = 0.6
    probs += 0.13
    probs = probs / probs.sum(axis=1, keepdims=True)
    oof = {i: probs[i] for i in range(n)}
    factor, table = select_draw_prior({"raw": probs}, oof, pd.Series(codes))
    assert table.loc[0, "rps"] == table["rps"].min()
    assert float(DRAW_PRIOR_GRID.min()) <= factor <= float(DRAW_PRIOR_GRID.max())
    assert len(table) == len(DRAW_PRIOR_GRID)
