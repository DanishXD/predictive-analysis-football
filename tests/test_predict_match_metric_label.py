"""Tests untuk label metrik model selection di predict_match.py.

CLI sebelumnya selalu mencetak "CV log loss" untuk semua model, padahal
angka Track A di ``cv_model_selection.csv`` dihitung IN-SAMPLE (model
dievaluasi di data yang sama dengan data latihnya). Dua baris output itu
jadi terlihat setara padahal tidak sebanding, dan ``basis:`` yang
ikutdicetak justru mengatakannya. Test di sini mengunci agar label tidak
kembali jadi hardcode.
"""

import pandas as pd
import pytest

from predict_match import describe_selection_metric


def _selection(value: float, basis: str) -> pd.Series:
    return pd.Series({"cv_log_loss_mean": value, "selection_basis": basis})


def test_in_sample_basis_is_not_called_cv():
    """Angka Track A yang in-sample tidak boleh berlabel 'CV log loss'."""
    text = describe_selection_metric(
        _selection(0.983277, "train-only single split (no CV)")
    )
    assert "in-sample" in text
    assert "CV log loss" not in text
    assert "bukan out-of-fold" in text


def test_time_series_split_is_called_cv():
    text = describe_selection_metric(
        _selection(0.980369, "TimeSeriesSplit 5-fold CV")
    )
    assert text.startswith("CV log loss:")
    assert "in-sample" not in text


def test_walk_forward_basis_counts_as_out_of_sample():
    """Regression guard untuk basis yang tidak mengandung kata 'cv'.

    Basis 'walk-forward musiman, season training saja' itu out-of-fold.
    Heuristik berbasis kata 'cv' akan salah melabelinya in-sample.
    """
    text = describe_selection_metric(
        _selection(1.103723, "walk-forward musiman, season training saja")
    )
    assert text.startswith("CV log loss:")
    assert "in-sample" not in text


@pytest.mark.parametrize(
    "basis",
    [
        "train-only single split (no CV)",
        "no CV",
        "in-sample evaluation",
        "train-only",
        "tidak tersedia (model belum disimpan)",
    ],
)
def test_explicit_in_sample_markers(basis):
    text = describe_selection_metric(_selection(0.9, basis))
    assert "in-sample log loss:" in text
    assert "bukan out-of-fold" in text


def test_value_is_always_formatted_six_decimals():
    for value in (0.9832765, 1.0, 0.0):
        text = describe_selection_metric(
            _selection(value, "TimeSeriesSplit 5-fold CV")
        )
        assert f"{value:.6f}" in text


def test_basis_text_is_preserved():
    """Basis asli harus ikut ditampilkan, tidak diganti ringkasan."""
    basis = "TimeSeriesSplit 5-fold CV"
    text = describe_selection_metric(_selection(0.98, basis))
    assert basis in text


def test_real_track_a_and_track_b_differ_in_label():
    """Angka Track A dan Track B tidak boleh berlabel sama.

    Inilah regresi asli: kedua model berada di tabel yang sama dengan kolom
    cv_log_loss_mean yang sama, tapi keduanya dicetak dengan label identik.
    """
    track_a = describe_selection_metric(
        _selection(0.983277, "train-only single split (no CV)")
    )
    track_b = describe_selection_metric(
        _selection(0.980369, "TimeSeriesSplit 5-fold CV")
    )
    assert track_a != track_b
    assert "in-sample" in track_a and "CV log loss" in track_b


def test_function_handles_numpy_float():
    import numpy as np

    selection = pd.Series(
        {"cv_log_loss_mean": np.float64(0.5), "selection_basis": "no CV"}
    )
    text = describe_selection_metric(selection)
    assert "0.500000" in text
