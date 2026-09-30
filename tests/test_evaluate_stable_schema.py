"""Tests untuk skema stabil evaluation_summary.csv.

Sebelum ini, ``evaluation_summary.csv`` hanya punya baris untuk model yang
prediksinya ada. Karena ``random_forest_stacked_xg`` bergantung pada file
opsional ``stacking_test_predictions.csv``, jumlah barisnya berubah dari 7
jadi 8 tergantung skrip mana yang pernah dijalankan. Artefak "canonical"
Fase 7 jadi tidak reproducible di mesin berbeda tanpa error apa pun.

Test di sini mengunci skema: SELALU satu baris per model di MODEL_ORDER,
dengan flag ``available`` dan NaN untuk model yang belum punya prediksi.
"""

import numpy as np
import pandas as pd
import pytest

import evaluate
from evaluate import (
    MODEL_ORDER,
    MODEL_TRACKS,
    _unavailable_confusion_row,
    _unavailable_class_rows,
    _unavailable_summary_row,
)


def _fake_predictions(models, n_matches=12):
    rng = np.random.default_rng(3)
    rows = []
    for index in range(n_matches):
        for model in models:
            raw = rng.uniform(0.1, 1.0, 3)
            prob = raw / raw.sum()
            rows.append(
                {
                    "match_id": f"m{index}",
                    "result": "HDAA"[index % 4],
                    "predicted_result": "HDAA"[index % 4],
                    "prob_home": prob[0],
                    "prob_draw": prob[1],
                    "prob_away": prob[2],
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Baris tidak-tersedia
# --------------------------------------------------------------------------


def test_unavailable_summary_row_shape():
    row = _unavailable_summary_row("random_forest_stacked_xg")
    assert row["model"] == "random_forest_stacked_xg"
    assert row["available"] is False
    assert row["unavailable_reason"]
    for metric in ("accuracy", "log_loss", "brier_score", "rps", "mean_ece"):
        assert np.isnan(row[metric]), f"{metric} harus NaN"


def test_unavailable_class_rows_has_three_classes():
    rows = _unavailable_class_rows("poisson")
    assert len(rows) == 3
    assert {r["class"] for r in rows} == set(evaluate.CLASS_NAMES)
    assert all(np.isnan(r["precision"]) for r in rows)


def test_unavailable_confusion_row_has_nine_cells():
    rows = _unavailable_confusion_row("poisson")
    assert len(rows) == 9
    assert all(r["count"] == 0 for r in rows)


# --------------------------------------------------------------------------
# Skema selalu lengkap
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", MODEL_ORDER)
def test_summary_always_has_every_model(missing):
    """Model mana pun yang hilang, summary tetap 8 baris dengan 11 kolom."""
    present = [m for m in MODEL_ORDER if m != missing]
    predictions = _fake_predictions(present)
    summary, class_metrics, confusion = evaluate.evaluate_predictions(predictions)

    assert len(summary) == len(MODEL_ORDER)
    assert set(summary["model"]) == set(MODEL_ORDER)
    assert list(summary.columns) == list(
        _unavailable_summary_row(MODEL_ORDER[0]).keys()
    )
    assert len(class_metrics) == len(MODEL_ORDER) * 3
    assert len(confusion) == len(MODEL_ORDER) * 9

    row = summary.loc[summary["model"] == missing].iloc[0]
    assert bool(row["available"]) is False
    assert np.isnan(row["rps"])
    others = summary.loc[summary["available"]]
    assert len(others) == len(MODEL_ORDER) - 1
    assert others["rps"].notna().all()


def test_all_models_present_marks_all_available():
    predictions = _fake_predictions(MODEL_ORDER)
    summary, _, _ = evaluate.evaluate_predictions(predictions)
    assert len(summary) == len(MODEL_ORDER)
    assert summary["available"].all()
    assert summary["rps"].notna().all()


def test_row_count_does_not_depend_on_optional_files():
    """Regression guard langsung untuk bug aslinya.

    Jumlah baris harus sama tidak peduli model mana yang tersedia.
    """
    counts = set()
    for missing in (None, "random_forest_stacked_xg", "elo", "poisson"):
        present = [m for m in MODEL_ORDER if m != missing] if missing else list(MODEL_ORDER)
        summary, class_metrics, _ = evaluate.evaluate_predictions(
            _fake_predictions(present)
        )
        counts.add((len(summary), len(class_metrics)))
    assert len(counts) == 1, f"jumlah baris berubah-ubah: {counts}"


def test_columns_are_identical_regardless_of_availability():
    reference = None
    for missing in (None, "random_forest_stacked_xg", "xgboost"):
        present = [m for m in MODEL_ORDER if m != missing] if missing else list(MODEL_ORDER)
        summary, _, _ = evaluate.evaluate_predictions(_fake_predictions(present))
        columns = tuple(summary.columns)
        if reference is None:
            reference = columns
        assert columns == reference


def test_available_models_sorted_before_unavailable():
    predictions = _fake_predictions([m for m in MODEL_ORDER if m != "poisson"])
    summary, _, _ = evaluate.evaluate_predictions(predictions)
    flags = summary["available"].tolist()
    assert flags == sorted(flags, reverse=True), "available harus di atas"


def test_track_column_populated_for_every_model():
    predictions = _fake_predictions([m for m in MODEL_ORDER if m != "elo"])
    summary, class_metrics, confusion = evaluate.evaluate_predictions(predictions)
    for model in MODEL_ORDER:
        assert summary.loc[summary["model"] == model, "track"].iloc[0] == MODEL_TRACKS[model]
    assert class_metrics["track"].notna().all()
    assert confusion["track"].notna().all() if "track" in confusion.columns else True


# --------------------------------------------------------------------------
# validate_outputs mengikuti skema baru
# --------------------------------------------------------------------------


def test_validate_outputs_accepts_stable_schema():
    base = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(12)],
            "result": ["H", "D", "A"] * 4,
        }
    )
    predictions = _fake_predictions([m for m in MODEL_ORDER if m != "poisson"])
    predictions["track"] = predictions["model"].map(MODEL_TRACKS)
    summary, class_metrics, confusion = evaluate.evaluate_predictions(predictions)
    evaluate.validate_outputs(base, predictions, summary, class_metrics)


def test_validate_outputs_rejects_missing_row():
    base = pd.DataFrame(
        {"match_id": [f"m{i}" for i in range(12)], "result": ["H", "D", "A"] * 4}
    )
    predictions = _fake_predictions(list(MODEL_ORDER))
    predictions["track"] = predictions["model"].map(MODEL_TRACKS)
    summary, class_metrics, _ = evaluate.evaluate_predictions(predictions)
    summary = summary.loc[summary["model"] != "poisson"]
    with pytest.raises(ValueError, match="skema summary"):
        evaluate.validate_outputs(base, predictions, summary, class_metrics)


def test_validate_outputs_rejects_available_row_with_nan():
    base = pd.DataFrame(
        {"match_id": [f"m{i}" for i in range(12)], "result": ["H", "D", "A"] * 4}
    )
    predictions = _fake_predictions(list(MODEL_ORDER))
    predictions["track"] = predictions["model"].map(MODEL_TRACKS)
    summary, class_metrics, _ = evaluate.evaluate_predictions(predictions)
    summary.loc[summary["model"] == "elo", "rps"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        evaluate.validate_outputs(base, predictions, summary, class_metrics)


def test_validate_outputs_rejects_unavailable_row_with_value():
    base = pd.DataFrame(
        {"match_id": [f"m{i}" for i in range(12)], "result": ["H", "D", "A"] * 4}
    )
    predictions = _fake_predictions([m for m in MODEL_ORDER if m != "poisson"])
    predictions["track"] = predictions["model"].map(MODEL_TRACKS)
    summary, class_metrics, _ = evaluate.evaluate_predictions(predictions)
    summary.loc[summary["model"] == "poisson", "rps"] = 0.2
    with pytest.raises(ValueError, match="NaN"):
        evaluate.validate_outputs(base, predictions, summary, class_metrics)
