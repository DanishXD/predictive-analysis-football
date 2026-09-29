"""Eksperimen koreksi probabilitas output model (Phase 18).

Dua pendekatan untuk masalah kelas Draw, keduanya TIDAK mengubah training:

1. Post-hoc calibration (Platt/sigmoid & isotonic) memakai
   ``CalibratedClassifierCV`` dengan ``cv=TimeSeriesSplit``. Mapping
   kalibrasi di-fit pada held-out fold sehingga tidak memakai data yang
   sama dengan training base model (anti-leakage).

2. Draw-prior adjustment: kali kan P(draw) dengan faktor lambda, lalu
   renormalisasi. Lambda DIPILIH dengan mencari nilai yang meminimalkan
   RPS di CV training -- bukan di test season.

Modul ini murni eksperimen: tidak menulis model produksi, tidak menyentuh
artefak Fase 5/7. Hasilnya hanya laporan + audit trail di data/metadata/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from config import (
    DATA_DIR,
    PROCESSED_DIR,
    TARGET_MAPPING,
    TEST_SEASON,
)
from train_model import build_models, date_based_time_splits, load_modeling_data

COMPARISON_PATH = PROCESSED_DIR / "calibration_comparison.csv"
SUMMARY_PATH = DATA_DIR / "metadata" / "calibration_summary.json"

# CalibratedClassifierCV(cv=...) mewajibkan splitter non-overlapping, jadi
# ensemble=False tidak bisa dipakai bersama TimeSeriesSplit. ensemble=True
# juga justru yang kita mau: tiap base model di-fit pada satu fold dan
# dikalibrasi pada held-out fold-nya.
CALIBRATION_INNER_SPLITS = 3

# Kandidat faktor draw-prior. Dicari di CV training, bukan di test.
DRAW_PRIOR_GRID = np.round(np.arange(1.0, 2.01, 0.1), 2)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Pastikan tiap baris menjumlah 1 (isotonic/sigmoid kadang tidak presisi)."""
    probabilities = np.asarray(probabilities, dtype=float)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Ada baris probabilitas yang nol total")
    return probabilities / totals


def apply_draw_prior(probabilities: np.ndarray, factor: float) -> np.ndarray:
    """Kalikan P(draw) dengan `factor`, lalu renormalisasi."""
    adjusted = np.asarray(probabilities, dtype=float).copy()
    adjusted[:, TARGET_MAPPING["D"]] *= factor
    return normalize_probabilities(adjusted)


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray,
                               bins: int = 10) -> float:
    """Mean one-vs-rest ECE, definisi sama dengan evaluate.py Fase 7.

    Penting: definisi HARUS sama dengan yang dipakai evaluate.py supaya
    angka di modul ini bisa dibandingkan langsung dengan
    evaluation_summary.csv. Versi top-label menghasilkan angka jauh lebih
    besar karena mengukur keyakinan kelas yang benar, bukan keyakinan
    prediksi.
    """
    codes = np.asarray(y_true, dtype=int)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    class_errors = []
    for class_index in range(probabilities.shape[1]):
        predicted = probabilities[:, class_index]
        observed = (codes == class_index).astype(float)
        class_error = 0.0
        for bin_index in range(bins):
            lower = bin_edges[bin_index]
            upper = bin_edges[bin_index + 1]
            if bin_index == bins - 1:
                mask = (predicted >= lower) & (predicted <= upper)
            else:
                mask = (predicted >= lower) & (predicted < upper)
            if mask.any():
                class_error += mask.mean() * abs(
                    predicted[mask].mean() - observed[mask].mean()
                )
        class_errors.append(class_error)
    return float(np.mean(class_errors))


def draw_recall(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    codes = np.asarray(y_true, dtype=int)
    draw = TARGET_MAPPING["D"]
    actual_draw = int((codes == draw).sum())
    if actual_draw == 0:
        return float("nan")
    predicted = probabilities.argmax(axis=1)
    return float(((codes == draw) & (predicted == draw)).sum() / actual_draw)


def summarise(probabilities: np.ndarray, y_true: np.ndarray) -> dict:
    """Metrik utama untuk satu varian prediksi."""
    probabilities = normalize_probabilities(probabilities)
    codes = np.asarray(y_true, dtype=int)
    return {
        "rps": float(pb.metrics.rps_average(probabilities, codes)),
        "log_loss": float(log_loss(codes, probabilities, labels=[0, 1, 2])),
        "accuracy": float((probabilities.argmax(axis=1) == codes).mean()),
        "draw_recall": draw_recall(probabilities, codes),
        "draw_predictions": int((probabilities.argmax(axis=1) == TARGET_MAPPING["D"]).sum()),
        "mean_ece": expected_calibration_error(probabilities, codes),
        "mean_pred_draw": float(probabilities[:, TARGET_MAPPING["D"]].mean()),
    }


def build_calibrated(base_model, method: str, n_splits: int = CALIBRATION_INNER_SPLITS):
    """Bungkus base model dengan CalibratedClassifierCV yang time-aware."""
    return CalibratedClassifierCV(
        estimator=clone(base_model),
        method=method,
        cv=TimeSeriesSplit(n_splits=n_splits),
        ensemble=True,
        n_jobs=-1,
    )


def _make_variant(base_model, method: str):
    """Factory: 'raw' memakai model apa adanya, sisanya dibungkus kalibrator."""
    def factory(X_train, y_train):
        if method == "raw":
            return clone(base_model).fit(X_train, y_train)
        return build_calibrated(base_model, method).fit(X_train, y_train)

    return factory


def out_of_fold_probabilities(
    base_model,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    methods: tuple[str, ...],
) -> dict[str, dict[int, np.ndarray]]:
    """Probabilitas out-of-fold untuk tiap varian, di forward-only folds.

    `dates` sengaja berada di luar matriks fitur: kolom `date` adalah kolom
    identitas, bukan fitur, jadi diteruskan terpisah.

    PENTING: hanya baris yang muncul sebagai validation fold yang punya
    prediksi. Baris paling awal tidak pernah divalidasi (TimeSeriesSplit
    butuh history), jadi metrik WAJIB dihitung hanya pada baris tercakup.
    Menghitung di semua baris akan menghasilkan log loss palsu yang jauh
    lebih buruk karena probabilitas 0.
    """
    collected: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}
    for _, train_idx, validation_idx in date_based_time_splits(
        pd.DataFrame({"date": dates.to_numpy()})
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[validation_idx]
        for name in methods:
            model = _make_variant(base_model, name)(X_train, y_train)
            probabilities = normalize_probabilities(model.predict_proba(X_val))
            for position, row_index in enumerate(validation_idx):
                collected[name][int(row_index)] = probabilities[position]
    return collected


def evaluate_variants(
    base_model,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test: pd.DataFrame,
    y_test: np.ndarray,
    methods: tuple[str, ...] = ("raw", "sigmoid", "isotonic"),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Bandingkan raw vs tiap metode kalibrasi di OOF dan di test season."""
    oof = out_of_fold_probabilities(base_model, X, y, dates, methods)

    covered = np.array(sorted(oof["raw"]))
    codes = y.to_numpy()
    rows = []
    for name in methods:
        stack = np.array([oof[name][int(i)] for i in covered])
        summary = summarise(stack, codes[covered])
        summary["variant"] = name
        summary["split"] = "oof"
        summary["n_rows"] = len(covered)
        rows.append(summary)

    test_predictions = {}
    for name in methods:
        model = _make_variant(base_model, name)(X, y)
        probabilities = normalize_probabilities(model.predict_proba(test))
        test_predictions[name] = probabilities
        summary = summarise(probabilities, y_test)
        summary["variant"] = name
        summary["split"] = "test"
        summary["n_rows"] = len(y_test)
        rows.append(summary)

    return pd.DataFrame(rows), test_predictions


def select_draw_prior(
    raw_test_predictions: dict[str, np.ndarray],
    oof_probabilities: dict[int, np.ndarray],
    y: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """Pilih faktor draw-prior yang meminimalkan RPS di CV TRAINING saja.

    Fungsi ini tidak pernah melihat test season -- inilah yang membedakan
    ini dari tuning di atas test set.
    """
    covered = np.array(sorted(oof_probabilities))
    stack = np.array([oof_probabilities[int(i)] for i in covered])
    codes = y.to_numpy()[covered]
    rows = []
    for factor in DRAW_PRIOR_GRID:
        summary = summarise(apply_draw_prior(stack, factor), codes)
        summary["factor"] = float(factor)
        rows.append(summary)
    table = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    return float(table.loc[0, "factor"]), table


def main() -> None:
    print("=" * 74)
    print("Phase 18: koreksi probabilitas output model (eksperimen)")
    print("=" * 74)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    data, feature_columns = load_modeling_data()
    data["target"] = data["result"].map(TARGET_MAPPING)
    train = data.loc[data["season"] != TEST_SEASON].reset_index(drop=True)
    test = data.loc[data["season"] == TEST_SEASON].reset_index(drop=True)

    X_train = train[feature_columns]
    y_train = train["target"]
    train_dates = train["date"]
    X_test = test[feature_columns]
    y_test = test["target"].to_numpy()

    base_model = build_models()["random_forest"]
    print(f"Base model    : random_forest (Fase 5, tidak diubah)")
    print(f"Train         : {len(train)} match, {len(feature_columns)} fitur")
    print(f"Test          : {len(test)} match ({TEST_SEASON})")

    print("\n--- A. Post-hoc calibration (OOF + test) ---")
    comparison, test_predictions = evaluate_variants(
        base_model, X_train, y_train, train_dates, X_test, y_test
    )
    for split in ("oof", "test"):
        block = comparison[comparison["split"] == split]
        print(f"\n  [{split.upper()}]  n={int(block['n_rows'].iloc[0])}")
        print(
            block[["variant", "rps", "log_loss", "draw_recall",
                   "draw_predictions", "mean_ece"]]
            .to_string(index=False, float_format=lambda v: f"{v:.6f}")
        )

    # --- B. draw-prior adjustment ---
    print("\n--- B. Draw-prior adjustment ---")
    oof_raw = out_of_fold_probabilities(
        base_model, X_train, y_train, train_dates, ("raw",)
    )["raw"]
    best_factor, prior_table = select_draw_prior(test_predictions["raw"], oof_raw, y_train)
    print(f"  Faktor terbaik (dipilih via CV training): {best_factor:.2f}")
    print("\n  Kurva RPS di CV training terhadap faktor:")
    print(
        prior_table[["factor", "rps", "log_loss", "draw_recall", "draw_predictions"]]
        .to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )

    draw_prior_test = {
        "raw_draw_prior": apply_draw_prior(test_predictions["raw"], best_factor)
    }
    for name, probabilities in draw_prior_test.items():
        summary = summarise(probabilities, y_test)
        summary["variant"] = name
        summary["split"] = "test"
        summary["n_rows"] = len(y_test)
        comparison = pd.concat(
            [comparison, pd.DataFrame([summary])], ignore_index=True
        )
    summary_row = summarise(draw_prior_test["raw_draw_prior"], y_test)
    print(f"\n  Di TEST season dengan faktor {best_factor:.2f}:")
    print(
        f"    RPS         : {summary_row['rps']:.6f} "
        f"(raw {summarise(test_predictions['raw'], y_test)['rps']:.6f})"
    )
    print(f"    log loss    : {summary_row['log_loss']:.6f}")
    print(f"    draw recall : {summary_row['draw_recall']:.6f} "
        f"(raw {summarise(test_predictions['raw'], y_test)['draw_recall']:.6f})")

    # --- C. Kesimpulan ---
    raw_test = summarise(test_predictions["raw"], y_test)
    print("\n" + "=" * 74)
    print("C. RINGKASAN EFEKTIFITAS (test season)")
    print("=" * 74)
    verdict_rows = []
    for variant in ["sigmoid", "isotonic", "raw_draw_prior"]:
        row = summarise(
            draw_prior_test["raw_draw_prior"] if variant == "raw_draw_prior"
            else test_predictions[variant],
            y_test,
        )
        delta_rps = row["rps"] - raw_test["rps"]
        verdict_rows.append({
            "variant": variant,
            "rps": row["rps"],
            "delta_rps": delta_rps,
            "delta_log_loss": row["log_loss"] - raw_test["log_loss"],
            "draw_recall": row["draw_recall"],
            "delta_draw_recall": row["draw_recall"] - raw_test["draw_recall"],
            "better_rps": delta_rps < 0,
        })
    verdict = pd.DataFrame(verdict_rows)
    print(verdict.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))

    comparison.to_csv(COMPARISON_PATH, index=False)
    prior_table.to_csv(PROCESSED_DIR / "draw_prior_curve.csv", index=False)
    summary_payload = {
        "purpose": "Eksperimen koreksi output probability. Model produksi tidak diubah.",
        "base_model": "random_forest",
        "test_season": TEST_SEASON,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "oof_rows_covered": int(comparison["n_rows"].iloc[0]),
        "selected_draw_prior_factor": best_factor,
        "raw_test": raw_test,
        "verdict": verdict.to_dict("records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nTersimpan: {COMPARISON_PATH}")
    print(f"Tersimpan: {PROCESSED_DIR / 'draw_prior_curve.csv'}")
    print(f"Tersimpan: {SUMMARY_PATH}")
    print("\nCatatan: model produksi TIDAK diganti. Semua ini hanya laporan.")


if __name__ == "__main__":
    main()
