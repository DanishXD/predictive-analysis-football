"""Eksperimen hyperparameter tuning dengan TimeSeriesSplit (Phase 19).

Berbeda dari Fase 5 yang memakai satu set parameter tetap, modul ini mencari
parameter terbaik per model lewat GridSearchCV dengan:

- ``cv`` = DateTimeSeriesSplit, adapter dari ``date_based_time_splits``, jadi
  urutan waktu dijaga persis seperti disiplin di seluruh project.
- ``scoring`` = RPS (custom scorer), karena RPS adalah metrik utama Fase 7
  dan metrik itu yang diprioritaskan. Accuracy TIDAK dipakai: pada 3 kelas
  dengan Draw minoritas, accuracy hampir tidak membedakan model.

Modul ini murni eksperimen: tidak menulis model produksi dan tidak menyentuh
artefak Fase 5/7.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.base import clone
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV

from config import (
    DATA_DIR,
    N_SPLITS,
    PROCESSED_DIR,
    TARGET_MAPPING,
    TEST_SEASON,
)
from train_model import build_models, date_based_time_splits, load_modeling_data

RESULTS_PATH = PROCESSED_DIR / "tuning_results.csv"
CV_CURVE_PATH = PROCESSED_DIR / "tuning_cv_results.csv"
SUMMARY_PATH = DATA_DIR / "metadata" / "tuning_summary.json"

# Search space sengaja dijaga kecil supaya total biaya tetap wajar
# (RF ~2 menit, XGBoost ~3 menit di mesin ini).
RANDOM_FOREST_GRID = {
    "classifier__n_estimators": [250, 400],
    "classifier__max_depth": [6, 10, None],
    "classifier__min_samples_leaf": [3, 8],
}

XGBOOST_GRID = {
    "classifier__n_estimators": [200, 350, 500],
    "classifier__learning_rate": [0.02, 0.05, 0.1],
    "classifier__max_depth": [2, 3, 5],
}


class DateTimeSeriesSplit:
    """Adapter ``date_based_time_splits`` ke protokol cv sklearn.

    Dipakai supaya GridSearchCV membagi berdasarkan tanggal, bukan baris:
    satu tanggal tidak boleh terbelah ke dua fold berbeda.
    """

    def __init__(self, dates, n_splits: int = N_SPLITS):
        self.dates = np.asarray(dates)
        self.n_splits = n_splits

    def split(self, X=None, y=None, groups=None):
        frame = pd.DataFrame({"date": self.dates})
        for _, train_idx, validation_idx in date_based_time_splits(
            frame, n_splits=self.n_splits
        ):
            yield train_idx, validation_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def rps_scorer(y_true, y_proba) -> float:
    """Rata-rata RPS dari matriks probabilitas.

    ``pb.metrics.rps_average`` sudah mengembalikan mean RPS per-sample, jadi
    bisa langsung dipakai sebagai custom scorer.
    """
    return float(pb.metrics.rps_average(np.asarray(y_proba), np.asarray(y_true, dtype=int)))


def make_rps_scorer():
    return make_scorer(rps_scorer, greater_is_better=False, response_method="predict_proba")


def single_threaded(model):
    """Matikan nested parallelism.

    ``build_models()`` sudah memakai ``n_jobs=-1``. Kalau GridSearchCV juga
    ``n_jobs=-1``, tiap worker RF/XGB akan membuat thread sendiri sehingga
    mesin oversubscribe dan proses bisa jauh lebih lambat.
    """
    model = clone(model)
    classifier = model.named_steps.get("classifier")
    if classifier is not None and hasattr(classifier, "n_jobs"):
        classifier.set_params(n_jobs=1)
    return model


def search_model(
    name: str,
    base_model,
    grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    dates: np.ndarray,
    scoring,
) -> tuple[dict, pd.DataFrame, float]:
    """Jalankan GridSearchCV dengan RPS + date-based TimeSeriesSplit."""
    model = single_threaded(base_model)
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        cv=DateTimeSeriesSplit(dates),
        scoring=scoring,
        n_jobs=-1,
        refit=False,
        return_train_score=False,
    )
    search.fit(X, y)
    frame = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(
        columns={
            "params": "params",
            "mean_test_score": "cv_rps_mean",
            "std_test_score": "cv_rps_std",
            "rank_test_score": "rank",
        }
    )
    # greater_is_better=False membuat sklearn membalik tanda skor; balik lagi
    # supaya angka yang ditampilkan benar-benar RPS (positif, lebih kecil = lebih baik).
    frame["cv_rps_mean"] = -frame["cv_rps_mean"]
    frame["cv_rps_std"] = -frame["cv_rps_std"]
    frame.insert(0, "model", name)
    frame = frame.sort_values("rank").reset_index(drop=True)
    return search.best_params_, frame, float(-search.best_score_)


def evaluate_with_dates(
    base_model, params: dict, X, y, dates, test, y_test
) -> dict:
    """Nilai parameter tertentu di OOF (CV) dan di test season.

    `dates` diterima terpisah karena kolom date adalah kolom identitas,
    bukan bagian dari matriks fitur.
    """
    model = single_threaded(base_model).set_params(**params)
    covered_pred = {}
    for _, train_idx, validation_idx in date_based_time_splits(
        pd.DataFrame({"date": dates})
    ):
        probabilities = clone(model).fit(
            X.iloc[train_idx], y.iloc[train_idx]
        ).predict_proba(X.iloc[validation_idx])
        for position, row_index in enumerate(validation_idx):
            covered_pred[int(row_index)] = probabilities[position]
    covered = np.array(sorted(covered_pred))
    stack = np.array([covered_pred[int(i)] for i in covered])
    codes = y.to_numpy()[covered]
    oof = {
        "oof_rps": float(pb.metrics.rps_average(stack, codes)),
        "oof_log_loss": float(log_loss(codes, stack, labels=[0, 1, 2])),
    }
    test_probabilities = clone(model).fit(X, y).predict_proba(test)
    test_summary = {
        "test_rps": float(pb.metrics.rps_average(test_probabilities, y_test)),
        "test_log_loss": float(log_loss(y_test, test_probabilities, labels=[0, 1, 2])),
        "test_accuracy": float((test_probabilities.argmax(axis=1) == y_test).mean()),
        "test_draw_predictions": int(
            (test_probabilities.argmax(axis=1) == TARGET_MAPPING["D"]).sum()
        ),
    }
    return {**oof, **test_summary}


def main() -> None:
    print("=" * 74)
    print("Phase 19: hyperparameter tuning (TimeSeriesSplit + objective RPS)")
    print("=" * 74)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    data, feature_columns = load_modeling_data()
    data["target"] = data["result"].map(TARGET_MAPPING)
    train = data.loc[data["season"] != TEST_SEASON].reset_index(drop=True)
    test = data.loc[data["season"] == TEST_SEASON].reset_index(drop=True)

    X_train = train[feature_columns]
    y_train = train["target"]
    train_dates = train["date"].to_numpy()
    X_test = test[feature_columns]
    y_test = test["target"].to_numpy()

    models = build_models()
    scoring = make_rps_scorer()
    grids = {
        "random_forest": (models["random_forest"], RANDOM_FOREST_GRID),
        "xgboost": (models["xgboost"], XGBOOST_GRID),
    }

    results, cv_tables, best_params_map = [], [], {}
    for name, (base_model, grid) in grids.items():
        n_combinations = int(np.prod([len(v) for v in grid.values()]))
        print(f"\n--- {name}: {n_combinations} kombinasi x {N_SPLITS} fold ---")
        best_params, cv_frame, best_score = search_model(
            name, base_model, grid, X_train, y_train, train_dates, scoring
        )
        best_params_map[name] = best_params
        print(f"  RPS CV terbaik : {best_score:.6f}")
        print("  Parameter terbaik:")
        for key, value in best_params.items():
            print(f"    {key} = {value}")
        print("  5 teratas:")
        print(
            cv_frame.head(5)[
                ["rank", "cv_rps_mean", "cv_rps_std", "params"]
            ].to_string(index=False, float_format=lambda v: f"{v:.6f}")
        )
        results.append({"model": name, "best_cv_rps": best_score,
                        "params": json.dumps(best_params, default=str)})
        cv_tables.append(cv_frame)

    # --- before vs after ---
    print("\n" + "=" * 74)
    print("PERBANDINGAN: parameter default (Fase 5) vs hasil tuning")
    print("=" * 74)
    comparison_rows = []
    for name in grids:
        base_model = models[name]
        default_metrics = evaluate_with_dates(
            base_model, {}, X_train, y_train, train_dates, X_test, y_test
        )
        tuned_metrics = evaluate_with_dates(
            base_model, best_params_map[name], X_train, y_train, train_dates,
            X_test, y_test,
        )
        row = {"model": name}
        row.update({f"default_{k}": v for k, v in default_metrics.items()})
        row.update({f"tuned_{k}": v for k, v in tuned_metrics.items()})
        row["delta_oof_rps"] = tuned_metrics["oof_rps"] - default_metrics["oof_rps"]
        row["delta_test_rps"] = tuned_metrics["test_rps"] - default_metrics["test_rps"]
        row["delta_test_log_loss"] = (
            tuned_metrics["test_log_loss"] - default_metrics["test_log_loss"]
        )
        comparison_rows.append(row)
        print(f"\n  {name}:")
        print(
            f"    OOF RPS      : default {default_metrics['oof_rps']:.6f} "
            f"-> tuned {tuned_metrics['oof_rps']:.6f} "
            f"({row['delta_oof_rps']:+.6f})"
        )
        print(
            f"    Test RPS     : default {default_metrics['test_rps']:.6f} "
            f"-> tuned {tuned_metrics['test_rps']:.6f} "
            f"({row['delta_test_rps']:+.6f})"
        )
        print(
            f"    Test log loss: default {default_metrics['test_log_loss']:.6f} "
            f"-> tuned {tuned_metrics['test_log_loss']:.6f} "
            f"({row['delta_test_log_loss']:+.6f})"
        )

    comparison = pd.DataFrame(comparison_rows)
    all_cv = pd.concat(cv_tables, ignore_index=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    all_cv.to_csv(CV_CURVE_PATH, index=False)
    comparison.to_csv(PROCESSED_DIR / "tuning_before_after.csv", index=False)

    summary_payload = {
        "purpose": "Eksperimen tuning. Model produksi tidak diubah.",
        "cv_strategy": f"DateTimeSeriesSplit({N_SPLITS} fold, forward-only by date)",
        "scoring": "RPS (custom scorer, lower is better)",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "grids": {"random_forest": RANDOM_FOREST_GRID, "xgboost": XGBOOST_GRID},
        "best_params": best_params_map,
        "before_after": comparison.to_dict("records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    print(f"\nTersimpan: {RESULTS_PATH}")
    print(f"Tersimpan: {CV_CURVE_PATH}")
    print(f"Tersimpan: {SUMMARY_PATH}")
    print("\nCatatan: model produksi TIDAK diganti. Semua ini hanya laporan.")


if __name__ == "__main__":
    main()
