"""Eksperimen: home advantage terpisah untuk musim tanpa penonton (2020/21).

Motifasi
--------
Poisson dan Dixon-Coles punya SATU parameter ``home_advantage`` global.
Musim 2020/2021 dimainkan tanpa penonton, dan HFA musim itu jelas berbeda
(HFA 2020/21 = -0.071 vs musim lain +0.425 poin/match, bootstrap 95% CI
[-0.385, -0.099]). Satu parameter global tidak bisa merepresentasikan itu.

Eksperimen ini menguji varian dengan DUA parameter HFA: satu di-fit dari
match non-no-fans, satu dari match no-fans, sementara parameter
attack/defense tetap di-fit dari seluruh data training. Prediksi memakai
HFA no-fans hanya untuk match di ``EMPTY_STADIUM_SEASONS``.

Keterbatasan struktural (baca sebelum menarik kesimpulan)
--------------------------------------------------------
Season 2020/2021 ada di TRAINING, dan test season 2025/2026 bukan musim
no-fans. Konsekuensinya:

1. Varian ini hanya mengubah prediksi untuk match no-fans. Di walk-forward
   modul ``track_a_cv``, season 2020/2021 jadi validation di fold 1, tapi
   season itu sendiri baru masuk training mulai fold 2. Jadi satu-satunya
   fold yang bisa menguji varian no-fans justru tidak punya musim no-fans
   di training, dan dilewati. Varian B (HFA khusus no-fans) secara
   struktural tidak akan pernah bisa diuji out-of-sample selama tidak ada
   musim no-fans kedua.
2. Test season juga tidak pernah bisa mengujinya, karena 2025/2026 bukan
   musim no-fans. Batasan yang sama seperti fitur ``is_empty_stadium`` di
   HANDOFF 5.5.

Yang benar-benar bisa diuji lintas fold adalah varian A: HFA global
di-profile-kan ulang hanya dari match non-no-fans, lalu dipakai untuk semua
prediksi. Itu mengukur apakah perbaikan HFA memberi keuntungan di musim
biasa, dan hasilnya dilaporkan terpisah.

Karena itu modul ini melaporkan DUA hal terpisah: selisih HFA antar musim
(diuji dengan bootstrap, signifikan) dan dampak ke metrik varian A (tidak
signifikan). Modul ini eksperimen: tidak menulis model produksi, tidak
menyentuh evaluate.py/predict_match.py, dan tidak mengubah
cv_model_selection.csv.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, log_loss

from config import (
    BOOTSTRAP_SAMPLES,
    DATA_DIR,
    EMPTY_STADIUM_SEASONS,
    MAX_GOALS,
    PROCESSED_DIR,
    SEASONS,
    TARGET_MAPPING,
    TEST_SEASON,
    TIME_DECAY_XI,
)
from track_a_cv import Fold, build_folds, score_fold, validate_folds

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
COMPARISON_PATH = PROCESSED_DIR / "seasonal_hfa_comparison.csv"
HFA_ESTIMATE_PATH = PROCESSED_DIR / "seasonal_hfa_estimates.csv"
DETAIL_PATH = PROCESSED_DIR / "seasonal_hfa_fold_details.csv"
METADATA_PATH = DATA_DIR / "metadata" / "seasonal_hfa_summary.json"

RANDOM_SEED = 20260929


def _hfa_log_likelihood(
    hfa: float,
    expected_home: np.ndarray,
    expected_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Log-likelihood Poisson dengan HFA satu variabel, attack/defense tetap."""
    home_rate = expected_home * np.exp(hfa)
    away_rate = expected_away
    home_rate = np.clip(home_rate, 1e-10, None)
    away_rate = np.clip(away_rate, 1e-10, None)
    return float(
        np.sum(
            weights
            * (
                goals_home * np.log(home_rate)
                - home_rate
                + goals_away * np.log(away_rate)
                - away_rate
            )
        )
    )


def refit_home_advantage(
    model,
    subset: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """Re-optimasi HFA saja atas subset, dengan attack/defense dibekukan.

    Pendekatan profile likelihood: parameter tim tetap di nilai hasil fit
    penuh, hanya HFA yang dicari ulang. Ini mengisolasi efek HFA tanpa
    mengulang seluruh fit, jadi turnaround-nya singkat.
    """
    parameters = model._params
    n_teams = model.n_teams
    goals_home = subset["goals_home"].to_numpy(dtype=float)
    goals_away = subset["goals_away"].to_numpy(dtype=float)
    home_idx = np.array(
        [model.team_to_idx[team] for team in subset["team_home"]], dtype=int
    )
    away_idx = np.array(
        [model.team_to_idx[team] for team in subset["team_away"]], dtype=int
    )
    attack = parameters[:n_teams]
    defense = parameters[n_teams : 2 * n_teams]
    expected_home = np.exp(attack[home_idx] + defense[away_idx])
    expected_away = np.exp(attack[away_idx] + defense[home_idx])

    result = minimize_scalar(
        lambda hfa: -_hfa_log_likelihood(
            hfa, expected_home, expected_away, goals_home, goals_away, weights
        ),
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    return float(result.x)


def _model_with_hfa(model, home_advantage: float):
    """Salinan model dengan HFA diganti, tanpa mengubah model asli."""
    shifted = copy.deepcopy(model)
    shifted._params = shifted._params.copy()
    shifted._params[-1] = home_advantage
    return shifted


def fit_with_seasonal_hfa(train: pd.DataFrame):
    """Fit Poisson baseline + dua estimasi HFA (no-fans vs sisanya)."""
    weights = pb.models.dixon_coles_weights(
        train["date"],
        xi=TIME_DECAY_XI,
        base_date=train["date"].max(),
    )
    model = pb.models.PoissonGoalsModel(
        train["goals_home"],
        train["goals_away"],
        train["team_home"],
        train["team_away"],
        weights=weights,
    )
    model.fit()

    is_empty = train["season"].isin(EMPTY_STADIUM_SEASONS).to_numpy()
    subset_weights = np.asarray(weights, dtype=float)

    hfa_normal = refit_home_advantage(
        model, train.loc[~is_empty], subset_weights[~is_empty]
    )
    hfa_empty = refit_home_advantage(
        model, train.loc[is_empty], subset_weights[is_empty]
    )
    return model, hfa_normal, hfa_empty


def _hfa_bootstrap_delta(
    model, train: pd.DataFrame, weights: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Bootstrap 95% CI untuk selisih HFA (no-fans dikurangi normal).

    Resampling dilakukan per-match dalam masing-masing subset. Selisihnya
    yang di-distribusi, bukan tiap HFA terpisah, karena yang ingin
    dijawab adalah apakah kedua musim itu benar-benar berbeda.
    """
    is_empty = train["season"].isin(EMPTY_STADIUM_SEASONS).to_numpy()
    empty_rows = train.loc[is_empty].reset_index(drop=True)
    normal_rows = train.loc[~is_empty].reset_index(drop=True)
    empty_weights = np.asarray(weights, dtype=float)[is_empty]
    normal_weights = np.asarray(weights, dtype=float)[~is_empty]

    def delta_for(empty_index: np.ndarray, normal_index: np.ndarray) -> float:
        hfa_empty = refit_home_advantage(
            model, empty_rows.loc[empty_index], empty_weights[empty_index]
        )
        hfa_normal = refit_home_advantage(
            model, normal_rows.loc[normal_index], normal_weights[normal_index]
        )
        return hfa_empty - hfa_normal

    point_estimate = delta_for(
        np.arange(len(empty_rows)), np.arange(len(normal_rows))
    )

    deltas = np.empty(BOOTSTRAP_SAMPLES)
    for draw in range(BOOTSTRAP_SAMPLES):
        deltas[draw] = delta_for(
            rng.integers(0, len(empty_rows), len(empty_rows)),
            rng.integers(0, len(normal_rows), len(normal_rows)),
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(point_estimate), float(low), float(high)


def evaluate_fold(
    fold: Fold, matches: pd.DataFrame, validation: pd.DataFrame, y_true: np.ndarray
) -> dict:
    """Bandingkan baseline vs varian seasonal-HFA pada satu fold."""
    train_seasons = [season for season in SEASONS if season <= fold.last_train_season]
    train = matches.loc[matches["season"].isin(train_seasons)].copy()

    if train["date"].max() >= validation["date"].min():
        raise ValueError(f"Fold {fold.number}: overlap waktu training/validation")
    if (train["season"] == TEST_SEASON).any():
        raise ValueError(f"Fold {fold.number}: TEST_SEASON bocor ke training")
    if not train["season"].isin(EMPTY_STADIUM_SEASONS).any():
        # Season no-fans belum ada di training, jadi tidak ada HFA no-fans
        # yang bisa di-profile-kan. Fold dilewati, bukan dipaksakan.
        return {
            "fold": fold.number,
            "validation_season": fold.validation_season,
            "validation_is_no_fans": bool(
                validation["season"].isin(EMPTY_STADIUM_SEASONS).any()
            ),
            "applicable": False,
            "skip_reason": "tidak ada musim no-fans di data training fold ini",
            "n_matches": len(validation),
        }

    model, hfa_normal, hfa_empty = fit_with_seasonal_hfa(train)

    baseline = score_fold(model, validation, dixon_coles=False)
    # Varian A: HFA global di-profile-kan ulang hanya dari match non-no-fans,
    # lalu dipakai untuk SEMUA match validasi. Ini bagian yang bisa diuji
    # lintas fold.
    variant_a = score_fold(_model_with_hfa(model, hfa_normal), validation, dixon_coles=False)
    # Varian B: HFA no-fans dipakai khusus untuk match no-fans. Karena season
    # validasi di fold yang bisa diuji tidak pernah no-fans, varian B secara
    # struktural tidak bisa dibedakan dari varian A di modul ini.
    if validation["season"].isin(EMPTY_STADIUM_SEASONS).any():
        variant_b = score_fold(
            _model_with_hfa(model, hfa_empty), validation, dixon_coles=False
        )
    else:
        variant_b = variant_a

    return {
        "fold": fold.number,
        "validation_season": fold.validation_season,
        "validation_is_no_fans": bool(
            validation["season"].isin(EMPTY_STADIUM_SEASONS).any()
        ),
        "applicable": True,
        "skip_reason": "",
        "n_matches": len(validation),
        "hfa_global": float(model._params[-1]),
        "hfa_normal": hfa_normal,
        "hfa_empty": hfa_empty,
        "hfa_delta_empty_minus_normal": hfa_empty - hfa_normal,
        "baseline_log_loss": _log_loss(y_true, baseline),
        "variant_a_log_loss": _log_loss(y_true, variant_a),
        "variant_b_log_loss": _log_loss(y_true, variant_b),
        "baseline_rps": _rps(y_true, baseline),
        "variant_a_rps": _rps(y_true, variant_a),
        "variant_b_rps": _rps(y_true, variant_b),
        "baseline_accuracy": _accuracy(y_true, baseline),
        "variant_a_accuracy": _accuracy(y_true, variant_a),
        "variant_b_accuracy": _accuracy(y_true, variant_b),
    }


def _log_loss(y_true: np.ndarray, frame: pd.DataFrame) -> float:
    return float(log_loss(y_true, _probabilities(frame), labels=[0, 1, 2]))


def _rps(y_true: np.ndarray, frame: pd.DataFrame) -> float:
    return float(pb.metrics.rps_average(_probabilities(frame), y_true))


def _accuracy(y_true: np.ndarray, frame: pd.DataFrame) -> float:
    return float(accuracy_score(y_true, _probabilities(frame).argmax(axis=1)))


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    """Konversi frame probabilitas dari ``score_fold`` jadi matriks numpy."""
    return frame[["prob_home", "prob_draw", "prob_away"]].to_numpy()


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan seluruh fold dan kembalikan (ringkasan, detail per fold)."""
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["datetime", "date"])
    folds = build_folds()
    validate_folds(folds)

    details = []
    for fold in folds:
        validation = matches.loc[matches["season"] == fold.validation_season].copy()
        y_true = validation["result"].map(TARGET_MAPPING).to_numpy()
        row = evaluate_fold(fold, matches, validation, y_true)
        details.append(row)
        if not row["applicable"]:
            print(
                f"Fold {fold.number}/{len(folds)} | validasi: {fold.validation_season} | "
                f"dilewati: {row['skip_reason']}"
            )
        else:
            print(
                f"Fold {fold.number}/{len(folds)} | validasi: {fold.validation_season} | "
                f"HFA normal {row['hfa_normal']:+.4f} vs no-fans {row['hfa_empty']:+.4f} "
                f"(delta {row['hfa_delta_empty_minus_normal']:+.4f})"
            )

    details_frame = pd.DataFrame(details)
    applicable = details_frame.loc[details_frame["applicable"]].copy()
    for label in ("a", "b"):
        applicable[f"delta_{label}_log_loss"] = (
            applicable[f"variant_{label}_log_loss"] - applicable["baseline_log_loss"]
        )
        applicable[f"delta_{label}_rps"] = (
            applicable[f"variant_{label}_rps"] - applicable["baseline_rps"]
        )

    # Hanya fold dengan season validasi no-fans yang bisa membedakan varian B
    # dari varian A. Kalau tidak ada, dampaknya nol secara struktural.
    sensitive = applicable.loc[applicable["validation_is_no_fans"]]
    summary = pd.DataFrame(
        [
            {
                "applicable_folds": int(len(applicable)),
                "skipped_folds": int(len(details_frame) - len(applicable)),
                "sensitive_folds": int(len(sensitive)),
                "total_folds": int(len(details_frame)),
                "n_matches_affected": int(sensitive["n_matches"].sum()),
                "mean_hfa_normal": float(applicable["hfa_normal"].mean()),
                "mean_hfa_empty": float(applicable["hfa_empty"].mean()),
                "mean_hfa_delta": float(
                    applicable["hfa_delta_empty_minus_normal"].mean()
                ),
                "delta_a_log_loss_mean": float(applicable["delta_a_log_loss"].mean()),
                "delta_a_rps_mean": float(applicable["delta_a_rps"].mean()),
                "delta_a_folds_better": int((applicable["delta_a_log_loss"] < 0).sum()),
                "delta_b_log_loss_mean": (
                    float(applicable["delta_b_log_loss"].mean())
                    if len(applicable)
                    else np.nan
                ),
                "delta_b_rps_mean": (
                    float(applicable["delta_b_rps"].mean())
                    if len(applicable)
                    else np.nan
                ),
            }
        ]
    )
    return summary, applicable


def estimate_hfa_difference() -> pd.DataFrame:
    """Bootstrap CI untuk selisih HFA no-fans vs normal pada fit penuh.

    Dihitung hanya dari data training, sesuai aturan bahwa angka penentu
    tidak boleh berasal dari test set.
    """
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["datetime", "date"])
    train = matches.loc[matches["season"] != TEST_SEASON].copy()
    weights = np.asarray(
        pb.models.dixon_coles_weights(
            train["date"], xi=TIME_DECAY_XI, base_date=train["date"].max()
        ),
        dtype=float,
    )
    model, _, _ = fit_with_seasonal_hfa(train)

    rng = np.random.default_rng(RANDOM_SEED)
    point, low, high = _hfa_bootstrap_delta(model, train, weights, rng)
    print(
        f"\nSelisih HFA (no-fans dikurangi normal): {point:+.4f} "
        f"[95% CI {low:+.4f}, {high:+.4f}]"
    )
    return pd.DataFrame(
        [
            {
                "hfa_delta_empty_minus_normal": point,
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap": BOOTSTRAP_SAMPLES,
                "significant": bool(low * high > 0),
                "empty_stadium_seasons": ", ".join(EMPTY_STADIUM_SEASONS),
            }
        ]
    )


def print_summary(
    summary: pd.DataFrame, details: pd.DataFrame, estimates: pd.DataFrame
) -> None:
    """Cetak hasil eksperimen beserta batasan eksperimentasinya."""
    width = 78
    print("\n" + "=" * width)
    print("EKSERIMEN: HOME ADVANTAGE TERPISAH UNTUK MUSIM NO-FANS")
    print("=" * width)

    print("\nPer fold (varian A = HFA di-profile-kan tanpa musim no-fans):")
    print(
        details[
            [
                "fold",
                "validation_season",
                "validation_is_no_fans",
                "hfa_normal",
                "hfa_empty",
                "delta_a_log_loss",
                "delta_a_rps",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:+.6f}")
    )

    row = summary.iloc[0]
    applicable = int(row["applicable_folds"])
    skipped = int(row["skipped_folds"])
    sensitive = int(row["sensitive_folds"])
    total = int(row["total_folds"])
    print(f"\nFold berlaku: {applicable} (dilewati: {skipped})")
    print(
        f"Rata-rata HFA: normal {row['mean_hfa_normal']:+.4f} vs "
        f"no-fans {row['mean_hfa_empty']:+.4f} "
        f"(delta {row['mean_hfa_delta']:+.4f})"
    )
    estimate = estimates.iloc[0]
    print(
        f"Bootstrap 95% CI untuk selisih HFA: "
        f"[{estimate['ci_low']:+.4f}, {estimate['ci_high']:+.4f}] "
        f"-> {'signifikan' if estimate['significant'] else 'TIDAK signifikan'}"
    )
    print(
        f"\nVarian A lintas {applicable} fold: delta log loss "
        f"{row['delta_a_log_loss_mean']:+.6f} "
        f"({int(row['delta_a_folds_better'])}/{applicable} fold membaik), "
        f"delta RPS {row['delta_a_rps_mean']:+.6f}"
    )
    print(
        f"Fold dengan season validasi no-fans: {sensitive} dari {total} -> "
        "varian B (HFA khusus no-fans) secara struktural tidak bisa "
        "dibedakan dari varian A."
    )

    print("=" * width)
    print(
        "KESIMPULAN STRUKTURAL: test season 2025-2026 bukan musim no-fans,\n"
        "jadi tidak ada match no-fans yang bisa diuji di test season. Dan dari\n"
        f"{total} fold walk-forward, {sensitive} punya season validasi no-fans,\n"
        "sehingga bukti multi-fold untuk varian B tidak akan pernah tersedia\n"
        "selama tidak ada musim no-fans kedua. Modul ini laporan, bukan\n"
        "kandidat fitur produksi."
    )
    print("=" * width)


def main() -> None:
    print("=" * 78)
    print("Eksperimen HFA terpisah untuk musim no-fans (2020/2021)")
    print("=" * 78)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary, details = run_experiment()
    estimates = estimate_hfa_difference()

    details.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(COMPARISON_PATH, index=False)
    estimates.to_csv(HFA_ESTIMATE_PATH, index=False)

    print_summary(summary, details, estimates)

    payload = {
        "purpose": (
            "Eksperimen. Model produksi, evaluate.py, dan predict_match.py "
            "tidak diubah."
        ),
        "empty_stadium_seasons": list(EMPTY_STADIUM_SEASONS),
        "test_season": TEST_SEASON,
        "test_season_can_validate_this": False,
        "test_season_reason": (
            "Test season bukan musim no-fans, jadi tidak ada match no-fans "
            "yang bisa diuji. Batasan struktural yang sama seperti fitur "
            "is_empty_stadium (HANDOFF 5.5)."
        ),
        "max_goals": MAX_GOALS,
        "time_decay_xi": TIME_DECAY_XI,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "folds_sensitive": int(summary.iloc[0]["sensitive_folds"]),
        "folds_total": int(summary.iloc[0]["total_folds"]),
        "folds_applicable": int(summary.iloc[0]["applicable_folds"]),
        "folds_skipped": int(summary.iloc[0]["skipped_folds"]),
        "hfa_delta": estimates.iloc[0].to_dict(),
        "summary": summary.to_dict("records"),
        "per_fold": details.to_dict("records"),
    }
    METADATA_PATH.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nTersimpan: {DETAIL_PATH}")
    print(f"Tersimpan: {COMPARISON_PATH}")
    print(f"Tersimpan: {HFA_ESTIMATE_PATH}")
    print(f"Tersimpan: {METADATA_PATH}")
    print(
        "\nCatatan: angka HFA dihitung dari data training saja. Test season "
        "tidak pernah dipakai\nsebagai sumber angka penentu."
    )


if __name__ == "__main__":
    main()
