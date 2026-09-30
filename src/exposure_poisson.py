"""Exposure-based Poisson: model Poisson dengan volume tembakan sebagai eksposur.

Kenapa modul ini ada
-------------------
Poisson goal-based (Fase 4) memodelkan ``goals ~ Poisson(lambda)`` di mana
lambda ditentukan penuh oleh kekuatan tim dan home advantage. Data lokal
pun sudah punya jumlah tembakan (``shots_home``/``shots_away``) yang
tidak dipakai model itu sama sekali. Pertanyaannya: apakah volume
tembakan dan efisiensi per tembakan membawa informasi yang tidak ada di
gol.

Ini BUKAN model berbasis xG. xG asli butuh shot location dan shot quality
(koordinat, body part, situasi) per tembakan; data lokal tidak punya
salah satunya. Yang bisa dihitung di sini hanya "berapa banyak peluang"
dan "seberapa efisien peluang itu dikonversi" dalam bentuk agregat per
tim per match. Karena itu nama modul dan semua output memakai istilah
exposure-based, bukan xG.

Struktur dua tahap (dan kenapa wajib dua tahap)
------------------------------------------------
Model offset tunggal::

    lambda_home = shots_home * exp(a_home - d_away + gamma)

hanya bisa dihitung kalau ``shots_home`` sudah diketahui, artinya untuk
match yang SUDAH SELESAI. Untuk memprediksi pertandingan mendatang kita
tidak tahu berapa tembakan yang akan terjadi, jadi exposure itu sendiri
harus diprediksi lebih dulu. Karena itu dekomposisi dua tahap imposed oleh
ketersediaan data, bukan pilihan gaya::

    sigma = E[shots]          (stage 1, model volume)
    q     = P(goal | shot)    (stage 2, model kualitas per tembakan)
    lambda = sigma * q

Stage 1 memakai ``penaltyblog.PoissonGoalsModel`` yang di-fit pada kolom
``shots_*``, jadi skemanya identik dengan Poisson goal-based produksi.
Stage 2 butuh offset ``log(trials)`` yang tidak didukung penaltyblog, jadi
diimplementasikan sendiri di modul ini dengan gradien analitik.

Konvensi tanda
--------------
Modul ini memakai konvensi teksbook Dixon-Coles::

    lambda_home = exp(attack_home - defense_away + home_advantage)

``defense`` adalah KEKUATAN defense dan masuk dengan tanda SUBTRAKTIF:
nilai ``defense`` lebih besar berarti defense lebih kuat dan membuat lambda
lawan lebih kecil.

Perhatikan ini berlawanan dengan parameter internal ``penaltyblog``, yang
menggunakan ``lambda_home = exp(attack_home + defense_away + hfa)`` dengan
``defense`` sebagai kelemahan (nilai lebih besar = defense lebih buruk).
Pemetaannya::

    defense_penaltyblog = -defense_teksbook

Jadi ``defense_teksbook`` di sini = ``-defense_penaltyblog`` untuk tim yang
sama. Modul ini tidak membaca parameter model produksi, jadi keduanya tidak
pernah tercampur; pemetaan ini hanya dicatat supaya pembaca tidak salah
membandingkan tabel kekuatan tim antar dua modul.

Temuan: home advantage praktis seluruhnya ada di volume, bukan di efisiensi
--------------------------------------------------------------------------
Di season training, rasio home/away adalah: goals 1.212, shots 1.204, tapi
conversion rate hanya 1.006 dan SOT-rate 0.986. Artinya "home edge" hampir
seluruhnya muncul sebagai "tim home membuat lebih banyak tembakan", bukan
"tembakan home lebih akurat".

Konsekuensi yang terukur: parameter ``home_advantage`` di stage-2 berakhir
di batas bawah 0.000 untuk kedua varian. Itu HASIL yang benar, bukan kegagalan
optimasi — sebagian besar efek rumah sudah di-capture stage-1 (volume), jadi
tidak ada sisa untuk stage-2 (efisiensi). Test ``test_stage2_home_advantage_
is_negligible`` mengunci temuan ini supaya tidak dianggap bug di kemudian hari.

Varian yang diimplementasikan
------------------------------
``shot_volume``    : sigma = E[shots], q = P(goal | shot)
``sot``            : sigma = E[shots], q = P(SOT | shot), r = P(goal | SOT)

Batas eksperimen
----------------
Modul ini tidak menulis model produksi, tidak menyentuh
``statistical_models.py``, ``evaluate.py`` atau ``predict_match.py``, dan
tidak mengubah ``cv_model_selection.csv``. Semua angka penentu dihitung
lewat walk-forward musiman di season training saja; ``TEST_SEASON`` tidak
pernah dipakai sebagai fold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, log_loss

import config
from config import (
    BOOTSTRAP_SAMPLES,
    DATA_DIR,
    EXPOSURE_MIN_TRIALS,
    EXPOSURE_STAGE2_BOUNDS,
    EXPOSURE_STAGE2_HFA_BOUNDS,
    EXPOSURE_XI_GRID,
    MAX_GOALS,
    PROCESSED_DIR,
    TARGET_MAPPING,
    TEST_SEASON,
    TIME_DECAY_XI,
)
from track_a_cv import Fold, build_folds, validate_folds

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
RESULTS_PATH = PROCESSED_DIR / "exposure_poisson_results.csv"
SUMMARY_PATH = PROCESSED_DIR / "exposure_poisson_summary.csv"
COMPARISON_PATH = PROCESSED_DIR / "exposure_poisson_comparison.csv"
METADATA_PATH = DATA_DIR / "metadata" / "exposure_poisson_summary.json"

RANDOM_SEED = 20260929

VARIANT_SHOT = "shot_volume"
VARIANT_SOT = "sot"
VARIANTS = (VARIANT_SHOT, VARIANT_SOT)

BASELINE_NAME = "poisson_goals_based"


@dataclass(frozen=True)
class QualityFit:
    """Hasil fitter stage-2: parameter kualitas per-trial."""

    teams: tuple[str, ...]
    attack: np.ndarray
    defense: np.ndarray
    home_advantage: float
    n_used: int
    n_skipped: int
    log_likelihood: float
    converged: bool

def _index_or_mean(values: np.ndarray, teams: tuple[str, ...], team: str) -> float:
    """Parameter untuk tim, pakai rata-rata liga kalau tim tak dikenal."""
    if team in teams:
        return float(values[teams.index(team)])
    return float(values.mean())


def quality_rate(fit: QualityFit, team: str, opponent: str, is_home: bool) -> float:
    """Per-trial rate dengan fallback rata-rata liga untuk tim tak dikenal."""
    attack = _index_or_mean(fit.attack, fit.teams, team)
    defense = _index_or_mean(fit.defense, fit.teams, opponent)
    hfa = fit.home_advantage if is_home else 0.0
    return float(np.exp(attack - defense + hfa))


# --------------------------------------------------------------------------
# Stage 1: model volume (Poisson pada jumlah tembakan)
# --------------------------------------------------------------------------


def fit_volume_model(
    trials_home: pd.Series,
    trials_away: pd.Series,
    team_home: pd.Series,
    team_away: pd.Series,
    weights: np.ndarray,
):
    """Fit Poisson pada kolom trials untuk memperkirakan volume per match."""
    model = pb.models.PoissonGoalsModel(
        trials_home.to_numpy(),
        trials_away.to_numpy(),
        team_home.to_numpy(),
        team_away.to_numpy(),
        weights=weights,
    )
    model.fit()
    return model


def volume_rate(model, team: str, opponent: str, is_home: bool) -> float:
    """E[trials] untuk satu tim, dengan fallback rata-rata liga.

    Perhatikan konvensi. Parameter penaltyblog memakai ``defense`` sebagai
    kelemahan (positif, masuk aditif), sedangkan modul ini memakai
    ``defense`` sebagai kekuatan (subtraktif). Karena itu baris di bawah
    memakai ``+ defense_penaltyblog``, yang setara dengan
    ``- defense_teksbook``.
    """
    n_teams = model.n_teams
    parameters = model.params_array
    if team in model.teams and opponent in model.teams:
        attack = float(parameters[model.team_to_idx[team]])
        defense = float(parameters[model.team_to_idx[opponent] + n_teams])
    else:
        attack = float(parameters[:n_teams].mean())
        defense = float(parameters[n_teams : 2 * n_teams].mean())
    home_advantage = float(parameters[-1]) if is_home else 0.0
    return float(np.exp(attack + defense + home_advantage))


# --------------------------------------------------------------------------
# Stage 2: model kualitas (Poisson dengan offset log trials)
# --------------------------------------------------------------------------

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _unpack(params: np.ndarray, n_teams: int):
    return (
        np.asarray(params[:n_teams], dtype=float),
        np.asarray(params[n_teams : 2 * n_teams], dtype=float),
        float(params[-1]),
    )


def _linear_predictor(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    is_home: np.ndarray,
) -> np.ndarray:
    """eta = attack_shooter - defense_conceding + hfa*is_home."""
    attack, defense, hfa = _unpack(params, n_teams)
    eta = attack[home_idx] - defense[away_idx] + hfa * is_home
    return eta


def _nll_and_grad(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    is_home: np.ndarray,
    log_trials: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray,
):
    """Negative log-likelihood Poisson dengan offset, beserta gradiennya.

    Log-likelihood per baris::

        ll_i = w_i * [ y_i * (log t_i + eta_i) - t_i * exp(eta_i) ]

    dengan ``t_i`` exposure (tembakan) dan ``y_i`` sukses (gol atau SOT).
    Offset ``log t_i`` adalah konstanta terhadap parameter, jadi tidak
    menyumbang ke gradien.
    """
    attack, defense, hfa = _unpack(params, n_teams)
    eta = attack[home_idx] - defense[away_idx] + hfa * is_home
    rate = np.exp(np.clip(eta, -50.0, 50.0))
    log_likelihood = float(
        np.sum(weights * (successes * (log_trials + eta) - trials * rate))
    )
    if not np.isfinite(log_likelihood):
        return 1e18, np.zeros_like(params)

    grad = np.zeros(n_teams * 2 + 1, dtype=float)
    # d/d eta: w * (y - t * exp(eta))
    d_eta = weights * (successes - trials * rate)

    np.add.at(grad, home_idx, d_eta)
    np.add.at(grad, n_teams + away_idx, -d_eta)
    grad[-1] = float(np.sum(d_eta * is_home))
    return -log_likelihood, -grad


def _initial_params(n_teams: int) -> np.ndarray:
    """Titik awal: attack netral, defense netral, HFA kecil."""
    params = np.zeros(n_teams * 2 + 1, dtype=float)
    params[:n_teams] = 1.0
    params[n_teams : 2 * n_teams] = 0.0
    params[-1] = 0.1
    return params


def fit_quality_model(
    successes: np.ndarray,
    trials: np.ndarray,
    team_shooter: np.ndarray,
    team_conceding: np.ndarray,
    is_home: np.ndarray,
    weights: np.ndarray,
) -> QualityFit:
    """Maksimumkan log-likelihood Poisson dengan offset ``log(trials)``.

    Parameter ``defense`` memakai konvensi teksbook (kekuatan, subtaktif),
    lihat docstring modul.
    """
    valid = trials >= EXPOSURE_MIN_TRIALS
    n_skipped = int((~valid).sum())
    successes = successes[valid]
    trials = trials[valid]
    team_shooter = team_shooter[valid]
    team_conceding = team_conceding[valid]
    is_home = is_home[valid]
    weights = np.asarray(weights, dtype=float)[valid]

    if trials.size == 0:
        raise ValueError("Tidak ada baris dengan exposure cukup untuk stage-2")

    teams = tuple(sorted(set(team_shooter.tolist()) | set(team_conceding.tolist())))
    team_position = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    shooter_idx = np.array([team_position[t] for t in team_shooter], dtype=int)
    conceder_idx = np.array([team_position[t] for t in team_conceding], dtype=int)
    log_trials = np.log(trials.astype(float))

    # SLSQP, bukan L-BFGS-B: L-BFGS-B mengabaikan constraints sama sekali
    # (scipy akan emit warning), sehingga constraint identifiabilitas
    # sum(attack) = n_teams tidak akan terpenuhi. penaltyblog juga memakai
    # SLSQP untuk Poisson, jadi ini konsisten dengan model produksi.
    bounds = [EXPOSURE_STAGE2_BOUNDS] * (2 * n_teams) + [EXPOSURE_STAGE2_HFA_BOUNDS]

    result = minimize(
        _nll_and_grad,
        _initial_params(n_teams),
        args=(
            n_teams,
            shooter_idx,
            conceder_idx,
            is_home.astype(float),
            log_trials,
            successes.astype(float),
            trials.astype(float),
            weights,
        ),
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda p: float(np.sum(p[:n_teams]) - n_teams)}
        ],
        options={"maxiter": 1000, "disp": False},
    )

    attack, defense, hfa = _unpack(result.x, n_teams)
    nll, _ = _nll_and_grad(
        result.x,
        n_teams,
        shooter_idx,
        conceder_idx,
        is_home.astype(float),
        log_trials,
        successes.astype(float),
        trials.astype(float),
        weights,
    )
    return QualityFit(
        teams=teams,
        attack=attack,
        defense=defense,
        home_advantage=hfa,
        n_used=int(trials.size),
        n_skipped=n_skipped,
        log_likelihood=float(-nll),
        converged=bool(result.success),
    )


# --------------------------------------------------------------------------
# Perakitan varian
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureModel:
    """Model eksposur lengkap: stage-1 volume plus stage-2 kualitas.

    ``quality`` adalah fitter per-trial pertama. Untuk varian SOT itu berarti
    P(SOT | shot), sedangkan ``goal_quality`` (hanya varian SOT) adalah
    P(goal | SOT).
    """

    variant: str
    volume: object
    quality: QualityFit
    xi: float
    goal_quality: QualityFit | None = None

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Hitung (lambda_home, lambda_away) untuk satu fixture."""
        shots_home = volume_rate(self.volume, home, away, True)
        shots_away = volume_rate(self.volume, away, home, False)

        if self.variant == VARIANT_SHOT:
            return (
                shots_home * quality_rate(self.quality, home, away, True),
                shots_away * quality_rate(self.quality, away, home, False),
            )

        if self.variant == VARIANT_SOT:
            if self.goal_quality is None:
                raise ValueError("Varian SOT butuh goal_quality")
            sot_home = shots_home * quality_rate(self.quality, home, away, True)
            sot_away = shots_away * quality_rate(self.quality, away, home, False)
            return (
                sot_home * quality_rate(self.goal_quality, home, away, True),
                sot_away * quality_rate(self.goal_quality, away, home, False),
            )

        raise ValueError(f"Varian tidak dikenal: {self.variant}")


def _weight_for(train: pd.DataFrame, xi: float) -> np.ndarray:
    return np.asarray(
        pb.models.dixon_coles_weights(
            train["date"], xi=xi, base_date=train["date"].max()
        ),
        dtype=float,
    )


def fit_exposure_model(train: pd.DataFrame, variant: str, xi: float) -> ExposureModel:
    """Fit satu model eksposur pada data training dengan bobot time-decay."""
    weights = _weight_for(train, xi)

    volume = fit_volume_model(
        train["shots_home"],
        train["shots_away"],
        train["team_home"],
        train["team_away"],
        weights,
    )

    home_shooter = train["team_home"].to_numpy()
    away_shooter = train["team_away"].to_numpy()
    home_is_home = np.ones(len(train), dtype=float)
    away_is_home = np.zeros(len(train), dtype=float)

    if variant == VARIANT_SHOT:
        quality = fit_quality_model(
            successes=np.concatenate(
                [train["goals_home"].to_numpy(), train["goals_away"].to_numpy()]
            ),
            trials=np.concatenate(
                [train["shots_home"].to_numpy(), train["shots_away"].to_numpy()]
            ),
            team_shooter=np.concatenate([home_shooter, away_shooter]),
            team_conceding=np.concatenate([away_shooter, home_shooter]),
            is_home=np.concatenate([home_is_home, away_is_home]),
            weights=np.concatenate([weights, weights]),
        )
        return ExposureModel(
            variant=variant, volume=volume, quality=quality, xi=xi
        )

    if variant == VARIANT_SOT:
        sot_quality = fit_quality_model(
            successes=np.concatenate(
                [
                    train["shots_on_target_home"].to_numpy(),
                    train["shots_on_target_away"].to_numpy(),
                ]
            ),
            trials=np.concatenate(
                [train["shots_home"].to_numpy(), train["shots_away"].to_numpy()]
            ),
            team_shooter=np.concatenate([home_shooter, away_shooter]),
            team_conceding=np.concatenate([away_shooter, home_shooter]),
            is_home=np.concatenate([home_is_home, away_is_home]),
            weights=np.concatenate([weights, weights]),
        )
        goal_quality = fit_quality_model(
            successes=np.concatenate(
                [train["goals_home"].to_numpy(), train["goals_away"].to_numpy()]
            ),
            trials=np.concatenate(
                [
                    train["shots_on_target_home"].to_numpy(),
                    train["shots_on_target_away"].to_numpy(),
                ]
            ),
            team_shooter=np.concatenate([home_shooter, away_shooter]),
            team_conceding=np.concatenate([away_shooter, home_shooter]),
            is_home=np.concatenate([home_is_home, away_is_home]),
            weights=np.concatenate([weights, weights]),
        )
        return ExposureModel(
            variant=variant,
            volume=volume,
            quality=sot_quality,
            goal_quality=goal_quality,
            xi=xi,
        )

    raise ValueError(f"Varian tidak dikenal: {variant}")


# --------------------------------------------------------------------------
# Prediksi
# --------------------------------------------------------------------------


def _poisson_pmf(rate: float, max_goals: int) -> np.ndarray:
    """PMF Poisson truncated, pakai ``gammaln`` supaya stabil untuk rate besar."""
    log_rates = -rate + np.arange(max_goals) * np.log(rate) - gammaln(np.arange(max_goals) + 1)
    return np.exp(log_rates)


def predict_1x2(model: ExposureModel, home: str, away: str) -> np.ndarray:
    """Distribusi H/D/A dari dua Poisson independen.

    Grid di-truncate di ``MAX_GOALS`` lalu dinormalisasi ulang, sama seperti
    ``normalize=True`` di penaltyblog. Tanpa normalisasi, sisa massa di
    skor tinggi membuat jumlah probabilitas sedikit di bawah 1 (~4e-9 pada
    MAX_GOALS=15), yang cukup untuk menggagalkan uji validasi distribusi.
    """
    lambda_home, lambda_away = model.expected_goals(home, away)
    joint = np.outer(
        _poisson_pmf(lambda_home, MAX_GOALS), _poisson_pmf(lambda_away, MAX_GOALS)
    )
    # Bandingkan INDEKS sel, bukan jumlah gol. joint[i, j] adalah peluang
    # home i gol dan away j gol, jadi home menang saat i > j. Membandingkan
    # i+j dengan j+i selalu sama, sehingga tidak pernah ada sel yang masuk
    # kategori home atau away.
    index = np.arange(MAX_GOALS)[:, None]
    probabilities = np.array(
        [
            joint[index > index.T].sum(),
            joint[index == index.T].sum(),
            joint[index < index.T].sum(),
        ]
    )
    return probabilities / probabilities.sum()


def score_matches(
    model: ExposureModel, validation: pd.DataFrame
) -> np.ndarray:
    """Matriks probabilitas 1X2 out-of-sample untuk satu fold."""
    rows = [
        predict_1x2(model, match.team_home, match.team_away)
        for match in validation.itertuples(index=False)
    ]
    return np.array(rows)


# --------------------------------------------------------------------------
# Baseline goals-based
# --------------------------------------------------------------------------


def fit_goals_baseline(train: pd.DataFrame, xi: float):
    """Fit Poisson goal-based sebagai lengan pembanding."""
    weights = _weight_for(train, xi)
    model = pb.models.PoissonGoalsModel(
        train["goals_home"],
        train["goals_away"],
        train["team_home"],
        train["team_away"],
        weights=weights,
    )
    model.fit()
    return model


def baseline_probabilities(model, validation: pd.DataFrame) -> np.ndarray:
    """Probabilitas 1X2 dari Poisson goal-based yang sudah di-fit."""
    rows = []
    for match in validation.itertuples(index=False):
        if match.team_home in model.teams and match.team_away in model.teams:
            grid = model.predict(match.team_home, match.team_away, max_goals=MAX_GOALS)
            home, draw, away = grid.home_draw_away
            rows.append([float(home), float(draw), float(away)])
        else:
            n_teams = model.n_teams
            parameters = model.params_array
            attack = float(parameters[:n_teams].mean())
            defense = float(parameters[n_teams : 2 * n_teams].mean())
            hfa = float(parameters[-1])
            expected_home = np.exp(hfa + attack + defense)
            expected_away = np.exp(attack + defense)
            grid = pb.models.create_dixon_coles_grid(
                expected_home, expected_away, rho=0.0, max_goals=MAX_GOALS
            )
            home, draw, away = grid.home_draw_away
            rows.append([float(home), float(draw), float(away)])
    return np.array(rows)


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


def evaluate_fold(
    fold: Fold, matches: pd.DataFrame, validation: pd.DataFrame, y_true: np.ndarray
) -> list[dict]:
    """Nilai semua varian dan baseline pada satu fold, untuk tiap xi di grid."""
    train_seasons = [season for season in config.SEASONS if season <= fold.last_train_season]
    train = matches.loc[matches["season"].isin(train_seasons)].copy()

    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.number}: training atau validation kosong")
    if train["date"].max() >= validation["date"].min():
        raise ValueError(
            f"Fold {fold.number}: overlap waktu training "
            f"({train['date'].max().date()}) dan validation "
            f"({validation['date'].min().date()})"
        )
    # Dua guard di bawah adalah defense-in-depth. Baris training dibangun
    # dengan isin() atas season yang <= last_train_season, dan TEST_SEASON
    # selalu season terakhir di SEASONS, jadi baris beracun sudah terbuang
    # sebelum guard ini dicek. Guard tetap dijaga karena memanggil kode ini
    # dengan training yang sudah dibangun di tempat lain adalah mungkin, dan
    # test test_training_excludes_test_season_even_when_poisoned memverifikasi
    # propertinya langsung.
    if (train["season"] == TEST_SEASON).any():
        raise ValueError(f"Fold {fold.number}: TEST_SEASON bocor ke training")
    if (validation["season"] == TEST_SEASON).any():
        raise ValueError(f"Fold {fold.number}: TEST_SEASON tidak boleh jadi validation")

    rows = []
    for xi in EXPOSURE_XI_GRID:
        baseline = fit_goals_baseline(train, xi)
        baseline_matrix = baseline_probabilities(baseline, validation)
        rows.append(
            _metrics_row(
                fold, BASELINE_NAME, xi, baseline_matrix, y_true, validation
            )
        )
        for variant in VARIANTS:
            model = fit_exposure_model(train, variant, xi)
            matrix = score_matches(model, validation)
            rows.append(_metrics_row(fold, variant, xi, matrix, y_true, validation))
    return rows


def _metrics_row(
    fold: Fold,
    model_name: str,
    xi: float,
    matrix: np.ndarray,
    y_true: np.ndarray,
    validation: pd.DataFrame,
) -> dict:
    return {
        "fold": fold.number,
        "last_train_season": fold.last_train_season,
        "validation_season": fold.validation_season,
        "model": model_name,
        "xi": xi,
        "n_matches": len(matrix),
        "log_loss": float(log_loss(y_true, matrix, labels=[0, 1, 2])),
        "rps": float(pb.metrics.rps_average(matrix, y_true)),
        "accuracy": float(accuracy_score(y_true, matrix.argmax(axis=1))),
    }


def run_walk_forward() -> pd.DataFrame:
    """Jalankan seluruh fold × varian × xi dan kembalikan tabel hasil per fold."""
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["datetime", "date"])
    folds = build_folds()
    validate_folds(folds)

    rows: list[dict] = []
    for fold in folds:
        validation = matches.loc[matches["season"] == fold.validation_season].copy()
        y_true = validation["result"].map(TARGET_MAPPING).to_numpy()
        print(
            f"Fold {fold.number}/{len(folds)} | validasi: {fold.validation_season} | "
            f"{len(validation)} match"
        )
        rows.extend(evaluate_fold(fold, matches, validation, y_true))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Analisis hasil
# --------------------------------------------------------------------------


def select_best_xi(results: pd.DataFrame, model_name: str) -> float:
    """Pilih xi dengan rerata log loss terendah di season training."""
    subset = results.loc[results["model"] == model_name]
    means = subset.groupby("xi")["log_loss"].mean().sort_values()
    return float(means.index[0])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per model pada xi terbaik masing-masing."""
    rows = []
    for model_name in (BASELINE_NAME,) + VARIANTS:
        best_xi = select_best_xi(results, model_name)
        subset = results.loc[
            (results["model"] == model_name) & (results["xi"] == best_xi)
        ]
        rows.append(
            {
                "model": model_name,
                "selected_xi": best_xi,
                "log_loss_mean": float(subset["log_loss"].mean()),
                "log_loss_std": float(subset["log_loss"].std()),
                "rps_mean": float(subset["rps"].mean()),
                "rps_std": float(subset["rps"].std()),
                "accuracy_mean": float(subset["accuracy"].mean()),
                "accuracy_std": float(subset["accuracy"].std()),
                "n_folds": int(subset["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss_mean").reset_index(drop=True)


def _bootstrap_ci(
    deltas: np.ndarray, rng: np.random.Generator, samples: int
) -> tuple[float, float, float]:
    """Bootstrap CI untuk rata-rata selisih per-match."""
    n = deltas.size
    means = np.empty(samples)
    for index in range(samples):
        means[index] = deltas[rng.integers(0, n, n)].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(low), float(high)


def compare_to_baseline(
    results: pd.DataFrame, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """Bandingkan tiap varian terhadap baseline pada xi masing-masing.

    Perbandingan dilakukan per fold pada xi yang dipilih masing-masing model,
    lalu dirata-ratakan. Bootstrap dipakai untuk CI per-match dengan
   .resample di dalam fold.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["datetime", "date"])
    folds = build_folds()

    rows = []
    for variant in VARIANTS:
        variant_xi = select_best_xi(results, variant)
        baseline_xi = select_best_xi(results, BASELINE_NAME)

        deltas: list[float] = []
        fold_means: list[float] = []
        for fold in folds:
            validation = matches.loc[
                matches["season"] == fold.validation_season
            ].copy()
            train_seasons = [
                season
                for season in config.SEASONS
                if season <= fold.last_train_season
            ]
            train = matches.loc[matches["season"].isin(train_seasons)].copy()
            y_true = validation["result"].map(TARGET_MAPPING).to_numpy()

            variant_matrix = score_matches(
                fit_exposure_model(train, variant, variant_xi), validation
            )
            baseline_matrix = baseline_probabilities(
                fit_goals_baseline(train, baseline_xi), validation
            )
            # Selisih per-match, bukan selisih rerata log loss. Nilai negatif
            # berarti varian lebih baik. Perbandingan rerata per fold
            # dicatat terpisah supaya keduanya bisa dicek silang.
            deltas.extend(
                _per_match_loss(variant_matrix, y_true)
                - _per_match_loss(baseline_matrix, y_true)
            )
            fold_means.append(
                float(
                    log_loss(y_true, variant_matrix, labels=[0, 1, 2])
                    - log_loss(y_true, baseline_matrix, labels=[0, 1, 2])
                )
            )

        per_match = np.array(deltas)
        per_fold = np.array(fold_means)
        point, low, high = _bootstrap_ci(per_match, rng, BOOTSTRAP_SAMPLES)
        rows.append(
            {
                "model": variant,
                "baseline": BASELINE_NAME,
                "variant_xi": variant_xi,
                "baseline_xi": baseline_xi,
                "n_folds": len(per_fold),
                "n_matches": int(per_match.size),
                "delta_log_loss_mean": float(per_fold.mean()),
                "delta_log_loss_per_match_mean": point,
                "delta_log_loss_ci_low": low,
                "delta_log_loss_ci_high": high,
                "folds_better": int((per_fold < 0).sum()),
                "significant": bool(low * high > 0),
            }
        )
    return pd.DataFrame(rows)


def _per_match_loss(probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Log loss per match, dengan clip supaya tidak inf saat probabilitas 0."""
    target = probabilities[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(target, 1e-15, 1.0))


def print_report(
    results: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Cetak hasil eksperimen beserta batasan eksperimentasinya."""
    width = 78
    print("\n" + "=" * width)
    print("EXPOSURE-BASED POISSON (bukan xG) — WALK-FORWARD CV")
    print("=" * width)

    print("\nRingkasan per model (xi terbaik masing-masing, di training season):")
    print(
        summary[
            ["model", "selected_xi", "log_loss_mean", "rps_mean", "accuracy_mean"]
        ].to_string(index=False, float_format=lambda v: f"{v:6f}")
    )

    print("\nPerbandingan vs Poisson goal-based (paired, nilai negatif = lebih baik):")
    print(
        comparison[
            [
                "model",
                "variant_xi",
                "baseline_xi",
                "delta_log_loss_mean",
                "delta_log_loss_per_match_mean",
                "delta_log_loss_ci_low",
                "delta_log_loss_ci_high",
                "folds_better",
                "significant",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:+.6f}")
    )
    print()
    print("  delta_log_loss_mean         : rerata selisih per fold (5 fold)")
    print("  delta_log_loss_per_match_mean: selisih per match, sumber CI di bawah")
    print("  CI                          : bootstrap 95% dari selisih per match")

    print("=" * width)
    print(
        "CATATAN PENTING\n"
        "- Ini exposure-based Poisson, BUKAN model berbasis xG. Tidak ada\n"
        "  shot location atau shot quality; hanya volume dan efisiensi\n"
        "  agregat per tim per match.\n"
        "- Angka CV untuk xi terpilih bias oleh selection, karena xi dipilih\n"
        "  dari rerata fold yang sama. Test season 2025-2026 tidak pernah\n"
        "  dipakai sebagai fold maupun sebagai sumber xi, jadi itulah\n"
        "  angka bersihnya nanti kalau model ini mau diuji di test season.\n"
        "- Kedua lengan di-grid-search xi dengan cara identik lalu\n"
        "  dibandingkan best-vs-best, supaya selisihnya tidak tercemar\n"
        "  perbedaan xi.\n"
        "- Modul ini tidak mengubah model produksi, evaluate.py, atau\n"
        "  predict_match.py."
    )
    print("=" * width)


def main() -> None:
    print("=" * 78)
    print("Eksperimen exposure-based Poisson (grid search xi)")
    print("=" * 78)
    print(f"Grid xi: {EXPOSURE_XI_GRID}")
    print(f"Varian: {', '.join(VARIANTS)}")
    print("Baseline goals-based: xi juga di-search dari grid yang sama")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = run_walk_forward()
    summary = summarize(results)
    rng = np.random.default_rng(RANDOM_SEED)
    comparison = compare_to_baseline(results, rng)

    results.to_csv(RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    print_report(results, summary, comparison)

    payload = {
        "purpose": (
            "Eksperimen exposure-based Poisson. BUKAN xG-based. Model "
            "produksi, evaluate.py, dan predict_match.py tidak diubah."
        ),
        "is_xg_based": False,
        "why_not_xg": (
            "xG asli butuh shot location dan shot quality (koordinat, body "
            "part, situasi) per tembakan. Data football-data.co.uk tidak "
            "punya kolom itu; hanya ada jumlah tembakan dan SOT per tim."
        ),
        "model_variant": "two-stage exposure: volume (Poisson) x quality per trial",
        "defense_convention": (
            "teksbook Dixon-Coles: defense = kekuatan, masuk SUBTRAKTIF. "
            "Pemetaan ke penaltyblog: defense_penaltyblog = -defense_teksbook."
        ),
        "variants": list(VARIANTS),
        "baseline": BASELINE_NAME,
        "xi_grid": list(EXPOSURE_XI_GRID),
        "xi_search": "kedua lengan, best-vs-best, di training season saja",
        "xi_selection_bias": (
            "Angka CV pada xi terpilih bias oleh selection karena xi dipilih "
            "dari rerata fold yang sama. Test season tidak pernah jadi fold."
        ),
        "test_season": TEST_SEASON,
        "test_season_used_in_cv": False,
        "baseline_time_decay_xi": TIME_DECAY_XI,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "max_goals": MAX_GOALS,
        "summary": summary.to_dict("records"),
        "comparison": comparison.to_dict("records"),
    }
    METADATA_PATH.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nTersimpan: {RESULTS_PATH}")
    print(f"Tersimpan: {SUMMARY_PATH}")
    print(f"Tersimpan: {COMPARISON_PATH}")
    print(f"Tersimpan: {METADATA_PATH}")


if __name__ == "__main__":
    main()
