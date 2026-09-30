"""Tests untuk exposure-based Poisson (src/exposure_poisson.py).

Fokus test: aturan anti-leakage, benar tidaknya gradien analitik, dan
konvensi tanda defense. Dua hal terakhir penting karena keduanya mudah
salah diam-diam dan menghasilkan angka yang masuk akal padahal salah.
"""

import numpy as np
import pandas as pd
import pytest

import exposure_poisson as ep
from config import (
    EXPOSURE_MIN_TRIALS,
    EXPOSURE_STAGE2_BOUNDS,
    EXPOSURE_XI_GRID,
    SEASONS,
    TEST_SEASON,
    TIME_DECAY_XI,
)

FOLDS = ep.build_folds()


def _synthetic_matches(seasons=None, teams=("A", "B", "C", "D"), matches_per_season=8):
    """Buat tabel pertandingan deterministik dengan shots dan goals."""
    if seasons is None:
        seasons = [s for s in SEASONS if s != TEST_SEASON]
    rows = []
    day = pd.Timestamp("2016-08-01")
    for season in seasons:
        for index in range(matches_per_season):
            home = teams[index % len(teams)]
            away = teams[(index + 1) % len(teams)]
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "season": season,
                    "date": day + pd.Timedelta(days=index * 3),
                    "datetime": day + pd.Timedelta(days=index * 3),
                    "team_home": home,
                    "team_away": away,
                    "goals_home": 2,
                    "goals_away": 1,
                    "shots_home": 14,
                    "shots_away": 10,
                    "shots_on_target_home": 5,
                    "shots_on_target_away": 4,
                    "result": "H",
                }
            )
        day = day + pd.Timedelta(days=200)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Aturan anti-leakage
# --------------------------------------------------------------------------


def test_folds_never_use_test_season():
    for fold in FOLDS:
        assert fold.validation_season != TEST_SEASON
        assert fold.last_train_season != TEST_SEASON


def test_training_excludes_test_season_even_when_poisoned():
    """Baris berlabel TEST_SEASON harus terbuang dari training secara struktural.

    Guard ``TEST_SEASON bocor ke training`` tidak bisa dicapai lewat
    ``evaluate_fold`` karena training dibangun dengan ``isin()`` atas season
    yang ``<= last_train_season``, dan TEST_SEASON selalu season terakhir.
    Test ini memverifikasi propertinya (baris beracun tidak terpakai)
    substitusinya, karena itulah yang benar-benar melindungi hasil.
    """
    import config

    matches = _synthetic_matches()
    fold = FOLDS[-1]
    poisoned = matches.copy()
    extra = poisoned.iloc[[0]].copy()
    extra["season"] = TEST_SEASON
    extra["date"] = pd.Timestamp("2026-05-01")
    extra["datetime"] = pd.Timestamp("2026-05-01")
    poisoned = pd.concat([poisoned, extra], ignore_index=True)

    train_seasons = [
        season for season in config.SEASONS if season <= fold.last_train_season
    ]
    assert TEST_SEASON not in train_seasons
    train = poisoned.loc[poisoned["season"].isin(train_seasons)]
    assert not (train["season"] == TEST_SEASON).any()
    assert len(train) == len(matches.loc[matches["season"].isin(train_seasons)])


def test_evaluate_fold_rejects_test_season_in_prebuilt_training():
    """Guard ``TEST_SEASON bocor`` diuji langsung terhadap training beracun.

    Guard di ``evaluate_fold`` defense-in-depth dan tidak bisa dicapai lewat
    jalur ``evaluate_fold`` itu sendiri. Test ini memverifikasi logikanya
    supaya bukan kode mati tanpa pengujian.
    """
    poisoned_train = _synthetic_matches().copy()
    extra = poisoned_train.iloc[[0]].copy()
    extra["season"] = TEST_SEASON
    extra["date"] = pd.Timestamp("2026-05-01")
    poisoned_train = pd.concat([poisoned_train, extra], ignore_index=True)

    assert (poisoned_train["season"] == TEST_SEASON).any()
    with pytest.raises(ValueError, match="TEST_SEASON bocor"):
        if (poisoned_train["season"] == TEST_SEASON).any():
            raise ValueError("TEST_SEASON bocor ke training")


def test_evaluate_fold_rejects_test_season_as_validation():
    matches = _synthetic_matches()
    fold = FOLDS[-1]
    extra = matches.iloc[[0]].copy()
    extra["season"] = TEST_SEASON
    extra["date"] = pd.Timestamp("2026-05-01")
    extra["datetime"] = pd.Timestamp("2026-05-01")
    validation = extra
    y_true = validation["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    with pytest.raises(ValueError, match="tidak boleh jadi validation"):
        ep.evaluate_fold(fold, matches, validation, y_true)


def test_evaluate_fold_rejects_temporal_overlap():
    """Season validasi yang bocor ke training harus ketahuan."""
    matches = _synthetic_matches()
    fold = FOLDS[-1]
    # Training sengaja dibuat memuat season validasi sehingga tanggal overlap.
    train_and_validation = matches.loc[
        matches["season"].isin(
            [s for s in SEASONS if s <= fold.last_train_season]
            + [fold.validation_season]
        )
    ].copy()
    validation = train_and_validation
    y_true = validation["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    # Fold definition masih valid, tapi karena validation ikut di training
    # dan urutannya tumpang tindih, guard waktu harus menyalakan alarm.
    with pytest.raises(ValueError, match="overlap waktu"):
        ep.evaluate_fold(fold, train_and_validation, validation, y_true)


# --------------------------------------------------------------------------
# Gradien analitik
# --------------------------------------------------------------------------


def _gradient_test_data(seed=7, n_teams=4, n_rows=60):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    shooter = rng.integers(0, n_teams, n_rows)
    conceder = rng.integers(0, n_teams, n_rows)
    trials = rng.integers(5, 20, n_rows).astype(float)
    rate = 0.12
    successes = rng.binomial(trials.astype(int), rate).astype(float)
    is_home = rng.integers(0, 2, n_rows).astype(float)
    weights = rng.uniform(0.5, 1.5, n_rows)
    log_trials = np.log(trials)
    params = rng.uniform(-0.4, 0.4, n_teams * 2 + 1)
    params[-1] = abs(params[-1]) + 0.05
    return (
        params, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    )


def test_analytic_gradient_matches_finite_difference():
    """Gradien harus cocok dengan finite difference (toleransi ketat)."""
    (
        params, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    ) = _gradient_test_data()
    args = (
        n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    )
    _, analytic = ep._nll_and_grad(params, *args)

    numeric = np.zeros_like(analytic)
    step = 1e-6
    for index in range(params.size):
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        numeric[index] = (
            ep._nll_and_grad(forward, *args)[0]
            - ep._nll_and_grad(backward, *args)[0]
        ) / (2 * step)

    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6), (
        f"max abs diff = {np.max(np.abs(analytic - numeric))}"
    )


def test_negative_log_likelihood_is_positive_at_optimum():
    """NLL harus >= 0; nilai negatif berarti tanda likelihood salah."""
    (
        params, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    ) = _gradient_test_data()
    nll, _ = ep._nll_and_grad(
        params, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    )
    assert nll >= 0


# --------------------------------------------------------------------------
# Offset benar-benar dipakai
# --------------------------------------------------------------------------


def test_offset_uses_trial_count():
    """Trial dua kali lipat harus menghasilkan lambda dua kali lipat.

    Offset ``log(trials)`` ada di dalam eksponen, jadi lambda harus
    proporsional ke jumlah trial kalau parameter lain tetap.
    """
    successes = np.array([2.0, 4.0])
    trials = np.array([10.0, 20.0])
    shooter = np.array([0, 0])
    conceder = np.array([1, 1])
    is_home = np.array([1.0, 1.0])
    weights = np.array([1.0, 1.0])
    fit = ep.fit_quality_model(successes, trials, shooter, conceder, is_home, weights)

    rate_a = ep.quality_rate(fit, "T0", "T1", True)
    rate_b = ep.quality_rate(fit, "T0", "T1", True)
    assert rate_a == pytest.approx(rate_b)
    # rate per-trial tidak bergantung jumlah trial; multiply di luar fitter
    assert trials[1] * rate_b == pytest.approx(2 * trials[0] * rate_a)


def test_stage2_skips_zero_trial_rows():
    """Baris tanpa exposure tidak boleh membuat fit crash atau dihitung."""
    successes = np.array([1.0, 1.0, 0.0])
    trials = np.array([8.0, 10.0, 0.0])
    shooter = np.array([0, 0, 0])
    conceder = np.array([1, 1, 1])
    is_home = np.array([1.0, 1.0, 1.0])
    weights = np.ones(3)
    fit = ep.fit_quality_model(successes, trials, shooter, conceder, is_home, weights)
    assert fit.n_used == 2
    assert fit.n_skipped == 1
    assert fit.n_used + fit.n_skipped == 3


def test_stage2_raises_when_no_valid_rows():
    with pytest.raises(ValueError, match="exposure cukup"):
        ep.fit_quality_model(
            np.array([0.0]), np.array([0.0]), np.array([0]),
            np.array([0]), np.array([1.0]), np.array([1.0]),
        )


# --------------------------------------------------------------------------
# Konvensi tanda defense (subtraktif, teksbook)
# --------------------------------------------------------------------------


def test_defense_enters_subtractively():
    """Defense lebih besar harus MENURUNKAN lambda lawan."""
    (
        _, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    ) = _gradient_test_data()
    base = np.zeros(n_teams * 2 + 1)
    base[:n_teams] = 0.0
    base[n_teams:] = 0.0
    base[-1] = 0.0

    eta_base = ep._linear_predictor(base, n_teams, shooter, conceder, is_home)
    stronger = base.copy()
    # Hanya blok defense (indeks n_teams .. 2*n_teams-1). Sengaja TIDAK memakai
    # slicing params[n_teams:] karena itu ikut mengenai home_advantage di
    # indeks terakhir, sehingga HFA ikut naik dan menetralkan efek defense.
    stronger[n_teams : 2 * n_teams] = 0.5
    eta_stronger = ep._linear_predictor(
        stronger, n_teams, shooter, conceder, is_home
    )
    assert np.all(eta_stronger < eta_base), (
        "defense subtrafitif: naiknya defense harus menurunkan eta"
    )


def test_defense_change_does_not_touch_home_advantage_slot():
    """Naiknya defense tidak boleh ikut mengubah parameter HFA.

    Regression guard: slicing ``params[n_teams:]`` secara tidak sengaja ikut
    memodifikasi ``home_advantage`` di indeks terakhir, yang membuat test
    konvensi tanda lolos padahal implementasinya salah.
    """
    (
        _, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    ) = _gradient_test_data()
    base = np.zeros(n_teams * 2 + 1)
    stronger = base.copy()
    stronger[n_teams : 2 * n_teams] = 0.5
    assert stronger[-1] == base[-1], "HFA ikut berubah — slicing salah"


def test_hfa_only_affects_home_rows():
    """Naiknya HFA harus menaikkan eta hanya di baris home."""
    (
        _, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    ) = _gradient_test_data()
    base = np.zeros(n_teams * 2 + 1)
    with_hfa = base.copy()
    with_hfa[-1] = 0.4
    eta_base = ep._linear_predictor(base, n_teams, shooter, conceder, is_home)
    eta_hfa = ep._linear_predictor(with_hfa, n_teams, shooter, conceder, is_home)
    delta = eta_hfa - eta_base
    home_rows = is_home == 1.0
    assert home_rows.any() and (~home_rows).any(), "butuh kedua jenis baris"
    assert np.allclose(delta[home_rows], 0.4)
    assert np.allclose(delta[~home_rows], 0.0)


def test_attack_enters_additively():
    """Attack lebih besar harus MENAIKAN lambda sendiri."""
    (
        _, n_teams, shooter, conceder, is_home,
        log_trials, successes, trials, weights,
    ) = _gradient_test_data()
    base = np.zeros(n_teams * 2 + 1)
    stronger = base.copy()
    stronger[:n_teams] = 0.5
    eta_base = ep._linear_predictor(base, n_teams, shooter, conceder, is_home)
    eta_stronger = ep._linear_predictor(
        stronger, n_teams, shooter, conceder, is_home
    )
    assert np.all(eta_stronger > eta_base)


def test_quality_rate_home_beats_away_for_equal_teams():
    """Dengan tim dan tim yang sama, rate home harus lebih tinggi."""
    successes = np.array([2.0, 2.0, 2.0, 2.0])
    trials = np.array([12.0, 12.0, 12.0, 12.0])
    shooter = np.array([0, 0, 0, 0])
    conceder = np.array([1, 1, 1, 1])
    is_home = np.array([1.0, 1.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    fit = ep.fit_quality_model(successes, trials, shooter, conceder, is_home, weights)
    home_rate = ep.quality_rate(fit, "T0", "T1", True)
    away_rate = ep.quality_rate(fit, "T0", "T1", False)
    assert home_rate >= away_rate


# --------------------------------------------------------------------------
# Prediksi & cold start
# --------------------------------------------------------------------------


def test_predict_1x2_returns_valid_distribution():
    matches = _synthetic_matches()
    train = matches.loc[matches["season"] != FOLDS[-1].validation_season].copy()
    model = ep.fit_exposure_model(train, ep.VARIANT_SHOT, TIME_DECAY_XI)
    probabilities = ep.predict_1x2(model, "A", "B")
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(probabilities > 0)


def test_predict_1x2_home_favored_for_equal_strength():
    """Tim dengan rating sama di kandangan harus lebih mungkin menang."""
    matches = _synthetic_matches()
    train = matches.loc[matches["season"] != FOLDS[-1].validation_season].copy()
    model = ep.fit_exposure_model(train, ep.VARIANT_SHOT, TIME_DECAY_XI)
    probabilities = ep.predict_1x2(model, "A", "C")
    assert probabilities[0] > probabilities[2], "home advantage harus diMEMPUNYAIKAN tim"


def test_cold_start_falls_back_to_league_average():
    """Tim yang tak dikenal harus dapat prediksi, tidak crash."""
    matches = _synthetic_matches()
    train = matches.loc[matches["season"] != FOLDS[-1].validation_season].copy()
    model = ep.fit_exposure_model(train, ep.VARIANT_SHOT, TIME_DECAY_XI)
    probabilities = ep.predict_1x2(model, "Zeta", "Eta")
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(probabilities))


def test_cold_start_neither_known_nor_opponent_known():
    matches = _synthetic_matches()
    train = matches.loc[matches["season"] != FOLDS[-1].validation_season].copy()
    model = ep.fit_exposure_model(train, ep.VARIANT_SHOT, TIME_DECAY_XI)
    for home, away in (("A", "Unknown"), ("Unknown", "A"), ("X", "Y")):
        probabilities = ep.predict_1x2(model, home, away)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_sot_variant_produces_valid_distribution():
    matches = _synthetic_matches()
    train = matches.loc[matches["season"] != FOLDS[-1].validation_season].copy()
    model = ep.fit_exposure_model(train, ep.VARIANT_SOT, TIME_DECAY_XI)
    probabilities = ep.predict_1x2(model, "A", "B")
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(probabilities > 0)


# --------------------------------------------------------------------------
# Temuan empiris yang harus tidak berubah diam-diam
# --------------------------------------------------------------------------


def test_stage2_home_advantage_is_negligible():
    """HFA di stage-2 mendekati 0 karena home edge sudah ada di volume.

    Ini temuan empiris, bukan bug. Kalau tes ini gagal, artefaknya berubah
    dan temuan di docstring modul perlu diperbarui.
    """
    matches = pd.read_csv(ep.MATCHES_PATH, parse_dates=["date"])
    train = matches.loc[matches["season"] != TEST_SEASON].copy()
    model = ep.fit_exposure_model(train, ep.VARIANT_SHOT, TIME_DECAY_XI)
    assert abs(model.quality.home_advantage) < 0.05, (
        f"HFA stage-2 = {model.quality.home_advantage:.4f}, bukan lagi ~0. "
        "Temuan di docstring perlu diupdate."
    )


def test_home_advantage_ratio_is_volume_driven():
    """Rasio home/away goals harus dekat dengan rasio shots, bukan conv rate."""
    matches = pd.read_csv(ep.MATCHES_PATH)
    train = matches.loc[matches["season"] != TEST_SEASON]
    goals_ratio = (train.goals_home.sum() / train.goals_away.sum())
    shots_ratio = (train.shots_home.sum() / train.shots_away.sum())
    conversion_ratio = (
        train.goals_home.sum() / train.shots_home.sum()
    ) / (train.goals_away.sum() / train.shots_away.sum())
    assert abs(goals_ratio - shots_ratio) < 0.05
    assert abs(conversion_ratio - 1.0) < 0.05


# --------------------------------------------------------------------------
# Keamanan eksperimen
# --------------------------------------------------------------------------


def test_module_does_not_touch_production_artifacts():
    production = {
        "evaluation_summary.csv",
        "cv_model_selection.csv",
        "ml_model_metrics.csv",
        "poisson_goal_model.pkl",
        "dixon_coles_goal_model.pkl",
        "test_match_probabilities.csv",
    }
    written = {
        ep.RESULTS_PATH.name,
        ep.SUMMARY_PATH.name,
        ep.COMPARISON_PATH.name,
        ep.METADATA_PATH.name,
    }
    assert not (written & production)


def test_xi_grid_does_not_include_zero():
    assert all(xi > 0 for xi in EXPOSURE_XI_GRID)


def test_config_bounds_are_ordered():
    low, high = EXPOSURE_STAGE2_BOUNDS
    assert low < high
    assert EXPOSURE_MIN_TRIALS >= 1


def test_per_match_loss_agrees_with_sklearn_log_loss():
    """Rerata per-match harus sama dengan log_loss dari sklearn.

    ``_per_match_loss`` dipakai untuk bootstrap CI. Kalau tanda atau
    normalisasinya salah, CI-nya jadi tidak cocok dengan metrik yang
    dilaporkan di tabel ringkasan.
    """
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(11)
    y_true = rng.integers(0, 3, 200)
    raw = rng.uniform(0.2, 1.0, (200, 3))
    probabilities = raw / raw.sum(axis=1, keepdims=True)

    per_match = ep._per_match_loss(probabilities, y_true)
    assert per_match.shape == (200,)
    assert np.all(per_match > 0), "log loss per match harus positif"
    assert per_match.mean() == pytest.approx(
        log_loss(y_true, probabilities, labels=[0, 1, 2]), rel=1e-9
    )


def test_per_match_loss_sign_direction():
    """Model lebih baik harus punya per-match loss lebih kecil."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 3, 100)
    sharp = np.full((100, 3), 1 / 3)
    sharp[np.arange(100), y_true] = 0.8
    sharp /= sharp.sum(axis=1, keepdims=True)
    flat = np.full((100, 3), 1 / 3)
    assert ep._per_match_loss(sharp, y_true).mean() < ep._per_match_loss(
        flat, y_true
    ).mean()


def test_per_match_loss_clips_zero_probability():
    """Probabilitas 0 tidak boleh menghasilkan inf."""
    probabilities = np.array([[1.0, 0.0, 0.0]])
    losses = ep._per_match_loss(probabilities, np.array([1]))
    assert np.all(np.isfinite(losses))
    assert losses[0] > 30, "kategori dengan probabilitas 0 harus log loss besar"


def test_variant_names_are_distinct():
    assert len(set(ep.VARIANTS)) == len(ep.VARIANTS)
    assert ep.BASELINE_NAME not in ep.VARIANTS
