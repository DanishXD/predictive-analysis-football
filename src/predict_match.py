"""Interactive CLI: score, W/D/L, key players, corner predictions, BTTS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import penaltyblog as pb

from config import (
    CORNER_OVER_UNDER_DEFAULT,
    FBREF_SEASON,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    YELLOW_OVER_UNDER_DEFAULT,
)
from corner_model import predict_corner_fixture, expected_corners, over_under_probability as corner_over_under
from discipline_model import (
    predict_discipline_fixture as predict_yellow_fixture,
    expected_cards as expected_yellow_cards,
    over_under_probability as yellow_over_under,
    first_card_probability,
)
from player_match_model import predict_motm_candidate
from player_stats import fetch_player_stats, get_key_players
from statistical_models import btts_probability, predict_fixture
from value_betting import build_ml_feature_rows


@dataclass(frozen=True)
class _Paths:
    """Direktori artefak, supaya CLI dan web bisa membaca snapshot yang sama.

    Default menunjuk ke output pipeline (``data/processed`` dan ``models/``).
    Web di Streamlit Community Cloud memanggil :func:`use_snapshot_directory`
    lebih dulu supaya membaca folder ``deploy/`` yang ikut ter-commit, karena
    kedua direktori pipeline di-gitignore dan tidak ada di repo.
    """

    processed: Path
    models: Path


_ACTIVE_PATHS: _Paths | None = None
_PREDICTOR_CACHE: Predictor | None = None


def use_snapshot_directory(snapshot_dir: str | Path) -> None:
    """Arahkan pembacaan artefak ke folder snapshot yang ter-commit."""
    global _ACTIVE_PATHS
    root = Path(snapshot_dir)
    _ACTIVE_PATHS = _Paths(processed=root, models=root / "models")
    global _PREDICTOR_CACHE
    _PREDICTOR_CACHE = None


def _pipeline_paths() -> _Paths:
    return _Paths(processed=PROCESSED_DIR, models=MODELS_DIR)


def _paths() -> _Paths:
    return _ACTIVE_PATHS or _pipeline_paths()


MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
# cv_model_selection.csv berisi CV metrics (bukan test metrics) — dipakai untuk
# model selection saja. evaluation_summary.csv (test metrics) tetap ada sebagai
# artefak pelaporan final dan TIDAK boleh dipakai untuk memilih model.
CV_SELECTION_PATH = PROCESSED_DIR / "cv_model_selection.csv"
ELO_RATINGS_PATH = PROCESSED_DIR / "elo_current_ratings.csv"
# Nama file artefak. Direktori come dari _paths() supaya bisa diarahkan ke
# folder deploy/ saat dipanggil dari web; lihat use_snapshot_directory().
_MATCHES_FILENAME = "matches_clean.csv"
_CV_SELECTION_FILENAME = "cv_model_selection.csv"
_ELO_FILENAME = "elo_current_ratings.csv"
_BEST_ML_FILENAME = "best_ml_model.pkl"
_BEST_ML_METADATA_FILENAME = "best_ml_model_metadata.json"
_CORNER_MODEL_FILENAME = "corner_poisson_model.pkl"
_YELLOW_MODEL_FILENAME = "yellow_card_model.pkl"

# cv_model_selection.csv berisi CV metrics (bukan test metrics) — dipakai untuk
# model selection saja. evaluation_summary.csv (test metrics) tetap ada sebagai
# artefak pelaporan final dan TIDAK boleh dipakai untuk memilih model.

GOAL_MODEL_FILENAMES = {
    "poisson": "poisson_goal_model.pkl",
    "dixon_coles": "dixon_coles_goal_model.pkl",
}
GOAL_MODEL_CLASSES = {
    "poisson": pb.models.PoissonGoalsModel,
    "dixon_coles": pb.models.DixonColesGoalModel,
}
GOAL_MODEL_NAMES = {
    "poisson": "Poisson",
    "dixon_coles": "Dixon-Coles",
}
CLASSIFICATION_MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
GOAL_MODEL_CANDIDATES = set(GOAL_MODEL_FILENAMES)
CLASSIFICATION_MODEL_CANDIDATES = set(CLASSIFICATION_MODEL_NAMES)


def select_best_model(
    evaluation: pd.DataFrame,
    candidates: set[str],
) -> pd.Series:
    """Select model dengan cv_log_loss_mean terendah dari kandidat yang tersedia.

    Menggunakan CV metrics (bukan test set) agar model selection tidak
    terkontaminasi oleh performa test — test set hanya untuk pelaporan final.
    Tie-breaker: cv_accuracy_mean tertinggi, lalu nama model (deterministik).
    """
    available = evaluation.loc[evaluation["model"].isin(candidates)].copy()
    if set(available["model"]) != candidates:
        missing = sorted(candidates - set(available["model"]))
        raise ValueError(f"Hasil evaluasi model belum lengkap: {missing}")
    if available["cv_log_loss_mean"].isna().any():
        raise ValueError("cv_log_loss_mean model kandidat tidak lengkap")
    return (
        available
        .sort_values(["cv_log_loss_mean", "cv_accuracy_mean", "model"],
                     ascending=[True, False, True])
        .iloc[0]
    )


def describe_selection_metric(selection: pd.Series) -> str:
    """Label metrik yang jujur mengikuti basis angka selection tersebut.

    Angka Track A di ``cv_model_selection.csv`` sekarang diambil dari
    walk-forward CV (``src/track_a_cv.py``) sehingga sudah out-of-sample dan
    sebanding dengan angka Track B. Tapi jalur in-sample masih ada sebagai
    fallback kalau ``track_a_cv.py`` belum dijalankan, dan basis itulah yang
    harus dicetak apa adanya. Menyatakannya sebagai "CV log loss" membuat
    angka fallback terlihat setara dengan angka Track B padahal tidak
    sebanding, jadi labelnya mengikuti ``selection_basis``, bukan hardcode.
    """
    value = float(selection["cv_log_loss_mean"])
    basis = str(selection["selection_basis"])
    basis_lower = basis.lower()
    # Default-nya out-of-fold, dan hanya turun ke in-sample kalau basisnya
    # secara eksplisit menyatakan begitu. Kalau dibalik, basis seperti
    # "walk-forward musiman, season training saja" akan salah dilabeli
    # in-sample padahal itu justru out-of-sample.
    in_sample_markers = ("no cv", "in-sample", "insample", "train-only", "tidak tersedia")
    is_in_sample = any(marker in basis_lower for marker in in_sample_markers)
    if is_in_sample:
        return f"in-sample log loss: {value:.6f}, basis: {basis} (bukan out-of-fold)"
    return f"CV log loss: {value:.6f}, basis: {basis}"


def display_teams(team_stats: pd.DataFrame, latest_season: str) -> None:
    """Print available teams with compact data-coverage context."""
    print("\nTim EPL yang tersedia di dataset:")
    for index, row in enumerate(team_stats.itertuples(index=False), start=1):
        status = "latest season" if row.last_season == latest_season else f"last: {row.last_season}"
        print(
            f"  {index:>2}. {row.team:<28} "
            f"({row.matches:>3} laga, {row.seasons} musim, {status})"
        )


def prompt_team(
    prompt: str,
    team_stats: pd.DataFrame,
    excluded_team: str | None = None,
) -> str:
    """Prompt until the user enters a valid name or list number."""
    teams = team_stats["team"].tolist()
    casefold_lookup = {team.casefold(): team for team in teams}

    while True:
        value = input(prompt).strip()
        selected = None
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(teams):
                selected = teams[index - 1]
        else:
            selected = casefold_lookup.get(value.casefold())

        if selected is None:
            print(
                "  Tim tidak ditemukan. Masukkan nama persis seperti daftar "
                "atau nomor tim."
            )
            continue
        if selected == excluded_team:
            print("  Tim 2 harus berbeda dari Tim 1. Silakan pilih lagi.")
            continue
        return selected


def prompt_home_team(team_one: str, team_two: str) -> tuple[str, str]:
    """Prompt until the user chooses which selected team plays at home."""
    while True:
        print("\nPilih tim kandang:")
        print(f"  1. {team_one}")
        print(f"  2. {team_two}")
        value = input("Pilihan Home [1/2]: ").strip()
        if value == "1":
            return team_one, team_two
        if value == "2":
            return team_two, team_one
        print("  Input tidak valid. Masukkan 1 atau 2.")


def build_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed EPL coverage for input warnings."""
    home = matches[["season", "date", "team_home"]].rename(
        columns={"team_home": "team"}
    )
    away = matches[["season", "date", "team_away"]].rename(
        columns={"team_away": "team"}
    )
    appearances = pd.concat([home, away], ignore_index=True)
    return (
        appearances.groupby("team")
        .agg(
            matches=("team", "size"),
            seasons=("season", "nunique"),
            last_season=("season", "max"),
            last_match=("date", "max"),
        )
        .reset_index()
        .sort_values("team", ignore_index=True)
    )


def get_prediction_datetime(matches: pd.DataFrame) -> pd.Timestamp:
    """Use current local time, but never a time inside the historical data."""
    current = pd.Timestamp.now().floor("min")
    latest_history = matches["datetime"].max()
    return max(current, latest_history + pd.Timedelta(days=1))


def season_for_date(value: pd.Timestamp) -> str:
    """Convert a date to the usual European football season label."""
    start_year = value.year if value.month >= 7 else value.year - 1
    return f"{start_year}-{start_year + 1}"


def top_scorelines(grid, count: int = 3) -> list[tuple[int, int, float]]:
    """Return the most probable exact scores from a probability grid."""
    flat = grid.grid.ravel()
    top_indices = np.argsort(flat)[::-1][:count]
    rows = []
    for flat_index in top_indices:
        home_goals, away_goals = np.unravel_index(flat_index, grid.grid.shape)
        rows.append((int(home_goals), int(away_goals), float(flat[flat_index])))
    return rows


def team_warnings(
    team: str,
    team_stats: pd.DataFrame,
    latest_season: str,
) -> list[str]:
    """Describe sparse or stale historical coverage for a selected team."""
    row = team_stats.loc[team_stats["team"] == team].iloc[0]
    warnings = []
    if row["seasons"] <= 1 or row["matches"] < 100:
        warnings.append(
            f"{team} cuma punya {int(row['matches'])} laga dari "
            f"{int(row['seasons'])} musim di dataset; estimasinya lebih tidak stabil."
        )
    if row["last_season"] != latest_season:
        warnings.append(
            f"Data EPL terakhir {team} berasal dari {row['last_season']}, "
            "jadi form/rating bisa stale."
        )
    return warnings


def load_match_player_predictions(
    home_team: str,
    away_team: str,
) -> dict:
    """Get top players with season G+A. Match-level SOT skipped (FBref blocked)."""
    try:
        season_data = fetch_player_stats(FBREF_SEASON)
        home_raw = get_key_players(home_team, season_data, top_n=3)
        away_raw = get_key_players(away_team, season_data, top_n=3)
        season_label = FBREF_SEASON[:2] + "/" + FBREF_SEASON[2:]
    except Exception as exc:
        return {
            "home_players": [],
            "away_players": [],
            "source": f"Data pemain tidak tersedia ({exc})",
        }

    def format_players(players_df):
        formatted = []
        for _, row in players_df.iterrows():
            goals = int(row["goals"]) if pd.notna(row.get("goals")) else 0
            assists = int(row["assists"]) if pd.notna(row.get("assists")) else 0
            formatted.append(
                {
                    "player": row["player"],
                    "goals": goals,
                    "assists": assists,
                    "goal_contribution": goals + assists,
                }
            )
        return formatted

    return {
        "home_players": format_players(home_raw),
        "away_players": format_players(away_raw),
        "source": f"FBref season {season_label}",
        "sot_disclaimer": (
            "Estimasi SOT per pemain per match tidak tersedia "
            "- FBref match stats diblokir. Hanya kontribusi gol musim ini."
        ),
    }


def goal_model_path(model_key: str) -> Path:
    """Path file goal model, mengikuti direktori yang sedang aktif."""
    return _paths().models / GOAL_MODEL_FILENAMES[model_key]


def load_corner_model():
    """Load the saved corner Poisson model."""
    return pb.models.PoissonGoalsModel.load(
        str(_paths().models / _CORNER_MODEL_FILENAME)
    )


@dataclass(frozen=True)
class Predictor:
    """Artefak bersama yang dibutuhkan untuk satu prediksi.

    Dipisah dari logika prediksi supaya CLI dan web memuat file yang sama
    dengan cara yang sama. ``load_predictor()`` meng-cache satu instance:
    Streamlit mengeksekusi ulang seluruh script pada setiap perubahan widget,
    jadi tanpa cache setiap perubahan dropdown akan membaca ulang beberapa
    CSV dan membuka ulang model.
    """

    matches: pd.DataFrame
    cv_selection: pd.DataFrame
    metadata: dict
    elo_ratings: dict[str, float]
    team_stats: pd.DataFrame
    goal_selection: pd.Series
    classification_selection: pd.Series
    latest_season: str


def load_predictor() -> Predictor:
    """Load dan cache artefak prediksi yang dipakai bersama CLI dan web.

    Pemilihan model terjadi di sini, dari ``cv_model_selection.csv``, yang
    berisi metrik CV out-of-sample di season training. File itu TIDAK boleh
    digantikan ``evaluation_summary.csv``: yang latter berisi metrik test
    season, jadi memakainya untuk memilih model adalah kontaminasi test set
    (AGENTS.md aturan keras 3).
    """
    global _PREDICTOR_CACHE
    if _PREDICTOR_CACHE is not None:
        return _PREDICTOR_CACHE

    paths = _paths()
    matches = pd.read_csv(
        paths.processed / _MATCHES_FILENAME, parse_dates=["datetime", "date"]
    )
    cv_selection = pd.read_csv(paths.processed / _CV_SELECTION_FILENAME)
    goal_selection = select_best_model(cv_selection, GOAL_MODEL_CANDIDATES)
    classification_selection = select_best_model(
        cv_selection, CLASSIFICATION_MODEL_CANDIDATES
    )

    metadata = json.loads(
        (paths.models / _BEST_ML_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    selected_classifier = classification_selection["model"]
    if metadata["model_name"] != selected_classifier:
        raise ValueError(
            "Model klasifikasi terbaik dari evaluasi adalah "
            f"{selected_classifier}, tetapi artefak tersimpan adalah "
            f"{metadata['model_name']}. Simpan ulang model terpilih dari Fase 5."
        )

    elo_frame = pd.read_csv(paths.processed / "elo_current_ratings.csv")
    elo_ratings = dict(zip(elo_frame["team"], elo_frame["elo_rating"]))

    _PREDICTOR_CACHE = Predictor(
        matches=matches,
        cv_selection=cv_selection,
        metadata=metadata,
        elo_ratings=elo_ratings,
        team_stats=build_team_stats(matches),
        goal_selection=goal_selection,
        classification_selection=classification_selection,
        latest_season=matches["season"].max(),
    )
    return _PREDICTOR_CACHE


def list_teams() -> list[str]:
    """Tim yang pernah muncul di dataset, untuk dropdown.

    Dipakai web supaya user tidak bisa salah ketik nama: CLI memvalidasi
    input teks secara manual, dropdown membuat kategori tertutup.
    """
    return sorted(load_predictor().team_stats["team"].tolist())


def get_prediction(home_team: str, away_team: str) -> dict:
    """Hitung prediksi lengkap untuk satu pasang tim.

    Fungsi ini adalah inti prediksi yang dipakai CLI maupun web, supaya
    keduanya tidak pernah berbeda perhitungan. Prediksi pemain (key player dan
    MOTM) TIDAK termasuk di sini: itu butuh scraping FBref saat runtime, dan
    web v1 memang tidak menampilkannya. CLI menghitungnya sendiri lalu
    meneruskannya ke ``print_prediction()``.
    """
    predictor = load_predictor()
    matches = predictor.matches
    metadata = predictor.metadata

    prediction_datetime = get_prediction_datetime(matches)
    fixture = pd.DataFrame(
        [
            {
                "fixture_id": "interactive-fixture",
                "season": season_for_date(prediction_datetime),
                "datetime": prediction_datetime,
                "team_home": home_team,
                "team_away": away_team,
            }
        ]
    )

    # 1. W/D/L probabilities (ML)
    ml_features = build_ml_feature_rows(
        fixture,
        matches,
        predictor.elo_ratings,
        metadata["feature_columns"],
    )
    ml_model = joblib.load(_paths().models / _BEST_ML_FILENAME)
    probabilities = ml_model.predict_proba(
        ml_features[metadata["feature_columns"]]
    )[0]
    class_to_index = {int(label): index for index, label in enumerate(ml_model.classes_)}
    target_mapping = metadata["target_mapping"]
    ml_probabilities = {
        result: float(probabilities[class_to_index[int(target_mapping[result])]])
        for result in ("H", "D", "A")
    }

    # 2. Score grid (goal model)
    goal_model_key = predictor.goal_selection["model"]
    goal_model = GOAL_MODEL_CLASSES[goal_model_key].load(
        str(goal_model_path(goal_model_key))
    )
    score_grid, goal_cold_start = predict_fixture(
        goal_model,
        home_team,
        away_team,
        dixon_coles=goal_model_key == "dixon_coles",
    )

    # Yellow card predictions
    yellow_threshold = YELLOW_OVER_UNDER_DEFAULT
    try:
        yellow_model = pb.models.PoissonGoalsModel.load(
            str(_paths().models / _YELLOW_MODEL_FILENAME)
        )
        yellow_grid, yellow_cold_start = predict_yellow_fixture(
            yellow_model, home_team, away_team
        )
    except (OSError, ValueError, KeyError, AttributeError):
        yellow_grid = None
        yellow_cold_start = False

    # Corner predictions
    corner_threshold = CORNER_OVER_UNDER_DEFAULT
    try:
        corner_model = load_corner_model()
        corner_grid, corner_cold_start = predict_corner_fixture(
            corner_model, home_team, away_team
        )
    except (OSError, ValueError, KeyError, AttributeError):
        corner_grid = None
        corner_cold_start = False

    # Warnings
    warnings = team_warnings(home_team, predictor.team_stats, predictor.latest_season)
    warnings.extend(team_warnings(away_team, predictor.team_stats, predictor.latest_season))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction_datetime": prediction_datetime,
        "goal_selection": predictor.goal_selection,
        "classification_selection": predictor.classification_selection,
        "score_grid": score_grid,
        "ml_probabilities": ml_probabilities,
        "yellow_grid": yellow_grid,
        "yellow_cold_start": yellow_cold_start,
        "yellow_threshold": yellow_threshold,
        "corner_grid": corner_grid,
        "corner_cold_start": corner_cold_start,
        "corner_threshold": corner_threshold,
        "goal_cold_start": goal_cold_start,
        "warnings": warnings,
    }


def print_prediction(
    home_team: str,
    away_team: str,
    prediction_datetime: pd.Timestamp,
    goal_selection: pd.Series,
    classification_selection: pd.Series,
    score_grid,
    ml_probabilities: dict[str, float],
    match_player_preds: dict,
    motm_result: dict,
    corner_grid,
    corner_cold_start: bool,
    corner_threshold: float,
    yellow_grid,
    yellow_cold_start: bool,
    yellow_threshold: float,
    warnings: list[str],
    goal_cold_start: bool,
) -> None:
    """Render a complete multi-section prediction output."""
    goal_model_key = goal_selection["model"]
    classification_key = classification_selection["model"]
    width = 78
    line = "-" * width

    print("\n" + "=" * width)
    print("PREDIKSI MATCH EPL (ANALISIS PROBABILISTIK)")
    print("=" * width)
    print(f"Fixture : {home_team} (Home) vs {away_team} (Away)")
    print(f"As-of   : {prediction_datetime:%Y-%m-%d %H:%M}")

    # Section 1: Top-3 exact scores
    print(f"\n{line}")
    print("TOP-3 PREDIKSI SKOR")
    print(line)
    print(
        f"Model   : {GOAL_MODEL_NAMES[goal_model_key]} "
        f"({describe_selection_metric(goal_selection)})"
    )
    for rank, (home_goals, away_goals, probability) in enumerate(
        top_scorelines(score_grid), start=1
    ):
        print(
            f"  {rank}. {home_team} {home_goals}-{away_goals} {away_team} "
            f"({probability:.2%})"
        )

    # Section 2: W/D/L probabilities
    print(f"\n{line}")
    print("PELUANG HASIL PERTANDINGAN (W/D/L)")
    print(line)
    print(
        f"Model   : {CLASSIFICATION_MODEL_NAMES[classification_key]} "
        f"({describe_selection_metric(classification_selection)})"
    )
    print(f"  Home Win - {home_team:<28}: {ml_probabilities['H']:.2%}")
    print(f"  Draw{'':<35}: {ml_probabilities['D']:.2%}")
    print(f"  Away Win - {away_team:<28}: {ml_probabilities['A']:.2%}")

    # BTTS
    btts = btts_probability(score_grid)
    print(f"  BTTS (both teams to score)           : {btts:.2%}")

    # Section 3: Match-level player predictions (season G+A)
    print(f"\n{line}")
    print("PREDIKSI LEVEL PEMAIN (KONTRIBUSI GOL)")
    print(line)
    print(f"Sumber  : {match_player_preds.get('source', 'tidak tersedia')}")
    for label, team_name, players in [
        ("KANDANG", home_team, match_player_preds.get("home_players", [])),
        ("TANDANG", away_team, match_player_preds.get("away_players", [])),
    ]:
        print(f"\n  [{label}] {team_name}")
        if not players:
            print("    Data pemain tidak tersedia.")
        else:
            for p in players:
                print(
                    f"    {p['player']:<24}  "
                    f"{p['goal_contribution']:>2} G+A "
                    f"({p['goals']}g, {p['assists']}a)"
                )
    print(f"\n  {match_player_preds.get('sot_disclaimer', '')}")

    # Section 3b: MOTM candidates
    print(f"\n{line}")
    print("KANDIDAT MAN OF THE MATCH")
    print(line)
    print(f"Sumber  : {motm_result.get('label', 'tidak tersedia')}")
    for role, candidates in [
        (f"KANDANG - {home_team}", motm_result.get("home_candidates", [])),
        (f"TANDANG - {away_team}", motm_result.get("away_candidates", [])),
    ]:
        print(f"\n  [{role}]")
        if not candidates:
            print("    Data tidak tersedia.")
        else:
            for candidate in candidates:
                xg_str = f", {candidate['expected_goals']:.1f} xG" if candidate.get("expected_goals") is not None else ""
                print(
                    f"    {candidate['player']:<24}  "
                    f"{candidate['goal_contribution']:>2} G+A "
                    f"({candidate['goals']} gol, {candidate['assists']} assist)"
                    f"{xg_str}"
                )
    print(f"\n  {motm_result.get('disclaimer', '')}")

    # Section 4b: Yellow card predictions
    print(f"\n{line}")
    print("PREDIKSI KARTU KUNING")
    print(line)
    print(f"Model   : Poisson discipline model")
    if yellow_grid is None:
        print("  Model kartu kuning tidak tersedia - prediksi dilewati.")
    else:
        y_exp_home, y_exp_away = expected_yellow_cards(yellow_grid)
        y_ou = yellow_over_under(yellow_grid, yellow_threshold)
        y_fc = first_card_probability(yellow_grid)
        print(f"  Expected yellow cards")
        print(f"    {home_team:<30}: {y_exp_home:.2f}")
        print(f"    {away_team:<30}: {y_exp_away:.2f}")
        print(f"  Total expected                    : {y_exp_home + y_exp_away:.2f}")
        print(f"  Over {y_ou['threshold']:.0f} yellow cards              : {y_ou['over']:.2%}")
        print(f"  Under {y_ou['threshold']:.0f} yellow cards             : {y_ou['under']:.2%}")
        print(f"  {home_team:<30} more yellows: {y_fc['home_more']:.2%}")
        print(f"  {away_team:<30} more yellows: {y_fc['away_more']:.2%}")
        if yellow_cold_start:
            print("  (Memakai prior rata-rata liga untuk cold-start tim)")
        print("  Akurasi model kartu kuning LEBIH RENDAH dari model gol")
        print("  karena dipengaruhi faktor situasional (wasit, tensi pertandingan, dll).")

    # Section 4: Corner predictions
    print(f"\n{line}")
    print("PREDIKSI CORNER")
    print(line)
    if corner_grid is None:
        print("  Model corner tidak tersedia - prediksi dilewati.")
    else:
        print(f"Model   : Poisson corner model")
        exp_home, exp_away = expected_corners(corner_grid)
        ou = corner_over_under(corner_grid, threshold=corner_threshold)
        print(f"  Expected corners")
        print(f"    {home_team:<30}: {exp_home:.2f}")
        print(f"    {away_team:<30}: {exp_away:.2f}")
        print(f"  Total expected                    : {exp_home + exp_away:.2f}")
        print(f"  Over {ou['threshold']:.0f} corners                     : {ou['over']:.2%}")
        print(f"  Under {ou['threshold']:.0f} corners                    : {ou['under']:.2%}")
        if corner_cold_start:
            print("  (Memakai prior rata-rata liga untuk cold-start tim)")
        print("  Akurasi model corner LEBIH RENDAH dari model gol")
        print("  karena corner lebih noisy/random (taktik, gaya main, dll).")

    # Section 5: Warnings
    all_warnings = list(warnings)
    if goal_cold_start:
        all_warnings.append(
            "Model goal-based memakai prior rata-rata liga karena ada tim yang "
            "tidak dikenal saat model di-fit."
        )
    if classification_selection["draw_recall"] < 0.1:
        all_warnings.append(
            f"Recall Draw {CLASSIFICATION_MODEL_NAMES[classification_key]} pada "
            f"test Fase 7 cuma {classification_selection['draw_recall']:.2%}."
        )

    if all_warnings:
        print(f"\n{line}")
        print("PERINGATAN KETERBATASAN")
        print(line)
        for warning in all_warnings:
            print(f"  - {warning}")

    # Final disclaimer
    print(f"\n{line}")
    print("DISCLAIMER")
    print(line)
    print("Ini adalah exercise data science untuk BELAJAR, BUKAN alat rekomendasi bet.")
    print("Model prediksi bola jarang bisa konsisten ngalahin odds bandar.")
    print("Jangan mempertaruhkan uang berdasarkan output project ini.")
    print("=" * width)


def main() -> None:
    """Run the interactive multi-section prediction flow."""
    predictor = load_predictor()
    team_stats = predictor.team_stats
    display_teams(team_stats, predictor.latest_season)

    try:
        team_one = prompt_team("\nPilih Tim 1 (nama/nomor): ", team_stats)
        team_two = prompt_team(
            "Pilih Tim 2 (nama/nomor): ", team_stats, excluded_team=team_one
        )
        home_team, away_team = prompt_home_team(team_one, team_two)
    except (EOFError, KeyboardInterrupt):
        print("\nInput dibatalkan. Tidak ada prediksi yang dibuat.")
        return

    # Perhitungan inti dipakai bersama dengan web, supaya CLI dan web tidak
    # pernah menghasilkan angka berbeda untuk INPUT YANG SAMA. Player/MOTM
    # tetap dihitung di sini saja karena butuh scraping FBref dan tidak
    # dipakai tampilan web.
    result = get_prediction(home_team, away_team)

    match_player_preds = load_match_player_predictions(home_team, away_team)

    try:
        motm_result = predict_motm_candidate(home_team, away_team)
    except Exception as exc:
        motm_result = {
            "home_candidates": [],
            "away_candidates": [],
            "label": "Kandidat MOTM tidak tersedia",
            "disclaimer": f"Tidak dapat memuat data ({exc})",
        }

    # Output
    print_prediction(
        home_team,
        away_team,
        result["prediction_datetime"],
        result["goal_selection"],
        result["classification_selection"],
        result["score_grid"],
        result["ml_probabilities"],
        match_player_preds,
        motm_result,
        result["corner_grid"],
        result["corner_cold_start"],
        result["corner_threshold"],
        result["yellow_grid"],
        result["yellow_cold_start"],
        result["yellow_threshold"],
        result["warnings"],
        result["goal_cold_start"],
    )


if __name__ == "__main__":
    main()
