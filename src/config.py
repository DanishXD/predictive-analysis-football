"""Shared project constants — single source of truth for all pipeline modules.

Setiap fase (data_collection, feature_engineering, statistical_models,
train_model, stacking, evaluate, value_betting, predict_match, corner_model,
discipline_model, player_stats, player_match_model) mengimport konstanta dari
sini. Jika ada nilai yang perlu diubah (misal ganti test season menjelang musim
baru), cukup ubah di satu tempat ini.
"""

from pathlib import Path

# --- Path dasar project ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Cakupan data ---
COMPETITION = "ENG Premier League"
SEASONS = (
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
)
TEST_SEASON = "2025-2026"

if TEST_SEASON not in SEASONS:
    raise ValueError(
        f"TEST_SEASON ({TEST_SEASON}) harus ada di SEASONS agar split "
        "train/test pipeline tetap valid."
    )

# Season FBref untuk fitur pemain (format FBref: YYYY mis. 2526 = 2025/26).
FBREF_SEASON = "2526"

# --- Target klasifikasi W/D/L ---
TARGET_MAPPING = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = {value: key for key, value in TARGET_MAPPING.items()}

TIME_DECAY_XI = 0.0018

ROLLING_WINDOW = 5

MAX_GOALS = 15
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_DEFAULT_RATING = 1500.0

N_SPLITS = 5
RANDOM_STATE = 42

BOOTSTRAP_SAMPLES = 5000

CORNER_OVER_UNDER_DEFAULT = 9.5
YELLOW_OVER_UNDER_DEFAULT = 4.5
