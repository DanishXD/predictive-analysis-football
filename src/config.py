"""Shared project constants — single source of truth for all pipeline modules.

Setiap fase (statistical_models, train_model, evaluate, feature_engineering,
corner_model, discipline_model) mengimport konstanta dari sini. Jika ada nilai
yang perlu diubah (misal ganti test season menjelang musim baru), cukup ubah
di satu tempat ini.
"""

TEST_SEASON = "2025-2026"

TIME_DECAY_XI = 0.0018

ROLLING_WINDOW = 5

MAX_GOALS = 15
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_DEFAULT_RATING = 1500.0

N_SPLITS = 5
RANDOM_STATE = 42

CORNER_OVER_UNDER_DEFAULT = 9.5
YELLOW_OVER_UNDER_DEFAULT = 4.5
