import re
from pathlib import Path

import config

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

SHARED_CONSTANT_NAMES = (
    "TEST_SEASON",
    "TIME_DECAY_XI",
    "ROLLING_WINDOW",
    "MAX_GOALS",
    "ELO_K",
    "N_SPLITS",
    "RANDOM_STATE",
    "CORNER_OVER_UNDER_DEFAULT",
    "YELLOW_OVER_UNDER_DEFAULT",
)


def test_config_defines_expected_constants():
    for name in SHARED_CONSTANT_NAMES:
        assert hasattr(config, name), f"config.py tidak mendefinisikan {name}"


def test_no_module_redefines_shared_constants():
    pattern = re.compile(
        r"^(" + "|".join(SHARED_CONSTANT_NAMES) + r")\s*=", re.MULTILINE
    )
    offenders = []
    for path in SRC_DIR.glob("*.py"):
        if path.name == "config.py":
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert offenders == [], (
        "Konstanta ini harus diimport dari src/config.py, bukan didefinisikan ulang:\n"
        + "\n".join(offenders)
    )
