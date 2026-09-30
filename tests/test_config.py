"""Guard agar konstanta shared hanya didefinisikan di src/config.py."""

import ast
from pathlib import Path

import config

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

SHARED_CONSTANT_NAMES = (
    "TEST_SEASON",
    "TIME_DECAY_XI",
    "ROLLING_WINDOW",
    "MAX_GOALS",
    "ELO_K",
    "ELO_HOME_ADVANTAGE",
    "ELO_DEFAULT_RATING",
    "N_SPLITS",
    "RANDOM_STATE",
    "CORNER_OVER_UNDER_DEFAULT",
    "YELLOW_OVER_UNDER_DEFAULT",
    "SEASONS",
    "COMPETITION",
    "FBREF_SEASON",
    "BOOTSTRAP_SAMPLES",
    "TARGET_MAPPING",
    "TARGET_NAMES",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "EXPOSURE_XI_GRID",
    "EXPOSURE_STAGE2_BOUNDS",
    "EXPOSURE_STAGE2_HFA_BOUNDS",
    "EXPOSURE_MIN_TRIALS",
)


def test_config_defines_expected_constants():
    for name in SHARED_CONSTANT_NAMES:
        assert hasattr(config, name), f"config.py tidak mendefinisikan {name}"


def test_test_season_is_part_of_seasons():
    assert config.TEST_SEASON in config.SEASONS


def _module_level_assigned_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(
                        elt.id for elt in target.elts if isinstance(elt, ast.Name)
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_no_module_redefines_shared_constants():
    offenders = []
    for path in sorted(SRC_DIR.glob("*.py")):
        if path.name == "config.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigned = _module_level_assigned_names(tree)
        offenders.extend(
            f"{path.name}: {name}"
            for name in sorted(assigned.intersection(SHARED_CONSTANT_NAMES))
        )
    assert offenders == [], (
        "Konstanta ini harus diimport dari src/config.py, bukan didefinisikan "
        "ulang:\n" + "\n".join(offenders)
    )
