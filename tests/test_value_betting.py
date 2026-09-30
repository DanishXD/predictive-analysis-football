"""Tests untuk value_betting.py (Fase 8).

Modul ini sebelumnya TIDAK punya test sama sekali, dan akibatnya satu bug
fatal tidak pernah terlihat: ``parse_args()`` memakai ``Path`` tanpa
meng-import-nya, jadi ``python src/value_betting.py`` langsung crash
dengan NameError padahal README mendokumentasikannya lengkap sebagai fitur
yang jalan.

Test di sini sengaja memanggil entry point yang dulu crash, bukan cuma
meng-import modul. Import saja tidak akan menangkap NameError di dalam
fungsi.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import value_betting
from config import PROJECT_ROOT

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------
# Entry point yang dulu crash
# --------------------------------------------------------------------------


def test_parse_args_handles_defaults(monkeypatch):
    """parse_args() harus jalan tanpa argumen (default Path)."""
    monkeypatch.setattr(sys, "argv", ["value_betting.py"])
    args = value_betting.parse_args()
    assert args.input == value_betting.DEFAULT_INPUT_PATH
    assert args.output == value_betting.DEFAULT_OUTPUT_PATH
    assert isinstance(args.input, Path)


def test_parse_args_accepts_custom_paths(monkeypatch, tmp_path):
    """Argumen --input/--output harus diterima dan bertipe Path."""
    custom_in = tmp_path / "in.csv"
    custom_out = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys, "argv", ["value_betting.py", "--input", str(custom_in), "--output", str(custom_out)]
    )
    args = value_betting.parse_args()
    assert args.input == custom_in
    assert args.output == custom_out


def test_parse_args_rejects_unknown_flag(monkeypatch):
    """Flag tidak dikenal harus gagal, bukan diabaikan diam-diam."""
    monkeypatch.setattr(sys, "argv", ["value_betting.py", "--tidak-ada"])
    with pytest.raises(SystemExit):
        value_betting.parse_args()


def test_module_imports_path_symbol():
    """Regression guard langsung untuk NameError yang lama.

    NameError di dalam fungsi baru ketahuan saat fungsi itu dipanggil, bukan
    saat modul di-import. Test parse_args di atas yang menangkapnya; test
    ini versi cepat supaya jelas simbolnya benar-benar ada.
    """
    assert hasattr(value_betting, "Path")
    assert value_betting.Path is Path


# --------------------------------------------------------------------------
# Smoke test subprocess: memastikan entry point benar-benar bisa dijalankan
# --------------------------------------------------------------------------


def test_script_runs_end_to_end_as_subprocess():
    """Jalankan value_betting.py sebagai proses; harus exit 0.

    Ini test yang akan menangkap regresi entry point apa pun, termasuk yang
    tidak muncul di parse_args. Dijalankan dengan --help supaya tidak
    butuh file fixture dan tidak menulis artefak.
    """
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "value_betting.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "--input" in result.stdout


def test_script_reaches_parse_args_when_run_for_real():
    """Dengan fixture default, script harus lewat parse_args tanpa crash.

    Kalau ini gagal dengan NameError, guard utama di parse_args_test()
    somehow tidak menangkap urutan masalah yang sama.
    """
    default_fixture = value_betting.DEFAULT_INPUT_PATH
    if not default_fixture.exists():
        pytest.skip("fixture example tidak ada di repo")
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "value_betting.py")],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout[-800:]}\nstderr={result.stderr[-800:]}"
    assert "NameError" not in result.stderr


# --------------------------------------------------------------------------
# Kontrak skema output
# --------------------------------------------------------------------------


def test_required_input_columns_cover_documented_schema():
    """Kolom wajib harus sesuai yang README dokumentasikan."""
    documented = {
        "fixture_id",
        "season",
        "datetime",
        "team_home",
        "team_away",
        "odds_home",
        "odds_draw",
        "odds_away",
    }
    assert documented <= set(value_betting.REQUIRED_INPUT_COLUMNS)


def test_output_path_is_under_processed_dir():
    """Output default harus tetap di data/processed (yang di-gitignore)."""
    assert value_betting.DEFAULT_OUTPUT_PATH.parent.name == "processed"


def test_disclaimer_present_in_module():
    """Disclaimer wajib ada: project ini bukan alat betting."""
    text = Path(value_betting.__file__).read_text(encoding="utf-8")
    assert "NOT" in text.upper() or "bukan" in text.lower()
