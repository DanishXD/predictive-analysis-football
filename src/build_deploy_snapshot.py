"""Generate the tracked ``deploy/`` snapshot for the Streamlit web app.

Streamlit Community Cloud clones the repo dan menjalankan ``streamlit run
app.py``. Dua direktori artefak utama ada di ``.gitignore``
(``data/processed/`` dan ``models/``), jadi di Cloud file-nya tidak ada dan
app akan crash sebelum sempat merender.

Daripada membuka pengecualian di ``.gitignore`` untuk seluruh direktori
artefak, script ini menyalin hanya file yang benar-benar dibaca jalur
prediksi ke folder ``deploy/`` yang kecil dan ter-commit. Tidak ada
exception di ``.gitignore`` sama sekali.

Yang TIDAK ikut: ``stacked_random_forest.pkl`` (6.1 MB, tidak dipakai
predict_match), data mentah, dan semua output evaluasi selain dua CSV yang
dibaca aplikasi.

Snapshot bukan sumber kebenaran. Kalau ``TEST_SEASON`` atau pipeline
berubah, jalankan ulang script ini supaya web dan CLI memakai angka yang
sama.

Jalankan dari root project::

    .venv\\Scripts\\python.exe src\\build_deploy_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import MODELS_DIR, PROCESSED_DIR, TEST_SEASON

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "deploy"
DEPLOY_MODELS = DEPLOY_DIR / "models"

# Processed files: matches_clean.csv untuk features, cv_model_selection.csv
# untuk pemilihan model, evaluation_summary.csv sebagai artefak pelaporan,
# elo_current_ratings.csv untuk fitur Elo.
PROCESSED_FILES = [
    "matches_clean.csv",
    "cv_model_selection.csv",
    "evaluation_summary.csv",
    "elo_current_ratings.csv",
]

# Model files. Kedua goal model ikut karena winner dipilih saat runtime dari
# cv_model_selection.csv: kalau winner berubah, snapshot tetap valid tanpa
# perlu regenerate.
MODEL_FILES = [
    "best_ml_model.pkl",
    "best_ml_model_metadata.json",
    "poisson_goal_model.pkl",
    "dixon_coles_goal_model.pkl",
    "corner_poisson_model.pkl",
    "yellow_card_model.pkl",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> dict:
    """Copy satu artefak dan catat provenance-nya."""
    if not source.exists():
        raise FileNotFoundError(
            f"{source} tidak ada. Jalankan pipeline lebih dulu "
            "(Fase 1, 4, 5, 6, 7)."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "file": str(destination.relative_to(DEPLOY_DIR)).replace("\\", "/"),
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def build_snapshot() -> list[dict]:
    """Salin artefak terpilih ke ``deploy/`` dan tulis README-nya."""
    manifest = []
    for name in PROCESSED_FILES:
        manifest.append(_copy(PROCESSED_DIR / name, DEPLOY_DIR / name))
    for name in MODEL_FILES:
        manifest.append(_copy(MODELS_DIR / name, DEPLOY_MODELS / name))

    manifest.append(_write_manifest(manifest))
    _write_readme(manifest)
    return manifest


def _write_manifest(manifest: list[dict]) -> dict:
    """Tulis machine-readable manifest supaya file ini bisa diaudit."""
    path = DEPLOY_DIR / "manifest.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_season": TEST_SEASON,
        "generator": "src/build_deploy_snapshot.py",
        "files": [entry for entry in manifest if entry.get("file") != "manifest.json"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "file": "manifest.json",
        "source": "(dihasilkan)",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_readme(manifest: list[dict]) -> None:
    lines = [
        "# deploy/ snapshot",
        "",
        "Folder ini dibuat oleh `src/build_deploy_snapshot.py`, bukan disalin manual.",
        "Jalankan ulang script itu kalau pipeline, `TEST_SEASON`, atau model berubah.",
        "",
        "## Kenapa ada",
        "",
        "Web app (`app.py`) di Streamlit Community Cloud membaca folder ini, bukan",
        "`data/processed/` dan `models/`. Kedua direktori itu ada di `.gitignore`",
        "karena isinya artefak turunan pipeline, jadi tidak ada di repo hasil clone.",
        "",
        "Alih-alih membuka pengecualian di `.gitignore` untuk seluruh direktori",
        "artefak, hanya file yang benar-benar dibaca jalur prediksi yang disalin ke",
        "sini. Isi folder ini bisa diaudit lewat `manifest.json` (sha256 per file).",
        "",
        "## File",
        "",
        "| file | sumber | ukuran |",
        "|---|---|---|",
    ]
    for entry in manifest:
        if entry["file"] == "manifest.json":
            continue
        lines.append(
            f"| `{entry['file']}` | `{entry['source']}` | "
            f"{entry['bytes'] / (1 << 20):.2f} MB |"
        )

    lines += [
        "",
        "## Catatan pemilihan model",
        "",
        "`cv_model_selection.csv` yang dipakai untuk auto-selection model.",
        "`evaluation_summary.csv` ikut disertakan sebagai artefak pelaporan, tapi",
        "TIDAK boleh dipakai untuk memilih model: isinya metrik test season, dan",
        "memakainya untuk selection adalah kontaminasi test set (AGENTS.md aturan 3).",
        "",
        f"## Test season saat snapshot dibuat",
        "",
        f"`{TEST_SEASON}`",
        "",
        "Kalau `TEST_SEASON` di `src/config.py` berubah, snapshot ini jadi tidak",
        "cocok dan harus di-generate ulang.",
        "",
    ]
    (DEPLOY_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def verify_snapshot() -> pd.DataFrame:
    """Cek sha256 setiap file terhadap manifest."""
    manifest_path = DEPLOY_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("deploy/manifest.json tidak ada; jalankan build_snapshot()")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for entry in payload["files"]:
        path = DEPLOY_DIR / entry["file"]
        actual = _sha256(path) if path.exists() else None
        rows.append(
            {
                "file": entry["file"],
                "exists": path.exists(),
                "sha256_match": actual == entry["sha256"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    manifest = build_snapshot()
    total = sum(entry["bytes"] for entry in manifest)
    print(f"Snapshot ditulis ke {DEPLOY_DIR.relative_to(ROOT)}/ ({len(manifest)} file, "
          f"{total / (1 << 20):.2f} MB)")
    for entry in manifest:
        print(f"  {entry['file']:<48} {entry['bytes'] / (1 << 20):6.2f} MB")
    print()
    print("Verifikasi:")
    print(verify_snapshot().to_string(index=False))


if __name__ == "__main__":
    main()