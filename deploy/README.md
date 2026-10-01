# deploy/ snapshot

Folder ini dibuat oleh `src/build_deploy_snapshot.py`, bukan disalin manual.
Jalankan ulang script itu kalau pipeline, `TEST_SEASON`, atau model berubah.

## Kenapa ada

Web app (`app.py`) di Streamlit Community Cloud membaca folder ini, bukan
`data/processed/` dan `models/`. Kedua direktori itu ada di `.gitignore`
karena isinya artefak turunan pipeline, jadi tidak ada di repo hasil clone.

Alih-alih membuka pengecualian di `.gitignore` untuk seluruh direktori
artefak, hanya file yang benar-benar dibaca jalur prediksi yang disalin ke
sini. Isi folder ini bisa diaudit lewat `manifest.json` (sha256 per file).

## File

| file | sumber | ukuran |
|---|---|---|
| `matches_clean.csv` | `data/processed/matches_clean.csv` | 0.77 MB |
| `cv_model_selection.csv` | `data/processed/cv_model_selection.csv` | 0.00 MB |
| `evaluation_summary.csv` | `data/processed/evaluation_summary.csv` | 0.00 MB |
| `elo_current_ratings.csv` | `data/processed/elo_current_ratings.csv` | 0.00 MB |
| `models/best_ml_model.pkl` | `models/best_ml_model.pkl` | 0.52 MB |
| `models/best_ml_model_metadata.json` | `models/best_ml_model_metadata.json` | 0.00 MB |
| `models/poisson_goal_model.pkl` | `models/poisson_goal_model.pkl` | 0.70 MB |
| `models/dixon_coles_goal_model.pkl` | `models/dixon_coles_goal_model.pkl` | 0.70 MB |
| `models/corner_poisson_model.pkl` | `models/corner_poisson_model.pkl` | 0.70 MB |
| `models/yellow_card_model.pkl` | `models/yellow_card_model.pkl` | 0.70 MB |

## Catatan pemilihan model

`cv_model_selection.csv` yang dipakai untuk auto-selection model.
`evaluation_summary.csv` ikut disertakan sebagai artefak pelaporan, tapi
TIDAK boleh dipakai untuk memilih model: isinya metrik test season, dan
memakainya untuk selection adalah kontaminasi test set (AGENTS.md aturan 3).

## Test season saat snapshot dibuat

`2025-2026`

Kalau `TEST_SEASON` di `src/config.py` berubah, snapshot ini jadi tidak
cocok dan harus di-generate ulang.
