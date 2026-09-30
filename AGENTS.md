# AGENTS.md — football-predictive-analysis

Project pembelajaran data science end-to-end untuk prediksi hasil pertandingan EPL
(H/D/A + over/under/BTTS/kartu). **Bukan alat taruhan.**

## Wajib dibaca sebelum kerja
- `scope-predictive-analysis-bola.md` — decision log metodologi (leakage, split, metrik). Jangan menabrak keputusan di sana tanpa alasan yang ditulis.
- `README.md` — arsitektur pipeline lengkap.

## Struktur & konvensi
- Semua konstanta pipeline di `src/config.py` — single source of truth. **Jangan** mendefinisikan ulang `TEST_SEASON`, `TIME_DECAY_XI`, dll di module lain; import dari `config`.
- Bahasa output/script: Indonesia casual. Komentar hanya jika diminta.
- Docstring & pesan validasi mengikuti gaya yang ada (bahasa Inggris untuk teknik, Indonesia untuk user-facing).
- Data mentah/processed/models/figures di-gitignore — jangan commit artefak.

## Cara menjalankan pipeline (urutan penting)
Semua dijalankan dari root project dengan interpreter venv:

```
.venv\Scripts\python.exe src\data_collection.py        # Fase 1: fetch + clean (butuh internet)
.venv\Scripts\python.exe src\feature_engineering.py    # Fase 3: fitur anti-leakage
.venv\Scripts\python.exe src\statistical_models.py     # Fase 4: Poisson, Dixon-Coles, Elo
.venv\Scripts\python.exe src\train_model.py            # Fase 5: LogReg/RF/XGBoost
.venv\Scripts\python.exe src\track_a_cv.py             # Fase 6: walk-forward CV Track A (WAJIB)
.venv\Scripts\python.exe src\evaluate.py               # Fase 7: evaluasi semua model vs odds
.venv\Scripts\python.exe src\predict_match.py          # Fase 9: CLI prediksi interaktif
```

**Fase 6 (`src/track_a_cv.py`) wajib, bukan opsional.** Output-nya
(`data/processed/track_a_cv_results.csv`) adalah sumber angka Track A di
`cv_model_selection.csv`. Kalau dilewati, `evaluate.py` jatuh ke angka
**in-sample** yang tidak sebanding dengan Track B dan bisa memilih goal model
yang berbeda — CLI lalu diam-diam memakai Dixon-Coles alih-alih Poisson.
`evaluate.py` akan mencetak peringatan eksplisit kalau file ini belum ada, dan
menolak file yang fold-nya sudah tidak cocok dengan `config.SEASONS` /
`config.TEST_SEASON` (artinya `TEST_SEASON` berubah tapi Fase 6 tidak diulang).
`data/processed/` di-gitignore, jadi di mesin baru file ini selalu hilang
sampai Fase 6 dijalankan.

HATI-HATI: `src/walk_forward_track_b_including_test.py` BUKAN pengganti Fase 6.
Modul itu 6 fold dan fold terakhirnya memakai `TEST_SEASON`, jadi hanya untuk
pelaporan dan tidak boleh dipakai untuk model selection.

Opsional: `src/stacking.py`, `src/corner_model.py`, `src/discipline_model.py`, `src/value_betting.py`.

Eksperimen (tidak mengubah model produksi, tidak mengubah `cv_model_selection.csv`):
`src/model_calibration.py`, `src/model_tuning.py`,
`src/exposure_poisson.py`, `src/blend.py`, `src/seasonal_hfa.py`.
Hasil negatif yang sudah tercatat ada di `HANDOFF.md` bagian 5 — baca dulu sebelum
mencoba pendekatan yang sama lagi.

Fitur pemain (`src/player_stats.py`, `src/player_match_model.py`) butuh `pip install soccerdata` (tidak di-install default). Tanpa itu, script menampilkan pesan fallback dan tidak crash.

## Tes
```
.venv\Scripts\python.exe -m pytest tests/ -q
```
Jalankan tes setiap kali ubah apapun di `src/`, terutama:
- `src/config.py` (konstanta)
- `src/feature_engineering.py` (logika anti-leakage)
- `src/team_mapping.py` (mapping nama tim)

## Aturan keras
1. **Anti-leakage:** fitur match T hanya boleh memakai data sebelum T. Split waktu: semua season != `TEST_SEASON` = train, `TEST_SEASON` = test. TimeSeriesSplit untuk CV, bukan random k-fold.
2. **Odds bandar adalah benchmark, bukan fitur.**
3. Pilihan model terbaik harus berdasarkan CV log loss, **bukan** hasil test set.
4. Evaluasi jujur: jika metrik berubah setelah refactor, jelaskan kenapa. Differences pada digit float terakhir (~1e-16) akibat non-determinisme library adalah wajar; perbedaan materiil harus diinvestigasi.
5. `models/` dan `data/processed/` bisa diambil lagi dengan menjalankan ulang pipeline — jangan memanipulasi isinya secara manual kecuali metadata JSON.

## Mengganti test season
1. Ubah `TEST_SEASON` di `src/config.py` (satu tempat saja).
2. Jalankan ulang Fase 1 → 3 → 4 → 5 → **6** → 7. Fase 6 wajib diulang juga:
   `track_a_cv.py` menghitung fold-nya dari `config.SEASONS`, jadi mengganti
   `TEST_SEASON` membuat fold di `track_a_cv_results.csv` yang tertinggal tidak
   lagi valid. Kalau lupa, `evaluate.py` menolaknya dengan pesan fold stale,
   bukan diam-diam memakai angka training period lama.
3. Cek output warning staleness dari `predict_match.py`.

## Known limitations (jangan "perbaiki" tanpa diskusi)
- Elo: promoted teams coba rating ClubElo dulu; kalau gagal, fallback ke rata-rata bottom-3 musim sebelumnya (bukan Elo kontinental). Elo kontinental rasanya terlalu kompleks untuk scope pembelajaran.
- Dixon-Coles rho ~ -0.004 (tidak banyak membantu), tetap dipakai karena bagian dari scope.
- Kenapa mesin odds bandar? Odds mengandung info yang tidak diwujudnyatakan di fitur kita (cedera, line-up, market wisdom). Target realistis: memperkecil gap RPS/log loss, bukan mengalahkannya dengan tipuan.
