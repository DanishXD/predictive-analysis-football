# Ringkasan Project: football-predictive-analysis

Project pembelajaran data science end-to-end untuk memprediksi hasil pertandingan English Premier League (EPL). Mulai dari pengumpulan data mentah, feature engineering, pemodelan statistik &amp; machine learning, evaluasi jujur, sampai alat eksplorasi interaktif (CLI).

> ⚠️ **Disclaimer:** Project ini murni untuk pembelajaran data science, **bukan alat rekomendasi taruhan**.

---

## 1. Approach / Model &amp; Target Prediksi

### Track A — Model Statistik (via `penaltyblog`)
| Model | Target |
|---|---|
| Poisson Goal Model | Distribusi skor exact → peluang W/D/L, Over/Under 2.5 |
| Dixon-Coles | Sama seperti Poisson + koreksi skor rendah (parameter `rho`) |
| Elo / Pi-ratings | Selisih rating → peluang W/D/L |

### Track B — Machine Learning
| Model | Target |
|---|---|
| Logistic Regression | 3 kelas: Home Win / Draw / Away Win |
| Random Forest | 3 kelas (model ML terbaik saat ini) |
| XGBoost | 3 kelas |

- Validasi memakai **TimeSeriesSplit** (bukan random k-fold) + `class_weight` dieksplorasi untuk menangani kelas Draw yang minoritas.
- **Stacking (opsional):** output Track A (selisih Elo, expected goals Poisson) ditambahkan sebagai fitur ke model Track B terbaik.

### Model Tambahan
| Model | Target |
|---|---|
| Poisson Corner | Expected corner + Over/Under corner (HC/AC) |
| BTTS | Dihitung langsung dari grid probabilitas skor Poisson/Dixon-Coles (skor ≥1–1) |
| Poisson Team Discipline | Expected kartu kuning (opsional kartu merah) per tim |
| Player Match-Level | Rentang ekspektasi Shots on Target + kandidat "Man of the Match" berdasarkan proyeksi kontribusi gol (xG + xA) |

> Kandidat MOTM adalah proxy statistik, bukan prediksi sesungguhnya — MOTM keputusan subjektif.

---

## 2. Dataset

| Aspek | Detail |
|---|---|
| Sumber utama | **football-data.co.uk** via scraper `penaltyblog` (fallback: download CSV manual) |
| Liga | English Premier League |
| Rentang | **10 musim, 2016/17 s.d. 2025/26** (~**3.800 matches**, 34 tim unik) |
| Split | 3.420 match train (time-decay weighting) vs 380 match test (2025/26, out-of-sample) |
| Data pemain | **FBref** via `soccerdata` |
| Rating awal tim promosi | **ClubElo** (fallback ke rata-rata 3 tim terbawah musim sebelumnya) |

Fitur utama yang dibangun (semua point-in-time, anti data leakage): form 5 pertandingan terakhir, head-to-head, home/away performance split, posisi klasemen (running table), rest days, dan rating Elo. Tim promosi tanpa histori diberi fallback nilai netral.

---

## 3. Stack Teknis

**Python** dengan library:
- `pandas`, `numpy` — data wrangling
- `penaltyblog` — model statistik (Poisson, Dixon-Coles, Elo)
- `scikit-learn`, `xgboost` — model machine learning
- `soccerdata` — data pemain (FBref, ClubElo)
- `jupyter`, `matplotlib`, `seaborn` — eksplorasi &amp; visualisasi
- `joblib` — persist model

Tidak ada neural net / PyTorch / web app / API. Interface hanya **Jupyter notebook** untuk eksplorasi dan **CLI interaktif** (`predict_match.py`).

---

## 4. Tahap Pengerjaan

Sudah selesai **end-to-end** (bukan baru EDA):

1. ✅ Data collection &amp; cleaning (10 musim, 3.800 baris bersih)
2. ✅ Exploratory Data Analysis
3. ✅ Feature engineering (point-in-time, tanpa leakage)
4. ✅ Track A — model statistik
5. ✅ Track B — model ML
6. ✅ Stacking (opsional, percobaan)
7. ✅ Evaluasi menyeluruh vs odds bandar
8. ✅ Value betting test harness
9. ✅ Interactive match predictor (CLI)
10. ✅ Model corner, BTTS, disiplin kartu, prediksi pemain match-level
11. ✅ Integrasi semua output ke satu CLI
12. ✅ **Audit menyeluruh (QA pass): 25/25 item PASS**

---

## 5. Metrik Hasil (test out-of-sample 2025/26, 380 match)

Angka nyata dari `data/processed/evaluation_summary.csv`:

| Model | Accuracy | Log loss | Brier | RPS | ECE |
|---|---|---|---|---|---|
| **Bookmaker odds (benchmark)** | **0.495** | **1.015** | 0.610 | **0.205** | 0.040 |
| Random Forest (best ML) | 0.474 | 1.036 | 0.623 | 0.211 | 0.050 |
| Logistic Regression | 0.476 | 1.047 | 0.633 | 0.215 | 0.063 |
| XGBoost | 0.474 | 1.056 | 0.637 | 0.216 | 0.063 |
| Elo | 0.484 | 1.083 | 0.644 | 0.217 | 0.080 |
| Dixon-Coles | 0.466 | 1.072 | 0.646 | 0.222 | 0.066 |
| Poisson | 0.466 | 1.072 | 0.646 | 0.222 | 0.067 |

**Catatan penting:**
- Odds bandar masih **mengalahkan semua model** pada log loss &amp; RPS — wajar, pasar taruhan relatif efisien.
- Semua model punya **Draw recall ≈ 0** (hanya 0–1 prediksi Draw dari 380 match) — kelas minoritas paling sulit; dicatat sebagai limitation yang acceptable per scope (RPS diprioritaskan).
- Dixon-Coles vs Poisson hampir identik: log-likelihood gain hanya **+0.0027**, parameter `rho ≈ -0.004`.

---

## 6. Status Publikasi / Reusable

Sudah disiapkan serius: **README lengkap**, scope dokumen (`scope-predictive-analysis-bola.md`), `requirements.txt`, `.gitignore`, repo git (`DanishXD/predictive-analysis-football`), dan laporan audit bertahap. Namun masih **eksplorasi / pembelajaran personal** dengan disclaimer eksplisit "bukan alat rekomendasi taruhan". Demo hanya CLI lokal, belum dipublikasikan ke pengguna lain.

---

## 7. Isu &amp; Catatan Terkini

1. **`soccerdata` tidak terinstall** (di-revert) → `player_stats.py` &amp; `player_match_model.py` akan gagal `import` tanpa fallback; cold-start ClubElo otomatis pakai fallback rata-rata 3 tim terbawah. `statistical_models.py` aman karena sudah pakai `try/except`.
2. **`class_weight='balanced'` tidak dipakai** (FIX1 di-revert) — RPS diprioritaskan; Draw recall rendah dianggap acceptable.
3. **Inkonsistensi minor:** docstring `data_collection.py` masih menyebut 5 musim padahal data sudah 10 musim.
4. **Artefak stale:** `models/stacked_random_forest_metadata.json` masih ada, padahal model `.pkl`-nya sudah dihapus saat cleanup stacking.

---

## Struktur Singkat
```
football-predictive-analysis/
├── data/processed/        # CSV bersih, fitur, evaluasi, prediksi
├── notebooks/             # EDA + figure
├── src/                   # 13 script pipeline (Fase 1–15)
├── models/                # Model .pkl tersimpan + metadata JSON
├── *.md                   # README, scope, laporan fix &amp; audit
└── requirements.txt
```
