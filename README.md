# EPL Predictive Analysis

Project pembelajaran end-to-end untuk memprediksi hasil pertandingan English Premier League — dari pengumpulan data sampai model statistik, machine learning, dan alat eksplorasi interaktif.

> ⚠️ **Disclaimer singkat:** Project ini murni untuk pembelajaran data science, **bukan alat rekomendasi taruhan**. Lihat bagian [Keterbatasan & Disclaimer](#keterbatasan--disclaimer) sebelum menggunakan output apa pun untuk keputusan finansial.

Scope dan keputusan teknis lengkap (termasuk alasan di balik setiap pilihan metodologi — split training/test, penanganan data leakage, dsb) tersedia di [`scope-predictive-analysis-bola.md`](./scope-predictive-analysis-bola.md).

## Daftar Isi
- [Tech Stack](#tech-stack)
- [Struktur Project](#struktur-project)
- [Instalasi](#instalasi)
- [Sumber Data](#sumber-data)
- [Pipeline](#pipeline)
  - [Fase 1: Data Collection & Cleaning](#fase-1-data-collection--cleaning)
  - [Fase 2: Exploratory Data Analysis](#fase-2-exploratory-data-analysis)
  - [Fase 3: Feature Engineering](#fase-3-feature-engineering)
  - [Fase 4: Track A — Model Statistik](#fase-4-track-a--model-statistik)
  - [Fase 5: Track B — Model Machine Learning](#fase-5-track-b--model-machine-learning)
  - [Fase 6: Stacking (Opsional)](#fase-6-stacking-opsional)
  - [Fase 7: Evaluasi Menyeluruh](#fase-7-evaluasi-menyeluruh)
  - [Fase 8: Value Betting Testing](#fase-8-value-betting-testing)
  - [Fase 9: Interactive Match Predictor](#fase-9-interactive-match-predictor)
  - [Fase 10 & 13: Player Statistics & Match-Level Predictions](#fase-10--13-player-statistics--match-level-predictions)
  - [Fase 11: Corner Model & BTTS](#fase-11-corner-model--btts)
  - [Fase 12 & 15: Integrasi ke Predictor](#fase-12--15-integrasi-ke-predictor)
  - [Fase 14: Team Discipline Model](#fase-14-team-discipline-model)
  - [Fase 16: Report/Dashboard (Opsional)](#fase-16-reportdashboard-opsional)
  - [Fase 17: Audit Menyeluruh](#fase-17-audit-menyeluruh)
- [Keterbatasan & Disclaimer](#keterbatasan--disclaimer)
- [Kredit & Sumber Data](#kredit--sumber-data)

## Tech Stack
- **Data wrangling:** pandas, numpy
- **Model statistik:** [`penaltyblog`](https://github.com/martineastwood/penaltyblog) (Poisson, Dixon-Coles, Elo/Pi-rating)
- **Model machine learning:** scikit-learn, XGBoost/LightGBM
- **Data pemain:** [`soccerdata`](https://github.com/probberechts/soccerdata) (FBref, ClubElo)
- **Eksplorasi:** Jupyter Notebook
- **Visualisasi:** matplotlib, seaborn

## Struktur Project
```
football-predictive-analysis/
├── data/
│   ├── raw/                    # data mentah (tidak di-commit — lihat .gitignore)
│   └── processed/              # data bersih + fitur (tidak di-commit)
├── notebooks/                  # EDA, eksperimen model
├── src/
│   ├── team_mapping.py         # standardisasi nama tim (single source of truth)
│   ├── data_collection.py      # Fase 1
│   ├── feature_engineering.py  # Fase 3
│   ├── statistical_models.py   # Fase 4 — Track A: Poisson, Dixon-Coles, Elo, BTTS
│   ├── train_model.py          # Fase 5 — Track B: ML classifiers
│   ├── evaluate.py             # Fase 7
│   ├── value_betting.py        # Fase 8
│   ├── predict_match.py        # Fase 9, 12, 15
│   ├── player_stats.py         # Fase 10 — statistik musiman pemain
│   ├── corner_model.py         # Fase 11 — model corner
│   ├── player_match_model.py   # Fase 13 — prediksi SOT & kandidat MOTM per match
│   └── discipline_model.py     # Fase 14 — model kartu kuning/merah
├── models/                     # model tersimpan (tidak di-commit)
├── requirements.txt
└── scope-predictive-analysis-bola.md
```

## Instalasi
```powershell
git clone https://github.com/DanishXD/predictive-analysis-football.git
cd predictive-analysis-football
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Sumber Data
| Sumber | Kegunaan |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | Hasil pertandingan, odds 1X2, corner (HC/AC), kartu (HY/AY/HR/AR) — data historis EPL |
| [`penaltyblog`](https://github.com/martineastwood/penaltyblog) | Scraper + model statistik (Poisson, Dixon-Coles, Elo) |
| [FBref](https://fbref.com/) via [`soccerdata`](https://github.com/probberechts/soccerdata) | Statistik pemain (goals, assists, shots, shots on target, xG) |
| [ClubElo](http://clubelo.com/) via `soccerdata.ClubElo` | Rating awal (cold-start) untuk tim promosi |

> **Catatan penggunaan data:** Beberapa sumber di atas (terutama FBref) memiliki Terms of Service yang membatasi automated scraping. Project ini menggunakan data tersebut untuk keperluan personal/edukasi dengan caching yang wajar (bukan bulk scraping berulang), bukan untuk redistribusi komersial. Data mentah **tidak di-commit** ke repo ini — lihat `.gitignore`.

## Pipeline

### Fase 1: Data Collection & Cleaning

Pipeline mengambil data EPL musim 2016/17 sampai 2025/26 melalui scraper
`penaltyblog`. Jika scraper gagal untuk suatu musim, pipeline otomatis mengunduh
CSV dari football-data.co.uk.

```powershell
python -m pip install -r requirements.txt
python src/data_collection.py
```

Output utama disimpan di `data/processed/matches_clean.csv`. Data per musim
sebelum proses cleaning disimpan di `data/raw/`. Nama tim distandardisasi lewat
`src/team_mapping.py`, yang dipakai ulang di seluruh fase berikutnya (Fase 10, 11, 13, 14).

### Fase 2: Exploratory Data Analysis

Notebook EDA yang mengecek distribusi hasil pertandingan (Home Win/Draw/Away Win),
home advantage, dan tren antar musim.

```powershell
jupyter notebook notebooks/eda.ipynb
```

### Fase 3: Feature Engineering

Membentuk fitur rolling — form 5 pertandingan terakhir, head-to-head record,
home/away performance split, dan posisi klasemen (running table) — yang **selalu
dihitung point-in-time**, hanya menggunakan data sebelum tanggal pertandingan yang
diprediksi, untuk menghindari data leakage. Tim promosi yang belum punya histori
diberi fallback nilai netral.

```powershell
python src/feature_engineering.py
```

Output: `data/processed/features.csv`

### Fase 4: Track A — Model Statistik

Fit Poisson Goal Model, Dixon-Coles, dan Elo/Pi-rating menggunakan `penaltyblog`.
Parameter attack/defense di-fit **hanya pada training period** — evaluasi dilakukan
out-of-sample di test period, sama disiplinnya dengan Track B, supaya perbandingan
model di Fase 7 adil. Tim promosi memakai rating awal dari ClubElo (fallback ke
rata-rata Elo 3 tim terbawah musim sebelumnya jika ClubElo gagal).

```powershell
python src/statistical_models.py
```

### Fase 5: Track B — Model Machine Learning

Logistic Regression → Random Forest → XGBoost, menggunakan `TimeSeriesSplit`
(bukan random k-fold) dan `class_weight='balanced'` untuk mengurangi bias model
yang cenderung mengabaikan kelas Draw (minoritas).

```powershell
python src/train_model.py
```

### Fase 6: Stacking (Opsional)

Menambahkan output Track A (Elo rating gap, expected goals dari Poisson) sebagai
fitur tambahan ke model Track B terbaik.

```powershell
python src/train_model.py --stacking
```

### Fase 7: Evaluasi Menyeluruh

Menghasilkan `evaluation_summary.csv` yang berisi accuracy, confusion matrix per
kelas, log loss, Brier score, Ranked Probability Score (RPS), calibration check,
dan perbandingan terhadap odds bandar yang sudah di-decode (overround dihilangkan).
File ini yang dibaca otomatis oleh Fase 9 untuk memilih model terbaik.

```powershell
python src/evaluate.py
```

### Fase 8: Value Betting Testing

Fase ini membandingkan probabilitas model dengan odds 1X2 manual dan menghitung:

```text
EV = (probabilitas model x odds decimal) - 1
```

Jalankan contoh dengan odds fiktif:

```powershell
python src/value_betting.py
```

Atau gunakan CSV sendiri dengan schema yang sama:

```powershell
python src/value_betting.py --input data/upcoming_fixtures.csv --output data/processed/value_betting_output.csv
```

Kolom wajib: `fixture_id`, `season`, `datetime`, `team_home`, `team_away`,
`odds_home`, `odds_draw`, dan `odds_away`. Odds harus memakai format decimal.

**Disclaimer:** Fitur ini hanya exercise data science untuk mempelajari perbedaan
probabilitas model dan pasar, **bukan alat rekomendasi taruhan**. Dataset hanya
sekitar 3.800 pertandingan dan pada evaluasi Fase 7 odds pasar mengalahkan semua
model pada RPS dan log loss. EV tinggi bisa berasal dari noise, data yang sudah
stale, cold-start, atau model yang belum terkalibrasi dengan baik — bukan bukti
adanya edge maupun jaminan profit. Jangan mempertaruhkan uang berdasarkan output
project ini.

### Fase 9: Interactive Match Predictor

Jalankan CLI interaktif:

```powershell
python src/predict_match.py
```

Script menampilkan semua tim yang pernah muncul dalam dataset. Tim bisa dipilih
dengan mengetik nomor atau nama lengkapnya, lalu user menentukan tim kandang.

Pemilihan model tidak di-hardcode: script membaca `evaluation_summary.csv`, memilih
RPS terendah (dengan log loss sebagai tie-breaker) antara Poisson/Dixon-Coles untuk
top-3 skor, dan antara Logistic Regression/Random Forest/XGBoost untuk peluang
Home Win/Draw/Away Win. Nama model beserta metrik Fase 7 selalu ditampilkan.

Karena fixture bersifat hipotetis dan tidak meminta tanggal, script memakai waktu
saat dijalankan sebagai tanggal prediksi, tetapi tidak pernah memakai tanggal
sebelum cutoff data historis. Tim dengan histori sedikit, tim yang sudah lama tidak
tampil di EPL, cold-start model, dan kelemahan recall Draw akan diberi peringatan
di output. Sejak Fase 12/15, output juga menyertakan corner, BTTS, prediksi
match-level pemain kunci, dan kartu kuning — lihat bagian terkait di bawah.

### Fase 10 & 13: Player Statistics & Match-Level Predictions

**Fase 10 — statistik musiman (konteks skuad):**

```powershell
python -m pip install soccerdata
python src/player_stats.py --season 2526
```

Untuk melihat top-3 pemain dari tim tertentu:

```powershell
python src/player_stats.py --season 2526 --team "Arsenal"
```

**Format season:** gunakan format FBref seperti `2526` untuk musim 2025/26.

**Fase 13 — prediksi match-level (SOT & kandidat MOTM):**

```powershell
python src/player_match_model.py
```

Menghitung rentang ekspektasi Shots on Target per pemain kunci untuk *match
spesifik* (bukan angka tunggal — SOT per pemain per pertandingan variansnya
tinggi), dan meranking kandidat "Man of the Match" berdasarkan proyeksi
kontribusi gol (xG + xA). Label output selalu **"kandidat MOTM paling mungkin"**,
bukan "prediksi MOTM" — MOTM adalah keputusan subjektif yang tidak bisa
benar-benar diprediksi model statistik.

**Disclaimer (berlaku untuk Fase 10 & 13):** Data pemain ini **hanya konteks
tambahan**, TIDAK dipakai sebagai fitur di model prediksi skor/W-D-L. Data ini
juga tidak mencakup informasi cedera, suspensi, atau rotasi tim menit-menit
terakhir. Script memakai caching bawaan soccerdata untuk menghindari request
berulang ke FBref.

### Fase 11: Corner Model & BTTS

**Corner Model**

```powershell
python src/corner_model.py
```

Fits a Poisson model for corner counts (HC/AC) with the same architecture as the
goal model: each team gets corner-attack and corner-defense parameters + home
advantage. Same train/test split as Phase 4 (2021-2025 train, 2025-26 test) with
time-decay weights. Output:

- `corner_team_strengths.csv` — corner-attack/defense per team
- `corner_test_predictions.csv` — expected corners + over/under probabilities per match

**Catatan:** Corner model menggunakan arsitektur Poisson yang sama dengan model
gol, tetapi akurasi model corner biasanya **lebih rendah** karena corner lebih
noisy/random (dipengaruhi taktik, defensive block, gaya main, dll).

**BTTS (Both Teams to Score)**

BTTS **tidak butuh model baru**. Dihitung langsung dari grid probabilitas skor
Poisson/Dixon-Coles (Fase 4) dengan menjumlahkan semua probabilitas skor di mana
`home_goals >= 1 AND away_goals >= 1`.

Fungsi `btts_probability(grid)` di `src/statistical_models.py` mengambil
`FootballProbabilityGrid` dan mengembalikan probabilitas BTTS.

```python
from statistical_models import btts_probability, btts_probabilities_for_test
```

### Fase 12 & 15: Integrasi ke Predictor

Menggabungkan seluruh output — skor (top-3), peluang W/D/L, key player
match-level (SOT & kandidat MOTM), prediksi corner, BTTS, dan kartu kuning — ke
dalam satu output `predict_match.py` (Fase 9). Setiap bagian output selalu
mencantumkan nama model/sumber yang dipakai, dan bagian key player di Fase 15
menggantikan tampilan season-stats lama dari Fase 10 dengan prediksi match-level
dari Fase 13.

### Fase 14: Team Discipline Model

Model Poisson untuk kartu kuning (opsional kartu merah) per tim, menggunakan
kolom HY/AY (dan HR/AR) yang sudah tersedia di data football-data.co.uk —
arsitektur dan split training/test sama dengan corner model di Fase 11. Kartu
merah datanya jauh lebih jarang, sehingga confidence model untuk itu lebih rendah.

```powershell
python src/discipline_model.py
```

### Fase 16: Report/Dashboard (Opsional)

Notebook yang merangkum seluruh hasil Fase 1–15 — overview dataset, insight EDA,
hasil Track A & B, dan tabel perbandingan semua metrik evaluasi — menjadi satu
laporan.

### Fase 17: Audit Menyeluruh

Checklist QA yang memverifikasi tidak ada data leakage di fitur, split
training/test konsisten antara Track A & B, pemilihan model di Fase 9/12/15
otomatis (bukan hardcode), serta seluruh disclaimer dan label model tampil
dengan benar. Prompt audit lengkap tersedia di
`prompt-agent-predictive-analysis-bola.md` (Prompt 17).

## Keterbatasan & Disclaimer

- Dataset hanya ~3.800 pertandingan (10 musim EPL) — tergolong kecil untuk model
  ML kompleks, risiko overfitting.
- Pada evaluasi Fase 7, odds bandar mengalahkan semua model pada metrik RPS dan
  log loss.
- Model classifier cenderung under-predict kelas Draw (kelas minoritas paling
  sulit ditangkap polanya).
- Data pemain (Fase 10/13) tidak mencakup cedera, suspensi, atau rotasi tim
  menit-menit terakhir.
- Kandidat "Man of the Match" adalah proxy statistik (expected goal
  contribution), bukan prediksi sesungguhnya — MOTM adalah keputusan subjektif
  yang tidak bisa diprediksi model statistik apa pun.
- Data historis dan model perlu diperbarui secara berkala (terutama menjelang
  musim baru, karena jendela transfer bisa mengubah kekuatan tim) sebelum
  dipakai menganalisis fixture baru.
- **Project ini bukan alat rekomendasi taruhan.** Jangan mempertaruhkan uang
  berdasarkan output project ini.

## Kredit & Sumber Data

- [football-data.co.uk](https://www.football-data.co.uk/)
- [penaltyblog](https://github.com/martineastwood/penaltyblog) oleh Martin Eastwood
- [soccerdata](https://github.com/probberechts/soccerdata) oleh Pieter Robberechts
- [FBref](https://fbref.com/) / Sports Reference LLC
- [ClubElo](http://clubelo.com/)