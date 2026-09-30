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
  - [Fase 18: Koreksi Output Probability (Eksperimen)](#fase-18-koreksi-output-probability-eksperimen)
  - [Fase 19: Hyperparameter Tuning (Eksperimen)](#fase-19-hyperparameter-tuning-eksperimen)
  - [Eksperimen Track A: Walk-Forward CV](#eksperimen-track-a-walk-forward-cv)
  - [Eksperimen Exposure-Based Poisson (bukan xG)](#eksperimen-exposure-based-poisson-bukan-xg)
  - [Eksperimen Blend Sistematis (semua model)](#eksperimen-blend-sistematis-semua-model)
  - [Eksperimen Seasonal HFA](#eksperimen-seasonal-hfa)
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
│   ├── model_calibration.py    # Fase 18 (eksperimen — hasil negatif)
│   ├── model_tuning.py         # Fase 19 (eksperimen)
│   ├── track_a_cv.py           # eksperimen walk-forward CV Track A
│   ├── exposure_poisson.py     # eksperimen exposure-based Poisson (bukan xG)
│   ├── blend.py                # eksperimen blend semua model (hasil negatif)
│   ├── seasonal_hfa.py         # eksperimen HFA musim no-fans (hasil negatif)
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
(bukan random k-fold). Eksperimen `class_weight='balanced'` sempat dicoba untuk
menangani kelas Draw (minoritas), tetapi di-revert karena RPS lebih diprioritaskan
— recall Draw rendah dicatat sebagai keterbatasan yang acceptable.

```powershell
python src/train_model.py
```

### Fase 6: Stacking (Opsional)

Menambahkan expected goals dari Poisson **dan** Dixon-Coles sebagai fitur tambahan
ke Random Forest (Elo rating gap sudah jadi fitur dasar sejak Fase 5). Percobaan
controlled dengan OOF xG anti-leakage + bootstrap CI.

```powershell
python src/stacking.py
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

Pemilihan model **tidak di-hardcode dan tidak memakai test metrics**. Script
membaca `cv_model_selection.csv` (metrik berbasis data training), lalu memilih
`cv_log_loss_mean` terendah — dengan `cv_accuracy_mean` tertinggi sebagai
tie-breaker, lalu nama model secara deterministik — antara Poisson/Dixon-Coles
untuk top-3 skor, dan antara Logistic Regression/Random Forest/XGBoost untuk
peluang Home Win/Draw/Away Win.

`evaluation_summary.csv` (metrik test season) tetap ada sebagai artefak
pelaporan akhir, tetapi **tidak boleh dipakai memilih model** — memakainya
untuk selection adalah kontaminasi test set. Nama model beserta metrik yang
dipakai selalu ditampilkan di output.

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
advantage. Same train/test split as Phase 4 (2016-2017 through 2024-2025 train,
2025-2026 test) with time-decay weights. Output:

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

> **Catatan:** `discipline_model.py` juga menyimpan `red_card_model.pkl`, tetapi
> model kartu merah itu **belum terintegrasi** ke output `predict_match.py`
> (data kartu merah terlalu jarang sehingga confidencenya rendah) — disimpan
> sebagai artefak eksperimen saja.

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

### Fase 18: Koreksi Output Probability (Eksperimen)

```powershell
python src/model_calibration.py
```

Menguji dua teknik koreksi output probability pada model Random Forest:
post-hoc calibration (sigmoid/Platt dan isotonic) serta penyesuaian
draw-prior (P(draw) x lambda). Metrik dihitung **out-of-fold di data training**,
bukan di test season, supaya tidak ada kontaminasi test set.

lambda draw-prior dicari dengan minimasi RPS di CV training. Hasilnya
**lambda = 1.0, artinya tidak ada penyesuaian yang membantu** — kurva RPS
monoton memburuk seiring naiknya lambda. Kedua metode calibration justru
memperbaiki ECE out-of-fold (0.021557 -> 0.016833) tapi **memperbaiki keyakinan
sambil merusak urutan probabilitas**, dan RPS hanya peduli urutan.

> **Hasil negatif — jangan diulang.** Detail lengkap di
> [`HANDOFF.md`](./HANDOFF.md) bagian 5.

### Fase 19: Hyperparameter Tuning (Eksperimen)

```powershell
python src/model_tuning.py
```

`GridSearchCV` dengan `DateTimeSeriesSplit` (adapter date-based, forward-only)
dan objective RPS. Random Forest dan XGBoost masing-masing diberi grid kecil
supaya biaya komputasi tetap wajar.

Hasil XGBoost (`learning_rate=0.02`, `max_depth=2`, `n_estimators=200`) kini
dipakai di `train_model.build_models()` dan tersimpan di
`config.XGBOOST_TUNED_PARAMS`.

> **Penting — pilihan ini belum meyakinkan.** XGBoost menang tipis di CV log loss
> (0.980369 vs 0.982640 untuk Random Forest) tetapi selisihnya **tidak
> signifikan**: t-test 5 fold menghasilkan p=0.28 dan hanya 3 dari 5 fold
> menguntungkan XGBoost. Di test season 2025-26 Random Forest justru lebih baik
> pada RPS (0.211421 vs 0.213533) dan log loss (1.037602 vs 1.045309). XGBoost
> dipilih karena **aturan main project ini** (selection berbasis CV, bukan test
> set), bukan karena bukti kuat. Kalau bukti di musim berikutnya tetap slim,
> kembalikan `XGBOOST_TUNED_PARAMS` di `config.py` ke nilai lama
> (`n_estimators=300`, `learning_rate=0.03`, `max_depth=3`).

Tuning Random Forest sendiri nyaris tidak bergerak: delta RPS hanya +0.000271
dengan 95% CI [-0.000327, +0.000874] -> tidak signifikan.

> **Hasil negatif — jangan diulang.** Detail lengkap di
> [`HANDOFF.md`](./HANDOFF.md) bagian 5.

### Eksperimen Track A: Walk-Forward CV

```powershell
python src/track_a_cv.py
```

Menghitung metrik **out-of-sample** untuk Track A (Poisson, Dixon-Coles, Elo)
lewat walk-forward musiman: 5 fold expanding-window dengan season validasi
2020-2021 s.d. 2024-2025. `TEST_SEASON` tidak pernah dipakai sebagai fold.

Alasan modul ini ada: angka Track A di `cv_model_selection.csv` dulu
dihitung **in-sample** (model dievaluasi di data yang sama dengan data
latihnya), sedangkan angka Track B di tabel yang sama berasal dari
TimeSeriesSplit out-of-fold. Modul ini menghasilkan angka yang benar-benar
sebanding.

```text
model          LogLoss       RPS      Acc
poisson       1.103723  0.243066  0.4489
dixon_coles   1.105310  0.243113  0.4479
elo           1.106281  0.241574  0.4616
```

Dixon-Coles tidak memberi peningkatan bermakna di luar sampel (paired per
fold: `+0.001586` log loss, hanya 1 dari 5 fold membaik) — konsisten dengan
`rho ~ -0.004` yang sudah dicatat sebagai limitation.

> **Status: sudah disambungkan (30 September 2026).** `evaluate.py` membaca
> `data/processed/track_a_cv_results.csv` dan memakai angka di atas untuk
> selection Track A, dengan guard keras yang menolak file bila ada fold
> dengan `validation_season == TEST_SEASON`. Konsekuensinya **goal-model
> produksi flip dari Dixon-Coles ke Poisson** — see `HANDOFF.md` §2.1.
> Track B tidak berubah.
>
> Catatan: Elo di modul ini tanpa cold-start ClubElo (semua tim mulai dari
> 1500) supaya deterministik dan tidak memanggil API luar yang bisa rate-limit.
> Jadi angka Elo di sini tidak langsung sebanding dengan Elo produksi.

### Eksperimen Exposure-Based Poisson (bukan xG)

```powershell
python src/exposure_poisson.py
```

Menguji apakah volume tembakan (`shots`) dan efisiensi per tembakan membawa
informasi yang tidak ada di data gol. **Ini bukan model berbasis xG** — tidak
ada shot location atau shot quality, hanya agregat per tim per match.

Struktur dua tahap. Model offset tunggal hanya bisa dipakai untuk match yang
sudah selesai, jadi exposure itu sendiri harus diprediksi lebih dulu:

```text
sigma  = E[shots]         (stage 1, penaltyblog Poisson di-fit pada shots)
q      = P(goal | shot)   (stage 2, Poisson dengan offset log(shots))
lambda = sigma * q
```

Varian SOT menambah tahap: `q = P(SOT | shot) * P(goal | SOT)`.

```text
model                   log loss      RPS  accuracy
sot                     0.997688  0.209706  0.517368
poisson_goals_based     1.002893  0.211598  0.520000
shot_volume             1.004002  0.212677  0.508421
```

Varian **SOT menang di kelima fold** pada log loss dan RPS, dengan 95% CI
[-0.008340, -0.002153] (tidak melintasi nol). Varian `shot_volume` justru
tidak signifikan — jadi efeknya bukan dari "tembakan" secara umum, tapi dari
pemisahan shots -> SOT -> goal.

Temuan sampingan: home advantage praktis seluruhnya ada di volume, bukan di
efisiensi. Rasio home/away di training: goals 1.212, shots 1.204, tapi
conversion rate hanya 1.006. Karena itu parameter home-advantage di stage-2
berakhir di ~0 — hasil yang benar, bukan kegagalan optimasi.

> **Status: belum jadi produksi.** Angka di atas masih CV di training season
> dengan bias selection pada `xi`, dan `evaluate.py`/`predict_match.py`
> sengaja tidak disentuh. Detail di [`HANDOFF.md`](./HANDOFF.md) bagian 5.8.

### Eksperimen Blend Sistematis (semua model)

```powershell
python src/blend.py
```

Meta-learner (Logistic Regression) di atas probabilitas 1X2 dari 7 base
model: `poisson`, `dixon_coles`, `elo`, `logistic_regression`,
`random_forest`, `xgboost`, dan `exposure_sot`.

Anti-leakage berlapis: base model di-fit ulang per fold, meta-learner
dilatih **hanya** pada prediksi out-of-fold, dan meta-learner sendiri
dievaluasi walk-forward di level kedua. Test season hanya dibaca sekali
untuk pelaporan.

> **Hasil: blend TIDAK mengalahkan model tunggal terbaik.** Ranking CV juga
> tidak transfer ke test season — rata-rata sederhana menang di CV
> (0.979351 vs 0.984392) tapi kalah di test (1.043516 vs 1.037602).
> Detail lengkap di [`HANDOFF.md`](./HANDOFF.md) bagian 5.9.

### Eksperimen Seasonal HFA

```powershell
python src/seasonal_hfa.py
```

Menguji apakah musim 2020/2021 (tanpa penonton) butuh parameter
`home_advantage` sendiri. Hasilnya: HFA antar musim memang **berbeda
signifikan** (selisih -0.1407, bootstrap 95% CI [-0.2849, -0.0723]), tapi
**tidak layak diimplementasikan** — tidak ada jalur validasi out-of-sample
yang bisa mengujinya, dan varian yang bisa diuji justru tidak significant.

> **Hasil negatif.** HFA global tetap dipakai di produksi. Detail di
> [`HANDOFF.md`](./HANDOFF.md) bagian 5.7.

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