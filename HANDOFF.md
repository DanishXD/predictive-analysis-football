# HANDOFF — football-predictive-analysis

Dokumen ini untuk **agent yang mulai bekerja di repo ini dari IDE (Orca)**, agar tidak
perlu membaca ulang 12 commit + seluruh percakapan sebelumnya.

**Dibuat:** 29 September 2026
**Project root:** `C:\Users\danis\Documents\football-predictive-analysis`

> PENTING: kalau CLI Anda berada di `C:\Users\danis`, itu **bukan** project root.
> Semua perintah harus dijalankan dari `C:\Users\danis\Documents\football-predictive-analysis`.

---

## 0. TL;DR

| | |
|---|---|
| Model produksi | **xgboost** (winner CV log loss) |
| Tes | **125 lulus**, 0 gagal (92 sebelum sesi 29 Sep) |
| Git | `main` == `origin/main`, working tree bersih |
| Status | Pipeline Fase 1-19 + 4 eksperimen (kalibrasi, tuning, Track A CV, seasonal HFA) |
| Hasil negatif | **8 tercatat** (§5.1-§5.7) — baca sebelum eksperimen baru |
| Topik terbuka | Explorasi xG-based model — **Understat DITOLAK**, butuh keputusan user |

---

## 1. Project apa ini

Project **pembelajaran data science**: prediksi hasil pertandingan EPL (H/D/A + corner +
kartu + pemain). Bukan alat rekomendasi betting, dan selalu ada disclaimer di output.

Sumber data utama: football-data.co.uk (goals, shot, corner, kartu, odds, wasit),
ClubElo (rating tim cold-start), FBref (pemain, opsional).

10 musim EPL 2016/17 - 2025/26, 3.800 pertandingan, 34 tim.
Test season: **2025-2026** (380 match, out-of-sample).

---

## 2. Status produksi saat ini

### 2.1 Pemilihan model (CV log loss, semua track out-of-sample)

Track B pakai TimeSeriesSplit 5-fold; Track A pakai walk-forward musiman
5 fold (lihat §13). Tidak ada lagi angka in-sample di tabel ini.

| model | track | cv_log_loss_mean | cv_accuracy_mean | basis |
|---|---|---|---|---|
| **xgboost** | Track B | **0.980369** | 0.546874 | TimeSeriesSplit 5-fold CV |
| random_forest | Track B | 0.982640 | 0.545098 | TimeSeriesSplit 5-fold CV |
| logistic_regression | Track B | 1.016022 | 0.526199 | TimeSeriesSplit 5-fold CV |
| poisson | Track A | 1.103723 | 0.448947 | walk-forward musiman, 5 fold (out-of-sample) |
| dixon_coles | Track A | 1.105310 | 0.447895 | walk-forward musiman, 5 fold (out-of-sample) |
| elo | Track A | 1.106281 | 0.461579 | walk-forward musiman, 5 fold (out-of-sample) |

> **Goal-model FLIP (30 September 2026).** Sebelumnya angka Track A di tabel
> ini dihitung in-sample dan winner-nya **Dixon-Coles** (0.983277) beating
> Poisson (0.983279) dengan selisih **0.000002** — itu noise, bukan bukti.
> Setelah disambungkan ke walk-forward CV, urutannya **Poisson**
> (1.103723) > Dixon-Coles (1.105310) > Elo (1.106281), dengan selisih
> 0.001586 dan berlaku konsisten di 5 fold. `predict_match.py` sekarang
> memakai **Poisson** sebagai goal model produksi.
>
> Perhatikan juga: angka Track A in-sample (0.983) dan Track B out-of-fold
> (0.980) tidak pernah sebanding. Sekarang keduanya out-of-sample, jadi
> jarak 0.12 antara track adalah perbedaan kompleksitas model yang nyata,
> bukan perbedaan metrik.

### 2.2 Performa di test season 2025-26 (380 match)

| model | accuracy | log_loss | brier | RPS | ECE | draw_recall |
|---|---|---|---|---|---|---|
| bookmaker_avg_odds | 0.494737 | 1.015252 | 0.610025 | **0.205280** | 0.040011 | 0.0 |
| random_forest | 0.473684 | 1.037602 | 0.623542 | 0.211421 | 0.040824 | 0.0 |
| xgboost | 0.486842 | 1.045309 | 0.629293 | 0.213533 | 0.045375 | 0.0 |
| logistic_regression | 0.478947 | 1.044979 | 0.630406 | 0.214109 | 0.051588 | 0.0 |
| elo | 0.484211 | 1.075032 | 0.641466 | 0.216117 | 0.070728 | 0.0 |
| dixon_coles | 0.465789 | 1.072190 | 0.645820 | 0.222052 | 0.065594 | 0.0 |
| poisson | 0.465789 | 1.072295 | 0.645906 | 0.222058 | 0.066965 | 0.0 |

Tidak ada model yang mengalahkan bookmaker. Ini jujur dan sesuai scope.

### 2.3 Konfigurasi kunci (`src/config.py`)

```python
TEST_SEASON = "2025-2026"     N_SPLITS = 5              RANDOM_STATE = 42
TIME_DECAY_XI = 0.0018        ELO_K = 20.0              ELO_HOME_ADVANTAGE = 100.0
ELO_DEFAULT_RATING = 1500.0   BOOTSTRAP_SAMPLES = 5000
CLUBELO_CACHE_MAX_AGE_DAYS = 7
EMPTY_STADIUM_SEASONS = ("2020-2021",)
REFEREE_ENABLED = True        REFEREE_MIN_MATCHES = 30  REFEREE_PRIOR_STRENGTH = 30.0
ELO_CLUBELO_INTERCEPT = -469.1728                      ELO_CLUBELO_SLOPE = 1.0893
XGBOOST_TUNED_PARAMS = {"learning_rate": 0.02, "max_depth": 2, "n_estimators": 200}
```

---

## 3. Peta project

### 3.1 Modul `src/` (15 file)

| File | Fase | Fungsi |
|---|---|---|
| `config.py` | - | **Single source of truth semua konstanta**. Jangan definisikan ulang di module lain |
| `team_mapping.py` | - | Standardisasi nama tim (3 arah: football-data, FBref, ClubElo) |
| `data_collection.py` | 1 | Fetch + clean -> `matches_clean.csv` |
| `feature_engineering.py` | 3 | Fitur point-in-time anti-leakage -> `features.csv` |
| `statistical_models.py` | 4 | Poisson, Dixon-Coles, Elo/Pi-rating + ClubElo cold-start |
| `train_model.py` | 5 | LogReg / RandomForest / XGBoost |
| `stacking.py` | 6 | Eksperimen stacking xG (opsional) |
| `evaluate.py` | 7 | Evaluasi semua model + bootstrap + `cv_model_selection.csv` |
| `value_betting.py` | 8 | Harness EV (edukasi) |
| `predict_match.py` | 9 | CLI prediksi interaktif |
| `player_stats.py` | 10 | Data pemain FBref (opsional, butuh soccerdata) |
| `corner_model.py` | 11 | Model corner + BTTS |
| `player_match_model.py` | 13 | Prediksi SOT & kandidat MOTM |
| `discipline_model.py` | 14 | Kartu kuning/merah + fitur wasit |
| `model_calibration.py` | 18 | Eksperimen koreksi output probability |
| `model_tuning.py` | 19 | Eksperimen hyperparameter tuning |

### 3.2 Test (`tests/`, 16 file)

`conftest.py`, `test_config.py`, `test_team_mapping.py`, `test_feature_engineering.py`,
`test_btts_probability.py`, `test_over_under_probability.py`, `test_select_best_model.py`,
`test_date_based_time_splits.py`, `test_data_collection_cleaning.py`, `test_clubelo_rating.py`,
`test_clubelo_cache.py`, `test_cv_model_selection.py`, `test_discipline_referee.py`,
`test_feature_empty_stadium.py`, `test_model_calibration.py`, `test_model_tuning.py`,
`test_tuned_xgboost_params.py`

---

## 4. 12 commit sesi ini

| Commit | Isi |
|---|---|
| `c87b8e6` | Checkpoint: `config.py`, test suite, docs masuk git untuk pertama kali |
| `1a0195b` | Bug fix: double-print `sot_disclaimer`, validasi stacking sebelum save, error handling corner/kartu |
| `bf8fe61` | Sentralisasi semua konstanta & path ke `config.py`, guard AST baru |
| `44aca7a` | Sinkronisasi docs (hapus klaim `class_weight`, fix `--stacking` phantom) |
| `d9572ae` | Cache FBref via lru_cache, guard `int(NaN)`, batch fitur, bersihkan notebook |
| `0b96c55` | 6 file unit test baru (+ bonus fix bug push threshold bulat) |
| `2c4cc61` | ClubElo rescaling + disk cache + log cold-start |
| `37eb633` | Fitur `is_empty_stadium` + fitur wasit |
| `1740114` | `evaluate.py` generate `cv_model_selection.csv` (tadinya file manual) |
| `7c7cef1` | Phase 18: kalibrasi + draw-prior (hasil negatif) |
| `a7288e8` | Phase 19: hyperparameter tuning |
| `34876d0` | XGBoost tuned jadi kandidat produksi |

---

## 5. HASIL NEGATIF — jangan mengulang percobaan ini

Ini bagian terpenting dokumen. Semua sudah diuji dengan cara yang benar (anti-leakage,
data terpisah, angka penentu dari training/CV). Semuanya gagal.

> **Perbarui 29 Sep 2026:** §5.7 (HFA terpisah per musim no-fans) ditambahkan.
> Catatan khusus §5.3 berlaku untuk semua eksperimen baru: kalau sensitivitas
> dihitung di TEST set, hasilnya akan terlihat bagus tapi salah.

### 5.1 `class_weight='balanced'` untuk kelas Draw
Memperbaiki sedikit Draw recall tapi **merusak RPS**. Sudah di-revert. Jangan dicoba lagi.

### 5.2 Post-hoc calibration (Phase 18) - `src/model_calibration.py`

Test season 2025-26, Random Forest:

| variant | RPS | log loss | draw_recall | ECE |
|---|---|---|---|---|
| raw | **0.211421** | **1.037602** | 0.000 | **0.040824** |
| sigmoid (Platt) | 0.213966 (+0.0025) | 1.044824 (+0.0072) | 0.000 | 0.041028 |
| isotonic | 0.215273 (+0.0039) | 1.050215 (+0.0126) | 0.000 | 0.053895 |

**Kedua metode merusak RPS.** Menarik: sigmoid justru *memperbaiki* ECE out-of-fold
(0.021557 -> 0.016833), jadi teknik ini bekerja secara teknis, tapi memperbaiki keyakinan
sambil merusak *urutan* probabilitas. RPS hanya peduli urutan.

### 5.3 Draw-prior adjustment (P(draw) x lambda)

`lambda` dipilih dengan minimasi RPS di CV training (bukan test).
**Hasil: lambda = 1.00, artinya tidak ada penyesuaian yang membantu.**

Kurva RPS di CV training monoton memburuk seiring naiknya lambda:

```
lambda   1.0      1.1      1.2      1.3      1.4      1.5
RPS    0.203482 0.203675 0.204021 0.204497 0.205087 0.205774
```

> **PELAJARAN PENTING:** eksplorasi awal menghitung sensitivitas lambda di **TEST SET**
> dan menyimpulkan "lambda 1.3 memperbaiki RPS" (0.210647 vs 0.211421). Itu **salah** -
> itu test-set overfitting. Begitu dihitung lewat CV training yang benar, sinyalnya hilang.
> Kalau kamu mengulang eksperimen ini, pastikan hitung angkanya di training, bukan test.

### 5.4 Fitur wasit untuk discipline model

Komis `Referee` **ada** di football-data.co.uk (380/380 per musim) dan sudah dipakai.
Implementasi: pengali multiplikatif expected cards, rate di-fit dari training saja,
wasit < 30 match di-group jadi `wasit_lain`, sisanya di-shrink ke rata-rata liga.

**Hasil: MAE kartu kuning memburuk.** 1.5986 -> 1.6205, 95% CI [-0.0047, +0.0481],
P(delta<=0) = 0.0504. Bukti cenderung negatif tapi belum meyakinkan (sample kecil).
Di-toggle lewat `config.REFEREE_ENABLED`.

### 5.5 Fitur `is_empty_stadium` (musim 2020/21 no-fans)

Dibatasi: HFA 2020/21 = -0.071 vs musim lain +0.425 poin/match, bootstrap
95% CI [-0.385, -0.099] -> nyata berbeda. Fitur biner sudah ditambahkan.

**Hasil: tidak signifikan.** Out-of-fold per-match log loss membaik di 3 model tapi CI semua
melintasi 0 (LogReg +0.0013, RF +0.0007, XGB +0.0014). Feature importance 0.0042,
peringkat 36 dari 38.

> Keterbatasan struktural: `is_empty_stadium` bernilai **0 untuk 100% test set** (2025-26
> bukan musim no-fans), jadi mustahil diuji di test season mana pun selama tidak ada
> musim no-fans lagi. Dampaknya hanya terlihat via CV di training.

### 5.6 Tuning hyperparameter Random Forest

GridSearchCV 12 kombinasi, DateTimeSeriesSplit, objective RPS.
Parameter terbaik: `max_depth=None, min_samples_leaf=8, n_estimators=250`.

Delta RPS hanya +0.000271, CI [-0.000327, +0.000874] -> **tidak signifikan**.
Secara praktis RF tidak bergerak dari setting default.

### 5.7 HFA terpisah untuk musim no-fans 2020/21 - `src/seasonal_hfa.py`

Rekomendasi sendiri di §7.5, sekarang sudah diuji. Profile likelihood: HFA
di-search ulang dengan attack/defense dibekukan dari fit penuh.

```
HFA normal  +0.1954     HFA no-fans  +0.0547     delta -0.1407
Bootstrap 95% CI selisih HFA: [-0.2849, -0.0723]  -> SIGNIFIKAN
Varian A (HFA tanpa musim no-fans) di 4 fold: delta log loss +0.000277,
  2 dari 4 fold membaik, delta RPS -0.000031  -> TIDAK signifikan
```

**HFA antar musim memang berbeda nyata, tapi tidak layak diimplementasikan:**

1. Varian B (HFA khusus no-fans) **tidak punya jalur validasi sama sekali.**
   Season 2020/21 jadi validation hanya di fold 1, tapi season itu sendiri
   baru masuk training di fold 2. Jadi satu-satunya fold yang bisa menguji
   varian no-fans justru tidak punya musim no-fans di training. Selama tidak
   ada musim no-fans kedua, ini mustahil diuji out-of-sample.
2. Test season juga tidak bisa mengujinya, karena 2025/26 bukan musim no-fans.
   Sama seperti §5.5.
3. Yang bisa diuji lintas fold cuma varian A, dan hasilnya tidak significant.

Jalankan `python src/seasonal_hfa.py` untuk angkanya. Modul ini murni laporan:
tidak menulis model produksi dan tidak menyentuh `evaluate.py`.

### 5.8 Exposure-based Poisson (bukan xG) - `src/exposure_poisson.py`

Topik xG (§9) berakhir dengan kesimpulan: tidak ada sumber xG EPL gratis yang
berlisensi jelas. Yang invece adalah memakai kolom `shots`/`shots_on_target`
yang sudah ada di dataset.

Struktur dua tahap. Model offset tunggal `lambda = shots * exp(...)` hanya bisa
dihitung untuk match yang sudah selesai, jadi exposure itu sendiri harus
diprediksi lebih dulu:

```
sigma = E[shots]            (stage 1, penaltyblog Poisson di-fit pada shots)
q     = P(goal | shot)      (stage 2, Poisson dengan offset log(shots))
lambda = sigma * q
```

Varian SOT menambah satu tahap: `q = P(SOT | shot) * P(goal | SOT)`.

**Hasil walk-forward CV (5 fold, season validasi 2020-2021 s.d. 2024-2025,
xi di-grid-search untuk kedua lengan lalu best-vs-best):**

| model | xi terbaik | log loss | RPS | accuracy |
|---|---|---|---|---|
| **sot** | 0.0025 | **0.997688** | **0.209706** | 0.517368 |
| poisson_goals_based | 0.0025 | 1.002893 | 0.211598 | 0.520000 |
| shot_volume | 0.0025 | 1.004002 | 0.212677 | 0.508421 |

Paired vs baseline (negatif = lebih baik):

```
sot           delta log loss -0.005205, CI 95% [-0.008340, -0.002153],
              SIGNIFIKAN, menang 5 dari 5 fold (log loss DAN RPS)
shot_volume   delta log loss +0.001109, CI 95% [-0.005091, +0.007150],
              tidak signifikan, menang 3 dari 5 fold
```

**Kesimpulan: varian SOT berhasil, shot-count biasa gagal.** Varian SOT
membaik di kelima fold pada dua metrik sekaligus, dan CI-nya tidak melintasi
nol — pola yang jarang di eksperimen manapun di project ini (§5.1-§5.7 semuanya
tidak signifikan). Varian `shot_volume` justru sedikit memburuk, jadi efeknya
bukan dari "tembakan" secara umum, tapi dari pemisahan tiga tahap
shots -> SOT -> goal.

**Temuan sampingan yang tak terduga: home advantage praktis seluruhnya ada di
volume, bukan di efisiensi.** Rasio home/away di training season: goals 1.212,
shots 1.204, tapi conversion rate hanya 1.006 dan SOT-rate 0.986. Karena itu
parameter `home_advantage` di stage-2 berakhir di batas bawah ~0.000 — itu
HASIL yang benar, bukan kegagalan optimasi. Test
`test_stage2_home_advantage_is_negligible` mengunci temuan ini.

> **Belum jadi produksi.** Angka di atas masih CV di training season dengan
> bias selection pada xi. `evaluate.py`, `predict_match.py`, dan
> `cv_model_selection.csv` sengaja tidak disentuh. Kalau mau diethyl di test
> season, itu keputusan terpisah.

### 5.9 Blend sistematis semua model (meta-learner di atas OOF) - `src/blend.py`

Korrection premis lebih dulu, karena dua asumsi yang masuk ke permintaan
tidak sesuai kenyataan repo:

- **Tidak ada RF "ter-tuning & terkalibrasi".** Tuning RF: delta RPS
  -0.000013, tidak significant (§5.6). Kalibrasi post-hoc: raw 0.211421,
  sigmoid 0.213966, isotonic 0.215273 — raw menang (§5.2). `train_model.py`
  masih pakai setting Phase 5. Yang ter-tuning hanya XGBoost.
- **Tidak ada model xG.** Yang Week 3 hasil = exposure-based Poisson, bukan xG.

Jadi 7 base model yang di-blend: `poisson`, `dixon_coles`, `elo`,
`logistic_regression`, `random_forest`, `xgboost`, `exposure_sot`.

**Level-2 walk-forward (training season) — angka penentu:**

| model | log loss | RPS |
|---|---|---|
| **blend_simple_average** | **0.979351** | **0.203578** |
| xgboost | 0.984392 | 0.204230 |
| random_forest | 0.985576 | 0.204280 |
| blend_meta | 0.991770 | 0.205512 |
| poisson | 0.992172 | 0.208308 |
| exposure_sot | 0.995162 | 0.209164 |
| elo | 1.103300 | 0.240926 |

**Test season 2025-26 (pelaporan saja):**

| model | log loss | RPS |
|---|---|---|
| bookmaker_avg_odds | **1.015252** | **0.205280** |
| random_forest | 1.037602 | 0.211421 |
| blend_meta | 1.042192 | 0.212752 |
| blend_simple_average | 1.043516 | 0.213098 |
| exposure_sot | 1.076536 | 0.222888 |

**HASIL: blend TIDAK mengalahkan model tunggal terbaik.** Dan lebih penting,
**ranking CV tidak transfer ke test season:**

```
CV  : simple_average 0.979351 vs xgboost 0.984392  -> blend menang -0.005041
Test: simple_average 1.043516 vs random_forest 1.037602 -> blend KALAH +0.005914
      CI 95% [-0.010938, +0.022800], tidak significant
```

blend_meta vs random_forest di test: `+0.004591`, CI `[-0.012508, +0.021644]`,
tidak significant. Ini pola yang sama seperti kesalahan lambda di §5.3:
perbaikan yang terlihat di CV hilang (bahkan berbalik) di test.

**Catatan fold 2.** Di level-2 CV, fold 2 hanya punya 599 baris untuk
melatih meta-learner, dan `blend_meta` di fold itu 1.045805 vs simple average
1.003654. Exclusion fold 2 menaikkan simple_average ke 0.971250 dan
blend_meta ke 0.973758, tapi urutan relatif tidak berubah.

### 5.10 Jarak ke odds bandar setelah 4 minggu perbaikan

Ini jawaban langsung atas pertanyaan "seberapa dekat ke bandar dibanding
sebelum semua perbaikan dimulai". Angka "sebelum" dari capture pre-upgrade di
`research/evaluations/baseline.md` (random_forest, RPS 0.211, log loss 1.036;
bookmaker RPS 0.205, log loss 1.015).

| kandidat | gap log loss sekarang | gap RPS sekarang | perubahan vs dulu |
|---|---|---|---|
| best_single_model (random_forest) | +0.022349 | +0.006141 | **+0.001349** |
| blend_meta | +0.026940 | +0.007473 | +0.005940 |
| blend_simple_average | +0.028264 | +0.007819 | +0.007264 |

Gap SEBELUM: log loss +0.021000, RPS +0.006000.

**Kesimpulan jujur: empat minggu kerja tidak memperkecil jarak ke bandar sama
sekali — jaraknya justru sedikit melebar (+0.0013 log loss).** Dari 7
eksperimen, hanya exposure-based SOT yang signifikan, dan itu baru terbukti di
CV training season, belum diuji di test season. Sisanya (§5.1-§5.7) gagal.

### 5.11 Yang TIDAK berhasil dan tidak dicoba lagi
- Eksclude musim 2020/21 dari training (membuang data, tidak sesuai scope)
- Menambah fitur wasit sebagai kategorikal langsung (Poisson penaltyblog hanya support
  param per-tim + home advantage, bukan kategorikal)
- HFA terpisah per musim untuk musim no-fans (lihat §5.7)
- Blend meta-learner di atas seluruh base model (lihat §5.9) — menang di CV
  tapi tidak transfer ke test season.
- Rata-rata aritmetik semua base model tanpa bobot (§5.9)

---

## 6. Jebakan teknis yang sudah dibayar mahal

### 6.1 `CalibratedClassifierCV(ensemble=False)` TIDAK bisa dipakai dengan TimeSeriesSplit
sklearn `cross_val_predict` mewajibkan splitter **non-overlapping**, sedangkan
TimeSeriesSplit training fold-nya overlap. Akan crash dengan
`ValueError: cross_val_predict only works for partitions`. Harus `ensemble=True`.

### 6.2 Out-of-fold hanya mencakup 2783 dari 3420 baris
Baris paling awal tidak pernah masuk validation fold (TimeSeriesSplit butuh history).
Menghitung metrik di SEMUA baris akan menghasilkan log loss palsu ~7.5 karena probabilitas
0. **Selalu batasi ke baris tercakup.**

### 6.3 Nested parallelism
`build_models()` sudah pakai `n_jobs=-1`. Kalau `GridSearchCV(n_jobs=-1)` juga, setiap
worker membuat thread sendiri -> mesin oversubscribe, proses jauh lebih lambat.
Gunakan `single_threaded()` di `model_tuning.py`.

### 6.4 Definisi ECE harus one-vs-rest
`evaluate.py` menghitung ECE per kelas lalu rata-rata (one-vs-rest). Versi top-label
menghasilkan angka ~8x lebih besar (0.32 vs 0.041) dan TIDAK sebanding dengan
`evaluation_summary.csv`. `model_calibration.py` sudah pakai definisi yang sama.

### 6.5 `runpy.run_path(..., run_name='__main__')` mereset patch
Kalau kamu patch konstanta module lalu jalankan skrip via runpy, `from config import ...`
di dalam skrip akan meng-`reset` global dan patch hilang. Panggil `main()` langsung.

### 6.6 ClubElo rate limit
`api.clubelo.com` sering 502, `clubelo.com` ikut rate-limit. Tanpa jeda + backoff, satu
tim bisa dapat rating di run A dan jatuh ke fallback bottom-3 di run B. Itu membikin
eksperimen before/after tidak apples-to-apples. Cache disk + retry sudah enshrined.

---

## 7. Keputusan desain aktif

### 7.1 ClubElo rescaling
ClubElo (skala ~1580-1748) dan Elo internal (start 1500, K=20, HFA=100) beda skala, jadi
9 tim promosi yang dapat ClubElo terlihat lebih kuat artifisial vs 18 tim fallback.

```
elo_internal = -469.1728 + 1.0893 * elo_clubelo
```
Di-fit OLS dari 1309 pasangan rating per-tanggal, 9 tim non-promosi, window chart
ClubElo, R2 = 0.93. Diterapkan di `get_clubelo_rating` (satu funnel, berlaku untuk API
maupun chart). 18 tim fallback tetap di skala internal apa adanya.

Verifikasi ulang independen menghasilkan `a=-438.18, b=1.0919, R2=0.9287` -> koefisien
di config tervalidasi (slope hampir identik, intercept beda 31 poin / 0.8% range).

### 7.2 Disk cache ClubElo
`data/cache/clubelo_cache.json`, fresh < 7 hari. Kegagalan fetch TIDAK di-cache.
Retry + backoff di kedua fetcher (`CLUBELO_MAX_ATTEMPTS=4`) karena tanpa itu cache tidak
pernah terisi penuh (rate limit parah: run 1 hanya 2/25 tim tanpa retry).

### 7.3 Log cold-start
`data/metadata/elo_coldstart_log.csv` - tim, musim, source yang BENAR-BENAR dipakai,
raw sebelum rescale, dan nilai sesudah. Row fallback punya `raw_clubelo` kosong supaya
nilai fallback tidak bisa salah dilabeli sebagai ClubElo.

### 7.4 XGBoost jadi produksi - dengan catatan jujur

XGBoost dipilih karena **aturan main** (selection berbasis CV log loss), bukan karena
bukti kuat:

- Selisih CV hanya 0.0023, **tidak signifikan** (t-test 5 fold, p=0.28; hanya 3 dari 5
  fold menguntungkan XGBoost)
- Di test season Random Forest justru lebih baik pada RPS (0.211421 vs 0.213533)
  dan log loss (1.037602 vs 1.045309)
- `evaluate.py` sendiri melaporkan RF vs XGB pairwise tidak signifikan (p=0.100)

> Kalau bukti di musim berikutnya tetap slim, kembalikan `XGBOOST_TUNED_PARAMS` di
> `config.py` ke nilai lama (`n_estimators=300, learning_rate=0.03, max_depth=3`).

### 7.5 Track A perlu perlakuan khusus untuk musim 2020/21 - SUDAH DIUJI, TIDAK DIJADIKAN FITUR
Time-decay menyiske musim ini ke 3.6% bobot - efeknya ter-diskon tapi BELUM dihilangkan.
Poisson/Dixon-Coles punya SATU parameter home_advantage global, jadi secara teori tidak
bisa merepresentasikan musim tanpa penonton.

Rekomendasi HFA terpisah sudah diimplementasikan dan diuji di `src/seasonal_hfa.py`.
Hasilnya: **signifikan secara statistik tapi tidak implementable.** Detail dan
batasan strukturalnya di §5.7. Intinya, tidak ada jalur validasi out-of-sample
untuknya selama tidak ada musim no-fans kedua, dan varian yang bisa diuji justru
tidak significant. Keputusan: **jangan dijadikan fitur produksi**, HFA global tetap.

---

## 8. Koordinasi dengan sesi lain

File berikut **milik sesi/agent lain yang jalan paralel**. Jangan disentuh atau
di-overwrite:

```
research/                              # jurnal riset, audit trail (AUDIT-0001, JOURNAL.md)
src/walk_forward.py                    # modul walk-forward
data/metadata/dataset_metadata.json    # provenance dataset
```

Sudah ikut ter-commit di `1740114` (karena satu file): `src/evaluate.py` (bootstrap CI),
`src/predict_match.py` (model selection berbasis CV), `tests/test_select_best_model.py`.

---

## 9. TOPIK TERBUKA: model berbasis xG

User minta eksplorasi model Poisson berbasis xG (expected goals) sebagai pelengkap model
goals-based yang sekarang.

### 9.1 Understat DITOLAK - hasil checking

```
https://understat.com/robots.txt  ->  HTTP 200
User-agent: *
Disallow: /
```

Seluruh situs closed untuk semua crawler. Di-fetch 2x, SHA identik. **Tidak ada** Terms of
Service, dokumentasi API, maupun kontak lisensi di homepage. Library komunitas
(`understatapi` dll) tetap ada, tapi dukungan Understat hanya pernah konfirmasi via email
**8 November 2018** bahwa data boleh dipakai non-komersial, dengan catatan "This stance
is subject to change". Itu informal 8 tahun lalu, bukan lisensi.

Alasan menolak: konsistensi dengan preseden project ini. FBref sudah ditolak karena
Selenium + ToS. Kalau Understat lolos hanya karena "praktis ditegakkan", standar yang
dijaga jadi tidak konsisten.

### 9.2 Data lokal tidak punya xG
`matches_clean.csv` (33 kolom) dan `data/raw/*.csv` (football-data.co.uk) tidak punya
kolom xG sama sekali. Tidak ada jalan pintas tanpa sumber baru.

### 9.3 Opsi (BELUM DIPILIH - butuh keputusan user)

| Sumber | Lisensi | xG match-level tim? | Tier |
|---|---|---|---|
| **OpenFootAPI** | API resmi, ada metadata lisensi | ya, + arsip 17 musim PL | Free (5k req/bulan) |
| API-Football | API resmi | perlu verifikasi endpoint | Free 100 req/hari |
| TheStatsAPI | API berbayar | ya | $50/bulan |
| BigBallsData | API resmi | **TIDAK** - hanya agregat per-pemain per-musim | Free |

Catatan: BigBallsData dicoret karena endpoint xG-nya season-aggregate, bukan per-match.

**Poin penting**: model xG berbeda antar provider menghasilkan angka yang TIDAK akan
cocok untuk tembakan yang sama. Kalau ganti sumber, hasilnya tidak sebanding langsung
dengan eksperimen yang sudah ada.

> User sudah bilang "Oke" tapi **belum memilih opsi A / B / C**. Tanya dulu sebelum kerja.

---

## 10. Cara menjalankan

Semua dari root project dengan interpreter venv:

```powershell
# Test (selalu jalankan setelah ubah src/)
.\.venv\Scripts\python.exe -m pytest tests/ -q

# Pipeline (urutan penting)
.\.venv\Scripts\python.exe src\data_collection.py      # Fase 1 (butuh internet)
.\.venv\Scripts\python.exe src\feature_engineering.py  # Fase 3
.\.venv\Scripts\python.exe src\statistical_models.py   # Fase 4
.\.venv\Scripts\python.exe src\train_model.py          # Fase 5
.\.venv\Scripts\python.exe src\evaluate.py             # Fase 7

# Prediksi interaktif
.\.venv\Scripts\python.exe src\predict_match.py

# Eksperimen (opsional)
.\.venv\Scripts\python.exe src\model_calibration.py    # Fase 18
.\.venv\Scripts\python.exe src\model_tuning.py         # Fase 19
```

---

## 11. Peringatan operasional

1. **Push setelah commit.** Sesi ini 12 commit tertahan lokal selama berjam-jam karena
   lupa push; user kaget GitHub masih 2 bulan. Commit tanpa push = kerja hilang dari
   pandangan user.

2. **Cek output non-ASCII sebelum commit.** Tiga kali sempat ada karakter Mandarin nyasar
   di docstring/komentar/commit message. Alasan: menulis teks campuran ID/EN kadang
   gagal encode. Selalu scan sebelum commit.

3. **Jalankan `pytest` setelah setiap ubahan `src/`**, terutama `config.py`,
   `feature_engineering.py`, `team_mapping.py`. `tests/test_config.py` punya guard AST
   yang menangkap konstanta yang didefinisikan ulang di module lain.

4. **Jalankan `git status` sebelum commit** dan cek file mana yang milik sesi lain.

5. **Jangan edit `data/processed/` atau `models/` manual.** Semua harus bisa diambil ulang
   dengan menjalankan ulang pipeline.

---

## 14. Ringkasan temuan audit (29 September 2026)

Empat temuan dari pembacaan kode + artefak pada sesi ini. Hanya satu yang
menyentuh `src/`, sudah diperbaiki di commit terpisah.

### 14.1 Hardcoded constants di `evaluate.py` (SUDAH DIPERBAIKI)
Tiga nilai ditulis langsung, bukan import dari `config.py`:

| Lokasi | Nilai lama | Constant yang benar |
|---|---|---|
| `evaluate.py:151` | `k=20.0` | `ELO_K` |
| `evaluate.py:843` | `k=20.0` | `ELO_K` |
| `evaluate.py:151,843` | `home_field_advantage=100.0` | `ELO_HOME_ADVANTAGE` |
| `evaluate.py:836` | `max_goals=15` | `MAX_GOALS` |

Guard AST di `tests/test_config.py` **tidak menangkap ini** karena dia hanya
mencari assignment module-level, sedangkan ketiga nilai ini muncul sebagai
keyword argument di dalam fungsi. Konsekuensinya kalau tidak diperbaiki:
mengubah `ELO_K` atau `MAX_GOALS` di `config.py` membuat
`statistical_models.py` dan `evaluate.py` memakai skala berbeda tanpa error —
metrik jadi tidak apples-to-apple dan perbandingan model diam-diam tidak
valid. Regression test baru mengunci kebocoran ini lewat monkeypatch.

### 14.2 Angka Track A di `cv_model_selection.csv` itu in-sample — **SELESAI**
`evaluate._track_a_selection_metrics()` me-load model yang **di-fit di data
training**, lalu menilainya di **data training yang sama**. Angka Track B di
tabel yang sama berasal dari TimeSeriesSplit out-of-fold. Jadi kolom
`cv_log_loss_mean` mencampur dua basis yang berbeda.

**Status: sudah diperbaiki (30 September 2026).** `evaluate.py` sekarang
membaca `data/processed/track_a_cv_results.csv` hasil `src/track_a_cv.py`
dan menghitung ulang mean/std per fold. Ditambah guard keras: kalau ada
baris dengan `validation_season == TEST_SEASON`, `evaluate.py` melempar
ValueError alih-alih diam-diam memakainya.

File agregat `track_a_cv_summary.csv` sengaja **tidak** dibaca untuk
selection, karena tidak menyimpan kolom season sehingga guard kontaminasi
tidak bisa diverifikasi darinya.

Konsekuensi: **goal-model flip** dari Dixon-Coles ke Poisson (§2.1).
In-sample fallback masih ada kalau `track_a_cv.py` belum dijalankan, dengan
`selection_basis` yang jujur menandainya in-sample.

### 14.3 `README.md` salah menulis rentang training
Tercantum "2021-2025 train" untuk corner model. Sebenarnya training period
adalah **2016-2017 s.d. 2024-2025** (9 musim, 3.420 match) dan test season
2025-2026 (380 match), sama seperti Phase 4. Sudah dikoreksi.

### 14.4 Angka `research/evaluations/baseline.md` berbeda dari sekarang
Tabel di `research/evaluations/baseline.md` (capture 2026-09-21) tidak cocok
dengan `data/processed/evaluation_summary.csv` sekarang:

| model | baseline (21 Sep) | sekarang | penjelasan |
|---|---|---|---|
| Random Forest | 1.036 | 1.037602 | refit, angka stabil |
| XGBoost | 1.056 | 1.045309 | `XGBOOST_TUNED_PARAMS` (commit 34876d0) |
| Elo | 1.083 | 1.075032 | ClubElo rescaling (commit 2c4cc61) |
| RF ECE | 0.050 | 0.040824 | definisi ECE one-vs-rest sudah diseragamkan |

**Ini bukan error** — `baseline.md` memang sengaja capture kondisi *pre-upgrade*
sebelum commit Phase 19 dan ClubElo rescale, jadi perbedaan angka memang
diharapkan. Dicatat supaya tidak salah dibaca sebagai regresi. File tersebut
milik sesi paralel, jadi tidak diubah.

---

## 13. Sesi 29 September 2026 (sesi setelah HANDOFF.md pertama)

### 13.1 Yang dikerjakan
1. **C1** — Sinkronisasi `README.md` (Fase 18/19, koreksi rentang training,
   konfirmasi model selection berbasis CV).
2. **C2** — Hardcoded constants di `evaluate.py` diganti import dari `config.py`
   + regression test baru (§14.1).
3. **C3** — `src/track_a_cv.py`: walk-forward CV untuk Track A, hasil ditulis ke
   file terpisah. **`evaluate.py` dan `predict_match.py` sengaja TIDAK diubah**
   sampai angkanya direview user.
4. **C4** — `src/seasonal_hfa.py`: eksperimen HFA terpisah untuk musim no-fans.

### 13.2 Baseline test
`92 passed` saat mulai sesi, `main` == `origin/main` — tidak ada commit tertahan.
Setelah C1-C4: **`125 passed`**.

### 13.3 Hasil Track A walk-forward CV (`src/track_a_cv.py`)

5 fold musiman, season validasi 2020-2021 s.d. 2024-2025, `TEST_SEASON` tidak
pernah jadi fold. Angka out-of-sample per model (mean 5 fold):

| model | log loss | RPS | accuracy |
|---|---|---|---|
| poisson | **1.103723** | 0.243066 | 0.4489 |
| dixon_coles | 1.105310 | 0.243113 | 0.4479 |
| elo | 1.106281 | **0.241574** | **0.4616** |

Perbandingan paired per fold terhadap Poisson: Dixon-Coles `+0.001586` (1 dari 5
fold membaik), Elo `+0.002557` (2 dari 5 fold). Dixon-Coles tidak memberi
peningkatan bermakna di luar sampel — konsisten dengan `rho ~ -0.004` yang sudah
dicatat sebagai limitation di `AGENTS.md`.

> **Angka-angka ini TIDAK disambungkan ke `evaluate.py`.** `cv_model_selection.csv`
> masih pakai angka in-sample Track A seperti sebelumnya, dan
> `predict_match.py` selection goal-model masih membaca angka itu. Tersedia tapi
> belum dipakai, supaya perubahannya tidak terjadi tanpa review.

> **Catatan comparability:** Elo di `track_a_cv.py` sengaja tanpa cold-start
> ClubElo (semua tim mulai dari 1500), supaya deterministik dan tidak memanggil
> API luar yang bisa rate-limit. Jadi **angka Elo di sana tidak langsung
> sebanding** dengan Elo produksi yang memakai ClubElo.

### 13.4 Hasil eksperimen HFA no-fans (`src/seasonal_hfa.py`)

Detail lengkap di §5.7. Ringkasnya: selisih HFA antar musim **signifikan**
(-0.1407, CI [-0.2849, -0.0723]) tapi tidak implementable karena tidak ada jalur
validasi out-of-sample. **HFA global tetap dipakai di produksi.**

### 13.5 File baru hasil C3/C4

| File | Isi |
|---|---|
| `src/track_a_cv.py` | Walk-forward CV Track A |
| `src/seasonal_hfa.py` | Eksperimen HFA no-fans |
| `tests/test_track_a_cv.py` | 16 test |
| `tests/test_evaluate_config_constants.py` | 6 test (guard hardcoded constants) |
| `tests/test_seasonal_hfa.py` | 11 test |

Artefak output di `data/processed/`: `track_a_cv_{results,summary,comparison}.csv`,
`seasonal_hfa_{comparison,estimates,fold_details}.csv`. Metadata JSON di
`data/metadata/{track_a_cv,seasonal_hfa}_summary.json`.

Kedua modul diverifikasi **deterministik** (dua run berturut-turut menghasilkan
CSV identik) dan terbukti **tidak menyentuh** `cv_model_selection.csv`.

### 13.6 Catatan penting untuk sesi berikutnya
- **Topik terbuka xG belum selesai.** Sumber data belum dipilih user. Semua
  opsi berlisensi (OpenFootAPI / API-Football / TheStatsAPI) masih menunggu.
  Jangan scraping apa pun untuk ini sebelum user memutuskan.
- Angka Track A di `cv_model_selection.csv` sudah **disambung** ke walk-forward
  CV pada 30 September 2026; goal-model flip ke Poisson (§2.1). Kalau
  `track_a_cv_results.csv` dihapus, `evaluate.py` otomatis jatuh ke angka
  in-sample dengan label jujur — bukan crash.
- **Jangan mengulang HFA terpisah per musim** (§5.7). Sudah diuji dan tidak ada
  jalur validasinya.

---

## 12. Aturan keras project (dari AGENTS.md)

1. **Anti-leakage**: fitur match T hanya boleh memakai data sebelum T. Split waktu:
   `season != TEST_SEASON` = train, `TEST_SEASON` = test. TimeSeriesSplit untuk CV.
2. **Odds bandar adalah benchmark, bukan fitur.**
3. **Pilihan model terbaik harus berdasarkan CV, bukan hasil test set.**
4. **Evaluasi jujur**: kalau metrik berubah setelah refactor, jelaskan kenapa.
5. **`models/` dan `data/processed/`** bisa diambil lagi dengan menjalankan ulang pipeline.
6. **JANGAN menabrak keputusan di `scope-predictive-analysis-bola.md`** tanpa alasan yang ditulis.

Semua konstanta pipeline di `src/config.py` - single source of truth. Output user-facing
dalam Bahasa Indonesia casual; docstring teknis dalam Bahasa Inggris.
