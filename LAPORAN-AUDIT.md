# Laporan Audit Menyeluruh — football-predictive-analysis

**Tanggal:** 30 September 2026
**Untuk:** konsultasi ke AI lain
**Metode:** baca semua dokumen + kode, lalu jalankan test suite, seluruh pipeline offline, dan keempat skrip eksperimen. Semua temuan di bawah diverifikasi dengan perintah yang bisa diulang.
**Status saat audit:** `main` == `origin/main`, 178 test lulus, 12 commit ter-push.

---

## 0. Ringkasan eksekutif

Tiga temuan yang paling penting, urut dari yang paling serius:

1. **`value_betting.py` crash saat dijalankan** — `NameError: name 'Path' is not defined`. Script yang README dokumentasikan sebagai Fase 8 tidak bisa jalan sama sekali, dan tidak ada test yang menutupinya.
2. **\valuation_summary.csv\ tidak reproducible** - isinya bergantung pada apakah \stacking.py\ pernah dijalankan. andom_forest_stacked_xg\ muncul atau tidak tergantung file opsional yang ada.
3. **Tidak ada satu pun model yang bisa menebak Draw.** Di 380 match test, model terbaik menebak Draw 2 kali, dan **semuanya 0% draw recall**. Ini kegagalan substantif yang sudah 4 minggu tidak tersentuh.

Secara-positive: 4 minggu kerja **tidak memperkecil jarak ke odds bandar sama sekali** (gap log loss +0.021000 → +0.022349, justru melebar). Dari 9 eksperimen, hanya exposure-based Poisson yang signifikan, dan itu pun baru di CV training season.

---

## 1. Temuan tingkat 1 — Bug yangconfirmed

### 1.1 `value_betting.py` crash — FATAL untuk fitur Fase 8

```
$ python src\value_betting.py
NameError: name 'Path' is not defined
  File "src\value_betting.py", line 66, in parse_args
    type=Path,
```

`Path` dipakai di `parse_args()` tapi tidak pernah di-import. Cek import di `value_betting.py:5-6` hanya ada `argparse` dan `json`.

**Kenapa lolos:** tidak ada satu pun test yang meng-import atau memanggil `value_betting`. `tests/` tidak menyebut nama file itu sama sekali. Modul ini tidak pernah dieksekusi sejak commit terakhir yang menyentuhnya, jadi bug-nya tidak pernah terlihat.

**Dampak:** README mendokumentasikan Fase 8 lengkap dengan contoh perintah, tapi perintahnya tidak bisa dijalankan. AGENTS.md juga menyebut `src/value_betting.py` sebagai opsional yang jalan.

**Perbaikan:** tambah `from pathlib import Path` + test yang memanggil `parse_args()` dengan argumen dummy.

### 1.2 `evaluation_summary.csv` bergantung pada file opsional

`evaluate.py:231` — `if STACKING_PATH.exists():` baru masukkan `random_forest_stacked_xg` ke tabel.-arts Konsekuensinya:

```
Sebelum stacking.py dijalankan : 7 model di evaluation_summary.csv
Setelah stacking.py dijalankan : 8 model (tambah random_forest_stacked_xg)
```

Artefak "canonical" Fase 7 jadi berubah isi hanya karena ada/tidaknya file opsional. Siapa pun yang menjalankan `evaluate.py` di mesin berbeda bisa mendapat tabel berbeda tanpa ada error.

** exacerbated oleh:** `HANDOFF.md` §2.2 mendokumentasikan tabel 7 model, jadi dokumentasi itu hanya cocok untuk mesin yang tidak pernah menjalankan `stacking.py`.

**Perbaikan:** di `evaluate.py`, selalu tulis 8 baris dengan `NaN` untuk yang tidak ada, plus kolom `available` — supaya skema file stabil.

### 1.3 Label "CV log loss" dipamerkan untuk angka yang BUKAN CV

`predict_match.py` mencetak:

```
Model   : Dixon-Coles (CV log loss: 0.983277, basis: train-only single split (no CV))
Model   : XGBoost (CV log loss: 0.980369, basis: TimeSeriesSplit 5-fold CV)
```

Untuk Dixon-Coles, angka 0.983277 itu **in-sample** — model dievaluasi di data yang sama dengan training-nya. Label "CV log loss" di user-facing output salah dan bersamaan dengan `basis:` yang justru mengatakannya. User yang tidak teliti akan baca dua baris itu sebagai setara.

**Perbaikan:** cetak label dinamis mengikuti `selection_basis`, misal "train log loss (in-sample)" vs "CV log loss".

### 1.4 `elo_coldstart_log.csv` tidak deterministik antar-run

`statistical_models.py:463` `identify_promoted_teams()` mengembalikan `set`, lalu di-iterasi langsung. Iterasi set di Python tidak punya urutan stabil antar-proses (hash randomization untuk string). Akibatnya setiap `statistical_models.py` run menulis ulang file yang **sudah di-commit** dengan urutan baris berbeda.

Verifikasi: isi 27 baris identik disregarding urutan, hanya urutan yang berubah. `git status` jadishowing `M data/metadata/elo_coldstart_log.csv` setiap pipeline run, dan diff-nya noise.

**Perbaikan:** `sorted(promoted)` sebelum iterasi.

---

## 2. Temuan tingkat 2 — Inkonsistensi metodologi

### 2.1 Angka Track A di tabel model selection itu in-sample

`evaluate.py:753` `_track_a_selection_metrics()` me-load model yang di-fit di data training, lalu menilainya di data training yang sama. Angka Track B di tabel yang sama out-of-fold. Jadi kolom `cv_log_loss_mean` mencampur dua basis berbeda.

Label `selection_basis` sudah jujur (`"train-only single split (no CV)"`), jadi tidak ada klaim palsu. Dampaknya belum merusak `select_best_model` karena kandidat goal-model semuanya Track A dan kandidat klasifikasi semuanya Track B.

**Yang sudah ada perbaikannya:** `src/track_a_cv.py` menghasilkan angka Track A out-of-sample yang benar (5 fold, TEST_SEASON tidak pernah jadi fold). Hanya belum disambungkan.

### 2.2 Dulu ada DUA implementasi "walk-forward" dengan fold berbeda

| file | fold | cakupan | status |
|---|---|---|---|
| `src/track_a_cv.py` | 5 | Track A, **season training saja** | commit oleh sesi ini |
| `src/walk_forward.py` | 6 | Track B, **termasuk 2025-2026 sebagai fold** | untracked, milik sesi paralel |

Keduanya menyebut diri "walk-forward" tapi punya jumlah fold, cakupan season, dan cakupan track berbeda. `walk_forward.py` memasukkan test season sebagai fold evaluasi — itu sah untuk *pelaporan*, tapi berbahaya kalau output-nya dipakai untuk model selection. Kalau ada dua orang baca keduanya, mereka akan menyimpulkan angka berbeda untuk "model terbaik".

Yang sudah diverifikasi aman: implementasi RPS lokal di `walk_forward.py` **identik** dengan `pb.metrics.rps_average` (selisih 0.00e+00). Jadi minimal definisi metriknya tidak ganda.

**Saran:** rename salah satu supaya tidak ada dua "walk-forward" dengan arti berbeda.

### 2.3 `AUDIT_COMPLETE.md` mengklaim sesuatu yang sudah direvert

Dokumen itu (23 Agustus, tidak pernah diupdate sejak) mengklaim:

> "All fixes: IMPLEMENTED & VERIFIED"
> "FIX 1: Class Imbalance Handling - Logistic Regression: added `class_weight='balanced'`"
> "Random Forest: added `class_weight='balanced'`"
> "Final state: **25 PASS / 0 FAIL**"

Kenyataan sekarang: `class_weight` **tidak ada sama sekali** di `train_model.py`. `HANDOFF.md` §5.1 mencatat itu sudah di-revert karena memperburuk RPS.

Jadi ada dokumen di root yang secara eksplisit menyatakan kondisi yang sudah dibatalkan. readership yang baca `AUDIT_COMPLETE.md` tanpa `HANDOFF.md` akan mendapat gambaran yang salah besar.

**Saran:** hapus `AUDIT_COMPLETE.md` atau tambah banner di bagian atas bahwa isinya sudah usang dan digantikan `HANDOFF.md`.

### 2.4 Stacking (Fase 6) adalah hasil negatif yang tidak tercatat

`stacking.py` saat dijalankan:

```
random_forest_controlled_baseline  cv 0.984091  test 1.035149
random_forest_stacked_xg           cv 0.989173  test 1.040039
improvement -0.004890, CI 95% [-0.016272, +0.006570]  -> tidak significant, dan lebih BURUK
```

Artinya: menumpuki expected goals ke Random Forest **membuat lebih buruk**, baik di CV maupun di test. Ini hasil negatif. Tapi `HANDOFF.md` §5 (daftar hasil negatif) **tidak menyebut stacking sama sekali**. Sesi berikutnya yang membaca §5 akan mengulang percobaan yang sudah selesai.

---

## 3. Temuan tingkat 3 — Technical debt & gap

### 3.1 `scipy` dipakai langsung tapi tidak ada di `requirements.txt`

```
blend.py:62         from scipy.special import gammaln
exposure_poisson.py:97  from scipy.optimize import minimize
exposure_poisson.py:98  from scipy.special import gammaln
seasonal_hfa.py:51  from scipy.optimize import minimize_scalar
```

`scipy` hanya ada sebagai dependensi transitif scikit-learn. Kalau scikit-learn suatu saat mengubah dependensinya, ketiga modul eksperimen itu langsung rusak, dan `requirements.txt` tidak memberi clue.

### 3.2 Tidak ada feature importance yang disimpan

`train_model.py` tidak menyimpan `feature_importances_` maupun permutation importance. Padahal 38 fitur feeding 3 model ML, dan tidak ada cara untuk tahu fitur mana yang benar-benar dipakai. `HANDOFF.md` §5.5 mengutip "Feature importance 0.0042, peringkat 36 dari 38" — angka itu dari percakapan sebelumnya yang tidak direproduksi kode-nya, jadi tidak bisa diverifikasi ulang.

### 3.3 `penaltyblog` sudah tidak kompatibel dengan numpy 2.5

Setiap run menghasilkan 6.890 warning:

```
DeprecationWarning: Setting the shape on a NumPy array has been
deprecated in NumPy 2.5.
  penaltyblog/models/poisson.py:240
  penaltyblog/models/dixon_coles.py:257
```

`penaltyblog` di-pin `==1.11.0` di `requirements.txt`. Saat numpy 2.5 jadi error, seluruh Track A mati. Versi terpasang sekarang sangat baru: numpy 2.5.1, pandas 3.0.5, xgboost 3.3.0, sklearn 1.9.0 — semua major version yang lebih baru dari yang biasa dipakai di tutorial football analytics.

### 3.4 `data/metadata/*.json` hasil eksperimen tidak pernah di-commit

Precedent ada: `calibration_summary.json` dan `tuning_summary.json` **sudah** di-commit. Tapi empat JSON baru (`track_a_cv_summary.json`, `seasonal_hfa_summary.json`, `exposure_poisson_summary.json`, `blend_summary.json`) tidak, dan `.gitignore` juga tidak mencakup `data/metadata/`. Jadi statusnya limbo: tidak di-commit maupun di-ignore.

### 3.5 Tidak ada `[project]` di `pyproject.toml` / packaging

Semua modul di-import dengan `from config import ...`, yang hanya jalan kalau `src/` ada di `sys.path`.-repo tidak punya `pyproject.toml`, `setup.py`, atau `conftest.py` yang menambah path. `tests/conftest.py` hanya 106 byte. Artinya menjalankan test dari root repo **harus** lewat `.venv\Scripts\python.exe -m pytest` dengan PYTHONPATH, dan `import config` dari REPL biasa akan gagal.

Verifikasi: `python -c "import config"` dari root tanpa PYTHONPATH → `ModuleNotFoundError`.

---

## 4. Temuan tingkat 4 — Risiko proses

### 4.1 File milik sesi paralel tidak pernah di-commit

```
?? research/                              (10 file, 70 KB)
?? src/walk_forward.py                    (8.4 KB)
?? data/metadata/dataset_metadata.json
```

`git log -- research/` kosong — **tidak pernah ada satu pun commit** untuk direktori itu. Kalau disk ini hilang atau di-reset, semua audit trail (`AUDIT-0001.md` 14 findings, `JOURNAL.md` 5 task selesai, `MANIFEST.md`, 5 template) hilang permanen.

`research/JOURNAL.md` terakhir di-update 22 September — sudah 8 hari basi, dan 4 minggu kerja terakhir sama sekali tidak tercatat di sana.

### 4.2 `dataset_metadata.json` provenance-nya basi sebulan

Isinya: `last_updated: 2026-08-01`, `provenance_recorded_at: 2026-09-22`. Sekarang 30 September. Sementara `matches_clean.csv` sendiri di-refetch 29 September. Jadi metadata itu menggambarkan dataset yang berbeda dari yang ada di disk.

---

## 5. Yang NON-ASCII TIDAK jadi masalah (saya sempat salah, dikoreksi)

Awal audit saya melihat tanda `?` dan karakter aneh di beberapa dokumen dan hampir mencatatnya sebagai kerusakan encoding. Setelah ditelusuri dengan `repr()` dan kode ordinal, **`?` itu artefak rendering terminal PowerShell**, bukan isi file. `AUDIT_COMPLETE.md` berisi `✅` dan `→` yang valid UTF-8.

Jadi: **tidak ada kerusakan encoding di dokumen mana pun.** Saya catat ini karena saya sendiri sempat salah dan sudah dikoreksi, supaya tidak diteruskan ke Anda sebagai temuan palsu.

---

## 6. Ringkasan hasil 4 minggu kerja

K confronto langsung dengan capture pre-upgrade (`research/evaluations/baseline.md`):

```
                                         SEBELUM      SEKARANG     PERUBAHAN
jarak log loss ke bandar (RF)           +0.021000    +0.022349    +0.001349
jarak RPS ke bandar (RF)                 +0.006000    +0.006141    +0.000141
```

**Jarak tidak berkurang sama sekali, justru sedikit melebar.**

Dari 9 eksperimen yang pernah dijalankan:

| # | eksperimen | hasil |
|---|---|---|
| 1 | class_weight balanced | gagal, direvert (§5.1) |
| 2 | sigmoid/isotonic calibration | gagal (§5.2) |
| 3 | draw-prior lambda | gagal (§5.3) |
| 4 | fitur wasit discipline | tidak significant (§5.4) |
| 5 | is_empty_stadium | tidak significant (§5.5) |
| 6 | tuning Random Forest | tidak significant (§5.6) |
| 7 | HFA terpisah no-fans | significant tapi tidak implementable (§5.7) |
| 8 | exposure-based Poisson SOT | **SIGNIFIKAN** di CV (§5.8) |
| 9 | blend 7 model | menang CV, kalah test (§5.9) |
| — | stacking Fase 6 | tidak significant, **tidak tercatat** (§2.4 di laporan ini) |

Hanya #8 yang significant, dan itu pun baru diuji di CV training season — belum pernah disentuh test season.

---

## 7. Masalah substantif yang belum Priority: Draw

Ini yang paling mengganjal dan tidak ada yang menyentuhnya sejak awal.

```
380 match test season
bookmaker_avg_odds   menebak Draw: 0 kali
random_forest        menebak Draw: 2 kali
logistic_regression  menebak Draw: 2 kali
xgboost              menebab Draw: 0 kali
poisson/DC/elo       menebak Draw: 0 kali

draw recall: SEMUANYA 0.0
```

Artinya di 380 pertandingan, **tidak satu pun model pernah benar menebak hasil Draw.** Draw adalah ~25% dari pertandingan EPL. Jadi setiap model systematic salah pada seperempat data.

Yang sudah dicoba dan gagal: `class_weight='balanced'` (§5.1) dan draw-prior adjustment (§5.3). Keduanya merusak RPS.

Pertanyaan yang layak dibawa ke AI lain: apakah approaches yang benar-benar berbeda sudah dicoba? Misalnya:
- Two-stage / hurdle model: model terpisah untuk "apakah ini Draw" lalu "siapa yang menang kalau bukan Draw"
- Loss function yang menghukum overconfidence pada non-Draw secara asimetris (RPS sudah semi-ordinal, tapi mungkinISP-nya yang perlu diubah)
- Recalibration conditional: kalibrasi berbeda untuk prediksi yang kemungkinan besar Draw vs bukan

---

## 8. Daftar perintah verifikasi

Semua temuan di atas bisa dicek ulang:

```powershell
# 1.1 bug value_betting
.\.venv\Scripts\python.exe src\value_betting.py
# -> NameError: name 'Path' is not defined

# 1.2 evaluation_summary bergantung file opsional
.\.venv\Scripts\python.exe -c "import pandas as pd; print(sorted(pd.read_csv('data/processed/evaluation_summary.csv').model.unique()))"
# 8 model SEKARANG; 7 model sebelum stacking.py dijalankan

# 1.3 label CV untuk angka in-sample
$env:PYTHONPATH='src'; "1`nArsenal`n2`n1" | .\.venv\Scripts\python.exe src\predict_match.py
# lihat baris "Model : Dixon-Coles (CV log loss: 0.983277, basis: train-only single split (no CV))"

# 1.4 log cold-start tidak deterministik
.\.venv\Scripts\python.exe src\statistical_models.py
git diff --stat data/metadata/elo_coldstart_log.csv
# 27 baris, isi sama, urutan berubah

# 2.3 klaim class_weight yang usang
Select-String -Path AUDIT_COMPLETE.md -Pattern "class_weight"
Select-String -Path src\train_model.py -Pattern "class_weight"
# dokumen: ada; kode: TIDAK

# 2.4 stacking negatif tidak tercatat
.\.venv\Scripts\python.exe src\stacking.py
# improvement -0.004890, CI [-0.016272, +0.006570]

# 3.1 scipy tidak di requirements
Select-String -Path requirements.txt -Pattern scipy   # kosong
Select-String -Path src\*.py -Pattern "from scipy"     # 4 hasil

# 3.3 deprecation penaltyblog
.\.venv\Scripts\python.exe -m pytest tests/ -q
# 6890 DeprecationWarning dari penaltyblog

# 4.1 file sesi paralel tidak ter-commit
git log -- research/            # kosong
git status --short              # research/, src/walk_forward.py untracked
```

---

## 9. Rekomendasi prioritas

**Perbaikan cepat (sudah ada bukti jelas):**
1. `value_betting.py` — tambah `from pathlib import Path` + test. 30 menit.
2. `identify_promoted_teams` — `sorted()`. 2 menit.
3. `requirements.txt` — tambah `scipy`. 1 menit.
4. `predict_match.py` — label dinamis untuk in-sample vs CV. 15 menit.

**Perbaikan metodologi:**
5. `evaluate.py` — skema output stabil (8 baris selalu, dengan flag tersedia/tidak).
6. `AUDIT_COMPLETE.md` — hapus atau beri banner "usang".
7. `HANDOFF.md` §5 — tambahkan stacking sebagai hasil negatif ke-8.
8. `cv_model_selection.csv` — sambungkan angka `track_a_cv.py` (perlu-hatian: goal-model selection bisa flip).

**Risiko proses:**
9. Commit `research/` dan `src/walk_forward.py` — ini Auditori loss risk, tidak boleh diabaikan.
10. Sinkronkan `research/JOURNAL.md` dengan 4 minggu terakhir.

**Yang perlu diputuskan (bukan sekadar dikerjakan):**
11. Rename salah satu dari dua "walk-forward" supaya tidak ambigu.
12. Apakah eksposur xG mau dilanjutkan. Rekomendasi saya: **tidak ada sumber gratis yang layak**, tapi kalau mau, TheStatsAPI $50/bulan dengan provenance yang harus dikonfirmasi dulu.
13. Draw — ini masalah yang belum priorities dan butuh approaches yang benar-benar baru.

---

*Dokumen ini dibuat dengan menjalankan kode, bukan hanya membacanya. Semua angka di sini berasal dari run aktual pada 30 September 2026. Tidak ada temuan yang FIXME berdasarkan asumsi — kalau tidak bisa diverifikasi, tidak ditulis.*
