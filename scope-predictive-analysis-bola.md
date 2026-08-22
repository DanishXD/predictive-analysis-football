# Predictive Analysis — Hasil Pertandingan Sepak Bola (EPL) — v2

## 1. Tujuan
Bikin model machine learning & statistik yang bisa memprediksi hasil pertandingan sepak bola (Menang / Seri / Kalah) berdasarkan data historis pertandingan & statistik tim. Fokus utama: belajar end-to-end workflow data science yang metodologinya benar (bukan cari "alpha" buat ngalahin bandar).

## 2. Keputusan Konkret (Default — ganti kalau mau beda)
- **Liga:** English Premier League (EPL) — data paling lengkap & bersih
- **Rentang data:** 10 musim terakhir — 2016/17 s.d. 2025/26 (musim 2025/26 tetap jadi test period)
  > *Naik dari 5 ke 10 musim: 5 musim (~1.900 match) masih tergolong kecil buat model ML kompleks. Catatan: musim 2016/17-2018/19 itu sebelum VAR diperkenalkan di EPL (VAR masuk musim 2019/20), jadi training data mencampur 2 era yang karakteristiknya bisa beda (misal jumlah penalti/kartu). Belum ditangani dengan fitur khusus — dicatat sebagai limitation.*
- **Target prediksi (Fase utama):** Full-Time Result — 3 kelas (Home Win / Draw / Away Win)
- **Target lanjutan (opsional):** Over/Under 2.5 Goals, Both Teams to Score (BTTS)

> **Penting soal season:** data 2016/17–2024/25 di atas itu buat **TRAINING** (ngajarin model pola historis), dan 2025/26 jadi test period — **BUKAN** musim yang mau diprediksi. Model yang udah jadi dipakai buat prediksi pertandingan ke depan (termasuk musim 2026/27) lewat `predict_match.py` (Fase 9), asalkan fitur tim (form, Elo rating) di-update pakai data paling baru yang tersedia. Catatan: jendela transfer musim panas 2026 (sebelum musim 2026/27 mulai) bisa bikin rating/form dari data 2025/26 udah agak basi begitu musim baru mulai, terutama buat tim yang belanja pemain besar-besaran — perlu update data begitu pertandingan 2026/27 mulai jalan.

## 3. Model yang Dipakai — 2 Track

**Track A — Model statistik klasik (basis kuat, standar industri football analytics)**
- **Poisson Goal Model** — tiap tim punya parameter attack/defense strength + home advantage; dari situ dihitung peluang tiap kemungkinan skor
- **Dixon-Coles** — versi improved Poisson yang koreksi skor rendah (0-0, 1-0, 0-1, 1-1) + time-decay (match lama pengaruhnya makin kecil)
- **Elo / Pi-rating** — rating kekuatan tim yang di-update tiap pertandingan; hasilnya juga dipakai sebagai *fitur* buat Track B
  > *Cold-start tim promosi: pakai rating awal dari **ClubElo** (`soccerdata.ClubElo` — API publik gratis di api.clubelo.com, udah nge-rate tim lintas divisi termasuk Championship) sebagai starting point, lebih akurat dari sekadar rata-rata generik karena tiap tim promosi kekuatannya beda-beda. Fallback ke rata-rata Elo 3 tim terbawah klasemen musim sebelumnya kalau ClubElo API down/nama tim ga ketemu. (Alternatif lanjutan kalau mau independen dari servis luar: tarik data EFL Championship sendiri dari football-data.co.uk kode `E1`, tapi butuh penyesuaian tambahan buat gap kualitas antar divisi.)*
- Semua ini pakai library **`penaltyblog`** (Python) — udah ada implementasi Poisson, Dixon-Coles, Elo/Massey/Colley/Pi rating, scraper football-data.co.uk, dan fungsi decode odds bandar (hilangin overround)

**Track B — Model machine learning (fleksibel, bisa gabung banyak fitur)**
- Baseline: Logistic Regression
- Random Forest
- XGBoost / LightGBM
- **(Opsional, Fase lanjutan) Stacking:** masukin output Track A (Elo rating gap, expected goals dari Poisson) sebagai fitur tambahan ke model ML — teknik umum di football analytics buat gabungin kekuatan model statistik + fleksibilitas ML
- **Class imbalance handling:** Draw itu kelas minoritas & polanya paling susah ditangkep — coba `class_weight='balanced'` (atau teknik sejenis) pas training, jangan cuma didiagnosis pas evaluasi doang

## 4. Sumber Data
- **`penaltyblog` scraper** (`pb.scrapers.FootballData`) — built-in buat narik data football-data.co.uk per musim, ga perlu nulis scraper manual
- **football-data.co.uk manual** (fallback) — pattern URL: `https://www.football-data.co.uk/mmz4281/[musim]/E0.csv`
- **API-Football (RapidAPI)** — opsional, buat fixture pertandingan yang akan datang
- **Understat.com** — opsional, data xG
- **ClubElo** (via `soccerdata.ClubElo`) — khusus buat cold-start rating tim promosi (Fase 4), gratis & udah nyakup lintas divisi

> **Standardisasi nama tim:** football-data.co.uk, FBref, dan API-Football masing-masing punya format nama tim yang beda ("Man United" vs "Manchester Utd" vs "Manchester United"). Jangan mapping manual terpisah di tiap fase — bikin `src/team_mapping.py` di Fase 1 sebagai **satu-satunya sumber kebenaran**, dipakai bareng-bareng sama Fase 10 (player stats) dan Fase 11 (corner model) biar konsisten.

## 5. Struktur Folder Project
```
football-predictive-analysis/
├── data/
│   ├── raw/                    # data mentah dari penaltyblog/football-data.co.uk
│   └── processed/              # data bersih + fitur
├── notebooks/                  # EDA, eksperimen model
├── src/
│   ├── team_mapping.py         # single source of truth: standardisasi nama tim lintas sumber data
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── statistical_models.py   # Track A: Poisson, Dixon-Coles, Elo (pakai penaltyblog)
│   ├── train_model.py          # Track B: ML classifiers
│   ├── evaluate.py             # metrik gabungan Track A & B
│   ├── value_betting.py        # bandingin peluang model vs odds bandar, hitung EV
│   ├── player_stats.py         # data key player: xG, shots on target, goals, assist (FBref)
│   ├── player_match_model.py   # prediksi match-level: SOT per pemain, kandidat MOTM
│   ├── discipline_model.py     # model Poisson buat kartu kuning/merah per tim
│   ├── corner_model.py         # model Poisson buat prediksi corner per tim
│   └── predict_match.py        # CLI interaktif: skor, W/D/L, key player, corner, BTTS
├── models/                     # model tersimpan (.pkl / .json)
├── requirements.txt
└── README.md
```

## 6. Fitur (Feature Engineering) — WAJIB rolling / point-in-time
- Form 5 pertandingan terakhir (W/D/L, goal difference) — **rolling, cutoff sebelum tanggal match**
- Head-to-head record — **rolling, bukan agregat sepanjang waktu penuh**
- Home/away performance split — **rolling per-tim sampe sebelum match date** (ini yang paling rawan leakage kalau dihitung sebagai agregat musim penuh)
- Rest days antar pertandingan
- Posisi/poin di klasemen liga — running table, bukan final table
- Elo / Pi-rating dari Track A — sebagai fitur, diambil nilainya **sebelum** match dimainkan
- xG rolling average (opsional, Fase lanjutan)
- Implied probability dari odds bandar (setelah di-decode/dihilangin overround pakai `penaltyblog`) — **HANYA buat baseline pembanding, JANGAN jadi fitur training model ML** (kalau dipakai jadi fitur, itu curang — modelnya cuma niruin odds, bukan belajar sendiri)

## 7. Metodologi Validasi (PENTING)
- **JANGAN pakai random k-fold cross-validation biasa.** Data ini time-series (pertandingan berurutan secara waktu) — random split bisa bikin model "belajar dari masa depan buat nebak masa lalu" alias data leakage temporal.
- **Yang benar: time-based split / walk-forward validation**
  - Opsi simpel: train di musim-musim awal, test di musim paling akhir
  - Opsi lebih realistis: walk-forward — train sampe matchweek N, prediksi matchweek N+1, geser terus ke depan
- Pakai `sklearn.model_selection.TimeSeriesSplit`, bukan `KFold` biasa
- **Ini berlaku juga buat Track A** (Poisson/Dixon-Coles), bukan cuma Track B: parameter attack/defense HARUS di-fit cuma pakai data training period, terus dievaluasi out-of-sample di test period. Kalau Track A di-fit pakai SEMUA data (termasuk yang harusnya jadi test set) sementara Track B dievaluasi jujur out-of-sample, perbandingan di Fase 7 jadi TIDAK ADIL — Track A bakal keliatan lebih bagus padahal cuma karena udah "liat" data test-nya duluan

## 8. Tech Stack
- **Data wrangling:** pandas, numpy
- **Model statistik (Track A):** `penaltyblog`
- **Model ML (Track B):** scikit-learn (Logistic Regression, Random Forest), XGBoost/LightGBM
- **Eksplorasi:** Jupyter Notebook
- **Visualisasi:** matplotlib, seaborn

## 9. Roadmap per Fase
| Fase | Isi | Output |
|---|---|---|
| 1. Data collection & cleaning | Tarik data 5 musim pakai `penaltyblog` scraper, gabungin, standardisasi nama tim, handle missing value | `data/processed/matches_clean.csv` |
| 2. EDA | Distribusi hasil, home advantage, tren per musim | Notebook + insight singkat |
| 3. Feature engineering | Bikin fitur di poin 6, semua rolling/point-in-time | `data/processed/features.csv` |
| 4. Model Track A (statistik) | Poisson, Dixon-Coles, Elo/Pi-rating pakai `penaltyblog` | Rating per tim + peluang per skor |
| 5. Model Track B (ML) | Logistic Regression → Random Forest → XGBoost, pakai `TimeSeriesSplit` | Model tersimpan di `models/` |
| 6. (Opsional) Stacking | Gabungin output Track A jadi fitur tambahan di Track B | Model gabungan |
| 7. Evaluasi menyeluruh | Accuracy, confusion matrix per kelas, log loss, Brier score, RPS, calibration check, dibanding odds bandar (decoded) | Report evaluasi |
| 8. (Opsional) Value betting testing | Bandingin peluang model vs odds bandar fixture yang akan datang, hitung Expected Value, ranking pertandingan | `src/value_betting.py` + tabel EV per pertandingan |
| 9. Interactive match predictor (CLI) | Pilih 2 tim manual, tentuin home/away, keluar prediksi skor + peluang W/D/L pakai model terbaik | `src/predict_match.py` |
| 10. Key player stats | Tarik data level pemain (xG, shots on target, goals, assist) pakai `soccerdata`/FBref | `src/player_stats.py` |
| 11. Model markets tambahan | Fit model Poisson buat corner (data HC/AC udah ada di football-data.co.uk), turunin BTTS dari grid skor Poisson/Dixon-Coles yang udah ada | `src/corner_model.py` + fungsi BTTS |
| 12. Integrasi ke predictor | Gabungin output Fase 9–11 jadi satu output CLI: skor, W/D/L, key player, corner, BTTS | `src/predict_match.py` (update) |
| 13. Match-level player predictions | Prediksi SOT per pemain kunci (rentang, bukan angka pasti) + kandidat MOTM proxy buat match yang dipilih | `src/player_match_model.py` |
| 14. Team discipline model | Model Poisson buat kartu kuning (& opsional merah) per tim, reuse arsitektur corner model | `src/discipline_model.py` |
| 15. Integrasi update — key player jadi match-level | Ganti tampilan key player season-stats (Fase 10) jadi prediksi match-level + tambah kartu ke `predict_match.py` | `src/predict_match.py` (update lagi) |
| 16. (Opsional) Report/dashboard | Rangkum hasil & insight | Notebook report rapi |
| 17. Audit menyeluruh (QA pass) | Cek ulang semua fase terhadap scope ini, cari gap/bug/inkonsistensi sebelum dipakai serius | Laporan audit (PASS/FAIL per item) |

## 10. Metrik Evaluasi & Sanity Check
- **Accuracy** — gampang dipahami, tapi **bisa menyesatkan**: model sering "males" nebak Draw (hasil paling jarang & polanya paling susah ditangkep), jadi WAJIB cek juga confusion matrix / recall per kelas, bukan cuma accuracy total
- **Log loss / Brier score** — buat kualitas probabilitas
- **Ranked Probability Score (RPS)** — metrik standar di literatur football prediction, karena W/D/L semi-ordinal (nebak Draw padahal aslinya Home Win itu kesalahan yang lebih kecil ketimbang nebak Away Win padahal Home Win); `penaltyblog` udah ada fungsi buat ini
- **Calibration check** — plot predicted probability vs actual frequency (misal: dari semua prediksi "55% Home Win", beneran ~55%-nya kejadian ga)
- **Baseline sanity check:** home win rate EPL historis ~45%
- **Benchmark utama:** RPS & log loss model dibandingkan ke odds bandar yang udah di-decode

## 11. Catatan Ukuran Data
10 musim (~3.800 match) masih tergolong dataset kecil buat ML kompleks kayak XGBoost dengan banyak fitur. Mitigasi: jaga jumlah fitur tetap moderate, pakai regularization / batasi kedalaman model, dan selalu evaluasi di test set (bukan cuma train set).

## 12. Value Betting — Testing Riil (Opsional, setelah Fase 7)
Kalau mau nyoba bandingin model vs pasar buat testing (bukan bet beneran), alurnya:
1. Ambil fixture EPL minggu yang akan datang
2. Jalanin model (Track A/B) → keluar peluang H/D/A tiap pertandingan
3. Ambil odds bandar buat fixture yang sama, decode jadi implied probability pakai `penaltyblog` (hilangin overround)
4. Hitung Expected Value: `EV = (peluang model × odds decimal) − 1`. EV > 0 = model liat ada "value"

**Sebelum dipakai buat testing beneran, wajib inget:**
- Valid cuma kalau model udah dikalibrasi dengan baik (cek ulang calibration check di Fase 7)
- Dataset ~3.800 match masih kecil — "value" yang keliatan gede bisa jadi overfitting/noise, bukan edge beneran
- Bandar udah sangat efisien — kalaupun ada edge, biasanya tipis dan cepet ilang
- Ini exercise data science buat belajar, bukan strategi yang dijamin cuan

## 13. Interactive Match Score Predictor (CLI)
Tools tambahan: script CLI yang bisa pilih 2 tim sendiri (bukan cuma fixture yang udah terjadwal), tentuin siapa home/away, terus keluar prediksi skor akhir.

**Catatan penting soal "prediksi skor":**
- Model klasifikasi (Track B: Logistic Regression/Random Forest/XGBoost) cuma bisa keluarin peluang W/D/L — TIDAK bisa langsung keluarin skor exact (misal 2-1), karena memang didesain buat klasifikasi kategori, bukan jumlah gol.
- Yang bisa keluarin prediksi skor adalah model Track A yang goal-based (Poisson / Dixon-Coles) — dari situ dihitung peluang tiap kombinasi skor (0-0, 1-0, 2-1, dst), lalu diambil **3 skor dengan peluang tertinggi** (bukan cuma 1) biar ga keliatan seolah-olah pasti — skor tunggal paling mungkin biasanya cuma punya peluang 15-20% aja.

**Pemilihan "model terbaik" (otomatis, bukan hardcode):**
- Script ambil hasil evaluasi dari Fase 7 (RPS & log loss tersimpan), pilih model goal-based (Poisson vs Dixon-Coles) dengan performa terbaik buat generate skor
- Peluang W/D/L tetap ditampilin dari model klasifikasi terbaik (Track B) sebagai pembanding
- Nama model yang dipakai buat tiap output HARUS disebutin jelas, biar ga rancu mana yang mana

**Alur pemakaian:**
1. Jalanin script, tampilin daftar tim EPL yang ada di dataset
2. User input Tim 1
3. User input Tim 2
4. User pilih siapa Home, siapa Away
5. Program keluarin: top-3 skor paling mungkin (dari model goal-based terbaik) + peluang W/D/L (dari model klasifikasi terbaik) + nama model yang dipakai buat masing-masing

## 14. Key Player Stats (Informasi Tambahan di Predictor)
Data level pemain buat ditampilin di `predict_match.py`, biar user liat siapa pemain kunci tiap tim pas milih match (contoh: Man United vs Man City → keliatan siapa key player masing-masing beserta xG/SOT/goals/assist-nya).

**Sumber data (baru):**
- Library **`soccerdata`** (`pip install soccerdata`) — scraper buat FBref, Understat, dll. Gunakan sesuai *usage notice*-nya: request secukupnya & manfaatin caching bawaan library, jangan spam request ke situsnya.
- `sd.FBref('ENG-Premier League', musim).read_player_season_stats(stat_type="standard")` → goals, assists
- `stat_type="shooting"` → shots, shots on target, xG

**Definisi "key player" (default):**
- Top 3 pemain per tim berdasarkan kontribusi gol (goals + assists) musim berjalan
- Tampilin: nama pemain, goals, assists, shots on target, xG musim ini

**Batasan penting (WAJIB disebutin ke user di output):**
- Ini murni informasi tambahan/konteks — **TIDAK** dipakai sebagai fitur di model prediksi skor/W-D-L (Track A/B cuma pakai data level tim, bukan level pemain)
- **TIDAK** otomatis nge-handle info cedera/suspend — kalau key player lagi cedera, itu ga kepantul di angka statistiknya, perlu dicek manual dari sumber lain
- Nama tim antar sumber data (football-data.co.uk vs FBref) bisa beda ejaan — perlu mapping/standardisasi nama tim yang sama kayak yang udah dilakuin di Fase 1

## 15. Model Markets Tambahan — Corner & BTTS
**BTTS (Both Teams to Score):**
- **TIDAK butuh model baru** — tinggal dihitung dari grid peluang skor yang udah dihasilkan model goal-based (Poisson/Dixon-Coles) di Fase 4/9: jumlahin semua peluang skor di mana home_goals ≥ 1 DAN away_goals ≥ 1

**Corner (jumlah corner per tim):**
- football-data.co.uk **udah ada kolom HC (home corners) & AC (away corners)** — ga perlu data source baru
- Fit model Poisson dengan arsitektur yang sama kayak model gol (Fase 4), tapi buat count corner: tiap tim punya parameter "corner-attack" & "corner-defense" strength
- Output: expected corner per tim + peluang over/under total corner (misal over/under 9.5)

**Catatan:** corner jauh lebih noisy/random dibanding gol (dipengaruhi taktik, defensive block, gaya main, dll), jadi akurasi model corner biasanya lebih rendah dari model gol — sampaikan ini juga di output biar user ga overconfident.

## 16. Match-Level Player Predictions (SOT & Man of the Match)
Upgrade dari Fase 10 (yang cuma nampilin stat musiman) — sekarang prediksi buat **match spesifik** yang dipilih user, bukan cuma info umum.

**Prediksi SOT (Shots on Target) per pemain:**
- Data: butuh match log per pemain (bukan season aggregate), pakai `soccerdata.FBref.read_player_match_stats(stat_type="shooting")` — scraping ini **jauh lebih berat** dari Fase 10 (per-match kali ratusan pemain), pastikan pakai caching serius
- Model: rata-rata SOT pemain dari beberapa match terakhir, disesuaikan sama kekuatan defense tim lawan (mirip logic attack/defense di Poisson, tapi level pemain)
- Output: **rentang ekspektasi** (misal "1.5–2.5 SOT"), **BUKAN angka tunggal** — variansnya tinggi banget per pemain per match, angka pasti bakal misleading

**Prediksi Man of the Match:**
- **Realistis dulu:** MOTM itu keputusan subjektif (moment individu, penyelamatan gemilang, dll) yang **statistik apapun ga bisa bener-bener prediksi**
- Pendekatan yang dipakai: proxy statistik — ranking pemain kedua tim berdasarkan **expected goal contribution** (xG + xA) buat match itu
- Output WAJIB dilabelin jelas: **"Kandidat MOTM paling mungkin (berdasarkan proyeksi kontribusi gol)"** — bukan "prediksi MOTM", biar user ga salah ekspektasi
- Ga ngecover rotasi/cedera/keputusan taktik menit terakhir — sebutin ini juga di output

## 17. Team Discipline Model — Kartu Kuning/Merah
- **TIDAK butuh data source baru** — football-data.co.uk udah ada kolom HY/AY (home/away yellow) dan HR/AR (home/away red), sama kayak corner
- Arsitektur **sama persis** kayak corner model (Fase 11): Poisson per tim dengan parameter discipline-attack strength, pakai split training/test period yang sama
- Kartu merah datanya jauh lebih jarang dari kuning, jadi modelnya bakal kurang stabil — kasih tau confidence-nya lebih rendah
- Output: expected kartu kuning per tim + peluang tim tertentu dapet kartu duluan/lebih banyak

## 18. Catatan Realistis
Model prediksi bola **jarang** bisa konsisten ngalahin odds bandar — pasar taruhan udah efisien. Tujuan project ini adalah nguasain workflow yang metodologinya BENAR (data → fitur tanpa leakage → model dengan validasi yang tepat → evaluasi yang jujur), bukan alat betting yang "pasti untung".