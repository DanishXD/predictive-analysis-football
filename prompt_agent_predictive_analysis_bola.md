# Prompt Agent — Predictive Analysis Sepak Bola v2 (buat opencode)

> Cara pakai: taro file ini bareng `scope-predictive-analysis-bola.md` di root folder project lu. Prompt dipisah per fase — paste satu-satu, tunggu review, baru lanjut. Jangan digabung semua sekaligus.

---

## PROMPT 1 — Plan Mode (paste sekali di awal)

```
Kamu adalah AI engineering assistant yang bakal bantu gua bikin project Predictive Analysis
untuk hasil pertandingan sepak bola EPL, dari nol sampe jadi model yang jalan.

Baca file scope-predictive-analysis-bola.md di root folder ini — itu adalah scope, keputusan
teknis (termasuk 2 track model: Track A statistik/penaltyblog, Track B machine learning),
dan roadmap lengkap project-nya. Jangan improvisasi keluar dari scope itu tanpa ngomong ke
gua dulu.

Tugas kamu sekarang (masih di Plan mode, JANGAN nulis kode dulu):
1. Breakdown Fase 1 (Data collection & cleaning) jadi langkah-langkah konkret, termasuk cara
   pakai library penaltyblog buat narik data football-data.co.uk
2. Sebutin file apa aja yang bakal dibikin dan isinya ngapain
3. Sebutin dependency/library Python apa aja yang dibutuhin (termasuk penaltyblog)
4. Kalau ada bagian scope yang menurut kamu ambigu atau bisa jadi masalah teknis, sebutin
   sekarang sebelum kita mulai build

Preferensi gua:
- Jelasin pake Bahasa Indonesia casual, gua masih belajar jadi jelasin juga KENAPA suatu
  keputusan diambil, bukan cuma daftar langkah doang
- Jangan overengineer, mulai dari yang paling simple dulu
```

---

## PROMPT 2 — Build Mode, Fase 1 (Data Collection & Cleaning)

```
Lanjut ke Build mode. Eksekusi Fase 1 sesuai plan yang barusan kita bikin:
- Bikin struktur folder sesuai scope-predictive-analysis-bola.md
- Bikin src/team_mapping.py dulu — dictionary/fungsi standardisasi nama tim (satu-satunya sumber
  kebenaran, bakal dipakai lagi di Fase 10 & 11, jadi desain biar reusable)
- Bikin src/data_collection.py yang pakai penaltyblog (pb.scrapers.FootballData) buat narik
  data EPL musim 2021/22 s.d. 2025/26 — kalau penaltyblog ga bisa cover semua musim, fallback
  ke download manual CSV dari football-data.co.uk (pattern URL ada di scope doc)
- Gabungin jadi satu dataframe, standardisasi nama tim (kadang beda ejaan antar musim),
  handle missing value
- Simpen hasil ke data/processed/matches_clean.csv
- Kasih summary singkat: berapa baris data, kolom apa aja, ada masalah data ga

Jangan lanjut ke EDA/fase 2 dulu sebelum gua bilang oke.
```

---

## PROMPT 3 — Build Mode, Fase 2 (EDA)

```
Lanjut Fase 2 dari roadmap. Bikin notebook EDA yang ngecek:
- Distribusi hasil (Home Win / Draw / Away Win) — berapa persen masing-masing
- Home advantage — beneran ada gap signifikan ga
- Tren per musim — ada pola yang berubah antar musim ga

Kasih insight singkat di bawah tiap chart, jangan cuma taro grafik doang.
Jangan lanjut ke feature engineering dulu sebelum gua review hasil EDA-nya.
```

---

## PROMPT 4 — Build Mode, Fase 3 (Feature Engineering)

```
Lanjut Fase 3. Bikin src/feature_engineering.py buat generate fitur ini (list lengkap ada
di scope doc bagian 6):
- Form 5 pertandingan terakhir per tim (W/D/L, goal difference)
- Head-to-head record
- Home/away performance split
- Rest days antar pertandingan
- Posisi/poin di klasemen liga

PENTING — cutoff waktu (WAJIB, ini bukan opsional):
Semua fitur di atas HARUS dihitung rolling/point-in-time — cuma pakai data SEBELUM tanggal
match yang mau diprediksi. Ini termasuk "home/away performance split" dan "head-to-head
record" yang paling gampang kelewat dihitung sebagai agregat sepanjang waktu penuh — kalau
gitu bakal data leakage (model "curi liat" hasil masa depan pas training).

Simpen hasil ke data/processed/features.csv. Kasih tau kalau ada fitur yang susah dibikin
tanpa leakage atau butuh keputusan tambahan dari gua.

Catatan tambahan: kalau ada tim promosi yang baru main di EPL (ga punya histori 5 match
terakhir/H2H di dataset kita), kasih fallback yang masuk akal (misal nilai netral/rata-rata
liga) buat beberapa match pertama tim itu, jangan biarin kosong/error.
```

---

## PROMPT 5 — Build Mode, Fase 4 (Track A: Model Statistik)

```
Lanjut Fase 4. Bikin src/statistical_models.py pakai library penaltyblog:
- Split data dulu jadi training period & test period (pakai cutoff waktu yang SAMA kayak
  yang bakal dipakai Track B di Fase 5) — JANGAN fit model pakai semua data terus asal ambil
  subset buat "test", itu bikin perbandingan Track A vs Track B di Fase 7 ga adil
- Fit Poisson Goal Model & Dixon-Coles Model **cuma pakai data training period**
- Hitung Elo rating (atau Pi-rating) per tim, update tiap pertandingan selesai — buat tim
  promosi yang ga punya histori EPL, query rating awal dari ClubElo (soccerdata.ClubElo,
  gratis) sebagai starting point; kalau ClubElo gagal/nama tim ga ketemu, fallback ke rata-rata
  Elo 3 tim terbawah klasemen musim sebelumnya

Simpen output: rating attack/defense per tim, Elo rating historis per tim (per tanggal, biar
bisa dipakai sebagai fitur point-in-time nanti), dan peluang tiap hasil (H/D/A) dari model
Poisson/Dixon-Coles buat tiap match di test period (out-of-sample, model belum pernah "liat"
match-match ini pas fitting).

Kasih rangkuman: tim mana yang attack/defense-nya paling kuat menurut model, dan Dixon-Coles
improvement-nya seberapa besar dibanding Poisson biasa (bandingin log-likelihood/AIC).
```

---

## PROMPT 6 — Build Mode, Fase 5 (Track B: Model Machine Learning)

```
Lanjut Fase 5. Bikin src/train_model.py:
- Gabungin fitur dari Fase 3 dengan Elo rating dari Fase 4 (sebagai fitur tambahan)
- Split data pakai TimeSeriesSplit dari sklearn — JANGAN pakai KFold/random split biasa,
  karena ini data time-series dan random split bakal bikin data leakage temporal
- Train bertahap: mulai Logistic Regression, lanjut Random Forest, baru XGBoost
- Coba pakai class_weight='balanced' (atau setara) di tiap model, karena Draw itu kelas
  minoritas yang biasanya diabaikan model — jangan cuma didiagnosis pas evaluasi doang
- Bandingin performa tiap model di test set (bukan cuma train set)

Simpen model terbaik ke models/. Kasih rangkuman model mana yang paling bagus dan kenapa.
```

---

## PROMPT 7 — Build Mode, Fase 6 (Opsional: Stacking)

```
Lanjut Fase 6 (opsional, kerjain kalau Fase 5 udah oke). Coba tambahin expected goals dari
Poisson/Dixon-Coles (Fase 4) sebagai fitur tambahan ke model ML terbaik dari Fase 5. Cek
apakah performanya naik dibanding tanpa fitur itu. Kalau ga naik signifikan, ga apa-apa,
laporin aja hasilnya jujur — ga semua eksperimen harus berhasil.
```

---

## PROMPT 8 — Build Mode, Fase 7 (Evaluasi Menyeluruh)

```
Lanjut Fase 7. Evaluasi SEMUA model (Track A dan Track B) pakai:
1. Accuracy — tapi JANGAN cuma ini, karena bisa nutupin kalau model gagal total nebak Draw
2. Confusion matrix + precision/recall per kelas (Home Win / Draw / Away Win) — ini yang
   nunjukkin kalau model "males" nebak Draw
3. Log loss dan Brier score
4. Ranked Probability Score (RPS) — pakai fungsi dari penaltyblog kalau ada, karena ini
   metrik standar buat football prediction (nangkep sifat semi-ordinal dari W/D/L)
5. Calibration check — plot predicted probability vs actual frequency
6. Decode odds bandar (hilangin overround, pakai penaltyblog) buat dapet peluang asli, terus
   bandingin RPS & log loss semua model ke odds itu sebagai benchmark utama

Kasih rangkuman akhir: model mana yang paling bagus secara keseluruhan, apakah ada yang
ngalahin benchmark odds bandar (biasanya belum, dan itu wajar), dan model mana yang paling
buruk nebak Draw spesifik-nya.
```

---

## PROMPT 9 — Build Mode, Fase 8 (Opsional: Value Betting Testing)

```
Lanjut Fase 8 (opsional). Bikin src/value_betting.py:
- Load model terbaik dari Fase 5/6 (boleh juga load Track A dari Fase 4 buat dibandingin)
- Terima input fixture EPL yang akan datang beserta odds bandar-nya (buat awal, boleh manual
  lewat CSV/dict dulu — ga perlu langsung integrasi API real-time)
- Decode odds bandar jadi implied probability pakai penaltyblog (hilangin overround)
- Jalanin model buat keluarin peluang H/D/A tiap fixture
- Hitung Expected Value: EV = (peluang model x odds decimal) - 1, buat tiap kemungkinan hasil
- Output tabel: fixture, peluang model, peluang pasar, odds, EV — urutin dari EV tertinggi

PENTING: ini exercise buat belajar bandingin model vs pasar, BUKAN alat rekomendasi bet.
Kasih disclaimer itu juga di output/README, dan ingetin kalau dataset kita masih kecil jadi
EV yang keliatan tinggi bisa jadi cuma noise, bukan edge beneran.
```

---

## PROMPT 10 — Build Mode, Fase 9 (Interactive Match Score Predictor CLI)

```
Lanjut Fase 9. Bikin src/predict_match.py sebagai script CLI interaktif.

1. Baca hasil evaluasi dari Fase 7 (RPS & log loss tiap model) buat nentuin OTOMATIS
   (jangan di-hardcode manual):
   - Model goal-based terbaik antara Poisson vs Dixon-Coles → dipakai buat prediksi skor
   - Model klasifikasi terbaik antara Logistic Regression/Random Forest/XGBoost → dipakai
     buat peluang W/D/L

2. Pas dijalanin, program harus:
   - Tampilin daftar tim EPL yang tersedia di dataset
   - Minta user input/pilih Tim 1
   - Minta user input/pilih Tim 2
   - Minta user pilih siapa yang Home, siapa yang Away
   - Validasi input — kalau nama tim salah ketik/ga ada di dataset, kasih tau & minta input
     ulang, jangan langsung crash

3. Setelah input lengkap, program cetak ke terminal dengan format rapi (bukan dataframe
   mentah):
   - Top-3 skor akhir paling mungkin beserta persentase peluangnya (misal "2-1 (18%)"),
     dari model goal-based terbaik
   - Peluang Home Win / Draw / Away Win, dari model klasifikasi terbaik
   - SEBUTIN JELAS nama model yang dipakai buat masing-masing bagian output (misal:
     "Prediksi skor: Dixon-Coles (RPS Fase 7: X)" dan "Peluang W/D/L: Random Forest
     (RPS Fase 7: Y)") — user harus selalu tau model apa yang lagi kepake

4. Kalau salah satu tim datanya kurang (misal tim promosi yang baru semusim main di
   dataset), kasih peringatan soal keterbatasan itu di output, jangan diem-diem aja.
```

---

## PROMPT 11 — Build Mode, Fase 10 (Key Player Stats)

```
Lanjut Fase 10. Bikin src/player_stats.py:
- Install & pakai library soccerdata (pip install soccerdata)
- Pakai sd.FBref('ENG-Premier League', musim) buat narik:
  - read_player_season_stats(stat_type="standard") → goals, assists
  - read_player_season_stats(stat_type="shooting") → shots, shots on target, xG
- Gabungin data ini per tim, pakai src/team_mapping.py (dari Fase 1) buat standardisasi nama
  tim biar cocok sama data football-data.co.uk — JANGAN bikin mapping baru yang terpisah
- Bikin fungsi get_key_players(nama_tim) yang return top-3 pemain berdasarkan (goals + assists)
  musim berjalan, beserta goals, assists, shots on target, xG masing-masing

PENTING: pakai caching bawaan soccerdata, jangan request berulang-ulang ke FBref tanpa perlu.
Kasih tau juga di README/output kalau data ini murni informasi konteks, TIDAK dipakai sebagai
fitur di model prediksi skor/W-D-L, dan TIDAK ngecover info cedera/suspend pemain.
```

---

## PROMPT 12 — Build Mode, Fase 11 (Model Markets Tambahan: Corner & BTTS)

```
Lanjut Fase 11. Bikin src/corner_model.py (pakai src/team_mapping.py dari Fase 1 buat nama tim,
konsisten sama modul lain):
- Fit model Poisson buat jumlah corner, pakai kolom HC (home corners) & AC (away corners) yang
  udah ada di data football-data.co.uk — arsitekturnya sama kayak model gol di Fase 4 (tiap tim
  punya parameter corner-attack & corner-defense strength), termasuk split training/test period
  yang sama biar konsisten
- Fungsi buat keluarin expected corner per tim + peluang over/under total corner (misal 9.5)

Terus, di src/statistical_models.py (atau file baru kecil), bikin fungsi hitung BTTS:
- BTTS TIDAK butuh model baru — tinggal jumlahin peluang dari grid skor Poisson/Dixon-Coles
  yang udah ada (Fase 4) buat semua kombinasi skor di mana home_goals >= 1 DAN away_goals >= 1

Kasih rangkuman: akurasi model corner biasanya lebih rendah dari model gol (corner lebih
random/noisy) — sebutin ini di output biar user ga overconfident sama prediksi corner.
```

---

## PROMPT 13 — Build Mode, Fase 12 (Integrasi ke Interactive Predictor)

```
Lanjut Fase 12. Update src/predict_match.py (dari Fase 9) biar pas user pilih 2 tim + home/away,
output-nya jadi lengkap satu paket:

1. Top-3 skor akhir paling mungkin + peluangnya (model goal-based terbaik, dari Fase 9)
2. Peluang Home Win / Draw / Away Win (model klasifikasi terbaik, dari Fase 9)
3. Key player tiap tim (top-3 goal contributor) beserta goals/assists/shots on target/xG,
   dari src/player_stats.py (Fase 10) — kasih label jelas ini cuma info konteks
4. Prediksi corner per tim + peluang over/under, dari src/corner_model.py (Fase 11)
5. Peluang BTTS, dihitung dari grid skor yang sama di poin 1

Setiap bagian output HARUS jelas nama model/sumber datanya masing-masing (misal: "Skor:
Dixon-Coles", "Corner: Poisson corner model", "Key player: FBref musim 2025/26"). Format
rapi di terminal, dipisah per section biar gampang dibaca, bukan satu blok teks panjang.

Di bagian paling bawah output, kasih satu baris disclaimer: ini exercise data science buat
belajar, bukan alat rekomendasi bet.
```

---

## PROMPT 14 — Build Mode, Fase 13 (Match-Level Player Predictions: SOT & MOTM)

```
Lanjut Fase 13 (upgrade dari Fase 10). Bikin src/player_match_model.py:

1. Pakai soccerdata.FBref.read_player_match_stats(stat_type="shooting") buat narik data SOT
   per pemain PER PERTANDINGAN (bukan season aggregate kayak Fase 10) — ini scraping yang
   jauh lebih berat, PASTIKAN pakai caching soccerdata biar ga request berulang-ulang ke FBref

2. Bikin fungsi predict_player_sot(nama_pemain, tim_lawan):
   - Hitung rata-rata SOT pemain itu dari beberapa match terakhir
   - Sesuaikan dengan kekuatan defense tim lawan (pakai defense strength dari Track A kalau
     ada, atau rata-rata SOT yang biasa dikasih ke lawan itu)
   - Return RENTANG ekspektasi (misal "1.5-2.5"), JANGAN angka tunggal — SOT per pemain per
     match variansnya tinggi, angka tunggal bakal misleading

3. Bikin fungsi predict_motm_candidate(tim_a, tim_b):
   - Ranking pemain kedua tim berdasarkan expected goal contribution (xG + xA) buat match ini
   - Return top-2/3 kandidat per tim
   - LABEL OUTPUT-nya harus jelas: "Kandidat MOTM paling mungkin (berdasarkan proyeksi
     kontribusi gol)" — BUKAN "prediksi MOTM", karena MOTM itu keputusan subjektif yang
     statistik ga bisa bener-bener prediksi (moment magis, penyelamatan, dll ga kecover)

Kasih tau juga keterbatasannya di output: prediksi ini ga ngecover rotasi/cedera/keputusan
taktik menit terakhir.
```

---

## PROMPT 15 — Build Mode, Fase 14 (Team Discipline Model: Kartu Kuning/Merah)

```
Lanjut Fase 14. Bikin src/discipline_model.py (pakai src/team_mapping.py buat nama tim, tetap
konsisten):
- Fit model Poisson buat kartu kuning, pakai kolom HY (home yellow) & AY (away yellow) yang
  udah ada di data football-data.co.uk — arsitektur SAMA PERSIS kayak corner model (Fase 11),
  tiap tim punya parameter discipline-attack strength, pakai split training/test period yang
  sama biar konsisten
- (Opsional) lakuin hal yang sama buat kartu merah pakai kolom HR/AR, meski datanya jauh lebih
  jarang jadi modelnya bakal kurang stabil — kasih tau kalau confidence-nya rendah di output

Fungsi keluarin: expected kartu kuning per tim + peluang tim tertentu dapet kartu duluan atau
total kartu di atas/bawah angka tertentu (misal over/under 4.5 kartu kuning).
```

---

## PROMPT 16 — Build Mode, Fase 15 (Integrasi Update: Key Player jadi Match-Level)

```
Lanjut Fase 15. Update src/predict_match.py (dari Fase 12) buat GANTI bagian key player yang
lama (season stats doang dari Fase 10) jadi prediksi match-level yang baru:

1. Ganti tampilan key player: dari "top-3 goal contributor musim ini" jadi prediksi match-level
   dari src/player_match_model.py (Fase 13) — rentang ekspektasi SOT per pemain kunci +
   kandidat MOTM
2. Tambahin prediksi kartu kuning per tim dari src/discipline_model.py (Fase 14)
3. Tetep pertahanin: skor, W/D/L, corner, BTTS (dari Fase 12) — cuma bagian key player yang
   diupgrade, jangan diulang dari nol

Pastiin tiap bagian output tetep dilabelin jelas sumbernya, dan disclaimer di bagian bawah
ga berubah (exercise data science, bukan alat rekomendasi bet).
```

---

## PROMPT 17 — Full Project Audit (Evaluasi Semua Fase, QA Pass)

```
Kamu sekarang mode AUDIT, BUKAN build baru. Tugas kamu: cek ulang SEMUA fase yang udah
dikerjain terhadap scope-predictive-analysis-bola.md, dan laporin gap/bug yang ketemu.
JANGAN langsung fix apapun dulu — laporin semuanya dulu, biar direview bareng gua sebelum
ada perubahan kode.

Baca ulang scope-predictive-analysis-bola.md dari awal sampai akhir sebelum mulai audit.

Cek satu-satu poin ini, kasih status PASS / FAIL / TIDAK YAKIN buat masing-masing beserta
penjelasan singkat (dan tunjukin di file/baris kode mana kamu ngeceknya):

FASE 1 (Data Collection)
- src/team_mapping.py ada dan isinya lengkap (bukan placeholder kosong)
- Data cover 5 musim (2021/22-2025/26), bukan kurang
- Missing value / nama tim beda ejaan udah ke-handle

FASE 3 (Feature Engineering) — PALING RAWAN LEAKAGE
- Form 5 match terakhir dihitung ROLLING (cutoff sebelum tanggal match), bukan agregat
- Head-to-head record juga rolling, bukan agregat sepanjang waktu
- Home/away performance split ROLLING per match, BUKAN agregat musim penuh — paling sering
  kelewat
- Posisi klasemen pakai running table, bukan final table musim itu
- Tim promosi yang ga punya histori form/H2H dikasih fallback (bukan error/kosong)

FASE 4 (Track A: Poisson/Dixon-Coles/Elo)
- Parameter attack/defense di-fit CUMA pakai training period — cek apa kodenya fit pakai
  seluruh dataset dulu baru "potong" test set (SALAH), atau split dulu baru fit (BENAR)
- Elo rating tim promosi pakai starting rating dari ClubElo (soccerdata.ClubElo), ada
  fallback kalau ClubElo gagal
- Cutoff training/test period SAMA PERSIS kayak yang dipakai Track B di Fase 5

FASE 5 (Track B: ML Classifiers)
- Pakai TimeSeriesSplit, BUKAN KFold/random split biasa — cek baris kodenya langsung
- Ada percobaan class_weight='balanced' (atau setara), bukan cuma default
- Evaluasi di test set yang out-of-sample, bukan train set

FASE 7 (Evaluasi)
- Ada confusion matrix / precision-recall PER KELAS, bukan cuma accuracy total
- Ada log loss DAN Brier score
- Ada RPS (Ranked Probability Score)
- Ada calibration check (predicted probability vs actual frequency)
- Ada perbandingan ke odds bandar yang udah di-decode (overround dihilangin)

FASE 9, 12, 15 (Interactive Predictor + integrasi)
- Pemilihan "model terbaik" OTOMATIS berdasarkan hasil Fase 7, BUKAN hardcode nama model
  tertentu di kode
- Skor ditampilin TOP-3 kemungkinan, bukan cuma 1 angka tunggal
- Tiap bagian output (skor, W/D/L, key player, corner, BTTS, kartu) nyebutin jelas nama
  model/sumber yang dipakai
- Validasi input tim (salah ketik ga langsung crash)
- Ada disclaimer "bukan alat rekomendasi bet" di output

FASE 11 (Corner & BTTS)
- BTTS dihitung dari grid skor Poisson/Dixon-Coles yang udah ada (bukan model baru terpisah)
- Corner model pakai split training/test yang sama kayak Fase 4/5

FASE 13 & 14 (Match-level player prediction & Discipline model)
- SOT ditampilin sebagai RENTANG (misal "1.5-2.5"), bukan angka tunggal pasti
- Output MOTM dilabelin "kandidat paling mungkin (proyeksi kontribusi gol)", BUKAN "prediksi
  MOTM" — cek teks label-nya persis
- Ada disclaimer soal keterbatasan (ga ngecover cedera/rotasi)
- Cek apa data shooting/xG dari FBref masih kereturn beneran (ada kemungkinan kepengaruh isu
  ketersediaan data terbaru di FBref) — kalau kosong/error, LAPORIN, jangan diem-diem dianggap
  "berhasil"
- Kartu kuning model reuse arsitektur & split yang sama kayak corner

CROSS-CUTTING (semua fase)
- src/team_mapping.py dipakai KONSISTEN di semua file (Fase 1, 10/13, 11/14) — bukan ada
  mapping nama tim yang beda-beda di file berbeda
- Ga ada satupun tempat yang pakai odds bandar sebagai FITUR training model ML (odds cuma
  boleh jadi baseline pembanding)
- requirements.txt: semua library yang dipakai (penaltyblog, soccerdata, xgboost, dst)
  tercantum dan versinya kompatibel

Setelah semua dicek, kasih rangkuman akhir: berapa item PASS, berapa FAIL, berapa TIDAK YAKIN,
dan urutin item yang FAIL dari yang paling kritis (misal: data leakage) ke yang paling minor.
JANGAN mulai fix apapun sampai gua bilang oke item mana yang mau diperbaiki duluan.
```

---

## PROMPT 18 — Resolve Audit Findings (Follow-up Fase 17)

```
Kamu sekarang mode FIX, berdasarkan hasil audit di report.md. Perbaiki SATU ITEM DULU,
tunjukin ringkasan perubahan & alasannya, TUNGGU konfirmasi gua sebelum lanjut ke item
berikutnya. Urutan prioritas (dari yang paling kritis, sesuai ranking di report.md):

### FIX 1 (paling prioritas): class_weight/class imbalance — Fase 5
- Tambahin class_weight='balanced' di Logistic Regression dan Random Forest
- Buat XGBoost (ga ada param class_weight langsung), hitung sample_weight pakai
  sklearn.utils.class_weight.compute_sample_weight('balanced', y_train) dan masukin
  ke .fit(sample_weight=...)
- Re-train (Fase 5), lalu re-run evaluate.py (Fase 7) biar evaluation_summary.csv
  ke-update — bandingin draw_recall SEBELUM vs SESUDAH, laporin apa ada perbaikan

### FIX 2: install soccerdata + update requirements.txt
- pip install soccerdata di .venv, tambahin ke requirements.txt
- Cek ulang apa FBref masih return 403 — kalau iya, JANGAN retry agresif/berulang
  (situs ini emang lagi ketat soal automated access), cukup konfirmasi fallback
  "Data pemain tidak tersedia" di predict_match.py masih jalan dengan benar dan
  keterbatasan ini dilaporkan jelas ke user, bukan gagal diam-diam
- Kalau ternyata FBref BISA diakses (403 cuma sementara), lanjut test
  player_stats.py & player_match_model.py normal

### FIX 3: ClubElo cold-start buat Elo — Fase 4 (butuh soccerdata dari FIX 2 dulu)
- Di statistical_models.py, pas ada tim promosi tanpa histori EPL, query rating
  awal dari soccerdata.ClubElo (bukan generic default kayak sekarang)
- Fallback ke rata-rata Elo 3 tim terbawah klasemen musim sebelumnya kalau ClubElo
  gagal/nama tim ga ketemu — sesuai yang udah didokumentasiin di scope
- Re-run Fase 4, terus Fase 5 (karena Elo dipakai jadi fitur Track B), terus Fase 7
  buat re-evaluate

### FIX 4: konsolidasi team name mapping
- Ganti FBREF_TO_CANONICAL yang terpisah di player_stats.py — pakai
  src/team_mapping.py aja, jangan bikin sumber kebenaran kedua
- Tambahin import eksplisit src/team_mapping.py di corner_model.py juga (meski
  sekarang "kebetulan" jalan karena udah pakai nama yang ke-mapping, tetap harus
  eksplisit biar konsisten sama scope §4)

### Dua hal TAMBAHAN yang JANGAN langsung diubah — tanya dulu ke gua:
- H2H window sekarang cuma last-5 match (deque maxlen=5) — ini PASS di audit,
  tapi konfirmasi apa ini emang yang dimaksud, atau harusnya window lebih panjang
- MOTM proxy sekarang ranking pakai goals+assists, padahal scope bilang xG+xA —
  konfirmasi mana yang mau dipakai sebelum diubah

Sekali lagi: SATU FIX DULU, laporin hasilnya (termasuk apa perlu re-train/re-run fase
lain), BARU lanjut ke fix berikutnya kalau gua bilang oke.
```

---

## PROMPT 19 — Verifikasi Klaim Audit Fix (Follow-up)

```
Kamu sekarang mode VERIFIKASI, bukan fix baru. Cek 5 hal ini dari AUDIT_COMPLETE.md,
laporkan temuannya SEBELUM ada perubahan kode apapun:

1. SOCCERDATA + SELENIUM (paling penting, ini keputusan yang HARUS gua yang putusin,
   bukan kamu jalan sendiri):
   - Konfirmasi ulang: apa benar soccerdata pakai selenium buat "bypass" HTTP 403 dari
     FBref? Kalau iya, jelasin persis mekanismenya (browser automation buka fbref.com
     kayak user beneran, bukan direct HTTP request?)
   - JANGAN diubah apapun dulu. Cukup laporin detail teknisnya, biar gua yang putusin
     apa mau lanjut pakai cara ini atau ganti jadi fallback graceful aja (data pemain
     "tidak tersedia") tanpa selenium

2. METRIK RPS & LOG LOSS sebelum/sesudah FIX 1 (class_weight):
   - Laporan FIX 1 cuma nunjukkin draw_recall & accuracy, TIDAK ada RPS/log loss
   - Kasih tabel RPS dan log loss SEBELUM vs SESUDAH class_weight diterapkan, buat
     ketiga model (LogReg, RF, XGBoost)
   - Accuracy turun dari ~0.48-0.50 ke ~0.42-0.45 — apa RPS/log loss juga turun, atau
     malah membaik? Ini yang nentuin apa perubahannya net positif atau ga

3. Konfirmasi RF sebagai "model terbaik":
   - RF punya draw_recall paling rendah (0.106) dibanding LogReg (0.260) dan XGBoost
     (0.212) — tapi kalau RF yang otomatis kepilih di predict_match.py, itu artinya
     RPS-nya emang paling rendah/bagus?
   - Tunjukin angka RPS ketiga model biar kelihatan jelas kenapa RF yang menang

4. Cek model FINAL yang tersimpan di models/ punya DUA-DUANYA:
   - class_weight='balanced' (dari FIX 1) DAN
   - fitur Elo yang udah pakai ClubElo cold-start (dari FIX 3)
   - Ini penting karena dua fix itu sama-sama re-train Fase 5 — pastiin re-train
     terakhir menggabungkan keduanya, bukan salah satu ke-timpa fix yang lain

5. Cek import team_mapping di corner_model.py:
   - Laporan bilang ini "documentation" doang — apa impor itu BENERAN dipakai
     manggil fungsi/variabel dari team_mapping, atau cuma diimpor tapi ga dipanggil
     (dead import)?
   - Kalau dead import, tetap OK secara fungsional, tapi sebutin jelas biar ga
     dianggap "sudah terintegrasi" padahal cuma kosmetik

Laporin kelima hal ini dulu. Untuk poin 1 (selenium), JANGAN ambil keputusan sendiri
soal lanjut/tidaknya — itu keputusan gua. Untuk poin 2-5, laporin datanya biar gua
review bareng.
```

---

## PROMPT 20 — Revert FIX 1 (Class Weight) & Re-verify Cascade

```
Berdasarkan Verifikasi #2, FIX 1 (class_weight='balanced') terbukti NET NEGATIVE
diukur pakai RPS (metrik utama scope §10) — RPS memburuk di ketiga model meski
draw_recall membaik. Keputusan: REVERT FIX 1.

1. Hapus class_weight='balanced' dari Logistic Regression dan Random Forest, dan
   hapus sample_weight dari .fit() XGBoost — kembalikan ke kondisi default sebelum
   FIX 1
2. Re-train (Fase 5), lalu re-run evaluate.py (Fase 7) buat regenerate
   evaluation_summary.csv dan model tersimpan
3. Konfirmasi ulang model terbaik yang otomatis kepilih (harusnya tetap Random
   Forest, karena RF udah RPS terendah bahkan SEBELUM class_weight — cek ulang
   biar ga berubah tanpa disadari)
4. PENTING — pastiin model final yang tersimpan SEKARANG:
   - TIDAK ada class_weight='balanced' lagi (FIX 1 di-revert)
   - TETAP ada fitur Elo dari ClubElo cold-start (FIX 3 harus tetap utuh, jangan
     ikut ke-revert gara-gara re-train ulang)
5. Dokumentasiin di scope-predictive-analysis-bola.md (bagian catatan, bukan
   bagian keputusan utama) bahwa class_weight='balanced' sudah dicoba dan
   terbukti memperburuk RPS/log loss meski draw_recall membaik — jadi ga usah
   dicoba ulang di masa depan tanpa strategi berbeda (misal post-hoc calibration
   kayak Platt/isotonic scaling, yang menyasar kualitas probabilitas tanpa
   mengubah training langsung)
6. Buat Verifikasi #5 (dead import team_mapping di corner_model.py) — cukup
   tambahin komentar singkat di baris importnya kayak
   "# Imported for documentation — data already uses canonical names", ga perlu
   dihapus ataupun dipaksa dipakai fungsional

Laporin hasil akhirnya: konfirmasi RPS/log loss balik ke angka BEFORE yang udah
dilaporkan di Verifikasi #2, dan model final mengandung FIX 3 tapi TIDAK FIX 1.
```

---

## PROMPT 21 — Extend Training Data ke 10 Musim & Re-run Cascade

```
Ubah rentang musim training dari 5 musim (2021/22-2025/26) jadi 10 musim
(2016/17-2025/26) di src/data_collection.py. Season terbaru (2025/26) tetap jadi
test period (TEST_SEASON), sisanya (2016/17-2024/25) jadi training period — total
9 musim training + 1 musim test.

Setelah itu, jalanin FULL CASCADE dari awal (urutan ini WAJIB, jangan diloncat):
1. python src/data_collection.py       — narik data 10 musim
2. python src/feature_engineering.py   — regenerate fitur (form, H2H, dst)
3. python src/statistical_models.py    — regenerate Track A (Poisson/Dixon-Coles/Elo)
4. python src/corner_model.py          — regenerate model corner
5. python src/discipline_model.py      — regenerate model kartu
6. python src/train_model.py           — re-train Track B (TANPA class_weight,
   ini udah di-revert, jangan ditambahin balik)
7. python src/evaluate.py              — regenerate evaluation_summary.csv

Hal yang perlu diperhatikan:
- Cek apa ada nama tim yang berubah/rebrand dalam rentang 10 musim ini (jarang,
  tapi cek), pastiin ke-handle di src/team_mapping.py
- Bakal ada LEBIH BANYAK tim promosi/relegasi yang keluar-masuk selama 10 musim
  ini dibanding 5 musim kemarin — pastiin cold-start ClubElo (FIX 3) jalan
  konsisten buat semua kasus, bukan cuma yang kemarin udah ketest
- Musim 2016/17-2018/19 itu SEBELUM VAR diperkenalkan (VAR masuk EPL musim
  2019/20) — ini bikin data training mencampur 2 era yang beda karakteristik
  (misal jumlah penalti/kartu bisa beda). Jangan diperbaiki dulu, cukup sebutin
  ini sebagai catatan/limitation baru di scope-predictive-analysis-bola.md,
  bagian "Rentang data" atau "Catatan Ukuran Data"
- Proses scraping & training bakal lebih lama dari sebelumnya (data ~2x lipat) —
  ini normal, bukan berarti nge-hang

Setelah selesai, laporin: total jumlah pertandingan sekarang, RPS/log loss model
terbaik dibandingkan angka 5-musim yang lama (evaluation_summary.csv sebelumnya),
dan apa ada tim yang datanya bermasalah selama proses narik 10 musim ini.
```

---

## PROMPT 22 — Klarifikasi Stacking & Perbaiki ClubElo (Lightweight, Tanpa Selenium)

```
Dua hal yang perlu diklarifikasi/diperbaiki dari laporan cascade 10-musim kemarin.
Kerjain SATU DULU, laporin, tunggu konfirmasi, baru lanjut.

### BAGIAN 1 — Klarifikasi: kenapa "random_forest_stacked_xg" jadi model terpilih?

Laporan bilang model terbaik yang otomatis kepilih itu versi STACKED (Fase 6),
padahal PROMPT 21 kemarin cuma minta jalanin `python src/train_model.py` biasa
(tanpa flag --stacking yang didokumentasiin di README).

Jelasin: apa train_model.py SELALU nge-train & masukin varian stacking ke pool
kandidat model (tanpa perlu flag), atau kemarin ternyata --stacking ke-trigger
entah gimana? JANGAN ubah kode dulu — cuma jelasin behaviour aktualnya biar gua
paham gimana cara kerja seleksi model sekarang.

### BAGIAN 2 — PENTING: ClubElo 0/27 berhasil, kemungkinan besar akibat revert FIX 2

Laporan bilang SEMUA 27 tim promosi gagal dapet rating ClubElo (fallback ke
rata-rata bottom-3 tiap kalinya). Ini kemungkinan besar KARENA kita uninstall
`soccerdata` secara penuh pas revert FIX 2 (buat ngilangin Selenium/FBref) — dan
ClubElo (FIX 3) ternyata JUGA make library yang sama (`soccerdata.ClubElo`),
jadi ikut mati walau ClubElo sendiri ga ada masalah etis apapun.

Konfirmasi dugaan ini dulu: cek apa pemanggilan soccerdata.ClubElo() di
statistical_models.py sekarang gagal karena ModuleNotFoundError (soccerdata ga
ke-install), bukan karena API/data-nya beneran ga ketemu.

Kalau benar itu penyebabnya — PENTING: ClubElo API itu cuma endpoint CSV publik
sederhana (http://api.clubelo.com/{nama_tim} atau /{tanggal}), TIDAK butuh
Selenium/browser automation sama sekali (beda total sama kasus FBref yang
emang diproteksi anti-bot). Jadi solusinya:

1. Bikin fungsi kecil standalone pakai `requests` biasa buat query ClubElo API
   langsung (tanpa install ulang soccerdata/selenium) — ga perlu package
   tambahan yang berat, cukup requests + csv parsing bawaan Python
2. Ganti pemanggilan soccerdata.ClubElo() di statistical_models.py ke fungsi
   baru ini
3. Fallback ke rata-rata Elo bottom-3 TETAP dipertahankan buat kasus tim yang
   beneran ga ketemu di ClubElo (bukan dihapus)
4. Re-run Fase 4 (statistical_models.py), lalu cascade downstream: Fase 5
   (train_model.py) dan Fase 7 (evaluate.py), karena fitur Elo bakal berubah
5. Verifikasi: dari 27 tim promosi, berapa yang SEKARANG berhasil dapet
   ClubElo asli (harusnya > 0, bukan 0/27 lagi)

Laporin hasil akhirnya: apa dugaan akar masalahnya kekonfirmasi, berapa banyak
tim yang sekarang berhasil pakai ClubElo asli vs fallback, dan RPS/log loss
model terbaru dibandingkan laporan 10-musim kemarin.
```

---

## PROMPT 23 — Bersihin Stacking Artifact Lama & Verifikasi best_ml_model.pkl

```
Sebelum lanjut ke Bagian 2 (ClubElo), ada 2 hal yang perlu dibenerin dari temuan
Bagian 1 kemarin:

### 1. Verifikasi LANGSUNG isi best_ml_model.pkl (jangan cuma infer dari timestamp)

Laporan 10-musim kemarin sempat bilang "Winner: random_forest_stacked_xg ...
This is the model saved as models/best_ml_model.pkl" — tapi laporan klarifikasi
barusan bilang sebaliknya (pure RF). Ini kontradiksi yang perlu diselesaikan
dengan bukti konkret, bukan cuma timestamp file:
- Load models/best_ml_model.pkl dan models/best_ml_model_metadata.json langsung
- Print field model_name/model_type-nya
- Konfirmasi eksplisit: apa isinya PURE Random Forest atau ada komponen stacking
  di dalamnya? Kasih tau persis field mana yang nunjukkin ini.

### 2. Bersihin artifact stacking yang lama (Opsi C dari laporan kemarin)

- Hapus models/stacked_random_forest.pkl (yang timestamp-nya 28 Juli, dari data
  5-musim lama)
- Hapus data/processed/stacking_*.csv yang berkaitan
- Re-run python src/evaluate.py biar evaluation_summary.csv cuma isinya model
  yang beneran konsisten dari training period 10-musim yang sama

### 3. Tambahin safeguard biar ini ga kejadian lagi di masa depan

Di evaluate.py, tambahin pengecekan sederhana sebelum masukin suatu model ke
perbandingan: cek training period/jumlah training match dari tiap model
(simpen sebagai metadata pas training) — kalau ada model yang metadata training
period-nya BEDA dari model lain yang lagi dibandingkan, kasih WARNING di
output evaluate.py, jangan diem-diem dibandingin kayak kemarin.

Laporin: konfirmasi isi asli best_ml_model.pkl, hasil evaluation_summary.csv
yang baru (harusnya cuma ada model 10-musim), dan apa safeguard di poin 3
berhasil ditambahin.

Setelah ini selesai, baru lanjut ke Bagian 2 (ClubElo fix) yang udah dikasih di
PROMPT 22.
```

---

## PROMPT 24 — Implementasi Safeguard Level 2 (Toleran, Tanpa Paksa Retrain)

```
Level 1 (graceful skip kalau file hilang) udah bagus, tapi itu BUKAN safeguard yang
gua minta — itu ga nyegah kasus model basi yang FILE-nya masih ada tapi training
period-nya beda (persis bug kemarin). Implementasiin Level 2 SEKARANG, tapi didesain
toleran biar ga perlu maksa full retrain lagi hari ini:

1. Di train_model.py, statistical_models.py, corner_model.py, discipline_model.py —
   tambahin penyimpanan metadata training (training_matches, training_seasons,
   timestamp) di file terpisah kayak yang udah didesain di laporan kemarin (poin
   "Safeguard RECOMMENDED - Design proposal")

2. Di evaluate.py, tambahin fungsi check_training_consistency() SEBELUM loop
   evaluasi utama:
   - Kalau metadata file BELUM ADA buat suatu model (karena belum pernah di-retrain
     pakai kode baru ini) → cukup print "metadata tidak tersedia untuk <model>,
     skip pengecekan konsistensi" — JANGAN crash, JANGAN treat sebagai error
   - Kalau metadata ADA buat semua model yang dibandingkan, DAN training
     period/jumlah match-nya beda-beda → print WARNING jelas kayak yang udah
     dirancang kemarin ("Models trained on DIFFERENT data periods")
   - Kalau semua ada dan konsisten → lanjut normal tanpa warning

3. JANGAN retrain ulang apapun sekarang. Kode ini akan "nyala" otomatis mulai
   training berikutnya (kapanpun itu terjadi — entah PROMPT 21 diulang, entah ada
   perubahan lain di masa depan), tanpa perlu kerjaan tambahan sekarang.

Laporin: konfirmasi fungsi check_training_consistency() udah ada, dan test dengan
kondisi SEKARANG (metadata belum ada) — pastiin evaluate.py tetap jalan normal
tanpa crash walau metadata belum tersedia.

Setelah ini, lanjut ke BAGIAN 2 dari PROMPT 22 (ClubElo ModuleNotFoundError fix).
```

---

## Catatan buat opencode
- Selalu pastiin agent baca `scope-predictive-analysis-bola.md` dulu sebelum mulai fase baru.
- Paste satu prompt per giliran, tunggu output & review dulu, baru lanjut ke prompt berikutnya.
- Fase 4 (Track A) dan Fase 5 (Track B) bisa dituker urutannya kalau mau, keduanya independen
  satu sama lain sampai ketemu di Fase 6 (stacking).