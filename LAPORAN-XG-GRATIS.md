# Laporan: Opsi Data xG Gratis untuk Side Project

**Tanggal:** 29 September 2026
**Untuk:** konsultasi ke AI/lain — dokumen ini sengaja berisi temuan mentah + bukti, supaya bisa diverifikasi ulang.
**Status repo saat dokumen dibuat:** `main` == `origin/main`, 125 tes lulus, model produksi = XGBoost.

---

## 1. Ringkasan eksekutif

Permintaan: model Poisson berbasis xG, **biaya harus nol**.

Temuan utama: **tidak ada sumber xG match-level EPL 10 musim yang gratis DAN berlisensi jelas.** Tapi ada dua jalan keluar gratis yang layak, dan keduanya lebih mudah dipertanggungjawabkan daripada opsi berbayar yang tersedia.

| Opsi | Biaya | xG match-level EPL | Lisensi | Rekomendasi |
|---|---|---|---|---|
| **A. Exposure-based Poisson** (pakai `shots` yang sudah ada) | Nol | Bukan xG, tapi exposure | Sudah ada di repo | **YA** |
| **B. StatsBomb Open Data** | Nol | Ya, tapi cuma 2 musim PL | CC BY-SA 4.0 + atribusi | Terbatas, tapi sah |
| **C. Wyscout Open (Pappalardo 2019)** | Nol | Shot events, tanpa xG | CC BY 4.0 | Terbatas, 1 musim |
| D. Kaggle (dataset turunan Understat) | Nol | Ya | klaim CC0, tapi berasal dari Understat | **TOLAK** |
| E. OpenFootAPI | $14/bln | Ya | **Unlicensed-Beta (FotMob)** | **TOLAK** |
| F. TheStatsAPI | $50/bln | Ya | Tidak dinyatakan | Terlalu mahal + tidak transparan |
| G. API-Football | 100 req/hari gratis | **Tidak terverifikasi** | Tidak dinyatakan | Ditolak |

---

## 2. Konteks: apa yang sudah DITOLAK sebelumnya dan kenapa

Preseden project ini penting untuk membantu AI lain menilai konsistensi rekomendasi.

### 2.1 Understat — DITOLAK (29 Sep 2026)
```
https://understat.com/robots.txt  ->  HTTP 200
User-agent: *
Disallow: /
```
- Seluruh situs tertutup untuk semua crawler. Tidak ada Terms of Service, tidak ada API docs, tidak ada kontak lisensi.
- Hanya ada konfirmasi email 8 Nov 2018 bahwa data boleh dipakai non-komersial, dengan catatan *"This stance is subject to change"*. Itu informal, 8 tahun lalu.
- Alasan penolakan: **konsistensi**. FBref sudah ditolak karena Selenium + ToS. Kalau Understat lolos hanya karena "praktis ditegakkan", standar yang dijaga jadi tidak konsisten.

### 2.2 FBref — DITOLAK
ToS membatasi automated scraping. `soccerdata` tidak di-install default; script pemain hanya menampilkan pesan fallback.

### 2.3 Insight PENTING untuk diskusi
Hampir semua sumber xG EPL gratis di internet adalah **turunan Understat** atau **turunan FotMob**. Keduanya sudah ditolak. Inilah yang membuat opsi gratis jadi sulit, dan mengapa ada dua jalan alternatif di bawah.

---

## 3. Opsi A — Exposure-based Poisson (REKOMENDASI UTAMA)

### 3.1 Insight teknis
`football-data.co.uk` yang **sudah dipakai project** punya kolom tembakan per tim per match. Saya verifikasi di `data/processed/matches_clean.csv`:

```
null counts:
  shots_home             0 / 3800
  shots_away             0 / 3800
  shots_on_target_home   0 / 3800
  shots_on_target_away   0 / 3800

shots per team-match: mean 13.87  median 13.0  min 0  max 37
conversion rate home:  0.1122
conversion rate away:  0.1115
SOT share home:        0.3438
```

**0 null di 3.800 match.** Ini-proxy xG yang jauh lebih baik daripada yang Dibutuhkan.

### 3.2 Yang diuji
Poisson goal model yang biasa bicara `goals ~ attack + defense + HFA`. Variasinya: **ganti exposure**. Alih-alih menghitung λ dari goal historis, hitung λ dari *volume kesempatan*.

```text
Poisson biasa:     λ_home = exp(μ + attack_home − defense_away + HFA)
Exposure-based:    λ_home = Σ_shots_team × P(shot → goal)
```

`P(shot → goal)` diestimasi sebagai fungsi conversion rate tim:
- `attack_h` (re_xG) = conversion rate × shots per match
- `defense_a` (re_xG conceded) = (1 − conversion rate lawas) × shots against per match

Tiga perbedaan fundamental dari xG asli:
1. **Snapshot, bukan shot-level.** xG asli_condition setiap peluang secara individual (posisi, jenis, situasi). Di sini kita cuma punya agregat.
2. **Tidak menangkap shot location.** Tembakan dari jauh di luar kotak penalti dihitung punya peluang yang sama dengan tembakan quality tinggi di depan gawang. Ini bias yang nyata.
3. **Tidak menangkap penalti.** Tidak ada informasi mana yang penalti.

### 3.3 Ekspektasi hasil ( Honest )
**Harapan:-netral, kemungkinan besar sedikit lebih baik dari baseline.** Alasannya:
- Conversion rate tim sudah sangat informatif dan TIDAK ada di baseline sekarang (baseline cuma pakai goals).
- Volume tembakan menambah informasi yang tidak ada di goals.
- Tapi tanpa shot location, sebagian besar nilai xG (posisi = 60-70% dari$xG variance) hilang.
- Konversi attack/defense dari shot-count ke goal-rate kena noise, karena goals itu low-count (~1.4 per match) sementara shots high-count (~13.9 per match). **Ini justru keunggulan** — estimasi lebih stabil.

### 3.4 Cara evaluasi (WAJIB anti-leakage)
Mengikuti aturan keras project:
- Angka penentu dari **walk-forward CV di training seasons saja** (`src/track_a_cv.py` sudah ada, 5 fold, 2020/21 s.d. 2024/25, `TEST_SEASON` tidak pernah jadi fold).
- **Jangan** pernah menghitung sensitivitas di test set. Ini persis kesalahan yang tercatat di HANDOFF §5.3: "lambda 1.3 memperbaiki RPS" ternyata salah karena dihitung di test set.
- Laporan: paired per-fold delta log loss + RPS + bootstrap CI.

### 3.5 Risiko & keterbatasan (untuk bahan diskusi)
- **Bukan xG.** Kalau mentor/reviewer butuh "Poisson berbasis xG" dengan arti harfiah, ini tidak memenuhi. Harus dikomunikasikan apa adanya.
- **Potensi bias eklektik.** Distribusi.location yang hilang bisa membuat model under-rate tim yang shoot dari jarak jauh (yang wedang overperform xG).
- **Time-decay interaksi.** `TIME_DECAY_XI=0.0018` diPoisson biasa. Exposure-based perlu test ulang apakah xi yang sama masih cocok, karena variabel target-nya berubah dari goals (skala ~1.4) ke re_xG (skala ~1.5, lebih smooth). **Jangan asumsikan xi yang sama.**
- **Bisa jadi tidak membaik sama sekali.** Kalau begitu, catat sebagai hasil negatif (jadi hasil negatif ke-8) dan berhenti.

---

## 4. Opsi B — StatsBomb Open Data (GRATIS, LENGKAP SECARA LISENSI, TAPI SANGAT SEDIKIT)

### 4.1 Lisensi — ini yang paling baik dari semua yang diperiksa
`https://github.com/hudl/open-data` (3.7k stars, 984 forks). README:
> "we have made certain leagues of StatsBomb Data freely available for public use for research projects and genuine interest in football analytics."
> "If you publish, share or distribute any research... please state the data source as StatsBomb and use our logo, available in our Media Pack."

Lisensi: **CC BY-SA 4.0** (dikonfirmasi dataset turunan di HuggingFace) + kewajiban atribusi. Non-komersial friendly.

### 4.2 Coverage EPL — INI MASALAHNYA
Saya hitung langsung dari `data/competitions.json` di repo tersebut:

| competition_id | season_id | season_name |
|---|---|---|
| 2 | 27 | 2015/2016 |
| 2 | 44 | 2003/2004 |

**Hanya 2 musim Premier League, dan keduanya di luar window project (2016/17–2025/26).** Musim 2015/2016 bersinggungan 0%.

Dataset turunan di HuggingFace menyebut "Premier League: 10,837 shots, 2 seasons" — konsisten dengan angka ini.

### 4.3 Verdict
**Secara lisensi: terbaik. Secara cakupan: tidak berguna untuk project ini.**

Nilai gunanya: sebagai **validasi metodologi** — bisa dipakai untuk menguji apakah algoritma re_xG benar-benarENN implement dengan benar, di domain yang sama. Bukan untuk dipakai di model produksi.

---

## 5. Opsi C — Wyscout Open / Pappalardo 2019 (GRATIS, SATU MUSIM)

### 5.1 Lisensi
**CC BY 4.0** — ini lisensi permisif yang benar. Figshare: `https://figshare.com/collections/Soccer_match_event_dataset/4415000`
Citable: Pappalardo et al. (2019), *Nature Scientific Data* 6:236, DOI 10.1038/s41597-019-0247-7

### 5.2 Coverage
Season **2017/18 saja**, 380 match PL + 4 liga besar lain + WC 2018 + Euro 2016. Total ~1.941 games.

### 5.3 Apakah ada xG?
**Tidak ada field xG.** Hanya event stream: passes, shots, fouls, dengan posisi dan waktu. xG harus **dibangun sendiri** dari shot location.

### 5.4 Verdict
Relevan hanya kalau ingin membangun xG model sendiri dari shot coordinates. 380 match terlalu sedikit untuk train model xG yang generalizable, tapi cukup untuk **sanity check** bahwa pipeline re_xG benar.

---

## 6. Opsi D — Dataset Kaggle (DITOLAK)

Beberapa dataset mengklaim CC0 / Public Domain dengan xG EPL:

| Dataset | Klaim lisensi | Sumber sebenarnya |
|---|---|---|
| `cpgaryyiy/premier-league-expected-goal-data-20142020` | CC0: Public Domain | **Understat** (dikonfirmasi di deskripsi: "made available by understat... understatr is used to grab") |
| `codytipton/understat-data` | — | Understat |
| `yarknyorulmaz/understat-match-team-metrics-dataset` | Open Database | **Understat** |
| `sinansaglam/premier-league-last-5-seasons-match-scores-and-xgs` | CC0 | Tidak jelas |

### Alasan penolakan
Orang yang meng-upload **tidak memegang hak** untuk melepas lisensi. Label "CC0 Public Domain" di Kaggle hanya berarti pengupload mengklaim tidak ada batasan — sedangkan Understat punya `Disallow: /` dan tidak punya ToS publik. File-file ini mungkin masih tersedia di Kaggle, tapi itu hasil scraping pihak lain, bukan pembebasan lisensi untuk kita.

Ini persis argumen yang dipakai menolak Understat, dan alasannya masih berlaku.

**Catatan tambahan**: `evangora/premier-league-data` (Apache 2.0) menyatakan sendiri bahwa "a majority of the games are missing... expected goals" — jadi bahkan kalau lisensinya benar, cakupannya tidak lengkap.

---

## 7. Opsi E — OpenFootAPI (DITOLAK, REKOMENDASI SAYA BERUBAH)

Saya Initially merekomendasikan ini. Setelah verifikasi, **saya berubah pikiran.**

### 7.1 Endpoint (terverifikasi, read-only, demo key publik)
```
GET /v1/matches?competition=comp_premier_league_eng&season=2024/25
GET /v1/matches/{id}/xg            <- Developer plan, 403 di free tier
GET /v1/analytics/xg?league=EPL
```
Arsip 17 musim: 2010/11 s.d. 2026/27. ID canonical: `match_fmb_*` (alias lama `match_of_epl_*` masih resolve).

### 7.2 Endpoint xG tidak gratis
Saya coba dengan demo key publik mereka (`of_demo_openfootapi_docs`):
```
GET /v1/analytics/xg?league=EPL  ->  HTTP 403 Forbidden
```
Free Starter ($0) = 5.000 req/bulan tapi **hanya** competitions, search, fixtures, results, scores, standings. xG butuh **Developer $14/bulan**.

**Ini membatalkan klaim "Free (5k req/bulan)" yang tertulis di HANDOFF.md §9.3.** Sudah dikoreksi di dokumen ini.

### 7.3 TEMUAN PENENTU: lisensi hulu
Saya hitung langsung dari `GET /v1/competitions` (endpoint free, tanpa auth):

```
31x  Free-Open-Data
 3x  ODbL-1.0
 6x  Public-Factual-Data
80x  Unlicensed-Beta (owner-approved factual data)
```

Premier League termasuk bucket terakhir:
```json
{
  "id": "comp_premier_league_eng",
  "code": "EPL",
  "coverage": { "advancedStats": true },
  "source": {
    "name": "FotMob",
    "license": "Unlicensed-Beta (owner-approved factual data)",
    "url": "https://www.fotmob.com/"
  }
}
```

**`"Unlicensed-Beta (owner-approved factual data)"` itu bukan lisensi.** Itu pernyataan bahwa mereka mendapat persetujuan lisan dari pemilik. Bandingkan dengan bucket lain yang benar-benar bernama lisensi (CC0, ODbL-1.0).

### 7.4 FotMob ToS — prohibitive
`https://www.fotmob.com/tos.txt`:
> "Use of the data, content, or any information displayed on FotMob for any purpose, including but not limited to **scraping, reproduction, redistribution**, or commercial purposes, without the express **written consent** of FotMob is strictly prohibited."
> "The use of automatic services (robots, spiders, indexing, etc.), as well as other methods for systematic, regular, or bulk retrieval of data, is expressly forbidden."
> Lisensi yang diberikan: "personal, worldwide, non-assignable license to use the software we provide you for your own **personal, non-commercial** use."

Preseden yang relevan: maintainer `worldfootballR` menghapus semua fungsi fotmob setelah menerima **cease-and-desist** dari CEO FotMob (PR #311).

### 7.5 OpenFootAPI ToS §7 mengakui sendiri
> "Access to a competition through the service is **not a licence to redistribute the underlying provider's feed**... Where a source licence imposes obligations of its own... those attach to your use **directly**, independently of these terms, and we **cannot waive them on the source's behalf**."

Artinya: membayar $14 **tidak memperbaiki** masalah lisensi. Kewajiban FotDb "attach to your use directly".

### 7.6 Verdict
**DITOLAK.** Pola identik dengan Understat, hanya dibungkus API berbayar. Membayar $14 tidak mengubah analisis risiko sama sekali.

---

## 8. Opsi F — TheStatsAPI (TIDAK LAYAK UNTUK BUDGET NOL)

### 8.1 Endpoint (terverifikasi dari docs)
```
GET /football/matches/{match_id}/stats        -> overview.expected_goals.all.{home,away}
                                              -> np_expected_goals
GET /football/matches/{match_id}/shotmap     -> data[].expected_goals (per shot)
GET /football/matches/{match_id}/player-stats -> shooting.expected_goals, expected_assists
```
Field `xg_available` untuk cek availability. 10 tahun history, 150 liga default.

### 8.2 Masalah
1. **$50/bulan** — di luar budget "full gratis".
2. **Sumber data tidak dinyatakan.** Tidak ada halaman yang menyebutkan provider hulu. Bandingkan: OpenFootAPI setidaknya menaruh `source.name` di setiap response.
3. **Trial 7 hari** — trial bisa dipakai untuk uji coba tanpa biaya, tapi tidak untuk dataset 3.800 match permanen.

### 8.3 ToS
LTD, non-transferable, revocable. Prohibit: resell, bulk redistribute, "cache or store beyond what is reasonably necessary", compete with the service. OK untuk side project pribadi, tapi tidak ada jaminan soal provenance.

---

## 9. Opsi G — API-Football (DITOLAK)

### 9.1 Temuan
Dokumentasi resmi **tidak pernah menyebut xG**. Field di `/fixtures/statistics`: possession, shots on/off goal, blocked, inside/outside box, fouls, corners, offsides, cards, passes, ball possession, yellow/red. **Tidak ada expected goals.**

Blog pihak ketiga mereka sendiri mengakui:
> "expected goals, npxG, and xA are **not safe to assume** as a standard field set in every core response or every league."
> "if xG is central to your product, test the exact competition and endpoint before committing to the provider."

### 9.2 Free tier
100 req/hari. Untuk backfill 3.800 match dengan `/fixtures?ids=` (maks 20 id per call) = 190 call untuk fixtures, tapi xG butuh 1 call per match = 3.800 call = **38 hari** di free tier. Tidak realistis.

### 9.3 Verdict
**DITOLAK** — tidak ada bukti endpoint xG, dan free tier tidak cukup walau ada.

---

## 10. Ringkasan bukti yang bisa diverifikasi ulang

Semua klaim di dokumen ini bisa dicek tanpa biaya:

```powershell
# OpenFootAPI — lisensi hulu (endpoint gratis, demo key publik)
Invoke-RestMethod "https://openfootapi.com/v1/competitions" `
  -Headers @{Authorization="Bearer of_demo_openfootapi_docs"} |
  Select-Object -ExpandProperty data | Group-Object {$_.source.license} |
  ForEach-Object { "$($_.Count)x  $($_.Name)" }

# OpenFootAPI — konfirmasi xG butuh plan berbayar
Invoke-RestMethod "https://openfootapi.com/v1/analytics/xg?league=EPL" `
  -Headers @{Authorization="Bearer of_demo_openfootapi_docs"}
# -> HTTP 403

# FotMob — robots.txt
Invoke-WebRequest "https://www.fotmob.com/robots.txt"
# -> Disallow: /api/*

# StatsBomb — hanya 2 musim PL
Invoke-RestMethod "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json" |
  Where-Object {$_.competition_name -eq "Premier League"} |
  Select-Object season_id, season_name

# Understat — sudah ditolak sebelumnya
Invoke-WebRequest "https://understat.com/robots.txt"
# -> Disallow: /
```

---

## 11. Pertanyaan terbuka untuk dikonsultasikan

Kalau Anda konsultasi ke AI lain, ini pertanyaan yang menurut saya paling layak untuk ditanyakan:

1. **Apakah exposure-based Poisson (opsi A) secara metodologis sepadan dengan "Poisson berbasis xG"?** Kalau tidak, apa framing yang benar supaya tidak overclaim?
2. **Berapa nilai tambah xG asli dibanding shot-count untuk prediksi H/D/A EPL?** Apakah sepadan dengan usaha dan biaya lisensi? Literatur relevan?
3. **Adakah sumber xG EPL gratis yang saya lewatkan** - terutama yang hulu-nya berlisensi jelas (bukan Understat/FotMob)? Misalnya data resmi EPL, FA, atau inisiatif open-data akademis?
4. **Apakah StatsBomb Open Data bisa diperluas** - apakah ada cara meminta akses season PL tambahan secara gratis untuk riset?
5. **Untuk exposure-based: Xi mana yang cocok?** `TIME_DECAY_XI=0.0018` sekarang dipakai di Poisson goal-based. Apakah harus di-search ulang? Berapa range yang masuk akal?
6. **Apakah ada variasi yang lebih baik dari re_xG berbasis conversion rate?** Misalnya kalibrasi shots ke xG dengan adjustment untuk SOT share, yang tersedia di data (`shots_on_target` ada, 0 null).

---

## 12. Rekomendasi saya (ringkas)

1. **Opsi A sudah diimplementasikan dan diuji.** Lihat §13.
2. Opsi B/C tetap berguna kalau butuh validasi algoritma re_xG di domain yang sama.
3. **Kalau tetap mau xG asli**, satu-satunya jalur yang belum ditolak adalah TheStatsAPI — tapi $50/bln + provenance tidak jelas. Tidak disarankan untuk budget nol.
4. **Jangan pernah** pakai dataset Kaggle turunan Understat, dan jangan pilih OpenFootAPI hanya karena "ada API resminya".

---

## 13. Hasil implementasi Opsi A (29 September 2026)

Opsi A sudah dikerjakan di `src/exposure_poisson.py`. Ringkas:

**Struktur dua tahap.** Model offset tunggal `lambda = shots * exp(...)` hanya bisa
dihitung untuk match yang sudah selesai, jadi exposure itu sendiri harus
diprediksi lebih dulu. Karena itu dekomposisi dua tahap bukan pilihan gaya,
melainkan imposed oleh ketersediaan data:

```
sigma  = E[shots]          (stage 1, penaltyblog Poisson di-fit pada shots)
q      = P(goal | shot)    (stage 2, Poisson dengan offset log(shots))
lambda = sigma * q
```

Varian SOT menambah tahap: `q = P(SOT | shot) * P(goal | SOT)`.

**Hasil walk-forward CV** (5 fold, season validasi 2020-2021 s.d. 2024-2025,
`xi` di-grid-search untuk kedua lengan lalu best-vs-best):

```text
model                   log loss      RPS  accuracy
sot                     0.997688  0.209706  0.517368
poisson_goals_based     1.002893  0.211598  0.520000
shot_volume             1.004002  0.212677  0.508421
```

**Paired vs baseline** (negatif = lebih baik):

```text
sot           delta log loss -0.005205, CI 95% [-0.008340, -0.002153],
              SIGNIFIKAN, menang 5 dari 5 fold pada log loss DAN RPS
shot_volume   delta log loss +0.001109, CI 95% [-0.005091, +0.007150],
              tidak signifikan, menang 3 dari 5 fold
```

### Jawaban untuk sebagian pertanyaan di §11

**Soal §11.1 (apakah exposure-based sepadan dengan label "Poisson berbasis xG")?**
Tidak, dan memang tidak diklaim demikian. Nama modul, docstring, dan output
memakai "exposure-based Poisson". Alasan teknisnya ada di §2.2 dokumen ini:
xG asli butuh shot location per tembakan, sedangkan
`football-data.co.uk` hanya punya agregat.

**Soal §11.2 (berapa nilai tambah dibanding shot-count)?**
Terjawab oleh eksperimen: shot-count saja **tidak** membantu (+0.001109, tidak
signifikan). Yang membantu adalah pemisahan tiga tahap shots -> SOT -> goal.
Jadi letupannya bukan dari "tembakan" secara umum.

**Soal §11.5 (Xi mana yang cocok)?**
`xi = 0.0025` untuk ketiga model. Ini **berbeda** dari `TIME_DECAY_XI = 0.0018`
yang dipakai produksi, jadi asumsi "xi yang sama berlaku untuk semua model"
ternyata salah. Tapi grid `EXPOSURE_XI_GRID` juga memilih 0.0025 untuk baseline
goals-based, jadi selisih di atas bukan efek xi.

**Soal §11.6 (variasi lebih baik dari re_xG berbasis conversion rate)?**
YA, dan itu persis yang menang. Alih-alih satu conversion rate
`goals/shots`, pisahkan jadi dua: `SOT/shots` lalu `goals/SOT`. Data mendukung
langkah ini (`shots_on_target` tersedia, 0 null dari 3.800 baris).

### Temuan sampingan yang tak terduga

Home advantage praktis seluruhnya ada di **volume**, bukan di efisiensi. Rasio
home/away di training season:

```text
goals          1.212
shots          1.204
conversion rate 1.006    <- hampir tidak ada home edge
SOT rate        0.986    <- bahkan sedikit di bawah 1
```

Konsekuensinya parameter `home_advantage` di stage-2 berakhir di batas bawah
~0.000. Itu hasil yang benar, bukan kegagalan optimasi — sebagian besar efek
rumah sudah ditangkap stage-1.

### Yang belum dilakukan

Angka di atas masih CV di **training season** dengan bias selection pada `xi`
(`xi` dipilih dari rerata fold yang sama). `evaluate.py`, `predict_match.py`,
dan `cv_model_selection.csv` sengaja tidak disentuh. Menguji varian SOT di test
season 2025-2026 adalah keputusan terpisah, dan angka itu akan menjadi satu-
satunya angka yang benar-benar bebas dari selection bias.

---

## 14. Ringkasan untuk sesi CONSULTASI

Kalau hanya boleh membawa satu bagian ke percakapan dengan AI lain, ambil ini.

```text
KONTEKS
  Project belajar DS, prediksi EPL H/D/A, 10 musim, test season 2025-2026.
  Tidak ada sumber xG EPL gratis yang berlisensi jelas (Understat/FotMob
  ditolak; sisanya berbayar). Kolom shots + shots_on_target sudah ada
  di football-data.co.uk, 0 null dari 3.800 baris.

MODEL
  Dua tahap: sigma = E[shots], q = P(goal|shot), lambda = sigma * q.
  Varian SOT tiga tahap: q = P(SOT|shot) * P(goal|SOT).
  BUKAN xG — tidak ada shot location. Jangan sebut "xG-based".

HASIL (walk-forward CV, 5 fold, training season saja)
  sot             log loss 0.997688  RPS 0.209706  menang 5/5 fold
  baseline goals  log loss 1.002893  RPS 0.211598
  shot_volume     log loss 1.004002  RPS 0.212677  menang 3/5 fold

  sot vs baseline: -0.005205, CI 95% [-0.008340, -0.002153], SIGNIFIKAN
  shot_volume  vs baseline: +0.001109, CI 95% [-0.005091, +0.007150], tidak

TEMUAN
  1. Shot-count saja tidak membantu. Yang membantu adalah pemisahan
     shots -> SOT -> goal.
  2. Home advantage seluruhnya di volume, bukan efisiensi (rasio conv rate
     home/away hanya 1.006), jadi HFA stage-2 ~0.
  3. xi optimal 0.0025, BUKAN 0.0018 yang dipakai produksi. Tapi xi=0.0025
     juga terbaik untuk baseline, jadi selisih di atas bukan efek xi.

STATUS
  Belum produksi. evaluate.py / predict_match.py / cv_model_selection.csv
  sengaja tidak disentuh. Angka masih punya selection bias pada xi.
```

Pertanyaan yang masih terbuka untuk didiskusi: apakah pola "shot-count biasa
gagal, tapi shots->SOT->gol berhasil" masuk akal secara football analytics,
dan apakah ada cara eksposur yang lebih baik lagi (misalnya memisahkan headed
vs open play, yang tidak ada di dataset ini)?

---

*Dokumen ini dibuat sebagai bahan konsultasi. Semua klaim sudah diverifikasi langsung pada 29 September 2026. Tidak ada scraping data yang dilakukan — hanya fetch dokumentasi, robots.txt, dan endpoint metadata read-only.*
