# EPL Predictive Analysis

Project pembelajaran end-to-end untuk memprediksi hasil pertandingan English Premier League.
Scope dan keputusan teknis lengkap tersedia di `scope-predictive-analysis-bola.md`.

## Fase 1: Data Collection & Cleaning

Pipeline mengambil data EPL musim 2021/22 sampai 2025/26 melalui scraper
`penaltyblog`. Jika scraper gagal untuk suatu musim, pipeline otomatis mengunduh
CSV dari football-data.co.uk.

```powershell
python -m pip install -r requirements.txt
python src/data_collection.py
```

Output utama disimpan di `data/processed/matches_clean.csv`. Data per musim
sebelum proses cleaning disimpan di `data/raw/`.

## Fase 8: Value Betting Testing

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

### Disclaimer

Fitur ini hanya exercise data science untuk mempelajari perbedaan probabilitas
model dan pasar, **bukan alat rekomendasi taruhan**. Dataset hanya sekitar 1.900
pertandingan dan pada evaluasi Fase 7 odds pasar mengalahkan semua model pada RPS
dan log loss. EV tinggi bisa berasal dari noise, data yang sudah stale, cold-start,
atau model yang belum terkalibrasi dengan baik, bukan bukti adanya edge maupun
jaminan profit. Data historis dan model juga harus diperbarui sebelum menganalisis
fixture baru. Jangan mempertaruhkan uang berdasarkan output project ini.

## Fase 9: Interactive Match Predictor

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
saat dijalankan sebagai tanggal prediksi, tetapi tidak pernah memakai tanggal sebelum
cutoff data historis. Tim dengan histori sedikit, tim yang sudah lama tidak tampil di
EPL, cold-start model, dan kelemahan recall Draw akan diberi peringatan di output.

## Fase 10: Player Statistics Context

Fase ini mengambil statistik pemain dari FBref untuk memberikan konteks skuad.

```powershell
python -m pip install soccerdata
python src/player_stats.py --season 2526
```

Untuk melihat top-3 pemain dari tim tertentu:

```powershell
python src/player_stats.py --season 2526 --team "Arsenal"
```

**Format season:** gunakan format FBref seperti `2526` untuk musim 2025/26.

### Disclaimer

Data pemain ini **hanya konteks tambahan**, TIDAK dipakai sebagai fitur di model
prediksi skor/W-D-L. Data ini juga tidak mencakup informasi cedera atau suspensi pemain.
Script memakai caching bawaan soccerdata untuk menghindari request berulang ke FBref.

## Fase 11: Corner Model & BTTS

### Corner Model

```powershell
python src/corner_model.py
```

Fits a Poisson model for corner counts (HC/AC) with the same architecture as the goal
model: each team gets corner-attack and corner-defense parameters + home advantage.
Same train/test split as Phase 4 (2021-2025 train, 2025-26 test) with time-decay
weights. Output:

- `corner_team_strengths.csv` — corner-attack/defense per team
- `corner_test_predictions.csv` — expected corners + over/under probabilities per match

**Catatan:** Corner model menggunakan arsitektur Poisson yang sama dengan model gol,
tetapi akurasi model corner biasanya **lebih rendah** karena corner lebih noisy/random
(dipengaruhi taktik, defensive block, gaya main, dll).

### BTTS (Both Teams to Score)

BTTS **tidak butuh model baru**. Dihitung langsung dari grid probabilitas skor
Poisson/Dixon-Coles (Fase 4) dengan menjumlahkan semua probabilitas skor di mana
home_goals >= 1 AND away_goals >= 1.

Fungsi `btts_probability(grid)` di `src/statistical_models.py` mengambil
`FootballProbabilityGrid` dan mengembalikan probabilitas BTTS.

```python
from statistical_models import btts_probability, btts_probabilities_for_test
```
