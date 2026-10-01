# Web app (Streamlit) — prediksi match EPL

Antarmuka web untuk prediksi match, memakai **logika yang sama persis** dengan
CLI `src/predict_match.py`. Web bukan implementasi terpisah: ia memanggil
fungsi yang sama, jadi tidak mungkin memilih model atau menghitung angka yang
berbeda dari CLI untuk input yang sama.

## Jalankan lokal

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka `http://localhost:8501`.

## Apa yang ditampilkan

Hanya prediksi yang punya nilai nyata:

- Skor paling mungkin
- W / D / L
- BTTS
- Corner
- Kartu kuning

**Tidak ada** key player dan Man of the Match. Keduanya butuh scraping FBref
saat runtime yang hampir selalu gagal di lingkungan server, jadi halamannya
akan permanent berisi pesan "tidak tersedia". CLI tetap menyediakan keduanya
seperti biasa. Bagian ini bisa ditambahkan ke web nanti kalau ada sumber data
pemain yang legitimately bisa diakses.

Feature lain (value betting, dashboard evaluasi, dst.) tetap CLI-only.

## Snapshot `deploy/`

Web membaca folder `deploy/`, bukan `data/processed/` dan `models/`. Kedua
direktori itu ada di `.gitignore` karena isinya artefak turunan pipeline, jadi
tidak ada di repo hasil clone — sedangkan Streamlit Community Cloud menjalankan
`streamlit run app.py` dari clone itu.

`.gitignore` punya pengecualian **hanya untuk `deploy/`**. `models/` dan
`data/processed/` milik pipeline tetap di-ignore.

Kalau pipeline, `TEST_SEASON`, atau model berubah, regenerate snapshot:

```powershell
.\.venv\Scripts\python.exe src\build_deploy_snapshot.py
```

Script itu juga menulis `deploy/manifest.json` berisi sha256 per file, dan
`tests/test_predict_match_refactor.py` memverifikasinya supaya snapshot tidak
bisa diam-diam berubah di luar build script.

## Pemilihan model

Otomatis dari `cv_model_selection.csv`, tanpa nama model yang di-hardcode di
`app.py`. Test suite menjaga ini.

`evaluation_summary.csv` ikut ter-deploy sebagai artefak pelaporan, tapi
**tidak** dipakai untuk memilih model: isinya metrik test season, dan
memakainya untuk selection adalah kontaminasi test set (`AGENTS.md` aturan
keras 3).

## Deploy

1. Push repo ke GitHub.
2. Buka <https://share.streamlit.io> dan login dengan GitHub.
3. "Deploy an app" → pilih repo ini → file `app.py`.
4. Deploy selesai; link-nya dikirim ke email.

Gratis, tanpa server sendiri.

## Tes

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app_streamlit.py tests/test_predict_match_refactor.py -q
```

`test_app_streamlit.py` menjalankan `app.main()` dengan widget yang di-mock,
sehingga regresi render (exception, nilai yang hilang, bagian yang kosong)
ketahuan di test suite, bukan pas user membuka link.