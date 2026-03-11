# Penjelasan Lengkap: naive_bayes.py

Dokumen ini menjelaskan struktur, alur, dan fungsi utama dari skrip `naive_bayes.py` yang membangun klasifikasi Gaussian Naive Bayes untuk dataset `heart.csv`.

## Tujuan & Alur Tinggi
- Tujuan: Memprediksi `target` (0/1) dari 13 fitur numerik menggunakan Gaussian Naive Bayes.
- Alur: Preprocessing (muat CSV, split) → Pengelompokkan (training/prediksi) → Evaluasi (akurasi, confusion matrix).

## Imports
- Lihat: [naive_bayes.py:L1-L4](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L1-L4)
- `csv`: membaca dataset dari file CSV
- `math`: operasi matematika (log, konstanta π)
- `random`: pengacakan indeks untuk pembagian train/test
- `typing`: anotasi tipe `List`, `Tuple`, `Dict`

## Ringkasan Modul
- Lihat: [naive_bayes.py:L6-L12](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L6-L12)
- Menandai batasan dan cakupan:
  - Preprocessing ringan: muat CSV dan split
  - Pengelompokkan (classification): Gaussian NB
  - Evaluasi: akurasi & confusion matrix
  - Tidak mencakup clustering (unsupervised)

## Model: GaussianNB
- Deklarasi: [naive_bayes.py:L15-L26](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L15-L26)
- Menyimpan parameter dan statistik:
  - `classes`: label kelas yang ditemukan saat training
  - `class_prior`: probabilitas prior per kelas
  - `theta`: mean per fitur per kelas
  - `sigma`: varians per fitur per kelas (dengan smoothing)
- Catatan: Asumsi Naive Bayes — fitur independen bersyarat dan mengikuti distribusi Gaussian pada tiap kelas.

### Training: fit
- Implementasi: [naive_bayes.py:L28-L64](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L28-L64)
- Langkah:
  - Hitung `counts` per kelas
  - Akumulasi `sums` dan `sq_sums` per fitur per kelas
  - `class_prior[c] = counts[c] / total`
  - Mean fitur: `theta[c][j] = sums[c][j] / counts[c]`
  - Varians fitur: `sigma[c][j] = (sq_sums[c][j] / counts[c]) - mean² + var_smoothing`
  - Smoothing mencegah varians nol agar stabil secara numerik

### Likelihood Gaussian
- Implementasi: [naive_bayes.py:L66-L67](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L66-L67)
- Rumus log-likelihood 1 fitur:
  - `-0.5 * log(2πσ) - (x−μ)² / (2σ)`
  - Dipakai untuk menjumlahkan log-likelihood per fitur

### Prediksi Satu Sampel: predict_one
- Implementasi: [naive_bayes.py:L69-L88](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L69-L88)
- Langkah:
  - Mulai dari `log(class_prior)`
  - Tambahkan log-likelihood Gaussian per fitur
  - Ambil kelas dengan log-probabilitas terbesar
  - Mengembalikan label kelas terbaik

### Prediksi Banyak Sampel: predict
- Implementasi: [naive_bayes.py:L90-L91](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L90-L91)
- Menerapkan `predict_one` untuk setiap baris fitur pada `X`

## Preprocessing

### Memuat CSV: load_csv
- Implementasi: [naive_bayes.py:L94-L109](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L94-L109)
- Langkah:
  - Baca header (nama kolom)
  - Untuk setiap baris:
    - `X`: 13 kolom pertama sebagai fitur (float)
    - `y`: kolom terakhir sebagai target (int: 0/1)
- Mengembalikan: `(X, y, headers)`
- Fitur yang digunakan: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
- Target: `target`

### Train/Test Split: train_test_split
- Implementasi: [naive_bayes.py:L112-L127](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L112-L127)
- Langkah:
  - Bentuk indeks 0..N-1, acak dengan `seed` deterministik (default 42)
  - Potong 80% untuk train, 20% untuk test (`test_size=0.2`)
  - Kembalikan `X_train, X_test, y_train, y_test`

## Evaluasi

### Akurasi
- Implementasi: [naive_bayes.py:L130-L132](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L130-L132)
- Definisi: `(jumlah prediksi benar) / (jumlah total)`

### Confusion Matrix
- Implementasi: [naive_bayes.py:L135-L140](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L135-L140)
- Menghitung frekuensi pasangan `(true_label, pred_label)` sebagai metrik kesalahan/keberhasilan per kelas

## Pipeline Utama: main
- Implementasi: [naive_bayes.py:L143-L167](file:///Users/dsi/projects/heart_attack/naive_bayes.py#L143-L167)
- Alur:
  1. `load_csv("heart.csv")`: muat data
  2. `train_test_split(...)`: bagi data menjadi train/test
  3. Buat `GaussianNB(var_smoothing=1e-9)`
  4. `fit(X_train, y_train)`: latih model
  5. `predict(X_test)`: prediksi pada data uji
  6. Hitung `accuracy` dan `confusion_matrix`
  7. Cetak ringkasan: jumlah fitur, ukuran train/test, akurasi, dan entri confusion matrix

## Parameter & Kustomisasi
- `var_smoothing`: stabilisasi varians untuk Gaussian (default `1e-9`)
- `test_size`: proporsi data uji (default `0.2`)
- `seed`: kontrol pengacakan indeks untuk reproducibility (default `42`)

## Asumsi & Catatan
- Asumsi Gaussian per fitur per kelas; fitur diperlakukan numerik.
- Tidak ada normalisasi atau imputasi; dataset sudah numerik dan lengkap.
- Tidak ada clustering (unsupervised) di skrip ini.
- Untuk fitur yang bersifat kategorikal (encoded angka), pertimbangkan model Bernoulli/Multinomial atau one-hot jika diperlukan.

## Referensi Berkas
- Skrip: [naive_bayes.py](file:///Users/dsi/projects/heart_attack/naive_bayes.py)
- Dataset: [heart.csv](file:///Users/dsi/projects/heart_attack/heart.csv)
