# Dokumentasi: Naive Bayes untuk Dataset Penyakit Jantung

Proyek ini membangun dan mengevaluasi model klasifikasi Naive Bayes (Gaussian) pada dataset penyakit jantung `heart.csv`. Model memprediksi kolom `target` (0: tidak ada penyakit jantung, 1: ada penyakit jantung) dari 13 fitur numerik.

## Struktur Proyek
- `heart.csv` — dataset dengan 14 kolom (13 fitur + 1 target)
- `naive_bayes.py` — skrip pelatihan dan evaluasi Gaussian Naive Bayes

## Dataset
Kolom: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`
- Semua fitur diperlakukan sebagai numerik dan dimodelkan dengan distribusi Gaussian
- `target` adalah label biner (0 atau 1)

### Ringkasan
- Jumlah data (baris, tidak termasuk header): 1025
- Jumlah fitur: 13
- Daftar fitur: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- Target: `target`

## Menjalankan
Pastikan Python 3 tersedia, lalu jalankan:

```bash
python3 naive_bayes.py
```

Output yang dicetak mencakup jumlah fitur, ukuran train/test, akurasi, serta entri confusion matrix.

## Metodologi
- Algoritma: Gaussian Naive Bayes dengan `var_smoothing=1e-9` untuk mencegah varians nol
- Pembagian data: split 80/20 dengan seed deterministik `42`
- Evaluasi: akurasi keseluruhan dan confusion matrix

## Contoh Hasil
Contoh hasil eksekusi pada dataset ini:
```
Features: 13
Train size: 820
Test size: 205
Accuracy: 0.8098
Confusion matrix entries:
(0, 0):75
(0, 1):32
(1, 0):7
(1, 1):91
```

## Kustomisasi
Parameter dapat diubah langsung di `naive_bayes.py`:
- `test_size`: proporsi data uji (mis. 0.2)
- `seed`: kontrol pemilihan train/test yang deterministik
- `var_smoothing`: penstabil varians Gaussian (mis. 1e-9)

## Pengembangan Lanjut
- Gunakan Bernoulli/Multinomial NB untuk fitur kategorikal/biner secara eksplisit
- Tambahkan k-fold cross-validation untuk evaluasi lebih kuat
- Normalisasi/standarisasi fitur jika diperlukan
- Simpan dan muat model terlatih (mis. via pickle)

## Referensi Berkas
- Dataset: `heart.csv`
- Skrip: `naive_bayes.py`
- Dokumentasi Peningkatan Akurasi: [README_Accuracy_Improvement.md](README_Accuracy_Improvement.md)
