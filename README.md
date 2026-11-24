# IF3170_G25_IMKSIAI

Kelompok 25 (IMKSIAI).

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027: *supervised learning* untuk memprediksi pembatalan pesanan penginapan (`is_canceled`). Metrik leaderboard: Macro F1-score.

## Struktur

```
src/
  Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
                                  notebook pengerjaan (dari templat asisten)
  preprocessing.py                Preprocessor, NaiveBayesPreprocessor, SVMPreprocessor, drop_duplicate_rows
  knn.py                          KNN from scratch (NumPy)
  knn_weight_search.py            pencarian bobot fitur jarak KNN (coordinate search berbasis CV)
  dtl.py                          Decision Tree C4.5 from scratch + Genetic Algorithm
  dtl_visualization.py            visualisasi pohon keputusan
  logistic_regression.py          Logistic Regression from scratch (SGA/Adam; L1, L2, Elastic Net)
  logistic_regression_visualization.py  kontur log-loss (video MP4)
  svm.py                          SVM from scratch (SMO; linear, polynomial, Gaussian) + Differential Evolution
  naive_bayes.py                  Naive Bayes kategorikal + Particle Swarm Optimization
  naive_bayes_visualization.py    graf Naive Bayes
  metrics.py                      Macro F1, akurasi, confusion matrix, biaya, stratified k-fold
docs/                             laporan (.pdf)
output/                           hasil run: submission_*.csv, models/, visualization/
data/                             train.csv, test.csv, sample_submission.csv (tidak di-commit)
```

## Setup dan cara menjalankan

1. Python 3.10+ dan paket di `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```
2. Unduh data kompetisi dari Kaggle, lalu letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` di folder `data/`.
3. Buka `src/Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb` (Jupyter, VS Code, atau Colab) lalu jalankan semua sel dari atas. Notebook bisa dijalankan dari folder `src/` maupun dari root repository.
   - Di Colab, unggah folder `src/` dan `data/` dengan struktur yang sama.
   - Video kontur log-loss membutuhkan FFmpeg (paket `imageio-ffmpeg`).
4. Submisi Kaggle tersimpan di `output/submission_*.csv` (KNN, DTL, Logistic Regression, SVM, dan Naive Bayes, termasuk versi optimasi). `output/submission_knn_v1.csv` membuat ulang submisi KNN sebelumnya sebagai pembanding.

Seluruh langkah deterministik (seed 42), sehingga menjalankan ulang notebook menghasilkan submisi yang sama. Preprocessing di-*fit* hanya pada data latih (tidak ada penggabungan train dan test).

## Status

Macro F1 pada validasi 20% (stratified, seed 42). Skor validasi yang dipakai untuk memilih konfigurasi bersifat estimasi lokal; skor akhir adalah skor leaderboard Kaggle.

| Model | Status | Macro F1 validasi |
| --- | --- | --- |
| KNN (k=31, Manhattan, voting 1/jarak², bobot fitur hasil coordinate search) | selesai | 0,8541 |
| DTL C4.5 (gain ratio, nilai hilang pecahan, cost-complexity pruning), tuning CV | selesai | 0,8119 |
| Logistic Regression (SGA; L1, L2, Elastic Net; konfigurasi terpilih tanpa penalti) | selesai | 0,7638 |
| SVM (linear 0,7599; polynomial 0,7771; Gaussian 0,7933) | selesai | 0,7933 (Gaussian) |
| Naive Bayes (diskretisasi 40 bin, Laplace alpha=1) | selesai | 0,7463 |

| Bonus | Status | Keterangan |
| --- | --- | --- |
| Cost-sensitive classification (FP 2x FN) | selesai | KNN: ambang Bayes 2/3 menurunkan biaya validasi 8,5% (992 → 908); ambang biaya minimum CV 0,71 menurunkan 7,1%; Macro F1 turun pada keduanya |
| Optimasi SVM dengan Differential Evolution | selesai | Macro F1 validasi 0,7933 → 0,8242; optimasi lanjutan (pembobotan fitur) 0,8264 |
| Optimasi Naive Bayes dengan PSO | selesai | 47 bin, alpha=0,0567, ambang=0,4804; Macro F1 CV 0,7477 → 0,7503, validasi 0,7463 → 0,7536 |
| Optimasi DTL dengan Genetic Algorithm | selesai | Macro F1 CV 0,8120 → 0,8186, validasi 0,8217 |
| Optimizer Adam untuk Logistic Regression | selesai | mencapai target log-loss lebih cepat per epoch; Macro F1 setara SGA |
| Visualisasi (pohon DTL, SVM 2D, kontur log-loss, graf Naive Bayes) | selesai | gambar dan video di `output/visualization/` |

## Pembagian tugas

| NIM | Nama | Tugas |
| --- | --- | --- |
| 13524014 | Yusuf Faishal Listyardi | Naive Bayes |
| 13524017 | Aziza Dharma Putri | KNN |
| 13524029 | Niko Samuel Simanjuntak | Logistic Regression |
| 13524038 | An-Dafa Anza Alvansyah | SVM |
| 13524071 | Kalyca Nathania Benedicta M. | Decision Tree Learning (DTL) |