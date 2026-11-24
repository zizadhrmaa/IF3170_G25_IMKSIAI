# IF3170_G25_IMKSIAI

Kelompok 25 (IMKSIAI).

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027: *supervised learning* untuk memprediksi pembatalan pesanan penginapan (`is_canceled`). Metrik utama kompetisi adalah Macro F1-score.

## Struktur repository

```text
src/
  Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
  preprocessing.py                    Pipeline preprocessing
  knn.py                              KNN from scratch
  knn_weight_search.py                Optimasi bobot fitur KNN
  dtl.py                              C4.5 dan Genetic Algorithm
  dtl_visualization.py                Visualisasi top-N percabangan
  logistic_regression.py              Logistic Regression, SGA, Adam, dan regularisasi
  logistic_regression_visualization.py Video kontur log-loss
  svm.py                              SVM tiga kernel dan Differential Evolution
  naive_bayes.py                      Naive Bayes kategorikal dan PSO
  naive_bayes_visualization.py        Visualisasi graf Naive Bayes
  metrics.py                          Metrik evaluasi dan stratified k-fold
  pca.py                              Modul PCA tambahan (tidak dipakai pada pipeline utama)
docs/                                 Laporan tugas besar (.pdf)
output/                               CSV submission, hasil eksperimen, dan visualisasi
  models/                             Model dan preprocessor (dihasilkan saat run)
  visualization/                      Gambar dan video
data/                                Dataset Kaggle (tidak di-commit)
requirements.txt
```

## Setup dan cara menjalankan

1. Gunakan Python 3.10 atau lebih baru. Buat virtual environment pada root repository, lalu instal dependensi:

   ```bash
   python -m venv .venv
   ```

   Aktifkan environment (`.venv\Scripts\Activate.ps1` pada PowerShell Windows atau `source .venv/bin/activate` pada Linux/macOS), kemudian jalankan:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Unduh dataset kompetisi Kaggle. Letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` pada folder `data/`.
3. Buka notebook utama pada folder `src/` menggunakan VS Code atau Jupyter. Pilih kernel Python dari environment yang sudah disiapkan. Notebook mendukung working directory root repository atau `src/`.
4. Simpan salinan submission yang pernah diunggah ke Kaggle sebelum menjalankan ulang eksperimen. Pilih **Restart Kernel**, lalu **Run All**. Jangan melewati sel karena beberapa eksperimen dan bonus memerlukan hasil dari bagian sebelumnya.
5. Setelah selesai, simpan notebook beserta output sel. Periksa bahwa tidak ada error, seluruh file yang diperlukan terbentuk, dan isi submission cocok dengan file yang diunggah ke Kaggle.

Notebook menggunakan seed tetap 42 dan konfigurasi eksperimen yang tercantum pada masing-masing bagian. Kecocokan hasil harus diverifikasi dengan menjalankan notebook menggunakan dataset asli dan environment yang sesuai. Waktu training dapat berbeda antarperangkat, terutama pada cross-validation dan optimasi SVM. `imageio-ffmpeg` membantu menghasilkan video bonus Logistic Regression jika FFmpeg tidak tersedia pada sistem.

## Output submission

| Model | File pada `output/` |
| --- | --- |
| KNN | `submission_knn.csv`, `submission_knn_v1.csv` (pembanding) |
| DTL | `submission_dtl.csv`, `submission_dtl_optimized.csv` |
| Logistic Regression | `submission_logistic_regression.csv`, `submission_logistic_regression_adam.csv` |
| SVM | `submission_svm.csv`, `submission_svm_optimized.csv`, `submission_svm_refined.csv` |
| Naive Bayes | `submission_naive_bayes.csv`, `submission_naive_bayes_optimized.csv` |

File CSV eksperimen `naive_bayes_cv.csv`, `naive_bayes_comparison.csv`, `svm_optimization_history.csv`, dan `svm_optimization_cv.csv` juga dihasilkan oleh notebook. File tersebut bukan submission Kaggle. Gambar dan video disimpan pada `output/visualization/`, sedangkan model tersimpan pada `output/models/`.

## Hasil eksperimen

Macro F1 berikut berasal dari output notebook yang tercatat pada repository, bukan skor leaderboard Kaggle. Nilai terbaru dapat dilihat pada tabel hasil setelah notebook dijalankan ulang.

| Model | Macro F1 validasi |
| --- | ---: |
| KNN (optimasi bobot fitur) | 0,8541 |
| DTL (tuning CV) | 0,81194 |
| DTL (Genetic Algorithm) | 0,82165 |
| Logistic Regression SGA (tuning dan ambang OOF) | 0,76349 |
| Logistic Regression Adam (tuning dan ambang OOF) | 0,76289 |
| SVM Gaussian | 0,79329 |
| SVM (Differential Evolution) | 0,82421 |
| SVM (optimasi lanjutan) | 0,82644 |
| Naive Bayes | 0,74633 |
| Naive Bayes (PSO) | 0,75360 |

## Bonus

Notebook memuat analisis cost-sensitive classification pada KNN, visualisasi Decision Tree dengan top-N percabangan, Genetic Algorithm DTL, video kontur log-loss dan optimasi Adam pada Logistic Regression, visualisasi tiga kernel SVM serta optimasi Differential Evolution, dan visualisasi graf serta optimasi PSO pada Naive Bayes. Penjelasan dan referensi untuk algoritma optimasi tersedia pada bagian bonus terkait.

## Pembagian tugas

| NIM | Nama | Tugas |
| --- | --- | --- |
| 13524014 | Yusuf Faishal Listyardi | Naive Bayes |
| 13524017 | Aziza Dharma Putri | KNN |
| 13524029 | Niko Samuel Simanjuntak | Logistic Regression |
| 13524038 | An-Dafa Anza Alvansyah | SVM |
| 13524071 | Kalyca Nathania Benedicta M. | Decision Tree Learning (DTL) |