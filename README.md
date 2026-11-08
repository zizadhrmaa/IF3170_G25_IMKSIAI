# IF3170_G25_IMKSIAI

Kelompok 25 (IMKSIAI).

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027: *supervised learning* untuk memprediksi pembatalan pesanan penginapan (`is_canceled`). Metrik leaderboard: Macro F1-score.

## Struktur

```
src/
  Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
                        notebook pengerjaan (dari templat asisten)
  preprocessing.py      Preprocessor: imputasi, fitur turunan, one-hot, standarisasi
  knn.py                KNN from scratch (NumPy)
  metrics.py            Macro F1, akurasi, confusion matrix, stratified k-fold
docs/                   laporan (.pdf)
output/                 hasil run: submission_*.csv dan model tersimpan (models/)
data/                   train.csv, test.csv, sample_submission.csv (tidak di-commit)
```

## Setup dan cara menjalankan

1. Python 3.10+ dan paket di `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```
2. Unduh data kompetisi dari Kaggle, lalu letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` di folder `data/`.
3. Buka `src/Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb` (Jupyter, VS Code, atau Colab) lalu jalankan semua sel dari atas. Notebook bisa dijalankan dari folder `src/` maupun dari root repository.
   - Di Colab, unggah folder `src/` dan `data/` dengan struktur yang sama.
4. Submisi Kaggle tersimpan di `output/submission_knn.csv`.

Seluruh langkah deterministik (seed 42), sehingga menjalankan ulang notebook menghasilkan submisi yang sama.

## Status

| Model | Status | Macro F1 validasi |
| --- | --- | --- |
| KNN (k=21, Manhattan, bobot jarak, jarak berbobot fitur) | selesai | 0,849 |
| DTL | belum |  |
| Logistic Regression (L1, L2, Elastic Net) | belum |  |
| SVM (linear, polynomial, Gaussian) | belum |  |
| Naive Bayes | belum |  |

| Bonus | Status | Keterangan |
| --- | --- | --- |
| Cost-sensitive classification (FP 2x FN) | selesai | KNN, ambang 0,65: biaya validasi turun 10,1% |

## Pembagian tugas

| NIM | Nama | Tugas |
| --- | --- | --- |
| 13524014 | Yusuf Faishal Listyardi | Naive Bayes |
| 13524017 | Aziza Dharma Putri | KNN |
| 13524029 | Niko Samuel Simanjuntak | Logistic Regression |
| 13524038 | An-Dafa Anza Alvansyah | SVM |
| 13524071 | Kalyca Nathania Benedicta M. | Decision Tree Learning (DTL) |
