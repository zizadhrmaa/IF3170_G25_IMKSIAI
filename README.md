# IF3170_G25_IMKSIAI

Kelompok 25 (IMKSIAI).

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027: *supervised learning* untuk memprediksi pembatalan pesanan Ronaldo Hotel™ (`is_canceled`). Metrik leaderboard: Macro F1-score. Semua algoritma utama (KNN, Decision Tree, Logistic Regression, SVM, Naive Bayes) diimplementasikan *from scratch* dengan NumPy dan dibandingkan dengan scikit-learn. SVM memakai `scipy.optimize.minimize` (SLSQP) untuk menyelesaikan dual QP.

## Struktur
```
src/
Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb notebook pengerjaan
preprocessing.py Preprocessor, SVMPreprocessor, NaiveBayesPreprocessor
pca.py PCA dengan SVD (NumPy) untuk eksperimen reduksi dimensi
knn.py KNN from scratch
knn_weight_search.py pencarian bobot fitur jarak KNN
dtl.py Decision Tree C4.5 from scratch + Genetic Algorithm (bonus)
logreg.py Logistic Regression from scratch (L1, L2, Elastic Net; GD dan Adam)
svm.py SVM from scratch (Linear, Polynomial, Gaussian) + Differential Evolution (bonus)
naive_bayes.py Naive Bayes from scratch + Particle Swarm Optimization (bonus)
metrics.py Macro F1, akurasi, confusion matrix, biaya, stratified k-fold
visualization.py gambar tree DTL, video Logistic Regression, graf Naive Bayes
docs/ laporan (.pdf)
output/ submission_*.csv, hasil eksperimen (.csv), visualization/, models/
data/ train.csv, test.csv, sample_submission.csv (tidak di-commit)
```

## Setup dan cara menjalankan

1. Gunakan Python 3.10 atau lebih baru, lalu pasang paket:
   `pip install -r requirements.txt`
2. Unduh data dari Kaggle Competition dan letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` di folder `data/` pada root repository.
3. Buka `src/Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb`, pilih *Restart Kernel and Run All Cells*. Notebook dapat dijalankan dari folder `src/` maupun dari root repository. Seluruh proses memakai seed 42 sehingga hasilnya sama setiap kali dijalankan. Waktu total dapat mencapai lebih dari satu jam karena pelatihan SVM dan pencarian parameter bonus.
4. Hasil run tersimpan di `output/`.

## Submission Kaggle

Submission yang dikumpulkan adalah `output/submission_knn.csv` (KNN *from scratch*, tanpa *ensemble*). Sel verifikasi di bagian *Kaggle Submission* pada notebook memastikan model yang dimuat kembali dan pelatihan ulang menghasilkan prediksi identik dengan berkas tersebut.

Skor leaderboard Kaggle (Macro F1, submission final `output/submission_knn.csv`): **\<isi skor setelah submisi final diunggah\>**

## Catatan preprocessing

- **KNN dan Logistic Regression** memakai `Preprocessor`: imputasi median untuk numerik dan kategori `"Unknown"` untuk kategorikal, penambahan fitur turunan (`log_lead_time`, `arrival_date_ordinal`), one-hot encoding dengan kategori langka digabung menjadi `"Other"`, dan standarisasi z-score. KNN juga memakai `feature_weights` hasil `knn_weight_search` (jarak Manhattan berbobot, coordinate search dengan CV).
- **SVM** memakai `SVMPreprocessor`, turunan `Preprocessor` dengan tiga opsi tambahan: `log_history` (log1p untuk `prior_cancellations` dan `prior_completed_bookings`), `binary_unscaled` (fitur biner `needs_parking` dan `is_returning_guest` tetap 0/1 tanpa z-score), dan `date_encoding` (`original`/`ordinal`/`cyclical`) untuk kolom tanggal kedatangan.
- **Naive Bayes** memakai `NaiveBayesPreprocessor`: diskretisasi kuantil untuk fitur numerik, pemetaan kategori ke kode `0` (missing), `1` (unseen), `2+` (kategori training). Sesuai QnA asisten, atribut numerik didiskretkan karena slide [5] hanya membahas atribut kategorikal.
- **Decision Tree** menerima DataFrame mentah karena DTL menangani missing value secara alami lewat split bercabang (fractional propagation), transformasi monoton seperti `log_lead_time` tidak mengubah split optimal, dan eksperimen di notebook (*Preprocessing Variations*) menunjukkan Macro F1 DTL mentah setara atau lebih baik dibanding DTL dengan `Preprocessor`.

## Berkas output

| Berkas | Isi |
| --- | --- |
| `output/submission_knn.csv` | submission Kaggle final (KNN) |
| `output/submission_knn_v1.csv` | KNN konfigurasi awal, pembanding |
| `output/submission_dtl.csv` | submission Decision Tree |
| `output/submission_logreg.csv` | submission Logistic Regression |
| `output/submission_svm.csv` | submission SVM kernel terbaik |
| `output/submission_svm_optimized.csv` | SVM hasil Differential Evolution |
| `output/submission_svm_refined.csv` | SVM hasil tuning tambahan |
| `output/submission_naive_bayes.csv` | submission Naive Bayes |
| `output/experiment_*.csv` | eksperimen PCA, feature selection, class balance |
| `output/dtl_sklearn_variants.csv` | perbandingan varian DTL (from scratch vs sklearn) |
| `output/inference_time.csv` | waktu inferensi per model |
| `output/visualization/dtl/` | gambar tree DTL dengan parameter `top_n` |
| `output/visualization/logreg/` | video kontur log-loss (`logreg_training.mp4` atau `.gif`) |
| `output/visualization/svm/` | bidang pemisah per kernel (`svm_linear.png`, `svm_polynomial.png`, `svm_gaussian.png`) |
| `output/visualization/naive_bayes/` | graf struktur, perbandingan, dan kurva PSO |
| `output/models/` | model dan preprocessor tersimpan (`.pkl`) |

## Status

| Model | Status | Macro F1 validasi |
| --- | --- | --- |
| KNN (Manhattan, voting berbobot jarak, bobot fitur hasil coordinate search) | selesai | lihat notebook |
| Decision Tree (C4.5 + gain ratio + numeric penalty + ccp_alpha pruning) | selesai | lihat notebook |
| Logistic Regression (L1, L2, Elastic Net; GD dan Adam) | selesai | lihat notebook |
| SVM (Linear, Polynomial, Gaussian; dual QP via SLSQP) | selesai | lihat notebook |
| Naive Bayes (diskretisasi kuantil, Laplace +1, MAP) | selesai | lihat notebook |

| Bonus | Status |
| --- | --- |
| Gambar tree DTL dengan parameter `top_n` | selesai |
| Video kontur log-loss Logistic Regression | selesai |
| Gambar bidang pemisah SVM per kernel | selesai |
| Graf Naive Bayes | selesai |
| Optimasi: Genetic Algorithm (DTL), Adam (Logistic Regression), Differential Evolution (SVM), PSO (Naive Bayes) | selesai |
| Cost-sensitive classification (FP 2x FN) | selesai |

## Pembagian tugas

| NIM | Nama | Tugas |
| --- | --- | --- |
| 13524014 | Yusuf Faishal Listyardi | Naive Bayes |
| 13524017 | Aziza Dharma Putri | KNN |
| 13524029 | Niko Samuel Simanjuntak | Logistic Regression |
| 13524038 | An-Dafa Anza Alvansyah | SVM |
| 13524071 | Kalyca Nathania Benedicta M. | Decision Tree Learning (DTL) |

## Referensi

Lihat daftar lengkap pada `docs/laporan.pdf`. Referensi kunci di kode: C4.5 (Quinlan, 1993), threshold penalty (Quinlan, JAIR 4, 1996), Logistic Regression L1 proximal (Parikh & Boyd, 2014), Adam (Kingma & Ba, 2015), SVM dual SLSQP (Kraft, 1988), Differential Evolution (Storn & Price, 1997), Particle Swarm (Kennedy & Eberhart, 1995), Bayesian Network (Maulidevi, IF3170 2024, slide 27-42), Genetic Algorithm (Holland, 1975; Goldberg, 1989).