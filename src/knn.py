"""
Implementasi k-Nearest Neighbors dengan NumPy.

KNN adalah lazy learner, sehingga fit hanya menyimpan
data latih. Kelas data baru ditentukan dari voting k tetangga terdekat dengan jarak Manhattan
(p=1), Euclidean (p=2), atau Minkowski orde p.

Voting berbobot 1/jarak^power mengacu pada P. Cunningham dan S. J. Delany, "k-Nearest Neighbour
Classifiers: 2nd Edition (with Python examples)", arXiv:2004.04523, 2020, Section 3: pangkat 1 adalah
pilihan umum, sedangkan pangkat lebih besar makin mengurangi pengaruh tetangga yang jauh. Gagasan voting
berbobot jarak berasal dari S. A. Dudani, "The Distance-Weighted k-Nearest-Neighbor Rule", IEEE Trans.
Systems, Man, and Cybernetics, SMC-6(4), 1976, hlm. 325-327.
"""

import pickle

import numpy as np


class KNN:
    """
    Klasifikasi k-Nearest Neighbors. Jarak dihitung per batch query agar matriks jarak
    tidak memenuhi memori ketika data latih besar.
    """

    #: Jenis voting yang didukung.
    VALID_WEIGHTS = ("uniform", "distance")

    def __init__(self, k=5, p=2, weights="uniform", batch_size=256, power=1.0):
        """
        :param k: jumlah tetangga yang ikut voting, minimal 1
        :param p: orde Minkowski; 1 = Manhattan, 2 = Euclidean
        :param weights: "uniform" jika setiap tetangga bernilai satu suara,
                        "distance" jika suara tetangga sebesar 1/jarak^power
        :param batch_size: jumlah baris query yang dihitung jaraknya sekaligus
        :param power: pangkat jarak pada voting "distance"; 1 = 1/jarak, 2 = 1/jarak^2.
                      Pangkat lebih besar membuat tetangga yang paling dekat lebih dominan.
        :raises ValueError: jika k kurang dari 1, weights tidak dikenal, atau power tidak positif
        """
        if k < 1:
            raise ValueError("k harus minimal 1, didapat " + str(k))
        if weights not in self.VALID_WEIGHTS:
            raise ValueError("weights harus 'uniform' atau 'distance', didapat '" + str(weights) + "'")
        if power <= 0:
            raise ValueError("power harus positif, didapat " + str(power))
        self.k = k
        self.p = p
        self.weights = weights
        self.batch_size = batch_size
        self.power = power

    def fit(self, X, y):
        """
        Menyimpan data latih. Label diubah menjadi indeks 0..n_kelas-1 dan nilai aslinya
        disimpan di classes_.

        :param X: matriks fitur data latih, berukuran (n, d)
        :param y: label data latih, sepanjang n
        :return: objek ini sendiri
        :raises ValueError: jika X bukan 2 dimensi, jumlah barisnya tidak sama dengan y,
                            atau k lebih besar dari jumlah data latih
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X harus 2 dimensi dengan jumlah baris sama dengan y")
        if self.k > len(X):
            raise ValueError("k = " + str(self.k) + " melebihi jumlah data latih " + str(len(X)))
        self.X_train_ = X
        self.classes_, self.y_train_ = np.unique(y, return_inverse=True)
        return self

    def _distances(self, Q):
        """
        Menghitung jarak setiap baris Q ke setiap baris data latih.

        :param Q: matriks query berukuran (m, d)
        :return: matriks jarak berukuran (m, n_latih)
        """
        X = self.X_train_
        if self.p == 2:
            # Bentuk ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x jauh lebih cepat daripada selisih per elemen.
            # Hasilnya bisa sedikit negatif karena pembulatan, jadi dipotong ke 0 sebelum diakarkan.
            sq = (Q ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2.0 * Q @ X.T
            return np.sqrt(np.maximum(sq, 0.0))
        diff = np.abs(Q[:, None, :] - X[None, :, :])
        if self.p == 1:
            return diff.sum(-1)
        return (diff ** self.p).sum(-1) ** (1.0 / self.p)

    def kneighbors(self, X, k=None):
        """
        Mencari k tetangga terdekat setiap baris X.

        :param X: matriks query berukuran (m, d)
        :param k: jumlah tetangga; None berarti memakai self.k
        :return: tuple (jarak, indeks), masing-masing berukuran (m, k) dan terurut dari yang terdekat
        """
        if k is None:
            k = self.k
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        all_dist = np.empty((n, k))
        all_idx = np.empty((n, k), dtype=np.int64)

        # Jarak selain Euclidean membentuk array 3 dimensi (m, n_latih, d), jadi batch-nya dikecilkan.
        step = self.batch_size if self.p == 2 else max(1, self.batch_size // 8)
        for start in range(0, n, step):
            end = start + step
            d = self._distances(X[start:end])
            cand = np.argpartition(d, k - 1, axis=1)[:, :k]
            cand_d = np.take_along_axis(d, cand, axis=1)
            # Kandidat diurutkan berdasarkan jarak. Jika jaraknya sama, indeks latih yang lebih kecil
            # didahulukan agar hasilnya deterministik.
            order = np.lexsort((cand, cand_d), axis=1)
            all_idx[start:end] = np.take_along_axis(cand, order, axis=1)
            all_dist[start:end] = np.take_along_axis(cand_d, order, axis=1)
        return all_dist, all_idx

    def _vote(self, dist, idx):
        """
        Menghitung probabilitas kelas dari jarak dan indeks tetangga.

        :param dist: jarak tetangga berukuran (m, k)
        :param idx: indeks tetangga di data latih berukuran (m, k)
        :return: probabilitas kelas berukuran (m, n_kelas)
        """
        labels = self.y_train_[idx]
        if self.weights == "uniform":
            w = np.ones_like(dist)
        else:
            # Model yang disimpan sebelum parameter power ada tetap memakai 1/jarak.
            power = getattr(self, "power", 1.0)
            with np.errstate(divide="ignore"):
                w = 1.0 / dist ** power
            # Tetangga berjarak 0 memiliki bobot tak hingga. Jika ada, hanya tetangga itu yang ikut voting.
            exact = dist == 0
            has_exact = exact.any(axis=1)
            w[has_exact] = exact[has_exact].astype(float)

        proba = np.zeros((len(dist), len(self.classes_)))
        for c in range(len(self.classes_)):
            proba[:, c] = (w * (labels == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        """
        :param X: matriks query berukuran (m, d)
        :return: probabilitas kelas berukuran (m, n_kelas), urutan kolom sesuai classes_
        """
        dist, idx = self.kneighbors(X)
        return self._vote(dist, idx)

    def predict(self, X):
        """
        :param X: matriks query berukuran (m, d)
        :return: label prediksi sepanjang m
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_from_neighbors(self, dist, idx, k):
        """
        Memprediksi dengan k kolom pertama hasil kneighbors yang dipanggil dengan k lebih besar.
        Dipakai saat tuning agar jarak cukup dihitung sekali untuk semua nilai k.

        :param dist: jarak tetangga hasil kneighbors
        :param idx: indeks tetangga hasil kneighbors
        :param k: jumlah tetangga yang dipakai, tidak boleh melebihi jumlah kolom dist
        :return: label prediksi
        """
        proba = self._vote(dist[:, :k], idx[:, :k])
        return self.classes_[proba.argmax(axis=1)]

    def save(self, path):
        """
        :param path: lokasi berkas pickle tujuan
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        :param path: lokasi berkas pickle hasil save
        :return: objek KNN yang tersimpan
        """
        with open(path, "rb") as f:
            return pickle.load(f)
