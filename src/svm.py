import pickle
import warnings

import numpy as np


class SVM:
    VALID_KERNELS = ("linear", "polynomial", "gaussian")

    def __init__(self, C=1.0, kernel="linear", degree=3, gamma=None,
                 coef0=1.0, tol=1e-3, max_iter=100000, batch_size=256):
        if kernel not in self.VALID_KERNELS:
            raise ValueError("kernel harus 'linear', 'polynomial', atau 'gaussian'")
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C harus positif dan finite")
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
            raise ValueError("gamma harus positif dan finite")
        if not np.isfinite(coef0) or coef0 < 0:
            raise ValueError("coef0 harus nonnegatif dan finite")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol harus positif dan finite")
        for name, value in [("degree", degree), ("max_iter", max_iter),
                            ("batch_size", batch_size)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + " harus bilangan bulat positif")
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.fitted_ = False

    def _kernel(self, X, Y):
        with np.errstate(over="ignore", invalid="ignore"):
            dot = X @ Y.T
            if self.kernel == "linear":
                result = dot
            elif self.kernel == "polynomial":
                result = (self.gamma_ * dot + self.coef0) ** self.degree
            else:
                squared_distance = ((X ** 2).sum(axis=1)[:, None]
                                    + (Y ** 2).sum(axis=1)[None, :] - 2 * dot)
                result = np.exp(-self.gamma_ * np.maximum(squared_distance, 0))
        if not np.isfinite(result).all():
            raise ValueError("Nilai kernel tidak finite; periksa scaling dan parameter kernel")
        return result

    def fit(self, X, y):
        self.fitted_ = False
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
            raise ValueError("X harus 2 dimensi dengan minimal 2 baris dan 1 fitur")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y harus 1 dimensi dengan panjang sama dengan jumlah baris X")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X dan y tidak boleh mengandung NaN atau infinity")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM ini memerlukan tepat 2 kelas")
        labels = np.where(y == self.classes_[0], -1.0, 1.0)
        self.n_features_in_ = X.shape[1]
        self.gamma_ = 1.0 / X.shape[1] if self.gamma is None else self.gamma
        alpha = np.zeros(len(X))
        gradient = labels.copy()
        self.n_iter_ = 0
        diagonal = np.empty(len(X))
        for index in range(len(X)):
            point = X[index:index + 1]
            diagonal[index] = self._kernel(point, point)[0, 0]

        # SMO mengubah dua alpha sekaligus agar jumlah alpha dikali label tetap nol
        # Referensi: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
        while True:
            can_increase = ((labels == 1) & (alpha < self.C)) | ((labels == -1) & (alpha > 0))
            can_decrease = ((labels == 1) & (alpha > 0)) | ((labels == -1) & (alpha < self.C))
            upper = np.flatnonzero(can_increase)
            lower = np.flatnonzero(can_decrease)
            i = upper[np.argmax(gradient[upper])]
            j = lower[np.argmin(gradient[lower])]
            gap = gradient[i] - gradient[j]
            if gap <= self.tol or self.n_iter_ >= self.max_iter:
                break

            # Pilih alpha kedua dengan memperhitungkan perubahan objektif dual.
            # Referensi: https://www.jmlr.org/papers/v6/fan05a.html
            column_i = self._kernel(X, X[i:i + 1])[:, 0]
            curvatures = np.maximum(diagonal[i] + diagonal[lower] - 2 * column_i[lower], 1e-12)
            differences = gradient[i] - gradient[lower]
            gains = np.where(differences > 0, differences ** 2 / curvatures, 0)
            j = lower[np.argmax(gains)]
            column_j = self._kernel(X, X[j:j + 1])[:, 0]
            curvature = diagonal[i] + diagonal[j] - 2 * column_i[j]
            limit_i = self.C - alpha[i] if labels[i] == 1 else alpha[i]
            limit_j = alpha[j] if labels[j] == 1 else self.C - alpha[j]
            # Batasi langkah agar kedua alpha tetap berada di antara 0 dan C
            step = min(limit_i, limit_j)
            if curvature > 0:
                step = min(step, gap / curvature)
            alpha[i] = np.clip(alpha[i] + labels[i] * step, 0, self.C)
            alpha[j] = np.clip(alpha[j] - labels[j] * step, 0, self.C)
            # Sesuaikan gradient setelah kedua alpha diperbarui
            gradient -= step * (column_i - column_j)
            self.n_iter_ += 1

        self.alpha_ = alpha
        self.b_ = (gradient[i] + gradient[j]) / 2
        self.optimality_gap_ = max(0.0, float(gap))
        self.converged_ = gap <= self.tol
        self.support_ = np.flatnonzero(alpha > 0)
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = alpha[self.support_] * labels[self.support_]
        self.w_ = self.dual_coef_ @ self.support_vectors_ if self.kernel == "linear" else None
        self.fitted_ = True
        if not self.converged_:
            warnings.warn("SVM belum konvergen; naikkan max_iter atau periksa scaling fitur",
                          RuntimeWarning, stacklevel=2)
        return self

    def decision_function(self, X):
        if not self.fitted_:
            raise RuntimeError("SVM belum di-fit, panggil fit sebelum prediksi")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X harus 2 dimensi dengan jumlah fitur sama seperti saat fit")
        if not np.isfinite(X).all():
            raise ValueError("X tidak boleh mengandung NaN atau infinity")
        if self.kernel == "linear":
            scores = X @ self.w_ + self.b_
        else:
            scores = np.empty(len(X))
            for start in range(0, len(X), self.batch_size):
                batch = X[start:start + self.batch_size]
                scores[start:start + self.batch_size] = (
                    self._kernel(batch, self.support_vectors_) @ self.dual_coef_ + self.b_
                )
        if not np.isfinite(scores).all():
            raise ValueError("Decision score tidak finite; periksa scaling fitur")
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("SVM belum di-fit, panggil fit sebelum save")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
