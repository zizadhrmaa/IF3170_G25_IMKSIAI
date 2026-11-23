import pickle

import numpy as np


class PCA:
    def __init__(self, n_components):
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components harus bilangan bulat positif")
        self.n_components = int(n_components)
        self.fitted_ = False

    def fit(self, X):
        self.fitted_ = False
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
            raise ValueError("X harus 2 dimensi dengan minimal 2 baris dan 1 fitur")
        if not np.isfinite(X).all():
            raise ValueError("X tidak boleh mengandung NaN atau infinity")
        if self.n_components > min(X.shape):
            raise ValueError("n_components melebihi min(jumlah baris, jumlah fitur)")
        self.mean_ = X.mean(axis=0)
        _, singular, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        anchor = vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)]
        signs = np.where(anchor < 0, -1.0, 1.0)
        vt = vt * signs[:, None]
        variance = singular ** 2 / (len(X) - 1)
        self.components_ = vt[:self.n_components]
        self.explained_variance_ = variance[:self.n_components]
        self.explained_variance_ratio_ = variance[:self.n_components] / variance.sum()
        self.n_features_in_ = X.shape[1]
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("PCA belum di-fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Jumlah fitur berbeda dengan data latih")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("PCA belum di-fit")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)