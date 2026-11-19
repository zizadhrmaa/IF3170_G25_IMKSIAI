import numpy as np


class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=50, seed=42):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate harus positif dan finite")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs harus bilangan bulat positif")
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self.coef_ = None
        self.intercept_ = 0.0

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X harus berupa matriks 2 dimensi dengan minimal satu fitur")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN atau infinity")
        return X

    @staticmethod
    def _validate_y(y, n_samples):
        y = np.asarray(y)
        if y.shape != (n_samples,) or n_samples == 0:
            raise ValueError("y harus berupa vektor tidak kosong dengan satu label per baris X")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Label y harus bernilai 0 atau 1")
        return y.astype(float, copy=False)

    @staticmethod
    def _sigmoid(z):
        return np.exp(-np.logaddexp(0.0, -z))

    def fit(self, X, y):
        X = self._validate_X(X)
        y = self._validate_y(y, len(X))
        weights = np.zeros(X.shape[1])
        intercept = 0.0
        rng = np.random.default_rng(self.seed)

        with np.errstate(over="raise", invalid="raise"):
            for _ in range(self.epochs):
                for i in rng.permutation(len(X)):
                    x_i = X[i]
                    score = x_i @ weights + intercept
                    p = self._sigmoid(score)
                    step = self.learning_rate * (y[i] - p)
                    weights += step * x_i
                    intercept += step

        self.coef_ = weights
        self.intercept_ = intercept
        return self

    def decision_function(self, X):
        if self.coef_ is None:
            raise RuntimeError("Panggil fit sebelum menghitung skor model")
        X = self._validate_X(X)
        if X.shape[1] != len(self.coef_):
            raise ValueError("Jumlah fitur X harus sama seperti saat fit")
        with np.errstate(over="raise", invalid="raise"):
            return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return np.column_stack((self._sigmoid(-scores), self._sigmoid(scores)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def log_loss(self, X, y):
        scores = self.decision_function(X)
        y = self._validate_y(y, len(scores))
        signed_scores = np.where(y == 1, -scores, scores)
        return float(np.mean(np.logaddexp(0.0, signed_scores)))
