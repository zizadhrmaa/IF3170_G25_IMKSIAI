import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class LogisticRegression:

    def __init__(
        self, learning_rate=0.01, epochs=50, seed=42, *, penalty="none",
        reg_strength=0.01, l1_ratio=0.5, optimizer="sga", beta1=0.9,
        beta2=0.999, epsilon=1e-8,
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate harus positif dan finite")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs harus bilangan bulat positif")
        if penalty not in ("none", "l1", "l2", "elasticnet"):
            raise ValueError("penalty harus 'none', 'l1', 'l2', atau 'elasticnet'")
        if not np.isfinite(reg_strength) or reg_strength < 0:
            raise ValueError("reg_strength harus nonnegatif dan finite")
        if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio harus berada dalam interval [0, 1]")
        if optimizer not in ("sga", "adam"):
            raise ValueError("optimizer harus 'sga' atau 'adam'")
        if not np.isfinite(beta1) or not 0 <= beta1 < 1:
            raise ValueError("beta1 harus berada dalam interval [0, 1)")
        if not np.isfinite(beta2) or not 0 <= beta2 < 1:
            raise ValueError("beta2 harus berada dalam interval [0, 1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon harus positif dan finite")
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.seed = seed
        self.penalty = penalty
        self.reg_strength = float(reg_strength)
        self.l1_ratio = float(l1_ratio)
        self.optimizer = optimizer
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.classes_ = np.array([0, 1])
        self.coef_ = None
        self.intercept_ = 0.0
        self.theta_history_ = None
        self.loss_history_ = None

    def __setstate__(self, state):
        self.__dict__.update(optimizer="sga", beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.__dict__.update(state)

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

    def _penalty_strengths(self):
        """Mengembalikan pengali untuk sum(abs(w)) dan sum(w**2)/2."""
        # Bentuk penalti mengikuti Daksh Patel (2024), "Understanding L1, L2, and Elastic Net Regularization".
        # https://medium.com/@helloitsdaksh007/understanding-l1-l2-and-elastic-net-regularization-2ba1aadf3159
        if self.penalty == "l1":
            return self.reg_strength, 0.0
        if self.penalty == "l2":
            return 0.0, self.reg_strength
        if self.penalty == "elasticnet":
            return self.reg_strength * self.l1_ratio, self.reg_strength * (1 - self.l1_ratio)
        return 0.0, 0.0

    def fit(self, X, y):
        X = self._validate_X(X)
        y = self._validate_y(y, len(X))
        weights = np.zeros(X.shape[1])
        intercept = 0.0
        rng = np.random.default_rng(self.seed)
        l1_strength, l2_strength = self._penalty_strengths()
        l1_step = self.learning_rate * l1_strength
        l2_step = self.learning_rate * l2_strength
        use_adam = self.optimizer == "adam"
        if use_adam:
            gradient = np.empty(X.shape[1] + 1)
            first_moment = np.zeros_like(gradient)
            second_moment = np.zeros_like(gradient)
            update_count = 0

        theta_history = np.empty((self.epochs + 1, X.shape[1] + 1))
        loss_history = np.empty(self.epochs + 1)
        theta_history[0] = 0.0
        loss_history[0] = np.log(2.0)

        with np.errstate(over="raise", invalid="raise"):
            for epoch in range(1, self.epochs + 1):
                for i in rng.permutation(len(X)):
                    x_i = X[i]
                    score = x_i @ weights + intercept
                    p = self._sigmoid(score)

                    if use_adam:
                        # Gradien log-loss per sampel, ditambah kontribusi penalti pada bobot.
                        gradient[0] = p - y[i]
                        gradient[1:] = gradient[0] * x_i
                        if l1_strength:
                            gradient[1:] += l1_strength * np.sign(weights)
                        if l2_strength:
                            gradient[1:] += l2_strength * weights

                        # https://medium.com/to-cut-a-long-paper-short/adam-a-method-for-stochastic-optimization-8be4ad7800c7
                        update_count += 1
                        first_moment *= self.beta1
                        first_moment += (1 - self.beta1) * gradient
                        second_moment *= self.beta2
                        second_moment += (1 - self.beta2) * np.square(gradient)
                        m_hat = first_moment / (1 - self.beta1**update_count)
                        v_hat = second_moment / (1 - self.beta2**update_count)
                        adam_step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        intercept -= adam_step[0]
                        weights -= adam_step[1:]
                        continue

                    step = self.learning_rate * (y[i] - p)
                    weight_update = step * x_i

                    if l1_step:
                        weight_update -= l1_step * np.sign(weights)
                    if l2_step:
                        weight_update -= l2_step * weights
                    weights += weight_update
                    intercept += step

                theta_history[epoch, 0] = intercept
                theta_history[epoch, 1:] = weights
                loss_history[epoch] = self._log_loss_from_scores(X @ weights + intercept, y)

        self.coef_ = weights
        self.intercept_ = intercept
        self.theta_history_ = theta_history
        self.loss_history_ = loss_history
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

    @staticmethod
    def _log_loss_from_scores(scores, y):
        signed_scores = np.where(y == 1, -scores, scores)
        losses = np.logaddexp(0.0, signed_scores)
        largest = losses.max()
        if largest >= np.finfo(losses.dtype).max / losses.size:
            return float(largest * np.mean(losses / largest))
        return float(losses.mean())

    def log_loss(self, X, y):
        scores = self.decision_function(X)
        y = self._validate_y(y, len(scores))
        return self._log_loss_from_scores(scores, y)

    def objective(self, X, y):
        loss = self.log_loss(X, y)
        l1_strength, l2_strength = self._penalty_strengths()
        if l1_strength > 0:
            loss += l1_strength * np.abs(self.coef_).sum()
        if l2_strength > 0:
            loss += 0.5 * l2_strength * (self.coef_ @ self.coef_)
        return float(loss)

    def save(self, path):
        if self.coef_ is None:
            raise RuntimeError("Panggil fit sebelum menyimpan model")
        path = Path(path)
        with TemporaryDirectory(dir=path.parent) as tmp_dir:
            tmp_path = Path(tmp_dir) / "model.pkl"
            with tmp_path.open("wb") as file:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise ValueError("Berkas tidak berisi model LogisticRegression")
        if model.coef_ is None:
            raise ValueError("Model dalam berkas belum dilatih")
        return model