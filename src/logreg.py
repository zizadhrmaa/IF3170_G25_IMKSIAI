"""
Regresi logistik biner dengan regularisasi L1, L2, dan Elastic Net, dioptimasi dengan
gradient descent atau Adam.

Referensi:
- Proximal gradient / soft-thresholding untuk L1:
  N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends in Optimization,
  vol. 1, no. 3, pp. 127-239, 2014.
- Adam:
  D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization,"
  in Proc. ICLR, 2015. arXiv:1412.6980.
"""

import pickle
import warnings

import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _soft_threshold(values, threshold):
    """
    Operator proximal untuk penalti L1 (soft-thresholding), lihat Parikh & Boyd (2014), Section 6.1.
    """
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


class LogisticRegression:
    VALID_PENALTIES = ("none", "l1", "l2", "elasticnet")
    VALID_OPTIMIZERS = ("gd", "adam")

    def __init__(self, penalty="l2", alpha=0.01, l1_ratio=0.5, lr=0.1, max_iter=500,
                 tol=1e-7, n_iter_no_change=5, batch_size=None, optimizer="gd",
                 beta1=0.9, beta2=0.999, epsilon=1e-8, class_weight=None,
                 decision_threshold=0.5, record_history=False, seed=42):
        if penalty not in self.VALID_PENALTIES:
            raise ValueError("penalty harus 'none', 'l1', 'l2', atau 'elasticnet'")
        if optimizer not in self.VALID_OPTIMIZERS:
            raise ValueError("optimizer harus 'gd' atau 'adam'")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha harus nonnegatif dan finite")
        if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio harus antara 0 dan 1")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr harus positif dan finite")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol harus nonnegatif dan finite")
        for name, value in [("max_iter", max_iter), ("n_iter_no_change", n_iter_no_change)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + " harus bilangan bulat positif")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size harus None atau bilangan bulat positif")
        for name, value in [("beta1", beta1), ("beta2", beta2)]:
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(name + " harus antara 0 dan 1")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon harus positif dan finite")
        if not np.isfinite(decision_threshold) or not 0 <= decision_threshold <= 1:
            raise ValueError("decision_threshold harus antara 0 dan 1")
        if class_weight is not None:
            if isinstance(class_weight, str):
                if class_weight != "balanced":
                    raise ValueError("class_weight harus None, 'balanced', atau dictionary")
            elif isinstance(class_weight, dict):
                for weight in class_weight.values():
                    if not np.isfinite(weight) or weight <= 0:
                        raise ValueError("Bobot kelas harus positif dan finite")
            else:
                raise ValueError("class_weight harus None, 'balanced', atau dictionary")
        self.penalty = penalty
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.lr = float(lr)
        self.max_iter = max_iter
        self.tol = float(tol)
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.class_weight = dict(class_weight) if isinstance(class_weight, dict) else class_weight
        self.decision_threshold = float(decision_threshold)
        self.record_history = record_history
        self.seed = seed
        self.fitted_ = False

    @staticmethod
    def alpha_from_C(C, n_samples):
        if not np.isfinite(C) or C <= 0 or n_samples < 1:
            raise ValueError("C harus positif dan n_samples minimal 1")
        return 1.0 / (C * n_samples)

    def _penalty_coefficients(self):
        if self.penalty == "l1":
            return self.alpha, 0.0
        if self.penalty == "l2":
            return 0.0, self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * self.l1_ratio, self.alpha * (1.0 - self.l1_ratio)
        return 0.0, 0.0

    def penalty_value(self, coef):
        l1, l2 = self._penalty_coefficients()
        coef = np.asarray(coef, dtype=float)
        return float(l1 * np.abs(coef).sum() + 0.5 * l2 * (coef ** 2).sum())

    def _weights_from_binary(self, yb, classes):
        weights = np.ones(len(yb))
        class_weight = getattr(self, "class_weight", None)
        if class_weight == "balanced":
            for k in (0, 1):
                count = np.sum(yb == k)
                if count > 0:
                    weights[yb == k] = len(yb) / (2.0 * count)
        elif isinstance(class_weight, dict):
            for label, weight in class_weight.items():
                if label not in classes:
                    raise ValueError("Label pada class_weight tidak ditemukan di y")
                weights[yb == (1.0 if label == classes[1] else 0.0)] = weight
        return weights

    def sample_weights(self, y):
        if not self.fitted_:
            raise RuntimeError("LogisticRegression belum di-fit")
        yb = (np.asarray(y) == self.classes_[1]).astype(float)
        return self._weights_from_binary(yb, self.classes_)

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] == 0:
            raise ValueError("X harus 2 dimensi dengan minimal 1 baris dan 1 fitur")
        if not np.isfinite(X).all():
            raise ValueError("X tidak boleh mengandung NaN atau infinity")
        return X

    @staticmethod
    def _loss(X, yb, w, coef, intercept):
        z = X @ coef + intercept
        return float((w * (np.logaddexp(0.0, z) - yb * z)).sum() / w.sum())

    def fit(self, X, y):
        self.fitted_ = False
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y harus 1 dimensi dengan panjang sama dengan jumlah baris X")
        if y.dtype.kind == "f" and not np.isfinite(y).all():
            raise ValueError("y tidak boleh mengandung NaN atau infinity")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LogisticRegression ini memerlukan tepat 2 kelas")
        yb = (y == classes[1]).astype(float)
        w = self._weights_from_binary(yb, classes)
        n, d = X.shape
        l1, l2 = self._penalty_coefficients()
        rng = np.random.default_rng(self.seed)
        batch = n if self.batch_size is None else min(int(self.batch_size), n)
        coef = np.zeros(d)
        intercept = 0.0
        m_w, v_w = np.zeros(d), np.zeros(d)
        m_b, v_b = 0.0, 0.0
        step_count = 0

        def objective():
            return self._loss(X, yb, w, coef, intercept) + self.penalty_value(coef)

        current = objective()
        loss_history = [current]
        history = [np.concatenate([[intercept], coef])] if self.record_history else None
        stalled = 0
        converged = False
        n_iter = 0
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(n) if batch < n else np.arange(n)
            for start in range(0, n, batch):
                rows = order[start:start + batch]
                Xb, yb_b, wb = X[rows], yb[rows], w[rows]
                residual = wb * (_sigmoid(Xb @ coef + intercept) - yb_b) / wb.sum()
                grad_w = Xb.T @ residual + l2 * coef
                grad_b = residual.sum()
                if self.optimizer == "gd":
                    coef = coef - self.lr * grad_w
                    intercept = intercept - self.lr * grad_b
                    if l1 > 0:
                        coef = _soft_threshold(coef, self.lr * l1)
                else:
                    # Adam dengan bias correction (Kingma & Ba, 2015), Algoritma 1.
                    step_count += 1
                    m_w = self.beta1 * m_w + (1 - self.beta1) * grad_w
                    v_w = self.beta2 * v_w + (1 - self.beta2) * grad_w ** 2
                    m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
                    v_b = self.beta2 * v_b + (1 - self.beta2) * grad_b ** 2
                    c1 = 1 - self.beta1 ** step_count
                    c2 = 1 - self.beta2 ** step_count
                    scale_w = self.lr / (np.sqrt(v_w / c2) + self.epsilon)
                    coef = coef - scale_w * (m_w / c1)
                    intercept = intercept - self.lr * (m_b / c1) / (np.sqrt(v_b / c2) + self.epsilon)
                    if l1 > 0:
                        coef = _soft_threshold(coef, scale_w * l1)
            n_iter = epoch
            previous, current = current, objective()
            if not np.isfinite(current):
                raise ValueError("Pelatihan divergen; turunkan lr atau standarisasi fitur")
            loss_history.append(current)
            if self.record_history:
                history.append(np.concatenate([[intercept], coef]))
            if abs(previous - current) <= self.tol:
                stalled += 1
                if stalled >= self.n_iter_no_change:
                    converged = True
                    break
            else:
                stalled = 0

        self.classes_ = classes
        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.n_features_in_ = d
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.loss_history_ = np.array(loss_history)
        self.history_ = np.array(history) if self.record_history else None
        self.fitted_ = True
        if not converged:
            warnings.warn("LogisticRegression belum konvergen, naikkan max_iter atau periksa lr",
                          RuntimeWarning, stacklevel=2)
        return self

    def objective(self, X, y, coef=None, intercept=None):
        if not self.fitted_:
            raise RuntimeError("LogisticRegression belum di-fit")
        X = self._validate_X(X)
        coef = self.coef_ if coef is None else np.asarray(coef, dtype=float)
        intercept = self.intercept_ if intercept is None else float(intercept)
        yb = (np.asarray(y) == self.classes_[1]).astype(float)
        w = self._weights_from_binary(yb, self.classes_)
        return self._loss(X, yb, w, coef, intercept) + self.penalty_value(coef)

    def decision_function(self, X):
        if not self.fitted_:
            raise RuntimeError("LogisticRegression belum di-fit")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Jumlah fitur berbeda dengan data latih")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return self.classes_[(p1 > self.decision_threshold).astype(int)]

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("LogisticRegression belum di-fit")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)