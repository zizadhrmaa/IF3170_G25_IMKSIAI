import pickle
import warnings

import numpy as np
from scipy.optimize import minimize


class SVM:
    """
    SVM biner dengan formulasi dual. Masalah QP dual diselesaikan dengan SLSQP
    (D. Kraft, "A Software Package for Sequential Quadratic Programming," DFVLR-FB 88-28, 1988)
    melalui scipy.optimize.minimize.
    """

    VALID_KERNELS = ("linear", "polynomial", "gaussian")

    def __init__(self, C=1.0, kernel="linear", degree=3, gamma=None,
                 coef0=1.0, tol=1e-3, max_iter=10000, batch_size=256,
                 decision_threshold=0.0, class_weight=None):
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
        if not np.isfinite(decision_threshold):
            raise ValueError("decision threshold harus finite")
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
        self.decision_threshold = decision_threshold
        self.class_weight = dict(class_weight) if isinstance(class_weight, dict) else class_weight
        self.fitted_ = False

    def _kernel(self, X, Y, squared_norms=None):
        with np.errstate(over="ignore", invalid="ignore"):
            dot = X @ Y.T
            if self.kernel == "linear":
                result = dot
            elif self.kernel == "polynomial":
                result = (self.gamma_ * dot + self.coef0) ** self.degree
            else:
                if squared_norms is None:
                    squared_norms = (X ** 2).sum(axis=1)
                squared_distance = (squared_norms[:, None]
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
        weights = np.ones(len(X))
        class_weight = getattr(self, "class_weight", None)
        if class_weight == "balanced":
            for label in self.classes_:
                weights[y == label] = len(y) / (2 * np.sum(y == label))
        elif isinstance(class_weight, dict):
            for label, weight in class_weight.items():
                if label not in self.classes_:
                    raise ValueError("Label pada class_weight tidak ditemukan di y")
                weights[y == label] = weight
        self.C_ = self.C * weights
        if not np.isfinite(self.C_).all():
            raise ValueError("C dikali bobot kelas harus finite")
        self.n_features_in_ = X.shape[1]
        self.gamma_ = 1.0 / X.shape[1] if self.gamma is None else self.gamma

        n = len(X)
        K = self._kernel(X, X)
        Q = (labels[:, None] * labels[None, :]) * K

        def objective(alpha):
            return 0.5 * alpha @ (Q @ alpha) - alpha.sum()

        def gradient(alpha):
            return Q @ alpha - 1.0

        def eq_constraint(alpha):
            return float(alpha @ labels)

        def eq_gradient(alpha):
            return labels

        alpha0 = np.zeros(n)
        bounds = [(0.0, float(c)) for c in self.C_]
        constraints = [{"type": "eq", "fun": eq_constraint, "jac": eq_gradient}]

        # SLSQP (Kraft, 1988) untuk QP dual dengan constraint kesetaraan dan box constraint.
        result = minimize(
            objective, alpha0, jac=gradient, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )

        alpha = np.clip(result.x, 0.0, self.C_)
        f_raw = (alpha * labels) @ K

        free = np.flatnonzero((alpha > 1e-8) & (alpha < self.C_ - 1e-8))
        if len(free) > 0:
            b = float(np.mean(labels[free] - f_raw[free]))
        else:
            lower_candidates, upper_candidates = [], []
            at_C = np.flatnonzero(alpha >= self.C_ - 1e-8)
            for idx in at_C:
                if labels[idx] > 0:
                    upper_candidates.append(1.0 - f_raw[idx])
                else:
                    lower_candidates.append(-1.0 - f_raw[idx])
            at_zero = np.flatnonzero(alpha <= 1e-8)
            for idx in at_zero:
                if labels[idx] > 0:
                    lower_candidates.append(1.0 - f_raw[idx])
                else:
                    upper_candidates.append(-1.0 - f_raw[idx])
            if lower_candidates and upper_candidates:
                b = float((np.max(lower_candidates) + np.min(upper_candidates)) / 2.0)
            else:
                b = 0.0

        self.b_ = b
        self.alpha_ = alpha
        self.n_iter_ = int(getattr(result, "nit", 0))
        self.converged_ = bool(result.success)
        self.support_ = np.flatnonzero(alpha > 1e-8)
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = alpha[self.support_] * labels[self.support_]
        self.w_ = self.dual_coef_ @ self.support_vectors_ if self.kernel == "linear" else None
        self.fitted_ = True
        if not self.converged_:
            warnings.warn("SVM belum konvergen: " + str(result.message),
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
        if self.kernel == "linear" and self.w_ is not None:
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
        threshold = getattr(self, "decision_threshold", 0.0)
        return np.where(scores >= threshold, self.classes_[1], self.classes_[0])

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("SVM belum di-fit, panggil fit sebelum save")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def differential_evolution(objective, bounds, initial_point, population_size=8,
                           generations=4, seed=42):
    """
    Differential Evolution (DE/rand/1/bin) untuk optimasi hyperparameter SVM.

    Referensi: R. Storn and K. Price, "Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces," Journal of Global
    Optimization, vol. 11, no. 4, pp. 341-359, 1997.
    """
    bounds = np.asarray(bounds, dtype=float)
    initial_point = np.asarray(initial_point, dtype=float)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("bounds harus berisi pasangan batas bawah dan atas yang valid")
    if (initial_point.shape != (len(bounds),) or not np.isfinite(initial_point).all()
            or np.any(initial_point < bounds[:, 0]) or np.any(initial_point > bounds[:, 1])):
        raise ValueError("initial_point harus berada dalam bounds")
    if not isinstance(population_size, (int, np.integer)) or population_size < 4:
        raise ValueError("population_size harus bilangan bulat minimal 4")
    if not isinstance(generations, (int, np.integer)) or generations < 1:
        raise ValueError("generations harus bilangan bulat positif")
    rng = np.random.default_rng(seed)
    population = rng.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, len(bounds)))
    population[0] = initial_point
    scores = np.array([objective(point.copy()) for point in population], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Objective harus menghasilkan skor finite")

    for generation in range(generations):
        previous = population.copy()
        for i in range(population_size):
            others = np.delete(np.arange(population_size), i)
            a, b, c = rng.choice(others, size=3, replace=False)
            mutant = np.clip(previous[a] + 0.8 * (previous[b] - previous[c]),
                             bounds[:, 0], bounds[:, 1])
            crossover = rng.random(len(bounds)) < 0.7
            crossover[rng.integers(len(bounds))] = True
            trial = np.where(crossover, mutant, previous[i])
            trial_score = float(objective(trial.copy()))
            if not np.isfinite(trial_score):
                raise ValueError("Objective harus menghasilkan skor finite")
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score
        print(f'Generasi {generation + 1}: skor terbaik = {scores.max():.4f}', flush=True)
    best = np.argmax(scores)
    return population[best].copy(), float(scores[best])