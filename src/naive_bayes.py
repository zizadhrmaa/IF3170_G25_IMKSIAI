import pickle

import numpy as np


# Referensi: PPT di kelas.
class NaiveBayes:
    def __init__(self, alpha=1.0, min_categories=None, decision_threshold=None):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha harus nonnegatif dan finite")
        if decision_threshold is not None and (
                not np.isscalar(decision_threshold) or not np.isfinite(decision_threshold)
                or not 0 <= decision_threshold <= 1):
            raise ValueError("decision_threshold harus None atau angka antara 0 dan 1")
        self.alpha = float(alpha)
        self.min_categories = min_categories
        self.decision_threshold = decision_threshold
        self.fitted_ = False

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X harus 2 dimensi dengan minimal 1 fitur")
        if X.dtype.kind not in "biuf" or not np.isfinite(X).all():
            raise ValueError("X harus berisi kode kategori numerik finite")
        if np.any(X < 0) or np.any(X != np.floor(X)) or np.any(X >= 2**63):
            raise ValueError("Kode kategori harus integer nonnegatif dalam rentang int64")
        return X.astype(np.int64)

    def fit(self, X, y):
        self.fitted_ = False
        X = self._validate_X(X)
        y = np.asarray(y)
        if len(X) == 0 or y.ndim != 1 or len(y) != len(X):
            raise ValueError("Data latih tidak boleh kosong; y harus 1 dimensi sepanjang X")
        if y.dtype.kind not in "biufUS" or (y.dtype.kind in "f" and not np.isfinite(y).all()):
            raise ValueError("Label harus berupa angka finite atau string")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        if getattr(self, "decision_threshold", None) is not None and len(self.classes_) != 2:
            raise ValueError("decision_threshold hanya untuk klasifikasi dua kelas")
        self.n_features_in_ = X.shape[1]
        self.n_categories_ = X.max(axis=0) + 1
        if self.min_categories is not None:
            minimum = np.asarray(self.min_categories)
            if minimum.ndim == 0:
                minimum = np.full(self.n_features_in_, minimum)
            if (minimum.shape != (self.n_features_in_,) or minimum.dtype.kind not in "biuf"
                    or not np.isfinite(minimum).all() or np.any(minimum < 1)
                    or np.any(minimum != np.floor(minimum)) or np.any(minimum >= 2**63)):
                raise ValueError("min_categories harus integer positif scalar atau per fitur")
            self.n_categories_ = np.maximum(self.n_categories_, minimum.astype(np.int64))

        self.class_count_ = np.bincount(encoded_y, minlength=len(self.classes_)).astype(float)
        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        self.category_count_ = []
        self.feature_log_prob_ = []
        for column, count in enumerate(self.n_categories_):
            frequencies = np.zeros((len(self.classes_), count), dtype=float)
            np.add.at(frequencies, (encoded_y, X[:, column]), 1)
            self.category_count_.append(frequencies)
            # Laplace smoothing untuk model wajib (alpha=1).
            # Referensi: https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes
            with np.errstate(divide="ignore"):
                log_prob = (np.log(frequencies + self.alpha)
                            - np.log(self.class_count_[:, None] + self.alpha * count))
            self.feature_log_prob_.append(log_prob)
        self.fitted_ = True
        return self

    def predict_joint_log_proba(self, X):
        if not self.fitted_:
            raise RuntimeError("NaiveBayes belum di-fit")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Jumlah fitur berbeda dengan data latih")
        if np.any(X >= self.n_categories_):
            raise ValueError("Kode kategori di luar model; gunakan preprocessor hasil fit")
        scores = np.tile(self.class_log_prior_, (len(X), 1))
        for column, log_prob in enumerate(self.feature_log_prob_):
            scores += log_prob[:, X[:, column]].T
        return scores

    def predict_log_proba(self, X):
        scores = self.predict_joint_log_proba(X)
        maximum = scores.max(axis=1, keepdims=True)
        if np.any(~np.isfinite(maximum)):
            raise ValueError("Semua kelas memiliki probabilitas nol; gunakan alpha > 0")
        # Log-sum-exp untuk normalisasi tanpa underflow.
        shifted = scores - maximum
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        threshold = getattr(self, "decision_threshold", None)
        if threshold is not None:
            return self.classes_[(np.exp(log_proba[:, 1]) > threshold).astype(int)]
        return self.classes_[log_proba.argmax(axis=1)]

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("NaiveBayes belum di-fit")
        with open(path, "wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path):
        with open(path, "rb") as stream:
            return pickle.load(stream)


def particle_swarm_optimization(objective, bounds, initial_point, swarm_size=12,
                                iterations=12, inertia=0.7, cognitive=1.5,
                                social=1.5, seed=42):
    bounds = np.asarray(bounds, dtype=float)
    initial_point = np.asarray(initial_point, dtype=float)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("bounds harus berisi pasangan batas bawah dan atas yang valid")
    if (initial_point.shape != (len(bounds),) or not np.isfinite(initial_point).all()
            or np.any(initial_point < bounds[:, 0]) or np.any(initial_point > bounds[:, 1])):
        raise ValueError("initial_point harus berada dalam bounds")
    for name, value, minimum in [("swarm_size", swarm_size, 2), ("iterations", iterations, 1)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} harus integer minimal {minimum}")
    for name, value in [("inertia", inertia), ("cognitive", cognitive), ("social", social)]:
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} harus nonnegatif dan finite")

    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]
    max_velocity = 0.2 * (upper - lower)
    positions = rng.uniform(lower, upper, size=(swarm_size, len(bounds)))
    positions[0] = initial_point
    velocities = rng.uniform(-max_velocity, max_velocity, size=positions.shape)

    def evaluate(points):
        scores = np.array([float(objective(point.copy())) for point in points])
        if not np.isfinite(scores).all():
            raise ValueError("Objective harus menghasilkan skor finite")
        return scores

    personal_best = positions.copy()
    personal_scores = evaluate(positions)
    best_index = np.argmax(personal_scores)
    global_best = personal_best[best_index].copy()
    global_score = float(personal_scores[best_index])

    # PSO dengan inertia, personal best, dan global best; objective dimaksimalkan.
    # Referensi: https://doi.org/10.1109/ICNN.1995.488968
    # Referensi: https://doi.org/10.1007/BFb0040810
    for iteration in range(iterations):
        r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
        velocities = (inertia * velocities
                      + cognitive * r1 * (personal_best - positions)
                      + social * r2 * (global_best - positions))
        velocities = np.clip(velocities, -max_velocity, max_velocity)
        proposed = positions + velocities
        positions = np.clip(proposed, lower, upper)
        velocities[proposed != positions] = 0.0
        scores = evaluate(positions)
        improved = scores > personal_scores
        personal_best[improved] = positions[improved]
        personal_scores[improved] = scores[improved]
        best_index = np.argmax(personal_scores)
        if personal_scores[best_index] > global_score:
            global_best = personal_best[best_index].copy()
            global_score = float(personal_scores[best_index])
        print(f'Iterasi {iteration + 1}: skor terbaik = {global_score:.4f}', flush=True)
    return global_best, global_score
