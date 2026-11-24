"""
Decision Tree Learning (C4.5) from scratch dengan penanganan missing value
secara fractional propagation.

Referensi:
- C4.5 dan gain ratio:
  J. R. Quinlan, "C4.5: Programs for Machine Learning," Morgan Kaufmann, 1993.
- Penalti threshold numerik:
  J. R. Quinlan, "Improved Use of Continuous Attributes in C4.5,"
  Journal of Artificial Intelligence Research, vol. 4, pp. 77-90, 1996.
"""

import pickle

import numpy as np

class _Node:
    def __init__(self, dist, total, depth):
        self.dist = dist
        self.total = total
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.categories = None
        self.lookup = None
        self.children = []
        self.child_fractions = None
        self.gain = 0.0

    @property
    def is_leaf(self):
        return len(self.children) == 0

def _entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, counts / total, 0.0)
        logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=-1)

def _missing_mask(col):
    if col.dtype.kind == "f":
        return np.isnan(col)
    if col.dtype.kind in "biu":
        return np.zeros(len(col), dtype=bool)
    return np.array([v is None or str(v) in ("nan", "<NA>", "NaT", "None") for v in col], dtype=bool)

class DecisionTree:
    MAX_DEPTH_LIMIT = 400

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_gain=1e-7, ccp_alpha=0.0, max_thresholds=128,
                 numeric_penalty=True, decision_threshold=None,
                 categorical_features=None):
        """
        :param numeric_penalty: penalti kecil pada gain split numerik, -log2(n_distinct-1)/total,
                                sebagai regularisasi terhadap split numerik beresolusi tinggi.
                                Terinspirasi MDL correction Quinlan (1996) untuk atribut kontinu,
                                bukan formula identik.
        """
        if max_depth is not None and (isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 1):
            raise ValueError("max_depth harus None atau bilangan bulat positif")
        if not np.isfinite(min_samples_split) or min_samples_split < 2:
            raise ValueError("min_samples_split harus minimal 2")
        if not np.isfinite(min_samples_leaf) or min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf harus positif")
        if not np.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain harus nonnegatif dan finite")
        if not np.isfinite(ccp_alpha) or ccp_alpha < 0:
            raise ValueError("ccp_alpha harus nonnegatif dan finite")
        if isinstance(max_thresholds, bool) or not isinstance(max_thresholds, (int, np.integer)) or max_thresholds < 1:
            raise ValueError("max_thresholds harus bilangan bulat positif")
        if decision_threshold is not None and (not np.isfinite(decision_threshold) or not 0 <= decision_threshold <= 1):
            raise ValueError("decision_threshold harus None atau angka antara 0 dan 1")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.ccp_alpha = ccp_alpha
        self.max_thresholds = max_thresholds
        self.numeric_penalty = numeric_penalty
        self.decision_threshold = decision_threshold
        self.categorical_features = None if categorical_features is None else list(categorical_features)
        self.fitted_ = False

    @staticmethod
    def _columns(X):
        if hasattr(X, "columns"):
            names = [str(c) for c in X.columns]
            cols = [X.iloc[:, j].to_numpy() for j in range(X.shape[1])]
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X harus 2 dimensi")
            names = ["x%d" % j for j in range(arr.shape[1])]
            cols = [arr[:, j] for j in range(arr.shape[1])]
        if len(cols) == 0:
            raise ValueError("X harus memiliki minimal 1 fitur")
        return names, cols

    def _forced_categorical(self, names):
        forced = set()
        for item in (self.categorical_features or []):
            if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                forced.add(int(item))
            elif item in names:
                forced.add(names.index(item))
            else:
                raise ValueError("categorical_features tidak dikenal: " + str(item))
        return forced

    def _encode(self, X, fit):
        if not fit and hasattr(X, "columns"):
            try:
                X = X[self.feature_names_]
            except KeyError:
                raise ValueError("Kolom X berbeda dengan data latih")
        names, cols = self._columns(X)
        n = len(cols[0])
        if fit:
            self.feature_names_ = names
            forced = self._forced_categorical(names)
            self.is_categorical_ = np.zeros(len(names), dtype=bool)
            self.categories_ = [None] * len(names)
            for j, col in enumerate(cols):
                if j in forced:
                    self.is_categorical_[j] = True
                elif col.dtype.kind not in "biuf":
                    try:
                        col.astype(float)
                    except (ValueError, TypeError):
                        self.is_categorical_[j] = True
        elif len(names) != len(self.feature_names_):
            raise ValueError("Jumlah fitur berbeda dengan data latih")
        Z = np.full((n, len(cols)), np.nan)
        for j, col in enumerate(cols):
            miss = _missing_mask(col)
            if self.is_categorical_[j]:
                text = col.astype(str)
                if fit:
                    self.categories_[j] = np.unique(text[~miss])
                cats = self.categories_[j]
                if len(cats) == 0:
                    continue
                pos = np.minimum(np.searchsorted(cats, text), len(cats) - 1)
                valid = (~miss) & (cats[pos] == text)
                Z[valid, j] = pos[valid]
            else:
                try:
                    values = col.astype(float)
                except (ValueError, TypeError):
                    raise ValueError("Kolom numerik '" + self.feature_names_[j] + "' berisi nilai non-numerik")
                values[miss] = np.nan
                Z[:, j] = values
        if np.isinf(Z).any():
            raise ValueError("X tidak boleh mengandung infinity")
        return Z

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = False
        Z = self._encode(X, fit=True)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(Z) or len(Z) == 0:
            raise ValueError("y harus 1 dimensi dengan panjang sama dengan jumlah baris X")
        if y.dtype.kind == "f" and not np.isfinite(y).all():
            raise ValueError("y tidak boleh mengandung NaN atau infinity")
        self.classes_, yenc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if sample_weight is None:
            w = np.ones(len(Z))
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != (len(Z),) or not np.isfinite(w).all() or np.any(w <= 0):
                raise ValueError("sample_weight harus positif, finite, dan sepanjang y")
        self.n_features_in_ = Z.shape[1]
        self._Z = Z
        self._y = yenc
        try:
            self.root_ = self._build(np.arange(len(Z)), w, 0)
            if self.ccp_alpha > 0:
                self._prune(self.root_, self.root_.total)
        finally:
            del self._Z
            del self._y
        importances = np.zeros(self.n_features_in_)
        stack = [self.root_]
        while stack:
            node = stack.pop()
            if not node.is_leaf:
                importances[node.feature] += node.total * node.gain
                stack.extend(node.children)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        self.fitted_ = True
        return self

    def _build(self, idx, w, depth):
        y = self._y[idx]
        dist = np.bincount(y, weights=w, minlength=self.n_classes_)
        total = float(w.sum())
        node = _Node(dist, total, depth)
        limit = self.MAX_DEPTH_LIMIT if self.max_depth is None else min(self.max_depth, self.MAX_DEPTH_LIMIT)
        if depth >= limit or total < self.min_samples_split or np.count_nonzero(dist > 1e-12) <= 1:
            return node
        split = self._best_split(idx, y, w, total)
        if split is None:
            return node
        feature, gain, extra = split
        col = self._Z[idx, feature]
        miss = np.isnan(col)
        with np.errstate(invalid="ignore"):
            if self.is_categorical_[feature]:
                groups = [(col == c) & ~miss for c in extra]
                node.categories = [int(c) for c in extra]
                lookup = np.full(len(self.categories_[feature]), -1, dtype=np.int64)
                for k, c in enumerate(extra):
                    lookup[int(c)] = k
                node.lookup = lookup
            else:
                groups = [(col <= extra) & ~miss, (col > extra) & ~miss]
                node.threshold = float(extra)
        known_weights = np.array([w[g].sum() for g in groups])
        fractions = known_weights / known_weights.sum()
        node.feature = int(feature)
        node.gain = float(gain)
        node.child_fractions = fractions
        for g, fraction in zip(groups, fractions):
            child_idx = np.concatenate([idx[g], idx[miss]])
            child_w = np.concatenate([w[g], w[miss] * fraction])
            node.children.append(self._build(child_idx, child_w, depth + 1))
        return node

    def _best_split(self, idx, y, w, total):
        candidates = []
        for f in range(self.n_features_in_):
            col = self._Z[idx, f]
            known = ~np.isnan(col)
            if not known.any():
                continue
            v, yk, wk = col[known], y[known], w[known]
            if self.is_categorical_[f]:
                result = self._categorical_split(v, yk, wk, total, len(self.categories_[f]))
            else:
                result = self._numeric_split(v, yk, wk, total)
            if result is None:
                continue
            gain, ratio, extra = result
            if gain > self.min_gain:
                candidates.append((gain, ratio, f, extra))
        if not candidates:
            return None
        average = np.mean([c[0] for c in candidates])
        eligible = [c for c in candidates if c[0] >= average - 1e-12]
        best = max(eligible, key=lambda c: (c[1], c[0]))
        return best[2], best[0], best[3]

    def _numeric_split(self, v, y, w, total):
        order = np.argsort(v, kind="stable")
        sv, sy, sw = v[order], y[order], w[order]
        boundaries = np.flatnonzero(sv[1:] > sv[:-1])
        if len(boundaries) == 0:
            return None
        n_distinct = len(boundaries) + 1
        if len(boundaries) > self.max_thresholds:
            chosen = np.unique(np.linspace(0, len(boundaries) - 1, self.max_thresholds).astype(np.int64))
            boundaries = boundaries[chosen]
        onehot = np.zeros((len(sv), self.n_classes_))
        onehot[np.arange(len(sv)), sy] = sw
        cumulative = np.cumsum(onehot, axis=0)
        all_counts = cumulative[-1]
        left = cumulative[boundaries]
        right = all_counts - left
        left_w, right_w = left.sum(axis=1), right.sum(axis=1)
        ok = (left_w >= self.min_samples_leaf) & (right_w >= self.min_samples_leaf)
        if not ok.any():
            return None
        known_total = float(sw.sum())
        info_after = (left_w * _entropy(left) + right_w * _entropy(right)) / known_total
        gain = (known_total / total) * (_entropy(all_counts) - info_after)
        if self.numeric_penalty and n_distinct > 2:
            gain = gain - np.log2(n_distinct - 1) / total
        gain = np.where(ok, gain, -np.inf)
        b = int(np.argmax(gain))
        position = boundaries[b]
        threshold = (sv[position] + sv[position + 1]) / 2.0
        if threshold >= sv[position + 1]:
            threshold = sv[position]
        parts = np.array([left_w[b], right_w[b], total - known_total])
        split_info = float(_entropy(parts))
        ratio = gain[b] / split_info if split_info > 1e-12 else 0.0
        return float(gain[b]), float(ratio), float(threshold)

    def _categorical_split(self, v, y, w, total, n_categories):
        table = np.zeros((n_categories, self.n_classes_))
        np.add.at(table, (v.astype(np.int64), y), w)
        branch_w = table.sum(axis=1)
        present = np.flatnonzero(branch_w > 0)
        if len(present) < 2 or np.count_nonzero(branch_w >= self.min_samples_leaf) < 2:
            return None
        known_total = float(branch_w.sum())
        info_after = float((branch_w[present] * _entropy(table[present])).sum() / known_total)
        gain = (known_total / total) * (float(_entropy(table.sum(axis=0))) - info_after)
        parts = np.append(branch_w[present], total - known_total)
        split_info = float(_entropy(parts))
        ratio = gain / split_info if split_info > 1e-12 else 0.0
        return float(gain), float(ratio), present

    def _prune(self, node, root_total):
        # Catatan: pruning di sini adalah cost-complexity bottom-up (weakest-link),
        # bukan pessimistic error pruning C4.5 standar. Parameter ccp_alpha mengikuti
        # konvensi scikit-learn; subtree dipangkas jika alpha <= ccp_alpha.
        if node.is_leaf:
            return (node.total / root_total) * float(_entropy(node.dist)), 1
        cost, leaves = 0.0, 0
        for child in node.children:
            c, l = self._prune(child, root_total)
            cost += c
            leaves += l
        leaf_cost = (node.total / root_total) * float(_entropy(node.dist))
        alpha = (leaf_cost - cost) / (leaves - 1)
        if alpha <= self.ccp_alpha:
            node.children = []
            node.feature = None
            node.threshold = None
            node.categories = None
            node.lookup = None
            node.child_fractions = None
            node.gain = 0.0
            return leaf_cost, 1
        return cost, leaves

    def predict_proba(self, X):
        if not self.fitted_:
            raise RuntimeError("DecisionTree belum di-fit")
        Z = self._encode(X, fit=False)
        out = np.zeros((len(Z), self.n_classes_))
        stack = [(self.root_, np.arange(len(Z)), np.ones(len(Z)))]
        while stack:
            node, idx, w = stack.pop()
            if node.is_leaf:
                out[idx] += w[:, None] * (node.dist / node.dist.sum())
                continue
            col = Z[idx, node.feature]
            miss = np.isnan(col)
            if node.threshold is not None:
                with np.errstate(invalid="ignore"):
                    branch = np.where(col <= node.threshold, 0, 1)
            else:
                codes = np.where(miss, 0, col).astype(np.int64)
                branch = np.where(miss, -1, node.lookup[codes])
                miss = miss | (branch < 0)
            known = ~miss
            any_missing = miss.any()
            for c, child in enumerate(node.children):
                sel = known & (branch == c)
                parts_idx, parts_w = [idx[sel]], [w[sel]]
                if any_missing:
                    parts_idx.append(idx[miss])
                    parts_w.append(w[miss] * node.child_fractions[c])
                child_idx = np.concatenate(parts_idx)
                if len(child_idx):
                    stack.append((child, child_idx, np.concatenate(parts_w)))
        return out

    def predict(self, X):
        proba = self.predict_proba(X)
        threshold = getattr(self, "decision_threshold", None)
        if threshold is not None:
            if self.n_classes_ != 2:
                raise ValueError("decision_threshold hanya untuk klasifikasi dua kelas")
            return self.classes_[(proba[:, 1] > threshold).astype(int)]
        return self.classes_[proba.argmax(axis=1)]

    def get_depth(self):
        if not self.fitted_:
            raise RuntimeError("DecisionTree belum di-fit")
        best, stack = 0, [self.root_]
        while stack:
            node = stack.pop()
            best = max(best, node.depth)
            stack.extend(node.children)
        return best

    def get_n_leaves(self):
        if not self.fitted_:
            raise RuntimeError("DecisionTree belum di-fit")
        count, stack = 0, [self.root_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                count += 1
            stack.extend(node.children)
        return count

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("DecisionTree belum di-fit")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

def genetic_algorithm(objective, bounds, initial_point, population_size=12, generations=8,
                      crossover_alpha=0.3, mutation_rate=0.2, mutation_scale=0.1,
                      tournament_size=3, elite=2, seed=42):
    """
    Genetic algorithm untuk optimasi hyperparameter model.

    Referensi:
    - J. H. Holland, "Adaptation in Natural and Artificial Systems", University of Michigan Press, 1975.
    - D. E. Goldberg, "Genetic Algorithms in Search, Optimization, and Machine Learning", Addison-Wesley, 1989.
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
    if not isinstance(elite, (int, np.integer)) or not 0 <= elite < population_size:
        raise ValueError("elite harus bilangan bulat antara 0 dan population_size - 1")
    if not isinstance(tournament_size, (int, np.integer)) or tournament_size < 1:
        raise ValueError("tournament_size harus bilangan bulat positif")
    for name, value in [("crossover_alpha", crossover_alpha), ("mutation_rate", mutation_rate),
                        ("mutation_scale", mutation_scale)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(name + " harus nonnegatif dan finite")
    if mutation_rate > 1:
        raise ValueError("mutation_rate tidak boleh lebih dari 1")

    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]

    def evaluate(points):
        scores = np.array([float(objective(p.copy())) for p in points])
        if not np.isfinite(scores).all():
            raise ValueError("Objective harus menghasilkan skor finite")
        return scores

    population = rng.uniform(lower, upper, size=(population_size, len(bounds)))
    population[0] = initial_point
    scores = evaluate(population)

    def tournament():
        contenders = rng.choice(population_size, size=min(tournament_size, population_size), replace=False)
        return population[contenders[np.argmax(scores[contenders])]]

    for generation in range(generations):
        order = np.argsort(-scores, kind="stable")
        next_population = [population[i].copy() for i in order[:elite]]
        next_scores = [scores[i] for i in order[:elite]]
        while len(next_population) < population_size:
            p1, p2 = tournament(), tournament()
            low, high = np.minimum(p1, p2), np.maximum(p1, p2)
            spread = high - low
            child = rng.uniform(low - crossover_alpha * spread, high + crossover_alpha * spread)
            mutate = rng.random(len(bounds)) < mutation_rate
            child = child + mutate * rng.normal(0.0, mutation_scale * (upper - lower))
            child = np.clip(child, lower, upper)
            next_population.append(child)
            next_scores.append(float(objective(child.copy())))
            if not np.isfinite(next_scores[-1]):
                raise ValueError("Objective harus menghasilkan skor finite")
        population = np.array(next_population)
        scores = np.array(next_scores, dtype=float)
        print(f'Generasi {generation + 1}: skor terbaik = {scores.max():.4f}', flush=True)
    best = int(np.argmax(scores))
    return population[best].copy(), float(scores[best])