"""
Pencarian bobot fitur jarak KNN dengan coordinate search berbasis cross-validation.

Jarak Manhattan berbobot antara baris q dan x adalah D(q, x) = sum_g w_g * delta_g(q, x), dengan
delta_g jumlah selisih mutlak kolom milik fitur g setelah standarisasi (untuk fitur kategorikal,
delta_g bernilai 0 atau 2 karena dua kolom one-hot berbeda). Bobot w_g dicari satu per satu
(coordinate search): untuk setiap fitur dicoba beberapa nilai dari grid, dan nilai baru diterima hanya
jika Macro F1 cross-validation naik lebih dari min_gain. Putaran diulang sampai tidak ada perubahan.
Pendekatan wrapper ini mengikuti D. Wettschereck, D. W. Aha, dan T. Mohri, "A Review and Empirical
Evaluation of Feature Weighting Methods for a Class of Lazy Learning Algorithms", Artificial Intelligence
Review 11, 1997, hlm. 273-314, serta R. Kohavi dan G. H. John, "Wrappers for Feature Subset Selection",
Artificial Intelligence 97, 1997, hlm. 273-324.

Agar setiap evaluasi murah, untuk setiap baris validasi disimpan M kandidat tetangga terdekat menurut
bobot awal beserta delta_g setiap kandidat. Jarak dengan bobot baru cukup dihitung ulang sebagai
perkalian delta dengan w, lalu k tetangga terdekat dipilih dari M kandidat tersebut. Jika M jauh lebih
besar dari k, hasilnya sama dengan KNN penuh selama bobot tidak berubah drastis; hasil akhir tetap
diverifikasi dengan KNN penuh di notebook.
"""

import numpy as np
import pandas as pd

from knn import KNN
from metrics import macro_f1
from preprocessing import CATEGORICAL_COLS, Preprocessor


def feature_groups(preprocessor):
    """
    Memetakan setiap kolom hasil transform ke nama fitur asalnya.

    :param preprocessor: Preprocessor yang sudah di-fit
    :return: tuple (nama_grup, indeks_grup_per_kolom); nama grup sama dengan key feature_weights
    """
    names, index = [], []
    for col in preprocessor.feature_names_:
        group = col.split("=", 1)[0] if "=" in col else col
        if group not in names:
            names.append(group)
        index.append(names.index(group))
    return names, np.array(index)


class FeatureWeightSearch:
    """
    Coordinate search bobot fitur KNN (jarak Manhattan, voting 1/jarak^power) dengan
    cross-validation. Statistik preprocessing di-fit ulang pada setiap fold latih.
    """

    def __init__(self, k, power, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                 n_candidates=150, min_gain=0.0005, max_rounds=4):
        """
        :param k: jumlah tetangga yang ikut voting
        :param power: pangkat jarak pada voting 1/jarak^power
        :param grid: nilai bobot yang dicoba untuk setiap fitur
        :param n_candidates: jumlah kandidat tetangga M yang disimpan per baris validasi
        :param min_gain: kenaikan Macro F1 minimum agar bobot baru diterima
        :param max_rounds: batas jumlah putaran coordinate search
        """
        if n_candidates <= k:
            raise ValueError("n_candidates harus lebih besar dari k")
        self.k = k
        self.power = power
        self.grid = tuple(float(v) for v in grid)
        self.n_candidates = n_candidates
        self.min_gain = min_gain
        self.max_rounds = max_rounds

    def _build_cache(self, df, y, folds, initial_weights):
        """
        Menyimpan delta per fitur dan label untuk M kandidat tetangga setiap baris validasi.
        """
        n = len(df)
        deltas, labels = None, None
        for tr_idx, va_idx in folds:
            fold_tr, fold_va = df.iloc[tr_idx], df.iloc[va_idx]
            pp = Preprocessor(feature_weights=None).fit(fold_tr)
            X_tr, X_va = pp.transform(fold_tr), pp.transform(fold_va)
            names, group_of = feature_groups(pp)
            if deltas is None:
                self.feature_names_ = names
                deltas = np.zeros((n, self.n_candidates, len(names)), dtype=np.float32)
                labels = np.zeros((n, self.n_candidates), dtype=np.float32)
            col_w = np.array([initial_weights.get(names[g], 1.0) for g in group_of])
            searcher = KNN(k=self.n_candidates, p=1).fit(X_tr * col_w, y[tr_idx])
            _, cand = searcher.kneighbors(X_va * col_w)
            # Matriks 0/1 untuk menjumlahkan selisih kolom menjadi selisih per fitur.
            to_group = np.zeros((X_tr.shape[1], len(names)))
            to_group[np.arange(X_tr.shape[1]), group_of] = 1.0
            for s in range(0, len(va_idx), 500):
                rows = cand[s:s + 500]
                diff = np.abs(X_va[s:s + 500, None, :] - X_tr[rows])
                deltas[va_idx[s:s + 500]] = diff @ to_group
                labels[va_idx[s:s + 500]] = y[tr_idx][rows]
        return deltas, labels

    def _score(self, w):
        """
        :param w: bobot per fitur (urutan feature_names_)
        :return: Macro F1 out-of-fold dari k tetangga terdekat di antara kandidat
        """
        dist = self.deltas_ @ w.astype(np.float32)
        idx = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k]
        d = np.take_along_axis(dist, idx, axis=1).astype(np.float64)
        lab = np.take_along_axis(self.labels_, idx, axis=1)
        with np.errstate(divide="ignore"):
            vote = 1.0 / d ** self.power
        # Tetangga berjarak 0 menentukan sendiri hasil voting, sama seperti KNN.predict.
        exact = d == 0
        has_exact = exact.any(axis=1)
        vote[has_exact] = exact[has_exact].astype(float)
        proba = (vote * lab).sum(axis=1) / vote.sum(axis=1)
        return macro_f1(self.y_, (proba > 0.5).astype(int))

    def fit(self, df, y, folds, initial_weights):
        """
        :param df: DataFrame data latih (fitur mentah)
        :param y: label data latih
        :param folds: list tuple (train_idx, val_idx) cross-validation
        :param initial_weights: dict bobot awal dengan format feature_weights Preprocessor
        :return: objek ini sendiri; hasil ada di weights_, cv_score_, dan history_
        """
        self.y_ = np.asarray(y)
        self.deltas_, self.labels_ = self._build_cache(df.reset_index(drop=True), self.y_, list(folds), initial_weights)
        w = np.array([float(initial_weights.get(f, 1.0)) for f in self.feature_names_])
        best = self._score(w)
        self.initial_score_ = best
        history = [{"putaran": 0, "fitur": "(bobot awal)", "bobot_lama": None, "bobot_baru": None, "cv_macro_f1": best}]
        for rnd in range(1, self.max_rounds + 1):
            changed = False
            for j, feature in enumerate(self.feature_names_):
                best_v = w[j]
                for v in self.grid:
                    if v == w[j]:
                        continue
                    trial = w.copy()
                    trial[j] = v
                    s = self._score(trial)
                    if s > best + self.min_gain:
                        best, best_v = s, v
                if best_v != w[j]:
                    history.append({"putaran": rnd, "fitur": feature, "bobot_lama": w[j], "bobot_baru": best_v,
                                    "cv_macro_f1": best})
                    w[j] = best_v
                    changed = True
            if not changed:
                break
        self.cv_score_ = best
        self.weights_ = {f: float(v) for f, v in zip(self.feature_names_, w)}
        self.history_ = pd.DataFrame(history)
        # Cache cukup besar; tidak perlu disimpan setelah pencarian selesai.
        del self.deltas_, self.labels_
        return self
