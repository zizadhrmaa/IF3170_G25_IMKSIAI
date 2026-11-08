"""
Metrik evaluasi klasifikasi dengan NumPy. Leaderboard Kaggle memakai Macro F1, yaitu rata-rata
F1 setiap kelas tanpa bobot, sehingga kelas minoritas berpengaruh sama besar dengan kelas mayoritas.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, labels=(0, 1)):
    """
    :param y_true: label asli
    :param y_pred: label prediksi
    :param labels: urutan label untuk baris dan kolom
    :return: matriks berukuran (n_label, n_label); baris = label asli, kolom = label prediksi
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            cm[i, j] = np.sum((y_true == t) & (y_pred == p))
    return cm


def f1_per_class(y_true, y_pred, labels=(0, 1)):
    """
    :param y_true: label asli
    :param y_pred: label prediksi
    :param labels: label yang dihitung F1-nya
    :return: F1 setiap label; bernilai 0 jika precision dan recall label tersebut sama-sama 0
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0)
    true_count = cm.sum(axis=1)

    # Label yang tidak pernah diprediksi atau tidak pernah muncul diberi nilai 0, bukan NaN.
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(y_true, y_pred, labels=(0, 1)):
    """
    :param y_true: label asli
    :param y_pred: label prediksi
    :param labels: label yang dirata-ratakan
    :return: rata-rata F1 semua label
    """
    return float(f1_per_class(y_true, y_pred, labels).mean())


def accuracy(y_true, y_pred):
    """
    :param y_true: label asli
    :param y_pred: label prediksi
    :return: proporsi prediksi yang benar
    """
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def stratified_kfold_indices(y, n_splits=5, seed=42):
    """
    Membagi indeks data menjadi n_splits fold dengan proporsi kelas yang sama di setiap fold.

    :param y: label seluruh data
    :param n_splits: jumlah fold
    :param seed: seed pengacakan agar pembagian fold selalu sama
    :return: generator tuple (train_idx, val_idx) untuk setiap fold
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    # Setiap kelas diacak dan dibagi rata ke semua fold secara terpisah.
    for c in np.unique(y):
        idx_c = rng.permutation(np.flatnonzero(y == c))
        for i, part in enumerate(np.array_split(idx_c, n_splits)):
            folds[i].extend(part.tolist())

    all_idx = np.arange(len(y))
    for i in range(n_splits):
        val_idx = np.sort(np.array(folds[i]))
        train_idx = np.setdiff1d(all_idx, val_idx)
        yield train_idx, val_idx
