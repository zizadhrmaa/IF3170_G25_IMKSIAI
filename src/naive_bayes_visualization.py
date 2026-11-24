import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


def _mutual_information(table):
    table = np.asarray(table, dtype=float)
    total = table.sum()
    if total <= 0:
        return 0.0
    joint = table / total
    row = joint.sum(axis=1, keepdims=True)
    col = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / (row @ col)[mask])).sum())


def plot_naive_bayes_graph(model, feature_names=None, X=None, top_pairs=0, path=None,
                           class_names=None, figsize=(11, 9), dpi=150, show=True):
    if not model.fitted_:
        raise RuntimeError("NaiveBayes belum di-fit")
    d = model.n_features_in_
    feature_names = ["x%d" % j for j in range(d)] if feature_names is None else list(feature_names)
    if len(feature_names) != d:
        raise ValueError("feature_names harus sepanjang jumlah fitur")
    class_label = "Kelas" if class_names is None else "/".join(str(c) for c in class_names)
    mi = np.array([_mutual_information(model.category_count_[j].T) for j in range(d)])
    angles = np.linspace(0, 2 * np.pi, d, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 3.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    if X is not None and top_pairs > 0:
        X = np.asarray(X, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError("X harus berukuran (n, jumlah fitur)")
        pairs = []
        for a in range(d):
            for b in range(a + 1, d):
                table = np.zeros((model.n_categories_[a], model.n_categories_[b]))
                np.add.at(table, (X[:, a], X[:, b]), 1)
                pairs.append((_mutual_information(table), a, b))
        pairs.sort(reverse=True)
        for value, a, b in pairs[:top_pairs]:
            ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], linestyle="--", color="gray",
                    linewidth=0.8, alpha=0.7, zorder=1)
            ax.text((pos[a, 0] + pos[b, 0]) / 2, (pos[a, 1] + pos[b, 1]) / 2, "%.3f" % value,
                    fontsize=6, color="gray", ha="center", va="center")
    scale = 6.0 / mi.max() if mi.max() > 0 else 1.0
    for j in range(d):
        arrow = FancyArrowPatch((0, 0), pos[j], arrowstyle="-|>", mutation_scale=12,
                                linewidth=0.5 + mi[j] * scale, color="#1f77b4", alpha=0.8,
                                shrinkA=22, shrinkB=18, zorder=2)
        ax.add_patch(arrow)
        ax.text(pos[j, 0], pos[j, 1], "%s\nMI=%.4f" % (feature_names[j], mi[j]), fontsize=7,
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1b8", edgecolor="#333333"))
    ax.text(0, 0, class_label, fontsize=10, ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffd6cc", edgecolor="#333333"))
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_title("Struktur Naive Bayes: kelas → fitur (tebal panah = informasi mutual)")
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
