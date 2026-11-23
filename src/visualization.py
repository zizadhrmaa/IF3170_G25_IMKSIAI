import heapq
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
import numpy as np

from logreg import LogisticRegression


def _edge_labels(model, node):
    name = model.feature_names_[node.feature]
    if node.threshold is not None:
        return ["%s <= %.4g" % (name, node.threshold), "%s > %.4g" % (name, node.threshold)]
    cats = model.categories_[node.feature]
    return ["%s = %s" % (name, cats[c]) for c in node.categories]


def plot_decision_tree(model, top_n=10, path=None, class_names=None, figsize=None, dpi=150, show=True):
    if not model.fitted_:
        raise RuntimeError("DecisionTree belum di-fit")
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n harus bilangan bulat positif")
    names = list(model.classes_) if class_names is None else list(class_names)
    if len(names) != len(model.classes_):
        raise ValueError("class_names harus sepanjang jumlah kelas")

    expanded = set()
    heap, counter = [], 0
    root = model.root_
    if not root.is_leaf:
        heapq.heappush(heap, (-root.total * root.gain, counter, id(root), root))
    while heap and len(expanded) < top_n:
        _, _, _, node = heapq.heappop(heap)
        expanded.add(id(node))
        for child in node.children:
            if not child.is_leaf:
                counter += 1
                heapq.heappush(heap, (-child.total * child.gain, counter, id(child), child))

    positions, edges = {}, []
    leaf_counter = [0]

    def layout(node):
        kids = node.children if id(node) in expanded else []
        if not kids:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
        else:
            labels = _edge_labels(model, node)
            xs = []
            for child, label in zip(kids, labels):
                xs.append(layout(child))
                edges.append((node, child, label))
            x = float(np.mean(xs))
        positions[id(node)] = (x, -node.depth)
        return x

    layout(root)
    nodes = {}
    stack = [root]
    while stack:
        node = stack.pop()
        nodes[id(node)] = node
        if id(node) in expanded:
            stack.extend(node.children)

    max_depth = max(node.depth for node in nodes.values())
    n_leaves = leaf_counter[0]
    if figsize is None:
        figsize = (max(8.0, 2.6 * n_leaves), max(5.0, 2.2 * (max_depth + 1)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    for parent, child, label in edges:
        x0, y0 = positions[id(parent)]
        x1, y1 = positions[id(child)]
        ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=1, zorder=1)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=7, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5), zorder=2)
    palette = ["#cfe8ff", "#ffd6cc", "#d9f2d0", "#fff1b8", "#e6d9f5", "#d9d9d9"]
    for key, node in nodes.items():
        x, y = positions[key]
        p = node.dist / node.dist.sum()
        k = int(np.argmax(p))
        lines = []
        if key in expanded:
            lines.append(model.feature_names_[node.feature])
        lines.append("n = %.0f" % node.total)
        lines.append("%s: %.1f%%" % (names[k], 100 * p[k]))
        if key not in expanded and not node.is_leaf:
            lines.append("(dipangkas tampilan)")
        ax.text(x, y, "\n".join(lines), fontsize=8, ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=palette[k % len(palette)], edgecolor="#333333"))
    ax.set_xlim(-0.8, max(n_leaves - 1, 0) + 0.8)
    ax.set_ylim(-max_depth - 0.6, 0.6)
    ax.set_title("Decision Tree (top-%d percabangan)" % len(expanded))
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _make_writer(path, fps):
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if path.lower().endswith(".mp4") and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps), path
    return animation.PillowWriter(fps=fps), os.path.splitext(path)[0] + ".gif"


def animate_logreg_training(X, y, feature_index=0, path="output/visualization/logreg/logreg_training.mp4",
                            penalty="l2", alpha=0.01, l1_ratio=0.5, lr=0.5, max_iter=300,
                            optimizer="gd", class_weight=None, max_frames=120, fps=15,
                            grid_size=80, feature_name=None, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or not 0 <= feature_index < X.shape[1]:
        raise ValueError("feature_index di luar jumlah fitur X")
    x = X[:, [feature_index]]
    model = LogisticRegression(penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, lr=lr, max_iter=max_iter,
                               optimizer=optimizer, class_weight=class_weight, record_history=True, seed=seed)
    model.fit(x, y)
    trail = model.history_
    theta0, theta1 = trail[:, 0], trail[:, 1]

    def axis_range(values):
        low, high = float(values.min()), float(values.max())
        pad = max(0.5 * (high - low), 1.0)
        return np.linspace(low - pad, high + pad, grid_size)

    g0, g1 = axis_range(theta0), axis_range(theta1)
    yb = (y == model.classes_[1]).astype(float)
    w = model.sample_weights(y)
    w = w / w.sum()
    surface = np.empty((len(g1), len(g0)))
    for i, a in enumerate(g1):
        z = g0[:, None] + a * x[None, :, 0]
        surface[i] = (np.logaddexp(0.0, z) - yb[None, :] * z) @ w + model.penalty_value([a])

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contour(g0, g1, surface, levels=25, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="objective (log-loss + penalti)")
    ax.set_xlabel("θ0 (intercept)")
    ax.set_ylabel("θ1 (koefisien %s)" % (feature_name or ("fitur %d" % feature_index)))
    line, = ax.plot([], [], color="red", linewidth=1.5)
    point, = ax.plot([], [], "o", color="red")
    ax.plot(theta0[0], theta1[0], "s", color="black", label="awal")
    ax.legend(loc="upper right")
    title = ax.set_title("")

    frame_ids = np.unique(np.linspace(0, len(trail) - 1, min(max_frames, len(trail))).astype(int))

    def update(frame):
        k = frame_ids[frame]
        line.set_data(theta0[:k + 1], theta1[:k + 1])
        point.set_data([theta0[k]], [theta1[k]])
        title.set_text("Epoch %d | objective = %.5f" % (k, model.loss_history_[k]))
        return line, point, title

    anim = animation.FuncAnimation(fig, update, frames=len(frame_ids), blit=False)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer, final_path = _make_writer(path, fps)
    anim.save(final_path, writer=writer, dpi=110)
    plt.close(fig)
    return final_path, model


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


def plot_svm_decision_boundary(model, X2d, y, ax=None, title=None, path=None, show=True,
                               grid_resolution=300, figsize=(7, 6), dpi=150,
                               class_names=None, feature_names=("Fitur 1", "Fitur 2")):
    if not model.fitted_:
        raise RuntimeError("SVM belum di-fit")
    if model.n_features_in_ != 2:
        raise ValueError("Model SVM harus di-fit pada 2 fitur untuk visualisasi ini")
    X2d = np.asarray(X2d, dtype=float)
    y = np.asarray(y)
    if X2d.ndim != 2 or X2d.shape[1] != 2:
        raise ValueError("X2d harus 2 dimensi dengan 2 fitur")
    if len(y) != len(X2d):
        raise ValueError("Panjang y harus sama dengan X2d")

    x_min = float(X2d[:, 0].min()) - 0.5
    x_max = float(X2d[:, 0].max()) + 0.5
    y_min = float(X2d[:, 1].min()) - 0.5
    y_max = float(X2d[:, 1].max()) + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True
    else:
        fig = ax.figure

    ax.contourf(xx, yy, scores, levels=[-1e9, 0, 1e9],
                colors=["#fde0dd", "#c8e6c9"], alpha=0.35)
    ax.contour(xx, yy, scores, levels=[0], colors="black", linewidths=1.6)
    ax.contour(xx, yy, scores, levels=[-1, 1], colors="dimgray",
               linestyles="--", linewidths=1.0)

    labels = list(model.classes_) if class_names is None else list(class_names)
    markers = ["o", "s"]
    colors = ["#1f77b4", "#d62728"]
    for k, cls in enumerate(model.classes_):
        mask = y == cls
        if mask.any():
            ax.scatter(X2d[mask, 0], X2d[mask, 1], marker=markers[k % len(markers)],
                       c=colors[k % len(colors)], s=22, edgecolors="k",
                       linewidths=0.4, alpha=0.75, label=str(labels[k]))

    if len(model.support_vectors_) > 0 and model.support_vectors_.shape[1] == 2:
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none",
                   edgecolors="orange", linewidths=1.6, label="Support Vector")

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title if title is not None else f"SVM kernel={model.kernel}")
    ax.legend(loc="best", fontsize=8)

    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    elif created:
        plt.close(fig)
    return fig