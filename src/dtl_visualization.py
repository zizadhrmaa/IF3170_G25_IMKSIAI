"""Visualisasi pohon keputusan untuk implementasi DecisionTree di dtl.py."""

import heapq
import os

import matplotlib.pyplot as plt
import numpy as np


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
