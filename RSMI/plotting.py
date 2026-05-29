"""
Optional 2D circle plotting for RSMI (matplotlib). Use for visualizing spins/coefficients on a circle layout.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
_linear_key_re = re.compile(
    r"^\s*x(?P<i>\d+)\s*,\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)


def _is_linear_key(key):
    if isinstance(key, (list, tuple)):
        return False
    s = str(key)
    if _linear_key_re.match(s):
        return True
    return len(re.findall(r"x\d+", s)) == 1


def _parse_pair_key(key):
    if _is_linear_key(key):
        return None
    if isinstance(key, (list, tuple)) and len(key) >= 2:
        nums = []
        for item in (key[0], key[1]):
            if isinstance(item, (int, np.integer)):
                nums.append(int(item))
            else:
                m = re.search(r"\d+", str(item))
                if m:
                    nums.append(int(m.group(0)))
        return (nums[0], nums[1]) if len(nums) == 2 else None
    s = str(key)
    x_hits = re.findall(r"x(\d+)", s)
    if len(x_hits) >= 2:
        return int(x_hits[0]), int(x_hits[1])
    ints = re.findall(r"\d+", s)
    if len(ints) >= 2:
        return int(ints[0]), int(ints[1])
    return None


def build_segments_and_weights(
    out, V_indices, top_k=None, min_abs=None, dedup=True, dedup_agg="sum", index_base=0
):
    """Build line segments and weights from (feature_name, coef) list. Returns segments, weights, skipped."""
    n = len(V_indices)
    coords = np.asarray(V_indices, dtype=float)
    edge_to_w = {}
    skipped = 0
    for key, w in out:
        parsed = _parse_pair_key(key)
        if parsed is None:
            skipped += 1
            continue
        i, j = parsed[0] - index_base, parsed[1] - index_base
        if not (0 <= i < n and 0 <= j < n):
            skipped += 1
            continue
        if dedup:
            k = (i, j) if i <= j else (j, i)
            if k in edge_to_w:
                edge_to_w[k] = edge_to_w[k] + float(w) if dedup_agg == "sum" else max(edge_to_w[k], float(w), key=lambda x: abs(x))
            else:
                edge_to_w[k] = float(w)
        else:
            edge_to_w[(i, j)] = edge_to_w.get((i, j), 0.0) + float(w)
    if not edge_to_w:
        return np.zeros((0, 2, 2)), np.zeros((0,)), skipped
    pairs = np.array(list(edge_to_w.keys()), dtype=int)
    vals = np.array(list(edge_to_w.values()), dtype=float)
    order = np.argsort(np.abs(vals))[::-1]
    if top_k is not None:
        order = order[:top_k]
    if min_abs is not None:
        order = order[np.abs(vals[order]) >= min_abs]
    pairs, vals = pairs[order], vals[order]
    segments = np.array([[coords[i], coords[j]] for i, j in pairs], dtype=float)
    return segments, vals, skipped


def plot_circle_graph(
    out,
    V_indices,
    node_size=12,
    top_k=300,
    min_abs=None,
    cmap_name="coolwarm",
    background="white",
    figsize=(7, 7),
    dedup=True,
    dedup_agg="sum",
    index_base=0,
    fc="white",
    ec="black",
):
    """Plot coefficient graph on circle layout. out = list of (feature_name, coef). Returns fig, ax."""
    segments, weights, skipped = build_segments_and_weights(
        out, V_indices, top_k=top_k, min_abs=min_abs, dedup=dedup, dedup_agg=dedup_agg, index_base=index_base
    )
    if weights.size:
        order = np.argsort(np.abs(weights))
        segments, weights = segments[order], weights[order]
    vmax = np.max(np.abs(weights)) if weights.size else 1.0
    vmin = -vmax
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    lw = 0.5 + 2.5 * (np.abs(weights) / vmax) if weights.size and vmax > 0 else 0.5
    lc = LineCollection(segments, array=weights, cmap=cmap, norm=norm, linewidths=lw, alpha=0.85, zorder=1)
    coords = np.asarray(V_indices, dtype=float)
    x, y = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.set_facecolor(background)
    ax.set_facecolor(background)
    ax.scatter(x, y, s=node_size, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_collection(lc)
    c = coords.mean(axis=0)
    r = 1.414 * np.linalg.norm(coords - c, axis=1).mean()
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), lw=0.5, alpha=1, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale()
    fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def DrawCircle(v, cors, node_size=12, cmap_name="coolwarm", background="white", figsize=(7, 7)):
    """Scatter plot of values v on 2D coordinates cors. Returns fig, ax."""
    coords = np.asarray(cors, dtype=float)
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    v = np.asarray(v)
    vmax = np.max(np.abs(v)) if v.size else 1.0
    vmin = -vmax
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.set_facecolor(background)
    ax.set_facecolor(background)
    scatter = ax.scatter(x_coords, y_coords, c=v, cmap=cmap, norm=norm, s=node_size, edgecolor="black", zorder=2)
    c = coords.mean(axis=0)
    r = 1.414 * np.linalg.norm(coords - c, axis=1).mean()
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), lw=0.5, alpha=1, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale()
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    plt.show()
    return fig, ax
