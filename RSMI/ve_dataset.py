import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List

from .symmetry import hyperoctahedral_permutations, identity_permutation


def _flat_indices_from_coords(
    coords: List[Tuple[int, ...]], shape: Tuple[int, ...]
) -> np.ndarray:
    """Convert list of D-dimensional index tuples to flat (C-order) indices."""
    if not coords:
        return np.array([], dtype=np.int32)
    arr = np.array(coords, dtype=np.int64)
    flat = np.ravel_multi_index(arr.T, shape, order="C")
    return np.asarray(flat, dtype=np.int32)


def _extract_hypercube_indices(
    center: Tuple[int, ...], side_length: int, L: int, dim: int
) -> List[Tuple[int, ...]]:
    """All lattice points in the axis-aligned box [center - side//2, center - side//2 + side) per axis, mod L."""
    lo = [center[d] - side_length // 2 for d in range(dim)]
    out = []

    def recurse(coord: List[int], d: int) -> None:
        if d == dim:
            out.append(tuple((coord[k] % L + L) % L for k in range(dim)))
            return
        for delta in range(side_length):
            coord.append(lo[d] + delta)
            recurse(coord, d + 1)
            coord.pop()

    recurse([], 0)
    return out


def _extract_hypercube_surface_indices(
    center: Tuple[int, ...], side_length: int, L: int, dim: int
) -> List[Tuple[int, ...]]:
    """Boundary of the hypercube: points on at least one face (one coord at min or max of box)."""
    lo = [center[d] - side_length // 2 for d in range(dim)]
    hi = [lo[d] + side_length - 1 for d in range(dim)]
    out_set = set()

    def recurse(coord: List[int], d: int, fix_axis: int, at_lo: bool) -> None:
        if d == dim:
            out_set.add(tuple((coord[k] % L + L) % L for k in range(dim)))
            return
        if d == fix_axis:
            coord.append(lo[d] if at_lo else hi[d])
            recurse(coord, d + 1, fix_axis, at_lo)
            coord.pop()
        else:
            for delta in range(side_length):
                coord.append(lo[d] + delta)
                recurse(coord, d + 1, fix_axis, at_lo)
                coord.pop()

    for axis in range(dim):
        for at_lo in (True, False):
            recurse([], 0, axis, at_lo)

    return sorted(out_set)




def _extract_hypersphere_indices(
    center: Tuple[int, ...], radius: int, L: int, dim: int
) -> List[Tuple[int, ...]]:
    """V region: lattice points in the ball (L2 distance <= radius)."""
    out = []
    rad_sq = (radius + 0.5) ** 2
    lo = [center[d] - radius for d in range(dim)]
    hi = [center[d] + radius for d in range(dim)]

    def recurse(coord: List[int], d: int) -> None:
        if d == dim:
            dist_sq = sum((coord[k] - center[k]) ** 2 for k in range(dim))
            if dist_sq <= rad_sq:
                out.append(tuple((coord[k] % L + L) % L for k in range(dim)))
            return
        for x in range(lo[d], hi[d] + 1):
            coord.append(x)
            recurse(coord, d + 1)
            coord.pop()

    recurse([], 0)
    out = sorted(set(out))
    return out


def _extract_hypersphere_surface_indices(
    center: Tuple[int, ...], radius: int, L: int, dim: int
) -> List[Tuple[int, ...]]:
    """E region: lattice points in the shell radius - 0.5 < d <= radius + 0.5."""
    out = []
    r_lo_sq = (radius - 0.5) ** 2
    r_hi_sq = (radius + 0.5) ** 2
    lo = [center[d] - radius - 1 for d in range(dim)]
    hi = [center[d] + radius + 1 for d in range(dim)]

    def recurse(coord: List[int], d: int) -> None:
        if d == dim:
            dist_sq = sum((coord[k] - center[k]) ** 2 for k in range(dim))
            if r_lo_sq < dist_sq <= r_hi_sq:
                out.append(tuple((coord[k] % L + L) % L for k in range(dim)))
            return
        for x in range(lo[d], hi[d] + 1):
            coord.append(x)
            recurse(coord, d + 1)
            coord.pop()

    recurse([], 0)
    out = sorted(set(out))
    return out


def build_ve_indices(
    shape: str,
    L: int,
    buf: int,
    size: Optional[int] = None,
    side_length: Optional[int] = None,
    radius: Optional[int] = None,
    dim: int = 2,
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], int, int, Tuple[int, ...], bool]:
    """
    Build V and E index lists for the given shape on a lattice of shape (L,) * dim.
    shape: "sphere" or "cube". buf: E is boundary of region expanded by buf.
    size: if set, used as radius for sphere and as side_length for cube (lets you switch shape without changing arg name).
    Returns V_indices, E_indices, V_size, E_size, center, half_integer_center.
    center and half_integer_center are chosen so symmetry (hyperoctahedral) maps the V set to itself.
    """
    if size is not None:
        if shape == "sphere":
            radius = size
        elif shape == "cube":
            side_length = size
    center = (L // 2,) * dim

    if shape == "cube":
        assert side_length is not None and radius is None
        V_indices = _extract_hypercube_indices(center, side_length, L, dim)
        side_e = side_length + 2 * buf
        E_indices = _extract_hypercube_surface_indices(center, side_e, L, dim)
    elif shape == "sphere":
        assert side_length is None and radius is not None
        V_indices = _extract_hypersphere_indices(center, radius, L, dim)
        r_e = radius + buf
        E_indices = _extract_hypersphere_surface_indices(center, r_e, L, dim)
    else:
        raise ValueError('shape must be "sphere" or "cube"')

    # Canonical order: lexicographic (index 0 most significant, then 1, ...)
    V_indices = sorted(V_indices)
    E_indices = sorted(E_indices)

    # Symmetry center: use bounding box of V. If extent is even in every axis, use half-integer center so hyperoctahedral group maps the set to itself.
    mins = [min(p[d] for p in V_indices) for d in range(dim)]
    maxs = [max(p[d] for p in V_indices) for d in range(dim)]
    extents = [maxs[d] - mins[d] + 1 for d in range(dim)]
    half_integer_center = all(e % 2 == 0 for e in extents)
    sym_center = tuple(mins) if half_integer_center else (L // 2,) * dim

    return V_indices, E_indices, len(V_indices), len(E_indices), sym_center, half_integer_center


def samples_to_ve_dataset(L, shape, buf, size=None, side_length=None, radius=None, use_symmetry=True, dim=2):
    """
    Compositional API: return V/E geometry and a pipeline function.

    size: if set, used as radius for sphere and as side_length for cube (switch shape without changing arg name).
    Returns (V_indices, E_indices, V_size, E_size, permutations, to_ve).
    Input dataset must yield batches of shape (batch_size,) + (L,) * dim.
    """
    V_indices_list, E_indices_list, V_size, E_size, center, half_integer_center = build_ve_indices(
        shape=shape, L=L, buf=buf, size=size, side_length=side_length, radius=radius, dim=dim
    )
    V_indices = np.array(V_indices_list, dtype=np.int32)
    E_indices = np.array(E_indices_list, dtype=np.int32)
    lattice_shape = (L,) * dim
    V_flat_idx = _flat_indices_from_coords(V_indices_list, lattice_shape)
    E_flat_idx = _flat_indices_from_coords(E_indices_list, lattice_shape)
    V_flat_idx_tf = tf.constant(V_flat_idx)
    E_flat_idx_tf = tf.constant(E_flat_idx)
    if use_symmetry:
            permutations = hyperoctahedral_permutations(
                V_indices_list, center, dim, L=L,
                half_integer_center=half_integer_center,
            )
    else:
        permutations = identity_permutation(V_size)

    def _map_fn(batch):
        flat = tf.reshape(batch, (tf.shape(batch)[0], -1))
        V = tf.gather(flat, V_flat_idx_tf, axis=1)
        E = tf.gather(flat, E_flat_idx_tf, axis=1)
        return (V, E)

    def to_ve(dataset):
        return dataset.map(
            _map_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    return V_indices, E_indices, V_size, E_size, permutations, to_ve