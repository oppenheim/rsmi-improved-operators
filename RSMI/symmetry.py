"""
Symmetry utilities for RSMI.

The rest of RSMI expects permutations as np.ndarray of shape (P, D), where P is the
number of permutation maps and D is the number of sites (e.g. spins). Each row is
a permutation: permutation[i] gives the index in the original configuration that
maps to position i under that symmetry.
"""

import itertools
import numpy as np


def _find_transformation_rule(list1, list2):
    """Indices such that list1[i] == list2[rule[i]]. Returns list of ints.
    Raises ValueError if any transformed point is not in list2 (symmetry does not map set to itself)."""
    list1 = [tuple(int(x) for x in p) for p in list1]
    list2 = [tuple(int(x) for x in p) for p in list2]
    index_map = {list2[i]: i for i in range(len(list2))}
    try:
        return [index_map[idx] for idx in list1]
    except KeyError as e:
        raise ValueError(
            "Symmetry does not map the index set to itself. "
            "For even×even shapes (e.g. 2×2 square) use half_integer_center=True."
        ) from e


def hyperoctahedral_permutations(indices, center, dim, L=None, half_integer_center=False):
    """
    Build the 2^dim * dim! permutations for the hyperoctahedral group (signed permutations
    of axes). Single generic implementation for all dimensions and shapes.
    Use half_integer_center=True for even×even 2D regions so the set is invariant.

    Parameters
    ----------
    indices : list of tuples of length dim
        Site coordinates (e.g. lattice positions).
    center : tuple of length dim
        Center of rotation/reflection. If half_integer_center=True, geometric center is center + 0.5 per axis.
    dim : int
        Lattice dimension.
    L : int, optional
        If provided, transformed coordinates are taken mod L (torus). If None, no wrapping.
    half_integer_center : bool
        If True, use geometric center at center + 0.5 per axis (for even×even regions).

    Returns
    -------
    np.ndarray of shape (2^dim * dim!, len(indices))
        Permutation array. Row p gives the image of each site under the p-th symmetry.
    """
    indices = list(indices)
    indices = [tuple(int(x) for x in idx) for idx in indices]
    center = tuple(int(c) for c in center)
    offset = 0.5 if half_integer_center else 0.0
    perms = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product((1, -1), repeat=dim):
            transformed = []
            for idx in indices:
                new_idx = tuple(
                    center[i] + offset + signs[i] * (idx[perm[i]] - center[perm[i]] - offset)
                    for i in range(dim)
                )
                new_idx = tuple(int(round(x)) for x in new_idx)
                if L is not None:
                    new_idx = tuple(new_idx[i] % L for i in range(dim))
                transformed.append(new_idx)
            rule = _find_transformation_rule(transformed, indices)
            perms.append(rule)
    return np.stack(perms, axis=0).astype(np.int32)


def identity_permutation(D):
    """
    No symmetry: single identity permutation.

    Returns
    -------
    np.ndarray of shape (1, D)
    """
    return np.arange(D, dtype=np.int32).reshape(1, -1)
