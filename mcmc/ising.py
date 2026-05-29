"""
Ising simulator using Wolff cluster updates in arbitrary dimension.
"""

import numpy as np
import numba

from .mcmc import (
    lattice_simulation_kwargs_from_args,
    make_lattice_simulation_parser,
    run_lattice_simulation,
)


def critical_coupling_2d() -> float:
    """Critical inverse temperature for 2D Ising: β_c = (1/2) * ln(1 + sqrt(2))"""
    return 0.5 * np.log(1.0 + np.sqrt(2.0))


def critical_coupling_3d() -> float:
    """Critical inverse temperature for 3D Ising (numerical, cubic lattice)."""
    return 0.221654  # commonly cited value for 3D Ising


def get_critical_J(dim: int) -> float:
    """Critical coupling for Ising in dimension dim (2 or 3)."""
    if dim == 2:
        return critical_coupling_2d()
    if dim == 3:
        return critical_coupling_3d()
    raise ValueError(f"Critical coupling not implemented for dim={dim}")


@numba.njit
def _wolff_steps_numba(lat: np.ndarray, add_prob: float, seed: int, n_steps: int) -> None:
    """Wolff cluster updates in arbitrary dimension. lat is modified in place; shape is (L,)*D."""
    shape = lat.shape
    D = len(shape)
    N = lat.size
    L = shape[0]
    np.random.seed(seed)

    # Strides for flat <-> multi-index: flat = sum(coords[d] * stride[d])
    stride = np.empty(D, dtype=np.int64)
    stride[D - 1] = 1
    for d in range(D - 2, -1, -1):
        stride[d] = stride[d + 1] * shape[d + 1]

    lat_flat = lat.ravel()
    stack = np.empty(N, dtype=np.int32)
    coords = np.empty(D, dtype=np.int32)

    for _ in range(n_steps):
        f0 = np.random.randint(0, N)
        old_spin = lat_flat[f0]
        new_spin = -old_spin
        lat_flat[f0] = new_spin
        stack[0] = f0
        top = 1

        while top > 0:
            top -= 1
            f = stack[top]
            # Unravel f -> coords
            rest = f
            for d in range(D):
                coords[d] = rest // stride[d]
                rest = rest % stride[d]
            # Visit 2*D neighbors
            for d in range(D):
                for delta in (1, -1):
                    nd = (coords[d] + delta + L) % L
                    # neighbor flat index
                    nf = 0
                    for k in range(D):
                        ck = nd if k == d else coords[k]
                        nf += ck * stride[k]
                    if lat_flat[nf] == old_spin and np.random.random() < add_prob:
                        lat_flat[nf] = new_spin
                        stack[top] = nf
                        top += 1


def _add_prob_ising(J: float) -> float:
    return 1.0 - np.exp(-2.0 * J)


def _init_lattice_ising(L: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        2 * rng.integers(0, 2, size=(L,) * dim, dtype=np.int8) - 1,
        dtype=np.int8,
    )


def _main():
    parser = make_lattice_simulation_parser(description="Ising Monte Carlo: Wolff cluster updates, TFRecord output.")
    args = parser.parse_args()
    run_lattice_simulation(
        **lattice_simulation_kwargs_from_args(args),
        model_name="ising",
        get_critical_J=get_critical_J,
        add_prob_fn=_add_prob_ising,
        init_lattice_fn=_init_lattice_ising,
        cluster_steps_fn=_wolff_steps_numba,
    )


if __name__ == "__main__":
    _main()
