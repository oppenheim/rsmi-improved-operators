"""
q=3 Potts simulator using Wolff-style cluster updates in arbitrary dimension.
"""

import numpy as np
import numba

from .mcmc import (
    lattice_simulation_kwargs_from_args,
    make_lattice_simulation_parser,
    run_lattice_simulation,
)


def critical_coupling_potts_2d() -> float:
    """Critical inverse temperature for 2D q=3 Potts: J_c = ln(1+√3)."""
    return float(np.log(1.0 + np.sqrt(3.0)))


def critical_coupling_potts_3d() -> float:
    """Critical inverse temperature for 3D q=3 Potts (numerical, cubic lattice)."""
    return 0.550565  # 3D ferromagnetic 3-state Potts, high-precision estimate


def get_critical_J(dim: int) -> float:
    """Critical coupling for q=3 Potts in dimension dim (2 or 3)."""
    if dim == 2:
        return critical_coupling_potts_2d()
    if dim == 3:
        return critical_coupling_potts_3d()
    raise ValueError(f"Critical coupling not implemented for dim={dim}")


@numba.njit
def _potts_cluster_steps(lat: np.ndarray, add_prob: float, seed: int, n_steps: int) -> None:
    """Potts cluster updates in arbitrary dimension. lat is modified in place; shape is (L,)*D."""
    shape = lat.shape
    D = len(shape)
    N = lat.size
    L = shape[0]
    np.random.seed(seed)

    stride = np.empty(D, dtype=np.int64)
    stride[D - 1] = 1
    for d in range(D - 2, -1, -1):
        stride[d] = stride[d + 1] * shape[d + 1]

    lat_flat = lat.ravel()
    stack = np.empty(N, dtype=np.int32)
    coords = np.empty(D, dtype=np.int32)

    for _ in range(n_steps):
        f0 = np.random.randint(0, N)
        old_state = lat_flat[f0]
        r = np.random.random()
        if old_state == 0:
            new_state = np.int8(1) if r < 0.5 else np.int8(2)
        elif old_state == 1:
            new_state = np.int8(0) if r < 0.5 else np.int8(2)
        else:
            new_state = np.int8(0) if r < 0.5 else np.int8(1)
        lat_flat[f0] = new_state
        stack[0] = f0
        top = 1

        while top > 0:
            top -= 1
            f = stack[top]
            rest = f
            for d in range(D):
                coords[d] = rest // stride[d]
                rest = rest % stride[d]
            for d in range(D):
                for delta in (1, -1):
                    nd = (coords[d] + delta + L) % L
                    nf = 0
                    for k in range(D):
                        ck = nd if k == d else coords[k]
                        nf += ck * stride[k]
                    if lat_flat[nf] == old_state and np.random.random() < add_prob:
                        lat_flat[nf] = new_state
                        stack[top] = nf
                        top += 1


def _add_prob_potts(J: float) -> float:
    return 1.0 - np.exp(-float(J))


def _init_lattice_potts(L: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        rng.integers(0, 3, size=(L,) * dim, dtype=np.int8),
        dtype=np.int8,
    )


def _main():
    parser = make_lattice_simulation_parser(description="q=3 Potts Monte Carlo: cluster updates, TFRecord output.")
    args = parser.parse_args()
    run_lattice_simulation(
        **lattice_simulation_kwargs_from_args(args),
        model_name="potts",
        get_critical_J=get_critical_J,
        add_prob_fn=_add_prob_potts,
        init_lattice_fn=_init_lattice_potts,
        cluster_steps_fn=_potts_cluster_steps,
    )


if __name__ == "__main__":
    _main()
