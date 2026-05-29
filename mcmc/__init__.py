"""
Monte Carlo samplers (Wolff cluster updates) for 2D/3D Ising and q=3 Potts.

Each model script can be invoked as a module from the repo root, e.g.

    python -m mcmc.ising  --L 40 --num-thermal 200 --num-cluster-moves 10 \
                          --num-measure 100000 --output-dir Data
    python -m mcmc.potts  --L 40 --num-thermal 200 --num-cluster-moves 10 \
                          --num-measure 100000 --output-dir Data

Samples are written as TFRecord shards under ``<output-dir>/<model>/L{L}/``
together with a ``config.json`` describing the run.
"""
from .mcmc import (
    generate_lattice_samples,
    lattice_simulation_kwargs_from_args,
    load_config,
    make_lattice_simulation_parser,
    run_lattice_simulation,
    save_config,
    serialize_sample_int8,
)

__all__ = [
    "generate_lattice_samples",
    "lattice_simulation_kwargs_from_args",
    "load_config",
    "make_lattice_simulation_parser",
    "run_lattice_simulation",
    "save_config",
    "serialize_sample_int8",
]
