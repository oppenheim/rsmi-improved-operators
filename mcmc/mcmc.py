"""
Shared I/O and run logic for lattice models (Ising, Potts) in arbitrary dimension:
TFRecord serialization, config, generic sample generation loop, and TFRecord writing loop.
"""

import json
import os
from typing import Callable, Iterator, Optional

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def _bytes_feature(v: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64_feature(v: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def serialize_sample_int8(sample: np.ndarray, L: int) -> bytes:
    """Serialize int8 lattice sample (shape (L,)*dim) to TFRecord Example bytes. Same schema for Ising and Potts."""
    ex = tf.train.Example(features=tf.train.Features(feature={
        "L": _int64_feature(L),
        "sample": _bytes_feature(sample.tobytes(order="C")),
    }))
    return ex.SerializeToString()


def save_config(output_dir: str, config: dict) -> None:
    """Write config dict to output_dir/config.json."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_config(output_dir: str) -> dict:
    """Load config from output_dir/config.json."""
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f)


# Type for cluster step function: (lat, add_prob, seed, n_steps) -> None; lat has shape (L,)*dim
ClusterStepsFn = Callable[[np.ndarray, float, int, int], None]


def generate_lattice_samples(
    L: int,
    num_thermal: int,
    num_cluster_moves: int,
    num_measure: int,
    J: float,
    seed: Optional[int] = None,
    initial_state: Optional[np.ndarray] = None,
    dim: int = 2,
    *,
    add_prob_fn: Callable[[float], float],
    init_lattice_fn: Callable[[int, int, np.random.Generator], np.ndarray],
    cluster_steps_fn: ClusterStepsFn,
) -> Iterator[np.ndarray]:
    """
    Generic thermalize-once then yield samples loop for lattice cluster algorithms in arbitrary dimension.
    Used by Ising and Potts with model-specific add_prob, init_lattice, and cluster_steps.
    """
    shape = (L,) * dim
    rng = np.random.default_rng(seed)
    add_prob = add_prob_fn(J)
    if initial_state is not None:
        lat = np.asarray(initial_state, dtype=np.int8).copy()
        if lat.shape != shape:
            raise ValueError(f"initial_state must have shape {shape}, got {lat.shape}")
    else:
        lat = init_lattice_fn(L, dim, rng)

    n_therm = num_thermal * num_cluster_moves
    seed_therm = int(rng.integers(0, 2**63))
    cluster_steps_fn(lat, add_prob, seed_therm, n_therm)

    n = 0
    while num_measure == -1 or n < num_measure:
        seed_step = int(rng.integers(0, 2**63))
        cluster_steps_fn(lat, add_prob, seed_step, num_cluster_moves)
        yield lat
        n += 1


def run_lattice_simulation(
    L: int,
    num_thermal: int,
    num_measure: int,
    num_cluster_moves: int,
    output_dir: str,
    records_per_file: int = 10000,
    J: Optional[float] = None,
    seed: Optional[int] = None,
    compression: Optional[str] = None,
    dim: int = 2,
    *,
    model_name: str,
    get_critical_J: Callable[[int], float],
    add_prob_fn: Callable[[float], float],
    init_lattice_fn: Callable[[int, int, np.random.Generator], np.ndarray],
    cluster_steps_fn: ClusterStepsFn,
) -> str:
    """
    Common run: create output_dir/model_name/L{L} subdir, write config, run generator with tqdm,
    write TFRecord shards, handle KeyboardInterrupt, remove incomplete shard. Model-specific bits
    passed via model_name, get_critical_J(dim), add_prob_fn, init_lattice_fn, cluster_steps_fn.
    The lattice dimension is recorded in config.json (no separate subdir per dim).
    """
    # output_dir / model_name / L{L}, e.g. output-dir/ising/L20
    output_dir = os.path.join(output_dir, model_name, f"L{L}")
    os.makedirs(output_dir, exist_ok=True)

    if J is None:
        J = get_critical_J(dim)
    J_val = float(J)

    config = {
        "L": L,
        "dim": dim,
        "num_thermal": num_thermal,
        "num_measure": num_measure,
        "num_cluster_moves": num_cluster_moves,
        "records_per_file": records_per_file,
        "J": J_val,
        "model": model_name,
        "format": "tfrecord",
        "compression": compression
    }
    save_config(output_dir, config)

    gen = generate_lattice_samples(
        L=L,
        num_thermal=num_thermal,
        num_cluster_moves=num_cluster_moves,
        num_measure=num_measure,
        J=J_val,
        seed=seed,
        dim=dim,
        add_prob_fn=add_prob_fn,
        init_lattice_fn=init_lattice_fn,
        cluster_steps_fn=cluster_steps_fn,
    )
    gen = tqdm(gen, total=num_measure, desc=f"Sampling at J={J_val:.4f}", leave=False)

    options = None
    suffix = ".tfrecord"
    if compression == "GZIP":
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        suffix = ".tfrecord.gz"

    file_idx = 0
    in_shard = 0
    writer = None

    def open_writer(idx: int):
        path = os.path.join(output_dir, f"{idx}{suffix}")
        return tf.io.TFRecordWriter(path, options=options)

    try:
        for sample in gen:
            if writer is None:
                writer = open_writer(file_idx)
            rec = serialize_sample_int8(sample, L)
            writer.write(rec)
            in_shard += 1
            if in_shard == records_per_file:
                writer.close()
                writer = None
                file_idx += 1
                in_shard = 0

        if writer is not None:
            writer.close()
    except KeyboardInterrupt:
        if writer is not None:
            writer.close()

    # Remove last (incomplete) shard so readers see only full records_per_file shards
    if in_shard > 0:
        last_path = os.path.join(output_dir, f"{file_idx}{suffix}")
        if os.path.isfile(last_path):
            os.remove(last_path)

    return output_dir


def make_lattice_simulation_parser(
    description: str = "Lattice simulation: write TFRecord shards to output_dir/<model>/d{dim}/L{L}.",
) -> "argparse.ArgumentParser":
    """Build an ArgumentParser with common options for run_lattice_simulation."""
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--L", type=int, required=True, help="Linear size per dimension")
    p.add_argument("--dim", type=int, default=2, help="Lattice dimension (default: 2)")
    p.add_argument("--num-thermal", type=int, required=True, help="Thermalization steps (each = num_cluster_moves)")
    p.add_argument("--num-measure", type=int, required=True, help="Number of samples to record")
    p.add_argument("--num-cluster-moves", type=int, required=True, help="Cluster moves between measurements")
    p.add_argument("--records-per-file", type=int, default=10000, help="Samples per TFRecord file")
    p.add_argument("--output-dir", "-o", type=str, required=True, help="Output root (writes to output_dir/<model>/L{L}, e.g. Data/ising/L20; lattice dim is stored in config.json)")
    p.add_argument("--J", type=float, default=None, help="Coupling (default: critical)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--gzip", action="store_true", help="Use GZIP compression for TFRecord files")
    return p


def lattice_simulation_kwargs_from_args(args) -> dict:
    """Convert parsed namespace from make_lattice_simulation_parser() to kwargs for run_lattice_simulation."""
    return {
        "L": args.L,
        "dim": getattr(args, "dim", 2),
        "num_thermal": args.num_thermal,
        "num_measure": args.num_measure,
        "num_cluster_moves": args.num_cluster_moves,
        "output_dir": args.output_dir,
        "records_per_file": args.records_per_file,
        "J": args.J,
        "seed": args.seed,
        "compression": "GZIP" if getattr(args, "gzip", False) else None,
    }