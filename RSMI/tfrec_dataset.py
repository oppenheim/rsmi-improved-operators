import os
import json
import random

import tensorflow as tf
from typing import Optional, Tuple, Union


def _parse_example(example_proto: tf.Tensor, L: int, dim: int) -> tf.Tensor:
    feature_spec = {
        "sample": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    raw = ex["sample"]
    x = tf.io.decode_raw(raw, out_type=tf.int8)
    x = tf.reshape(x, [L] * dim)
    x = tf.cast(x, tf.float32)
    return x


def _build_single_pipeline(
    files: list,
    L: int,
    dim: int,
    compression: Optional[str],
    batch_size: int,
    shuffle_files: bool,
    shuffle_samples: bool,
    shuffle_buffer_size: int,
    repeat: bool,
    seed: Optional[int],
) -> tf.data.Dataset:
    """Build one dataset from a list of TFRecord files."""
    file_ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files and len(files) > 1:
        file_ds = file_ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)

    ds = tf.data.TFRecordDataset(
        file_ds,
        compression_type=("GZIP" if compression == "GZIP" else None),
        num_parallel_reads=tf.data.AUTOTUNE,
    )

    if shuffle_samples:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda ex: _parse_example(ex, L, dim), num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_tfrecord_dataset(
    data_dir: str,
    batch_size: int,
    shuffle_files: bool = True,
    shuffle_samples: bool = True,
    shuffle_buffer_size: int = 10000,
    repeat: bool = False,
    seed: Optional[int] = None,
    train_frac: Optional[float] = None,
) -> Union[Tuple[tf.data.Dataset, int, int, int], Tuple[tf.data.Dataset, tf.data.Dataset, int, int, int, int]]:
    """
    Build TFRecord dataset(s) from data_dir.

    Parameters
    ----------
    data_dir : str
        Directory containing config.json and TFRecord shards.
    batch_size : int
        Batch size.
    shuffle_files : bool
        Shuffle file order (each pipeline).
    shuffle_samples : bool
        Shuffle samples within the dataset.
    shuffle_buffer_size : int
        Buffer size for sample shuffling.
    repeat : bool
        If True, repeat dataset indefinitely.
    seed : int, optional
        Random seed for shuffling.
    train_frac : float, optional
        If set (e.g. 0.8), split by files into train (first train_frac) and test (rest).
        Returns (train_ds, test_ds, L, dim, train_size, test_size). Split is by file (no skip).

    Returns
    -------
    If train_frac is None: (ds, L, dim, n_samples). n_samples = len(files) * records_per_file.
    If train_frac is set: (train_ds, test_ds, L, dim, train_size, test_size).
    """
    with open(os.path.join(data_dir, "config.json")) as f:
        cfg = json.load(f)
    L = int(cfg["L"])
    dim = int(cfg.get("dim", 2))
    compression = cfg["compression"]
    records_per_file = int(cfg["records_per_file"])

    pattern = "*.tfrecord.gz" if compression == "GZIP" else "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(data_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord shards matching {pattern} in {data_dir}")

    if train_frac is None:
        ds = _build_single_pipeline(
            files, L, dim, compression, batch_size,
            shuffle_files, shuffle_samples, shuffle_buffer_size, repeat, seed,
        )
        n_samples = len(files) * records_per_file
        return ds, L, dim, n_samples

    # Split by files so train and test pipelines are independent (no slow skip)
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0, 1)")
    shuffled = list(files)
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_frac))
    train_files = shuffled[:n_train]
    test_files = shuffled[n_train:]
    if not test_files:
        raise ValueError(
            "train_frac leaves no file for test; use fewer files or a smaller train_frac"
        )

    train_ds = _build_single_pipeline(
        train_files, L, dim, compression, batch_size,
        shuffle_files, shuffle_samples, shuffle_buffer_size, repeat, seed,
    )
    test_ds = _build_single_pipeline(
        test_files, L, dim, compression, batch_size,
        shuffle_files, shuffle_samples, shuffle_buffer_size, False, seed,
    )
    train_size = len(train_files) * records_per_file
    test_size = len(test_files) * records_per_file
    return train_ds, test_ds, L, dim, train_size, test_size