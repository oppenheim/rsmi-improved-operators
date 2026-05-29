import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from scipy import stats
from tqdm.auto import tqdm
from .tfrec_dataset import build_tfrecord_dataset

def nearest_neighbors_hamming_one(indices):
    """
    Given a list of index tuples (x1, x2, ..., x_dim) of arbitrary dimension,
    return a list of index pairs (i, j) such that:
      - They differ in exactly one coordinate,
      - The difference in that coordinate is exactly 1,
      - All other coordinates are the same.
    """
    result = []
    n = len(indices)

    for i in range(n):
        if indices[i] is None:
            continue
        a = indices[i]
        for j in range(i + 1, n):
            if indices[j] is None:
                continue
            b = indices[j]
            if len(a) != len(b):
                continue
            diffs = [k for k in range(len(a)) if a[k] != b[k]]
            if len(diffs) != 1:
                continue
            k = diffs[0]
            if abs(a[k] - b[k]) == 1:
                result.append((i, j))
    return result
                
def bin_bootstrap_analysis(data,min_sample_size=128,func_boot=None,n_boot=1000):
    assert((func_boot!=None or np.shape(data)[0]==1) and data.ndim == 2)
    data_size=np.shape(data)[1]
    chopped_data_size=2**np.floor(np.log2(data_size))
    chopped_data=data[:,int(data_size-chopped_data_size):]
    if func_boot == None:
        stds=[np.std(chopped_data,ddof=1)/np.sqrt(chopped_data_size)]
    else:
        es = stats.bootstrap(chopped_data,func_boot,n_resamples=n_boot, vectorized=False, paired=True,method="basic")
        stds=[es.standard_error]

    bin_size=2
    while min_sample_size < int(chopped_data_size/bin_size):
        length_bin=int(chopped_data_size/bin_size)
        binned=np.reshape(chopped_data,(-1,length_bin,bin_size))
        mean_binned = np.mean(binned,axis=2)
        if func_boot == None:
            std_bin=np.std(mean_binned,ddof=1)/np.sqrt(length_bin)
        else:
            es = stats.bootstrap(mean_binned.tolist(),func_boot,n_resamples=n_boot, vectorized=False, paired=True,method="basic")
            std_bin = es.standard_error
        bin_size=bin_size*2
        stds.append(std_bin)
    return stds

def create_shifted_grid(indices, L, dim=2):
    """
    For each cell in an L^dim grid, shift all index tuples by that cell (mod L).
    indices: list of length-dim tuples (e.g. 2D (x,y) or 3D (x,y,z)).
    Returns array of shape (L,)*dim + (len(indices),) + (dim,).
    """
    n_sites = len(indices)
    indices_arr = np.array(indices, dtype=np.int64)  # (n_sites, dim)
    # Build grid of shifts: (L,)*dim + (dim,) -> each point is a shift vector
    grid_shape = (L,) * dim + (dim,)
    # Iterate over all L^dim shift cells
    out = np.zeros((L,) * dim + (n_sites,) + (dim,), dtype=np.int64)
    for flat_idx in range(L ** dim):
        shift = np.array(np.unravel_index(flat_idx, (L,) * dim), dtype=np.int64)
        # shifted = (indices_arr + shift) % L  shape (n_sites, dim)
        shifted = (indices_arr + shift) % L
        # Write into out at position shift (as tuple)
        out[tuple(shift) + (slice(None),)] = shifted
    return out    

def LogConnectedCorrelation(corr,mean,ratio=2):
    return 0.5*np.log(np.abs(np.mean(corr)-np.mean(mean)**2))/np.log(ratio)

def ConnectedCorrelation(corr,mean):
    return np.mean(corr)-np.mean(mean)**2

def CalculateCorrelationWithError(corr_accum, mean_accum, bin_counts, half_L, dim=2):
    """Calculate connected correlation at antipodal index (half_L,) * dim."""
    antipodal_tuple = (slice(None),) + (half_L,) * dim
    (corr_avg, mean_avg) = (((corr_accum.T) / bin_counts).T, ((mean_accum.T) / bin_counts).T)
    val = np.mean(corr_avg[antipodal_tuple]) - np.abs(np.mean(mean_avg)) ** 2
    log_err = bin_bootstrap_analysis(np.vstack((corr_avg[antipodal_tuple], mean_avg)), func_boot=LogConnectedCorrelation)[-1]
    err = bin_bootstrap_analysis(np.vstack((corr_avg[antipodal_tuple], mean_avg)), func_boot=ConnectedCorrelation)[-1]
    return val, err, log_err



def _log_ratio_connected_correlation(corr1, mean1, corr2, mean2):
    """Scaling dimension from ratio of connected correlations: 0.5 * log(C1/C2) / log(ratio)."""
    c1 = np.mean(corr1) - np.mean(mean1) ** 2
    c2 = np.mean(corr2) - np.mean(mean2) ** 2
    return 0.5 * np.log2(np.abs(c1) / np.abs(c2)) 


def _correlation_ratio_with_error(
    corr_accum1, mean_accum1, bin_counts1, half_L1,
    corr_accum2, mean_accum2, bin_counts2, half_L2,
    n_boot=1024,
    dim=2,
):
    """
    Scaling dimension from L1 vs L2 antipodal connected correlations (L2 = 2*L1).
    Antipodal at (half_L,) * dim. Returns (delta, log_err).
    """
    assert half_L2 > half_L1 and half_L2 % half_L1 == 0
    antipodal_tuple1 = (slice(None),) + (half_L1,) * dim
    antipodal_tuple2 = (slice(None),) + (half_L2,) * dim
    (corr_avg1, mean_avg1) = (((corr_accum1.T) / bin_counts1).T, ((mean_accum1.T) / bin_counts1).T)
    (corr_avg2, mean_avg2) = (((corr_accum2.T) / bin_counts2).T, ((mean_accum2.T) / bin_counts2).T)
    val1 = np.mean(corr_avg1[antipodal_tuple1]) - np.abs(np.mean(mean_avg1)) ** 2
    val2 = np.mean(corr_avg2[antipodal_tuple2]) - np.abs(np.mean(mean_avg2)) ** 2
    delta = 0.5 * np.log2(np.abs(val1) / np.abs(val2))
    data = np.vstack((
        corr_avg1[antipodal_tuple1],
        mean_avg1,
        corr_avg2[antipodal_tuple2],
        mean_avg2,
    ))
    log_err = bin_bootstrap_analysis(
        data,
        n_boot=n_boot,
        func_boot=lambda c1, m1, c2, m2: _log_ratio_connected_correlation(c1, m1, c2, m2),
    )[-1]
    return delta, log_err


def scaling_dimensions_from_correlations(
    corr_dict,
    n_boot=1024,
    dim=None,
):
    """If dim is None, infer from first corr_accum shape (corr_accum.ndim - 1)."""
    Ls = sorted(corr_dict.keys())
    if not Ls:
        return {}
    if dim is None:
        corr_accum0 = corr_dict[Ls[0]][0]
        dim = corr_accum0.ndim - 1
    out = {}
    for L in Ls:
        L2 = 2 * L
        if L2 not in corr_dict:
            continue
        corr_accum1, mean_accum1, bin_counts1 = corr_dict[L]
        corr_accum2, mean_accum2, bin_counts2 = corr_dict[L2]
        half_L1 = L // 2
        half_L2 = L2 // 2
        delta, log_err = _correlation_ratio_with_error(
            corr_accum1, mean_accum1, bin_counts1, half_L1,
            corr_accum2, mean_accum2, bin_counts2, half_L2,
            n_boot=n_boot,
            dim=dim,
        )
        out[L] = {"delta": float(delta), "err": float(log_err)}
    return out


def _correlation_fft(b, L, dim):
    """Apply FFT-based correlation: b has shape (batch_size,) + (L,)*dim. Returns same shape."""
    x = tf.cast(b, dtype=tf.complex64)
    if dim == 2:
        r = tf.signal.fft2d(x)
        corr = tf.signal.ifft2d(r * tf.math.conj(r))
    elif dim == 3:
        r = tf.signal.fft3d(x)
        corr = tf.signal.ifft3d(r * tf.math.conj(r))
    else:
        raise ValueError("connected_correlation supports only dim=2 or dim=3 (FFT)")
    return corr


@tf.function
def CalculateCorrelationForOperator(model, V, batch_size, L, dim=2):
    """Evaluate the operator on V and return the correlations and the means.
    V has shape (batch_size,) + (L,)*dim + (V_size,). Output correlation shape (L,)*dim."""
    v_size = tf.shape(V)[-1]
    if dim == 2:
        reshape_tuple = (batch_size, L, L)
    else:
        reshape_tuple = (batch_size, L, L, L)
    b = tf.reshape(model(tf.reshape(V, (-1, v_size))), reshape_tuple)
    denom = tf.cast(tf.cast(L, tf.float32) ** dim, tf.complex64)
    corr = _correlation_fft(b, L, dim)
    return tf.math.real(tf.reduce_mean(corr, axis=0) / denom), tf.reduce_mean(b)


def _sample_to_V(s, trans_tf, dim):
    """Extract V from lattice samples s using trans_tf. s: (batch_size,) + (L,)*dim.
    trans_tf: (L,)*dim + (n_sites, dim). Returns V (batch_size,) + (L,)*dim + (n_sites,)."""
    # Put batch last so gather_nd indexes spatial dims: (L,)*dim + (batch_size,)
    perm_to_batch_last = list(range(1, dim + 1)) + [0]
    s_reordered = tf.transpose(s, perm_to_batch_last)
    # gather_nd: indices (L,)*dim + (n_sites, dim) -> result (L,)*dim + (n_sites,) + (batch_size,)
    V_gathered = tf.gather_nd(s_reordered, trans_tf)  # (L,)*dim + (n_sites, batch_size)
    # Move batch to front: batch is at index dim in V_gathered
    perm_to_batch_first = (dim + 1,) + tuple(range(dim)) + (dim,)
    V = tf.transpose(V_gathered, perm_to_batch_first)  # (batch_size,) + (L,)*dim + (n_sites,)
    return V


@tf.function
def CalculateCorrelationForSample(s, trans_tf, model, batch_size, L, dim=2):
    """Extract the Vs from a sample (using trans_tf) and operate on them with the model.
    s: (batch_size,) + (L,)*dim. trans_tf: (L,)*dim + (n_sites, dim)."""
    V = _sample_to_V(s, trans_tf, dim)
    corrs, means = CalculateCorrelationForOperator(model, V, batch_size, L, dim)
    return corrs, means


    
def compute_correlations_from_dataset(
    dataset,
    trans_tf,
    model,
    batch_size,
    L,
    sample_size,
    max_batches,
    time_threshold=None,
    log_err_threshold=0.01,
    temp_file=None,
    pbar=None,
    dim=2,
):
    """
    Run the correlation accumulation loop over an arbitrary dataset that yields
    batches of shape (batch_size,) + (L,)*dim float32 raw lattices.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Yields (batch_size,) + (L,)*dim float32.
    trans_tf : tf.Tensor
        From create_shifted_grid(..., L, dim) + tf.constant(np.mod(res, L), dtype=tf.int32).
    model : callable
        Keras model or callable: input (None, V_size), output reshapable to (batch_size,) + (L,)*dim.
    batch_size, L : int
    sample_size : int
        Number of bins for accumulation.
    max_batches : int
        If set, stop after this many batches (bin_counts may be uneven).
    time_threshold : float, optional
        If set, every time_threshold seconds when c%sample_size==0 print/save and optionally break.
    log_err_threshold : float, optional
        If set with time_threshold, break when log_err < this.
    temp_file : str, optional
        If set with time_threshold, save checkpoint pickle here.
    pbar : tqdm or None
        If provided, update(1) each batch; else no progress bar.
    dim : int
        Lattice dimension (2 or 3).

    Returns
    -------
    corr_accum, mean_accum, bin_counts : np.ndarray
        Accumulated correlations (sample_size,) + (L,)*dim, means (sample_size,), and batch counts per bin (sample_size,).
    """
    assert dim in (2, 3), "connected_correlation only supports dim=2 or dim=3, got %d" % dim

    half_L = int(L / 2)
    correlation_tuple = (sample_size,) + (L,) * dim

    bin_counts = np.zeros(sample_size)
    corr_accum = np.zeros(correlation_tuple)
    mean_accum = np.zeros(sample_size)
    c = 0
    cur_time = time.time()
    print("  [L=%d] started at %s" % (L, time.ctime()))
    for s in dataset:
        if pbar is not None:
            pbar.update(1)
        corrs, means = CalculateCorrelationForSample(s, trans_tf, model, batch_size, L, dim)
        corr_accum[c % sample_size] += corrs
        mean_accum[c % sample_size] += means
        bin_counts[c % sample_size] += 1
        c += 1
        if (
            time_threshold is not None
            and time.time() - cur_time > time_threshold
            and c % sample_size == 0
        ):
            cur_time = time.time()
            if temp_file:
                open(temp_file, "wb").write(pickle.dumps((corr_accum, mean_accum, bin_counts)))
            val, err, log_err = CalculateCorrelationWithError(corr_accum, mean_accum, bin_counts, half_L, dim)
            print(
                "  [L=%d] %d batches | connected correlation = %.6e +/- %.2e "
                "(relative log-error %.4f)" % (L, c, val, err, log_err)
            )
            if log_err_threshold is not None and log_err < log_err_threshold:
                print(
                    "  [L=%d] target log-error %.4f reached after %d batches"
                    % (L, log_err_threshold, c)
                )
                break
        if c >= max_batches:
            break
    print("  [L=%d] finished at %s" % (L, time.ctime()))
    assert np.var(bin_counts) == 0
    return (corr_accum, mean_accum, bin_counts)


def run_connected_correlation(
    model,
    data_dir,
    V_indices,
    *,
    batch_size,
    sample_size=256,
    max_batches=None,
    time_threshold=None,
    log_err_threshold=None,
    temp_file=None,
    seed=0,
    shuffle_files=True,
    shuffle_samples=True,
    out=None
):
    """
    Run connected correlation estimation with a callable model and data directory.

    Loads config.json from data_dir (for L and TFRecord layout), builds the dataset
    and shifted-grid extraction, runs the accumulation loop, and optionally saves results.
    The model is used as-is (no loading from disk).

    Parameters
    ----------
    model : callable
        Keras Model or tf.function: input (None, V_size) float32, output reshapeable to (batch_size,) + (L,)*dim.
    data_dir : str
        Folder containing config.json (L, optional dim) and TFRecord shards.
    V_indices : array-like
        List or array of length-dim lattice index tuples for the operator (used by create_shifted_grid).
    batch_size : int
        Batch size for the dataset.
    sample_size : int
        Number of bins for accumulation.
    max_batches : int, optional
        If set, stop after this many batches.
    time_threshold : float, optional
        If set, every time_threshold seconds when c%%sample_size==0 print/save and optionally break.
    log_err_threshold : float, optional
        If set with time_threshold, break when log_err < this.
    temp_file : str, optional
        If set with time_threshold, save checkpoint pickle here.
    seed : int, optional
        Random seed for dataset shuffle.
    shuffle_files : bool
        Whether to shuffle TFRecord file order.
    shuffle_samples : bool
        Whether to shuffle samples within the dataset.
    out : str, optional
        If set, save (corr_accum, mean_accum, bin_counts) to this pickle path.

    Returns
    -------
    corr_accum, mean_accum, bin_counts : np.ndarray
        Accumulated correlation maps, means per bin, and batch counts per bin.
    result : dict
        With keys "val", "err", "log_err" for the connected correlation at antipodal (half_L).
    """
    # Load config from data dir (for L, dim)
    data_config_path = os.path.join(data_dir, "config.json")
    if not os.path.isfile(data_config_path):
        raise FileNotFoundError("config.json not found in data dir: %s" % data_dir)
    with open(data_config_path) as f:
        data_cfg = json.load(f)
    L = int(data_cfg["L"])
    dim = int(data_cfg["dim"])
    assert dim in (2, 3), "connected_correlation only supports dim=2 or dim=3, got %d" % dim

    # Total samples and batches
    records_per_file = data_cfg["records_per_file"]
    compression = data_cfg["compression"]
    pattern = "*.tfrecord.gz" if compression == "GZIP" else "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(data_dir, pattern))
    num_files = len(files)
    assert num_files > 0, "no TFRecord files found in data_dir"

    total_batches = math.ceil(num_files * records_per_file / batch_size)
    total_batches_flat = (total_batches // sample_size) * sample_size
    if max_batches is None:
        max_batches = total_batches_flat
    else:
        max_batches = (max_batches // sample_size) * sample_size  # round down so bin_counts stay flat

    # TFRecord dataset
    dataset, _, _, _ = build_tfrecord_dataset(
        data_dir,
        batch_size=batch_size,
        shuffle_files=shuffle_files,
        shuffle_samples=shuffle_samples,
        repeat=False,
        seed=seed,
    )
    # Shifted grid for V extraction
    res = create_shifted_grid(V_indices, L, dim)
    trans_tf = tf.constant(np.mod(res, L), dtype=tf.int32)

    pbar = tqdm(total=max_batches, desc="Batches", unit="batch") 

    corr_accum, mean_accum, bin_counts = compute_correlations_from_dataset(
        dataset,
        trans_tf,
        model,
        batch_size,
        L,
        sample_size=sample_size,
        max_batches=max_batches,
        time_threshold=time_threshold,
        log_err_threshold=log_err_threshold,
        temp_file=temp_file,
        pbar=pbar,
        dim=dim,
    )

    pbar.close()

    half_L = int(L / 2)
    val, err, log_err = CalculateCorrelationWithError(corr_accum, mean_accum, bin_counts, half_L, dim)
    result = {"val": val, "err": err, "log_err": log_err}

    if out is not None:
        with open(out, "wb") as f:
            pickle.dump((corr_accum, mean_accum, bin_counts), f)

    return (corr_accum, mean_accum, bin_counts), result