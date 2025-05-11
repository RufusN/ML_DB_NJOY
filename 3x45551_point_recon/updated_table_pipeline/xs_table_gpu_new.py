#!/usr/bin/env python3
"""
xs_table_gpu.py – batched GPU inference via precomputed weight table
==================================================================
Uses the precomputed W_tab and b_tab (and the first-layer weights) to reconstruct
many XS queries in one shot on the GPU.  No Keras model is loaded at runtime.

Prerequisites:
  • 3x45551_950_1050_spec_scalers.h5  (T-scale/mean)
  • w_table.h5                        (W_tab, b_tab, E_idxs, W0, b0, alpha)

Usage:
    python xs_table_gpu_new.py [--batch 50000000] [--chunk 5000000]
"""
import argparse
import time
import os
import h5py
import tensorflow as tf

# Paths to HDF5 files
SCALER_PATH = "/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/3x45551_950_1050_spec_scalers.h5"
TABLE_PATH  = "/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/w_table.h5"

# Maximum sub-batch size (tune this so chunk_size*16*4 bytes ≲ your free GPU memory)
DEFAULT_BATCH_SIZE = 10_000_000
DEFAULT_CHUNK_SIZE = 1_000_000

@tf.function
def xs_from_table(T_batch, E_batch,
                  W0, b0, alpha,
                  W_tab, b_tab,
                  T_scale, T_mean):
    """Compute XS batch via hidden-layer and weight table dot."""
    T_norm = (T_batch - T_mean) / T_scale                             # [N]
    hidden = tf.nn.leaky_relu(tf.expand_dims(T_norm, 1) @ W0 + b0, alpha)  # [N,16]
    wvec = tf.gather(W_tab, E_batch)   # [N,16]
    bvec = tf.gather(b_tab, E_batch)   # [N]
    xs = tf.reduce_sum(hidden * wvec, axis=1) + bvec  # [N]
    return xs

def batched_xs(T_batch, E_batch,
               W0, b0, alpha,
               W_tab, b_tab,
               T_scale, T_mean,
               chunk_size):
    """Break a big batch into chunks and run xs_from_table on each."""
    N = T_batch.shape[0]
    outputs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xs_chunk = xs_from_table(
            T_batch[start:end],
            E_batch[start:end],
            W0, b0, alpha,
            W_tab, b_tab,
            T_scale, T_mean
        )
        outputs.append(xs_chunk)
    return tf.concat(outputs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                    help="Total number of random queries to benchmark")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE,
                    help="Sub-batch size for GPU (avoid OOM)")
    args = ap.parse_args()
    N = args.batch
    chunk_size = args.chunk

    # --- load T scalers ---
    with h5py.File(SCALER_PATH, 'r') as hf:
        T_scale = tf.constant(hf['T_scale'][:].astype('float32'))  # [1]
        T_mean  = tf.constant(hf['T_mean'] [:].astype('float32'))  # [1]

    # --- load weight table ---
    with h5py.File(TABLE_PATH, 'r') as hf:
        W_tab  = tf.constant(hf['W_tab'][:].astype('float32'))   # [N_E,16]
        b_tab  = tf.constant(hf['b_tab'][:].astype('float32'))   # [N_E]
        E_idxs = hf['E_idxs'][:].astype('int32')                 # [N_E]
        W0     = tf.constant(hf['W0'][:].astype('float32'))     # [1,16]
        b0     = tf.constant(hf['b0'][:].astype('float32'))     # [16]
        alpha  = float(hf['alpha'][()])                         # scalar

    # --- generate random queries ---
    temps = tf.random.uniform([N], 950., 1050., dtype=tf.float32)
    idxs = tf.random.uniform([N], 0, E_idxs.shape[0], dtype=tf.int32)
    E_batch = tf.gather(E_idxs, idxs)

    # --- warm-up compile ---
    xs_from_table(temps[:1], E_batch[:1],
                  W0, b0, alpha,
                  W_tab, b_tab,
                  T_scale, T_mean)

    # --- timed run with chunking ---
    t0 = time.perf_counter()
    xs_vals = batched_xs(
        temps, E_batch,
        W0, b0, alpha,
        W_tab, b_tab,
        T_scale, T_mean,
        chunk_size
    )
    dt = time.perf_counter() - t0

    print(f"{N:,} total points (chunk={chunk_size:,}) → {dt*1e3:.1f} ms  |  {dt/N*1e6:.3f} µs per point")

if __name__ == '__main__':
    main()
