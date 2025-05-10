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
    python xs_table_gpu.py --batch 8192
"""
import argparse
import time
import h5py
import tensorflow as tf

# Paths to HDF5 files
SCALER_PATH = "3x45551_950_1050_spec_scalers.h5"
TABLE_PATH  = "w_table.h5"

# JIT-compiled inference: only inputs T_batch, E_batch
@tf.function
def xs_from_table(T_batch, E_batch,
                  W0, b0, alpha,
                  W_tab, b_tab,
                  T_scale, T_mean):
    """Compute XS batch via hidden-layer and weight table dot."""
    # 1) Temperature → normalized → leaky-ReLU hidden vector [N,16]
    T_norm = (T_batch - T_mean) / T_scale                        # [N]
    hidden = tf.nn.leaky_relu(tf.expand_dims(T_norm,1) @ W0 + b0, alpha)  # [N,16]

    # 2) Gather precomputed weight vectors and biases
    wvec = tf.gather(W_tab, E_batch)  # [N,16]
    bvec = tf.gather(b_tab, E_batch)  # [N]

    # 3) Dot-product and bias
    xs = tf.reduce_sum(hidden * wvec, axis=1) + bvec    # [N]
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192,
                    help="Number of random queries to benchmark")
    args = ap.parse_args()
    N = args.batch

    # --- load T scalers ---
    with h5py.File(SCALER_PATH, 'r') as hf:
        T_scale = tf.constant(hf['T_scale'][:].astype('float32'))  # [1]
        T_mean  = tf.constant(hf['T_mean'] [:].astype('float32'))  # [1]

    # --- load weight table (includes first-layer + hidden→XS) ---
    with h5py.File(TABLE_PATH, 'r') as hf:
        W_tab  = tf.constant(hf['W_tab'][:].astype('float32'))   # [N_E,16]
        b_tab  = tf.constant(hf['b_tab'][:].astype('float32'))   # [N_E]
        E_idxs = hf['E_idxs'][:].astype('int32')                # [N_E]
        W0     = tf.constant(hf['W0'][:].astype('float32'))     # [1,16]
        b0     = tf.constant(hf['b0'][:].astype('float32'))     # [16]
        alpha  = float(hf['alpha'][()])                         # scalar

    # --- random test batch of (T, E_idx) ---
    temps = tf.random.uniform([N], 950., 1050., dtype=tf.float32)
    # sample energy indices (with replacement)
    idxs = tf.random.uniform([N], 0, E_idxs.shape[0], dtype=tf.int32)
    E_batch = tf.gather(E_idxs, idxs)

    # --- warm-up (compile) ---
    xs_from_table(temps[:1], E_batch[:1],
                  W0, b0, alpha,
                  W_tab, b_tab,
                  T_scale, T_mean)

    # --- timed run ---
    t0 = time.perf_counter()
    xs_vals = xs_from_table(temps, E_batch,
                             W0, b0, alpha,
                             W_tab, b_tab,
                             T_scale, T_mean)
    dt = time.perf_counter() - t0

    print(f"{N:,} points -> {dt*1e3:.1f} ms   |   {dt/N*1e6:.3f} µs per point")

if __name__ == '__main__':
    main()
