#!/usr/bin/env python3
"""
xs_table_gpu.py – batched GPU inference via precomputed weight table
==================================================================
Uses the precomputed W_tab and b_tab to reconstruct many XS queries in one
shot on the GPU.  Only a small 16-element dot and bias per query.

Usage:
    python xs_table_gpu.py --batch 8192

Prerequisites:
  • 3x45551_950_1050_spec_scalers.h5
  • 3x45551_950_1050.keras
  • w_table.h5             (built by build_w_table.py)
"""
import argparse
import time
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

# ───── constants ─────
SCALER_PATH = "3x45551_950_1050_spec_scalers.h5"
MODEL_PATH  = "3x45551_950_1050.keras"
TABLE_PATH  = "w_table.h5"

# JIT-compile once with dynamic batch size
@tf.function
# xs_from_table will be traced once on first call; all constants are globals.
def xs_from_table(T_batch, E_batch, W0, b0, alpha, W_tab, b_tab, T_scale, T_mean):
    """Compute XS batch via one 16-element dot per sample."""
    # normalize T and compute hidden [N,16]
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm,1), W0) + b0, alpha)
    # gather table vectors
    wvec = tf.gather(W_tab, E_batch)
    bvec = tf.gather(b_tab, E_batch)
    # dot + bias
    xs = tf.reduce_sum(hidden * wvec, axis=1) + bvec
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()
    N = args.batch

    # ----- load scalers -----
    with h5py.File(SCALER_PATH, 'r') as hf:
        T_scale = tf.constant(hf['T_scale'][:].astype('float32'))
        T_mean  = tf.constant(hf['T_mean'] [:].astype('float32'))

    # ----- load model first-layer weights -----
    model = load_model(MODEL_PATH, compile=False)
    dense0 = model.layers[1]
    W0 = tf.constant(dense0.kernel.numpy().astype('float32'))
    b0 = tf.constant(dense0.bias.numpy().astype('float32'))
    lrelu = model.layers[2]
    alpha = float(getattr(lrelu, 'alpha', getattr(lrelu, 'negative_slope', 0.3)))

    # ----- load table and indices -----
    with h5py.File(TABLE_PATH, 'r') as hf:
        W_tab = tf.constant(hf['W_tab'][:].astype('float32'))  # [N_E,16]
        b_tab = tf.constant(hf['b_tab'][:].astype('float32'))  # [N_E]
        E_idxs = hf['E_idxs'][:].astype('int32')               # [N_E]

        # ----- random test batch -----
    temps = tf.random.uniform([N], 950., 1050., dtype=tf.float32)
    # sample E_batch from precomputed energy index list, with replacement
    E_source = tf.constant(E_idxs)
    idxs = tf.random.uniform([N], 0, tf.shape(E_source)[0], dtype=tf.int32)
    E_batch = tf.gather(E_source, idxs)

    # ----- warm-up -----
    xs_from_table(temps[:1], E_batch[:1], W0, b0, alpha, W_tab, b_tab, T_scale, T_mean)

    # ----- timed run -----
    t0 = time.perf_counter()
    xs_vals = xs_from_table(temps, E_batch, W0, b0, alpha, W_tab, b_tab, T_scale, T_mean)
    dt = time.perf_counter() - t0

    print(f"{N:,} points -> {dt*1e3:.1f} ms   |   {dt/N*1e6:.6f} µs per point")


if __name__ == '__main__':
    main()
