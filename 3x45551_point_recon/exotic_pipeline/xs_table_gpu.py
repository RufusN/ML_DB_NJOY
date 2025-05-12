#!/usr/bin/env python3
"""
Reconstruct XS from precomputed table (w_table.h5).
"""

import argparse
import h5py
import tensorflow as tf

TABLE_PATH = "w_table.h5"
SCALER_PATH = "3x45551_950_1050_spec_scalers.h5"

@tf.function
def reconstruct_xs(T, E_idx, W_tab, b_tab, W0, b0, alpha, T_scale, T_mean):
    T_norm = (T - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm, 1), W0) + b0, alpha)
    wvec = tf.gather(W_tab, E_idx)
    bvec = tf.gather(b_tab, E_idx)
    xs = tf.reduce_sum(hidden * wvec, axis=1) + bvec
    return xs

def main():
    import argparse
    import h5py
    import numpy as np
    import tensorflow as tf

    TABLE_PATH = "w_table.h5"
    SCALER_PATH = "3x45551_950_1050_spec_scalers.h5"

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=1000.0, help="Temperature (K)")
    parser.add_argument("--E_idx", type=int, default=700, help="Energy index")
    args = parser.parse_args()

    # Load scalers
    with h5py.File(SCALER_PATH, 'r') as hf:
        T_scale = tf.constant(hf["T_scale"][:], dtype=tf.float32)
        T_mean = tf.constant(hf["T_mean"][:], dtype=tf.float32)

    # Load table
    with h5py.File(TABLE_PATH, 'r') as hf:
        W_tab = tf.constant(hf["W_tab"][:], dtype=tf.float32)
        b_tab = tf.constant(hf["b_tab"][:], dtype=tf.float32)
        W0 = tf.constant(hf["W0"][:], dtype=tf.float32)
        b0 = tf.constant(hf["b0"][:], dtype=tf.float32)
        alpha = float(hf["alpha"][()])

    # Inputs
    T = tf.constant([args.T], dtype=tf.float32)
    E_idx = tf.constant([args.E_idx], dtype=tf.int32)

    # Normalize temperature and compute hidden layer
    T_norm = (T - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm, 1), W0) + b0, alpha)

    # Gather weights and bias
    wvec = tf.gather(W_tab, E_idx)
    bvec = tf.gather(b_tab, E_idx)

    # Reconstruct XS
    xs = tf.reduce_sum(hidden * wvec, axis=1) + bvec
    xs_val = float(xs.numpy().squeeze())

    print(f"[T={args.T}, E_idx={args.E_idx}]")
    print(f"→ Table-based XS: {xs_val:.8e}")

if __name__ == '__main__':
    main()

