#!/usr/bin/env python3
"""
pipeline_with_table.py — Unified inference pipeline: optimized table lookup
==============================================================================

This version specializes the table backend for maximum throughput:
  • Enables XLA JIT compilation
  • Loads all constants onto the GPU once
  • Uses tf.function with static input_signature
  • Uses tf.nn.embedding_lookup (optimized gather)

Usage:
  python pipeline_with_table.py --batch 20000000 --tmin 1000.0 --tmax 1000.0 --precision float32
"""
import argparse, time
import h5py
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable XLA JIT globally
tf.config.optimizer.set_jit(True)

# Globals: will be populated once
W_tab = None
b_tab = None
W0 = None
b0 = None
alpha = 0.0
T_scale = None
T_mean = None

# Load constants onto GPU
def initialize_table(table_h5: str):
    global W_tab, b_tab, W0, b0, alpha, T_scale, T_mean
    with h5py.File(table_h5, 'r') as hf:
        # Move constants to GPU memory immediately
        with tf.device('/GPU:0'):
            W_tab   = tf.constant(hf['W_tab'][:])
            b_tab   = tf.constant(hf['b_tab'][:])
            W0      = tf.constant(hf['W0'][:])
            b0      = tf.constant(hf['b0'][:])
            T_scale = tf.constant(hf['T_scale'][:])
            T_mean  = tf.constant(hf['T_mean'][:])
        alpha = float(hf['alpha'][()])

# Optimized query: static signature for XLA, embedding lookup
@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None], tf.int32)
    ],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    # normalize temperature
    T_norm = (T_batch - T_mean) / T_scale  # [N]
    # MLP hidden layer: [N,1] x [1,16] -> [N,16]
    hidden = tf.nn.leaky_relu(
        tf.matmul(tf.expand_dims(T_norm, 1), W0) + b0,
        alpha
    )
    # embed weights and biases for each index
    W_vec = tf.nn.embedding_lookup(W_tab, E_batch)  # [N,16]
    b_vec = tf.nn.embedding_lookup(b_tab, E_batch)  # [N]
    # fused elementwise matmul+sum
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec  # [N]

# Benchmark helper
def benchmark(table_h5: str, batch_size: int, tmin: float, tmax: float,
              precision: str):
    # precision policy
    mixed_precision.set_global_policy(precision)
    # load constants
    initialize_table(table_h5)
    # random inputs
    T = tf.random.uniform([batch_size], tmin, tmax, tf.float32)
    E = tf.random.uniform([batch_size], 0, W_tab.shape[0], tf.int32)
    # warm-up
    _ = query_table(T[:1], E[:1])
    # timed
    start = time.perf_counter()
    xs = query_table(T, E)
    dur = time.perf_counter() - start
    print(f"{batch_size:,} points → {dur*1e6/batch_size:.5f} µs/point")
    return xs

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8192,
                        help='Number of points')
    parser.add_argument('--tmin', type=float, default=1000.0,
                        help='Minimum temperature')
    parser.add_argument('--tmax', type=float, default=1000.0,
                        help='Maximum temperature')
    parser.add_argument('--precision', choices=['float16','mixed_float16','float32'],
                        default='mixed_float16',
                        help='Precision policy')
    parser.add_argument('--table', type=str, default='w_table.h5',
                        help='Precomputed table HDF5')
    args = parser.parse_args()

    benchmark(
        table_h5=args.table,
        batch_size=args.batch,
        tmin=args.tmin,
        tmax=args.tmax,
        precision=args.precision
    )
