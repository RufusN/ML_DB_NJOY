#!/usr/bin/env python3
"""
pipeline_fp16.py – ultrafast half-precision inference with fused lookup
======================================================================
Key changes vs v1
-----------------
• Loads all constants as **float16** by default                      (memory ↓2×)
• Global policy set by --precision  (float16 | mixed_float16 | float32)
• XLA JIT on, and the `@tf.function` kernel is *fully fused*
• W_tab already stored latent-major, so no post-transpose needed
"""
import argparse, time, h5py, tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable graph-level optimisations
tf.config.optimizer.set_jit(True)

# ----------------------- GPU-resident constants -----------------------
W_tab = b_tab = W0 = b0 = T_scale = T_mean = None
alpha = 0.0

def initialize_table(table_h5: str, target_dtype: tf.dtypes.DType):
    global W_tab, b_tab, W0, b0, alpha, T_scale, T_mean
    with h5py.File(table_h5, 'r') as hf, tf.device('/GPU:0'):
        cast = lambda d: tf.constant(hf[d][:], dtype=target_dtype)
        W_tab, b_tab = cast('W_tab'), cast('b_tab')
        W0, b0 = cast('W0'), cast('b0')
        T_scale, T_mean = cast('T_scale'), cast('T_mean')
        alpha = float(hf['alpha'][()])

# --------------------------- fused kernel -----------------------------
@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float16),  # T_batch
        tf.TensorSpec([None], tf.int32)     # E_batch
    ],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    # FP16 normalisation & hidden layer --------------------------------
    T_norm = (T_batch - T_mean) / T_scale               # [N]
    hidden = tf.nn.leaky_relu(
        tf.matmul(tf.expand_dims(T_norm, 1), W0) + b0,
        alpha
    )                                                   # [N,16]
    # Gather & fused elementwise GEMM ------------------------------
    W_vec = tf.gather(W_tab, E_batch)                   # [N,16]
    b_vec = tf.gather(b_tab, E_batch)                   # [N]
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

# --------------------------- benchmark -------------------------------
def benchmark(table_h5: str, batch_size: int,
              tmin: float, tmax: float,
              precision_flag: str):
    # precision policy -------------------------------------------------
    mixed_precision.set_global_policy(precision_flag)
    compute_dtype = tf.float16 if precision_flag != 'float32' else tf.float32

    # load constants ---------------------------------------------------
    initialize_table(table_h5, compute_dtype)

    # synth inputs (already on GPU) -----------------------------------
    with tf.device('/GPU:0'):
        T = tf.random.uniform([batch_size],
                              tmin, tmax,
                              dtype=compute_dtype)
        E = tf.random.uniform([batch_size],
                              0, W_tab.shape[0],
                              dtype=tf.int32)

    # warm-up (XLA compilation) ---------------------------------------
    _ = query_table(T[:1], E[:1])

    # timed run --------------------------------------------------------
    start = time.perf_counter()
    xs = query_table(T, E)
    dur = time.perf_counter() - start
    print(f"{batch_size:,} points → {dur*1e6/batch_size:.4f} µs/point")
    return xs

# ----------------------------- CLI -----------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=8192,
                   help='Number of points')
    p.add_argument('--tmin', type=float, default=1000.0)
    p.add_argument('--tmax', type=float, default=1000.0)
    p.add_argument('--precision',
                   choices=['float16', 'mixed_float16', 'float32'],
                   default='float16',
                   help='TensorFlow mixed-precision policy')
    p.add_argument('--table', default='w_table.h5',
                   help='Precomputed FP16 table HDF5')
    args = p.parse_args()

    _ = benchmark(args.table, args.batch,
                  args.tmin, args.tmax,
                  args.precision)
