# pipeline_fp16_int8_block4_rowq.py – inference w/ 4-block INT8 quant
#======================================================================
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import time
import glob
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# constants
E_MIN, E_MAX   = 1e4 * 1e-6, 1e6 * 1e-6
BASE_FILE_PATH = '/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
TEMP_DATA_DIR  = '/mnt/d/800_1200'
PAD            = 4

# globals
W0 = b0 = T_scale = T_mean = None
alpha = 0.0
QW_tab      = None
scale_blocks= None
b_tab       = None

# load table
def load_table(table_h5: str):
    global W0, b0, T_scale, T_mean, alpha, QW_tab, scale_blocks, b_tab
    dt = tf.keras.mixed_precision.global_policy().compute_dtype
    with h5py.File(table_h5, 'r') as hf, tf.device('/GPU:0'):
        W0           = tf.constant(hf['W0'][:], dtype=dt)
        b0           = tf.constant(hf['b0'][:], dtype=dt)
        T_scale      = tf.constant(hf['T_scale'][:], dtype=dt)
        T_mean       = tf.constant(hf['T_mean'][:], dtype=dt)
        alpha        = float(hf['alpha'][()])
        QW_tab       = tf.constant(hf['QW_tab'][:], dtype=tf.int8)
        scale_blocks = tf.constant(hf['scale_blocks'][:], dtype=tf.float32)  # float32 [E,4]
        b_tab        = tf.constant(hf['b_tab'][:], dtype=dt)

@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None], tf.int32)
    ], experimental_compile=True
)
def query_table(T_batch: tf.Tensor, E_batch: tf.Tensor) -> tf.Tensor:
    # MLP hidden
    Tn     = (tf.cast(T_batch, tf.float16) - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(Tn[:, None], W0) + b0, alpha)
    # gather + dequantize per block at full precision
    Q_sel = tf.gather(QW_tab, E_batch)            # [N,16] int8
    s_sel = tf.gather(scale_blocks, E_batch)      # [N,4] float32 scales
    parts = []
    for blk in range(4):
        lo, hi    = blk*4, blk*4 + 4
        scale_blk = tf.expand_dims(s_sel[:, blk], 1)  # [N,1] float32
        qblk_i8   = tf.cast(Q_sel[:, lo:hi], tf.float32)
        wblk_f32  = qblk_i8 * scale_blk
        parts.append(tf.cast(wblk_f32, tf.float16))
    W_vec = tf.concat(parts, axis=1)                # [N,16]
    b_vec = tf.gather(b_tab, E_batch)
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

# ══════════════════════════ Benchmark harness ═══════════════════════════
def benchmark(table_h5: str, batch: int, tmin: float, tmax: float):
    load_table(table_h5)
    with tf.device('/GPU:0'):
        T = tf.random.uniform([batch], tmin, tmax, dtype=tf.float32)
        E = tf.random.uniform([batch], PAD, 90000, dtype=tf.int32)
    # warm-up
    _ = query_table(T[:1], E[:1])
    t0 = time.perf_counter()
    xs = query_table(T, E)
    _  = xs.numpy()
    dur = (time.perf_counter() - t0) * 1e6 / batch
    print(f"{batch:,} points → {dur:.4f} µs/point")
    return xs, T, E

# ═══════════════════════ helper: base grid ═════════════════════════════
def load_base(e_min, e_max, base_t=200.0):
    df     = pd.read_hdf(BASE_FILE_PATH, key='xs_data', compression='gzip')
    subset = df[(df['T'] == base_t) & (df['ERG'] >= e_min) & (df['ERG'] <= e_max)]
    E      = subset['ERG'].to_numpy()
    return np.sort(E), len(E)

# ═══════════════════════ helper: pad & interp ═════════════════════════════
def load_temperature(test_temp: float, base_e: np.ndarray, file_dir: str):
    for fp in glob.glob(os.path.join(file_dir, '*.h5')):
        df = pd.read_hdf(fp, key='xs_data', compression='gzip')
        if test_temp not in df['T'].values:
            continue
        subset = df[(df['T'] == test_temp) & (df['ERG'] >= E_MIN) & (df['ERG'] <= E_MAX)]
        if len(subset) < 2:
            continue
        E  = subset['ERG'].to_numpy(); xs = subset['XS'].to_numpy()
        idx = np.argsort(E)
        interp    = interp1d(E[idx], xs[idx], kind='cubic', fill_value='extrapolate')
        padded_e  = np.pad(base_e, (PAD, PAD), mode='constant')
        padded_xs = np.pad(interp(base_e), (PAD, PAD), mode='constant')
        return padded_e, padded_xs
    return None, None

# ═════════════════════════════ Accuracy plot ═════════════════════════════
def analyse(base_e, xs_rec, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(float(temps[0]), base_e, file_dir)
    if padded is None:
        print("Ground-truth data not found – skipping accuracy plot.")
        return
    xs_vals = xs_rec.numpy(); idxs = eidxs.numpy()
    orig    = padded[idxs]; relerr = np.abs(xs_vals - orig) / np.abs(orig) * 100
    os.makedirs("./results_final_inference", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(base_e, padded[PAD:-PAD], label='Ground truth')
    plt.scatter(padded_e[idxs-PAD], xs_vals, c='r', s=10, label='Reconstructed')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy'); plt.ylabel('XS')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("./results_final_inference/xs.png", dpi=200); plt.close()

    relerr  = np.abs(xs_vals - orig) / np.abs(orig) * 100
    sorted_idx = np.argsort(idxs)
    idxs_sorted = idxs[sorted_idx]
    rel_err_sorted = relerr[sorted_idx]
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs_sorted-PAD], rel_err_sorted, marker='o', linestyle='-')
    plt.xlabel('Energy'); plt.ylabel('Relative error (%)')
    plt.grid(True); plt.tight_layout()
    plt.savefig("./results_final_inference/relative_error_linear.png", dpi=200)
    plt.close()
    print("Accuracy plots saved to ./results_final_inference")

# ═════════════════════════════════ CLI ═════════════════════════════════
if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument("--table", default="w_table_final.h5", help="HDF5 INT8-quantized table file")
    P.add_argument("--batch", type=int, default=8192)
    P.add_argument("--tmin",  type=float, default=1000.0)
    P.add_argument("--tmax",  type=float, default=1000.0)
    P.add_argument("--accuracy", action="store_true", help="produce error plots")
    args = P.parse_args()
    xs, temps, eidxs = benchmark(args.table, args.batch, args.tmin, args.tmax)
    if args.accuracy:
        base_e = load_base(E_MIN, E_MAX)[0]
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)