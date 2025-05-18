#!/usr/bin/env python3
"""
pipeline_fp16_unified.py – latent-major Doppler-XS lookup
========================================================
Single script that understands **both** table formats:

* Full table  – datasets **W_tab (E,16)** and **b_tab (E,)**
* Compact     – datasets **rows (N_K,)**, **W_k (N_K,16)**, **b_k (N_K,)**
                plus scalar attribute **K**

The kernel auto-switches; CLI is unchanged.

Example
-------
# compact
python pipeline_fp16_unified.py --table w_table_compact.h5 --batch 2_000_000

# full
python pipeline_fp16_unified.py --table w_table_original.h5 --batch 2_000_000
"""
from __future__ import annotations
import argparse, time, glob, os
import h5py, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#tf.config.optimizer.set_jit(True)

# ─────────────────────────── domain constants ──────────────────────────
E_MIN, E_MAX = 1e4 * 1e-6, 1e6 * 1e-6       # MeV
WINDOW_SIZE  = 0.00004628                   # match original pipeline
STEP_SIZE    = 0.00002314
H, W_TIME, C = 3, 45551, 2

BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
WEIGHTS_H5     = 'model_weights.h5'
SCALER_PATH    = '3x45551_950_1050_spec_scalers.h5'
TEMP_DATA_DIR  = r'/mnt/d/800_1200'

# ──────────────────────────── GPU globals ────────────────────────────
W_k = b_k = rows = None
K    = None
W0 = b0 = T_scale = T_mean = None
alpha = 0.0
FULL_TABLE = False
TABLE_FILE = None
# for padding in accuracy mode
global PAD, WINDOW_SAMPS, STEP_SAMPS

# ═════════════════════════════ Load table ═══════════════════════════════
def load_table(table_h5: str, dtype: tf.dtypes.DType):
    global W_k, b_k, rows, K, W0, b0, T_scale, T_mean, alpha, FULL_TABLE, TABLE_FILE
    TABLE_FILE = table_h5
    with h5py.File(table_h5, "r") as hf, tf.device("/GPU:0"):
        cast = lambda d: tf.constant(hf[d][:], dtype=dtype)

        # MLP input→latent
        W0      = cast("W0")
        b0      = cast("b0")
        T_scale = cast("T_scale")
        T_mean  = cast("T_mean")
        alpha   = float(hf["alpha"][()])

        # detect low-rank (not shown here)
        if "U_r" in hf and "V_r" in hf:
            raise NotImplementedError("Low-rank format not supported in this script")

        # full table
        if "W_tab" in hf:
            print("----------------------------- HERE")
            FULL_TABLE = True
            W_k  = cast("W_tab")
            b_k  = cast("b_tab")
            rows = tf.range(tf.shape(W_k)[0], dtype=tf.int32)
            K    = 1
        else:
            print("----------------------------- HERE22222")
            # compact table
            FULL_TABLE = False
            K    = int(hf["K"][()])
            rows = tf.constant(hf["rows"][:], dtype=tf.int32)
            W_k  = cast("W_k")
            b_k  = cast("b_k")

# ═════════════════════════════ Query kernel ═════════════════════════════
@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None], tf.int32)
    ],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    seg_idx  = tf.math.floordiv(E_batch, K)
    seg_idxR = tf.minimum(seg_idx + 1, tf.shape(rows)[0] - 1)
    t        = tf.cast(E_batch % K, tf.float32) / tf.cast(K, tf.float32)

    W_left = tf.gather(W_k, seg_idx)
    b_left = tf.gather(b_k, seg_idx)

    W_right = tf.gather(W_k, seg_idxR)
    b_right = tf.gather(b_k, seg_idxR)
    W_vec = W_left + tf.expand_dims(t, 1) * (W_right - W_left)
    b_vec = b_left + t * (b_right - b_left)

    # MLP forward
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:, None], W0) + b0, alpha)
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

# ══════════════════════════ Benchmark harness ═══════════════════════════
def benchmark(table_h5: str, batch: int, tmin: float, tmax: float,
              policy: str):
    mixed_precision.set_global_policy(policy)
    dtype = tf.float32
    load_table(table_h5, dtype)

    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch], tmin, tmax, dtype=dtype)
        E = tf.random.uniform([batch], 4, 90000, dtype=tf.int32)

    _ = query_table(T[:], E[:])  # warm-up
    t0 = time.perf_counter()
    xs = query_table(T, E)
    _  = xs.numpy()
    dur = (time.perf_counter() - t0) * 1e6 / batch
    print(f"{batch:,} points → {dur:.4f} µs/point")
    return xs, T, E

# ═══════════════════════ helper: base grid ════════════════════════
def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key='xs_data', compression='gzip')
    subset = df[(df['T'] == base_t) &
                (df['ERG'] >= e_min) &
                (df['ERG'] <= e_max)]
    E = subset['ERG'].to_numpy()
    idx = np.argsort(E)
    return E[idx], len(E)

# ═══════════════════════ helper: pad & interp ════════════════════════
def init_full():
    global PAD, WINDOW_SAMPS, STEP_SAMPS
    # read window/step from table metadata
    with h5py.File(TABLE_FILE, 'r') as hf:
        win  = int(hf['window'][()])
        step = int(hf['step'][()])
    WINDOW_SAMPS = win
    STEP_SAMPS   = step
    PAD = WINDOW_SAMPS
    base_e, _ = load_base(E_MIN, E_MAX)
    return base_e


def load_temperature(test_temp: float, base_e: np.ndarray, file_dir: str):
    for fp in glob.glob(os.path.join(file_dir, '*.h5')):
        df = pd.read_hdf(fp, key='xs_data', compression='gzip')
        if test_temp not in df['T'].values:
            continue
        subset = df[(df['T'] == test_temp) &
                    (df['ERG'] >= E_MIN) &
                    (df['ERG'] <= E_MAX)]
        if len(subset) < 2:
            continue
        E = subset['ERG'].to_numpy()
        xs = subset['XS'].to_numpy()
        idx = np.argsort(E)
        interp = interp1d(E[idx], xs[idx], kind='cubic', fill_value='extrapolate')
        padded_e   = np.pad(base_e, (PAD, PAD), mode='constant')
        padded_xs  = np.pad(interp(base_e), (PAD, PAD), mode='constant')
        return padded_e, padded_xs
    return None, None

# ═════════════════════════════ Accuracy plot ═════════════════════════════
def analyse(base_e, xs_rec, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(float(temps[0]), base_e, file_dir)
    if padded is None:
        print("Ground-truth data not found – skipping accuracy plot.")
        return
    xs_vals = xs_rec.numpy()
    idxs    = eidxs.numpy()
    orig    = padded[idxs]
    relerr  = np.abs(xs_vals - orig) / np.abs(orig) * 100

    os.makedirs("./results_exotic_inference", exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(base_e, padded[4:-4], label='Ground truth')
    plt.scatter(padded_e[idxs-4], xs_vals, c='r', s=10, label='Reconstructed')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy'); plt.ylabel('XS')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results_exotic_inference/xs.png", dpi=200)
    plt.close()

    sorted_idx = np.argsort(idxs)
    idxs_sorted = idxs[sorted_idx]
    rel_err_sorted = relerr[sorted_idx]
    print(len(base_e), len(relerr))
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs_sorted-4], rel_err_sorted, marker='o', linestyle='-')
    plt.xlabel('Energy'); plt.ylabel('Relative error (%)')
    plt.title('Relative error vs energy'); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results_exotic_inference/relative_error_linear.png", dpi=200)
    plt.close()
    print("Accuracy plots saved to ./results_exotic_inference")

# ═════════════════════════════════ CLI ═════════════════════════════════
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--table",  default="w_table_exotic.h5")
    P.add_argument("--batch",  type=int,   default=8192)
    P.add_argument("--tmin",   type=float, default=1000.0)
    P.add_argument("--tmax",   type=float, default=1000.0)
    P.add_argument("--policy", choices=["float16","mixed_float16","float32"],
                   default="float32",
                   help="Precision policy")
    P.add_argument("--accuracy", action="store_true",
                   help="produce error plots against ground truth data")
    args = P.parse_args()

    xs, temps, eidxs = benchmark(args.table, args.batch,
                                 args.tmin, args.tmax,
                                 args.policy)
    if args.accuracy:
        base_e = init_full()
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)