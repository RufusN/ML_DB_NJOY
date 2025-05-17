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

tf.config.optimizer.set_jit(True)

# ─────────────────────────── domain constants ──────────────────────────
E_MIN, E_MAX = 1e4 * 1e-6, 1e6 * 1e-6       # MeV
WINDOW_SIZE  = 0.00004744
STEP_SIZE    = 0.00002314
H, W_TIME, C = 3, 45551, 2

BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
WEIGHTS_H5     = 'model_weights.h5'
SCALER_PATH    = '3x45551_950_1050_spec_scalers.h5'
TEMP_DATA_DIR  = r'/mnt/d/800_1200'

# ──────────────────────────── GPU constants ────────────────────────────
W_k = b_k = rows = None      # compact / full weights
K    = None
W0 = b0 = T_scale = T_mean = None
alpha = 0.0
FULL_TABLE = False           # True → W_k = W_tab, no interpolation
# lowrank
U_r = V_r = None
RANK = 0
FMT = "full"

# ═════════════════════════════ Load table ═══════════════════════════════
def load_table(table_h5: str, dtype: tf.dtypes.DType):
    global W_k, b_k, rows, K, W0, b0, T_scale, T_mean, alpha, FULL_TABLE, W_tab, b_tab, W_q, scale_q, zero_q, U_r, V_r, RANK

    with h5py.File(table_h5, "r") as hf, tf.device("/GPU:0"):
        cast = lambda d: tf.constant(hf[d][:], dtype=dtype)

        W0, b0  = cast("W0"), cast("b0")
        T_scale = cast("T_scale")
        T_mean  = cast("T_mean")
        alpha   = float(hf["alpha"][()])

        if "U_r" in hf and "V_r" in hf:
            FMT  = "lowrank"
            U_r  = cast("U_r")      # shape (E, r)
            V_r  = cast("V_r")      # shape (r, 16)
            RANK = int(hf["rank"][()])
            b_tab = cast("b_tab")
            rows  = tf.range(tf.shape(U_r)[0], dtype=tf.int32)
            K = 1
            return

        elif "W_tab" in hf:                      # ───── full table ─────
            FULL_TABLE = True
            W_k  = cast("W_tab")               # (E,16)
            b_k  = cast("b_tab")               # (E,)
            rows = tf.range(tf.shape(W_k)[0], dtype=tf.int32)
            K    = 1                           # stride 1 → no lerp

        else:                                  # ─── compact table ───
            FULL_TABLE = False
            K    = int(hf["K"][()])
            rows = tf.constant(hf["rows"][:], dtype=tf.int32)
            W_k  = cast("W_k")                 # (N_K,16)
            b_k  = cast("b_k")                 # (N_K,)

# ═════════════════════════════ Query kernel ═════════════════════════════
@tf.function(
    input_signature=[tf.TensorSpec([None], tf.float16),
                     tf.TensorSpec([None], tf.int32)],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    """Return XS values for (T, E_idx) pairs."""
    seg_idx  = tf.math.floordiv(E_batch, K)                    # (N,)
    seg_idxR = tf.minimum(seg_idx + 1, tf.shape(rows)[0] - 1)  # (N,)
    t        = tf.cast(E_batch % K, T_batch.dtype) / tf.cast(K, T_batch.dtype)

    W_left  = tf.gather(W_k, seg_idx)                          # (N,16)
    b_left  = tf.gather(b_k, seg_idx)

    if FMT == "lowrank":
        U_sel = tf.gather(U_r, E_batch)       # (N, r)
        W_vec = tf.matmul(U_sel, V_r)         # back to (N, 16)
        b_vec = tf.gather(b_tab, E_batch)
    
    elif FULL_TABLE:                                             # no lerp
        W_vec, b_vec = W_left, b_left
    else:
        W_right = tf.gather(W_k, seg_idxR)
        b_right = tf.gather(b_k, seg_idxR)
        W_vec = W_left + tf.expand_dims(t, 1) * (W_right - W_left)
        b_vec = b_left + t * (b_right - b_left)

    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:, None], W0) + b0, alpha)
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

# ══════════════════════════ Benchmark harness ═══════════════════════════
def benchmark(table_h5: str, batch: int, tmin: float, tmax: float,
              policy: str):
    mixed_precision.set_global_policy(policy)
    dtype = tf.float16 if policy != "float32" else tf.float32
    load_table(table_h5, dtype)

    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch], tmin, tmax, dtype=dtype)
        E = tf.random.uniform([batch], 0, rows[-1] + 1, dtype=tf.int32)

    _ = query_table(T, E)               # JIT warm-up
    t0 = time.perf_counter()
    xs = query_table(T, E)
    _  = xs.numpy()                     # host sync
    dur = (time.perf_counter() - t0) * 1e6 / batch
    print(f"{batch:,} points → {dur:.4f} µs/point")
    return xs, T, E                     # temps & indices for analysis

# ═══════════════════════ reference data helpers ════════════════════════
def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key='xs_data', compression='gzip')
    subset = df[(df['T'] == base_t) &
                (df['ERG'] >= e_min) &
                (df['ERG'] <= e_max)]
    E = subset['ERG'].to_numpy()
    idx = np.argsort(E)
    return E[idx], len(E)

def init_full():
    global fs, WINDOW_SAMPS, STEP_SAMPS, PAD
    base_e, fs = load_base(E_MIN, E_MAX)
    WINDOW_SAMPS = int(WINDOW_SIZE * fs)
    STEP_SAMPS   = int(STEP_SIZE  * fs)
    PAD = WINDOW_SAMPS
    return base_e

def load_temperature(test_temp, base_e, file_dir):
    for fp in glob.glob(os.path.join(file_dir, '*.h5')):
        df = pd.read_hdf(fp, key='xs_data', compression='gzip')
        if test_temp not in df['T'].values:
            continue
        subset = df[(df['T'] == test_temp) &
                    (df['ERG'] >= E_MIN) &
                    (df['ERG'] <= E_MAX)]
        if len(subset) < 2:
            continue
        E, xs = subset['ERG'].to_numpy(), subset['XS'].to_numpy()
        idx = np.argsort(E)
        interp = interp1d(E[idx], xs[idx],
                          kind='cubic', fill_value='extrapolate')
        return (np.pad(base_e, (PAD, PAD), mode='constant'),
                np.pad(interp(base_e), (PAD, PAD), mode='constant'))
    return None, None

def analyse(base_e, xs_rec, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(float(temps[0]), base_e, file_dir)
    if padded is None:
        print("Ground-truth data not found – skipping accuracy plot.")
        return
    xs_rec = xs_rec.numpy()
    idxs   = eidxs.numpy()
    orig   = padded[idxs]
    relerr = np.abs(xs_rec - orig) / np.abs(orig) * 100

    os.makedirs("./data_reconstruct", exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(padded_e, padded, label='Ground truth')
    plt.scatter(padded_e[idxs], xs_rec, c='r', s=10, label='Reconstructed')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy'); plt.ylabel('XS')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data_reconstruct/xs.png", dpi=200); plt.close()

    srt = np.argsort(idxs)
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs[srt]-PAD], relerr[srt], marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Energy'); plt.ylabel('Relative error (%)')
    plt.title('Relative error vs energy'); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data_reconstruct/relative_error.png", dpi=200); plt.close()
    print("Accuracy plots saved to ./data_reconstruct")

# ═════════════════════════════════  CLI  ════════════════════════════════
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--table",  default="w_table_compact.h5")
    P.add_argument("--batch",  type=int,   default=8192)
    P.add_argument("--tmin",   type=float, default=1000.0)
    P.add_argument("--tmax",   type=float, default=1000.0)
    P.add_argument("--policy", choices=["float16","mixed_float16","float32"],
                   default="float16")
    P.add_argument("--accuracy", action="store_true",
                   help="produce error plots against ground truth data")
    args = P.parse_args()

    xs, temps, eidxs = benchmark(args.table, args.batch,
                                 args.tmin, args.tmax,
                                 args.policy)

    if args.accuracy:
        base_e = init_full()
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)
