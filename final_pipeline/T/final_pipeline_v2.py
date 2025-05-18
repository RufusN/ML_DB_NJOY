# pipeline_fp16_coarse.py – inference with linear interpolation using FP32 table
# =============================================================================
#!/usr/bin/env python3
from __future__ import annotations
import argparse, time, glob
import h5py, numpy as np, tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import os

mixed_precision.set_global_policy('mixed_float16')

E_MIN, E_MAX = 1e4 * 1e-6, 1e6 * 1e-6
BASE_FILE_PATH = '/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
TEMP_DATA_DIR  = '/mnt/d/800_1200'
PAD            = 4

# globals
W0 = b0 = T_scale = T_mean = None
alpha = 0.0
W_coarse = None
b_coarse = None
stride   = None

# load FP32 coarse table + MLP params
def load_table(table_h5: str):
    global W0, b0, T_scale, T_mean, alpha, W_coarse, b_coarse, stride
    with h5py.File(table_h5, 'r') as hf, tf.device('/GPU:0'):
        W_coarse = tf.constant(hf['W_coarse'][:], dtype=tf.float32)
        b_coarse = tf.constant(hf['b_coarse'][:], dtype=tf.float32)
        stride   = int(hf['stride'][()])
        W0       = tf.cast(tf.constant(hf['W0'][:]), tf.float16)
        b0       = tf.cast(tf.constant(hf['b0'][:]), tf.float16)
        T_scale  = tf.cast(tf.constant(hf['T_scale'][:]), tf.float16)
        T_mean   = tf.cast(tf.constant(hf['T_mean'][:]), tf.float16)
        alpha    = float(hf['alpha'][()])

@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None], tf.int32)
    ], experimental_compile=True
)
def query_table(T_batch, E_batch):
    # MLP hidden in FP16
    Tn     = (tf.cast(T_batch, tf.float16) - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(Tn[:,None], W0) + b0, alpha)

    # interpolation in FP32
    t      = tf.cast(E_batch % stride, tf.float32) / float(stride)
    i0     = tf.math.floordiv(E_batch, stride)
    i1     = tf.minimum(i0 + 1, tf.shape(W_coarse)[0] - 1)

    W0_c   = tf.gather(W_coarse, i0)   # FP32
    W1_c   = tf.gather(W_coarse, i1)
    W_vec  = W0_c + tf.expand_dims(t,1) * (W1_c - W0_c)

    b0_c   = tf.gather(b_coarse, i0)
    b1_c   = tf.gather(b_coarse, i1)
    b_vec  = b0_c + t * (b1_c - b0_c)

    # cast hidden to FP32 for dot
    hidden_f32 = tf.cast(hidden, tf.float32)
    return tf.reduce_sum(hidden_f32 * W_vec, axis=1) + b_vec



# ────────────────────────────────────────────────────────────────────────
def benchmark(table_h5: str, batch: int, tmin: float, tmax: float):
    load_table(table_h5)
    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch], tmin, tmax, dtype=tf.float32)
        E = tf.random.uniform([batch], PAD, 90000, dtype=tf.int32)

    # warm-up + timed run
    _  = query_table(T[:1], E[:1])
    t0 = time.perf_counter()
    xs = query_table(T, E)
    _  = xs.numpy()
    dur = (time.perf_counter() - t0) * 1e6 / batch
    print(f"{batch:,} points → {dur:.4f} µs/point")
    return xs, T, E

# ────────────────────────────────────────────────────────────────────────
def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key="xs_data", compression="gzip")
    subset = df[(df["T"] == base_t) & (df["ERG"] >= e_min) & (df["ERG"] <= e_max)]
    return np.sort(subset["ERG"].to_numpy()), len(subset)

def load_temperature(test_temp: float, base_e: np.ndarray, file_dir: str):
    for fp in glob.glob(os.path.join(file_dir, "*.h5")):
        df = pd.read_hdf(fp, key="xs_data", compression="gzip")
        if test_temp not in df["T"].values:
            continue
        subset = df[(df["T"] == test_temp) & (df["ERG"] >= E_MIN) & (df["ERG"] <= E_MAX)]
        if len(subset) < 2:
            continue
        E  = subset["ERG"].to_numpy()
        xs = subset["XS"].to_numpy()
        idx = np.argsort(E)
        fn = interp1d(E[idx], xs[idx], kind="cubic", fill_value="extrapolate")
        padded_e  = np.pad(base_e, (PAD, PAD), mode="constant")
        padded_xs = np.pad(fn(base_e), (PAD, PAD), mode="constant")
        return padded_e, padded_xs
    return None, None

def analyse(base_e, xs_rec, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(float(temps[0]), base_e, file_dir)
    if padded is None:
        print("Ground-truth data not found – skipping accuracy plot.")
        return

    xs_vals = xs_rec.numpy(); idxs = eidxs.numpy()
    orig    = padded[idxs]
    relerr  = np.abs(xs_vals - orig) / np.abs(orig) * 100

    os.makedirs("./results_final_inference_v2", exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(base_e, padded[PAD:-PAD], label="Ground truth")
    plt.scatter(padded_e[idxs-PAD], xs_vals, c="r", s=10, label="Reconstructed")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Energy"); plt.ylabel("XS")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("./results_final_inference_v2/xs.png", dpi=200)
    plt.close()

    sorted_idx      = np.argsort(idxs)
    idxs_sorted     = idxs[sorted_idx]
    rel_err_sorted  = relerr[sorted_idx]
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs_sorted-PAD], rel_err_sorted, marker="o", linestyle="-")
    plt.xlabel("Energy"); plt.ylabel("Relative error (%)")
    plt.grid(True); plt.tight_layout()
    plt.savefig("./results_final_inference_v2/relative_error_linear.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table",    default="w_table_final_v2.h5")
    p.add_argument("--batch",    type=int,   default=8192)
    p.add_argument("--tmin",     type=float, default=1000.0)
    p.add_argument("--tmax",     type=float, default=1000.0)
    p.add_argument("--accuracy", action="store_true")
    args = p.parse_args()

    xs, temps, eidxs = benchmark(args.table, args.batch, args.tmin, args.tmax)
    if args.accuracy:
        base_e = load_base(E_MIN, E_MAX)[0]
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)
