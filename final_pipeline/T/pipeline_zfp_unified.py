#!/usr/bin/env python3
"""
pipeline_fp16_tiered.py – HVQ/AHVQ inference with fp32+fp16 caches  (v2)
========================================================================
Order used per query:
    1. fp32 rows   (rows32 / W32 / b32)
    2. fp16 rows   (rows16 / W16 / b16)
    3. HVQ fallback (C1/C2 + b_tab)

If `--policy float32` is selected **XLA is disabled automatically** to
avoid the CUDA tiling overflow that crashes the driver.
"""

from __future__ import annotations
import argparse, time, glob, os, h5py, tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from  scipy.interpolate import interp1d

# ───────────────────────── domain paths / constants ─────────────────────
E_MIN, E_MAX = 1e4*1e-6, 1e6*1e-6
BASE_FILE_PATH = r"/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
TEMP_DATA_DIR  = r"/mnt/d/800_1200"

# ───────────────────────── GPU-resident tensors ─────────────────────────
W0=b0=T_scale=T_mean=None; alpha=0.0
C1=C2=idx1=idx2=b_tab=None
rows32=W32=b32=rows16=W16=b16=None

# ——————— reset globals each load so we never reuse stale tensors ——————
def _clear_globals():
    global W0,b0,T_scale,T_mean,alpha
    global C1,C2,idx1,idx2,b_tab,rows32,W32,b32,rows16,W16,b16
    W0=b0=T_scale=T_mean=None; alpha=0.0
    C1=C2=idx1=idx2=b_tab=None
    rows32=W32=b32=rows16=W16=b16=None

# ───────────────────────── table loader ─────────────────────────
def load_table(path:str, dtype=tf.float16):
    _clear_globals()
    g = globals()     # local shorthand

    with h5py.File(path) as f, tf.device("/GPU:0"):
        cast = lambda k: tf.constant(f[k][:], dtype=dtype)

        g["W0"], g["b0"]           = cast("W0"), cast("b0")
        g["T_scale"], g["T_mean"]  = cast("T_scale"), cast("T_mean")
        g["alpha"]                 = float(f["alpha"][()])

        g["C1"],  g["C2"]          = cast("C1"), cast("C2")
        g["idx1"] = tf.constant(f["idx1"][:], tf.int32)
        g["idx2"] = tf.constant(f["idx2"][:], tf.int32)
        g["b_tab"]= cast("b_tab")

        if "rows32" in f:
            g["rows32"] = tf.constant(f["rows32"][:], tf.int32)
            g["W32"], g["b32"] = cast("W32"), cast("b32")
        if "rows16" in f:
            g["rows16"] = tf.constant(f["rows16"][:], tf.int32)
            g["W16"], g["b16"] = cast("W16"), cast("b16")

    print(f"[load] fp32 rows: {int(g['rows32'].shape[0]) if g['rows32'] is not None else 0}   "
          f"fp16 rows: {int(g['rows16'].shape[0]) if g['rows16'] is not None else 0}")

# ───────────────────────── core helpers ─────────────────────────
@tf.function(jit_compile=False)   # always eager-compile safe helper
def _hidden(T):
    T = tf.cast(T, T_scale.dtype)
    return tf.nn.leaky_relu(tf.matmul(((T - T_mean) / T_scale)[:, None], W0) + b0, alpha)

def _apply_cache(rows, Wc, bc, W_prev, b_prev, E):
    if rows is None or tf.size(rows) == 0:
        return W_prev, b_prev        # nothing to do
    n = tf.shape(rows)[0]
    rows_pad = tf.concat([rows, [-1]], 0)
    W_pad = tf.concat([Wc, tf.zeros([1, tf.shape(Wc)[1]], Wc.dtype)], 0)
    b_pad = tf.concat([bc, tf.zeros([1], bc.dtype)], 0)
    idx = tf.searchsorted(rows, E, side="left")
    idx_c = tf.minimum(idx, n)
    matched = tf.equal(tf.gather(rows_pad, idx_c), E)
    W_new = tf.where(matched[:, None], tf.gather(W_pad, idx_c), W_prev)
    b_new = tf.where(matched, tf.gather(b_pad, idx_c), b_prev)
    return W_new, b_new

# Decide at definition time whether to let XLA fuse the whole kernel
def _make_query_fn(xla: bool):
    @tf.function(experimental_compile=xla)
    def query_fn(T, E):
        h = _hidden(T)
        W = tf.gather(C1, tf.gather(idx1, E)) + tf.gather(C2, tf.gather(idx2, E))
        b = tf.gather(b_tab, E)
        W, b = _apply_cache(rows16, W16, b16, W, b, E)
        W, b = _apply_cache(rows32, W32, b32, W, b, E)
        return tf.reduce_sum(h * W, axis=1) + b
    return query_fn

# ───────────────────────── benchmark wrapper ─────────────────────
def benchmark(table:str, batch:int, tmin:float, tmax:float, policy:str):
    mixed_precision.set_global_policy(policy)
    dtype = tf.float16 if policy != "float32" else tf.float32
    use_xla = policy != "float32"      # XLA off when pure fp32

    if not use_xla:
        tf.config.optimizer.set_jit(False)

    load_table(table, dtype)
    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch], tmin, tmax, dtype=dtype)
        E = tf.random.uniform([batch], 4, 90000, dtype=tf.int32)

    query = _make_query_fn(use_xla)
    _ = query(T[:1], E[:1])            # warm-up

    t0 = time.perf_counter()
    xs = query(T, E)
    _  = xs.numpy()
    dt = (time.perf_counter() - t0) * 1e6 / batch
    mode = "XLA" if use_xla else "no-XLA"
    print(f"{batch:,} pts → {dt:.2f} µs/pt   ({policy}, {mode})")
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
    WINDOW_SAMPS = 4
    STEP_SAMPS   = 2
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",  default="w_table_ahvq.h5")
    ap.add_argument("--batch",  type=int, default=1_000_000)
    ap.add_argument("--tmin",   type=float, default=1000.0)
    ap.add_argument("--tmax",   type=float, default=1000.0)
    ap.add_argument("--policy", choices=["float16","mixed_float16","float32"],
                    default="float32")
    ap.add_argument("--accuracy", action="store_true")
    args = ap.parse_args()

    xs, T, E = benchmark(args.table, args.batch,
                         args.tmin, args.tmax, args.policy)

    if args.accuracy:
        from __main__ import init_full, analyse   # your originals
        base_e = init_full()
        analyse(base_e, xs, T, E, TEMP_DATA_DIR)
