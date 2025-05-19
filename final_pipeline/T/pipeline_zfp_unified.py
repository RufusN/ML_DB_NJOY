#!/usr/bin/env python3
"""pipeline_fp16_hvq.py – **HVQ / AHVQ** inference harness (v4)
================================================================
Fixes the TensorFlow indexing error and clarifies accuracy‑plot axes.

Changes
-------
1. **Tensor indexing** – replaced `C1[tf.gather(idx1,E)]` with
   `tf.gather(C1, tf.gather(idx1, E))`, which is the legal way to gather
   from a constant with a tensor index.
2. **Query kernels** – same fix applied to AHVQ branch.
3. **Accuracy plot** – the scatter X‑coords were off by −4 when `PAD>0`.
4. **Docstring** – notes why a *topP = 100* AHVQ table is ~3.7 MB (fp16)
   whereas the original dense fp32 table was ~6 MB.
"""
from __future__ import annotations
import argparse, time, os, glob, h5py, numpy as np, pandas as pd, tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.keras import mixed_precision

E_MIN, E_MAX = 1e4 * 1e-6, 1e6 * 1e-6
BASE_FILE_PATH = r"/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
TEMP_DATA_DIR  = r"/mnt/d/800_1200"

# ────────────────────────── GPU globals ───────────────────────────
W0 = b0 = T_scale = T_mean = None
alpha: float = 0.0
C1 = C2 = idx1 = idx2 = b_tab = None
exc_rows = W_exc = b_exc = None
MODE = "HVQ"  # or "AHVQ"

# ══════════════════════════ Loader ════════════════════════════════

def load_table(fp: str, dtype=tf.float16):
    global W0, b0, T_scale, T_mean, alpha
    global C1, C2, idx1, idx2, b_tab, MODE, exc_rows, W_exc, b_exc

    with h5py.File(fp) as f, tf.device("/GPU:0"):
        c = lambda k: tf.constant(f[k][:], dtype=dtype)
        W0, b0          = c("W0"), c("b0")
        T_scale, T_mean = c("T_scale"), c("T_mean")
        alpha           = float(f["alpha"][()])

        C1, C2 = c("C1"), c("C2")
        idx1   = tf.constant(f["idx1"][:], tf.int32)
        idx2   = tf.constant(f["idx2"][:], tf.int32)
        b_tab  = c("b_tab")

        if "AHVQ" in f.attrs:
            MODE      = "AHVQ"
            exc_rows  = tf.constant(f["exc_rows"][:], tf.int32)
            W_exc     = c("W_exc")
            b_exc     = c("b_exc")
        else:
            MODE     = "HVQ"
            exc_rows = tf.constant([], tf.int32)  # empty tensor
    print(f"[load] {MODE} table loaded ({dtype.name})")

# ═══════════════════════ helper: MLP hidden ═══════════════════════
@tf.function(experimental_compile=True)
def _mlp_hidden(T):
    T = tf.cast(T, T_scale.dtype)
    return tf.nn.leaky_relu(tf.matmul(((T - T_mean) / T_scale)[:, None], W0) + b0, alpha)

# ═══════════════════════ Query kernels ═══════════════════════════=
@tf.function(experimental_compile=True)
def query_hvq(T, E):
    h = _mlp_hidden(T)
    W = tf.gather(C1, tf.gather(idx1, E)) + tf.gather(C2, tf.gather(idx2, E))
    return tf.reduce_sum(h * W, axis=1) + tf.gather(b_tab, E)

@tf.function(experimental_compile=True)
def query_ahvq(T, E):
    h = _mlp_hidden(T)
    W_base = tf.gather(C1, tf.gather(idx1, E)) + tf.gather(C2, tf.gather(idx2, E))
    b_base = tf.gather(b_tab, E)

    def no_exc():
        return W_base, b_base

    def with_exc():
        n = tf.shape(exc_rows)[0]
        rows_pad = tf.concat([exc_rows, [-1]], 0)
        W_pad = tf.concat([W_exc, tf.zeros([1, tf.shape(W_exc)[1]], W_exc.dtype)], 0)
        b_pad = tf.concat([b_exc, tf.zeros([1], b_exc.dtype)], 0)

        idx = tf.searchsorted(exc_rows, E, side="left")
        idx_c = tf.minimum(idx, n)
        m = tf.equal(tf.gather(rows_pad, idx_c), E)

        W = tf.where(m[:, None], tf.gather(W_pad, idx_c), W_base)
        b = tf.where(m, tf.gather(b_pad, idx_c), b_base)
        return W, b

    W_vec, b_vec = tf.cond(tf.size(exc_rows) > 0, with_exc, no_exc)
    return tf.reduce_sum(h * W_vec, axis=1) + b_vec

# ═══════════════════════ Benchmark ═══════════════════════════════

def benchmark(table: str, batch: int, tmin: float, tmax: float, policy: str):
    if batch > 5_000_000:
        raise SystemExit("Batch too large – split into <5 M chunks to avoid CUDA tiling limits")

    mixed_precision.set_global_policy(policy)
    dtype = tf.float16 if policy != "float32" else tf.float32
    load_table(table, dtype)

    T = tf.random.uniform([batch], tmin, tmax, dtype=dtype)
    E = tf.random.uniform([batch], 4, 90000, dtype=tf.int32)

    kernel = query_ahvq if MODE == "AHVQ" else query_hvq
    _ = kernel(T[:1], E[:1])  # warm‑up

    t0 = time.perf_counter()
    xs = kernel(T, E)
    _  = xs.numpy()
    dur = 1e6 * (time.perf_counter() - t0) / batch
    print(f"{batch:,} pts → {dur:.2f} µs/pt  [{MODE}, {policy}]")
    return xs, T, E

# ═══════════════════════ Accuracy helpers (optional) ═════════════


PAD = 4

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
    plt.plot(base_e, padded[4:-4], c = 'black')
    plt.scatter(padded_e[idxs-4], xs_vals, c='r', s=10)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy'); plt.ylabel('XS')
    #plt.legend(); plt.grid(True)
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

# ═══════════════════════ CLI ═══════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",  default="w_table_ahvq.h5")
    ap.add_argument("--batch",  type=int, default=1_000_000)
    ap.add_argument("--tmin",   type=float, default=1000.0)
    ap.add_argument("--tmax",   type=float, default=1000.0)
    ap.add_argument("--policy", choices=["float16", "mixed_float16", "float32"], default="float16")
    ap.add_argument("--accuracy", action="store_true")
    args = ap.parse_args()

    xs, T, E = benchmark(args.table, args.batch, args.tmin, args.tmax, args.policy)
    if args.accuracy:
        base_e = init_full()
        analyse(base_e, xs, T, E, TEMP_DATA_DIR)
