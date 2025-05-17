#!/usr/bin/env python3
"""
pipeline_fp16_unified.py – latent-major Doppler-XS lookup
========================================================
A single script that recognises *four* table layouts:

──────────────────────────────────────────────────────────────────────────
 layout name        required datasets in the HDF5 file
──────────────────────────────────────────────────────────────────────────
  full fp16         W_tab (E,16)      b_tab (E,)
  compact fp16      rows (N_K,)       W_k  (N_K,16)   b_k (N_K,)    K
  quant-full uint8  W_q  (E,16 uint8) b_tab (E,)      scale_q zero_q
  quant-compact     rows              W_q (N_K,16)    b_k            K
                                               + the same scale_q zero_q
──────────────────────────────────────────────────────────────────────────

The kernel automatically chooses:
    • direct gather                     (full / quant-full)
    • linear interpolation              (compact / quant-compact)
    • uint8 de-quantisation if present

CLI usage is identical for every format:

    python pipeline_fp16_unified.py \
           --table w_table_quant.h5   --batch 2_000_000 --accuracy
"""
from __future__ import annotations
import argparse, time, glob, os
import h5py, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d

tf.config.optimizer.set_jit(True)

# ─────────────────────────── domain constants ──────────────────────────
E_MIN, E_MAX = 1e4 * 1e-6, 1e6 * 1e-6    # MeV
WINDOW_SIZE  = 0.00004744
STEP_SIZE    = 0.00002314
H, W_TIME, C = 3, 45551, 2

BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
TEMP_DATA_DIR  = r'/mnt/d/800_1200'

# ──────────────────────────── GPU constants ────────────────────────────
rows = None                 # int32  knot indices
K    = 1                    # stride for compact variants
# weight & bias stores
W_tab = b_tab = None        # fp16 (full / compact)
W_q   = None                # uint8 (quantised)
scale_q = zero_q = None     # fp16 de-quant parameters
# layer-0 & temperature scalers
W0 = b0 = T_scale = T_mean = None
alpha = 0.0
FMT = "full"                # detected layout

# ═════════════════════════════ loader ═══════════════════════════════════
def load_table(table_h5: str, dtype: tf.dtypes.DType):
    """
    Upload all constants to the GPU and set the global `FMT` flag.
    """
    global rows, K, W_tab, b_tab, W_q, scale_q, zero_q
    global W0, b0, T_scale, T_mean, alpha, FMT

    with h5py.File(table_h5, "r") as hf, tf.device("/GPU:0"):
        c = lambda d: tf.constant(hf[d][:], dtype=dtype)

        # common tensors
        W0, b0    = c("W0"), c("b0")
        T_scale   = c("T_scale");  T_mean = c("T_mean")
        alpha     = float(hf["alpha"][()])

        has_Wq   = "W_q"   in hf
        has_rows = "rows"  in hf
        has_Wtab = "W_tab" in hf

        if has_Wq and not has_rows:          # ─── quant-full ───
            FMT    = "quant_full"
            W_q    = tf.constant(hf["W_q"][:], dtype=tf.uint8)
            scale_q = c("scale_q"); zero_q = c("zero_q")
            b_tab   = c("b_tab")
            rows    = tf.range(tf.shape(W_q)[0], dtype=tf.int32)
            K       = 1

        elif has_Wq and has_rows:            # ─ quant-compact ─
            FMT    = "quant_comp"
            W_q    = tf.constant(hf["W_q"][:], dtype=tf.uint8)
            scale_q = c("scale_q"); zero_q = c("zero_q")
            b_tab   = c("b_k")
            rows    = tf.constant(hf["rows"][:], dtype=tf.int32)
            K       = int(hf["K"][()])

        elif has_Wtab:                       # ─── full fp16 ───
            FMT   = "full"
            W_tab = c("W_tab");  b_tab = c("b_tab")
            rows  = tf.range(tf.shape(W_tab)[0], dtype=tf.int32)
            K     = 1

        else:                                # ─ compact fp16 ─
            FMT   = "compact"
            W_tab = c("W_k");    b_tab = c("b_k")
            rows  = tf.constant(hf["rows"][:], dtype=tf.int32)
            K     = int(hf["K"][()])

# ═════════════════════════════ query kernel ═════════════════════════════
@tf.function(
    input_signature=[tf.TensorSpec([None], tf.float16),
                     tf.TensorSpec([None], tf.int32)],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    """
    Vectorised cross-section lookup for (T, E_idx) batches.
    Automatically:
        • de-quantises uint8 weights if needed
        • linearly interpolates when K > 1
    """
    # ── gather / interpolate weights ──────────────────────────────────
    if FMT in ("full", "quant_full"):
        if FMT == "full":
            W_vec = tf.gather(W_tab, E_batch)
        else:
            W_vec = (tf.cast(tf.gather(W_q, E_batch), T_batch.dtype)
                     * scale_q + zero_q)
        b_vec = tf.gather(b_tab, E_batch)

    else:  # compact variants
        segL = tf.math.floordiv(E_batch, K)
        segR = tf.minimum(segL + 1, tf.shape(rows)[0] - 1)
        t    = (tf.cast(E_batch % K, T_batch.dtype)
                / tf.cast(K, T_batch.dtype))

        if FMT == "compact":
            W_l = tf.gather(W_tab, segL); W_r = tf.gather(W_tab, segR)
        else:  # quant_comp
            W_l = (tf.cast(tf.gather(W_q, segL), T_batch.dtype)
                   * scale_q + zero_q)
            W_r = (tf.cast(tf.gather(W_q, segR), T_batch.dtype)
                   * scale_q + zero_q)
        b_l  = tf.gather(b_tab, segL);  b_r = tf.gather(b_tab, segR)

        W_vec = W_l + tf.expand_dims(t, 1) * (W_r - W_l)
        b_vec = b_l + t * (b_r - b_l)

    # ── hidden layer & dot product ────────────────────────────────────
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:, None], W0) + b0, alpha)
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

# ═════════════════════ benchmark / accuracy harness ════════════════════
def benchmark(table, batch, tmin, tmax, policy):
    mixed_precision.set_global_policy(policy)
    dtype = tf.float16 if policy != "float32" else tf.float32
    load_table(table, dtype)

    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch], tmin, tmax, dtype=dtype)
        E = tf.random.uniform([batch], 0, rows[-1] + 1, dtype=tf.int32)

    _ = query_table(T[:1], E[:1])          # warm-up
    t0 = time.perf_counter()
    xs = query_table(T, E); _ = xs.numpy()
    dt = (time.perf_counter() - t0)*1e6 / batch
    print(f"{batch:,} points   ({FMT})   → {dt:.4f} µs/point")
    return xs, T, E

# ─── helpers for optional ground-truth accuracy plotting ───────────────
def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key='xs_data', compression='gzip')
    sub = df[(df['T'] == base_t) &
             (df['ERG'] >= e_min) &
             (df['ERG'] <= e_max)]
    e_sorted = np.sort(sub['ERG'].to_numpy())
    return e_sorted, len(e_sorted)

def init_full():
    global fs, WINDOW_SAMPS, STEP_SAMPS, PAD
    base_e, fs = load_base(E_MIN, E_MAX)
    WINDOW_SAMPS = int(WINDOW_SIZE * fs)
    STEP_SAMPS   = int(STEP_SIZE  * fs)
    PAD = WINDOW_SAMPS
    return base_e

def load_temperature(T_val, base_e, data_dir):
    for fp in glob.glob(os.path.join(data_dir, "*.h5")):
        df = pd.read_hdf(fp, key="xs_data", compression="gzip")
        if T_val not in df["T"].values:
            continue
        sub = df[(df["T"] == T_val) &
                 (df["ERG"] >= E_MIN) &
                 (df["ERG"] <= E_MAX)]
        if len(sub) < 2:
            continue
        E, xs = sub["ERG"].to_numpy(), sub["XS"].to_numpy()
        idx = np.argsort(E)
        interp = interp1d(E[idx], xs[idx],
                          kind="cubic", fill_value="extrapolate")
        return (np.pad(base_e, (PAD, PAD), mode="constant"),
                np.pad(interp(base_e), (PAD, PAD), mode="constant"))
    return None, None

def analyse(base_e, xs_rec, temps, idxs, data_dir):
    pad_e, pad_ref = load_temperature(float(temps[0]), base_e, data_dir)
    if pad_ref is None:
        print("Ground-truth data not available – skipping plots.")
        return
    xs_rec = xs_rec.numpy(); idxs = idxs.numpy()
    rel = np.abs(xs_rec - pad_ref[idxs]) / np.abs(pad_ref[idxs]) * 100

    os.makedirs("data_reconstruct", exist_ok=True)
    # XS curve
    plt.figure(figsize=(8,5))
    plt.plot(pad_e, pad_ref, label="truth")
    plt.scatter(pad_e[idxs], xs_rec, s=8, c='r', label="recon")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Energy"); plt.ylabel("XS")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("data_reconstruct/xs.png", dpi=200); plt.close()
    # rel-error curve
    ord = np.argsort(idxs)
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs[ord]-PAD], rel[ord], marker='o')
    plt.xscale("log"); plt.xlabel("Energy")
    plt.ylabel("Relative error (%)")
    plt.title("Relative error vs energy"); plt.grid(True); plt.tight_layout()
    plt.savefig("data_reconstruct/relative_error.png", dpi=200); plt.close()
    print("Plots saved in ./data_reconstruct")

# ═════════════════════════════ CLI front end ════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table",  default="w_table_quant.h5")
    p.add_argument("--batch",  type=int,   default=8192)
    p.add_argument("--tmin",   type=float, default=1.0)
    p.add_argument("--tmax",   type=float, default=1000.0)
    p.add_argument("--policy", choices=["float16", "mixed_float16", "float32"],
                   default="float16")
    p.add_argument("--accuracy", action="store_true",
                   help="produce ground-truth error plots")
    args = p.parse_args()

    xs, T, E_idx = benchmark(args.table, args.batch,
                             args.tmin, args.tmax, args.policy)

    if args.accuracy:
        base_e = init_full()
        analyse(base_e, xs, T, E_idx, TEMP_DATA_DIR)
