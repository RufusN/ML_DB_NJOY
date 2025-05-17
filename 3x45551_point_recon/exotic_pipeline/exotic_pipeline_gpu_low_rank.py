#!/usr/bin/env python3
"""
pipeline_fp16_unified.py – latent-major Doppler-XS lookup with built-in accuracy
================================================================================

Supports five table layouts:
  • full fp16         (W_tab,b_tab)
  • compact fp16      (rows,W_k,b_k,K)
  • quant-full uint8  (W_q,b_tab,scale_q,zero_q)
  • quant-compact     (rows,W_q,b_k,scale_q,zero_q,K)
  • lowrank fp16      (U_r,V_r,b_tab,rank)

Benchmarks throughput on GPU, and if --accuracy is given, then:
  1. Samples N random (T,E) pairs.
  2. Runs both the **reference** full fp16 table and your **test** table.
  3. Prints max & RMS relative error (%).
  4. Emits `xs.png` and `relative_error.png` over the sample.

Example
-------
# Speed only, lowrank table
python pipeline_fp16_unified.py --table w_table_lowrank.h5 --batch 2000000

# Speed + accuracy (100k random points, vs w_table_original.h5)
python pipeline_fp16_unified.py \
    --table w_table_lowrank.h5 \
    --accuracy --samples 100000
"""
from __future__ import annotations
import argparse, time
import h5py, numpy as np, tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import os
import argparse, time, glob, os
import h5py, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

tf.config.optimizer.set_jit(True)

E_MIN, E_MAX = 1e4 * 1e-6, 1e6 * 1e-6       # MeV
WINDOW_SIZE  = 0.00004744
STEP_SIZE    = 0.00002314
H, W_TIME, C = 3, 45551, 2

BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
WEIGHTS_H5     = 'model_weights.h5'
SCALER_PATH    = '3x45551_950_1050_spec_scalers.h5'
TEMP_DATA_DIR  = r'/mnt/d/800_1200'

# Default path to your full fp16 table for accuracy comparisons
REF_TABLE = "w_table_original.h5"

# ───────────────────────── GPU globals ──────────────────────────────────
rows = None    # knot indices if needed
K    = 1       # stride

# full/compact
W_tab = b_tab = None

# quantised
W_q     = None
scale_q = zero_q = None

# lowrank
U_r = V_r = None
RANK = 0

# layer-0 + normalization
W0 = b0 = T_scale = T_mean = None
alpha = 0.0

# detected format
FMT = "full"

# ─────────────────────────── table loader ───────────────────────────────
def load_table(path: str, dtype: tf.dtypes.DType):
    """Populate globals for whichever format is found in `path`."""
    global rows, K, W_tab, b_tab, W_q, scale_q, zero_q, U_r, V_r, RANK
    global W0, b0, T_scale, T_mean, alpha, FMT

    with h5py.File(path, "r") as hf, tf.device("/GPU:0"):
        to_tf = lambda d: tf.constant(hf[d][:], dtype=dtype)
        # layer-0 constants
        W0, b0    = to_tf("W0"), to_tf("b0")
        T_scale   = to_tf("T_scale")
        T_mean    = to_tf("T_mean")
        alpha     = float(hf["alpha"][()]) if "alpha" in hf else 0.0

        # lowrank?
        if "U_r" in hf and "V_r" in hf:
            FMT  = "lowrank"
            U_r  = to_tf("U_r")      # shape (E, r)
            V_r  = to_tf("V_r")      # shape (r, 16)
            RANK = int(hf["rank"][()])
            b_tab = to_tf("b_tab")
            rows  = tf.range(tf.shape(U_r)[0], dtype=tf.int32)
            K = 1
            return

        has_Wq   = "W_q"   in hf
        has_rows = "rows"  in hf
        has_Wtab = "W_tab" in hf

        if has_Wq and not has_rows:
            FMT     = "quant_full"
            W_q     = tf.constant(hf["W_q"][:], dtype=tf.uint8)
            scale_q = to_tf("scale_q")
            zero_q  = to_tf("zero_q")
            b_tab   = to_tf("b_tab")
            rows    = tf.range(tf.shape(W_q)[0], dtype=tf.int32)
            K = 1

        elif has_Wq and has_rows:
            FMT     = "quant_comp"
            W_q     = tf.constant(hf["W_q"][:], dtype=tf.uint8)
            scale_q = to_tf("scale_q")
            zero_q  = to_tf("zero_q")
            b_tab   = to_tf("b_k")
            rows    = tf.constant(hf["rows"][:], dtype=tf.int32)
            K = int(hf["K"][()])

        elif has_Wtab:
            FMT    = "full"
            W_tab  = to_tf("W_tab")
            b_tab  = to_tf("b_tab")
            rows   = tf.range(tf.shape(W_tab)[0], dtype=tf.int32)
            K = 1

        else:
            FMT    = "compact"
            W_tab  = to_tf("W_k")
            b_tab  = to_tf("b_k")
            rows   = tf.constant(hf["rows"][:], dtype=tf.int32)
            K = int(hf["K"][()])

# ───────────────────────── query kernel ─────────────────────────────────
@tf.function(
    input_signature=[tf.TensorSpec([None], tf.float16),
                     tf.TensorSpec([None], tf.int32)],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    print("***************************'")
    """Compute XS for batches of (T, E_idx), handling all FMT cases."""
    # full / quant-full
    if FMT in ("full", "quant_full"):
        if FMT == "full":
            W_vec = tf.gather(W_tab, E_batch)
        else:
            W_vec = (tf.cast(tf.gather(W_q, E_batch), T_batch.dtype)
                     * scale_q + zero_q)
        b_vec = tf.gather(b_tab, E_batch)

    # lowrank
    elif FMT == "lowrank":
        print("------------ HEEEEEEEEREEEEEEEEEEEEE")
        U_sel = tf.gather(U_r, E_batch)       # (N, r)
        W_vec = tf.matmul(U_sel, V_r)         # back to (N, 16)
        b_vec = tf.gather(b_tab, E_batch)

    # compact / quant-comp
    else:
        segL = tf.math.floordiv(E_batch, K)
        segR = tf.minimum(segL + 1, tf.shape(rows)[0] - 1)
        t    = (tf.cast(E_batch % K, T_batch.dtype)
                / tf.cast(K, T_batch.dtype))

        if FMT == "compact":
            W_l = tf.gather(W_tab, segL)
            W_r = tf.gather(W_tab, segR)
        else:  # quant_comp
            W_l = (tf.cast(tf.gather(W_q, segL), T_batch.dtype)
                   * scale_q + zero_q)
            W_r = (tf.cast(tf.gather(W_q, segR), T_batch.dtype)
                   * scale_q + zero_q)

        b_l = tf.gather(b_tab, segL)
        b_r = tf.gather(b_tab, segR)

        W_vec = W_l + tf.expand_dims(t, 1) * (W_r - W_l)
        b_vec = b_l + t * (b_r - b_l)

    # hidden layer + dot product
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:, None], W0) + b0, alpha)
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

# ───────────────────────── benchmark harness ────────────────────────────
def benchmark(table, batch, tmin, tmax, policy):
    mixed_precision.set_global_policy(policy)
    dtype = tf.float16 if policy != "float32" else tf.float32

    # load *test* table
    load_table(table, dtype)

    # example inputs
    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch], tmin, tmax, dtype=dtype)
        E = tf.random.uniform([batch], 4, 90000, dtype=tf.int32)

    _ = query_table(T[:1], E[:1])            # warm-up
    t0 = time.perf_counter()
    xs = query_table(T, E); _ = xs.numpy()    # host-sync
    dt = (time.perf_counter() - t0) * 1e6 / batch
    print(f"{batch:,} pts  ({FMT})  → {dt:.4f} µs/pt")
    return xs, T, E

# ═════════════════════ accuracy via random comparison ═════════════════════
def accuracy_check(ref_table, test_table, samples, policy,tmin, tmax):
    print(f"\n→ Accuracy check ({samples} random points)…")
    mixed_precision.set_global_policy(policy)
    dtype = tf.float16 if policy != "float32" else tf.float32

    # load reference (full fp16) table
    load_table(ref_table, dtype)
    @tf.function
    def query_ref(T_batch, E_batch):
        # always FMT == "full" here
        W_vec = tf.gather(W_tab, E_batch)
        b_vec = tf.gather(b_tab, E_batch)
        Tn = (T_batch - T_mean) / T_scale
        h  = tf.nn.leaky_relu(tf.matmul(Tn[:, None], W0) + b0, alpha)
        return tf.reduce_sum(h * W_vec, axis=1) + b_vec

    # load test table again
    load_table(test_table, dtype)

    # sample random inputs
    T_samp = tf.random.uniform([samples], tmin, tmax, dtype=dtype)
    E_samp = tf.random.uniform([samples],
                               0, rows[-1] + 1,
                               dtype=tf.int32)


    # run both
    xs_ref  = query_ref(T_samp, E_samp)
    xs_test = query_table(T_samp, E_samp)

    # to NumPy
    r_ref  = xs_ref .numpy()
    r_test = xs_test.numpy()

    # relative errors
    rel = np.abs(r_ref - r_test) / np.maximum(1e-8, np.abs(r_ref))
    print(len(rel))
    print(f" max rel-err = {rel.max()*100:.4f} %")
    print(f" rms rel-err = {np.sqrt((rel**2).mean())*100:.4f} %")

    # make plots
    idx = np.argsort(E_samp.numpy())
    es  = E_samp.numpy()[idx]
    rr  = rel[idx]

    os.makedirs("data_reconstruct", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.scatter(es, rr*100, s=5)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Energy index"); plt.ylabel("Rel-err (%)")
    plt.title("Relative error vs E_idx"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("data_reconstruct/relative_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.hist(rel*100, bins=100)
    plt.xlabel("Rel-err (%)"); plt.ylabel("Count")
    plt.title("Error distribution (log-y)") 
    plt.yscale("log"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("data_reconstruct/error_hist.png", dpi=200)
    plt.close()

    print("Plots written to data_reconstruct/*.png")

# ═════════════════════════════════ CLI ══════════════════════════════════

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

    #srt = np.argsort(idxs)
    #threshold = 91099
    #idxx = idxs[srt]
    #srt1 = np.array([i for i in idxx if i < threshold])

    # Plot relative error (sorted)
    sorted_idx = np.sort(idxs)#np.argsort(idxs)
    rel_err_sorted = relerr
    plt.figure(figsize=(8,5))
    plt.plot(base_e[sorted_idx], rel_err_sorted, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Energy'); plt.ylabel('Relative error (%)')
    plt.title('Relative error vs energy'); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data_reconstruct/relative_error_2.png", dpi=200); plt.close()
    print("Accuracy plots saved to ./data_reconstruct")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table",     default="w_table_lowrank.h5",
                   help="Your test table (compact/quant/lowrank)")
    p.add_argument("--batch",     type=int,   default=8192)
    p.add_argument("--tmin",      type=float, default=1.0)
    p.add_argument("--tmax",      type=float, default=1000.0)
    p.add_argument("--policy",    choices=["float16","mixed_float16","float32"],
                   default="float16")
    p.add_argument("--accuracy",  action="store_true",
                   help="compare against full table with random sampling")
    p.add_argument("--samples",   type=int, default=100000,
                   help="number of random points for accuracy check")
    p.add_argument("--ref-table", default=REF_TABLE,
                   help="path to full fp16 table for reference")
    args = p.parse_args()

    # benchmark speed
    xs, t_batch, e_batch = benchmark(
        args.table, args.batch, args.tmin, args.tmax, args.policy
    )

    # optional accuracy
    if args.accuracy:
        accuracy_check(args.ref_table, args.table,
                       args.samples, args.policy, args.tmin, args.tmax)

        base_e = init_full()
        analyse(base_e, xs, t_batch, e_batch, TEMP_DATA_DIR)
