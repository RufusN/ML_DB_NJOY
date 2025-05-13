#!/usr/bin/env python3
"""
table_xs_fp16.py – build a **float16** lookup table and run FP16 queries
=======================================================================
Changes vs original
-------------------
• Added --dtype {float16,float32} (default float16)  
• Everything written to HDF5 is cast to the requested dtype  
• Internal accumulation still uses float32 for numerical stability, but the
  final table is half-precision, cutting the file size in half.
"""

from __future__ import annotations
import argparse, time, h5py, numpy as np, tensorflow as tf

# ────────────────────────── helper: segment mapping ────────────────────────

def map_segments_and_locals(e_idx: int, window: int, step: int):
    n_over = window // step
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

# ────────────────────────── table builder  ─────────────────────────────────

def build_table(weights_h5: str, scaler_h5: str,
                window: int, step: int,
                out_h5: str = "w_table.h5",
                out_dtype: str = "float16"):
    H, C = 3, 2
    out_np_dtype = np.float16 if out_dtype == "float16" else np.float32

    with h5py.File(weights_h5, "r") as hf_w:
        full_width = hf_w["W_dec"].shape[1]
        W0    = hf_w["W0"][:].astype("float32")
        b0    = hf_w["b0"][:].astype("float32")
        alpha = float(hf_w["alpha"][()])
        W_dec = hf_w["W_dec"][:].astype("float32")
        b_dec = hf_w["b_dec"][:].astype("float32")

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    LATENT = 16
    n_over = window // step
    print(f"[build] window={window}, step={step}, overlaps={n_over}, rows={E_ROWS}")

    with h5py.File(scaler_h5, "r") as hf_s:
        spec_scale = hf_s["spec_scale"][:].astype("float32")
        spec_mean  = hf_s["spec_mean"] [:].astype("float32")
        T_scale    = hf_s["T_scale"]   [:].astype("float32")
        T_mean     = hf_s["T_mean"]    [:].astype("float32")

    hann = np.hanning(window).astype("float32")
    scale_fac = np.sum(hann**2) / step

    # latent-major already, so no transpose needed later
    W_tab = np.zeros((E_ROWS, LATENT), dtype="float32")
    b_tab = np.zeros(E_ROWS,              dtype="float32")
    flat_stride_f = N_STFT * C

    t0 = time.perf_counter()
    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue
        flat_idx = []
        for f in range(H):
            for s in segs:
                base = f * flat_stride_f + s * C
                flat_idx.extend((base, base + 1))
        flat_idx = np.asarray(flat_idx, np.int32)
        W_sub = W_dec[:, flat_idx] * spec_scale[flat_idx]
        b_sub = b_dec[ flat_idx]  * spec_scale[flat_idx] + spec_mean[flat_idx]
        W_c = W_sub.reshape(LATENT, H, n_over, 2)
        b_c = b_sub.reshape(       H, n_over, 2)
        W_c = (W_c[...,0] + 1j*W_c[...,1]).transpose(0,2,1)
        b_c = (b_c[...,0] + 1j*b_c[...,1]).transpose(1,0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann
        coeff = seg_W[np.arange(LATENT)[:,None], np.arange(n_over), local].sum(axis=1) / scale_fac
        bias  = seg_b[np.arange(n_over),             local].sum()            / scale_fac
        W_tab[e_idx] = coeff.real
        b_tab[e_idx] = bias.real
        if (e_idx & 0x1FFF) == 0:   # ~8k rows
            done = 100 * e_idx / E_ROWS
            print(f"  • {done:5.1f}%")

    print(f"[build] finished in {time.perf_counter()-t0:.1f}s → {out_h5}")

    # ---------- save (optionally down-cast to FP16) ----------
    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"]   = W_tab.astype(out_np_dtype)
        hf["b_tab"]   = b_tab.astype(out_np_dtype)
        hf["W0"]      = W0.astype(out_np_dtype)
        hf["b0"]      = b0.astype(out_np_dtype)
        hf["alpha"]   = alpha                       # scalar → saves as float64
        hf["T_scale"] = T_scale.astype(out_np_dtype)
        hf["T_mean"]  = T_mean.astype(out_np_dtype)
        hf["window"], hf["step"] = window, step

    print(f"[build] table saved ({out_dtype}) ✔︎")

# ────────────────────────── CLI wiring ─────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    bld = sub.add_parser("build", help="create w_table.h5 from weights/scalers")
    bld.add_argument("--weights", required=True)
    bld.add_argument("--scaler",  required=True)
    bld.add_argument("--window",  type=int, required=True)
    bld.add_argument("--step",    type=int, required=True)
    bld.add_argument("--out",     default="w_table.h5")
    bld.add_argument("--dtype",   choices=["float16","float32"], default="float16",
                     help="storage precision for the table")

    qry = sub.add_parser("query", help="query one XS value from the table")
    qry.add_argument("--T",      type=float, required=True)
    qry.add_argument("--E_idx",  type=int,   required=True)
    qry.add_argument("--table",  default="w_table.h5")

    args = p.parse_args()
    if args.cmd == "build":
        build_table(args.weights, args.scaler,
                    args.window, args.step,
                    args.out,   args.dtype)
    else:
        val = query_xs(args.table, args.T, args.E_idx)
        print(f"XS(T={args.T}, E_idx={args.E_idx}) = {val:.8e}")

# fast query (unchanged except for automatic dtype handling)
@tf.function
def _tf_query(T_batch, E_batch,
              W_tab, b_tab, W0, b0, alpha,
              T_scale, T_mean):
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:, None], W0) + b0, alpha)
    W_vec  = tf.gather(W_tab, E_batch)
    b_vec  = tf.gather(b_tab, E_batch)
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec

def query_xs(table_h5: str, T: float, E_idx: int):
    with h5py.File(table_h5, "r") as hf:
        cast = lambda d: tf.constant(hf[d][:])
        W_tab = cast("W_tab"); b_tab = cast("b_tab")
        W0    = cast("W0");    b0    = cast("b0")
        alpha = float(hf["alpha"][()])
        T_scale = cast("T_scale"); T_mean = cast("T_mean")
    xs = _tf_query(tf.constant([T]), tf.constant([E_idx]),
                   W_tab, b_tab, W0, b0, alpha, T_scale, T_mean)
    return float(xs.numpy()[0])

if __name__ == "__main__":
    main()
