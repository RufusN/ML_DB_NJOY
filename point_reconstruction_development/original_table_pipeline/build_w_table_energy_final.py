#!/usr/bin/env python3
"""table_xs.py – build a pre‑baked table and run **single** or **batched** queries
================================================================================
Now covers **every reconstructable energy index**:
for a window/hop pair `(window, step)` the number of rows is
```
E_ROWS = 2 * N_STFT + (window - 1)
```
where `N_STFT = W_dec.shape[1] // (3 × 2)` – i.e. all STFT frames that were
actually seen during training.  For your 4/2 setting that means
`E_ROWS = 91 105`, so indices up to 60 000 are valid.
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

def build_table(weights_h5: str, scaler_h5: str, window: int, step: int,
                out_h5: str = "w_table.h5"):
    H, C = 3, 2
    with h5py.File(weights_h5, "r") as hf_w:
        full_width = hf_w["W_dec"].shape[1]
        W0    = hf_w["W0"][:].astype("float32")
        b0    = hf_w["b0"][:].astype("float32")
        alpha = float(hf_w["alpha"][()])
        W_dec = hf_w["W_dec"][:].astype("float32")
        b_dec = hf_w["b_dec"][:].astype("float32")

    N_STFT = full_width // (H * C)                 # 45 551 for 4/2 example
    E_ROWS = 2 * N_STFT + (window - 1)             # 91 105 for 4/2
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

    W_tab = np.zeros((E_ROWS, LATENT), dtype="float32")
    b_tab = np.zeros(E_ROWS,              dtype="float32")
    flat_stride_f = N_STFT * C

    t0 = time.perf_counter()
    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            # requires unseen STFT column → leave zeros (same as pipeline)
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
        if e_idx % 10000 == 0:
            print(f"  • {e_idx}/{E_ROWS} rows done")

    print(f"[build] finished in {time.perf_counter()-t0:.1f}s → {out_h5}")

    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"], hf["b_tab"] = W_tab, b_tab
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["window"], hf["step"] = window, step

    print("[build] table saved ✔︎")

# ────────────────────────── fast query  ────────────────────────────────────

@tf.function
def _tf_query(T_batch, E_batch,
             W_tab, b_tab, W0, b0, alpha, T_scale, T_mean):
    T_norm = (T_batch - T_mean) / T_scale                      # [N]
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:, None], W0) + b0, alpha)  # [N,16]
    W_vec  = tf.gather(W_tab, E_batch)                         # [N,16]
    b_vec  = tf.gather(b_tab, E_batch)                         # [N]
    return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec       # [N]


def query_xs(table_h5: str, T: float, E_idx: int):
    with h5py.File(table_h5, "r") as hf:
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])

    xs = _tf_query(tf.constant([T], tf.float32), tf.constant([E_idx], tf.int32),
                   W_tab, b_tab, W0, b0, alpha, T_scale, T_mean)
    return float(xs.numpy()[0])

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

    qry = sub.add_parser("query", help="query one XS value from the table")
    qry.add_argument("--T",      type=float, required=True)
    qry.add_argument("--E_idx",  type=int,   required=True)
    qry.add_argument("--table",  default="w_table.h5")

    args = p.parse_args()
    if args.cmd == "build":
        build_table(args.weights, args.scaler, args.window, args.step, args.out)
    else:
        val = query_xs(args.table, args.T, args.E_idx)
        print(f"XS(T={args.T}, E_idx={args.E_idx}) = {val:.8e}")

if __name__ == "__main__":
    main()
