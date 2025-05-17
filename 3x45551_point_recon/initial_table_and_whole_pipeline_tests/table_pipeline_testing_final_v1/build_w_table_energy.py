#!/usr/bin/env python3
"""table_xs.py – build a tabulated decoder and query XS values
================================================================
Self‑contained utility that can **build** a pre‑baked linear table and **query**
MLP–Doppler cross‑sections.  Fixed after review: frequency axis handling and
per‑segment local‑sample gathering now match the reference pipeline exactly.
"""

from __future__ import annotations
import argparse, time, h5py, numpy as np, tensorflow as tf

# ────────────────────────── helper: segment mapping ────────────────────────

def map_segments_and_locals(e_idx: int, window: int, step: int):
    """Return the n_over segment indices and the *local sample index* inside
    every segment that together form energy sample e_idx (matches original
    `mapSegIdx` and GPU code)."""
    n_over = window // step                      # 2 when window=4, step=2
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window         # per‑segment offset
    return segs, local                                     # (n_over,) each

# ────────────────────────── table builder  ─────────────────────────────────

def build_table(weights_h5: str, scaler_h5: str,
                window: int, step: int, out_h5: str = "w_table.h5"):
    H, W_TIME, C = 3, 45_551, 2                 # trained model constants
    LATENT = 16
    n_over = window // step                     # 2 for 4/2 pair

    print(f"[build] window={window}, step={step}, overlaps={n_over}")

    # ── load network weights ───────────────────────────────────────────────
    with h5py.File(weights_h5, "r") as hf:
        W0    = hf["W0"][:].astype("float32")           # (1,16)
        b0    = hf["b0"][:].astype("float32")           # (16,)
        alpha = float(hf["alpha"][()])
        W_dec = hf["W_dec"][:].astype("float32")        # (16, H*W*C)
        b_dec = hf["b_dec"][:].astype("float32")        # (H*W*C,)

    # ── load scalers ───────────────────────────────────────────────────────
    with h5py.File(scaler_h5, "r") as hf:
        spec_scale = hf["spec_scale"][:].astype("float32")
        spec_mean  = hf["spec_mean"] [:].astype("float32")
        T_scale    = hf["T_scale"]   [:].astype("float32")
        T_mean     = hf["T_mean"]    [:].astype("float32")

    hann = np.hanning(window).astype("float32")
    scale_fac = np.sum(hann**2) / step          # overlap‑add correction

    W_tab = np.zeros((W_TIME, LATENT), dtype="float32")
    b_tab = np.zeros(W_TIME,              dtype="float32")

    flat_stride_f = W_TIME * C                  # stride for channel layout

    t0 = time.perf_counter()
    for e_idx in range(W_TIME):
        segs, local = map_segments_and_locals(e_idx, window, step)

        # build list of flat spectrogram indices (real / imag interleaved) --
        flat_idx = []
        for f in range(H):
            for s in segs:
                base = f * flat_stride_f + s * C
                flat_idx.extend((base, base + 1))
        flat_idx = np.asarray(flat_idx, np.int32)

        # slice decoder weights/bias & un‑scale ----------------------------
        W_sub = W_dec[:, flat_idx] * spec_scale[flat_idx]              # (16,n)
        b_sub = b_dec[ flat_idx]  * spec_scale[flat_idx] + spec_mean[flat_idx]

        # reshape to complex (latent, H, n_over) and bring freq axis last --
        W_c = (W_sub.reshape(LATENT, H, n_over, C)[..., 0] +
                1j * W_sub.reshape(LATENT, H, n_over, C)[..., 1])      # (16,H,n)
        b_c = (b_sub.reshape(        H, n_over, C)[..., 0] +
                1j * b_sub.reshape(        H, n_over, C)[..., 1])      # (H,n)

        W_c = np.swapaxes(W_c, -2, -1)   # (16,n_over,H) → freq axis last
        b_c = np.swapaxes(b_c,  -2, -1)   # (n_over,H)

        # iRFFT over *frequency* bins --------------------------------------
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann            # (16,n_over,window)
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann            # (n_over,window)

        # gather per‑segment local sample and sum overlaps -----------------
        coeff = np.zeros(LATENT, dtype="float32")
        bias  = 0.0
        for k, loc in enumerate(local):
            coeff += seg_W[:, k, loc]
            bias  += seg_b[  k, loc]
        coeff /= scale_fac
        bias  /= scale_fac

        W_tab[e_idx] = coeff.real
        b_tab[e_idx] = bias.real

        if e_idx % 5000 == 0:
            print(f"  • {e_idx}/{W_TIME} rows done")

    print(f"[build] finished in {time.perf_counter()-t0:.1f}s → {out_h5}")

    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"], hf["b_tab"] = W_tab, b_tab
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["window"], hf["step"] = window, step
        hf["E_idxs"] = np.arange(W_TIME, dtype="int32")

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
