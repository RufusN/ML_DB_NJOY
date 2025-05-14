#!/usr/bin/env python3
"""table_xs_fp16.py – pure‑fp16 Doppler‑XS lookup & benchmark
-----------------------------------------------------------
Everything—the table, hidden layer, matmul—is kept in **float16**.
Expect ~1.6× speed‑up vs. fp32 accumulate with ≲0.3 % RMS error.
"""
from __future__ import annotations
import argparse, time, h5py, numpy as np, tensorflow as tf

# ═════════════ helper ════════════════════════════════════════════════════

def map_segments_and_locals(e_idx: int, window: int, step: int):
    n_over = window // step
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

# ═════════════ builder (latent‑major, fp16) ═════════════════════════════=

def build_table(weights_h5: str, scaler_h5: str, window: int, step: int,
                out_h5: str = "w_table_fp16.h5"):
    dt = np.float16
    H, C, LATENT = 3, 2, 16
    with h5py.File(weights_h5, "r") as hf_w:
        full_width = hf_w["W_dec"].shape[1]
        W0, b0 = hf_w["W0"][:].astype(dt), hf_w["b0"][:].astype(dt)
        alpha  = float(hf_w["alpha"][()])
        W_dec, b_dec = hf_w["W_dec"][:].astype(dt), hf_w["b_dec"][:].astype(dt)

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step

    with h5py.File(scaler_h5, "r") as hf_s:
        spec_scale = hf_s["spec_scale"][:].astype(dt)
        spec_mean  = hf_s["spec_mean"] [:].astype(dt)
        T_scale    = hf_s["T_scale"]   [:].astype(dt)
        T_mean     = hf_s["T_mean"]    [:].astype(dt)

    hann = np.hanning(window).astype(dt)
    scale_fac = np.sum(hann**2) / step

    W_tab = np.zeros((LATENT, E_ROWS), dtype=dt)   # latent‑major
    b_tab = np.zeros(E_ROWS, dtype=dt)
    flat_stride_f = N_STFT * C

    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue
        flat_idx = []
        for f in range(H):
            for s in segs:
                flat_idx.extend((f * flat_stride_f + s * C, f * flat_stride_f + s * C + 1))
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
        W_tab[:, e_idx] = coeff.real
        b_tab[e_idx]    = bias.real

    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"], hf["b_tab"] = W_tab, b_tab
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["window"], hf["step"], hf["dtype"] = window, step, "float16"
    print("[build] fp16 table saved ✔️")

# ═════════════ fast fp16 query / batch ═══════════════════════════════════

def _tf_query16(T_batch, E_batch, W_tab, b_tab, W0, b0, alpha, T_scale, T_mean):
    hidden = tf.nn.leaky_relu(tf.matmul((T_batch - T_mean)[:,None] / T_scale, W0) + b0, alpha)
    W_sel  = tf.gather(W_tab, E_batch, axis=1)
    b_sel  = tf.gather(b_tab, E_batch)
    return tf.matmul(hidden, W_sel) + b_sel


def query_xs(table_h5: str, T: float, E_idx: int):
    with h5py.File(table_h5, "r") as hf:
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])
    xs = _tf_query16(tf.constant([T], tf.float16), tf.constant([E_idx], tf.int32),
                     W_tab, b_tab, W0, b0, alpha, T_scale, T_mean)
    return float(xs.numpy()[0])


def batch_query(table_h5: str, T_arr, E_arr, use_gpu: bool):
    with h5py.File(table_h5, "r") as hf:
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])

    device = "/GPU:0" if use_gpu and tf.config.list_logical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        T_tf = tf.constant(T_arr, tf.float16)
        hidden = tf.nn.leaky_relu(tf.matmul(((T_tf - T_mean)/T_scale)[:,None], W0) + b0, alpha)
        W_sel = tf.gather(W_tab, tf.constant(E_arr, tf.int32), axis=1)
        b_sel = tf.gather(b_tab, tf.constant(E_arr, tf.int32))
        start = time.perf_counter()
        xs_np = (tf.matmul(hidden, W_sel) + b_sel).numpy()
        dur = time.perf_counter() - start
    print(f"[batch] {xs_np.shape} on {device} in {dur*1e3:.1f} ms → {dur/xs_np.size*1e6:6f} µs/value")
    return xs_np

# ═════════════ tiny CLI ═════════════════════════════════════════════════=

def _parse(arg: str, as_int=False):
    if ":" in arg:
        a, b, n = map(float, arg.split(":"))
        arr = np.linspace(a, b, int(n))
    else:
        arr = np.fromstring(arg, sep=",")
    return arr.astype("int32" if as_int else "float32")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

        # ── build ─────────────────────────────────────────────
    b = sub.add_parser("build")
    b.add_argument("--weights", required=True)
    b.add_argument("--scaler",  required=True)
    b.add_argument("--window",  type=int, required=True)
    b.add_argument("--step",    type=int, required=True)
    b.add_argument("--out",     default="w_table_fp16.h5")

    # ── query ─────────────────────────────────────────────
    q = sub.add_parser("query")
    q.add_argument("--T",      type=float, required=True)
    q.add_argument("--E_idx",  type=int,   required=True)
    q.add_argument("--table",  default="w_table_fp16.h5")

    # ── batch ─────────────────────────────────────────────
    batch = sub.add_parser("batch")
    batch.add_argument("--T",      required=True, help="start:stop:num or list")
    batch.add_argument("--E",      required=True, help="start:stop:num or list")
    batch.add_argument("--table",  default="w_table_fp16.h5")
    batch.add_argument("--device", choices=["CPU", "GPU"], default="GPU")

    args = p.parse_args()

    if args.cmd == "build":
        build_table(args.weights, args.scaler, args.window, args.step, args.out)

    elif args.cmd == "query":
        val = query_xs(args.table, args.T, args.E_idx)
        print(f"XS(T={args.T}, E_idx={args.E_idx}) = {val:.8e}")

    else:  # batch
        T_arr = _parse(args.T)
        E_arr = _parse(args.E, as_int=True)
        batch_query(args.table, T_arr, E_arr, use_gpu=(args.device == "GPU"))

if __name__ == "__main__":
    main()
