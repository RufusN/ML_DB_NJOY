#!/usr/bin/env python3
"""
build_quant_table.py  –  make an 8-bit Doppler-XS lookup table
=============================================================
* Builds the **full latent-major table** (float32) right here — no
  external import needed.
* Quantises each of the 16 latent columns **independently** to uint8
  with linear scale = (max-min)/255 and per-column zero-point (=min).
* Bias vector stays in float16 (lossless enough, negligible size).
* Output HDF5 schema
      W_q      (E,16)  uint8
      b_tab    (E,)    float16
      scale_q  (16,)   float16
      zero_q   (16,)   float16
      W0, b0, alpha, T_scale, T_mean  (same dtype as checkpoint)
* Size for your 91 105-row model:  **≈ 0.28 MB on disk**,  
  **1.46 MB in GPU memory**  (half the fp16 table).
"""
from __future__ import annotations
import argparse, time, h5py, numpy as np, os, tempfile, shutil, sys

# ─────────────────────────── helper: STFT mapping ────────────────────────
def map_segments_and_locals(e_idx: int, window: int, step: int):
    n_over = window // step
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

# ─────────────────────────── full-table builder ──────────────────────────
def build_full_table(weights_h5: str, scaler_h5: str,
                     window: int, step: int):
    """
    Returns
    -------
    W_tab : (E_ROWS, 16)  float32
    b_tab : (E_ROWS,)     float32
    W0, b0 : (1×16 layer) float32
    alpha  : float
    T_scale, T_mean : (1,) float32
    """
    H, C, LATENT = 3, 2, 16
    with h5py.File(weights_h5, "r") as hf_w:
        full_width = hf_w["W_dec"].shape[1]
        W0    = hf_w["W0"][:].astype("float32")
        b0    = hf_w["b0"][:].astype("float32")
        alpha = float(hf_w["alpha"][()])
        W_dec = hf_w["W_dec"][:].astype("float32")
        b_dec = hf_w["b_dec"][:].astype("float32")

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step

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

    print(f"[full] building {E_ROWS:,} rows …")
    t0 = time.perf_counter()
    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue  # unseen STFT column
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

        coeff = seg_W[np.arange(LATENT)[:,None],
                      np.arange(n_over), local].sum(axis=1) / scale_fac
        bias  = seg_b[np.arange(n_over), local].sum()         / scale_fac

        W_tab[e_idx] = coeff.real
        b_tab[e_idx] = bias.real

        if e_idx % 10_000 == 0:
            print(f"  • {e_idx}/{E_ROWS} rows")

    print(f"[full] done in {time.perf_counter()-t0:.1f}s")
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean

# ─────────────────────────── quantisation helper ────────────────────────
def quantise_per_latent(W: np.ndarray):
    """uint8 linear quantisation 0-255 for each of 16 columns."""
    mins  = W.min(axis=0)      # (16,)
    maxs  = W.max(axis=0)
    scale = (maxs - mins) / 255.0
    scale[scale == 0] = 1e-9   # avoid zero spread
    W_q   = np.round((W - mins) / scale).astype(np.uint8)
    return W_q, scale.astype("float16"), mins.astype("float16")

# ─────────────────────────── main entry point ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scaler",  required=True)
    ap.add_argument("--window",  type=int, required=True)
    ap.add_argument("--step",    type=int, required=True)
    ap.add_argument("--out",     default="w_table_quant.h5")
    args = ap.parse_args()

    W_tab, b_tab, W0, b0, alpha, T_s, T_m = build_full_table(
        args.weights, args.scaler, args.window, args.step
    )

    W_q, scale_q, zero_q = quantise_per_latent(W_tab)
    print(f"[quant] raw size  = {W_q.nbytes/1024:.1f} kB")
    print(f"[quant] max |Δw|  = {np.max(scale_q)/2:.4e}")

    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("W_q",     data=W_q,
                          compression="gzip", shuffle=True, compression_opts=9)
        hf.create_dataset("b_tab",   data=b_tab.astype("float16"),
                          compression="gzip", shuffle=True)
        hf["scale_q"], hf["zero_q"]  = scale_q, zero_q
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"]     = T_s, T_m
    print(f"[quant] saved → {args.out}")

if __name__ == "__main__":
    main()
