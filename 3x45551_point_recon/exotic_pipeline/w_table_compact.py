#!/usr/bin/env python3
"""
build_compact_table.py  –  create a K-stride float16 XS table
============================================================
• Generates the *full* latent-major table once (float32)
• Keeps only every K-th row, stores it as float16 + row index array
• Typical size @K=64 :  110 kB   (vs 3 MB full fp16 table)
"""
from __future__ import annotations
import argparse, time, h5py, numpy as np, tempfile, os, shutil

# ───────────────────────────────────────────────────────────────────────────
#  1 · minimal copy of the original FULL builder (no tf needed here)
# ───────────────────────────────────────────────────────────────────────────
def _build_full(weights_h5: str, scaler_h5: str,
                window: int, step: int) -> tuple[np.ndarray, ...]:
    """Return W_tab (E×16), b_tab (E,), plus layer-0 & scalers."""
    H, C = 3, 2
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
    print(f"[full] window={window}, step={step}, rows={E_ROWS}")

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

    def map_segments_and_locals(e_idx: int):
        first  = int(np.ceil((e_idx - window) / step - 0.5))
        segs   = np.arange(first, first + n_over, dtype=np.int32)
        local  = (e_idx - (segs + 1) * step) % window
        return segs, local

    t0 = time.perf_counter()
    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx)
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
        if e_idx % 10000 == 0:
            print(f"  • {e_idx}/{E_ROWS} rows")

    print(f"[full] finished in {time.perf_counter()-t0:.1f}s")
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean

# ───────────────────────────────────────────────────────────────────────────
#  2 · compact builder
# ───────────────────────────────────────────────────────────────────────────
def build_compact(weights_h5: str, scaler_h5: str,
                  window: int, step: int, K: int,
                  out_h5: str):

    W_tab, b_tab, W0, b0, alpha, T_scale, T_mean = \
        _build_full(weights_h5, scaler_h5, window, step)

    rows = np.arange(0, W_tab.shape[0], K, dtype=np.int32)
    W_k  = W_tab[rows].astype(np.float16)
    b_k  = b_tab[rows].astype(np.float16)

    size = W_k.nbytes + b_k.nbytes + rows.nbytes
    print(f"[compact] store {len(rows)} rows (K={K}) → {size/1024:.1f} kB")

    with h5py.File(out_h5, "w") as hf:
        hf.create_dataset("rows", data=rows, dtype="int32")
        hf.create_dataset("W_k",  data=W_k,  compression="gzip", shuffle=True)
        hf.create_dataset("b_k",  data=b_k,  compression="gzip", shuffle=True)
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["K"] = K
    print("[compact] saved ✔︎")

# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--weights", required=True)
    P.add_argument("--scaler",  required=True)
    P.add_argument("--window",  type=int, required=True)
    P.add_argument("--step",    type=int, required=True)
    P.add_argument("--K",       type=int, default=64,
                   help="keep every K-th row (64 ≈ 110 kB)")
    P.add_argument("--out",     default="w_table_compact.h5")
    args = P.parse_args()

    build_compact(args.weights, args.scaler,
                  args.window, args.step, args.K, args.out)
