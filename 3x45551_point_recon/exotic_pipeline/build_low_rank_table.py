#increase r, increase memory
#!/usr/bin/env python3
"""
build_lowrank_table.py – rank-4 fp16 factorisation of W_tab
==========================================================
Reads the *full* fp32 table, does a rank-4 truncated SVD,
stores:

    U4   (E,4)  float16      ← U · diag(S_r)
    V4   (4,16) float16      ← V_rᵀ
    b_tab           float16
    + layer-0 weights, α, T_scale, T_mean

File size:  ~0.36 MB (gzip), GPU footprint: 0.91 MB.
"""
from __future__ import annotations
import argparse, time, h5py, numpy as np
from numpy.linalg import svd

# --- helper: build full fp32 table (identical to your original code) ---
def build_full(weights_h5, scaler_h5, window, step):
    from math import ceil
    H, C, LAT = 3, 2, 16
    with h5py.File(weights_h5, "r") as hf_w:
        W0, b0  = hf_w["W0"][:], hf_w["b0"][:]
        alpha   = float(hf_w["alpha"][()])
        W_dec   = hf_w["W_dec"][:]
        b_dec   = hf_w["b_dec"][:]
        full_w  = W_dec.shape[1]

    N_STFT = full_w // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step

    with h5py.File(scaler_h5, "r") as hf_s:
        spec_scale = hf_s["spec_scale"][:]
        spec_mean  = hf_s["spec_mean"] [:]
        T_scale    = hf_s["T_scale"]   [:]
        T_mean     = hf_s["T_mean"]    [:]

    hann = np.hanning(window)
    scale_fac = np.sum(hann**2) / step
    W_tab = np.zeros((E_ROWS, LAT), np.float32)
    b_tab = np.zeros(E_ROWS,        np.float32)
    flat_stride = N_STFT * C

    def segs_loc(e_idx):
        n_over = window // step
        first  = int(ceil((e_idx - window)/step - .5))
        segs   = np.arange(first, first+n_over, dtype=np.int32)
        local  = (e_idx - (segs+1)*step) % window
        return segs, local

    print(f"[full] building {E_ROWS:,} rows …")
    for e_idx in range(E_ROWS):
        segs, local = segs_loc(e_idx)
        if segs.max() >= N_STFT:
            continue
        flat = []
        for h in range(H):
            for s in segs:
                base = h*flat_stride + s*C
                flat.extend((base, base+1))
        flat = np.asarray(flat, np.int32)
        W_sub = W_dec[:, flat]*spec_scale[flat]
        b_sub = b_dec[flat]*spec_scale[flat] + spec_mean[flat]
        W_c = W_sub.reshape(LAT, H, n_over, 2)
        b_c = b_sub.reshape(      H, n_over, 2)
        W_c = (W_c[...,0]+1j*W_c[...,1]).transpose(0,2,1)
        b_c = (b_c[...,0]+1j*b_c[...,1]).transpose(1,0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1)*hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1)*hann
        coeff = seg_W[np.arange(LAT)[:,None], np.arange(n_over), local].sum(1)/scale_fac
        bias  = seg_b[np.arange(n_over),             local].sum()/scale_fac
        W_tab[e_idx] = coeff.real
        b_tab[e_idx] = bias.real
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean

# --------------------------- main script ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scaler",  required=True)
    ap.add_argument("--window",  type=int, required=True)
    ap.add_argument("--step",    type=int, required=True)
    ap.add_argument("--out",     default="w_table_lowrank.h5")
    ap.add_argument("--rank",    type=int, default=4, choices=range(1,100))
    args = ap.parse_args()

    W_tab, b_tab, W0, b0, alpha, T_s, T_m = \
        build_full(args.weights, args.scaler, args.window, args.step)

    r = args.rank
    print(f"[svd] computing rank-{r} SVD …")
    U, S, Vt = svd(W_tab, full_matrices=False)
    U_r = (U[:, :r] * S[:r])          .astype(np.float16)   # (E,r)
    V_r =  Vt[:r].astype(np.float16)                       # (r,16)

    rms = np.sqrt(np.mean((W_tab - U_r@V_r)**2))
    print(f"[svd] RMS weight error = {rms:.3e}")

    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("U_r", data=U_r, compression="gzip", shuffle=True)
        hf.create_dataset("V_r", data=V_r, compression="gzip", shuffle=True)
        hf.create_dataset("b_tab", data=b_tab.astype(np.float16),
                          compression="gzip", shuffle=True)
        hf["rank"] = r
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_s, T_m
    print(f"[svd] saved → {args.out}")

if __name__ == "__main__":
    main()
