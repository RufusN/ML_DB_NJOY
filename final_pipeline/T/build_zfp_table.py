#!/usr/bin/env python3
"""
table_tiered_ahvq.py – fp32 + fp16 + HVQ builder
===============================================

Memory tiers
------------
1. **fp32 cache**  – rows whose MSE are in the top `--fp32P %`
2. **fp16 cache**  – next `--fp16P %` rows
3. **HVQ**         – everything else (256 + 64 code-books)

CLI
---
python table_tiered_ahvq.py build \
       --weights   model_weights.h5 \
       --scaler    3x45551_950_1050_spec_scalers.h5 \
       --window 4 --step 2 \
       --fp32P  1.0   --fp16P  4.0    \\
       --k1 256 --k2 64               \
       --out w_table_ahvq.h5

*If `--fp32P + fp16P >= 100` the file is a **pure dense fp32** table.*

File layout
-----------
C1, C2, idx1, idx2, b_tab               – HVQ fallback  
rows32, W32, b32                        – fp32 cache (optional)  
rows16, W16, b16                        – fp16 cache (optional)  
W0, b0, alpha, T_scale, T_mean, window, step
"""

from __future__ import annotations
import argparse, time, os, h5py, numpy as np
from sklearn.cluster import MiniBatchKMeans

# ───────────────────────── helpers ─────────────────────────
def map_segments_and_locals(e, window, step):
    n_over = window // step
    first  = int(np.ceil((e - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e - (segs + 1) * step) % window
    return segs, local

def hvq(mat, k1=256, k2=64, seed=0):
    km1  = MiniBatchKMeans(k1, random_state=seed,   batch_size=4096).fit(mat)
    idx1 = km1.labels_.astype(np.uint8)
    resid = mat - km1.cluster_centers_[idx1]
    km2  = MiniBatchKMeans(k2, random_state=seed+1, batch_size=4096).fit(resid)
    idx2 = km2.labels_.astype(np.uint8)
    return (km1.cluster_centers_.astype(np.float16),
            km2.cluster_centers_.astype(np.float16),
            idx1, idx2)

# ───────────────────────── builder ─────────────────────────
def build_table(weights_h5, scaler_h5, window, step,
                out_h5="w_table_ahvq.h5",
                fp32P=1.0, fp16P=4.0,
                k1=256, k2=64):

    if not (0 <= fp32P <= 100 and 0 <= fp16P <= 100 and fp32P+fp16P <= 100):
        raise SystemExit("percentages must be 0-100 and fp32P+fp16P ≤ 100")

    # load model
    with h5py.File(weights_h5) as f:
        W0, b0 = f["W0"][:].astype("float32"), f["b0"][:].astype("float32")
        alpha  = float(f["alpha"][()])
        W_dec  = f["W_dec"][:].astype("float32")
        b_dec  = f["b_dec"][:].astype("float32")
        full_width = W_dec.shape[1]

    H, C = 3, 2
    N_STFT = full_width // (H*C)
    E_ROWS = 2*N_STFT + (window-1)
    LATENT, n_over = 16, window//step
    print(f"[build] rows={E_ROWS}, fp32P={fp32P}  fp16P={fp16P}")

    with h5py.File(scaler_h5) as f:
        spec_scale = f["spec_scale"][:].astype("float32")
        spec_mean  = f["spec_mean"] [:].astype("float32")
        T_scale    = f["T_scale"]   [:].astype("float32")
        T_mean     = f["T_mean"]    [:].astype("float32")

    hann = np.hanning(window).astype("float32")
    scale_fac = np.sum(hann**2) / step
    W_tab = np.zeros((E_ROWS, LATENT), np.float32)
    b_tab = np.zeros(E_ROWS,            np.float32)
    stride = N_STFT * C

    t0 = time.perf_counter()
    for e in range(E_ROWS):
        segs, local = map_segments_and_locals(e, window, step)
        if segs.max() >= N_STFT: continue
        idx = np.concatenate(
            [(f*stride + segs[:,None]*C + np.arange(C)).ravel() for f in range(H)]
        )
        W_sub = W_dec[:, idx] * spec_scale[idx]
        b_sub = b_dec[idx]    * spec_scale[idx] + spec_mean[idx]
        W_c = W_sub.reshape(LATENT,H,n_over,2)
        b_c = b_sub.reshape(H,n_over,2)
        W_c = (W_c[...,0]+1j*W_c[...,1]).transpose(0,2,1)
        b_c = (b_c[...,0]+1j*b_c[...,1]).transpose(1,0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann
        W_tab[e] = seg_W[np.arange(LATENT)[:,None], np.arange(n_over), local].sum(1)/scale_fac
        b_tab[e] = seg_b[np.arange(n_over),          local].sum()     /scale_fac
    print(f"[build] dense finished in {time.perf_counter()-t0:.1f}s")

    # ------- ranking by MSE wrt full-precision -------
    C1, C2, idx1, idx2 = hvq(W_tab, k1, k2)
    mse = ((W_tab - (C1[idx1] + C2[idx2]))**2).mean(1)
    order = np.argsort(-mse)        # descending

    n32  = int(E_ROWS * fp32P / 100 + 0.5)
    n16  = int(E_ROWS * fp16P / 100 + 0.5)
    rows32 = np.sort(order[:n32]).astype(np.int32)
    rows16 = np.sort(order[n32:n32+n16]).astype(np.int32)
    mask32 = np.zeros(E_ROWS, bool); mask32[rows32] = True
    mask16 = np.zeros(E_ROWS, bool); mask16[rows16] = True

    W32, b32 = W_tab[mask32].astype("float32"), b_tab[mask32].astype("float32")
    W16, b16 = W_tab[mask16].astype("float16"), b_tab[mask16].astype("float16")

    with h5py.File(out_h5,"w") as f:
        f["C1"],f["C2"],f["idx1"],f["idx2"] = C1,C2,idx1,idx2
        f["b_tab"] = b_tab.astype("float16")          # HVQ bias table (fp16)
        if n32: f["rows32"],f["W32"],f["b32"] = rows32,W32,b32
        if n16: f["rows16"],f["W16"],f["b16"] = rows16,W16,b16
        f["W0"],f["b0"],f["alpha"] = W0,b0,alpha
        f["T_scale"],f["T_mean"] = T_scale,T_mean
        f["window"],f["step"] = window,step
        f.attrs["AHVQ"]=1
    print(f"[build] saved {out_h5}  (size={os.path.getsize(out_h5)/1024:.1f} kB)")

# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("cmd", choices=["build"])
    P.add_argument("--weights", required=True)
    P.add_argument("--scaler",  required=True)
    P.add_argument("--window",  type=int, required=True)
    P.add_argument("--step",    type=int, required=True)
    P.add_argument("--fp32P",   type=float, default=1.0,
                   help="%% rows kept in fp32 (highest error first)")
    P.add_argument("--fp16P",   type=float, default=4.0,
                   help="additional %% rows kept in fp16")
    P.add_argument("--k1", type=int, default=256)
    P.add_argument("--k2", type=int, default=64)
    P.add_argument("--out", default="w_table_ahvq.h5")
    a = P.parse_args()
    if a.cmd == "build":
        build_table(a.weights, a.scaler, a.window, a.step,
                    a.out, a.fp32P, a.fp16P, a.k1, a.k2)
