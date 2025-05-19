#!/usr/bin/env python3
"""
table_ahvq.py – Adaptive HVQ builder (v2)
========================================
* --topP  : percentage (0–100) of rows stored verbatim (fp16).
* --mse-thr: absolute per-row MSE threshold (mutually exclusive with --topP).
* File-size print-out uses os.path.getsize for robustness.
"""

from __future__ import annotations
import argparse, time, os, h5py, numpy as np
from sklearn.cluster import MiniBatchKMeans

# ────────────────────── helpers ──────────────────────
def map_segments_and_locals(e_idx: int, window: int, step: int):
    n_over = window // step
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

def hierarchical_vq(mat: np.ndarray, k1=256, k2=64, seed=0):
    mb1 = MiniBatchKMeans(k1, random_state=seed, batch_size=4096).fit(mat)
    idx1 = mb1.labels_.astype(np.uint8)
    resid = mat - mb1.cluster_centers_[idx1]
    mb2 = MiniBatchKMeans(k2, random_state=seed + 1, batch_size=4096).fit(resid)
    idx2 = mb2.labels_.astype(np.uint8)
    return mb1.cluster_centers_.astype(np.float16), mb2.cluster_centers_.astype(np.float16), idx1, idx2

# ────────────────────── builder ──────────────────────
def build_table(weights_h5, scaler_h5, window, step,
                out_h5="w_table_ahvq.h5", mse_thr=None, topP=None,
                k1=256, k2=64):

    if topP is not None and mse_thr is not None:
        raise SystemExit("choose either --mse-thr or --topP, not both")
    if topP is not None and not (0 <= topP <= 100):
        raise SystemExit("--topP must be between 0 and 100")

    H, C = 3, 2
    print("[AHVQ] reading model …")
    with h5py.File(weights_h5) as f:
        full_width = f["W_dec"].shape[1]
        W0  = f["W0"][:].astype("float16")
        b0  = f["b0"][:].astype("float16")
        alpha = float(f["alpha"][()])
        W_dec = f["W_dec"][:].astype("float32")
        b_dec = f["b_dec"][:].astype("float32")

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    LATENT = 16
    n_over = window // step
    print(f"[AHVQ] building dense table ({E_ROWS} rows)…")

    with h5py.File(scaler_h5) as f:
        spec_scale = f["spec_scale"][:].astype("float32")
        spec_mean  = f["spec_mean"] [:].astype("float32")
        T_scale    = f["T_scale"]   [:].astype("float16")
        T_mean     = f["T_mean"]    [:].astype("float16")

    hann = np.hanning(window).astype("float32")
    scale_fac = np.sum(hann ** 2) / step
    W_tab = np.zeros((E_ROWS, LATENT), np.float32)
    b_tab = np.zeros(E_ROWS, np.float32)
    stride = N_STFT * C

    t0 = time.perf_counter()
    for e in range(E_ROWS):
        segs, local = map_segments_and_locals(e, window, step)
        if segs.max() >= N_STFT:
            continue
        idx = np.concatenate(
            [(f * stride + segs[:, None] * C + np.arange(C)).ravel() for f in range(H)]
        )
        W_sub = W_dec[:, idx] * spec_scale[idx]
        b_sub = b_dec[idx]   * spec_scale[idx] + spec_mean[idx]
        W_c = W_sub.reshape(LATENT, H, n_over, 2)
        b_c = b_sub.reshape(H, n_over, 2)
        W_c = (W_c[..., 0] + 1j * W_c[..., 1]).transpose(0, 2, 1)
        b_c = (b_c[..., 0] + 1j * b_c[..., 1]).transpose(1, 0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann
        W_tab[e] = (
            seg_W[np.arange(LATENT)[:, None], np.arange(n_over), local].sum(1) / scale_fac
        )
        b_tab[e] = seg_b[np.arange(n_over), local].sum() / scale_fac
    print(f"[AHVQ] dense done in {time.perf_counter() - t0:.1f}s")

    # HVQ
    print("[AHVQ] running HVQ …")
    C1, C2, idx1, idx2 = hierarchical_vq(W_tab, k1, k2)
    recon = C1[idx1] + C2[idx2]
    mse = ((W_tab - recon) ** 2).mean(1)

    if topP is not None:
        keep = int(E_ROWS * topP / 100 + 0.5)
        thr = np.partition(mse, -keep)[-keep] if keep > 0 else 0.0
    else:
        thr = mse_thr or 0.0

    mask = mse > thr
    exc_rows = np.where(mask)[0].astype(np.int32)
    W_exc = W_tab[mask].astype(np.float16)
    b_exc = b_tab[mask].astype(np.float16)
    print(f"[AHVQ] keeping {len(exc_rows)} / {E_ROWS} rows verbatim (thr={thr:.2e})")

    with h5py.File(out_h5, "w") as f:
        f["C1"], f["C2"], f["idx1"], f["idx2"] = C1, C2, idx1, idx2
        f["b_tab"] = b_tab.astype(np.float16)
        f["exc_rows"], f["W_exc"], f["b_exc"] = exc_rows, W_exc, b_exc
        f["W0"], f["b0"], f["alpha"] = W0, b0, alpha
        f["T_scale"], f["T_mean"] = T_scale, T_mean
        f["window"], f["step"] = window, step
        f.attrs["AHVQ"] = 1
    print(f"[AHVQ] wrote {out_h5}  (size={os.path.getsize(out_h5)/1024:.1f} kB)")

# ────────────────────── CLI ──────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--weights", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--step",   type=int, required=True)
    p.add_argument("--out",    default="w_table_ahvq.h5")
    p.add_argument("--mse-thr", type=float)
    p.add_argument("--topP",    type=float)
    p.add_argument("--k1", type=int, default=256)
    p.add_argument("--k2", type=int, default=64)

    a = p.parse_args()
    if a.cmd == "build":
        build_table(a.weights, a.scaler, a.window, a.step,
                    a.out, a.mse_thr, a.topP, a.k1, a.k2)
