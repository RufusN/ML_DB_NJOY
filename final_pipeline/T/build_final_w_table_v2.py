# table_xs_coarse.py – build a coarse-grid FP32 table for linear interpolation
# ====================================================================
# Usage:
#   python table_xs_coarse.py --weights model_weights.h5 \
#       --scaler spec_scalers.h5 --window 4 --step 2 --stride 8 \
#       --out w_table_coarse_fp32.h5
from __future__ import annotations
import argparse, time
import h5py
import numpy as np

def map_segments_and_locals(e_idx: int, window: int, step: int):
    from math import ceil
    H, C = 3, 2
    n_over = window // step
    first  = int(ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

def build_coarse_table(
    weights_h5: str,
    scaler_h5: str,
    window: int,
    step: int,
    stride: int,
    out_h5: str
):
    # load model
    with h5py.File(weights_h5, 'r') as hf_w:
        W0       = hf_w['W0'][:].astype(np.float32)
        b0       = hf_w['b0'][:].astype(np.float32)
        alpha    = float(hf_w['alpha'][()])
        W_dec    = hf_w['W_dec'][:].astype(np.float32)
        b_dec    = hf_w['b_dec'][:].astype(np.float32)
        full_w   = W_dec.shape[1]

    H, C, LAT = 3, 2, 16
    N_STFT = full_w // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step

    # load scalers
    with h5py.File(scaler_h5, 'r') as hf_s:
        spec_scale = hf_s['spec_scale'][:].astype(np.float32)
        spec_mean  = hf_s['spec_mean'] [:].astype(np.float32)
        T_scale    = hf_s['T_scale']   [:].astype(np.float32)
        T_mean     = hf_s['T_mean']    [:].astype(np.float32)

    hann = np.hanning(window).astype(np.float32)
    scale_fac = np.sum(hann**2) / step

    # build full table
    W_tab = np.zeros((E_ROWS, LAT), np.float32)
    b_tab = np.zeros(E_ROWS,        np.float32)

    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue
        flat = []
        for h in range(H):
            for s in segs:
                base = h * (N_STFT * C) + s * C
                flat.extend((base, base+1))
        flat = np.array(flat, np.int32)
        W_sub = W_dec[:, flat] * spec_scale[flat]
        b_sub = b_dec[flat] * spec_scale[flat] + spec_mean[flat]
        W_c = W_sub.reshape(LAT, H, n_over, 2)
        b_c = b_sub.reshape(     H, n_over, 2)
        W_c = (W_c[...,0] + 1j*W_c[...,1]).transpose(0,2,1)
        b_c = (b_c[...,0] + 1j*b_c[...,1]).transpose(1,0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann

        coeff = seg_W[
            np.arange(LAT)[:,None],
            np.arange(n_over),
            local
        ].sum(axis=1) / scale_fac
        bias  = seg_b[
            np.arange(n_over),
            local
        ].sum() / scale_fac

        W_tab[e_idx] = coeff
        b_tab[e_idx] = bias

    # sample coarse in FP32
    idxs     = np.arange(0, E_ROWS, stride, dtype=np.int32)
    W_coarse = W_tab[idxs].astype(np.float32)
    b_coarse = b_tab[idxs].astype(np.float32)

    # write
    with h5py.File(out_h5, 'w') as hf:
        hf.create_dataset('W_coarse', data=W_coarse, compression='gzip', shuffle=True)
        hf.create_dataset('b_coarse', data=b_coarse, compression='gzip', shuffle=True)
        hf['stride']   = stride
        hf['W0']       = W0.astype(np.float32)
        hf['b0']       = b0.astype(np.float32)
        hf['alpha']    = alpha
        hf['T_scale']  = T_scale.astype(np.float32)
        hf['T_mean']   = T_mean.astype(np.float32)
        hf['window']   = window
        hf['step']     = step

    print(f"[build] coarse table FP32 saved → {out_h5}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--scaler',  required=True)
    p.add_argument('--window',  type=int, required=True)
    p.add_argument('--step',    type=int, required=True)
    p.add_argument('--stride',  type=int, required=True)
    p.add_argument('--out',     default='w_table_final_v2.h5')
    args = p.parse_args()
    build_coarse_table(
        args.weights, args.scaler,
        args.window, args.step,
        args.stride, args.out
    )
