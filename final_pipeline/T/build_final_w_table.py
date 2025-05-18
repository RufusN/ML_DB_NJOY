# table_xs_int8_block4_rowq.py – per-row 4-block INT8 quantization (fixed scales)
# ====================================================================
# Builds full W_tab then quantizes each row into four 4‐dim blocks
# to INT8 with per‐block scales stored in FP32 for higher accuracy
# (~1.8 MB memory). Avoids FP16 scale truncation.
# Usage:
#   python table_xs_int8_block4_rowq.py --weights model_weights.h5 \
#       --scaler spec_scalers.h5 --window 4 --step 2 --out w_table_block4.h5

from __future__ import annotations
import argparse
import h5py
import numpy as np

# helper: segment mapping
def map_segments_and_locals(e_idx: int, window: int, step: int):
    n_over = window // step
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

# build 4-block per-row INT8 table
def build_int8_block4_rowq_table(
    weights_h5: str, scaler_h5: str,
    window: int, step: int, out_h5: str
):
    H, C, LAT = 3, 2, 16
    # load model + decoder
    with h5py.File(weights_h5, 'r') as hf_w:
        full_width = hf_w['W_dec'].shape[1]
        W0    = hf_w['W0'][:].astype(np.float32)
        b0    = hf_w['b0'][:].astype(np.float32)
        alpha = float(hf_w['alpha'][()])
        W_dec = hf_w['W_dec'][:].astype(np.float32)
        b_dec = hf_w['b_dec'][:].astype(np.float32)
    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step
    # load spectral scalers
    with h5py.File(scaler_h5, 'r') as hf_s:
        spec_scale = hf_s['spec_scale'][:].astype(np.float32)
        spec_mean  = hf_s['spec_mean'] [:].astype(np.float32)
        T_scale    = hf_s['T_scale']   [:].astype(np.float32)
        T_mean     = hf_s['T_mean']    [:].astype(np.float32)
    # assemble full W_tab
    hann = np.hanning(window).astype(np.float32)
    scale_fac = np.sum(hann**2) / step
    W_tab = np.zeros((E_ROWS, LAT), np.float32)
    b_tab = np.zeros(E_ROWS,       np.float32)
    flat_stride = N_STFT * C
    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue
        flat = []
        for h in range(H):
            for s in segs:
                base = h * flat_stride + s * C
                flat.extend((base, base + 1))
        flat = np.asarray(flat, np.int32)
        W_sub = W_dec[:, flat] * spec_scale[flat]
        b_sub = b_dec[flat] * spec_scale[flat] + spec_mean[flat]
        W_c   = W_sub.reshape(LAT, H, n_over, 2)
        b_c   = b_sub.reshape(H, n_over, 2)
        W_c   = (W_c[...,0] + 1j * W_c[...,1]).transpose(0,2,1)
        b_c   = (b_c[...,0] + 1j * b_c[...,1]).transpose(1,0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann
        W_tab[e_idx] = seg_W[np.arange(LAT)[:,None], np.arange(n_over), local].sum(1) / scale_fac
        b_tab[e_idx] = seg_b[np.arange(n_over), local].sum() / scale_fac
    # per-row 4-block INT8 quantization
    QW = np.zeros((E_ROWS, LAT), np.int8)
    scales = np.zeros((E_ROWS, 4), np.float32)
    for blk in range(4):
        lo, hi    = blk*4, blk*4 + 4
        block_vals= W_tab[:, lo:hi]
        blk_max   = np.max(np.abs(block_vals), axis=1, keepdims=True)
        scale_blk = blk_max / 127.0
        scales[:, blk] = scale_blk[:,0]
        QW[:, lo:hi] = np.round(block_vals / scale_blk).astype(np.int8)
    # save (scales in FP32)
    with h5py.File(out_h5, 'w') as hf:
        hf.create_dataset('QW_tab',      data=QW,             compression='gzip', shuffle=True)
        hf.create_dataset('scale_blocks', data=scales,         compression='gzip')  # float32 [E,4]
        hf.create_dataset('b_tab',        data=b_tab.astype(np.float16), compression='gzip', shuffle=True)
        hf['W0']       = W0.astype(np.float16)
        hf['b0']       = b0.astype(np.float16)
        hf['alpha']    = alpha
        hf['T_scale']  = T_scale.astype(np.float16)
        hf['T_mean']   = T_mean.astype(np.float16)
        hf['window']   = window
        hf['step']     = step
    print(f"[build] per-row 4-block INT8 table saved → {out_h5}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--scaler',  required=True)
    p.add_argument('--window',  type=int, required=True)
    p.add_argument('--step',    type=int, required=True)
    p.add_argument('--out',     default='w_table_final.h5')
    args = p.parse_args()
    build_int8_block4_rowq_table(args.weights, args.scaler, args.window, args.step, args.out)