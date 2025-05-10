#!/usr/bin/env python3
"""
build_w_table.py
================
Precompute and store the end-to-end weight table for XS reconstruction.
For each energy-index E in [START_IDX, END_IDX], computes the 16-dim
vector w(E) and scalar bias b(E) such that

    XS(T, E) = hidden(T) @ w(E) + b(E)

where hidden(T) = LeakyReLU((T - T_mean)/T_scale * W0 + b0).

Outputs:
  • w_table.h5 containing datasets:
      - W_tab   : float32 [N_E, 16]
      - b_tab   : float32 [N_E]
      - E_idxs  : int32   [N_E] (the energy indices)

Usage:
    python build_w_table.py --start 100 --end 45000
"""
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from scipy.signal.windows import hann

# Constants matching inference pipeline
H, W_TIME, C = 3, 45551, 2
WINDOW_SAMPS = 2048
STEP_SAMPS   = 512
SCALER_PATH  = "3x45551_950_1050_spec_scalers.h5"
MODEL_PATH   = "3x45551_950_1050.keras"
OUTPUT_PATH  = "w_table.h5"

# Precompute DFT basis and window as numpy arrays
dt = 2 * np.pi / WINDOW_SAMPS
freqs = np.arange(H)[:, None]
ns = np.arange(WINDOW_SAMPS)[None, :]
exp_coeffs = np.exp(1j * dt * (freqs * ns)).astype(np.complex64)  # [H, WINDOW_SAMPS]
hann_win = hann(WINDOW_SAMPS).astype(np.float32)
scale_fac = np.sum(hann_win**2) / STEP_SAMPS

# Utilities
def mapSegIdx(E_idx):
    n_over = WINDOW_SAMPS // STEP_SAMPS
    first = int(np.ceil((E_idx - WINDOW_SAMPS)/STEP_SAMPS - 0.5))
    segs = first + np.arange(n_over, dtype=np.int32)
    local = (E_idx - segs * STEP_SAMPS) % WINDOW_SAMPS
    return segs, local

def flat_idxs(segs):
    idxs = []
    for f in range(H):
        for t in segs:
            for ch in range(C):
                idxs.append(f * (W_TIME * C) + t * C + ch)
    return np.array(idxs, dtype=np.int32)

# Main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end',   type=int, required=True)
    args = ap.parse_args()
    E_list = np.arange(args.start, args.end+1, dtype=np.int32)
    N_E = len(E_list)

    # Load scalers and model weights
    with h5py.File(SCALER_PATH, 'r') as hf:
        T_scale = hf['T_scale'][:].astype(np.float32)
        T_mean  = hf['T_mean'] [:].astype(np.float32)
        spec_scale = hf['spec_scale'][:].astype(np.float32)
        spec_mean  = hf['spec_mean'] [:].astype(np.float32)

    model = load_model(MODEL_PATH, compile=False)
    W0 = model.layers[1].kernel.numpy().astype(np.float32).ravel()  # (1,16)->(16,)
    b0 = model.layers[1].bias.numpy().astype(np.float32)
    alpha = getattr(model.layers[2], 'alpha', getattr(model.layers[2], 'negative_slope', 0.3))
    W_dec = model.layers[3].kernel.numpy().astype(np.float32)
    b_dec = model.layers[3].bias.numpy().astype(np.float32)

    # Prepare tables
    W_tab = np.zeros((N_E, W0.size), dtype=np.float32)
    b_tab = np.zeros((N_E,), dtype=np.float32)

    for i, E in enumerate(E_list):
        # 1) compute hidden basis for each latent dim j
        # never need T here since weight table independent of T
        # Instead, for each j: hidden = zero except hidden[j]=1
        segs, local = mapSegIdx(int(E))
        fidx = flat_idxs(segs)
        # gather decoder weights
        W_small = W_dec[:, fidx]  # shape [latent, N_flat]
        b_small = b_dec[fidx]     # shape [N_flat]
        # DFT unscale and overlap-add for each latent basis vector
        for j in range(W0.size):
            # spec_scaled_flat = W_small[j,:] + b_small
            spec_flat = spec_scale[fidx] * (W_small[j,:] + b_small) + spec_mean[fidx]
            spec = spec_flat.reshape(H, -1, C)
            spec_c = spec[:,:,0] + 1j*spec[:,:,1]
            xsj = 0.0
            for k in range(spec_c.shape[1]):
                # inverse DFT at local[k]
                val = np.real(np.dot(spec_c[:,k], exp_coeffs[:, local[k]]))
                xsj += val * hann_win[local[k]]
            W_tab[i,j] = xsj / scale_fac
        # bias: hidden=0
        spec_flat = spec_scale[fidx] * b_small + spec_mean[fidx]
        spec = spec_flat.reshape(H, -1, C)
        spec_c = spec[:,:,0] + 1j*spec[:,:,1]
        bval = 0.0
        for k in range(spec_c.shape[1]):
            val = np.real(np.dot(spec_c[:,k], exp_coeffs[:, local[k]]))
            bval += val * hann_win[local[k]]
        b_tab[i] = bval / scale_fac

    # Save
    with h5py.File(OUTPUT_PATH, 'w') as hf:
        hf.create_dataset('E_idxs', data=E_list)
        hf.create_dataset('W_tab',  data=W_tab)
        hf.create_dataset('b_tab',  data=b_tab)
    print(f"Saved weight table for {N_E} indices → {OUTPUT_PATH}")
