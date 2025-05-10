#!/usr/bin/env python3
"""
build_w_table.py
================
Precompute and store the end-to-end weight table for XS reconstruction, including
first-layer (T→hidden) weights. For each energy-index E in [start, end],
computes:

  • A 16-dim weight vector w(E) and scalar bias b(E) such that:
      XS(T, E) = hidden(T) · w(E) + b(E)
    where hidden(T) = LeakyReLU((T - T_mean)/T_scale * W0 + b0)

Outputs HDF5 with datasets:
  • E_idxs : int32 [N_E]
  • W_tab  : float32 [N_E, 16]
  • b_tab  : float32 [N_E]
  • W0     : float32 [1, 16]
  • b0     : float32 [16]
  • alpha  : float32 scalar

Usage:
    python build_w_table.py --start 100 --end 45000
"""
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal.windows import hann

# Constants
H, W_TIME, C = 3, 45551, 2
WINDOW_SAMPS = 2048
STEP_SAMPS   = 512
SCALER_PATH  = "3x45551_950_1050_spec_scalers.h5"
MODEL_PATH   = "3x45551_950_1050.keras"
OUTPUT_PATH  = "w_table.h5"

# Precompute DFT basis and Hann window for direct DFT
dt = 2 * np.pi / WINDOW_SAMPS
freqs = np.arange(H)[:, None]
ns = np.arange(WINDOW_SAMPS)[None, :]
exp_coeffs = np.exp(1j * dt * (freqs * ns)).astype(np.complex64)
hann_win = hann(WINDOW_SAMPS).astype(np.float32)
scale_fac = np.sum(hann_win**2) / STEP_SAMPS

# Utilities
def mapSegIdx(E_idx: int):
    n_over = WINDOW_SAMPS // STEP_SAMPS
    first = int(np.ceil((E_idx - WINDOW_SAMPS)/STEP_SAMPS - 0.5))
    segs = first + np.arange(n_over, dtype=np.int32)
    local = (E_idx - segs * STEP_SAMPS) % WINDOW_SAMPS
    return segs, local


def flat_idxs(segs: np.ndarray):
    idxs = []
    for f in range(H):
        for t in segs:
            for ch in range(C):
                idxs.append(f * (W_TIME * C) + t * C + ch)
    return np.array(idxs, dtype=np.int32)

# Main
def main():
    # Load the canonical energy grid to determine full index range
    with h5py.File("base_energy_grid.h5", 'r') as hf:
        Base_E = hf['Base_E'][:]
    E_list = np.arange(len(Base_E), dtype=np.int32)
    N_E = len(E_list)

    # Load scalers
    with h5py.File(SCALER_PATH, 'r') as hf:
        T_scale = hf['T_scale'][:].astype(np.float32)
        T_mean  = hf['T_mean'] [:].astype(np.float32)
        spec_scale = hf['spec_scale'][:].astype(np.float32)
        spec_mean  = hf['spec_mean'] [:].astype(np.float32)

    # Load model and weights
    model = load_model(MODEL_PATH, compile=False)
    dense0 = model.layers[1]
    W0 = dense0.kernel.numpy().astype(np.float32).reshape(1, -1)  # [1,16]
    b0 = dense0.bias.numpy().astype(np.float32)                  # [16]
    lrelu_layer = model.layers[2]
    alpha = getattr(lrelu_layer, 'alpha', getattr(lrelu_layer, 'negative_slope', 0.3))
    dense_dec = model.layers[3]
    W_dec = dense_dec.kernel.numpy().astype(np.float32)          # [in_dim, out_dim]
    b_dec = dense_dec.bias.numpy().astype(np.float32)            # [out_dim]

    # Allocate table arrays
    W_tab = np.zeros((N_E, W0.shape[1]), dtype=np.float32)
    b_tab = np.zeros((N_E,), dtype=np.float32)

    # Compute table
    for i, E in enumerate(E_list):
        segs, local = mapSegIdx(int(E))
        fidx = flat_idxs(segs)
        # Decoder slice
        W_small = W_dec[:, fidx]   # [16, N_flat]
        b_small = b_dec[fidx]      # [N_flat]
        # Compute per-latent weights
        for j in range(W0.shape[1]):
            spec_flat = (W_small[j, :] + b_small) * spec_scale[fidx] + spec_mean[fidx]
            spec = spec_flat.reshape(H, -1, C)
            spec_c = spec[:, :, 0] + 1j * spec[:, :, 1]
            xsj = 0.0
            for k in range(spec_c.shape[1]):
                val = np.real(np.dot(spec_c[:, k], exp_coeffs[:, local[k]]))
                xsj += val * hann_win[local[k]]
            W_tab[i, j] = xsj / scale_fac
        # Bias term
        spec_flat = b_small * spec_scale[fidx] + spec_mean[fidx]
        spec = spec_flat.reshape(H, -1, C)
        spec_c = spec[:, :, 0] + 1j * spec[:, :, 1]
        bval = 0.0
        for k in range(spec_c.shape[1]):
            val = np.real(np.dot(spec_c[:, k], exp_coeffs[:, local[k]]))
            bval += val * hann_win[local[k]]
        b_tab[i] = bval / scale_fac

    # Save to HDF5
    with h5py.File(OUTPUT_PATH, 'w') as hf:
        hf.create_dataset('E_idxs', data=E_list)
        hf.create_dataset('W_tab',  data=W_tab)
        hf.create_dataset('b_tab',  data=b_tab)
        hf.create_dataset('W0',     data=W0)
        hf.create_dataset('b0',     data=b0)
        hf.create_dataset('alpha',  data=np.array(alpha, dtype=np.float32))

    print(f"Saved weight table and first-layer weights for {N_E} indices → {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
    main()
