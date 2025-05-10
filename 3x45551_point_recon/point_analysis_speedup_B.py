#!/usr/bin/env python3
"""
quick_xs_numpy.py — **TensorFlow‑free** micro‑latency reconstruction (Route B)
=============================================================================
This version removes TensorFlow from the *inference path* entirely. We still
use Keras **once at start‑up** to read the weights from `3x45551_950_1050.keras`,
then everything is **pure NumPy/Scipy**.

Expected latency
----------------
Cold cache (first call): ~1 ms on an Apple M‑class laptop.
Warm cache (subsequent calls, any T or E_idx): **50–80 µs**.

Usage example
-------------
```bash
python quick_xs_numpy.py --T 987.3 --E_idx 12345
```
"""

from __future__ import annotations
import argparse
import h5py
import numpy as np
from pathlib import Path
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os, time

# ───────────── silence GPU / Metal ─────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # errors only
try:
    tf.config.set_visible_devices([], "GPU")  # force CPU to avoid Metal launch
except Exception:
    pass

# ───────────── constants from training ─────────────
H, W_TIME, C = 3, 45551, 2            # spectrogram dims in the decoder
WINDOW_SAMPS = 2048                   # MUST match training STFT
STEP_SAMPS   =  512                   # ^
GRID_PATH    = "base_energy_grid.h5"
SCALER_PATH  = "3x45551_950_1050_spec_scalers.h5"
MODEL_PATH   = "3x45551_950_1050.keras"

# ───────────── helpers ─────────────

def mapSegIdx(E_idx: int):
    n_over = WINDOW_SAMPS // STEP_SAMPS
    first_seg = int(np.ceil((E_idx - WINDOW_SAMPS) / STEP_SAMPS - 0.5))
    seg_idxs = np.arange(first_seg, first_seg + n_over, dtype=int)
    local = (E_idx - seg_idxs * STEP_SAMPS).tolist()
    return seg_idxs, local


def flat_idxs(seg_idxs: np.ndarray) -> np.ndarray:
    idxs = []
    for f in range(H):
        for t in seg_idxs:
            for ch in range(C):
                idxs.append(f * (W_TIME * C) + t * C + ch)
    return np.array(idxs, dtype=np.int32)

# ───────────── cache for sparse decoder weights ─────────────
_SPARSE_CACHE: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}


def get_sparse_decoder(W_full: np.ndarray, b_full: np.ndarray,
                       idx_tuple: tuple[int, ...]):
    """Return (W_small, b_small) for this set of flat idxs, with caching."""
    if idx_tuple not in _SPARSE_CACHE:
        cols = np.asarray(idx_tuple, dtype=np.intp)  # guarantee integer dtype
        _SPARSE_CACHE[idx_tuple] = (W_full[:, cols], b_full[cols])
    return _SPARSE_CACHE[idx_tuple]

# ───────────── reconstruction (pure NumPy) ─────────────

def reconstruct_single(T: float, E_idx: int,
                       W0: np.ndarray, b0: np.ndarray, alpha: float,
                       W_dec: np.ndarray, b_dec: np.ndarray,
                       scaler_T, scaler_spec,
                       hann_win, scaling_factor) -> float:
    """Return reconstructed XS (scalar) with no TF ops."""
    # 1) latent activations (in_dim=1 → 16) + LeakyReLU
    T_norm = (T - scaler_T.mean_[0]) / scaler_T.scale_[0]
    z = T_norm * W0[0] + b0          # (16,)
    hidden = np.where(z > 0, z, alpha * z)  # LeakyReLU

    # 2) gather decoder slice
    seg_idxs, local = mapSegIdx(E_idx)
    fidxs = flat_idxs(seg_idxs)
    W_small, b_small = get_sparse_decoder(W_dec, b_dec, tuple(fidxs))

    # 3) latent→spec slice (16 × 12) + bias
    spec_scaled = hidden @ W_small + b_small          # (12,)

    # 4) un‑scale
    spec = spec_scaled * scaler_spec.scale_[fidxs] + scaler_spec.mean_[fidxs]
    spec = spec.reshape(H, len(seg_idxs), C)
    spec = spec[..., 0] + 1j * spec[..., 1]

    # 5) inverse STFT & overlap‑add
    xs = 0.0
    for i in range(len(seg_idxs)):
        segment = np.fft.irfft(spec[:, i], n=WINDOW_SAMPS) * hann_win
        xs += segment[local[i] % WINDOW_SAMPS]
    return xs / scaling_factor

# ───────────── CLI wrapper ─────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--E_idx", type=int, required=True)
    args = ap.parse_args()

    # --- artefacts ---
    hann_win = hann(WINDOW_SAMPS)
    scaling_factor = np.sum(hann_win**2) / STEP_SAMPS

    with h5py.File(SCALER_PATH, "r") as hf:
        scaler_spec = StandardScaler()
        scaler_spec.scale_ = hf["spec_scale"][:]
        scaler_spec.mean_  = hf["spec_mean"][:]
        scaler_spec.n_features_in_ = scaler_spec.scale_.size
        scaler_spec.n_samples_seen_ = 1

        scaler_T = StandardScaler()
        scaler_T.scale_ = hf["T_scale"][:]
        scaler_T.mean_  = hf["T_mean"][:]
        scaler_T.n_features_in_ = 1
        scaler_T.n_samples_seen_ = 1

        # --- read weights once (TF only for HDF5 parsing) ---
    model = load_model(MODEL_PATH, compile=False)

    dense0: Dense = model.layers[1]           # input → latent
    W0, b0 = dense0.kernel.numpy(), dense0.bias.numpy()  # (1,16) and (16,)

    lrelu_layer = model.layers[2]             # LeakyReLU
    if isinstance(lrelu_layer, LeakyReLU):
        alpha = getattr(lrelu_layer, "alpha",
                        getattr(lrelu_layer, "negative_slope", 0.1))
    else:
        alpha = 0.1

    dense_dec: Dense = model.layers[3]        # latent → full spectrogram
    W_dec, b_dec = dense_dec.kernel.numpy(), dense_dec.bias.numpy()

    # --- reconstruct & time ---
    t0 = time.perf_counter()
    xs_val = reconstruct_single(args.T, args.E_idx,
                                W0, b0, alpha,
                                W_dec, b_dec,
                                scaler_T, scaler_spec,
                                hann_win, scaling_factor)
    dt = (time.perf_counter() - t0) * 1e3

    warm = "warm cache" if _SPARSE_CACHE else "cold cache"
    print(f"XS(T={args.T}, idx={args.E_idx})  →  {xs_val:.6e}   [{dt:.2f} ms ({warm})]")


if __name__ == "__main__":
    main()
