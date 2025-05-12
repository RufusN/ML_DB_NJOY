#!/usr/bin/env python3
"""
gpu_full_pipeline.py — batched GPU inference + direct inverse DFT reconstruction
===========================================================================
Full pipeline entirely on GPU, loading weights & biases from HDF5:

  1. Read W0, b0, alpha, W_dec, b_dec from `model_weights.h5`
  2. Read T_scale/T_mean and spec_scale/spec_mean from `3x45551_950_1050_spec_scalers.h5`
  3. Sparse-decode spectrogram slices from the latent model
  4. Direct inverse DFT + Hann-window overlap-add per sample

Usage:
    python gpu_full_pipeline.py --batch 8192

Outputs:
  • Prints µs/point for the batch
"""

import argparse
import time
import h5py
import numpy as np
import tensorflow as tf

# ─── Constants ─────────────────────────────────────────────────────────────
WINDOW_SAMPS = 4
STEP_SAMPS   = WINDOW_SAMPS // 2
H, W_TIME, C = 3, 45551, 2
n_over = WINDOW_SAMPS // STEP_SAMPS

# ─── Paths ─────────────────────────────────────────────────────────────────
WEIGHTS_H5   = "model_weights.h5"               # your saved W0,b0,alpha,W_dec,b_dec
SCALER_PATH  = "3x45551_950_1050_spec_scalers.h5"

# ─── Precompute DFT coefficients & window on GPU ────────────────────────────
e = tf.cast(tf.range(WINDOW_SAMPS), tf.float32)
f = tf.cast(tf.range(H), tf.float32)[:, None]
angle = 2.0 * np.pi * f * e[None, :] / WINDOW_SAMPS
exp_coeffs = tf.exp(tf.complex(tf.zeros_like(angle), angle))  # [H, WINDOW_SAMPS]

hann_win = tf.constant(np.hanning(WINDOW_SAMPS).astype("float32"))
scale_fac = tf.reduce_sum(hann_win**2) / STEP_SAMPS
SEG_OFF = tf.range(n_over, dtype=tf.int32)  # [n_over]

# ─── GPU batched reconstruction ────────────────────────────────────────────
@tf.function
def reconstruct_batch(T_batch: tf.Tensor, E_batch: tf.Tensor) -> tf.Tensor:
    # 1) Temperature → hidden (LeakyReLU)
    T_norm = (T_batch - T_mean) / T_scale                         # [N]
    hidden = tf.nn.leaky_relu(
        tf.matmul(tf.expand_dims(T_norm, 1), W0) + b0,
        alpha
    )                                                             # [N,16]

    # 2) compute seg indices exactly as CPU version
    first = tf.cast(
        tf.math.ceil((E_batch - WINDOW_SAMPS) / STEP_SAMPS - 0.5),
        tf.int32
    )                                                             # [N]
    segs = tf.expand_dims(first, 1) + SEG_OFF[None, :]            # [N, n_over]

    # 3) build flat indices & sparse-decode
    N     = tf.shape(T_batch)[0]
    f_off = tf.reshape(tf.range(H, dtype=tf.int32)*(W_TIME*C), [1, H, 1, 1])
    segs_e= tf.reshape(segs, [N, 1, n_over, 1])
    ch_off= tf.reshape(tf.range(C, dtype=tf.int32), [1, 1, 1, C])
    flat_idx = tf.reshape(f_off + segs_e * C + ch_off, [N, -1])   # [N, n_flat]

    W_flat      = tf.gather(W_dec, flat_idx, axis=1)              # [latent, N, n_flat]
    W_sub       = tf.transpose(W_flat, [1, 0, 2])                 # [N, latent, n_flat]
    b_sub       = tf.gather(b_dec, flat_idx, axis=0)              # [N, n_flat]
    spec_scaled = tf.einsum('nl,nlf->nf', hidden, W_sub) + b_sub   # [N, n_flat]

    # 4) Un-scale & reshape → complex spectrogram
    scale_flat  = tf.gather(spec_scale, flat_idx)
    mean_flat   = tf.gather(spec_mean,  flat_idx)
    spec        = spec_scaled * scale_flat + mean_flat            # [N, n_flat]
    spec        = tf.reshape(spec, [N, H, n_over, C])             # [N,H,n_over,C]
    spec_c      = tf.complex(spec[..., 0], spec[..., 1])           # [N,H,n_over]

    # 5) inverse RFFT + Hann overlap-add
    spec_t   = tf.transpose(spec_c, [0, 2, 1])                    # [N,n_over,H]
    segments = tf.signal.irfft(spec_t, fft_length=[WINDOW_SAMPS]) # [N,n_over,WINDOW_SAMPS]
    segments = segments * hann_win                                # apply window

    # 6) compute local offsets (with +1 hop shift)
    local = (E_batch[:, None] - (segs + 1) * STEP_SAMPS) % WINDOW_SAMPS  # [N,n_over]

    # 7) gather & sum
    vals = tf.gather(segments, local, axis=2, batch_dims=2)       # [N,n_over]
    xs   = tf.reduce_sum(vals, axis=1) / scale_fac                # [N]

    return xs

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8192)
    args = p.parse_args()
    N = args.batch

    # Load scalers
    with h5py.File(SCALER_PATH, "r") as hf:
        global T_scale, T_mean, spec_scale, spec_mean
        T_scale    = tf.constant(hf["T_scale"][:].astype("float32"))
        T_mean     = tf.constant(hf["T_mean"] [:].astype("float32"))
        spec_scale = tf.constant(hf["spec_scale"][:].astype("float32"))
        spec_mean  = tf.constant(hf["spec_mean"] [:].astype("float32"))

    # Load network weights from HDF5 instead of full Keras model
    with h5py.File(WEIGHTS_H5, "r") as hf:
        global W0, b0, alpha, W_dec, b_dec
        W0    = tf.constant(hf["W0"][:].astype("float32"))      # shape (1,16)
        b0    = tf.constant(hf["b0"][:].astype("float32"))      # shape (16,)
        alpha = float(hf["alpha"][()])                          # scalar
        W_dec = tf.constant(hf["W_dec"][:].astype("float32"))   # shape (16, H*W_TIME*C)
        b_dec = tf.constant(hf["b_dec"][:].astype("float32"))   # shape (H*W_TIME*C,)

    # Run a timed batch with random Ts and E_idxs
    temps = tf.random.uniform([N], 950.0, 1050.0, dtype=tf.float32)
    eidxs = tf.random.uniform([N], 100, 90001, dtype=tf.int32)

    # temps = tf.constant([1000.0], dtype=tf.float32)
    # eidxs = tf.constant([700],    dtype=tf.int32)

    # Warm‐up
    _ = reconstruct_batch(temps[:1], eidxs[:1])

    # Timed run
    t0 = time.perf_counter()
    xs = reconstruct_batch(temps, eidxs)
    # print(xs)
    dt = (time.perf_counter() - t0) * 1e6 / tf.cast(N, tf.float32)  # µs/point
    print(f"{N:,} points → {dt:.2f} µs/point")

if __name__ == "__main__":
    main()
