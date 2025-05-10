#!/usr/bin/env python3
"""
gpu_full_pipeline.py — batched GPU inference + direct inverse DFT reconstruction
===========================================================================
Full pipeline entirely on GPU:
  1. Sparse-decode spectrogram slices from latent model
  2. Direct inverse DFT + Hann-window overlap-add per sample

Usage:
    python gpu_full_pipeline.py --batch 8192

Prerequisites:
  • 3x45551_950_1050_spec_scalers.h5
  • 3x45551_950_1050.keras
  • base_energy_grid.h5 (for completeness)

Outputs:
  • Prints total wall-clock and µs/point for batch
"""
import argparse
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Constants ─────────────────────────────────────────────────────────────
WINDOW_SAMPS = 2048
STEP_SAMPS   = 512
H, W_TIME, C = 3, 45551, 2
n_over = WINDOW_SAMPS // STEP_SAMPS

# ─── Precompute DFT coefficients & window on GPU ────────────────────────────
e = tf.cast(tf.range(WINDOW_SAMPS), tf.float32)
f = tf.cast(tf.range(H), tf.float32)[:, None]
angle = 2.0 * np.pi * f * e[None, :] / WINDOW_SAMPS
exp_coeffs = tf.exp(tf.complex(tf.zeros_like(angle), angle))  # [H, WINDOW_SAMPS]

hann_win = tf.constant(np.hanning(WINDOW_SAMPS).astype('float32'))
scale_fac = tf.reduce_sum(hann_win**2) / STEP_SAMPS
SEG_OFF = tf.range(n_over, dtype=tf.int32)  # [n_over]

# ─── GPU batched reconstruction ────────────────────────────────────────────
@tf.function
def reconstruct_batch(T_batch, E_batch):
    # Latent layer: T -> 16-dim hidden
    T_norm = (T_batch - T_mean) / T_scale                        # [N]
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm,1), W0) + b0, alpha)  # [N,16]

    # Segment indices
    first = tf.cast(tf.math.ceil((E_batch - WINDOW_SAMPS)/STEP_SAMPS - 0.5), tf.int32)  # [N]
    segs  = tf.expand_dims(first,1) + SEG_OFF[None,:]                                    # [N,n_over]

    # Flat decoder indices [N, H*n_over*C]
    N = tf.shape(T_batch)[0]
    f_off = tf.reshape(tf.range(H, dtype=tf.int32)*(W_TIME*C), [1,H,1,1])               # [1,H,1,1]
    segs_e = tf.reshape(segs, [N,1,n_over,1])                                          # [N,1,n_over,1]
    ch_off = tf.reshape(tf.range(C, dtype=tf.int32), [1,1,1,C])                       # [1,1,1,C]
    flat_idx = f_off + segs_e * C + ch_off                                             # [N,H,n_over,C]
    flat_idx = tf.reshape(flat_idx, [N, -1])                                           # [N,n_flat]

        # Sparse decode: gather per-sample decoder weights
    # 3) Sparse decode: gather weight columns per sample
    # W_dec: [latent, out_dim]; gather axis=1 yields [latent, N, n_flat]
    W_flat = tf.gather(W_dec, flat_idx, axis=1)     # [latent, N, n_flat]
    W_sub  = tf.transpose(W_flat, perm=[1,0,2])     # [N, latent, n_flat]
    # gather biases per sample
    b_sub  = tf.gather(b_dec, flat_idx, axis=0)     # [N, n_flat]
    # compute scaled spectrogram slice: [N, n_flat]
    spec_scaled = tf.einsum('nl,nlf->nf', hidden, W_sub) + b_sub

    # Un-scale and reshape to complex spectrogram
    scale_flat = tf.gather(spec_scale, flat_idx)
    mean_flat  = tf.gather(spec_mean,  flat_idx)
    spec = spec_scaled * scale_flat + mean_flat                                      # [N,n_flat]
    spec = tf.reshape(spec, [N, H, n_over, C])                                       # [N,H,n_over,C]
    spec_c = tf.complex(spec[:,:,:,0], spec[:,:,:,1])                                # [N,H,n_over]

    # Direct inverse DFT + window & overlap-add
    local = (E_batch[:,None] - segs*STEP_SAMPS) % WINDOW_SAMPS                        # [N,n_over]
    exp_sel = tf.gather(exp_coeffs, local, axis=1)                                    # [H,N,n_over]
    exp_sel = tf.transpose(exp_sel, [1,0,2])                                         # [N,H,n_over]

    seg_vals = tf.math.real(tf.reduce_sum(spec_c * exp_sel, axis=1)) / WINDOW_SAMPS   # [N,n_over]
    win_vals = tf.gather(hann_win, local)                                             # [N,n_over]
    xs = tf.reduce_sum(seg_vals * win_vals, axis=1)                                   # [N]
    return xs / scale_fac                                                            # [N]

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()
    N = args.batch

    # Load scalers
    with h5py.File('3x45551_950_1050_spec_scalers.h5','r') as hf:
        global T_scale, T_mean, spec_scale, spec_mean
        T_scale    = tf.constant(hf['T_scale'][:].astype('float32'))
        T_mean     = tf.constant(hf['T_mean'] [:].astype('float32'))
        spec_scale = tf.constant(hf['spec_scale'][:].astype('float32'))
        spec_mean  = tf.constant(hf['spec_mean'] [:].astype('float32'))

    # Load model weights
    model = load_model('3x45551_950_1050.keras', compile=False)
    dense0  = model.layers[1]
    global W0, b0, alpha, W_dec, b_dec
    W0    = tf.constant(dense0.kernel.numpy().astype('float32'))
    b0    = tf.constant(dense0.bias.numpy().astype('float32'))
    lrelu = model.layers[2]
    alpha = float(getattr(lrelu, 'alpha', getattr(lrelu, 'negative_slope', 0.3)))
    dense_dec = model.layers[3]
    W_dec = tf.constant(dense_dec.kernel.numpy().astype('float32'))
    b_dec = tf.constant(dense_dec.bias.numpy().astype('float32'))

    # Random batch
    temps = tf.random.uniform([N], 950., 1050., dtype=tf.float32)
    eidxs = tf.random.uniform([N], 100, 45001, dtype=tf.int32)

    # Warm-up
    reconstruct_batch(temps[:1], eidxs[:1])

    # Timed run
    t0 = time.perf_counter()
    xs = reconstruct_batch(temps, eidxs)
    dt = time.perf_counter() - t0
    print(f"{N:,} points -> {dt*1e3:.1f} ms | {dt/N*1e6:.2f} µs/point")

if __name__ == '__main__':
    main()
