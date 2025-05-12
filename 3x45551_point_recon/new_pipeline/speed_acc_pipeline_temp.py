#input: python whole_pipeline_temp.py --mode speed --batch 20000000 --chunk 10000000 --tmin 1000.0 --tmax 1000.0 --precision float32
#!/usr/bin/env python3
"""
gpu_full_pipeline.py — Batched GPU inference + direct inverse DFT reconstruction
===========================================================================
Full pipeline entirely on GPU, loading weights & biases from HDF5.
Handles large batches via optional chunking to avoid OOM.
"""

import argparse
import time
import glob
import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision

# ─── Global Constants ─────────────────────────────────────────────────────
E_MIN = 1e4 * 1e-6  # MeV
E_MAX = 1e5 * 1e-6  # MeV
WINDOW_SIZE = 0.000398
STEP_SIZE = 0.000199
H, W_TIME, C = 3, 5313, 2

# Paths (do not change)
BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
WEIGHTS_H5 = "model_weights.h5"
SCALER_PATH = "spec_scalers_3_5313.h5"
MODEL_PATH = '/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/martin_pipeline/best_model_3_5313.h5'
TEMP_DATA_DIR = r'/mnt/d/800_1200'

# Derived globals (init in main)
fs = None
WINDOW_SAMPS = None
STEP_SAMPS = None
PAD = None
spec_scale = spec_mean = T_scale = T_mean = None
W0 = b0 = alpha = W_dec = b_dec = None
exp_coeffs = hann_win = scale_fac = SEG_OFF = None


def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key="xs_data", compression="gzip")
    subset = df[df["T"] == base_t]
    subset = subset[(subset["ERG"] >= e_min) & (subset["ERG"] <= e_max)]
    E = subset["ERG"].to_numpy()
    idx = np.argsort(E)
    return E[idx], len(E)


def init_globals():
    global fs, WINDOW_SAMPS, STEP_SAMPS, PAD, exp_coeffs, hann_win, scale_fac, SEG_OFF
    base_e, fs = load_base(E_MIN, E_MAX)
    WINDOW_SAMPS = int(WINDOW_SIZE * fs)
    STEP_SAMPS = int(STEP_SIZE * fs)
    PAD = WINDOW_SAMPS

    e = tf.cast(tf.range(WINDOW_SAMPS), tf.float32)
    f = tf.cast(tf.range(H), tf.float32)[:, None]
    angle = 2.0 * np.pi * f * e[None, :] / WINDOW_SAMPS
    exp_coeffs = tf.exp(tf.complex(tf.zeros_like(angle), angle))

    hann_win = tf.constant(np.hanning(WINDOW_SAMPS).astype('float32'))
    scale_fac = tf.reduce_sum(hann_win**2) / STEP_SAMPS
    SEG_OFF = tf.range(WINDOW_SAMPS // STEP_SAMPS, dtype=tf.int32)

    return base_e


def load_scalers():
    global spec_scale, spec_mean, T_scale, T_mean
    with h5py.File(SCALER_PATH, 'r') as hf:
        spec_scale = tf.constant(hf['spec_scale'][:], tf.float32)
        spec_mean = tf.constant(hf['spec_mean'][:], tf.float32)
        T_scale = tf.constant(hf['T_scale'][:], tf.float32)
        T_mean = tf.constant(hf['T_mean'][:], tf.float32)


def load_weights():
    global W0, b0, alpha, W_dec, b_dec
    with h5py.File(WEIGHTS_H5, 'r') as hf:
        W0 = tf.constant(hf['W0'][:], tf.float32)
        b0 = tf.constant(hf['b0'][:], tf.float32)
        alpha = float(hf['alpha'][()])
        W_dec = tf.constant(hf['W_dec'][:], tf.float32)
        b_dec = tf.constant(hf['b_dec'][:], tf.float32)


# Use decorator to compile into graph for GPU
@tf.function(experimental_compile=True)
def reconstruct_batch(T_batch: tf.Tensor, E_batch: tf.Tensor) -> tf.Tensor:
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm, 1), W0) + b0, alpha)

    first = tf.cast(tf.math.ceil((E_batch - WINDOW_SAMPS) / STEP_SAMPS - 0.5), tf.int32)
    segs = first[:, None] + SEG_OFF[None, :]

    N = tf.shape(T_batch)[0]
    f_off = tf.reshape(tf.range(H, dtype=tf.int32) * W_TIME * C, [1, H, 1, 1])
    segs_e = tf.reshape(segs, [N, 1, -1, 1])
    ch_off = tf.reshape(tf.range(C, dtype=tf.int32), [1, 1, 1, C])
    flat_idx = tf.reshape(f_off + segs_e * C + ch_off, [N, -1])

    W_flat = tf.gather(W_dec, flat_idx, axis=1)
    b_sub = tf.gather(b_dec, flat_idx, axis=0)

    # Corrected einsum equation: 'nl,lnk->nk'
    spec_scaled = tf.einsum('nl,lnk->nk', hidden, W_flat) + b_sub

    scale_flat = tf.gather(spec_scale, flat_idx)
    mean_flat = tf.gather(spec_mean, flat_idx)

    spec = spec_scaled * scale_flat + mean_flat
    spec = tf.reshape(spec, [N, H, -1, C])
    spec_c = tf.complex(spec[..., 0], spec[..., 1])

    segments = tf.signal.irfft(
        tf.transpose(spec_c, [0, 2, 1]),
        fft_length=[WINDOW_SAMPS]
    )
    segments *= hann_win

    local = (E_batch[:, None] - (segs + 1) * STEP_SAMPS) % WINDOW_SAMPS
    vals = tf.gather(segments, local, axis=2, batch_dims=2)
    xs = tf.reduce_sum(vals, axis=1) / scale_fac
    return xs


def process_in_chunks(temps, eidxs, chunk_size):
    results = []
    total = tf.shape(temps)[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        xs_chunk = reconstruct_batch(temps[start:end], eidxs[start:end])
        results.append(xs_chunk)
    return tf.concat(results, axis=0)


def load_temperature(test_temp, base_e, file_dir):
    for fp in glob.glob(os.path.join(file_dir, '*.h5')):
        df = pd.read_hdf(fp, key='xs_data', compression='gzip')
        if test_temp not in df['T'].values:
            continue
        subset = df[df['T'] == test_temp]
        subset = subset[(subset['ERG']>=E_MIN)&(subset['ERG']<=E_MAX)]
        if len(subset)<2:
            continue
        E, xs = subset['ERG'].to_numpy(), subset['XS'].to_numpy()
        idx = np.argsort(E)
        E, xs = E[idx], xs[idx]
        interp = interp1d(E, xs, kind='cubic', fill_value='extrapolate')
        return np.pad(interp(base_e), (PAD,PAD), mode='constant')
    return None


def analyse(base_e, reconstructed, temps, eidxs, file_dir):
    padded = load_temperature(temps[0].numpy(), base_e, file_dir)
    if padded is None:
        return
    xs_rec = reconstructed.numpy()
    idxs = eidxs.numpy()
    orig_vals = padded[idxs]
    rel_err = np.abs(xs_rec - orig_vals)/np.abs(orig_vals)*100

    # Plot original vs reconstructed point
    plt.figure(figsize=(8,5))
    plt.plot(base_e, padded[PAD:-PAD], label='Original Signal')
    plt.scatter(base_e[idxs-PAD], xs_rec, c='r', s=10, label='Reconstructed')
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('Energy'); plt.ylabel('XS')
    plt.legend(); plt.grid(True)
    plt.savefig('./data_reconstruct/xs.png', dpi=200)
    plt.close()

    # Plot relative error (sorted)
    sorted_idx = np.argsort(idxs)
    idxs_sorted = idxs[sorted_idx]
    rel_err_sorted = rel_err[sorted_idx]
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs_sorted-PAD], rel_err_sorted, marker='o', linestyle='-')
    plt.xscale('log'); plt.xlabel('Energy Index'); plt.ylabel('Relative Error (%)')
    plt.title('Relative Error vs Energy'); plt.grid(True)
    plt.savefig('./data_reconstruct/relative_error.png', dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="speed")
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--tmin', type=float, default=1000.0)
    parser.add_argument('--tmax', type=float, default=1000.0)
    parser.add_argument('--chunk', type=int, default=None,
                        help='Optional chunk size for batch processing')
    parser.add_argument('--precision-policy', type=str,
                        choices=['float16','mixed_float16','float32'],
                        default='mixed_float16',
                        help='Precision policy')
    args = parser.parse_args()

    policy = args.precision_policy
    mixed_precision.set_global_policy(policy)

    base_e = init_globals()
    load_scalers()
    load_weights()

    temps = tf.random.uniform([args.batch], args.tmin, args.tmax, tf.float32)
    global fs
    eidxs = tf.random.uniform([args.batch], PAD, fs-PAD, tf.int32)

    # Warm-up
    _ = reconstruct_batch(temps[:1], eidxs[:1])

    # Timed run with optional chunking
    start = time.perf_counter()
    if args.chunk and args.chunk < args.batch:
        xs = process_in_chunks(temps, eidxs, args.chunk)
    else:
        xs = reconstruct_batch(temps, eidxs)
    dt = (time.perf_counter() - start) * 1e6 / tf.cast(args.batch, tf.float32)
    print(f"{args.batch:,} points → {dt:.2f} µs/point")

    if (args.mode == "accuracy"):
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)


if __name__ == '__main__':
    main()
