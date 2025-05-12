#!/usr/bin/env python3
"""
pipeline_with_table.py — Unified inference pipeline: full GPU vs pre‑baked table
==============================================================================

Supports two backends for MLP–Doppler XS lookup:
  • full: batched GPU inference + inverse DFT (original pipeline)
  • table: fast lookup in precomputed W_tab/b_tab table (original query)

User can choose speed benchmarking or accuracy analysis.

Usage examples:
  # speed test using table lookup
  python pipeline_with_table.py --backend table --mode speed \
      --batch 20000000 --chunk 1000000 --tmin 1000.0 --tmax 1000.0 --table w_table.h5

  # accuracy test using full pipeline
  python pipeline_with_table.py --backend full --mode accuracy --batch 8192 \
      --tmin 950.0 --tmax 1050.0 --chunk 1024 --precision float32
"""
import argparse, time, glob, os
import h5py, numpy as np, pandas as pd, tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Global constants for full pipeline (unchanged)
#E_MIN = 1e4 * 1e-6  # MeV
#E_MAX = 1e5 * 1e-6  # MeV
#WINDOW_SIZE = 0.000398
#STEP_SIZE = 0.000199

E_MIN = 1e4 * 1e-6  # MeV
E_MAX = 1e6 * 1e-6  # MeV
WINDOW_SIZE = 0.00004744
STEP_SIZE = 0.00002314
H, W_TIME, C = 3, 45551, 2
# Paths for full pipeline
BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
WEIGHTS_H5 = 'model_weights.h5'
SCALER_PATH = '3x45551_950_1050_spec_scalers.h5'
TEMP_DATA_DIR = r'/mnt/d/800_1200'

# Globals to initialize
fs = WINDOW_SAMPS = STEP_SAMPS = PAD = None
spec_scale = spec_mean = T_scale = T_mean = None
W0 = b0 = alpha = W_dec = b_dec = None
exp_coeffs = hann_win = scale_fac = SEG_OFF = None

# Subroutines for full pipeline ----------------------------------------------
def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key='xs_data', compression='gzip')
    subset = df[df['T']==base_t]
    subset = subset[(subset['ERG']>=e_min)&(subset['ERG']<=e_max)]
    E = subset['ERG'].to_numpy()
    idx = np.argsort(E)
    return E[idx], len(E)


def init_full():
    global fs, WINDOW_SAMPS, STEP_SAMPS, PAD, hann_win, scale_fac, SEG_OFF
    base_e, fs = load_base(E_MIN, E_MAX)
    WINDOW_SAMPS = int(WINDOW_SIZE * fs)
    STEP_SAMPS = int(STEP_SIZE  * fs)
    PAD = WINDOW_SAMPS
    hann_win = tf.constant(np.hanning(WINDOW_SAMPS).astype('float32'))
    scale_fac = tf.reduce_sum(hann_win**2) / STEP_SAMPS
    SEG_OFF = tf.range(WINDOW_SAMPS//STEP_SAMPS, dtype=tf.int32)
    return base_e


def load_scalers_full():
    global spec_scale, spec_mean, T_scale, T_mean
    with h5py.File(SCALER_PATH,'r') as hf:
        spec_scale = tf.constant(hf['spec_scale'][:], tf.float32)
        spec_mean  = tf.constant(hf['spec_mean'][:],  tf.float32)
        T_scale    = tf.constant(hf['T_scale'][:],    tf.float32)
        T_mean     = tf.constant(hf['T_mean'][:],     tf.float32)


def load_weights_full():
    global W0, b0, alpha, W_dec, b_dec
    with h5py.File(WEIGHTS_H5,'r') as hf:
        W0    = tf.constant(hf['W0'][:],   tf.float32)
        b0    = tf.constant(hf['b0'][:],   tf.float32)
        alpha = float(hf['alpha'][()])
        W_dec = tf.constant(hf['W_dec'][:],tf.float32)
        b_dec = tf.constant(hf['b_dec'][:],tf.float32)

@tf.function(experimental_compile=True)
def reconstruct_full(T_batch, E_batch):
    # normalize temperature
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm,1), W0)+b0, alpha)
    # compute segments
    first = tf.cast(tf.math.ceil((E_batch - WINDOW_SAMPS)/STEP_SAMPS - 0.5),tf.int32)
    segs = first[:,None] + SEG_OFF[None,:]
    # build flat indices
    N = tf.shape(T_batch)[0]
    f_off = tf.reshape(tf.range(H, dtype=tf.int32)*W_TIME*C, [1,H,1,1])
    segs_e = tf.reshape(segs, [N,1,-1,1])
    ch_off = tf.reshape(tf.range(C, dtype=tf.int32), [1,1,1,C])
    flat_idx = tf.reshape(f_off + segs_e* C + ch_off, [N,-1])
    # gather weights/bias
    W_flat = tf.gather(W_dec,flat_idx, axis=1)
    b_flat = tf.gather(b_dec, flat_idx, axis=0)
    spec_scaled = tf.einsum('nl,lnk->nk', hidden, W_flat) + b_flat
    # un-scale
    scale_flat = tf.gather(spec_scale,flat_idx)
    mean_flat  = tf.gather(spec_mean, flat_idx)
    spec = spec_scaled * scale_flat + mean_flat
    spec_c = tf.complex(
        tf.reshape(spec[...,0],[N,-1,H]),
        tf.reshape(spec[...,1],[N,-1,H])
    )
    segments = tf.signal.irfft(tf.transpose(spec_c,[0,2,1]), fft_length=[WINDOW_SAMPS])
    segments *= hann_win
    local = (E_batch[:,None] - (segs+1)*STEP_SAMPS) % WINDOW_SAMPS
    vals = tf.gather(segments, local, axis=2, batch_dims=2)
    xs = tf.reduce_sum(vals, axis=1)/scale_fac
    return xs

# Subroutines for table backend ---------------------------------------------
def load_table(table_h5):
    with h5py.File(table_h5,'r') as hf:
        W_tab   = tf.constant(hf['W_tab'][:])
        b_tab   = tf.constant(hf['b_tab'][:])
        W0      = tf.constant(hf['W0'][:])
        b0      = tf.constant(hf['b0'][:])
        alpha   = float(hf['alpha'][()])
        T_scale = tf.constant(hf['T_scale'][:])
        T_mean  = tf.constant(hf['T_mean'][:])
    @tf.function#(experimental_compile=True)
    def query_table(T_batch, E_batch):
        T_norm = (T_batch - T_mean) / T_scale
        hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm,1), W0)+b0, alpha)
        W_vec = tf.gather(W_tab, E_batch)
        b_vec = tf.gather(b_tab, E_batch)
        return tf.reduce_sum(hidden * W_vec, axis=1) + b_vec
    return query_table

# Generic timing and accuracy ------------------------------------------------
def benchmark(fn, temps, eidxs, chunk):
    # warm-up
    _ = fn(temps[:1], eidxs[:1])
    start = time.perf_counter()
    if chunk and chunk<temps.shape[0]:
        # chunked
        out = []
        N = temps.shape[0]
        for i in range(0,N,chunk):
            out.append(fn(temps[i:i+chunk], eidxs[i:i+chunk]))
        xs = tf.concat(out, axis=0)
    else:
        xs = fn(temps, eidxs)
    dt = (time.perf_counter()-start)*1e6/temps.shape[0]
    print(f"{temps.shape[0]:,} points → {dt:.5f} µs/point")
    return xs

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
        return np.pad(base_e, (PAD,PAD), mode='constant'), np.pad(interp(base_e), (PAD,PAD), mode='constant')
    return None

def analyse(base_e, reconstructed, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(temps[0].numpy(), base_e, file_dir)
    if padded is None:
        return
    xs_rec = reconstructed.numpy()
    idxs = eidxs.numpy()
    orig_vals = padded[idxs]
    rel_err = np.abs(xs_rec - orig_vals)/np.abs(orig_vals)*100
    # Plot original vs reconstructed point
    plt.figure(figsize=(8,5))
    plt.plot(padded_e, padded, label='Original Signal')
    plt.scatter(padded_e[idxs], xs_rec, c='r', s=10, label='Reconstructed')
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


# Main ------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['full','table'], required=True,
                   help='Inference backend')
    p.add_argument('--mode',    choices=['speed','accuracy'], default='speed',
                   help='Operation mode')
    p.add_argument('--batch',   type=int, default=8192)
    p.add_argument('--chunk',   type=int, default=None)
    p.add_argument('--tmin',    type=float, default=1000.0)
    p.add_argument('--tmax',    type=float, default=1000.0)
    p.add_argument('--precision', dest='precision_policy',
                   choices=['float16','mixed_float16','float32'],
                   default='mixed_float16')
    p.add_argument('--table',   type=str, default='w_table.h5',
                   help='Precomputed table file (for table backend)')
    args = p.parse_args()

    # set precision
    mixed_precision.set_global_policy(args.precision_policy)
    # prepare inputs
    temps = tf.random.uniform([args.batch], args.tmin, args.tmax, tf.float32)
    # energy indices
    if args.backend=='full':
        base_e = init_full()
        eidxs  = tf.random.uniform([args.batch], fs-WINDOW_SAMPS, fs-PAD,
                                   tf.int32)
    else:
        # table backend: valid E_idxs in [0, W_TIME)
        eidxs  = tf.random.uniform([args.batch], 4, 92763-4, tf.int32)

    # build fn
    if args.backend=='full':
        load_scalers_full(); load_weights_full()
        fn = reconstruct_full
    else:
        fn = load_table(args.table)

    # run benchmark
    xs = benchmark(fn, temps, eidxs, args.chunk)

    # accuracy
    if args.mode=='accuracy':
        #(base_e, reconstructed, temps, eidxs, file_dir)
        #analyse(xs, temps, eidxs, base_e if args.backend=='full' else None,
        #                 PAD if args.backend=='full' else None,
        #                 TEMP_DATA_DIR)
        base_e = init_full()
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)

if __name__=='__main__':
    main()
