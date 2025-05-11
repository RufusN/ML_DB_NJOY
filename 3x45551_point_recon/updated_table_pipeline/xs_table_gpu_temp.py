#!/usr/bin/env python3
"""
xs_table_gpu_with_accuracy.py – GPU inference timing and accuracy via precomputed weight table
===========================================================================================

This script does two things:
 1. Timed batched inference of XS values via table lookup (random queries).
 2. Accuracy evaluation over a range of temperatures: reconstruct specific points and save relative error plots.

Prerequisites (same directory or adjust paths):
  • 3x45551_950_1050_spec_scalers.h5  (T-scale/mean)
  • w_table.h5                        (W_tab, b_tab, E_idxs, W0, b0, alpha)
  • capture_xs_data_0.h5             (base cross-section table)
  • spectrogram *.h5 files under test directory

Usage examples:
    # 1) Time inference on 30M random queries at T=1000K, chunked in 1M
    python xs_table_gpu_with_accuracy.py --batch 30000000 --chunk 1000000

    # 2) Evaluate accuracy for T from 950 to 1050 in steps of 0.1, sampling every 100 points
    python xs_table_gpu_with_accuracy.py --tmin 950 --tmax 1050 --error-step 100

    # 3) Both tasks
    python xs_table_gpu_with_accuracy.py --batch 1000000 --chunk 200000 --test-temp 1000 --tmin 950 --tmax 1050 --error-step 50
"""
import argparse
import time
import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

#------------------------------------------------------------------------------
# I/O paths (edit as needed)
SCALER_PATH = "/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/3x45551_950_1050_spec_scalers.h5"
TABLE_PATH  = "/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/w_table.h5"
BASE_XS_FILE = "/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
SPEC_DIR     = "/mnt/d/new_specs_3_45551/"

# Defaults
DEFAULT_BATCH_SIZE = 10_000_000
DEFAULT_CHUNK_SIZE = 1_000_000

@tf.function
def xs_from_table(T_batch, E_batch,
                  W0, b0, alpha,
                  W_tab, b_tab,
                  T_scale, T_mean):
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.expand_dims(T_norm,1) @ W0 + b0, alpha)
    wvec = tf.gather(W_tab, E_batch)
    bvec = tf.gather(b_tab, E_batch)
    return tf.reduce_sum(hidden * wvec, axis=1) + bvec

# Batched inference
def batched_xs(T_batch, E_batch, W0, b0, alpha, W_tab, b_tab, T_scale, T_mean, chunk_size):
    parts = []
    for i in range(0, T_batch.shape[0], chunk_size):
        end = min(i + chunk_size, T_batch.shape[0])
        parts.append(xs_from_table(
            T_batch[i:end], E_batch[i:end],
            W0, b0, alpha, W_tab, b_tab, T_scale, T_mean
        ))
    return tf.concat(parts, axis=0)

# Load precomputed table + scalers
def load_table_and_scalers():
    with h5py.File(SCALER_PATH,'r') as hf:
        T_scale = hf['T_scale'][:].astype('float32')
        T_mean  = hf['T_mean'] [:].astype('float32')
    with h5py.File(TABLE_PATH,'r') as hf:
        W_tab  = hf['W_tab'][:].astype('float32')
        b_tab  = hf['b_tab'][:].astype('float32')
        E_idxs = hf['E_idxs'][:].astype('int32')
        W0     = hf['W0'][:].astype('float32')
        b0     = hf['b0'][:].astype('float32')
        alpha  = float(hf['alpha'][()])
    return (tf.constant(W_tab), tf.constant(b_tab), tf.constant(E_idxs),
            tf.constant(W0), tf.constant(b0), alpha,
            tf.constant(T_scale), tf.constant(T_mean))

def load_base():
    with h5py.File("/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/base_energy_grid.h5", 'r') as hf:
        base_E = hf['Base_E'][:]
    return base_E

# Load XS base values for a given temperature
def load_data(T_test, Eidx, E_min, E_max, base_E):
    file_path=r'/mnt/d/800_1200'
    all_files = glob.glob(os.path.join(file_path, "*.h5"))
    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        
        try:
            with h5py.File(file_path, "r") as h5_file:
                print(f"------ Processing: {file_path}")
                
                data = pd.read_hdf(file_path, key="xs_data", compression="gzip")
                
                # Check if the test_temp exists in the dataset
                print("Checking", T_test, data["T"].values)
                for i in data["T"].values:
                    print(i)
                if T_test not in data["T"].values:
                    continue
                
                print(f"Found data for T = {T_test}!")
                subset_T = data[data["T"] == T_test].copy()
                mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
                subset_range = subset_T[mask].copy()
                
                if len(subset_range) < 2:
                    print(f"[Warning] Insufficient data ({len(subset_range)}) in range [{E_min}, {E_max}]. Skipping.")
                    continue
                E = subset_range["ERG"].to_numpy()
                xs = subset_range["XS"].to_numpy()
                
                sort_idx = np.argsort(E)
                E = E[sort_idx]
                xs = xs[sort_idx]
                interp_func = interp1d(
                    E, xs, kind="cubic", fill_value="extrapolate"
                )
                signal = interp_func(base_E)
                
                return signal[Eidx] # Exit function as soon as a valid instance is found
        
        except Exception as e:
            print(f"Error reading HDF5 file {file_path}: {e}")

# Plot and save relative error for one temperature
def plot_relative_error(E_grid, true_xs, pred_xs, T_val):
    rel_err = np.abs(pred_xs - true_xs) / np.abs(true_xs) * 100
    plt.figure(); plt.plot(E_grid, rel_err, 'o-')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy (MeV)'); plt.ylabel('Relative Error (%)')
    plt.title(f'Relative Error @ T={T_val}K')
    plt.grid(True)
    fname = f"rel_err_T{T_val:.1f}.png"
    plt.savefig(fname, dpi=200)
    print(f"Saved rel error plot → {fname}")
    plt.close()

    plt.figure()
    plt.plot(E_grid, true_xs, label = 'true xs')
    plt.plot(E_grid, pred_xs, label = 'pred xs')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy (MeV)'); plt.ylabel('XS')
    plt.title(f'XS @ T={T_val}K')
    plt.grid(True)
    fname = f"XS_T{T_val:.1f}.png"
    plt.legend()
    plt.savefig(fname, dpi=200)
    print(f"Saved xs plot → {fname}")
    plt.close()
#------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--chunk', type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument('--tmin', type=float, default=950.0)
    p.add_argument('--tmax', type=float, default=1050.0)
    p.add_argument('--accuracy', type=bool, default=False)
    p.add_argument('--step', type=float, default=0.1)
    args = p.parse_args()

    # load once
    W_tab, b_tab, E_idxs, W0, b0, alpha, T_scale, T_mean = load_table_and_scalers()

    # 1) Timing
    print("Running Timing ............................ lol xd")
    N = args.batch
    tmin = args.tmin
    tmax = args.tmax 
    chunk_size = args.chunk
    temps = tf.random.uniform([N], tmin, tmax, dtype=tf.float32)
    idxs = tf.random.uniform([N], 0, E_idxs.shape[0], dtype=tf.int32)
    E_batch = tf.gather(E_idxs, idxs)

    xs_from_table(temps[:1], E_batch[:1], W0,b0,alpha, W_tab,b_tab, T_scale,T_mean)
    t0 = time.perf_counter()
    _ = batched_xs(temps, E_batch, W0,b0,alpha, W_tab,b_tab, T_scale,T_mean, chunk_size)
    dt = time.perf_counter()-t0
    print(f"{N:,} total points (chunk={chunk_size:,}) → {dt*1e3:.1f} ms  |  {dt/N*1e6:.3f} µs per point")

    # 2) Accuracy over T-range
    if (args.accuracy):
        print("Running Accuracy ............................ lol xd")

        base_E = load_base()
        E_min, E_max = base_E[0], base_E[-1]
        step = args.step
        temps = tf.range(tmin, tmax, delta=step, dtype=tf.float32)
        N = tf.shape(temps)[0].numpy()

        log_min, log_max = np.log(base_E[0]), np.log(base_E[-1])
        logs     = np.random.uniform(log_min, log_max, size=N).astype(np.float32)
        energies = np.exp(logs)
        Eidx = np.searchsorted(base_E, energies, side='left')
        Eidx = np.clip(Eidx, 0, base_E.shape[0]-1)
        Eidx = np.sort(Eidx)

        for Tval in temps:
            Tval = np.round(Tval,1)
            print("Tval", Tval)
            true_xs = load_data(Tval, Eidx, E_min, E_max, base_E)
            xs_from_table(temps[:1], Eidx[:1], W0,b0,alpha, W_tab,b_tab, T_scale,T_mean)
            pred = xs_from_table(temps, Eidx, W0,b0,alpha, W_tab,b_tab, T_scale,T_mean).numpy()
            print(len(true_xs), len(pred), true_xs[0], pred[0])
            plot_relative_error(base_E[Eidx], true_xs, pred, Tval)

if __name__=='__main__':
    main()
