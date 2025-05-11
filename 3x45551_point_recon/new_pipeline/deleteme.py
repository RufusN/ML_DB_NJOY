#!/usr/bin/env python3
"""
gpu_full_pipeline.py — batched GPU inference + direct inverse DFT reconstruction
===========================================================================
Full pipeline entirely on GPU, loading weights & biases from HDF5:

  1. Read W0, b0, alpha, W_dec, b_dec from model_weights.h5
  2. Read T_scale/T_mean and spec_scale/spec_mean from 3x45551_950_1050_spec_scalers.h5
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
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model

def load_base(E_min, E_max):
    base_file_path = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
    Base_Tval = 200.0

    with h5py.File(base_file_path, "r") as h5_file:
        xs_data = h5_file['xs_data']

        try:
            data = pd.read_hdf(base_file_path, key="xs_data", compression="gzip")
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
            exit()

        print(f"\nProcessing T = {Base_Tval} ...")

        # (a) Filter the DataFrame by the current T
        subset_T = data[data["T"] == Base_Tval].copy()

        mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
        subset_range = subset_T[mask].copy()

        print(f"Currently processing T={Base_Tval}...")

        subset_T = data[data["T"] == Base_Tval].copy()
        print(f" subset_T has {len(subset_T)} rows for T={Base_Tval}.")

        mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
        subset_range = subset_T[mask].copy()
        # Extract the columns as NumPy arrays
        Base_E = subset_range["ERG"].to_numpy()
        Base_xs = subset_range["XS"].to_numpy()

        sort_idx = np.argsort(Base_E)
        Base_E  = Base_E[sort_idx]
        Base_xs = Base_xs[sort_idx]

        print("Length of Base Energy Grid: ", len(Base_E))
        fs = len(Base_E)
    return Base_E, fs

# ─── Constants ─────────────────────────────────────────────────────────────
E_min = 1e4 * 1e-6  # MeV
E_max = 1e5 * 1e-6  # MeV
window_size = 0.000398
step_size = 0.000199
h = 3
w = 5313
c = 2

base_e, fs = load_base(E_min, E_max)
print("----- fs", fs)
WINDOW_SAMPS = int(window_size * fs)
STEP_SAMPS   = int(step_size   * fs)
pad          = WINDOW_SAMPS  
H, W_TIME, C = 3, 5313, 2
n_over = WINDOW_SAMPS // STEP_SAMPS

padded_base_e = np.pad(
    base_e, (pad, pad), mode='constant', constant_values=0
)

print("step samps, win samps, f", STEP_SAMPS, WINDOW_SAMPS, fs)

# ─── Paths ─────────────────────────────────────────────────────────────────
WEIGHTS_H5   = "model_weights.h5"               # your saved W0,b0,alpha,W_dec,b_dec
SCALER_PATH  = "spec_scalers_3_5313.h5"

# ─── Precompute DFT coefficients & window on GPU ────────────────────────────
e = tf.cast(tf.range(WINDOW_SAMPS), tf.float32)
f = tf.cast(tf.range(H), tf.float32)[:, None]
angle = 2.0 * np.pi * f * e[None, :] / WINDOW_SAMPS
exp_coeffs = tf.exp(tf.complex(tf.zeros_like(angle), angle))  # [H, WINDOW_SAMPS]

hann_win = tf.constant(np.hanning(WINDOW_SAMPS).astype("float32"))
scale_fac = tf.reduce_sum(hann_win**2) / STEP_SAMPS
SEG_OFF = tf.range(n_over, dtype=tf.int32)  # [n_over]


with h5py.File("/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/martin_pipeline/spec_scalers_3_5313.h5", "r") as hf:
    spec_scale = hf["spec_scale"][:]
    spec_mean  = hf["spec_mean"][:]
    T_scale    = hf["T_scale"][:]
    T_mean     = hf["T_mean"][:]

scaler_spec = StandardScaler()
scaler_spec.scale_ = spec_scale
scaler_spec.mean_  = spec_mean
scaler_spec.var_   = scaler_spec.scale_ ** 2
scaler_spec.n_features_in_ = scaler_spec.scale_.size
scaler_spec.n_samples_seen_ = 1  

scaler_T = StandardScaler()
scaler_T.scale_ = T_scale
scaler_T.mean_  = T_mean
scaler_T.var_   = scaler_T.scale_ ** 2
scaler_T.n_features_in_ = scaler_T.scale_.size
scaler_T.n_samples_seen_ = 1

model = load_model('/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/martin_pipeline/best_model_3_5313.h5', compile = False)

def mapSegIdx(energy_idx, step_samps, window_samps):
    n_overlaps = int(window_samps / step_samps)
    seg_idxs = []
    local_idxs = []
    energy_idx = energy_idx
    seg_idx = int(np.ceil((energy_idx - window_samps)/step_samps - 1 + 0.5))
    for _ in range(n_overlaps):
        seg_idxs.append(seg_idx)
        seg_idx += 1
        start = seg_idx*step_samps
        local_idxs.append(energy_idx-start)
    seg_idxs = np.array(seg_idxs)  
    return seg_idxs, local_idxs

def get_spectrogram_output(model, input_data, scaler_T, scaler_spec, seg_indices, h, w, c):
    T = input_data[0]

    slice_length = len(seg_indices)
    seg_indices = np.array(seg_indices)

    T_norm = scaler_T.transform([[T]])
    input_data = T_norm

    desired_indices = seg_indices
    pred_scaled, desired_indices = predict_subset(model, input_data, desired_indices, h, w, c)
    pred_scaled = np.array(pred_scaled)

    pred_scaled = pred_scaled.squeeze() 
    pred_scaled_flat = pred_scaled.reshape(1, -1)

    pred_unscaled_flat = pred_scaled_flat * scaler_spec.scale_[desired_indices] + scaler_spec.mean_[desired_indices]
    pred_unscaled = pred_unscaled_flat.reshape(h, slice_length, c)  

    spectrogram_real = pred_unscaled[..., 0]
    spectrogram_imag = pred_unscaled[..., 1]

    spectrogram = spectrogram_real + 1j * spectrogram_imag 
    return spectrogram

def predict_subset(model, input_data, seg_indices, h, w, c):
    latent_layer = model.layers[2]
    submodel = tf.keras.Model(inputs=model.input, outputs=latent_layer.output)
    
    hidden_activations = submodel(input_data)  # Expected shape: (batch_size, 16)
    
    # Get the Dense layer that maps the 16 latent nodes to the flattened spectrogram.
    output_dense_layer = model.layers[3]
    W = output_dense_layer.kernel  
    b = output_dense_layer.bias   
    
    desired_indices = []
    for f in range(h):
        for t in seg_indices:
            for ch in range(c):
                flat_idx = f * (w * c) + t * c + ch
                desired_indices.append(flat_idx)
    # Convert desired_indices to a TensorFlow constant
    desired_indices = tf.constant(desired_indices, dtype=tf.int32)
    
    W_subset = tf.gather(W, desired_indices, axis=1)  
    b_subset = tf.gather(b, desired_indices)          

    outputs = tf.matmul(hidden_activations, W_subset) + b_subset 
    outputs_reshaped = tf.reshape(outputs, (-1, h, len(seg_indices), c))
    return outputs_reshaped, desired_indices 

def point_reconstruction(spectrogram, window_samps, step_samps, local_indices):
    hann_window = hann(window_samps)
    reconstructed_segments = [
        np.fft.irfft(spectrogram[:, t], n=window_samps) * hann_window
        for t in range(spectrogram.shape[1])
    ]
    constructed_xs = 0 

    for i in range(len(reconstructed_segments)):
        constructed_xs += reconstructed_segments[i][local_indices[i]]

    scaling_factor = np.sum(hann_window**2) / step_samps
    constructed_xs /= scaling_factor

    return constructed_xs


# ─── GPU batched reconstruction ────────────────────────────────────────────
@tf.function
def reconstruct_batch(T_batch: tf.Tensor, E_batch: tf.Tensor) -> tf.Tensor:
    # 1) Temperature → hidden (LeakyReLU)
    #T_batch = T_batch.numpy()
    #E_batch = E_batch.numpy()
    #xs = []
    #for t,e_idx in zip(T_batch, E_batch): 
    #    e_idx = 3000
    #    seg_indices, local_indices = mapSegIdx(e_idx, STEP_SAMPS, WINDOW_SAMPS)
    #    print('---', seg_indices, local_indices)
    #    spectrogram = get_spectrogram_output(model, [t], scaler_T, scaler_spec, seg_indices, h, w, c)
    #    print(spectrogram)
    #    constructed_xs = point_reconstruction(spectrogram, WINDOW_SAMPS, STEP_SAMPS, local_indices)
    #    xs.append(constructed_xs)
    #    print('----', xs)


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

    print("----------", xs)
    return xs

def load_temperature(test_temp, padded_base_e, pad, E_min, E_max, file_path):    
    all_files = glob.glob(os.path.join(file_path, "*.h5"))
    
    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        
        try:
            with h5py.File(file_path, "r") as h5_file:
                print(f"------ Processing: {file_path}")
                
                data = pd.read_hdf(file_path, key="xs_data", compression="gzip")
                
                # Check if the test_temp exists in the dataset
                if test_temp not in data["T"].values:
                    continue
                
                print(f"Processing T = {test_temp} ...")
                subset_T = data[data["T"] == test_temp].copy()
                
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
                signal = interp_func(base_e)

                padded_sig = np.pad(
                    signal, (pad, pad), mode='constant', constant_values=0
                )

                return padded_sig  # Exit function as soon as a valid instance is found
        
        except Exception as e:
            print(f"Error reading HDF5 file {file_path}: {e}")
    
    print("Temperature not found in any files.")
    return None, None

def analyse(pad, padded_base_e, xs, temps, eidxs, E_min, E_max, file_path):
    for T in temps:
        print("Temp", T, "Eidx", eidxs)
        padded_signal = load_temperature(T, padded_base_e, pad, E_min, E_max, file_path)
        print(padded_signal[eidxs], "*********** ehrwehrhe", eidxs)
        filtered_singal = padded_signal[eidxs]


        rel_error = np.abs(xs - filtered_singal) / np.abs(filtered_singal) * 100
        print("relative error", rel_error)

        pivot_idx = eidxs[0]
        print("test IDX", pivot_idx)
        plt.figure(figsize=(8, 5))
        plt.plot(padded_base_e[pad:-pad], padded_signal[pad:-pad], label="Original Signal", lw=2)
        plt.plot(padded_base_e[eidxs], xs, 'ro', markersize=2, label="Reconstructed Point")
        plt.xlabel("padded_Base_E")
        plt.ylabel("padded_sig")
        plt.title("Signal with Reconstructed Point")
        plt.legend()
        plt.grid(True)
        plt.xscale("log")
        plt.yscale("log")
        #plt.show()
        fname = "./data_reconstruct/xs.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(base_e[eidxs], rel_error, marker='o', linestyle='-', color='r', label='Relative Error (%)')
        plt.xlabel("Energy Indices")
        plt.ylabel("Error (%)")
        plt.title("Relative Error vs. E_indices")
        plt.grid(True)
        plt.legend()
        #plt.show()
        fname = "./data_reconstruct/relative_error.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        return None

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--tmin", type=float, default=950.0)
    p.add_argument("--tmax", type=float, default=1050.0)
    args = p.parse_args()
    N = args.batch
    tmin = args.tmin
    tmax = args.tmax

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
    temps = tf.random.uniform([N], tmin, tmax, dtype=tf.float32)
    print("----- TEST TEMPS", temps, len(temps))
    eidxs = tf.random.uniform([N], pad, fs-pad, dtype=tf.int32)
    #eidxs = tf.fill([N], 3000)

    # Warm‐up
    _ = reconstruct_batch(temps[:1], eidxs[:1])

    # Timed run
    t0 = time.perf_counter()
    xs = reconstruct_batch(temps, eidxs)
    dt = (time.perf_counter() - t0) * 1e6 / tf.cast(N, tf.float32)  # µs/point
    print(f"{N:,} points → {dt:.2f} µs/point")

    analyse(pad, padded_base_e, xs, temps, eidxs, E_min, E_max, r'/mnt/d/800_1200')


if __name__ == "__main__":
    main()