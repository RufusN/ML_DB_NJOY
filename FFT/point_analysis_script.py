import os
import glob
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model
import scipy.ndimage as ndimage
from pathlib import Path

import re
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.saving import register_keras_serializable

def read_spectrogram_h5(h5_filepath):
    with h5py.File(h5_filepath, 'r') as h5f:
        required_datasets = {'time_bins', 'frequencies', 'spectrogram_real', 'spectrogram_imag'}
        missing = required_datasets - set(h5f.keys())
        if missing:
            raise KeyError(f"Missing datasets in HDF5 file: {missing}")
        time_bins = h5f['time_bins'][:]
        frequencies = h5f['frequencies'][:]
        spectrogram_real = h5f['spectrogram_real'][:]
        spectrogram_imag = h5f['spectrogram_imag'][:]
    spectrogram_complex = spectrogram_real + 1j * spectrogram_imag
    print("Spectrogram shape:", np.shape(spectrogram_complex))
    return time_bins, frequencies, spectrogram_complex

def load_base(E_min, E_max):
    base_file_path = r'/Volumes/T7 Shield/Base_E/capture_xs_data_0.h5'
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
    return Base_E, Base_xs, fs

def load_temperature(test_temp, Base_E, pad, E_min, E_max, file_path):    
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
                signal = interp_func(Base_E)
                
                padded_sig = np.pad(
                    signal, (pad, pad), mode='constant', constant_values=0
                )
                
                padded_Base_E = np.pad(
                    Base_E, (pad, pad), mode='constant', constant_values=0
                )
                
                return padded_Base_E, padded_sig  # Exit function as soon as a valid instance is found
        
        except Exception as e:
            print(f"Error reading HDF5 file {file_path}: {e}")
    
    print("Temperature not found in any files.")
    return None, None

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

def get_spectrogram_output(model, input_data, scaler_T, scaler_spec, seg_indices):
    T = input_data[0]

    w = 170 # spectrogram bins
    h = 3   # frequencies
    c = 2   # channels (real/imagninary)
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
    
    print("Input data shape:", input_data.shape)
    hidden_activations = submodel(input_data)  # Expected shape: (batch_size, 16)
    print("Hidden activations shape:", hidden_activations.shape)
    
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

def analyse(pad, padded_Base_E, padded_sig, E_indices, results):

    E_indices = np.array(E_indices)
    rel_error = np.abs(results - padded_sig[E_indices]) / np.abs(padded_sig[E_indices]) * 100
    print('Relative_Errors:', rel_error)
    print('Relative Max:', np.max(rel_error))
    print('Relative Mean:', np.mean(rel_error))

    plt.figure(figsize=(8, 5))
    plt.plot(padded_Base_E[pad:-pad], padded_sig[pad:-pad], label="Original Signal", lw=2)
    plt.plot((padded_Base_E)[E_indices], results, 'ro', markersize=2, label="Reconstructed Point")
    plt.xlabel("padded_Base_E")
    plt.ylabel("padded_sig")
    plt.title("Signal with Reconstructed Point")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(E_indices, rel_error, marker='o', linestyle='-', color='r', label='Relative Error (%)')
    plt.xlabel("Energy Indices")
    plt.ylabel("Error (%)")
    plt.title("Relative Error vs. E_indices")
    plt.grid(True)
    plt.legend()
    plt.show()


        

    return None

def log_ae_loss(y_true, y_pred, epsilon=1e-16):
    return tf.reduce_mean(tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon))

def main():

    with h5py.File("../MLP/spec_scalers.h5", "r") as hf:
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
    
    # E_min = 1e4 * 1e-6 #MeV
    # E_max = 1e5 * 1e-6 #MeV
    # window_size = 0.01
    # step_size   = 0.00075

    E_min = 2e4 * 1e-6 #MeV
    E_max = 2.1e4 * 1e-6 #MeV
    window_size = 0.014559
    step_size   = 0.007306

    test_temp = 1000.0
    input_data = [test_temp]

    Base_E, Base_xs, fs = load_base(E_min, E_max)
    window_samps = int(window_size * fs)
    step_samps   = int(step_size   * fs)
    pad          = window_samps  

    padded_Base_E, padded_sig = load_temperature(test_temp, Base_E, pad, E_min, E_max, file_path=r'/Volumes/T7 Shield/T_800_1200_data/800_1200')

    E_indices = [i for i in range(4,338,1)]
    results = []
    for E_idx in E_indices:
        seg_indices, local_indices = mapSegIdx(E_idx, step_samps, window_samps)

        model = load_model('../MLP/best_model_real_imag.keras', compile = False)

        # compare_full_vs_partial(model,input_data,scaler_T,scaler_spec,seg_indices)

        spectrogram = get_spectrogram_output(model, input_data, scaler_T, scaler_spec, seg_indices)
        constructed_xs = point_reconstruction(spectrogram, window_samps, step_samps, local_indices)
        results.append(constructed_xs)
        E_val = padded_Base_E[E_idx]  

    analyse(pad, padded_Base_E, padded_sig, E_indices, results)


if __name__ == "__main__":
    main()
