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
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.saving import register_keras_serializable


def predict_subset(model, input_data, seg_indices, h, w, c):
    # In our model built by build_real_imag_model, the layers are:
    #   0: InputLayer
    #   1: Dense(16) layer
    #   2: LeakyReLU activation (latent representation, shape=(batch, 16))
    #   3: Dense(final_size) layer (maps latent (16) to 163944 outputs)
    #   4: Reshape to (54, 1518, 2)
    #
    # We want to extract the latent representation from the LeakyReLU activation.
    latent_layer = model.layers[2]
    submodel = tf.keras.Model(inputs=model.input, outputs=latent_layer.output)
    
    print("Input data shape:", input_data.shape)
    hidden_activations = submodel(input_data)  # Expected shape: (batch_size, 16)
    print("Hidden activations shape:", hidden_activations.shape)
    
    # Get the Dense layer that maps the 16 latent nodes to the flattened spectrogram.
    output_dense_layer = model.layers[3]
    W = output_dense_layer.kernel  # Shape: (16, 163944)
    b = output_dense_layer.bias    # Shape: (163944,)
    
    # Compute the flattened indices corresponding to the two desired time bins.
    # For each frequency f in [0, h), for each channel ch in [0, c),
    # the flattened index for a given time t is: f * (w*c) + t*2 + ch.
    desired_indices = []
    for f in range(h):
        for t in seg_indices:
            for ch in range(c):
                flat_idx = f * (w * c) + t * c + ch
                desired_indices.append(flat_idx)
    # Convert desired_indices to a TensorFlow constant
    desired_indices = tf.constant(desired_indices, dtype=tf.int32)
    
    # Gather the corresponding columns from the dense layer weights and bias.
    W_subset = tf.gather(W, desired_indices, axis=1)  # New shape: (16, number_of_selected_neurons)
    b_subset = tf.gather(b, desired_indices)          # New shape: (number_of_selected_neurons,)
    
    # Compute the predicted output for the subset:
    outputs = tf.matmul(hidden_activations, W_subset) + b_subset  # Shape: (batch_size, number_of_selected_neurons)
    # The number_of_selected_neurons should be: h * 2 (time bins) * c = 54 * 2 * 2 = 216.
    
    # Reshape the output so that it has the correct spatial arrangement.
    # We want the extracted slice to be of shape (h, 2, c) per sample.
    outputs_reshaped = tf.reshape(outputs, (-1, h, len(seg_indices), c))
    return outputs_reshaped  # Shape: (batch_size, 54, 2, 2)

# def get_prescaled_spectrogram_output(model, input_data, seg_indices, h, w, c):
#     """
#     Returns the raw (scaled) output from the model corresponding to the selected segments.
    
#     Parameters:
#       model: the trained model.
#       input_data: list or array containing the temperature value.
#       seg_indices: list of time bin indices you want to extract.
#       h: number of frequency bins.
#       w: total number of spectrogram time bins.
#       c: number of channels (real, imaginary).
    
#     Returns:
#       A numpy array of shape (h, len(seg_indices), c) containing the model's output
#       before applying the inverse scaling.
#     """
#     # Get the prediction in the scaled space (output of predict_subset)
#     pred_scaled = predict_subset(model, input_data, seg_indices, h, w, c)
#     # Convert tensor to numpy and squeeze extra dimensions if needed.
#     prescaled_output = pred_scaled.numpy().squeeze()
#     return prescaled_output



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
    # Combine real and imaginary parts into a complex array
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
        fs = sampling_size = len(Base_E)
    return Base_E, Base_xs, fs

def load_temperature(test_temp, Base_E, pad, E_min, E_max, file_path=r'D:/800_1200/'):    
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
    seg_idx = int(np.round((energy_idx - window_samps)/step_samps -1 + 0.5))
    for _ in range(n_overlaps):
        seg_idxs.append(seg_idx)
        seg_idx += 1
        start = seg_idx*step_samps
        local_idxs.append(energy_idx-start)
    return seg_idxs, local_idxs

def get_spectrogram_output(model, input_data, scaler_T, scaler_spec, seg_indices):
    T = input_data[0]

    w = 170 # spectrogram bins
    h = 3   # frequencies
    c = 2   # channels (real/imagninary)
    slice_length = len(seg_indices)

    T_norm = scaler_T.transform([[T]])
    input_data = T_norm

    desired_indices = seg_indices
    pred_scaled = predict_subset(model, input_data, desired_indices, h, w, c)
    pred_scaled = np.array(pred_scaled)

    # For example, if you want to see the raw model output for seg_indices you choose:
    # prescaled_spec = get_prescaled_spectrogram_output(model, input_data, seg_indices, h=3, w=170, c=2)
    # print("Prescaled spectrogram shape:", prescaled_spec.shape)
    # print("Prescaled spectrogram values (first few):", prescaled_spec[:3, 83, :])



    pred_scaled = pred_scaled.squeeze() 
    pred_scaled_flat = pred_scaled.reshape(1, -1)

    # print(pred_scaled_flat)
    test = [ 83,84]
    desired_indices = []
    for f in range(h):
        for t in test:
            for ch in range(c):
                # Flattening in C order: index = f*(w*c) + t*c + ch
                desired_indices.append(f * (w * c) + t * c + ch)
    desired_indices = np.array(desired_indices) 

    pred_unscaled_flat = pred_scaled_flat * scaler_spec.scale_[desired_indices] + scaler_spec.mean_[desired_indices]

    print(scaler_spec.scale_)
    print(scaler_spec.mean_)

    # pred_unscaled_flat = scaler_spec.inverse_transform(arr)
    pred_unscaled = pred_unscaled_flat.reshape(h, slice_length, c)  

    spectrogram_real = pred_unscaled[..., 0]
    spectrogram_imag = pred_unscaled[..., 1]

    spectrogram = spectrogram_real + 1j * spectrogram_imag 
    return spectrogram

def point_reconstruction(spectrogram, window_samps, local_indices):
    hann_window = hann(window_samps)
    reconstructed_segments = [
        np.fft.irfft(spectrogram[:, t], n=window_samps) * hann_window
        for t in range(spectrogram.shape[1])
    ]
    constructed_xs = 0 

    for i in range(len(reconstructed_segments)):
        constructed_xs += reconstructed_segments[i][local_indices[i]]

    scaling_factor = 0.5625000000000002
    constructed_xs /= scaling_factor

    return constructed_xs

def analyse(pad, padded_Base_E, padded_sig, reconstructed_point, index):
    plt.figure(figsize=(8, 5))
    print(padded_sig[index], reconstructed_point)
    plt.plot(padded_Base_E, padded_sig, label="Original Signal", lw=2)
    plt.plot(padded_Base_E[index], reconstructed_point, 'ro', markersize=10, label="Reconstructed Point")
    plt.xlabel("padded_Base_E")
    plt.ylabel("padded_sig")
    plt.title("Signal with Reconstructed Point")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
    rel_error = np.abs(reconstructed_point - padded_sig[index]) / np.abs(padded_sig[index]) * 100
    print(rel_error)
    return None

def log_ae_loss(y_true, y_pred, epsilon=1e-16):
    return tf.reduce_mean(tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon))

def main():

    scaler_spec = StandardScaler()
    scaler_T = StandardScaler()

    with open("/Users/ru/FFT/Resonance/MLP_scripts/MLP_non_uniform/spectrogram_stats.txt", "r") as f:
    # with open("/Users/ru/FFT/Resonance/MLP_scripts/MLP_non_uniform/scaler_stats.txt", "r") as f:
        lines = f.read().splitlines()

    t_parts = lines[1].split(":")[1].split(",")
    mean_t = float(t_parts[0].split("=")[1])
    std_t = float(t_parts[1].split("=")[1])
    scaler_T.mean_ = np.array([mean_t])
    scaler_T.scale_ = np.array([std_t])
    scaler_T.var_ = scaler_T.scale_ ** 2
    scaler_T.n_features_in_ = 1

    # Read the entire file and filter out empty lines.
    with open("/Users/ru/FFT/Resonance/MLP_scripts/MLP_non_uniform/scaler_stats.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    # Find the indices of the labels.
    scale_label = "Scaler Spec Scale:"
    mean_label = "Scaler Spec Mean:"

    try:
        scale_idx = lines.index(scale_label)
        mean_idx = lines.index(mean_label)
    except ValueError as e:
        print("Error: Could not find the required labels in the file.")
        raise e

    # Extract lines that belong to each section.
    scale_lines = lines[scale_idx + 1 : mean_idx]
    mean_lines = lines[mean_idx + 1 :]

    # Join the lines into one string for each array.
    scale_str = " ".join(scale_lines)
    mean_str = " ".join(mean_lines)

    # Remove any surrounding square brackets.
    if scale_str.startswith("[") and scale_str.endswith("]"):
        scale_str = scale_str[1:-1]
    if mean_str.startswith("[") and mean_str.endswith("]"):
        mean_str = mean_str[1:-1]

    # Convert the strings to NumPy arrays.
    scaler_spec_scale_array = np.fromstring(scale_str, sep=",")
    scaler_spec_mean_array = np.fromstring(mean_str, sep=",")

    # Assign the arrays to your scaler attributes.
    scaler_spec.mean_ = scaler_spec_mean_array
    scaler_spec.scale_ = scaler_spec_scale_array
    scaler_spec.var_ = scaler_spec.scale_ ** 2
    scaler_spec.n_features_in_ = scaler_spec_scale_array.size

    # (If you have similar T values information in the file, you can parse it in a similar way.)
    print("Scaler Spec Scale:", scaler_spec.scale_)
    print("Scaler Spec Mean:", scaler_spec.mean_)


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
    pad        = window_samps  

    padded_Base_E, padded_sig = load_temperature(test_temp, Base_E, pad, E_min, E_max, file_path=r'/Volumes/T7 Shield/T_800_1200_data/800_1200')

    E_idx = 171
    seg_indices, local_indices = mapSegIdx(E_idx, step_samps, window_samps)

    # seg_indices = [i for i in range(172)]

    model = load_model('../../MLP_scripts/MLP_non_uniform/best_model_real_imag.keras', compile = False)

    spectrogram = get_spectrogram_output(model, input_data, scaler_T, scaler_spec, seg_indices)
    constructed_xs = point_reconstruction(spectrogram, window_samps, local_indices)

    E_val = padded_Base_E[E_idx] #choose something 
    # xs_val = padded_sig[E_idx]
    # print(spectrogram)
    # print(np.shape(spectrogram))
    # print(spectrogram[:3,83])    

    analyse(pad, padded_Base_E, padded_sig, constructed_xs, E_idx)


if __name__ == "__main__":
    main()


        # ml_spec = spectrogram


    # temperature = test_temp
    # path_temp = f'/Volumes/T7 Shield/T_800_1200_data/spectrogram_T_{temperature}.h5'
    # with h5py.File(path_temp, "r") as h5f:
    #     time_bins_gt = h5f["time_bins"][:]
    #     frequencies_gt = h5f["frequencies"][:]
    #     spectrogram_real_gt = h5f["spectrogram_real"][:]
    #     spectrogram_imag_gt = h5f["spectrogram_imag"][:]

    #     gt_spec = spectrogram_real_gt + 1j*spectrogram_imag_gt
    #     gt_spec = np.abs(gt_spec)
    #     print("wet", np.shape(gt_spec))
    #     plt.figure(figsize=(12, 6))
    #     plt.pcolormesh(
    #         time_bins_gt,
    #         frequencies_gt,
    #         10 * np.log10(np.abs(gt_spec) + 1e-12),
    #         shading='auto',
    #         cmap='viridis'
    #     )
    #     plt.colorbar(label='Power (dB)')
    #     plt.title(f"Spectrogram")
    #     plt.xlabel('Energy (eV) [Log Scale]')
    #     plt.ylabel('Frequency (Hz)')
    #     # plt.xscale("log")
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
        # 3. Select only the 50th and 51st slices
        # --------------------------
        # Here we assume that the first axis corresponds to the frequency slices.
        # slice_indices = slice(local_idxs[0], local_idxs[-1])

        # gt_slice = gt_spec[:, slice_indices]
        # ml_slice = ml_spec[:, slice_indices]

        # # --------------------------
        # # 4. Compute the relative error
        # # --------------------------
        # # Use a small epsilon to avoid division by zero if necessary.
        # eps = 1e-16


        # relative_error = np.abs(ml_slice - gt_slice) / (np.abs(gt_slice) + eps)
        # print("errors",relative_error, np.shape(relative_error))
        # relative_error = np.log(relative_error)

        # for i in range(len(gt_spec)):
        #     print(gt_slice[i,0],ml_slice[i,0])

        # # --------------------------
        # # 5. Plotting the results
        # # --------------------------
        # fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # # Plot ML spectrogram slice
        # im0 = axs[0].imshow(ml_slice, aspect='auto', origin='lower')
        # axs[0].set_title(f"ML Spectrogram (Slices {bin})")
        # axs[0].set_xlabel("Time")
        # axs[0].set_ylabel("Frequency")
        # fig.colorbar(im0, ax=axs[0])

        # # Plot ground truth spectrogram slice
        # im1 = axs[1].imshow(gt_slice, aspect='auto', origin='lower')
        # axs[1].set_title(f"Ground Truth Spectrogram (Slices {bin})")
        # axs[1].set_xlabel("Time")
        # axs[1].set_ylabel("Frequency")
        # fig.colorbar(im1, ax=axs[1])

        # # Plot relative error
        # im2 = axs[2].imshow(relative_error, aspect='auto', origin='lower')
        # axs[2].set_title("Relative Error (|ML - GT| / |GT|)")
        # axs[2].set_xlabel("Time")
        # axs[2].set_ylabel("Frequency")
        # fig.colorbar(im2, ax=axs[2])

        # plt.tight_layout()
        # plt.show()
    ###
