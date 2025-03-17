import os
import glob
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
import scipy.ndimage as ndimage

# ---------------------------
# 1) User-defined parameters
# ---------------------------

#54x1518 - old. 
# E_min = 1e4 * 1e-6 #MeV
# E_max = 1e5 * 1e-6 #MeV
# window_size = 0.01
# step_size   = 0.00075

#3x170
E_min = 2e4 * 1e-6 #MeV
E_max = 2.1e4 * 1e-6 #MeV
window_size = 0.014559
step_size   = 0.007306

# E_min = 1e4 * 1e-6 #MeV
# E_max = 1e6 * 1e-6 #MeV

#options - 91x1521
# window_size = 0.001988 
# step_size   = 0.000666

# window_size = 0.016412
# step_size = 0.005477

# window_size = 0.010619 
# step_size   =  0.003538

Analysis = False

# ---------------------------
# 2) Read the baseline h5
# ---------------------------

base_file_path = r"/Volumes/T7 Shield/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
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



file_path = r'/Volumes/T7 Shield/T_800_1200_data/800_1200'

all_files = glob.glob(os.path.join(file_path, "*.h5"))
combined_df = pd.DataFrame()

for file_path in all_files:
    file_path = os.path.normpath(file_path)
    with h5py.File(file_path, "r") as h5_file:
        xs_data = h5_file['xs_data']

        try:
            data = pd.read_hdf(file_path, key="xs_data", compression="gzip")
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
            exit()

    unique_T_values = data["T"].unique()

    for T_val in unique_T_values:
        if (T_val == Base_Tval):
            continue
        print(f"\nProcessing T = {T_val} ...")

        # (a) Filter the DataFrame by the current T
        subset_T = data[data["T"] == T_val].copy()

        mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
        subset_range = subset_T[mask].copy()

        print(f"Currently processing T={T_val}...")

        subset_T = data[data["T"] == T_val].copy()
        print(f" subset_T has {len(subset_T)} rows for T={T_val}.")

        if len(subset_range) < 2:
            print(f" [Warning] Insufficient data (length " + str(len(subset_range)) + ") in range [ " + str(E_min) + ", " + str(E_max) + "]. Skipping.")
            continue

        # Extract the columns as NumPy arrays
        E = subset_range["ERG"].to_numpy()
        xs = subset_range["XS"].to_numpy()

        # Sort by ascending energy if not already
        sort_idx = np.argsort(E)
        E  = E[sort_idx]
        xs = xs[sort_idx]

        # Linear interpolation
        interp_func = interp1d(
            E,
            xs,
            kind="cubic",
            fill_value="extrapolate"
        )
        signal = interp_func(Base_E)

        # ---------------------------
        # 3) Prepare Sliding Window FFT (Spectrogram)
        # ---------------------------
        window_samps = int(window_size * fs)
        step_samps   = int(step_size   * fs)

        pad        = window_samps
        padded_sig = np.pad(
            signal,
            (pad, pad),
            mode='constant',
            constant_values=0
        )

        padded_Base_E = np.pad(
                    Base_E,
                    (pad, pad),
                    mode='constant',
                    constant_values=0
        )
        frequencies = np.fft.rfftfreq(window_samps, d=1/fs)
        hann_window = hann(window_samps)

        starts      = range(0, len(padded_sig) - window_samps, step_samps) 
        windowed    = [padded_sig[s : s + window_samps] * hann_window for s in starts] 
        spectrogram = np.array([np.fft.rfft(w) for w in windowed])  
        spectrogram = spectrogram.T                             

        spectrogram = spectrogram[:,1:]   
        time_bins = np.array([i for i in range(spectrogram.shape[1])]) 

        #Optional Smoothing 
        # sigma = 0.2
        # spectrogram = ndimage.gaussian_filter(spectrogram, sigma=sigma)

        # ---------------------------
        # 4) Save Spectrogram to HDF5
        # ---------------------------

        h5_filename = f'/Volumes/T7 Shield/T_full_data/3x170_spectrograms/spectrogram_T_{T_val}.h5' 
        with h5py.File(h5_filename, "w") as h5f:
            h5f.create_dataset("time_bins",      data=time_bins)
            h5f.create_dataset("frequencies",    data=frequencies)
            h5f.create_dataset("spectrogram_real", data=spectrogram.real)
            h5f.create_dataset("spectrogram_imag", data=spectrogram.imag)

        print(f"Spectrogram saved: {h5_filename} with shape: {np.shape(spectrogram)}")

        if Analysis:

            # ---------------------------
            # 5) Plot the Spectrogram
            # ---------------------------
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins,
                frequencies,
                10 * np.log10(np.abs(spectrogram) + 1e-12),
                shading='auto',
                cmap='viridis'
            )
            plt.colorbar(label='Power (dB)')
            plt.title(f"Spectrogram (T={T_val})\nE in [{E_min}, {E_max}]")
            plt.xlabel('Energy (eV) [Log Scale]')
            plt.ylabel('Frequency (Hz)')
            # plt.xscale("log")
            plt.tight_layout()
            spectrogram_plot = f"spectrogram_T_{T_val}.png"
            #plt.savefig(spectrogram_plot, dpi=150)
            plt.show()
            plt.close()
            print(f"  [Info] Spectrogram plot saved: {spectrogram_plot}")

            # ---------------------------
            # 6) Reconstruct the Signal (Inverse Windowed-FFT)
            # ---------------------------
            reconstructed_segments = [
                np.fft.irfft(spectrogram[:, t], n=window_samps) * hann_window
                for t in range(len(time_bins))
            ]
            reconstructed_signal = np.zeros_like(padded_sig, dtype=np.float64)
            overlap_factor       = np.zeros_like(padded_sig, dtype=np.float64)
            for i, segment in enumerate(reconstructed_segments):
                start = (i+1) * step_samps
                reconstructed_signal[start:start + window_samps] += segment
            # Correct for Hann window overlap scaling
            # Scale by the sum of squared Hann window values
            scaling_factor = np.sum(hann_window**2) / step_samps
            reconstructed_signal /= scaling_factor
            # Remove padding (if applicable)
            start_pad = window_samps 
            end_pad = start_pad
            reconstructed_signal = reconstructed_signal[start_pad: -end_pad]
            original_energy = np.sum(signal**2)
            reconstructed_energy = np.sum(reconstructed_signal**2)
            print(f"Energy ratio (reconstructed/original): {reconstructed_energy / original_energy}")

            # ---------------------------
            # 11) Plot Original vs Reconstructed
            # ---------------------------
            plt.figure(figsize=(12, 6))
            plt.plot(Base_E, signal, label='Original Signal', alpha=0.8)
            plt.plot(Base_E, reconstructed_signal, label='Reconstructed Signal', linestyle='--', alpha=0.8)
            plt.legend()
            plt.xlabel('Energy (eV)')
            plt.ylabel('Cross Section')
            plt.title(f"Original vs Reconstructed (T={T_val})\nE in [{E_min}, {E_max}]")
            plt.xscale("log")
            plt.yscale("log")
            plt.tight_layout()
            orig_recon_plot = f"orig_vs_recon_T_{T_val}.png"
            #plt.savefig(orig_recon_plot, dpi=150)
            plt.show()
            plt.close()
            print(f"  [Info] Original vs Reconstructed plot saved: {orig_recon_plot}")
            # # # ---------------------------
            # # # 12) Calculate & Plot Relative Errors
            # # # ---------------------------
            signal_uniform_safe = np.where(np.abs(signal) < 1e-12, 1e-12, signal)
            relative_error = np.abs(signal - reconstructed_signal) / np.abs(signal_uniform_safe)
            plt.figure(figsize=(12, 6))
            plt.plot(Base_E, relative_error, label='Relative Error', color='red', alpha=0.8)
            plt.legend()
            plt.xlabel('Energy (eV)')
            plt.ylabel('Relative Error')
            plt.title(f"Relative Error (T={T_val})\nE in [{E_min}, {E_max}]")
            plt.xscale("log")
            plt.yscale("log")
            plt.tight_layout()
            error_plot = f"relative_error_T_{T_val}.png"
            #plt.savefig(error_plot, dpi=150)
            plt.show()
            plt.close()
            print(f"  [Info] Relative error plot saved: {error_plot}")

print("\nAll T values processed!")
