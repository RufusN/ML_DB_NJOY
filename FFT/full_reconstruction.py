import h5py
import numpy as np
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
import h5py
import os, glob
import numpy as np
import matplotlib.pyplot as plt


relative_errors_no_scale = dict()
relative_errors_no_scale_trunc = dict()

# ---------------------------
# 1) User-defined parameters
# ---------------------------

#54x1518
# E_min = 1e4 * 1e-6 #MeV
# E_max = 1e5 * 1e-6 #MeV
# window_size = 0.01
# step_size   = 0.00075

#3x170
E_min = 2e4 * 1e-6 #MeV
E_max = 2.1e4 * 1e-6 #MeV
window_size = 0.014559
step_size   = 0.007306

#91x1533
# E_min = 1e4 * 1e-6 #MeV
# E_max = 1e6 * 1e-6 #MeV
# window_size = 0.001988 
# step_size   = 0.000666 

# ---------------------------
# 2) Load base values
# ---------------------------
base_file_path = r"/Volumes/T7 Shield/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
Base_Tval = 200

print(f"\nProcessing T = {Base_Tval} ...")

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
    fs = len(Base_E) #sampling size


window_samps = int(window_size * fs)
step_samps = int(step_size * fs)
hann_window = hann(window_samps)


path = r'../ML_prediction/'
h5_files = list(Path(path).glob("*.h5"))

for file in h5_files:

    temperature = file.name.split('_')[2]
    with h5py.File(file, "r") as h5f:

        time_bins = h5f["time_bins"][:]
        frequencies = h5f["frequencies"][:]
        spectrogram_real = h5f["spectrogram_real"][:]
        spectrogram_imag = h5f["spectrogram_imag"][:]

    spectrogram = spectrogram_real + 1j * spectrogram_imag
    print('Spectrogram shape: ',spectrogram.shape)

    padded_signal_length = 2 * (window_samps) + fs
    reconstructed_signal = np.zeros(int(padded_signal_length), dtype=np.float64)
    reconstructed_segments = [
        np.fft.irfft(spectrogram[:, t], n=window_samps) * hann_window
        for t in range(len(time_bins))
    ]

    for i, segment in enumerate(reconstructed_segments):
        start = (i+1) * step_samps
        reconstructed_signal[start:start + window_samps] += segment

    scaling_factor = np.sum(hann_window**2) / step_samps
    reconstructed_signal /= scaling_factor

    # Remove padding (if applicable)
    start_pad = window_samps
    end_pad = start_pad
    reconstructed_signal = reconstructed_signal[start_pad: -end_pad]


    # ---------------------------
    # 3) Loop over each T value
    # ---------------------------
    T_val = float(temperature)
    print(f"\nProcessing T = {T_val} ...")


    # ---------------------------
    # 4) Read the h5
    # ---------------------------
    file_path = r'/Volumes/T7 Shield/T_800_1200_data/800_1200/'

    all_files = glob.glob(os.path.join(file_path, "*.h5"))

    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        with h5py.File(file_path, "r") as h5_file:
            xs_data = h5_file['xs_data']

            try:
                data = pd.read_hdf(file_path, key="xs_data", compression="gzip")
                if (T_val not in data["T"].unique()):
                    continue

            except Exception as e:
                print(f"Error reading HDF5 file: {e}")
                exit()

        # (a) Filter the DataFrame by the current T
        subset_T = data[data["T"] == T_val].copy()
        print(f" subset_T has {len(subset_T)} rows for T={T_val}.")

        mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
        subset_range = subset_T[mask].copy()

        if len(subset_range) < 2:
            print(f" [Warning] Insufficient data for T={T_val} in range [{E_min}, {E_max}]. Skipping.")

        # Extract the columns as NumPy arrays
        E = subset_range["ERG"].to_numpy()
        xs = subset_range["XS"].to_numpy()
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

        xs = signal
        E = Base_E
        

        relative_error = np.abs(xs - reconstructed_signal) / np.abs(xs)
        relative_errors_no_scale[T_val] = relative_error
        relative_errors_no_scale_trunc[T_val] = relative_error

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # First plot: Cross Section Data vs. Reconstructed Signal
        axs[0].plot(E, xs, label="Cross Section Data", linewidth=1.5)
        axs[0].plot(E, reconstructed_signal, label="Resampled Reconstructed Signal", linestyle="--", linewidth=1.5)
        axs[0].set_xlabel("Energy (eV)")
        axs[0].set_ylabel("xs")
        axs[0].set_title(f"Comparison of Cross Section Data and Reconstructed Signal: T = {T_val}")
        axs[0].legend()
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")

        # Second plot: Relative Error
        axs[1].plot(E, relative_error, label='Relative Error', color='red', alpha=0.8)
        axs[1].set_xlabel("Energy (eV)")
        axs[1].set_ylabel("Relative Error")
        axs[1].set_title(f"Relative Error (T={T_val})\nE in [{E_min:.2f}, {E_max:.2f}]")
        axs[1].legend()
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")

        plt.tight_layout()
        plt.show()
        break

 
print("Plotting Results ...")
target_relative_error = 10**(-3)

# Concatenate all relative errors
all_relative_errors_base_scaled_trunc = np.concatenate(list(relative_errors_no_scale_trunc.values()))
proportion_below_target = np.sum(all_relative_errors_base_scaled_trunc < target_relative_error) / len(all_relative_errors_base_scaled_trunc) * 100

# Define binning for histogram
min_val = max(np.min(all_relative_errors_base_scaled_trunc), 1e-10)  
max_val = np.max(all_relative_errors_base_scaled_trunc)
log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(all_relative_errors_base_scaled_trunc, bins=log_bins, color='skyblue', edgecolor='black', alpha=0.7)
plt.xscale('log')  
plt.xlabel('Relative Error', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Relative Error Distribution', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.axvline(target_relative_error, color='red', linestyle='--', linewidth=1.5, label='Target Relative Error')

# Convert values to scientific notation
mean_error = "{:.1e}".format(np.mean(all_relative_errors_base_scaled_trunc))
max_error = "{:.1e}".format(np.max(all_relative_errors_base_scaled_trunc))
performance_index = "{:.1f} %".format(proportion_below_target)

# Add annotations within the plot boundaries
text_x = min_val * 10  # Adjust position to fit within the left side of the graph
plt.text(text_x, plt.ylim()[1] * 0.9, f"Performance Index: {performance_index}", 
         color='black', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.text(text_x, plt.ylim()[1] * 0.8, f"Mean Relative Error: {mean_error}", 
         color='black', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.text(text_x, plt.ylim()[1] * 0.7, f"Max Relative Error: {max_error}", 
         color='black', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.legend(fontsize=12)
plt.show()

# Scatter plot of relative errors by temperature
temps = list(relative_errors_no_scale_trunc.keys())
errors = [relative_errors_no_scale_trunc[temp] for temp in temps]

x_jittered = []
y_values = []

for i, temp in enumerate(temps):
    x_jittered.extend(np.random.normal(loc=temp, scale=5, size=len(errors[i])))
    y_values.extend(errors[i])

plt.figure(figsize=(10, 6))
plt.scatter(x_jittered, y_values, alpha=0.6, s=10, color="skyblue", edgecolor="black")

plt.yscale('log')  
plt.xlabel('Temperature (K)', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.title('Relative Error Distribution Across Temperatures', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(temps)  
plt.axhline(target_relative_error, color='red', linestyle='--', linewidth=1.5, label='Target Relative Error')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Box plot
temps = sorted(list(relative_errors_no_scale_trunc.keys()))
errors = [relative_errors_no_scale_trunc[temp] for temp in temps]

plt.figure(figsize=(10, 6))
plt.boxplot(errors, patch_artist=False, labels=temps, showmeans=False, showfliers=False,
            whis=[0, 100],  
            boxprops=dict(color='black'),
            medianprops=dict(color='green', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5))

plt.yscale('log')  
plt.xlabel('Temperature (K)', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.title('Box Plot of Relative Error Distributions Across Test Temperatures', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.axhline(target_relative_error, color='red', linestyle='--', linewidth=1.5, label='Target Relative Error')
plt.legend(fontsize=12)
plt.show()
