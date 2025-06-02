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
E_min = 5 * 1e-6# MeV (10 keV)
E_max = 1e4 * 1e-6  # MeV (1 MeV)
Analysis = True  # Set True to enable plotting
only_integer = False  # Only process integer temperatures

# ---------------------------
# Helper: load PyTables-style HDF5 into DataFrame
# ---------------------------
def load_h5_dataframe(file_path, group_name="data"):
    with h5py.File(file_path, "r") as f:
        grp = f[group_name]
        # Read block0_items (column names)
        items = [name.decode('utf-8') for name in np.array(grp['block0_items'], dtype='S')]
        values = np.array(grp['block0_values'])  # shape (N, len(items))
        df = pd.DataFrame(values, columns=items)
        # Convert ERG from eV to MeV if present
        if 'ERG' in df.columns:
            df['ERG'] = df['ERG'] * 1e-6
        return df

# ---------------------------
# 2) Read the baseline HDF5 for T = 300 K
# ---------------------------
base_file_path = r'/Volumes/T7 Shield/NJOY/U238_data/U238_300.0.h5'
Base_Tval = 300.0

try:
    data_base = load_h5_dataframe(base_file_path)
except Exception as e:
    print(f"Error reading baseline HDF5 file: {e}")
    exit(1)

print(f"\nProcessing baseline T = {Base_Tval} from {base_file_path}...")
# Filter by temperature if column exists
if "T" in data_base.columns:
    subset_T = data_base[data_base["T"] == Base_Tval].copy()
else:
    subset_T = data_base.copy()
# Apply energy range mask on ERG (MeV)
mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
subset_range = subset_T[mask].copy()

# Extract and sort baseline grid using original energy points
Base_E = subset_range["ERG"].to_numpy()
Base_xs = subset_range["XS"].to_numpy()
sort_idx = np.argsort(Base_E)
Base_E = Base_E[sort_idx]
Base_xs = Base_xs[sort_idx]
fs = len(Base_E)
print(f"Length of baseline energy grid: {fs}")

# ---------------------------
# 3) Loop over new single-temperature HDF5 files (integers only)
# ---------------------------
file_dir = r'/Volumes/T7 Shield/NJOY/U238_data'
all_files = glob.glob(os.path.join(file_dir, 'U238_*.h5'))

for file_path in all_files:
    file_path = os.path.normpath(file_path)
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    try:
        T_val = float(name.split('_')[1])
    except Exception:
        print(f"Skipping file with unexpected name: {base_name}")
        continue

    # Only consider integer temperatures
    if only_integer:
        if not T_val.is_integer():
            T_val = int(T_val)  
            continue
        
    # Skip the baseline file
    if T_val == int(Base_Tval):
        continue

    print(f"\nProcessing integer T = {T_val} from {file_path}...")
    try:
        data = load_h5_dataframe(file_path)
    except Exception as e:
        print(f"Error reading HDF5 file {file_path}: {e}")
        continue

    # Filter by temperature if present
    if "T" in data.columns:
        subset = data[data["T"] == float(T_val)].copy()
    else:
        subset = data.copy()
    # Apply energy mask on ERG (MeV)
    mask = (subset["ERG"] >= E_min) & (subset["ERG"] <= E_max)
    subset_range = subset[mask].copy()
    if len(subset_range) < 2:
        print(f" [Warning] Insufficient data ({len(subset_range)}) for T={T_val} in range [{E_min}, {E_max}]. Skipping.")
        continue

    E = subset_range["ERG"].to_numpy()
    xs = subset_range["XS"].to_numpy()
    sort_idx = np.argsort(E)
    E = E[sort_idx]
    xs = xs[sort_idx]

    # Interpolate to baseline grid
    interp_func = interp1d(E, xs, kind="cubic", fill_value="extrapolate")
    signal = interp_func(Base_E)

    # ---------------------------
    # 4) Prepare Sliding Window FFT (Spectrogram)
    # ---------------------------
    window_samps = 4  # small window size
    step_samps = 2    # small step size
    pad = window_samps
    padded_sig = np.pad(signal, (pad, pad), mode='constant', constant_values=0)
    frequencies = np.fft.rfftfreq(window_samps, d=1/fs)
    hann_window = hann(window_samps)

    starts = range(0, len(padded_sig) - window_samps, step_samps)
    windowed = [padded_sig[s: s + window_samps] * hann_window for s in starts]
    spectrogram = np.array([np.fft.rfft(w) for w in windowed]).T
    spectrogram = spectrogram[:, 1:]  # trim DC component
    time_bins = np.arange(spectrogram.shape[1])

    print(f"Spectrogram shape (freq_bins x time_bins): {spectrogram.shape}")

    # Save Spectrogram to HDF5
    h5_filename = f'/Volumes/T7 Shield/NJOY/spectrograms/uranium/full_range/3x49681/spectrogram_T_{T_val}.h5'
    os.makedirs(os.path.dirname(h5_filename), exist_ok=True)
    with h5py.File(h5_filename, "w") as h5f:
        h5f.create_dataset("time_bins", data=time_bins)
        h5f.create_dataset("frequencies", data=frequencies)
        h5f.create_dataset("spectrogram_real", data=spectrogram.real)
        h5f.create_dataset("spectrogram_imag", data=spectrogram.imag)

    print(f"Spectrogram saved: {h5_filename} with shape: {spectrogram.shape}")

    # ---------------------------
    # 5) Optional Analysis and plotting
    # ---------------------------
    if Analysis:
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(time_bins, frequencies, 10 * np.log10(np.abs(spectrogram) + 1e-12), shading='auto', cmap='viridis')
        plt.colorbar(label='Power (dB)')
        plt.title(f"Spectrogram (T={T_val})")
        plt.xlabel('Window index')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

        reconstructed_segments = [
            np.fft.irfft(spectrogram[:, t], n=window_samps) * hann_window
            for t in range(spectrogram.shape[1])
        ]
        reconstructed_signal = np.zeros_like(padded_sig, dtype=np.float64)
        for i, segment in enumerate(reconstructed_segments):
            start = (i + 1) * step_samps
            reconstructed_signal[start:start + window_samps] += segment
        scaling_factor = np.sum(hann_window**2) / step_samps
        reconstructed_signal = reconstructed_signal / scaling_factor
        reconstructed_signal = reconstructed_signal[pad:-pad]

        plt.figure(figsize=(12, 6))
        plt.plot(Base_E, signal, label='Original')
        plt.plot(Base_E, reconstructed_signal, '--', label='Reconstructed')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Original vs Reconstructed (T={T_val})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        rel_error = np.abs(signal - reconstructed_signal) / np.where(np.abs(signal) < 1e-12, 1e-12, np.abs(signal))
        plt.figure(figsize=(12, 6))
        plt.plot(Base_E, rel_error, label='Relative Error')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Relative Error (T={T_val})")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("\nAll T values processed!")