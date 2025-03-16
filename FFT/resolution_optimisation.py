import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from scipy.optimize import differential_evolution

# -------------------------------------------
# Settings and data loading
# -------------------------------------------
# Energy range (in MeV)
# E_min = 2e4 * 1e-6  # MeV
# E_max = 2.1e4 * 1e-6  # MeV

E_min = 1e4 * 1e-6  # MeV
E_max = 1e6 * 1e-6  # MeV

# File and base temperature for reference
base_file_path = r"/Volumes/T7 Shield/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
Base_Tval = 200.0

# Load base data for reference grid
with h5py.File(base_file_path, "r") as h5_file:
    try:
        data = pd.read_hdf(base_file_path, key="xs_data", compression="gzip")
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        exit()

# Filter data for the base temperature
subset_T = data[data["T"] == Base_Tval].copy()
mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
subset_range = subset_T[mask].copy()

# Extract and sort the base energy grid and xs values
Base_E = subset_range["ERG"].to_numpy()
Base_xs = subset_range["XS"].to_numpy()
sort_idx = np.argsort(Base_E)
Base_E = Base_E[sort_idx]
Base_xs = Base_xs[sort_idx]

# Use the number of energy samples as a surrogate for the sampling frequency
fs = len(Base_E)

# -------------------------------------------
# Select a test temperature and prepare the signal
# -------------------------------------------
# For testing, use a different temperature (here: 205.0)
test_T = 205.0
subset_T_test = data[data["T"] == test_T].copy()
mask_test = (subset_T_test["ERG"] >= E_min) & (subset_T_test["ERG"] <= E_max)
subset_range_test = subset_T_test[mask_test].copy()

if len(subset_range_test) < 2:
    print(f"Not enough data for T = {test_T}")
    exit()

E_test = subset_range_test["ERG"].to_numpy()
xs_test = subset_range_test["XS"].to_numpy()
sort_idx = np.argsort(E_test)
E_test = E_test[sort_idx]
xs_test = xs_test[sort_idx]

# Interpolate the test data onto the base energy grid to create the signal
interp_func = interp1d(E_test, xs_test, kind="cubic", fill_value="extrapolate")
signal = interp_func(Base_E)

# -------------------------------------------
# Function: Compute error and resolution metric
# -------------------------------------------
def compute_error_and_resolution(window_size, step_size, signal, fs):
    """
    For a given window_size and step_size (in the same units as energy scale fraction),
    compute the sliding-window FFT spectrogram, perform an inverse FFT reconstruction,
    and return the maximum relative error (over central indices) and the spectrogram
    resolution defined as (freq_bins x time_bins).
    """
    # Convert to sample counts
    window_samps = int(window_size * fs)
    step_samps   = int(step_size * fs)
    if window_samps < 2 or step_samps < 1:
        return np.inf, None  # invalid configuration

    # Pad the signal symmetrically
    pad = window_samps // 2
    padded_sig = np.pad(signal, (pad, pad), mode='constant', constant_values=0)
    
    # Create Hann window and prepare sliding windows
    hann_window = hann(window_samps)
    starts = range(0, len(padded_sig) - window_samps + 1, step_samps)
    windowed = [padded_sig[s : s + window_samps] * hann_window for s in starts]
    if len(windowed) == 0:
        return np.inf, None

    # Compute spectrogram (each column is the FFT of a window)
    spectrogram = np.array([np.fft.rfft(w) for w in windowed]).T  # shape: (freq_bins, time_bins)
    n_freq, n_time = spectrogram.shape
    resolution_product = n_freq * n_time  # Metric we want to minimize

    # Reconstruction using inverse FFT and overlap-add
    reconstructed_segments = [
        np.fft.irfft(spectrogram[:, t], n=window_samps) * hann_window
        for t in range(n_time)
    ]
    reconstructed_signal = np.zeros_like(padded_sig, dtype=np.float64)
    for i, segment in enumerate(reconstructed_segments):
        start = i * step_samps
        reconstructed_signal[start:start + window_samps] += segment

    # Correct for Hann window overlap scaling
    scaling_factor = np.sum(hann_window**2) / step_samps
    reconstructed_signal /= scaling_factor
    
    # Remove the padding
    reconstructed_signal = reconstructed_signal[pad:-pad]
    
    # Compute relative error with a safeguard against division by zero
    signal_safe = np.where(np.abs(signal) < 1e-12, 1e-12, signal)
    relative_error = np.abs(signal - reconstructed_signal) / np.abs(signal_safe)
    
    # Evaluate maximum relative error (ignoring edge effects)
    max_error = np.max(relative_error[20:-20])
    return max_error, resolution_product

# -------------------------------------------
# Dynamic Optimization using Differential Evolution
# -------------------------------------------
error_threshold = 1.0e-4  # Maximum allowed error
penalty_weight = 1e12   # Large penalty weight to discourage violations

def objective(params):
    ws, ss = params
    max_error, res_product = compute_error_and_resolution(ws, ss, signal, fs)
    # If configuration is invalid, return a large penalty
    if np.isinf(max_error) or res_product is None or np.isnan(max_error):
        return penalty_weight
    # Apply a penalty if the error exceeds the threshold
    penalty = 0
    if max_error > error_threshold:
        penalty = penalty_weight * (max_error - error_threshold)
    return res_product + penalty

# Set bounds for window_size and step_size (adjust as needed)
# Here, the window and step sizes are expressed in fraction of the energy grid span
bounds = [(0.004, 0.02), (0.002, 0.01)]

# Run the differential evolution optimizer
result = differential_evolution(objective, bounds, strategy='best1bin', disp=True)
best_ws, best_ss = result.x
best_error, best_resolution = compute_error_and_resolution(best_ws, best_ss, signal, fs)

print("\nOptimal Parameters Found:")
print(f"Window Size: {best_ws:.6f}")
print(f"Step Size:   {best_ss:.6f}")
print(f"Max Relative Error: {best_error:.6e}")
print(f"Spectrogram Resolution (freq_bins x time_bins): {best_resolution}")