import os
import glob
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from scipy.optimize import differential_evolution

# -------------------------------------------
# 1) User-defined parameters and energy range
# -------------------------------------------
E_min = 1e4 * 1e-6  # MeV (10 keV)
E_max = 1e6 * 1e-6  # MeV (1 MeV)

# Baseline and test file paths and temperatures
base_file_path = r"/Volumes/T7 Shield/NJOY/Fe56_data/Fe56_300.0.h5"
Base_Tval = 300.0

test_T = 305.0  # example test temperature
test_file = f"/Volumes/T7 Shield/NJOY/Fe56_data/Fe56_{test_T}.h5"

# -------------------------------------------
# Helper: load PyTables-style HDF5 into DataFrame
# -------------------------------------------
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

# -------------------------------------------
# 2) Load baseline data for reference grid
# -------------------------------------------
try:
    df_base = load_h5_dataframe(base_file_path)
except Exception as e:
    print(f"Error reading baseline HDF5 file: {e}")
    exit(1)

print(f"\nProcessing baseline T = {Base_Tval} from {base_file_path}...")
# Filter by temperature if column exists
if "T" in df_base.columns:
    df_base = df_base[df_base["T"] == Base_Tval]
# Apply energy range mask (MeV)
mask = (df_base["ERG"] >= E_min) & (df_base["ERG"] <= E_max)
subset_base = df_base[mask].copy()
if subset_base.empty:
    print("No data in baseline energy range. Exiting.")
    exit(1)

# Extract and sort baseline grid\ nBase_E = subset_base["ERG"].to_numpy()
Base_xs = subset_base["XS"].to_numpy()
sort_idx = np.argsort(Base_E)
Base_E = Base_E[sort_idx]
Base_xs = Base_xs[sort_idx]
fs = len(Base_E)
print(f"Length of baseline energy grid: {fs}")

# -------------------------------------------
# 3) Load test data for signal generation
# -------------------------------------------
try:
    df_test = load_h5_dataframe(test_file)
except Exception as e:
    print(f"Error reading test HDF5 file: {e}")
    exit(1)

print(f"\nProcessing test T = {test_T} from {test_file}...")
# Filter by temperature if present
if "T" in df_test.columns:
    df_test = df_test[df_test["T"] == test_T]
mask_test = (df_test["ERG"] >= E_min) & (df_test["ERG"] <= E_max)
subset_test = df_test[mask_test].copy()
if len(subset_test) < 2:
    print(f"Insufficient test data for T = {test_T}")
    exit(1)

# Extract and sort test arrays
E_test = subset_test["ERG"].to_numpy()
xs_test = subset_test["XS"].to_numpy()
sort_idx = np.argsort(E_test)
E_test = E_test[sort_idx]
xs_test = xs_test[sort_idx]

# Interpolate onto baseline grid
interp_func = interp1d(E_test, xs_test, kind="cubic", fill_value="extrapolate")
signal = interp_func(Base_E)

# -------------------------------------------
# 4) Compute error and resolution for spectrogram
# -------------------------------------------
def compute_error_and_resolution(window_size, step_size, signal, fs):
    window_samps = int(window_size * fs)
    step_samps = int(step_size * fs)
    if window_samps < 2 or step_samps < 1:
        return np.inf, None
    pad = window_samps // 2
    padded = np.pad(signal, (pad, pad), mode='constant')
    hann_win = hann(window_samps)
    starts = range(0, len(padded) - window_samps + 1, step_samps)
    windows = [padded[s:s+window_samps] * hann_win for s in starts]
    if not windows:
        return np.inf, None
    spec = np.array([np.fft.rfft(w) for w in windows]).T
    n_freq, n_time = spec.shape
    res_product = n_freq * n_time
    # Reconstruction via overlap-add
    recon_segs = [np.fft.irfft(spec[:, t], n=window_samps) * hann_win for t in range(n_time)]
    recon = np.zeros_like(padded)
    for i, seg in enumerate(recon_segs):
        recon[i*step_samps:i*step_samps+window_samps] += seg
    recon /= (np.sum(hann_win**2) / step_samps)
    recon = recon[pad:-pad]
    safe = np.where(np.abs(signal) < 1e-12, 1e-12, signal)
    rel_err = np.abs(signal - recon) / np.abs(safe)
    max_err = np.max(rel_err[20:-20])
    return max_err, res_product

# -------------------------------------------
# 5) Differential Evolution Optimization
# -------------------------------------------
bounds = [(0.004, 0.02), (0.002, 0.01)]
error_threshold = 1e-4
penalty = 1e12

def objective(params):
    ws, ss = params
    max_err, res_prod = compute_error_and_resolution(ws, ss, signal, fs)
    if np.isinf(max_err) or res_prod is None or np.isnan(max_err):
        return penalty
    if max_err > error_threshold:
        return res_prod + penalty * (max_err - error_threshold)
    return res_prod

result = differential_evolution(objective, bounds, strategy='best1bin', disp=True)
best_ws, best_ss = result.x
best_err, best_res = compute_error_and_resolution(best_ws, best_ss, signal, fs)

print("\nOptimal Parameters Found:")
print(f"Window Size: {best_ws:.6f}")
print(f"Step Size:   {best_ss:.6f}")
print(f"Max Relative Error: {best_err:.6e}")
print(f"Spectrogram Resolution (freq_bins x time_bins): {best_res}")
