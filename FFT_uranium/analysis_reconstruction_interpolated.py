import os
import glob
import h5py
import pandas as pd
import numpy as np
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# ---------------------------
# 1) User-defined parameters
# ---------------------------
E_min = 1e4 * 1e-6  # MeV (10 keV)
E_max = 5e4 * 1e-6  # MeV (1 MeV)

only_integer = True


# Path to spectrogram files directory
spectrogram_dir = Path('/Users/ru/FFT/ML_DB_NJOY/ML_prediction')
# Path to XS HDF5 data files (one per temperature)
xs_data_dir = Path('/Volumes/T7 Shield/NJOY/Fe56_data_250k')

relative_errors_no_scale = {}

# ---------------------------
# 2) Load base values for interpolation (T=300)
# ---------------------------
base_file_path = r"./base_energy_grid_90k.h5"

try:
    with h5py.File(base_file_path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise KeyError("No datasets found in base energy grid file")
        dataset_name = keys[0]
        subset_T = f[dataset_name][:]
        mask = (subset_T >= E_min) & (subset_T <= E_max)
        subset_range = subset_T[mask].copy()

# Extract and sort baseline grid using original energy points
    Base_E = np.array(subset_range)
    sort_idx = np.argsort(Base_E)
    Base_E = Base_E[sort_idx]
    fs = len(Base_E)
    print(f"Loaded baseline energy grid from '{dataset_name}', length: {fs}")
except Exception as e:
    print(f"Error reading baseline energy grid file: {e}")
    exit(1)
print(f"Length of baseline energy grid: {fs}")

# Hann window for overlap-add
window_samps = 4
step_samps   = 2
hann_window = hann(window_samps)

# ---------------------------
# 3) Process each spectrogram
# ---------------------------
spectrogram_files = list(spectrogram_dir.glob('*.h5'))
for spec_file in spectrogram_files:
    # Extract temperature from filename, e.g. spectrogram_T_305.h5
    name = spec_file.stem  # 'spectrogram_T_305'
    try:
        T_val = float(name.split('_')[2])
    except Exception:
        print(f"Skipping unexpected file: {spec_file.name}")
        continue
    if only_integer:
        if not float(T_val).is_integer():
            continue
    T_val = int(T_val)

    print(f"\nProcessing spectrogram for T = {T_val} from {spec_file}")
    # Load spectrogram arrays
    with h5py.File(spec_file, 'r') as f:
        time_bins = f['time_bins'][:]
        frequencies = f['frequencies'][:]
        spec_real = f['spectrogram_real'][:]
        spec_imag = f['spectrogram_imag'][:]
    spectrogram = spec_real + 1j * spec_imag

    # Inverse overlap-add reconstruction
    padded_len = 2 * window_samps + fs
    recon_signal = np.zeros(padded_len)
    for i in range(len(time_bins)):
        segment = np.fft.irfft(spectrogram[:, i], n=window_samps) * hann_window
        start = (i + 1) * step_samps
        recon_signal[start:start + window_samps] += segment
    # Scale to compensate Hann overlap
    scaling = np.sum(hann_window**2) / step_samps
    recon_signal /= scaling
    # Remove padding
    recon_signal = recon_signal[window_samps:-window_samps]

    # ---------------------------
    # 4) Load XS data for T_val
    # ---------------------------
    xs_file = xs_data_dir / f'Fe56_{T_val}.0.h5'
    try:
        with h5py.File(xs_file, 'r') as f:
            grp = f['data']
            items = [name.decode('utf-8') for name in np.array(grp['block0_items'], dtype='S')]
            values = np.array(grp['block0_values'])
        df_xs = pd.DataFrame(values, columns=items)
    except Exception as e:
        print(f"Error reading XS file for T={T_val}: {e}")
        continue
    # Filter by temperature column if exists
    if 'T' in df_xs.columns:
        df_xs = df_xs[df_xs['T'] == T_val]
    mask = (df_xs['ERG'] * 1e-6 >= E_min) & (df_xs['ERG'] * 1e-6 <= E_max)
    subset_xs = df_xs[mask].copy()
    subset_xs['ERG'] = subset_xs['ERG'] * 1e-6
    E = subset_xs['ERG'].to_numpy()
    xs = subset_xs['XS'].to_numpy()
    sort_idx = np.argsort(E)
    E = E[sort_idx]
    xs = xs[sort_idx]

    # Interpolate onto Base_E grid
    interp_func = interp1d(E, xs, kind='cubic', fill_value='extrapolate')
    true_signal = interp_func(Base_E)

    # Compute relative error
    rel_err = np.abs(true_signal - recon_signal) / np.where(np.abs(true_signal) < 1e-12, 1e-12, np.abs(true_signal))
    relative_errors_no_scale[T_val] = rel_err

    # Plot and save relative error
    plt.figure()
    plt.plot(Base_E, rel_err * 100, color='black', alpha=0.8)
    plt.plot(Base_E, [0.1]*len(Base_E), color='red', label='Target 0.1%')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Relative Error (%)')
    plt.title(f'Relative Error for T={T_val}K')
    plt.savefig(f'./Figures/xs_relative_err_{T_val}.png', dpi=200)
    plt.close()

# ---------------------------
# 5) Aggregate and visualize errors
# ---------------------------
temps_sorted = sorted(relative_errors_no_scale.keys())
all_errors = np.concatenate([relative_errors_no_scale[T] for T in temps_sorted])

# 5a) Error distribution histograms
for threshold in [0.002, 0.0011]:
    prop = np.sum(all_errors < threshold) / all_errors.size * 100
    plt.figure(figsize=(8,6))
    bins = np.logspace(np.log10(max(all_errors.min(),1e-12)), np.log10(all_errors.max()), 100)
    plt.hist(all_errors, bins=bins, alpha=0.7, edgecolor='black')
    plt.xscale('log')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title(f'Relative Error Distribution (Proportion < {threshold}: {prop:.2f}%)')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=1.5)
    plt.tight_layout()
    plt.savefig(f'./Figures/dist_error_{threshold}.png', dpi=200)
    plt.close()

# 5b) Mean and max error vs temperature
means = [np.mean(relative_errors_no_scale[T]) for T in temps_sorted]
maxs = [np.max(relative_errors_no_scale[T]) for T in temps_sorted]
plt.figure(figsize=(8,6))
plt.plot(temps_sorted, means, marker='o', label='Mean Error')
plt.plot(temps_sorted, maxs, marker='s', label='Max Error')
plt.yscale('log')
plt.xlabel('Temperature (K)')
plt.ylabel('Relative Error')
plt.title('Mean and Max Relative Error vs Temperature')
plt.grid(which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/error_vs_temperature.png', dpi=200)
plt.close()

# 5c) Scatter and heatmap visualizations
from matplotlib import colors, cm
plt.figure(figsize=(10,6))
cmap = plt.get_cmap('coolwarm')
n = len(temps_sorted)
norm = colors.Normalize(vmin=min(temps_sorted), vmax=max(temps_sorted))
fig, ax = plt.subplots()
for i, T in enumerate(temps_sorted):
    errs = relative_errors_no_scale[T]
    idxs = np.arange(len(errs))
    ax.scatter(idxs, errs, color=cmap(norm(T)), alpha=(i+1)/n, s=10, edgecolors='none')
ax.set_yscale('log')
ax.set_xlabel('Energy Index')
ax.set_ylabel('Relative Error')
ax.set_title('Relative Error vs Energy Index (Colored by Temperature)')
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Temperature (K)')
plt.tight_layout()
plt.savefig('./Figures/scatter_error_energy_temp.png', dpi=200)
plt.close()

# Heatmap
error_matrix = np.vstack([relative_errors_no_scale[T] for T in temps_sorted]) * 100
plt.figure(figsize=(12,8))
plt.imshow(error_matrix, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Relative Error (%)')
plt.xlabel('Energy Index')
plt.ylabel('Temperature (K)')
plt.yticks(ticks=np.arange(len(temps_sorted)), labels=temps_sorted)
plt.title('Heatmap of Relative Errors')
plt.tight_layout()
plt.savefig('./Figures/heatmap_error.png', dpi=200)
plt.close()

# 5d) Thresholded points and CSV export
all_points = [(err, T, idx) for T, errs in relative_errors_no_scale.items() for idx, err in enumerate(errs)]
errors = np.array([pt[0] for pt in all_points])
thr_black = 0.002
thr_grey = 0.0011
black_pts = [(idx, T, err) for (err, T, idx) in all_points if err > thr_black]
grey_pts = [(idx, T, err) for (err, T, idx) in all_points if thr_grey < err <= thr_black]
df_black = pd.DataFrame(black_pts, columns=['energy_idx','temperature','rel_error'])
df_grey = pd.DataFrame(grey_pts, columns=['energy_idx','temperature','rel_error'])
df_black.to_csv('./Figures/points_above_0.2pct.csv', index=False)
df_grey.to_csv('./Figures/points_0.1_to_0.2pct.csv', index=False)

plt.figure(figsize=(8,6))
plt.scatter([p[0] for p in black_pts], [p[1] for p in black_pts], c='black', s=20, label='rel err > 0.2%')
plt.scatter([p[0] for p in grey_pts], [p[1] for p in grey_pts], c='grey', s=20, label='0.11% < rel err ≤ 0.2%')
plt.xscale('log')
plt.xlabel('Energy Index')
plt.ylabel('Temperature (K)')
plt.title('Thresholded Error Points')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./Figures/thresholded_error_scatter.png', dpi=200)
plt.close()
