import os
import glob
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
import sys


def read_spectrogram_h5(h5_filepath):
    # Not used when querying with batch script
    pass


def load_base(E_min, E_max):
    base_file_path = '/Volumes/T7 Shield/Base_E/capture_xs_data_0.h5'
    Base_Tval = 200.0
    data = pd.read_hdf(base_file_path, key='xs_data', compression='gzip')
    subset_T = data[data['T'] == Base_Tval]
    mask = (subset_T['ERG'] >= E_min) & (subset_T['ERG'] <= E_max)
    subset_range = subset_T[mask]
    Base_E = subset_range['ERG'].to_numpy()
    Base_xs = subset_range['XS'].to_numpy()
    sort_idx = np.argsort(Base_E)
    return Base_E[sort_idx], Base_xs[sort_idx]


def load_temperature(test_temp, Base_E, pad, E_min, E_max, file_dir):
    for file_path in glob.glob(os.path.join(file_dir, '*.h5')):
        data = pd.read_hdf(file_path, key='xs_data', compression='gzip')
        if test_temp not in data['T'].values:
            continue
        subset = data[data['T'] == test_temp]
        mask = (subset['ERG'] >= E_min) & (subset['ERG'] <= E_max)
        subset = subset[mask]
        E = subset['ERG'].to_numpy()
        xs = subset['XS'].to_numpy()
        idx = np.argsort(E)
        E, xs = E[idx], xs[idx]
        interp = interp1d(E, xs, kind='cubic', fill_value='extrapolate')
        signal = interp(Base_E)
        padded_sig = np.pad(signal, (pad, pad), mode='constant')
        padded_Base_E = np.pad(Base_E, (pad, pad), mode='constant')
        return padded_Base_E, padded_sig
    raise ValueError('Temperature not found')


def mapSegIdx(energy_idx, step_samps, window_samps):
    n_overlaps = window_samps // step_samps
    seg_idxs = []
    local_idxs = []
    seg_idx = int(np.ceil((energy_idx - window_samps)/step_samps - 1 + 0.5))
    for _ in range(n_overlaps):
        seg_idxs.append(seg_idx)
        start = seg_idx * step_samps
        local_idxs.append(energy_idx - start)
        seg_idx += 1
    return np.array(seg_idxs), np.array(local_idxs)


def analyse(pad, padded_Base_E, padded_sig, E_indices, results):
    E_indices = np.array(E_indices)
    rel_error = np.abs(results - padded_sig[E_indices]) / np.abs(padded_sig[E_indices]) * 100
    print('Relative Max Error (%):', np.max(rel_error))
    print('Relative Mean Error (%):', np.mean(rel_error))
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(padded_Base_E[pad:-pad], padded_sig[pad:-pad], label='True XS', lw=1)
    plt.scatter(padded_Base_E[E_indices], results, c='r', s=10, label='Table QS')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy (MeV)'); plt.ylabel('Cross-section'); plt.legend(); plt.show()
    plt.figure(figsize=(8,4))
    plt.plot(E_indices, rel_error, marker='o')
    plt.xlabel('Energy Index'); plt.ylabel('Error (%)'); plt.title('Quantized Table Relative Error'); plt.grid(True); plt.show()


def query_point(table_path, T, E_idx):
    """
    Query a single point (T, E_idx) using the batch script.
    Uses the same Python interpreter and correct script name.
    """
    # Determine path to the runtime script; assume it's in the same directory as this accuracy script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_script = os.path.join(script_dir, 'build_runtime_quant_2.py')
    cmd = [
        sys.executable, runtime_script, 'query',
        '--table', table_path,
        '--T', str(T),
        '--E_idx', str(E_idx)
    ]

    out = subprocess.check_output(cmd, cwd=script_dir).decode().strip()
    return float(out)


def main():
    # Base data settings
    E_min = 1e4 * 1e-6
    E_max = 1e6 * 1e-6
    Base_E, Base_xs = load_base(E_min, E_max)
    fs = len(Base_E)
    window_samps = int(0.00004628 * fs)
    step_samps   = int(0.00002314 * fs)
    pad = window_samps
    padded_Base_E, padded_sig = load_temperature(
        test_temp=1000.0,
        Base_E=Base_E,
        pad=pad,
        E_min=E_min,
        E_max=E_max,
        file_dir='/Volumes/T7 Shield/T_800_1200_data/800_1200'
    )
    E_indices = list(range(0, 91000, 1000)) 
    results = []
    table_file = 'w_hier.h5'
    for E_idx in E_indices:
        xs_val = query_point(table_file, 1000.0, E_idx)
        results.append(xs_val)
    analyse(pad, padded_Base_E, padded_sig, E_indices, np.array(results))

if __name__ == '__main__':
    main()
