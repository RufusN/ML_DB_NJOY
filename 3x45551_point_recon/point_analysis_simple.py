#!/usr/bin/env python3
"""
compute_single_point.py
---------------------------------
Reconstruct a single XS point from the latent‑space model as efficiently as
possible.  It now relies on a pre‑exported canonical energy grid stored in
`base_grid.h5`, eliminating the need to load the bulky base cross‑section
file at runtime.
"""

import os
import glob
import h5py
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_base_grid(grid_path: str = "base_energy_grid.h5") -> np.ndarray:
    """Return the canonical energy grid saved by `make_base_grid.py`."""
    with h5py.File(grid_path, "r") as hf:
        return hf["Base_E"][:]


def load_temperature(test_temp: float, Base_E: np.ndarray, pad: int,
                     E_min: float, E_max: float, data_dir: str):
    """Interpolate XS for `test_temp` onto `Base_E` and return padded arrays."""
    for fp in glob.glob(os.path.join(data_dir, "*.h5")):
        try:
            df = pd.read_hdf(fp, key="xs_data", compression="gzip")
        except Exception as e:
            print(f"[skip] {fp}: {e}")
            continue

        if test_temp not in df["T"].values:
            continue

        subset = df[df["T"] == test_temp]
        mask   = (subset["ERG"] >= E_min) & (subset["ERG"] <= E_max)
        subset = subset[mask]
        if len(subset) < 2:
            continue

        E  = subset["ERG"].to_numpy()
        xs = subset["XS"].to_numpy()
        sort = np.argsort(E)
        interp = interp1d(E[sort], xs[sort], kind="cubic",
                           fill_value="extrapolate")
        signal = interp(Base_E)

        padded_sig  = np.pad(signal,  (pad, pad))
        padded_E    = np.pad(Base_E,  (pad, pad))
        return padded_E, padded_sig

    raise RuntimeError(f"T={test_temp} not found in {data_dir}")

# ──────────────────────────────────────────────────────────────────────────────
#  Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def mapSegIdx(energy_idx: int, step_samps: int, window_samps: int):
    """Return segment indices and local indices for overlap‑add inverse FFT."""
    n_overlaps = window_samps // step_samps
    seg_idxs, local_idxs = [], []
    seg_idx = int(np.ceil((energy_idx - window_samps) / step_samps - 0.5))
    for _ in range(n_overlaps):
        seg_idxs.append(seg_idx)
        seg_idx += 1
        start = seg_idx * step_samps
        local_idxs.append(energy_idx - start)
    return np.asarray(seg_idxs, dtype=int), local_idxs


def predict_subset(model, input_data, seg_indices, h: int, w: int, c: int):
    """Decode only the time bins we need and return flat indices used."""
    latent_layer = model.layers[2]
    submodel = tf.keras.Model(model.input, latent_layer.output)
    hidden = submodel(input_data)

    dense = model.layers[3]
    W, b = dense.kernel, dense.bias

    flat_idxs = []
    for f in range(h):
        for t in seg_indices:
            for ch in range(c):
                flat_idxs.append(f * (w * c) + t * c + ch)

    flat_idxs_tf = tf.constant(flat_idxs, dtype=tf.int32)
    W_sub = tf.gather(W, flat_idxs_tf, axis=1)
    b_sub = tf.gather(b, flat_idxs_tf)

    out = tf.matmul(hidden, W_sub) + b_sub
    out = tf.reshape(out, (-1, h, len(seg_indices), c))
    return out, np.array(flat_idxs)


def get_spectrogram_output(model, T: float, scaler_T, scaler_spec,
                           seg_indices):
    h, w, c = 3, 45551, 2  # network architecture constants

    T_norm = scaler_T.transform([[T]])
    pred_scaled, flat_idxs = predict_subset(model, T_norm, seg_indices,
                                            h, w, c)

    pred_scaled = tf.squeeze(pred_scaled).numpy().reshape(1, -1)
    pred_unscaled = (pred_scaled * scaler_spec.scale_[flat_idxs]
                     + scaler_spec.mean_[flat_idxs])
    pred_unscaled = pred_unscaled.reshape(h, len(seg_indices), c)

    return pred_unscaled[..., 0] + 1j * pred_unscaled[..., 1]


def point_reconstruction(spectrogram, window_samps: int, step_samps: int,
                         local_indices):
    win = hann(window_samps)
    segments = [np.fft.irfft(spectrogram[:, t], n=window_samps) * win
                for t in range(spectrogram.shape[1])]
    xs = sum(seg[local_indices[i]] for i, seg in enumerate(segments))
    return xs / (np.sum(win**2) / step_samps)

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # --- user‑configurable ---------------------------------------------------
    E_min, E_max = 1e4 * 1e-6, 1e6 * 1e-6  # MeV
    window_size  = 0.00004628             # fraction of Base_E length
    step_size    = 0.00002314
    test_temp    = 1000.0                # K
    E_idx        = 500                   # energy‑grid index to reconstruct
    data_dir     = r"/Volumes/T7 Shield/T_800_1200_data/800_1200"

    # --- constant artefacts --------------------------------------------------
    Base_E = load_base_grid()
    fs = len(Base_E)
    window_samps, step_samps = int(window_size * fs), int(step_size * fs)
    pad = window_samps  # symmetric zero‑padding length

    # ---- scalers ------------------------------------------------------------
    with h5py.File("./3x45551_950_1050_spec_scalers.h5", "r") as hf:
        scaler_spec = StandardScaler()
        scaler_spec.scale_ = hf["spec_scale"][:]
        scaler_spec.mean_  = hf["spec_mean"][:]
        scaler_spec.var_   = scaler_spec.scale_ ** 2
        scaler_spec.n_features_in_ = scaler_spec.scale_.size
        scaler_spec.n_samples_seen_ = 1

        scaler_T = StandardScaler()
        scaler_T.scale_ = hf["T_scale"][:]
        scaler_T.mean_  = hf["T_mean"][:]
        scaler_T.var_   = scaler_T.scale_ ** 2
        scaler_T.n_features_in_ = scaler_T.scale_.size
        scaler_T.n_samples_seen_ = 1

    # ---- data + model -------------------------------------------------------
    padded_E, _ = load_temperature(test_temp, Base_E, pad,
                                   E_min, E_max, data_dir)
    seg_indices, local_indices = mapSegIdx(E_idx, step_samps, window_samps)

    model = load_model("./3x45551_950_1050.keras", compile=False)

    # ---- reconstruction -----------------------------------------------------
    spect = get_spectrogram_output(model, test_temp,
                                   scaler_T, scaler_spec, seg_indices)
    xs_val = point_reconstruction(spect, window_samps,
                                  step_samps, local_indices)

    print(f"Reconstructed XS @ index {E_idx} (E = {padded_E[E_idx]:.3e} MeV): "
          f"{xs_val:.6e}")


if __name__ == "__main__":
    main()
