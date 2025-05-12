#!/usr/bin/env python3
"""
benchmark_xs.py – time 5 000 random reconstructions
==================================================
Benchmarks the average latency of reconstructing a single XS value from the
latent‑space model.

• Temperatures are drawn uniformly in **[950 K, 1050 K]**.
• Energy‑grid indices are drawn uniformly in **[100, 45 000]**.

After the run, the script prints the total wall‑clock time and the average
milliseconds per point.

Prerequisites (same folder):
  • base_energy_grid.h5                   – canonical energy grid (dataset "Base_E")
  • 3x45551_950_1050_spec_scalers.h5     – spec/T scalers
  • 3x45551_950_1050.keras               – trained model
"""

from pathlib import Path
import time
import h5py
import numpy as np
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_base_grid(grid_path: str | Path = "base_energy_grid.h5") -> np.ndarray:
    """Load the 1‑D canonical energy grid exported by make_base_grid.py."""
    with h5py.File(grid_path, "r") as hf:
        return hf["Base_E"][:]


def mapSegIdx(energy_idx: int, step_samps: int, window_samps: int):
    """Return segment indices and per‑segment local sample indices."""
    n_over = window_samps // step_samps
    first_seg = int(np.ceil((energy_idx - window_samps) / step_samps - 0.5))
    seg_idxs = np.arange(first_seg, first_seg + n_over, dtype=int)
    local = energy_idx - seg_idxs * step_samps
    return seg_idxs, local.tolist()


def predict_subset(hidden, W, b, seg_indices, h: int, w: int, c: int):
    """Dense‑layer partial decode (only desired f/t/ch positions)."""
    flat = []
    for f in range(h):
        for t in seg_indices:
            for ch in range(c):
                flat.append(f * (w * c) + t * c + ch)
    flat_tf = tf.constant(flat, dtype=tf.int32)
    out = tf.matmul(hidden, tf.gather(W, flat_tf, axis=1)) + tf.gather(b, flat_tf)
    return tf.reshape(out, (-1, h, len(seg_indices), c)), np.array(flat)


def reconstruct_point(T: float, E_idx: int,
                      submodel, W, b,
                      scaler_T, scaler_spec,
                      window_samps: int, step_samps: int,
                      hann_win, scaling_factor):
    """Faithful to the original overlap‑add logic, with bounds‑safe indexing."""
    h, w, c = 3, 45551, 2
    seg_idxs, local = mapSegIdx(E_idx, step_samps, window_samps)

    # latent activations (1 × latent)
    hidden = submodel(scaler_T.transform([[T]]))

    # partial dense decode (only the required f/t/ch positions)
    spec_scaled, flat = predict_subset(hidden, W, b, seg_idxs, h, w, c)
    spec_scaled = tf.squeeze(spec_scaled).numpy().reshape(1, -1)

    # un‑scale
    spec = spec_scaled * scaler_spec.scale_[flat] + scaler_spec.mean_[flat]
    spec = spec.reshape(h, len(seg_idxs), c)
    spec = spec[..., 0] + 1j * spec[..., 1]

    # inverse STFT for each overlapped segment, then pick the sample of interest
    xs_val = 0.0
    for i, t in enumerate(seg_idxs):
        segment = np.fft.irfft(spec[:, i], n=window_samps) * hann_win
        idx = local[i]
        if idx < 0 or idx >= window_samps:
            # Safety clip (should not happen, but avoids IndexError)
            idx = idx % window_samps
        xs_val += segment[idx]

    return xs_val / scaling_factor

# ──────────────────────────────────────────────────────────────────────────────
#  Main benchmark
# ──────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)

    # --- artefacts ----------------------------------------------------------
    Base_E = load_base_grid()
    fs = len(Base_E)

    window_size = 4.628e-5
    step_size   = 2.314e-5
    window_samps = int(window_size * fs)
    step_samps   = int(step_size  * fs)

    hann_win = hann(window_samps)
    scaling_factor = np.sum(hann_win**2) / step_samps

    with h5py.File("3x45551_950_1050_spec_scalers.h5", "r") as hf:
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

    model = load_model("3x45551_950_1050.keras", compile=False)
    submodel = tf.keras.Model(model.input, model.layers[2].output)   # latent only
    dense = model.layers[3]
    W, b = dense.kernel, dense.bias

    # --- generate random combos -------------------------------------------
    n_tests = 5000
    temps  = np.random.uniform(950, 1050,  size=n_tests)
    e_idxs = np.random.randint(100, 45_001, size=n_tests)  # inclusive upper bound

    # --- benchmark loop ----------------------------------------------------
    t0 = time.perf_counter()
    for T, E_idx in zip(temps, e_idxs):
        _ = reconstruct_point(T, int(E_idx), submodel, W, b,
                               scaler_T, scaler_spec,
                               window_samps, step_samps,
                               hann_win, scaling_factor)
    elapsed = time.perf_counter() - t0

    print(f"{n_tests:,} reconstructions in {elapsed:.2f} s  →  "
          f"{elapsed / n_tests * 1_000:.2f} ms / point (avg)")


if __name__ == "__main__":
    main()
