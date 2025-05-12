#!/usr/bin/env python3
"""
benchmark_ops.py – average operation counts & timing (5 000 samples)
===================================================================
For **5 000** random (temperature, energy‑index) pairs, this script:
  • Reconstructs the XS value (same math as original pipeline)
  • Tallies an approximate count of *primitive floating‑point operations* in
    each logical stage of the pipeline
  • Accumulates wall‑clock time for each stage

At the end it prints average ops and milliseconds **per single reconstruction**.

Operation estimates (per call)
------------------------------
Stage              | Formula
------------------ | ---------------------------------------------
latent→subset matmul| `2 * latent_dim * len(flat_idxs)` (mul + add)
un‑scale           | `2 * flat_size` (mul + add)
IRFFT              | `2 * N * log2(N)` per IRFFT (Cooley‑Tukey estimate)
overlap‑add sum    | `len(seg_idxs)` (just the additions)
(The sub‑model forward pass before the subset‑decode isn’t counted because its
FLOPs are model‑specific; include it if you have the layer dims.)

Prerequisites (same folder)
---------------------------
  • base_energy_grid.h5                   – canonical energy grid (dataset "Base_E")
  • 3x45551_950_1050_spec_scalers.h5     – spec/T scalers
  • 3x45551_950_1050.keras               – trained model (first 3 layers: input → latent → dense)
"""

from __future__ import annotations
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from math import log2

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_base_grid(grid_path: str | Path = "base_energy_grid.h5") -> np.ndarray:
    with h5py.File(grid_path, "r") as hf:
        return hf["Base_E"][:]


def mapSegIdx(energy_idx: int, step_samps: int, window_samps: int):
    n_over = window_samps // step_samps
    first_seg = int(np.ceil((energy_idx - window_samps) / step_samps - 0.5))
    seg_idxs = np.arange(first_seg, first_seg + n_over, dtype=int)
    local = energy_idx - seg_idxs * step_samps
    return seg_idxs, local.tolist()


def predict_subset(hidden, W, b, seg_indices, h: int, w: int, c: int):
    flat = [f * (w * c) + t * c + ch
            for f in range(h) for t in seg_indices for ch in range(c)]
    flat_tf = tf.constant(flat, dtype=tf.int32)
    out = tf.matmul(hidden, tf.gather(W, flat_tf, axis=1)) + tf.gather(b, flat_tf)
    return tf.reshape(out, (-1, h, len(seg_indices), c)), np.array(flat)


def irfft_ops(n: int) -> float:
    """Rough FLOP estimate for a real FFT of length *n* (radix‑2)."""
    return 2 * n * log2(n)

# ──────────────────────────────────────────────────────────────────────────────
#  Reconstruction with instrumentation
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_point(T: float, E_idx: int,
                      submodel, W, b,
                      scaler_T, scaler_spec,
                      window_samps: int, step_samps: int,
                      hann_win, scaling_factor,
                      counters_ops: dict[str, float],
                      counters_time: dict[str, float]):
    h, w, c = 3, 45551, 2
    seg_idxs, local = mapSegIdx(E_idx, step_samps, window_samps)

    t0 = time.perf_counter()
    hidden = submodel(scaler_T.transform([[T]]))
    counters_time["latent"] += time.perf_counter() - t0  # latent‑layer forward
    # (latent ops not counted; add if you know the layer dims)

    # --- partial dense -----------------------------------------------------
    t0 = time.perf_counter()
    spec_scaled, flat = predict_subset(hidden, W, b, seg_idxs, h, w, c)
    dt = time.perf_counter() - t0
    counters_time["subset_dense"] += dt
    latent_dim = hidden.shape[1]
    counters_ops["subset_dense"] += 2 * latent_dim * len(flat)  # mul + add

    # --- un‑scale ----------------------------------------------------------
    t0 = time.perf_counter()
    spec_scaled = tf.squeeze(spec_scaled).numpy().reshape(1, -1)
    spec = spec_scaled * scaler_spec.scale_[flat] + scaler_spec.mean_[flat]
    counters_ops["unscale"] += 2 * spec.size
    spec = spec.reshape(h, len(seg_idxs), c)
    spec = spec[..., 0] + 1j * spec[..., 1]
    counters_time["unscale"] += time.perf_counter() - t0

    # --- inverse FFT & overlap‑add ----------------------------------------
    xs_val = 0.0
    for i in range(len(seg_idxs)):
        t0 = time.perf_counter()
        segment = np.fft.irfft(spec[:, i], n=window_samps) * hann_win
        counters_time["irfft"] += time.perf_counter() - t0
        counters_ops["irfft"] += irfft_ops(window_samps)

        idx = local[i] % window_samps  # safe wrap
        xs_val += segment[idx]
        counters_ops["overlap_add"] += 1  # one addition per segment

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
    scaling_factor = np.sum(hann_win ** 2) / step_samps

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
    submodel = tf.keras.Model(model.input, model.layers[2].output)
    dense = model.layers[3]
    W, b = dense.kernel, dense.bias

    # --- generate random combos -------------------------------------------
    n_tests = 5_000
    temps  = np.random.uniform(950, 1050,  size=n_tests)
    e_idxs = np.random.randint(100, 45_001, size=n_tests)

    # --- counters ----------------------------------------------------------
    ops = {k: 0.0 for k in ["subset_dense", "unscale", "irfft", "overlap_add"]}
    tms = {k: 0.0 for k in ["latent", "subset_dense", "unscale", "irfft"]}

    # --- benchmark loop ----------------------------------------------------
    t0_all = time.perf_counter()
    for T, E_idx in zip(temps, e_idxs):
        _ = reconstruct_point(T, int(E_idx), submodel, W, b,
                               scaler_T, scaler_spec,
                               window_samps, step_samps,
                               hann_win, scaling_factor,
                               ops, tms)
    elapsed_all = time.perf_counter() - t0_all

    # --- print averages ----------------------------------------------------
    print("\nAverage per‑point statistics (5 000 samples):")
    for k in ops:
        print(f"  {k:<12}: {ops[k] / n_tests:>12.2e} FLOPs   |  "
              f"time: {tms.get(k,0)/n_tests*1e3:6.3f} ms")

    print(f"\nTotal wall‑clock: {elapsed_all:.2f} s  →  "
          f"{elapsed_all / n_tests * 1_000:.2f} ms per point (avg)")


if __name__ == "__main__":
    main()
