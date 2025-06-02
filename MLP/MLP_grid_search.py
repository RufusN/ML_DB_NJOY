#!/usr/bin/env python3
"""
quick_grid_search.py – grid‑search wrapper around your single‑hidden‑layer
spectrogram MLP.

**What’s new (2025‑05‑22 b)**
──────────────────────────────
1. **Abs‑weighted noise, like your original script**
   Each noisy duplicate is now generated with
   `j_noisy = j + |j| * ε`,  ε ∼ 𝒩(0, σ²), instead of the previous
   multiplicative `j*(1+ε)`.  This matches your two‑channel (real/imag)
   augmentation exactly.

2. **(Optional) peak‑aware loss**
   A `--peak_k` flag lets you boost resonance bins in the loss via
   `1 + k·|y_true| / mean(|y_true|)`.  Default *k = 0* keeps the original
   `log_ae_loss`.

Grid parameters (unchanged)
    • Temperature transform        (T, √T, T², T³, log T)
    • Multiplicative noise σ       (0, 1e‑6, 1e‑5)
    • Leaky‑ReLU negative slope α  (0.01, 0.05, 0.1)

Run:
$ python quick_grid_search.py --data  /path/to/3x45551_spectrograms \
                              --epochs 25 --batch 32 --peak_k 1.0
"""

import argparse, os, glob, re, time, itertools, warnings, sys, random

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ───────────────────────────── helper io ───────────────────────────

def read_spectrogram_h5(fp):
    with h5py.File(fp, "r") as h5:
        req = {"time_bins", "frequencies", "spectrogram_real", "spectrogram_imag"}
        missing = req - set(h5.keys())
        if missing:
            raise KeyError(f"Missing datasets in {fp}: {missing}")
        spec = h5["spectrogram_real"][:] + 1j * h5["spectrogram_imag"][:]
        return h5["time_bins"][:], h5["frequencies"][:], spec


def load_data_from_h5(directory, T_min=300, T_max=600, n_holdout=40, seed=123):
    rng = random.Random(seed)
    holdout_Ts = [round(rng.uniform(T_min, T_max), 0) for _ in range(n_holdout)]

    T_vals, specs, files = [], [], []
    time_bins_ref = freq_ref = None
    for fp in glob.glob(os.path.join(directory, "spectrogram_T_*.h5")):
        m = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(fp))
        if not m:
            continue
        T = float(m.group(1))
        if not (T_min <= T <= T_max) or T in holdout_Ts:
            continue
        tb, fr, spec = read_spectrogram_h5(fp)
        if time_bins_ref is None:
            time_bins_ref, freq_ref = tb, fr
        T_vals.append(T)
        specs.append(np.stack([spec.real, spec.imag], axis=-1))
        files.append(os.path.basename(fp))
        print(f"Loaded {fp} (T={T})")

    if not T_vals:
        sys.exit("No training files found")

    return (np.array(T_vals, dtype=np.float32)[:, None],
            np.array(specs, dtype=np.float32),
            dict(files=files, time_bins=time_bins_ref, freq=freq_ref,
                 T_min=T_min, T_max=T_max, holdout=holdout_Ts))

# ───────────────────────────── model & losses ───────────────────────────

def build_mlp(input_shape, output_shape, alpha=0.1, l2=0.0):
    inp = layers.Input(shape=input_shape)
    x   = layers.Dense(16, kernel_regularizer=regularizers.l2(l2))(inp)
    x   = LeakyReLU(alpha=alpha)(x)
    x   = layers.Dense(np.prod(output_shape))(x)
    out = layers.Reshape(output_shape)(x)
    return models.Model(inp, out)

@tf.function
def log_ae_loss(y_true, y_pred, eps=1e-16):
    return tf.reduce_mean(tf.math.log(1.0 + tf.abs(y_pred - y_true) + eps))

@tf.function
def peak_aware_log_ae(y_true, y_pred, k=1.0, eps=1e-16):
    avg = tf.reduce_mean(tf.abs(y_true))
    weight = 1.0 + k * tf.abs(y_true) / (avg + 1e-12)
    err = tf.math.log(1.0 + tf.abs(y_pred - y_true) + eps)
    return tf.reduce_mean(weight * err)

# ───────────────────────────── trial runner ───────────────────────────

def run_trial(T_raw, spec_raw, transform_fn, sigma, alpha, k_peak, epochs=25, batch=32, seed=42):
    # temp transform & scale
    T_trans = transform_fn(T_raw.astype(np.float32))
    T_norm  = StandardScaler().fit_transform(T_trans)

    # flatten spec
    N, H, W, C = spec_raw.shape
    flat_clean = spec_raw.reshape(N, -1)

    # duplicate + abs‑weighted noise
    if sigma > 0.0:
        rng   = np.random.default_rng(seed)
        eps   = rng.normal(0.0, sigma, size=flat_clean.shape)
        flat_noisy = flat_clean + np.abs(flat_clean) * eps
        flat  = np.concatenate([flat_clean, flat_noisy], axis=0)
        T_norm = np.concatenate([T_norm, T_norm], axis=0)
    else:
        flat = flat_clean

    flat_scaled = StandardScaler().fit_transform(flat)
    spec_scaled = flat_scaled.reshape(-1, H, W, C)

    # split and model
    T_tr, T_val, X_tr, X_val = train_test_split(T_norm, spec_scaled, test_size=0.2, random_state=seed)
    model = build_mlp((1,), (H, W, C), alpha)
    loss_fn = (log_ae_loss if k_peak == 0 else
               lambda y, ŷ: peak_aware_log_ae(y, ŷ, k=k_peak))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn, metrics=["mse", "mae"])

    cb  = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
    t0  = time.time()
    hist= model.fit(T_tr, X_tr, validation_data=(T_val, X_val), epochs=epochs, batch_size=batch,
                    callbacks=[cb], verbose=0)
    dur = time.time() - t0

    return dict(val_loss=hist.history["val_loss"][-1],
                val_MAE=hist.history["val_mae"][-1],
                val_MSE=hist.history["val_mse"][-1],
                time_s=dur)

# ───────────────────────────── main driver ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--peak_k", type=float, default=0.0,
                    help="k=0 uses plain log‑AE; k>0 boosts resonance bins")
    args = ap.parse_args()

    T_raw, spec_raw, _ = load_data_from_h5(args.data)
    print(f"Loaded {len(T_raw)} clean samples — spec shape {spec_raw.shape[1:]}")

    T_TRANS = {
        "T":     lambda t: t,
        "sqrtT": np.sqrt,
        "T^2":   np.square,
        "T^3":   lambda t: np.power(t,3),
        "logT":  np.log,
    }
    SIGMAS = [0.0, 1e-6, 1e-5]
    ALPHAS = [0.01, 0.05, 0.1]

    rows = []
    for (name, fn), σ, α in itertools.product(T_TRANS.items(), SIGMAS, ALPHAS):
        print(f"→ {name:5s}  σ={σ:.0e}  α={α}  k={args.peak_k}")
        try:
            m = run_trial(T_raw, spec_raw, fn, σ, α, args.peak_k, args.epochs, args.batch)
            rows.append(dict(transform=name, sigma=σ, alpha=α, **m))
        except Exception as e:
            warnings.warn(f"FAILED ({name}, σ={σ}, α={α}): {e}")

    df = pd.DataFrame(rows).sort_values("val_loss")
    print("\nRESULTS (sorted by val_loss):")
    print(df.to_string(index=False, float_format="{:.3g}".format))
    df.to_csv("grid_results.csv", index=False)
    print("Saved grid_results.csv")

if __name__ == "__main__":
    main()

