#!/usr/bin/env python3
import os, glob, re
import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers

def read_spectrogram_h5(h5_filepath):
    with h5py.File(h5_filepath, 'r') as h5f:
        required = {'time_bins','frequencies','spectrogram_real','spectrogram_imag'}
        missing = required - set(h5f.keys())
        if missing:
            raise KeyError(f"Missing datasets: {missing}")
        real = h5f['spectrogram_real'][:]
        imag = h5f['spectrogram_imag'][:]
    return real + 1j*imag

def load_data_from_h5(directory):
    T_vals, RI_list = [], []
    all_files = glob.glob(os.path.join(directory, "spectrogram_T_*.h5"))
    print(f"[load_data] Looking in {directory!r}, found {len(all_files)} files")
    test_Ts = [1000.0]
    for path in all_files:
        temp = float(re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(path)).group(1))
        if 950.0 <= temp <= 1050.0 and temp not in test_Ts:
            spec = read_spectrogram_h5(path)
            T_vals.append([temp])
            # real+imag stacked along last axis
            RI_list.append(
                np.stack([np.real(spec), np.imag(spec)], axis=-1)
            )
    T_arr = np.array(T_vals, dtype=np.float32)
    RI_arr = np.array(RI_list, dtype=np.float32)
    return T_arr, RI_arr

def build_real_imag_model(input_shape, output_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(16, use_bias=True)(inp)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(np.prod(output_shape), activation='linear', use_bias=True)(x)
    out = layers.Reshape(output_shape)(x)
    return models.Model(inp, out)

def log_ae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(1.0 + tf.abs(y_pred - y_true) + 1e-12))

def main():
    DATA_DIR = "/mnt/d/new_specs_3_5313"

    # === 1) load & sanity-check ===
    T, RI = load_data_from_h5(DATA_DIR)
    if RI.ndim != 4 or RI.shape[0] == 0:
        raise RuntimeError(f"load_data_from_h5 returned RI.shape={RI.shape}, nothing to train on")
    N, h, w, c = RI.shape
    flat = h*w*c

    # === 2) fit scalers ===
    scaler_T = StandardScaler().fit(T)
    scaler_R = StandardScaler().fit(RI.reshape(N,flat))

    Tn = scaler_T.transform(T).astype(np.float32)
    Rn = scaler_R.transform(RI.reshape(N,flat)).astype(np.float32)

    # === 3) build & train ===
    model = build_real_imag_model((1,), (h,w,c))
    model.compile(optimizer="adam", loss=log_ae_loss)
    model.fit(
        Tn, Rn.reshape((N,h,w,c)),
        epochs=200, batch_size=32,
        validation_split=0.1,
        callbacks=[ tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) ],
        verbose=1
    )

    # === 4) extract weights & biases ===
    W0, b0 = model.layers[1].get_weights()  # Dense(1→16)
    W1, b1 = model.layers[3].get_weights()  # Dense(16→h*w*c)

    # === 5) save compactly ===
    np.savez_compressed(
      "mlp_pointwise.npz",
      W0=W0.astype(np.float32),
      b0=b0.astype(np.float32),
      W1=W1.astype(np.float32),
      b1=b1.astype(np.float32),
      T_scale=scaler_T.scale_.astype(np.float32),
      T_mean =scaler_T.mean_.astype(np.float32),
      R_scale=scaler_R.scale_.astype(np.float32),
      R_mean =scaler_R.mean_.astype(np.float32),
      h=h, w=w, c=c
    )
    print("Exported mlp_pointwise.npz")

if __name__=="__main__":
    main()
