#!/usr/bin/env python3
"""
generate_model_weights_h5.py — extract weights & biases from your Keras model
and save them into a single HDF5 file for fast GPU loading.

Usage:
    python generate_model_weights_h5.py \
        --model_path 3x45551_950_1050.keras \
        --output      model_weights.h5
"""

import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, Dense

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        type=str,
        default="best_model_3_5313.h5",
        help="Path to your Keras .keras model file"
    )
    p.add_argument(
        "--output",
        type=str,
        default="./model_weights.h5",
        help="Output HDF5 file to write weights into"
    )
    args = p.parse_args()

    # Load the Keras model (only to extract weights)
    model = load_model(args.model_path, compile=False)

    # Find the first Dense layer (input→latent)
    dense0 = next(l for l in model.layers if isinstance(l, Dense))
    W0 = dense0.kernel.numpy()  # shape (1,16)
    b0 = dense0.bias.numpy()    # shape (16,)

    # Find the LeakyReLU layer to get its alpha
    lrelu = next(l for l in model.layers if isinstance(l, LeakyReLU))
    # Keras stores slope in .alpha or .negative_slope depending on version
    alpha = getattr(lrelu, "alpha", getattr(lrelu, "negative_slope", 0.1))

    # Find the second Dense layer (latent→full spectrogram)
    # It will be the next Dense after dense0
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    if len(dense_layers) < 2:
        raise RuntimeError("Expected at least two Dense layers in the model")
    dense_dec = dense_layers[1]
    W_dec = dense_dec.kernel.numpy()  # shape (16, H*W_TIME*C)
    b_dec = dense_dec.bias.numpy()    # shape (H*W_TIME*C,)

    # Write everything to HDF5
    with h5py.File(args.output, "w") as hf:
        hf.create_dataset("W0",      data=W0,    dtype="float32")
        hf.create_dataset("b0",      data=b0,    dtype="float32")
        hf.create_dataset("alpha",   data=np.array(alpha, dtype="float32"))
        hf.create_dataset("W_dec",   data=W_dec, dtype="float32")
        hf.create_dataset("b_dec",   data=b_dec, dtype="float32")

    print(f"Saved weights to {args.output!r}:")
    print(f"  W0     shape {W0.shape}")
    print(f"  b0     shape {b0.shape}")
    print(f"  alpha  = {alpha}")
    print(f"  W_dec  shape {W_dec.shape}")
    print(f"  b_dec  shape {b_dec.shape}")

if __name__ == "__main__":
    main()
