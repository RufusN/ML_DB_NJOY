# ──────────────────────────────────────────────────────────
# 1 ▸ extract_weights_bias_deep.py
# ──────────────────────────────────────────────────────────
#!/usr/bin/env python3
"""
extract_weights_bias_deep.py — walk an arbitrary‑depth MLP encoder and write
all parameters + leaky‑ReLU α into HDF5.  Supports both *inline* activations
(`Dense(..., activation=LeakyReLU(...))`) and *separate* `LeakyReLU()` layers.

HDF5 layout (example 256‑128‑16 encoder → decoder):
    W_hidden_0   (  1,256)  b_hidden_0 (256,)  alpha_0 ()
    W_hidden_1   (256,128)  b_hidden_1 (128,)  alpha_1 ()
    W_latent     (128, 16)  b_latent   (16,)   alpha_latent ()
    W_dec        ( 16, N)   b_dec      (N,)    ← decoder
"""
import argparse, h5py, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LeakyReLU

DEFAULT_ALPHA = 0.1  # used when we can’t infer a slope

def _get_alpha(layer_or_activation):
    """Return the leaky‑ReLU negative‑slope for either a Layer or tf fn."""
    if isinstance(layer_or_activation, LeakyReLU):
        return float(getattr(layer_or_activation, "alpha", DEFAULT_ALPHA))
    # Inline activation case (tf.nn.leaky_relu or keras activ str)
    try:
        cfg = layer_or_activation.get_config()
        if cfg.get("activation", "") == "linear":
            return DEFAULT_ALPHA  # not actually activated here
    except AttributeError:
        pass
    return DEFAULT_ALPHA


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--model_path", default='./3x45551_300_500_model.keras')
    cli.add_argument("--output",     default="model_weights_deep.h5")
    args = cli.parse_args()

    model = load_model(args.model_path, compile=False)

    # -------- locate encoder / decoder --------
    dense_layers = [lyr for lyr in model.layers if isinstance(lyr, Dense)]
    if len(dense_layers) < 2:
        raise RuntimeError("Model needs at least one encoder Dense + decoder Dense")

    decoder_dense   = dense_layers[-1]
    encoder_dense   = dense_layers[:-1]                 # all before decoder
    latent_dense    = encoder_dense[-1]                 # final encoder Dense
    hidden_denses   = encoder_dense[:-1]                # the rest

    # For quick lookup of subsequent layers when seeking separate activations
    idx_of = {id(lyr): i for i, lyr in enumerate(model.layers)}

    with h5py.File(args.output, "w") as hf:
        # ---- hidden stack ----
        for i, d in enumerate(hidden_denses):
            W, b = d.kernel.numpy().astype("float32"), d.bias.numpy().astype("float32")
            # find matching activation — either inline or the next LeakyReLU in graph
            alpha = DEFAULT_ALPHA
            if hasattr(d, "activation") and getattr(d.activation, "__name__", "") == "leaky_relu":
                alpha = _get_alpha(d.activation)
            else:
                # look ahead for first LeakyReLU after this Dense
                for nxt in model.layers[idx_of[id(d)]+1:]:
                    if isinstance(nxt, LeakyReLU):
                        alpha = _get_alpha(nxt)
                        break
            hf[f"W_hidden_{i}"] = W
            hf[f"b_hidden_{i}"] = b
            hf[f"alpha_{i}"]    = np.array(alpha, dtype="float32")

        # ---- latent Dense ----
        hf["W_latent"]       = latent_dense.kernel.numpy().astype("float32")
        hf["b_latent"]       = latent_dense.bias.numpy().astype("float32")
        hf["alpha_latent"]  = np.array(_get_alpha(latent_dense.activation), dtype="float32")

        # ---- decoder ----
        hf["W_dec"] = decoder_dense.kernel.numpy().astype("float32")
        hf["b_dec"] = decoder_dense.bias.numpy().astype("float32")
    print(f"[extract] wrote {len(hidden_denses)} hidden layers + latent + decoder → {args.output}")


if __name__ == "__main__":
    main()