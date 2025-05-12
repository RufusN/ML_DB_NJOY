#!/usr/bin/env python3
"""
xs_batch_gpu.py – batched, XLA‑compiled GPU reconstruction
=========================================================
Reconstruct *many* (T, E_idx) pairs in one shot on the GPU.  Everything after
loading weights is TensorFlow, kept in float32 to avoid dtype mismatches.

• Batch size set by --batch (default 8192)
• Temperatures uniform [950, 1050] K; indices uniform [100, 45 000].
• Window/step match training (2048 / 512).
• Prints total time and µs per point after JIT warm‑up.

Run:

    python xs_batch_gpu.py --batch 8192
"""
import argparse, time, h5py, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LeakyReLU
from scipy.signal.windows import hann

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # quiet TF logs

# ───── constants ─────
H, W_TIME, C = 3, 45551, 2
WINDOW_SAMPS = 2048
STEP_SAMPS   =  512
MODEL_PATH   = "3x45551_950_1050.keras"
SCALER_PATH  = "3x45551_950_1050_spec_scalers.h5"

hann_win = tf.constant(hann(WINDOW_SAMPS).astype("float32"))
scale_fac = tf.constant((hann_win.numpy()**2).sum() / STEP_SAMPS, tf.float32)

# ───── precomputed segment offsets ─────
N_OVER = WINDOW_SAMPS // STEP_SAMPS
SEG_OFFSETS = tf.constant(list(range(N_OVER)), dtype=tf.int32)

# ───── utility to map energy idx → (seg_idxs, local_indices) ─────

def seg_and_local(e_idx):
    first = tf.cast(tf.math.ceil((e_idx - WINDOW_SAMPS)/STEP_SAMPS - 0.5), tf.int32)
    segs = first + SEG_OFFSETS  # shape [N_OVER]
    local = e_idx - segs * STEP_SAMPS
    return segs, local

# vectorised over batch
@tf.function
def xs_batch(T_batch, E_batch,
             W0, b0, alpha, W_dec, b_dec,
             spec_scale, spec_mean):
    # ---- latent ----
    T_norm = (T_batch - T_mean) / T_scale  # [N]
    hidden = tf.nn.leaky_relu(tf.linalg.matmul(tf.expand_dims(T_norm,1), W0) + b0, alpha)  # [N,latent]

    # ---- segment indices and local offsets ----
    first = tf.cast(tf.math.ceil((E_batch - WINDOW_SAMPS)/STEP_SAMPS - 0.5), tf.int32)  # [N]
    segs = tf.expand_dims(first,1) + tf.reshape(SEG_OFFSETS, [1,-1])                   # [N,N_OVER]
    local = E_batch[:,None] - segs * STEP_SAMPS
    # wrap local indices into [0, WINDOW_SAMPS)
    local = tf.math.floormod(local, WINDOW_SAMPS)

    # ---- flat decoder indices ----
    N = tf.shape(segs)[0]
    # frequency offset
    f_off = tf.reshape(tf.range(H, dtype=tf.int32) * (W_TIME * C), [1, H, 1, 1])  # [1, H, 1, 1]
    # expand segments for frequency and channel dims
    segs_e = tf.reshape(segs, [N, 1, N_OVER, 1])                                 # [N, 1, N_OVER, 1]
    ch_off = tf.reshape(tf.range(C, dtype=tf.int32), [1, 1, 1, C])                # [1, 1, 1, C]
    # combine offsets: frequency + time-bin + channel
    flat_idx = f_off + segs_e * C + ch_off                                        # [N, H, N_OVER, C]
    flat_idx_flat = tf.reshape(flat_idx, [N, -1])                                 # [N, N_flat]

    # ---- sparse decoder gather ----
    W_sub_raw = tf.gather(W_dec, flat_idx_flat, axis=1)                               # [latent,N,N_flat]
    W_sub = tf.transpose(W_sub_raw, [1,0,2])                                          # [N,latent,N_flat]
    b_sub = tf.gather(b_dec, flat_idx_flat, axis=0)                                   # [N,N_flat]

    # ---- decode and unscale ----
    spec_scaled_flat = tf.reduce_sum(tf.expand_dims(hidden,2) * W_sub, axis=1) + b_sub # [N,N_flat]
    spec_scaled = tf.reshape(spec_scaled_flat, [N,H,N_OVER,C])                        # [N,H,N_OVER,C]

    scale_flat = tf.gather(spec_scale, flat_idx_flat)
    mean_flat  = tf.gather(spec_mean,  flat_idx_flat)
    spec = spec_scaled * tf.reshape(scale_flat, [N,H,N_OVER,C]) + \
           tf.reshape(mean_flat,  [N,H,N_OVER,C])                                     # [N,H,N_OVER,C]
    spec_c = tf.complex(spec[:,:,:,0], spec[:,:,:,1])                                # [N,H,N_OVER]

            # ---- batch direct inverse DFT & overlap-add (avoid full IRFFT) ----
    # Precompute angle matrix [H, WINDOW_SAMPS]
    exp_range = tf.cast(tf.range(WINDOW_SAMPS), tf.float32)
    freq = tf.cast(tf.range(H), tf.float32)
    angle = tf.tensordot(freq, exp_range, axes=0) * (2 * np.pi / WINDOW_SAMPS)
    exp_coeffs = tf.exp(tf.complex(0.0, angle))  # complex64 [H, WINDOW_SAMPS]

    # Gather exponentials for each sample and segment
    # local: [N, N_OVER]
    exp_sel = tf.gather(exp_coeffs, local, axis=1)         # [H, N, N_OVER]
    exp_sel = tf.transpose(exp_sel, perm=[1, 0, 2])        # [N, H, N_OVER]

    # Compute inverse DFT at local sample positions
    # spec_c: [N, H, N_OVER]
    # Multiply and sum over freq
    seg_vals = tf.math.real(tf.reduce_sum(spec_c * exp_sel, axis=1)) / WINDOW_SAMPS  # [N, N_OVER]

    # Apply Hann window and sum overlap-add
    win_vals = tf.gather(hann_win, local)                   # [N, N_OVER]
    xs_vals = tf.reduce_sum(seg_vals * win_vals, axis=1) / scale_fac  # [N]
    return xs_vals

# ───── main ─────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()
    N = args.batch

    # ----- scalers (float32) -----
    global T_scale, T_mean
    with h5py.File(SCALER_PATH, "r") as hf:
        spec_scale = tf.constant(hf["spec_scale"][:].astype("float32"))
        spec_mean  = tf.constant(hf["spec_mean"] [:].astype("float32"))
        T_scale    = tf.constant(hf["T_scale"]   [:].astype("float32"))
        T_mean     = tf.constant(hf["T_mean"]    [:].astype("float32"))

    # ----- model weights -----
    model = load_model(MODEL_PATH, compile=False)
    dense0: Dense = model.layers[1]
    W0 = tf.constant(dense0.kernel.numpy().astype("float32"))  # shape (1,16)
    b0 = tf.constant(dense0.bias.numpy().astype("float32"))    # (16,)
    lrelu = model.layers[2]
    alpha = float(getattr(lrelu, "alpha", getattr(lrelu, "negative_slope", 0.3)))
    dense_dec: Dense = model.layers[3]
    W_dec = tf.constant(dense_dec.kernel.numpy().astype("float32"))
    b_dec = tf.constant(dense_dec.bias.numpy().astype("float32"))

    # ----- random test batch -----
    temps  = tf.random.uniform([N], 950., 1050., dtype=tf.float32)
    eidxs  = tf.random.uniform([N], 100, 45001, dtype=tf.int32)

    # ----- warm‑up (compile) -----
    xs_batch(temps[:1], eidxs[:1], W0, b0, alpha, W_dec, b_dec, spec_scale, spec_mean)

    # ----- timed run -----
    t0 = time.perf_counter()
    xs_vals = xs_batch(temps, eidxs, W0, b0, alpha, W_dec, b_dec, spec_scale, spec_mean)
    dt = time.perf_counter() - t0

    print(f"{N:,} points → {dt*1e3:.1f} ms   |   {dt/N*1e6:.1f} µs per point")

if __name__ == "__main__":
    main()
