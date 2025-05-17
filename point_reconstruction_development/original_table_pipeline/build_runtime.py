#!/usr/bin/env python3
"""table_xs.py – build a pre‑baked table and measure runtime
============================================================
Sub‑commands
------------
* **build**  – create `w_table.h5` from network weights & scaler.
* **query**  – return one XS value.
* **batch**  – benchmark an `N_T × N_E` grid on CPU or GPU; prints
  wall‑clock time and average µs/value.

Example benchmark (GPU)
----------------------
```bash
python table_xs.py batch \
    --T 300:1500:128 \
    --E 0:91000:4096 \
    --device GPU
```
"""
from __future__ import annotations
import argparse, time, h5py, numpy as np, tensorflow as tf

# ═════════════════ helper ════════════════════════════════════════════════

def map_segments_and_locals(e_idx: int, window: int, step: int):
    n_over = window // step
    first  = int(np.ceil((e_idx - window) / step - 0.5))
    segs   = np.arange(first, first + n_over, dtype=np.int32)
    local  = (e_idx - (segs + 1) * step) % window
    return segs, local

# ═════════════════ builder ═══════════════════════════════════════════════

def build_table(weights_h5: str, scaler_h5: str, window: int, step: int,
                out_h5: str = "w_table.h5"):
    H, C, LATENT = 3, 2, 16
    with h5py.File(weights_h5, "r") as hf_w:
        full_width = hf_w["W_dec"].shape[1]
        W0  = hf_w["W0"][:].astype("float32")
        b0  = hf_w["b0"][:].astype("float32")
        alpha = float(hf_w["alpha"][()])
        W_dec = hf_w["W_dec"][:].astype("float32")
        b_dec = hf_w["b_dec"][:].astype("float32")

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step
    print(f"[build] window={window}, step={step}, rows={E_ROWS}")

    with h5py.File(scaler_h5, "r") as hf_s:
        spec_scale = hf_s["spec_scale"][:].astype("float32")
        spec_mean  = hf_s["spec_mean"] [:].astype("float32")
        T_scale    = hf_s["T_scale"]   [:].astype("float32")
        T_mean     = hf_s["T_mean"]    [:].astype("float32")

    hann = np.hanning(window).astype("float32")
    scale_fac = np.sum(hann**2) / step

    W_tab = np.zeros((E_ROWS, LATENT), dtype="float32")
    b_tab = np.zeros(E_ROWS,              dtype="float32")
    flat_stride_f = N_STFT * C

    t0 = time.perf_counter()
    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue  # unseen STFT column – leave zeros
        flat_idx = []
        for f in range(H):
            for s in segs:
                base = f * flat_stride_f + s * C
                flat_idx.extend((base, base + 1))
        flat_idx = np.asarray(flat_idx, np.int32)
        W_sub = W_dec[:, flat_idx] * spec_scale[flat_idx]
        b_sub = b_dec[ flat_idx]  * spec_scale[flat_idx] + spec_mean[flat_idx]
        W_c = W_sub.reshape(LATENT, H, n_over, 2)
        b_c = b_sub.reshape(       H, n_over, 2)
        W_c = (W_c[...,0] + 1j*W_c[...,1]).transpose(0,2,1)
        b_c = (b_c[...,0] + 1j*b_c[...,1]).transpose(1,0)
        seg_W = np.fft.irfft(W_c, n=window, axis=-1) * hann
        seg_b = np.fft.irfft(b_c, n=window, axis=-1) * hann
        coeff = seg_W[np.arange(LATENT)[:,None], np.arange(n_over), local].sum(axis=1) / scale_fac
        bias  = seg_b[np.arange(n_over),             local].sum()            / scale_fac
        W_tab[e_idx] = coeff.real
        b_tab[e_idx] = bias.real
        if e_idx % 10000 == 0:
            print(f"  • {e_idx}/{E_ROWS} rows done")
    print(f"[build] finished in {time.perf_counter()-t0:.1f}s → {out_h5}")

    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"], hf["b_tab"] = W_tab, b_tab
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["window"], hf["step"] = window, step

# ═════════════════ single query ═══════════════════════════════════════════

@tf.function
def _tf_query(T_batch, E_batch, W_tab, b_tab, W0, b0, alpha, T_scale, T_mean):
    T_norm = (T_batch - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:,None], W0) + b0, alpha)
    W_vec  = tf.gather(W_tab, E_batch)
    b_vec  = tf.gather(b_tab, E_batch)
    return tf.reduce_sum(hidden * W_vec, axis=-1) + b_vec


def query_xs(table_h5: str, T: float, E_idx: int):
    with h5py.File(table_h5, "r") as hf:
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])
    val = _tf_query(tf.constant([T]), tf.constant([E_idx]),
                    W_tab, b_tab, W0, b0, alpha, T_scale, T_mean)
    return float(val.numpy()[0])

# ═════════════════ batch benchmark ════════════════════════════════════════

def _parse_range(arg: str, is_int: bool):
    if ":" in arg:
        start, stop, num = map(float, arg.split(":"))
        arr = np.linspace(start, stop, int(num))
    else:
        arr = np.fromstring(arg, sep=",")
    return arr.astype("int32" if is_int else "float32")


def batch_query(table_h5: str, T_arr: np.ndarray, E_arr: np.ndarray, use_gpu: bool):
    with h5py.File(table_h5, "r") as hf:
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])

    device = "/GPU:0" if use_gpu and tf.config.list_logical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        T_tf = tf.constant(T_arr)
        hidden = tf.nn.leaky_relu(tf.matmul(((T_tf - T_mean)/T_scale)[:,None], W0) + b0, alpha)
        W_sel = tf.gather(W_tab, tf.constant(E_arr, dtype=tf.int32))
        b_sel = tf.gather(b_tab, tf.constant(E_arr, dtype=tf.int32))

    start = time.perf_counter()

    with tf.device(device):
        xs = tf.matmul(hidden, W_sel, transpose_b=True) + b_sel
        xs_np = xs.numpy()                     # ← forces GPU → CPU sync

    dur = time.perf_counter() - start          # stop after sync completes
    print(f"[batch] {xs_np.shape} on {device} in {dur*1e3:.6f} ms "
        f"→ {dur/xs_np.size*1e6:.6f} µs/value")
    return xs_np

    #     start = time.perf_counter()
    #     xs = tf.matmul(hidden, W_sel, transpose_b=True) + b_sel
    #     dur = time.perf_counter() - start
    # xs_np = xs.numpy()
    # print(f"[batch] {xs_np.shape} on {device} in {dur*1e3:.1f} ms → {dur/xs_np.size*1e6:.8f} µs/value")
    # return xs.numpy()

# ═════════════════ CLI ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--weights", required=True)
    p_build.add_argument("--scaler", required=True)
    p_build.add_argument("--window", type=int, required=True)
    p_build.add_argument("--step",   type=int, required=True)
    p_build.add_argument("--out",    default="w_table.h5")

    p_query = sub.add_parser("query")
    p_query.add_argument("--T",      type=float, required=True)
    p_query.add_argument("--E_idx",  type=int,   required=True)
    p_query.add_argument("--table",  default="w_table.h5")

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--T",      required=True, help="start:stop:num or list")
    p_batch.add_argument("--E",      required=True, help="start:stop:num or list")
    p_batch.add_argument("--table",  default="w_table.h5")
    p_batch.add_argument("--device", choices=["CPU","GPU"], default="CPU")

    args = parser.parse_args()
    if args.cmd == "build":
        build_table(args.weights, args.scaler, args.window, args.step, args.out)
    elif args.cmd == "query":
        val = query_xs(args.table, args.T, args.E_idx)
        print(f"XS(T={args.T}, E_idx={args.E_idx}) = {val:.8e}")
    else:  # batch
        T_arr = _parse_range(args.T, is_int=False)
        E_arr = _parse_range(args.E, is_int=True)
        batch_query(args.table, T_arr, E_arr, args.device == "GPU")

if __name__ == "__main__":
    main()
