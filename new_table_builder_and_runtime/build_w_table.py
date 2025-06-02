# ──────────────────────────────────────────────────────────
# 2 ▸ build_runtime_deep.py
# ──────────────────────────────────────────────────────────
#!/usr/bin/env python3
"""
build_runtime_deep.py – identical CLI to your current build_runtime.py but now
supports an arbitrary‑depth encoder.  The latent‑major table is still the same
(shape LATENT × E_ROWS) so query speed for **each energy index** remains 1 matmul.
Any extra time comes from the three tiny Dense layers we execute per **batch**.
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
                out_h5: str = "w_table_deep.h5", dtype: str = "float32"):
    dt = np.float16 if dtype == "float16" else np.float32
    H, C, LATENT = 3, 2, 16  # unchanged

    with h5py.File(weights_h5, "r") as hf_w:
        # ---------- decoder (same as before) ----------
        W_dec = hf_w["W_dec"][:].astype(dt)
        b_dec = hf_w["b_dec"][:].astype(dt)
        full_width = W_dec.shape[1]

        # ---------- hidden stack ----------
        hidden_W, hidden_b, hidden_alpha = [], [], []
        idx = 0
        while f"W_hidden_{idx}" in hf_w:
            hidden_W.append(hf_w[f"W_hidden_{idx}"][:].astype(dt))
            hidden_b.append(hf_w[f"b_hidden_{idx}"][:].astype(dt))
            hidden_alpha.append(float(hf_w[f"alpha_{idx}"][()]))
            idx += 1
        # latent layer
        W_latent = hf_w["W_latent"][:].astype(dt)
        b_latent = hf_w["b_latent"][:].astype(dt)
        alpha_latent = float(hf_w["alpha_latent"][()])

    with h5py.File(scaler_h5, "r") as hf_s:
        spec_scale = hf_s["spec_scale"][:].astype(dt)
        spec_mean  = hf_s["spec_mean"] [:].astype(dt)
        T_scale    = hf_s["T_scale"]   [:].astype(dt)
        T_mean     = hf_s["T_mean"]    [:].astype(dt)

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step
    print(f"[build] rows={E_ROWS}, hidden_layers={len(hidden_W)+1}, dtype={dtype}")

    hann = np.hanning(window).astype(dt)
    scale_fac = np.sum(hann**2) / step

    # Build latent‑major table (unchanged) -------------------------------
    W_tab = np.zeros((LATENT, E_ROWS), dtype=dt)
    b_tab = np.zeros(E_ROWS,              dtype=dt)
    flat_stride_f = N_STFT * C

    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue
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
        W_tab[:, e_idx] = coeff.real
        b_tab[e_idx]    = bias.real

    # Write everything ----------------------------------------------------
    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"], hf["b_tab"] = W_tab, b_tab
        # store hidden stack
        for i, (W,b,a) in enumerate(zip(hidden_W, hidden_b, hidden_alpha)):
            hf[f"W_hidden_{i}"] = W; hf[f"b_hidden_{i}"] = b; hf[f"alpha_{i}"] = a
        hf["W_latent"], hf["b_latent"], hf["alpha_latent"] = W_latent, b_latent, alpha_latent
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["window"], hf["step"], hf["dtype"] = window, step, dtype
    print("[build] table + hidden stack saved ✔️")

# ═════════════════ runtime path ══════════════════════════════════════════

@tf.function
def _tf_query(T_batch, E_batch, W_tab, b_tab, hidden_W, hidden_b, hidden_alpha,
              W_latent, b_latent, alpha_latent, T_scale, T_mean, dt):
    # T_batch: (N,)   scalar input temperatures
    T_float = tf.cast(T_batch, tf.float32)
    T_cube  = tf.pow(T_float, 3.0)
    T_norm  = (T_cube - T_mean) / T_scale        # fp32 for precision
    if dt == tf.float16:
        T_norm = tf.cast(T_norm, tf.float16)
    x = T_norm[:, None]                          # shape (N,1)
    # ----- encoder stack -----
    for W,b,a in zip(hidden_W, hidden_b, hidden_alpha):
        x = tf.nn.leaky_relu(tf.matmul(x, W) + b, alpha=a)
    x = tf.nn.leaky_relu(tf.matmul(x, W_latent) + b_latent, alpha=alpha_latent)  # latent  (N,16)

    # ----- decode lookup row -----
    W_sel = tf.gather(W_tab, E_batch, axis=1)   # (16,|E|)
    b_sel = tf.gather(b_tab, E_batch)           # (|E|,)
    return tf.matmul(x, W_sel) + b_sel          # (N,|E|)


def query_xs(table_h5: str, T: float, E_idx: int):
    with h5py.File(table_h5, "r") as hf:
        dt = np.float16 if hf["dtype"].asstr()[()] == "float16" else np.float32
        tf_dt = tf.float16 if dt == np.float16 else tf.float32
        W_tab = tf.constant(hf["W_tab"][:], dtype=tf_dt)
        b_tab = tf.constant(hf["b_tab"][:], dtype=tf_dt)
        # hidden stack
        hidden_W, hidden_b, hidden_alpha = [], [], []
        idx = 0
        while f"W_hidden_{idx}" in hf:
            hidden_W.append(tf.constant(hf[f"W_hidden_{idx}"][:], dtype=tf_dt))
            hidden_b.append(tf.constant(hf[f"b_hidden_{idx}"][:], dtype=tf_dt))
            hidden_alpha.append(float(hf[f"alpha_{idx}"][()]))
            idx += 1
        W_latent = tf.constant(hf["W_latent"][:], dtype=tf_dt)
        b_latent = tf.constant(hf["b_latent"][:], dtype=tf_dt)
        alpha_latent = float(hf["alpha_latent"][()])
        T_scale = tf.constant(hf["T_scale"][:], dtype=tf.float32)
        T_mean  = tf.constant(hf["T_mean"] [:], dtype=tf.float32)

    val = _tf_query(tf.constant([T], dt), tf.constant([E_idx], tf.int32),
                    W_tab, b_tab, hidden_W, hidden_b, hidden_alpha,
                    W_latent, b_latent, alpha_latent,
                    T_scale, T_mean, tf_dt)
    return float(val.numpy()[0,0])

# ═════════════════ CLI (build/query/batch) identical to v1 ═══════════════

def _parse_range(arg: str, is_int: bool):
    if ":" in arg:
        start, stop, num = map(float, arg.split(":"))
        arr = np.linspace(start, stop, int(num))
    else:
        arr = np.fromstring(arg, sep=",")
    return arr.astype("int32" if is_int else "float32")


def batch_query(table_h5: str, T_arr, E_arr, use_gpu: bool, xla: bool):
    tf.config.optimizer.set_jit(xla)
    with h5py.File(table_h5, "r") as hf:
        dt = np.float16 if hf["dtype"].asstr()[()] == "float16" else np.float32
        tf_dt = tf.float16 if dt == np.float16 else tf.float32
        W_tab = tf.constant(hf["W_tab"][:], dtype=tf_dt)
        b_tab = tf.constant(hf["b_tab"][:], dtype=tf_dt)
        hidden_W, hidden_b, hidden_alpha = [], [], []
        idx = 0
        while f"W_hidden_{idx}" in hf:
            hidden_W.append(tf.constant(hf[f"W_hidden_{idx}"][:], dtype=tf_dt))
            hidden_b.append(tf.constant(hf[f"b_hidden_{idx}"][:], dtype=tf_dt))
            hidden_alpha.append(float(hf[f"alpha_{idx}"][()]))
            idx += 1
        W_latent = tf.constant(hf["W_latent"][:], dtype=tf_dt)
        b_latent = tf.constant(hf["b_latent"][:], dtype=tf_dt)
        alpha_latent = float(hf["alpha_latent"][()])
        T_scale = tf.constant(hf["T_scale"][:], dtype=tf.float32)
        T_mean  = tf.constant(hf["T_mean"] [:], dtype=tf.float32)

    device = "/GPU:0" if use_gpu and tf.config.list_logical_devices("GPU") else "/CPU:0"
    start = time.perf_counter()
    with tf.device(device):
        xs = _tf_query(tf.constant(T_arr, dt), tf.constant(E_arr, tf.int32),
                        W_tab, b_tab, hidden_W, hidden_b, hidden_alpha,
                        W_latent, b_latent, alpha_latent,
                        T_scale, T_mean, tf_dt)
        xs_np = xs.numpy()
    dur = time.perf_counter() - start
    rate_ns = dur / xs_np.size * 1e9
    unit = "ns/value" if rate_ns < 100 else "µs/value"
    rate = rate_ns if unit == "ns/value" else rate_ns / 1e3
    print(f"[batch] {xs_np.shape} on {device} → {rate:.2f} {unit}")
    return xs_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--weights", required=True)
    p_build.add_argument("--scaler",  required=True)
    p_build.add_argument("--window",  type=int, required=True)
    p_build.add_argument("--step",    type=int, required=True)
    p_build.add_argument("--out",     default="w_table_deep.h5")
    p_build.add_argument("--dtype",   choices=["float32","float16"], default="float32")

    p_query = sub.add_parser("query")
    p_query.add_argument("--T",     type=float, required=True)
    p_query.add_argument("--E_idx", type=int,   required=True)
    p_query.add_argument("--table", default="w_table_deep.h5")

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--T")
    p_batch.add_argument("--E")
    p_batch.add_argument("--table",  default="w_table_deep.h5")
    p_batch.add_argument("--device", choices=["CPU","GPU"], default="GPU")
    p_batch.add_argument("--xla", action="store_true")

    args = parser.parse_args()

    if args.cmd == "build":
        build_table(args.weights, args.scaler, args.window, args.step,
                    out_h5=args.out, dtype=args.dtype)
    elif args.cmd == "query":
        val = query_xs(args.table, args.T, args.E_idx)
        print(f"XS(T={args.T}, E_idx={args.E_idx}) = {val:.8e}")
    else:
        T_arr = _parse_range(args.T, False)
        E_arr = _parse_range(args.E, True)
        batch_query(args.table, T_arr, E_arr,
                    use_gpu=(args.device=="GPU"), xla=args.xla)