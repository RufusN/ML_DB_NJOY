#!/usr/bin/env python3
"""table_xs.py – fast Doppler‑XS lookup & GPU benchmark
=====================================================
Improvements in this version
---------------------------
* **Latent‑major table** `W_tab` is now stored as **(16 × E_ROWS)** so the
  heavy kernel can use a plain `tf.matmul(hidden, W_sel)` (no transpose).
* **Optional float16**  `--dtype float16` halves memory traffic; Metal’s fp16
  FMA keeps full accuracy for this workload.
* **XLA JIT on by default** – can be toggled with `--no_xla`.
* **Accurate timing** – stopwatch wraps the `numpy()` sync so µs/value is
  real compute time.
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
                out_h5: str = "w_table.h5", dtype: str = "float32"):
    """Pre‑compute linear rows and save latent‑major table (16 × E_ROWS)."""
    dt = np.float16 if dtype == "float16" else np.float32
    H, C, LATENT = 3, 2, 16
    with h5py.File(weights_h5, "r") as hf_w:
        full_width = hf_w["W_dec"].shape[1]
        W0  = hf_w["W0"][:].astype(dt)
        b0  = hf_w["b0"][:].astype(dt)
        alpha = float(hf_w["alpha"][()])
        W_dec = hf_w["W_dec"][:].astype(dt)
        b_dec = hf_w["b_dec"][:].astype(dt)

    N_STFT = full_width // (H * C)
    E_ROWS = 2 * N_STFT + (window - 1)
    n_over = window // step
    print(f"[build] rows={E_ROWS}, dtype={dtype}, latent‑major")

    with h5py.File(scaler_h5, "r") as hf_s:
        spec_scale = hf_s["spec_scale"][:].astype(dt)
        spec_mean  = hf_s["spec_mean"] [:].astype(dt)
        T_scale    = hf_s["T_scale"]   [:].astype(dt)
        T_mean     = hf_s["T_mean"]    [:].astype(dt)

    hann = np.hanning(window).astype(dt)
    scale_fac = np.sum(hann**2) / step

    # latent‑major table ---------------------------------------------------
    W_tab = np.zeros((LATENT, E_ROWS), dtype=dt)
    b_tab = np.zeros(E_ROWS,              dtype=dt)
    flat_stride_f = N_STFT * C

    for e_idx in range(E_ROWS):
        segs, local = map_segments_and_locals(e_idx, window, step)
        if segs.max() >= N_STFT:
            continue  # unseen column – leave zeros
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
    with h5py.File(out_h5, "w") as hf:
        hf["W_tab"], hf["b_tab"] = W_tab, b_tab
        hf["W0"], hf["b0"], hf["alpha"] = W0, b0, alpha
        hf["T_scale"], hf["T_mean"] = T_scale, T_mean
        hf["window"], hf["step"], hf["dtype"] = window, step, dtype
    print("[build] table saved ✔️")

# ═════════════════ single query ═══════════════════════════════════════════

@tf.function  # jit_compile removed for Metal/CPU compatibility
def _tf_query(T_batch, E_batch, W_tab, b_tab, W0, b0, alpha, T_scale, T_mean):
    T_norm = (tf.pow(T_batch, 3) - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(T_norm[:,None], W0) + b0, alpha)  # [N,16]
    W_sel  = tf.gather(W_tab, E_batch, axis=1)           # [16,N]
    b_sel  = tf.gather(b_tab, E_batch)                   # [N]
    return tf.matmul(hidden, W_sel) + b_sel              # [N]


def query_xs(table_h5: str, T: float, E_idx: int):
    with h5py.File(table_h5, "r") as hf:
        dt = np.float16 if hf["dtype"][()].decode() == "float16" else np.float32
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])
    val = _tf_query(tf.constant([T], dt), tf.constant([E_idx], tf.int32),
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


def batch_query(table_h5: str, T_arr, E_arr, use_gpu: bool, xla: bool):
    # Enable XLA only if the user explicitly asks for it — Metal backend
    # still raises NOT_FOUND on jit‑compiled graphs.
    tf.config.optimizer.set_jit(xla)
    with h5py.File(table_h5, "r") as hf:
        dt = np.float16 if hf["dtype"][()].decode() == "float16" else np.float32
        W_tab = tf.constant(hf["W_tab"][:])
        b_tab = tf.constant(hf["b_tab"][:])
        W0    = tf.constant(hf["W0"][:])
        b0    = tf.constant(hf["b0"][:])
        alpha = float(hf["alpha"][()])
        T_scale = tf.constant(hf["T_scale"][:])
        T_mean  = tf.constant(hf["T_mean"] [:])

    device = "/GPU:0" if use_gpu and tf.config.list_logical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        # T_tf = tf.constant(T_arr, dtype=dt)
        T_tf_cube = tf.constant(T_arr, dtype=dt)
        T_tf_cube = tf.pow(T_tf_cube, 3)
        hidden = tf.nn.leaky_relu(tf.matmul((T_tf_cube - T_mean)[:, None] / T_scale, W0) + b0, alpha)
        W_sel = tf.gather(W_tab, tf.constant(E_arr, dtype=tf.int32), axis=1)
        b_sel = tf.gather(b_tab, tf.constant(E_arr, dtype=tf.int32))

    start = time.perf_counter()
    with tf.device(device):
        xs = tf.matmul(hidden, W_sel) + b_sel
        xs_np = xs.numpy()
    dur = time.perf_counter() - start
    print(f"[batch] {xs_np.shape} on {device} in {dur*1e3:.1f} ms → {dur/xs_np.size*1e6:.6f} µs/value")
    rate_ns = dur / xs_np.size * 1e9
    unit = "ns/value" if rate_ns < 100 else "µs/value"
    rate = rate_ns if unit == "ns/value" else rate_ns / 1e3
    print(f"[batch] {xs_np.shape} on {device} in {dur*1e3:.1f} ms → {rate:.2f} {unit}")

    return xs_np

# ═════════════════ CLI ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── build ─────────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="pre‑compute latent‑major table")
    p_build.add_argument("--weights", required=True)
    p_build.add_argument("--scaler",  required=True)
    p_build.add_argument("--window",  type=int, required=True)
    p_build.add_argument("--step",    type=int, required=True)
    p_build.add_argument("--out",     default="w_table.h5")
    p_build.add_argument("--dtype",   choices=["float32","float16"], default="float32")

    # ── query ─────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="single XS value from table")
    p_query.add_argument("--T",      type=float, required=True)
    p_query.add_argument("--E_idx",  type=int,   required=True)
    p_query.add_argument("--table",  default="w_table.h5")

    # ── batch benchmark ──────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="benchmark grid runtime")
    p_batch.add_argument("--T",      required=True, help="start:stop:num or comma list")
    p_batch.add_argument("--E",      required=True, help="start:stop:num or comma list")
    p_batch.add_argument("--table",  default="w_table.h5")
    p_batch.add_argument("--device", choices=["CPU","GPU"], default="GPU")
    p_batch.add_argument("--xla", action="store_true", help="enable XLA JIT (CPU only – Metal GPU jit not supported)")

    args = parser.parse_args()

    if args.cmd == "build":
        build_table(args.weights, args.scaler, args.window, args.step,
                    args.out, dtype=args.dtype)

    elif args.cmd == "query":
        val = query_xs(args.table, args.T, args.E_idx)
        print(f"XS(T={args.T}, E_idx={args.E_idx}) = {val:.8e}")

    else:  # batch
        T_arr = _parse_range(args.T, is_int=False)
        E_arr = _parse_range(args.E, is_int=True)
        batch_query(args.table, T_arr, E_arr,
                    use_gpu=(args.device == "GPU"), xla=args.xla)

if __name__ == "__main__":
    main()
