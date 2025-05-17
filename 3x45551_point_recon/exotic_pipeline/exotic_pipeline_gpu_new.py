#!/usr/bin/env python3
"""
pipeline_fp16.py – ultrafast half-precision inference (latent-major v2.1)
=========================================================================

This revision fixes compatibility with TF ≤ 2.15 by replacing the
`tf.experimental.sync_devices()` call with a small helper that
– if the symbol is absent – forces a device sync via `.numpy()`.

Everything else is identical to the v2 code you just tried.
"""
from __future__ import annotations
import argparse, time, h5py, tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

E_MIN = 1e4 * 1e-6  # MeV
E_MAX = 1e6 * 1e-6  # MeV
WINDOW_SIZE = 0.00004744
STEP_SIZE = 0.00002314
H, W_TIME, C = 3, 45551, 2
# Paths for full pipeline
BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
WEIGHTS_H5 = 'model_weights.h5'
SCALER_PATH = '3x45551_950_1050_spec_scalers.h5'
TEMP_DATA_DIR = r'/mnt/d/800_1200'


# ─────────────────────────────── graph-level opts ───────────────────────────
tf.config.optimizer.set_jit(True)
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)

# ───────────────────────────── GPU-resident constants ───────────────────────
W_tab = b_tab = W0 = b0 = T_scale = T_mean = None
alpha = 0.0

def initialize_table(table_h5: str, target_dtype: tf.dtypes.DType):
    """Load constant tensors onto the GPU (transpose W_tab → latent-major)."""
    global W_tab, b_tab, W0, b0, alpha, T_scale, T_mean
    with h5py.File(table_h5, "r") as hf, tf.device("/GPU:0"):
        cast = lambda d: tf.constant(hf[d][:], dtype=target_dtype)
        W_tab = tf.transpose(cast("W_tab"))           # (16, E_ROWS)
        b_tab = cast("b_tab")                         # (E_ROWS,)
        W0, b0 = cast("W0"), cast("b0")               # (1×16) layer
        T_scale, T_mean = cast("T_scale"), cast("T_mean")
        alpha = float(hf["alpha"][()])

# ───────────────────────────────── fused kernel ─────────────────────────────
@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float16),   # T_batch
        tf.TensorSpec([None], tf.int32)      # E_batch
    ],
    experimental_compile=True
)
def query_table(T_batch, E_batch):
    """Vectorised XS lookup for a batch of (T, E_idx) pairs."""
    # normalise + hidden layer ------------------------------------------------
    T_norm = (T_batch - T_mean) / T_scale                 # (N,)
    hidden = tf.nn.leaky_relu(
        tf.matmul(T_norm[:, None], W0) + b0, alpha        # (N,16)
    )

    # gather column slice in latent-major layout ------------------------------
    W_sel = tf.gather(W_tab, E_batch, axis=1)             # (16,N)
    b_sel = tf.gather(b_tab, E_batch)                     # (N,)

    # small dot-product on Tensor Cores --------------------------------------
    xs = tf.einsum("ij,ji->i", hidden, W_sel) + b_sel     # (N,)
    return xs

# ───────────── device-sync helper (works on any TF 2.x version) ─────────────
def device_sync(tensor: tf.Tensor):
    """
    Block the host thread until all pending kernels on the tensor’s device
    have finished.  Uses the official API if present, else falls back to
    a cheap `.numpy()` round-trip.
    """
    exp = getattr(tf.experimental, "sync_devices", None)
    if callable(exp):
        exp()
    else:
        # Older TF – copying a single scalar to host is <10 µs and forces sync
        _ = tf.reduce_sum(tensor)  # keep it tiny
        _ = _.numpy()

# ───────────────────────────────── benchmark ────────────────────────────────
def benchmark(table_h5: str, batch_size: int,
              tmin: float, tmax: float,
              precision_flag: str):
    # 1) precision policy ----------------------------------------------------
    mixed_precision.set_global_policy(precision_flag)
    compute_dtype = tf.float16 if precision_flag != "float32" else tf.float32

    # 2) load constants (once) ----------------------------------------------
    initialize_table(table_h5, compute_dtype)

    # 3) synthetic inputs on GPU --------------------------------------------
    with tf.device("/GPU:0"):
        T = tf.random.uniform([batch_size], tmin, tmax, dtype=compute_dtype)
        E = tf.random.uniform([batch_size], 0, W_tab.shape[1], dtype=tf.int32)

    # 4) one-time warm-up ----------------------------------------------------
    _ = query_table(T, E)

    # 5) timed run (properly synchronised) -----------------------------------
    start = time.perf_counter()
    xs = query_table(T, E)
    device_sync(xs)                                  # <<<  blocks here
    dur = time.perf_counter() - start

    print(f"{batch_size:,} points → {dur*1e6/batch_size:.4f} µs/point")
    return xs





def init_full():
    global fs, WINDOW_SAMPS, STEP_SAMPS, PAD, hann_win, scale_fac, SEG_OFF
    base_e, fs = load_base(E_MIN, E_MAX)
    WINDOW_SAMPS = int(WINDOW_SIZE * fs)
    STEP_SAMPS = int(STEP_SIZE  * fs)
    PAD = WINDOW_SAMPS
    hann_win = tf.constant(np.hanning(WINDOW_SAMPS).astype('float32'))
    scale_fac = tf.reduce_sum(hann_win**2) / STEP_SAMPS
    SEG_OFF = tf.range(WINDOW_SAMPS//STEP_SAMPS, dtype=tf.int32)
    return base_e
def analyse(base_e, reconstructed, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(temps[0].numpy(), base_e, file_dir)
    if padded is None:
        return
    xs_rec = reconstructed.numpy()
    idxs = eidxs.numpy()
    orig_vals = padded[idxs]
    rel_err = np.abs(xs_rec - orig_vals)/np.abs(orig_vals)*100
    # Plot original vs reconstructed point
    plt.figure(figsize=(8,5))
    plt.plot(padded_e, padded, label='Original Signal')
    plt.scatter(padded_e[idxs], xs_rec, c='r', s=10, label='Reconstructed')
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('Energy'); plt.ylabel('XS')
    plt.legend(); plt.grid(True)
    plt.savefig('./data_reconstruct/xs.png', dpi=200)
    plt.close()

    # Plot relative error (sorted)
    sorted_idx = np.argsort(idxs)
    idxs_sorted = idxs[sorted_idx]
    rel_err_sorted = rel_err[sorted_idx]
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs_sorted-PAD], rel_err_sorted, marker='o', linestyle='-')
    plt.xscale('log'); plt.xlabel('Energy Index'); plt.ylabel('Relative Error (%)')
    plt.title('Relative Error vs Energy'); plt.grid(True)
    plt.savefig('./data_reconstruct/relative_error.png', dpi=200)
    plt.close()
# ───────────────────────────────────── CLI ──────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8192,
                   help="Number of points per benchmark run")
    p.add_argument("--tmin", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=1000.0)
    p.add_argument("--precision",
                   choices=["float16", "mixed_float16", "float32"],
                   default="float16",
                   help="TensorFlow mixed-precision policy")
    p.add_argument("--table", default="w_table_compact.h5",
                   help="Pre-computed table HDF5 (E_ROWS × 16 weights)")
    args = p.parse_args()

    xs = benchmark(args.table, args.batch,
                  args.tmin, args.tmax,
                  args.precision)
    
        # accuracy
    if True:
        #(base_e, reconstructed, temps, eidxs, file_dir)
        #analyse(xs, temps, eidxs, base_e if args.backend=='full' else None,
        #                 PAD if args.backend=='full' else None,
        #                 TEMP_DATA_DIR)
        base_e = init_full()
        temps = tf.random.uniform([args.batch],  args.tmin, args.tmax, dtype=tf.int32)
        eidxs = tf.random.uniform([args.batch], 0, W_tab.shape[1], dtype=tf.int32)
        analyse(base_e, xs, temps, eidxs, TEMP_DATA_DIR)
