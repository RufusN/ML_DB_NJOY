import argparse
import time
import h5py
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


def load_weights_and_scaler(table_path: str, scaler_path: str):
    """
    Build-time helper: load the full precomputed W_tab (16×E) and bias table,
    plus MLP parameters W0, b0, alpha, T_scale, T_mean from the original pipeline.
    """
    # Replace this stub with your actual loader of the latent-major table
    with h5py.File(table_path, 'r') as hf:
        W_tab   = hf['W_tab'][:]            # shape (16, n_E)
        b_tab   = hf['b_tab'][:]            # shape (n_E,)
        W0      = hf['W0'][:]               # shape (16,)
        b0      = hf['b0'][:]               # shape (16,)
        alpha   = float(hf['alpha'][()])    # scalar
        T_scale = hf['T_scale'][:]          # shape (1,) or scalar
        T_mean  = hf['T_mean'][:]           # shape (1,) or scalar
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean


def build_table(weights_path, scaler_path, window, step, out_path, *, dtype,
                quantize=0, quantize_residual=0, fallback_error=0.0005):
    # Load full table and MLP params
    W_tab, b_tab, W0, b0, alpha, T_scale, T_mean = load_weights_and_scaler(weights_path, scaler_path)
    n_E = W_tab.shape[1]

    # Stage 1: coarse k-means
    C1 = idx1 = None
    if quantize > 0:
        X = W_tab.T  # (n_E,16)
        km1 = KMeans(n_clusters=quantize, random_state=0).fit(X)
        C1   = km1.cluster_centers_.T   # (16, K1)
        idx1 = km1.labels_              # (n_E,)
    # Stage 2: residual k-means
    C2 = idx2 = None
    if C1 is not None and quantize_residual > 0:
        residuals = X - km1.cluster_centers_[idx1]
        km2 = KMeans(n_clusters=quantize_residual, random_state=0).fit(residuals)
        C2   = km2.cluster_centers_.T   # (16, K2)
        idx2 = km2.labels_              # (n_E,)

    # Compute representative hidden vector at T_mean
    T0 = np.array([T_mean]).reshape(-1)
    Tn = (T0 - T_mean) / T_scale
    h0 = np.maximum(0, Tn[:,None] * W0 + b0) + np.minimum(0, 0.01*(Tn[:,None] * W0 + b0))
    h0 = h0.reshape(-1)

    # True outputs and quantized outputs
    xs_true = h0 @ W_tab + b_tab
    if C1 is not None:
        Wq = C1[:, idx1]
        if C2 is not None:
            Wq += C2[:, idx2]
        xs_quant = h0 @ Wq + b_tab
    else:
        xs_quant = xs_true

    # Relative pointwise error
    rel_err = np.abs(xs_quant - xs_true) / (np.abs(xs_true) + 1e-12)
    fb_mask = rel_err > fallback_error

    # Fallback entries
    fb_idx = np.nonzero(fb_mask)[0].astype(np.int32)
    fb_W   = W_tab[:, fb_idx] if fb_idx.size else np.zeros((16,0), dtype=W_tab.dtype)
    fb_b   = b_tab[fb_idx]    if fb_idx.size else np.zeros((0,), dtype=b_tab.dtype)

    # Write compact HDF5
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('dtype', data=np.string_(dtype.__name__))
        hf.create_dataset('W0', data=W0, dtype=dtype)
        hf.create_dataset('b0', data=b0, dtype=dtype)
        hf.create_dataset('alpha', data=alpha)
        hf.create_dataset('T_scale', data=T_scale, dtype=dtype)
        hf.create_dataset('T_mean', data=T_mean, dtype=dtype)
        hf.create_dataset('b_tab_base', data=b_tab, dtype=dtype)
        hf.create_dataset('fb_idx', data=fb_idx, dtype='int32')
        hf.create_dataset('fb_W', data=fb_W, dtype=dtype)
        hf.create_dataset('fb_b', data=fb_b, dtype=dtype)
        if C1 is not None:
            hf.create_dataset('C1', data=C1, dtype=dtype)
            hf.create_dataset('idx1', data=idx1.astype('int32'), dtype='int32')
        if C2 is not None:
            hf.create_dataset('C2', data=C2, dtype=dtype)
            hf.create_dataset('idx2', data=idx2.astype('int32'), dtype='int32')


def query_xs(table_h5, T, E_idx):
    """
    Runtime: single-point lookup using either fallback weights or two-stage codebooks.
    """
    with h5py.File(table_h5, 'r') as hf:
        # Determine dtype
        dt = np.float16 if hf['dtype'][()].decode() == 'float16' else np.float32
        # Load MLP parameters
        W0_arr = hf['W0'][:]
        b0_arr = hf['b0'][:]
        alpha  = float(hf['alpha'][()])
        T_scale_arr = hf['T_scale'][:]
        T_mean_arr  = hf['T_mean'][:]
        # Full bias vector
        b_tab_np = hf['b_tab_base'][:]
        # Fallback entries
        fb_idx_np = hf['fb_idx'][:]
        fb_W_np   = hf['fb_W'][:]
        fb_b_np   = hf['fb_b'][:]
        # Codebooks
        C1_np = hf['C1'][:] if 'C1' in hf else None
        idx1_np = hf['idx1'][:] if 'C1' in hf else None
        C2_np = hf['C2'][:] if 'C2' in hf else None
        idx2_np = hf['idx2'][:] if 'C2' in hf else None

    # Convert to Tensors
    W0 = tf.constant(W0_arr, dtype=dt)
    b0 = tf.constant(b0_arr, dtype=dt)
    T_scale = tf.constant(T_scale_arr, dtype=dt)
    T_mean  = tf.constant(T_mean_arr, dtype=dt)
    b_tab   = tf.constant(b_tab_np, dtype=dt)
    # hidden layer
    Tt = tf.constant([T], dtype=dt)
    Tn = (Tt - T_mean) / T_scale
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(Tn,1), W0) + b0, alpha)

    # Determine if falling back
    pos = int(np.searchsorted(fb_idx_np, E_idx))
    if pos < fb_idx_np.size and fb_idx_np[pos] == E_idx:
        # exact fallback weight
        W_sel = tf.expand_dims(tf.constant(fb_W_np[:, pos], dtype=dt), 1)
        b_sel = tf.expand_dims(tf.constant(fb_b_np[pos], dtype=dt), 0)
    else:
        # quantized
        if C1_np is not None:
            w1 = tf.expand_dims(tf.constant(C1_np[:, idx1_np[E_idx]], dtype=dt), 1)
            if C2_np is not None:
                w2 = tf.expand_dims(tf.constant(C2_np[:, idx2_np[E_idx]], dtype=dt), 1)
                W_sel = w1 + w2
            else:
                W_sel = w1
        else:
            # no quantization available
            W_sel = tf.zeros((W0.shape[1],1), dtype=dt)
        # bias from full table
        b_sel = tf.expand_dims(tf.gather(b_tab, E_idx), 0)

    # final output dot
    xs = tf.matmul(hidden, W_sel) + b_sel
    return float(xs.numpy().squeeze())

def parse_range(spec, dtype=float):
    if ':' in spec:
        a,b,n = spec.split(':')
        return np.linspace(dtype(a), dtype(b), int(n), dtype=dtype)
    return np.array([dtype(x) for x in spec.split(',')], dtype=dtype)


def batch_xs(table, T_spec, E_spec, device, xla):
    Ts = parse_range(T_spec, float)
    Es = parse_range(E_spec, int)
    with h5py.File(table,'r') as hf:
        dt = np.float16 if hf['dtype'][()].decode()=='float16' else np.float32
        W0 = tf.constant(hf['W0'][:], dtype=dt)
        b0 = tf.constant(hf['b0'][:], dtype=dt)
        alpha = float(hf['alpha'][()])
        T_scale = tf.constant(hf['T_scale'][:], dtype=dt)
        T_mean = tf.constant(hf['T_mean'][:], dtype=dt)
        b_tab = tf.constant(hf['b_tab_base'][:], dtype=dt)
        fb_idx = hf['fb_idx'][:]
        fb_W   = tf.constant(hf['fb_W'][:], dtype=dt)
        fb_b   = tf.constant(hf['fb_b'][:], dtype=dt)
        has_C1 = 'C1' in hf
        if has_C1:
            C1 = tf.constant(hf['C1'][:], dtype=dt)
            idx1 = tf.constant(hf['idx1'][:], dtype=tf.int32)
        has_C2 = 'C2' in hf
        if has_C2:
            C2 = tf.constant(hf['C2'][:], dtype=dt)
            idx2 = tf.constant(hf['idx2'][:], dtype=tf.int32)

    with tf.device(f'/{device}:0'):
        if xla: tf.config.optimizer.set_jit(True)
        Tt = tf.constant(Ts, dtype=dt)
        Tn = (Tt - T_mean) / T_scale
        hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(Tn,1),W0) + b0, alpha)

        full_idx = []
        quant_idx = []
        for e in Es:
            pos = np.searchsorted(fb_idx, e)
            (full_idx if pos < fb_idx.size and fb_idx[pos]==e else quant_idx).append(e)

        parts = []
        if full_idx:
            parts.append((tf.gather(fb_W, full_idx, axis=1), tf.gather(fb_b, full_idx)))
        if quant_idx and has_C1:
            w1 = tf.gather(C1, tf.gather(idx1, quant_idx), axis=1)
            Wq = w1 + (tf.gather(C2, tf.gather(idx2, quant_idx), axis=1) if has_C2 else 0)
            parts.append((Wq, tf.gather(b_tab, quant_idx)))

        W_sel = tf.concat([p[0] for p in parts], axis=1)
        b_sel = tf.concat([p[1] for p in parts], axis=0)
        start = time.time()
        xs = tf.matmul(hidden, W_sel) + b_sel
        elapsed = (time.time()-start)*1000
        total = xs.shape[0]*xs.shape[1]
        print(f"[batch] ({xs.shape[0]},{xs.shape[1]}) on /{device}:0 in {elapsed:.1f} ms → {elapsed*1e3/total:.3f} µs/value")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd')

    pb = sp.add_parser('build')
    pb.add_argument('--weights', required=True)
    pb.add_argument('--scaler', required=True)
    pb.add_argument('--window', type=int, required=True)
    pb.add_argument('--step', type=int, required=True)
    pb.add_argument('--out', required=True)
    pb.add_argument('--dtype', choices=['float32','float16'], default='float32')
    pb.add_argument('--quantize', type=int, default=0)
    pb.add_argument('--quantize_residual', type=int, default=0)
    pb.add_argument('--fallback_error', type=float, default=0.001,
                    help='relative error cutoff for fallback')

    pq = sp.add_parser('query')
    pq.add_argument('--table', required=True)
    pq.add_argument('--T', type=float, required=True)
    pq.add_argument('--E_idx', type=int, required=True)

    pbatch = sp.add_parser('batch')
    pbatch.add_argument('--table', required=True)
    pbatch.add_argument('--T', required=True)
    pbatch.add_argument('--E', required=True)
    pbatch.add_argument('--device', choices=['CPU','GPU'], default='CPU')
    pbatch.add_argument('--xla', action='store_true')

    args = p.parse_args()
    if args.cmd == 'build':
        dt = np.float16 if args.dtype=='float16' else np.float32
        build_table(args.weights, args.scaler, args.window, args.step, args.out,
                    dtype=dt, quantize=args.quantize,
                    quantize_residual=args.quantize_residual,
                    fallback_error=args.fallback_error)
    elif args.cmd == 'query':
        print(query_xs(args.table,args.T,args.E_idx))
    elif args.cmd == 'batch':
        batch_xs(args.table,args.T,args.E,args.device,args.xla)
    else:
        p.print_help()

if __name__=='__main__':
    main()
