import argparse
import time
import h5py
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


def load_weights_and_scaler(table_path: str, _: str):
    # Load base table and MLP params
    with h5py.File(table_path, 'r') as hf:
        W_tab   = hf['W_tab'][:]            # [16, E_rows]
        b_tab   = hf['b_tab'][:]            # [E_rows]
        W0      = hf['W0'][:]               # [16]
        b0      = hf['b0'][:]               # [16]
        alpha   = float(hf['alpha'][()])    # scalar
        T_scale = hf['T_scale'][:]          # [E_rows] or scalar
        T_mean  = hf['T_mean'][:]           # [E_rows] or scalar
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean


def build_table(weights_path, scaler_path, window, step, out_path, *, dtype,
                quantize=0, quantize_residual=0, fallback_error=0.001):
    # Load full table
    W_tab, b_tab, W0, b0, alpha, T_scale, T_mean = load_weights_and_scaler(weights_path, scaler_path)
    E_rows = W_tab.shape[1]

    # Prepare fallback mask
    fallback_mask = np.zeros(E_rows, dtype=bool)

    # Stage 1: coarse quantization
    C1 = idx1 = C2 = idx2 = None
    if quantize > 0:
        X = W_tab.T  # (E_rows,16)
        km1 = KMeans(n_clusters=quantize, random_state=0).fit(X)
        C1 = km1.cluster_centers_.T
        idx1 = km1.labels_
        # Stage 2: residual quantization
        if quantize_residual > 0:
            residuals = X - km1.cluster_centers_[idx1]
            km2 = KMeans(n_clusters=quantize_residual, random_state=0).fit(residuals)
            C2 = km2.cluster_centers_.T
            idx2 = km2.labels_
            # fallback error mask
            recon = km1.cluster_centers_[idx1] + km2.cluster_centers_[idx2]
            rel_err = np.linalg.norm(recon - X, axis=1) / (np.linalg.norm(X, axis=1) + 1e-12)
            fallback_mask = rel_err > fallback_error

    # Store only fallback entries
    fb_indices = np.nonzero(fallback_mask)[0].astype(np.int32)
    fb_W = W_tab[:, fb_indices] if fb_indices.size else np.zeros((W_tab.shape[0], 0), dtype=W_tab.dtype)
    fb_b = b_tab[fb_indices]       if fb_indices.size else np.zeros((0,), dtype=b_tab.dtype)

    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('dtype', data=np.string_(dtype.__name__))
        hf.create_dataset('W0', data=W0, dtype=dtype)
        hf.create_dataset('b0', data=b0, dtype=dtype)
        hf.create_dataset('alpha', data=alpha)
        hf.create_dataset('T_scale', data=T_scale, dtype=dtype)
        hf.create_dataset('T_mean', data=T_mean, dtype=dtype)
        hf.create_dataset('b_tab_base', data=b_tab, dtype=dtype)
        hf.create_dataset('fb_idx', data=fb_indices, dtype='int32')
        hf.create_dataset('fb_W', data=fb_W, dtype=dtype)
        hf.create_dataset('fb_b', data=fb_b, dtype=dtype)
        if C1 is not None:
            hf.create_dataset('C1', data=C1, dtype=dtype)
            hf.create_dataset('idx1', data=idx1.astype('int32'), dtype='int32')
        if C2 is not None:
            hf.create_dataset('C2', data=C2, dtype=dtype)
            hf.create_dataset('idx2', data=idx2.astype('int32'), dtype='int32')


def query_xs(table_h5, T, E_idx):
    with h5py.File(table_h5, 'r') as hf:
        dt = np.float16 if hf['dtype'][()].decode() == 'float16' else np.float32
        # Load MLP parameters
        W0 = tf.constant(hf['W0'][:], dtype=dt)
        b0 = tf.constant(hf['b0'][:], dtype=dt)
        alpha = float(hf['alpha'][()])
        T_scale = tf.constant(hf['T_scale'][:], dtype=dt)
        T_mean = tf.constant(hf['T_mean'][:], dtype=dt)
        # Load bias base and fallback
        b_tab_base = tf.constant(hf['b_tab_base'][:], dtype=dt)
        fb_idx = hf['fb_idx'][:]
        fb_W = tf.constant(hf['fb_W'][:], dtype=dt)
        fb_b = tf.constant(hf['fb_b'][:], dtype=dt)

        # Compute hidden
        Tt = tf.constant([T], dt)
        Tn = (Tt - T_mean) / T_scale
        hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(Tn, 1), W0) + b0, alpha)

        # Determine if fallback
        pos = np.searchsorted(fb_idx, E_idx)
        if pos < fb_idx.size and fb_idx[pos] == E_idx:
            W_sel = tf.expand_dims(fb_W[:, pos], 1)
            b_sel = tf.expand_dims(fb_b[pos], 0)
        else:
            # Quantized lookup
            if 'C1' in hf:
                C1 = tf.constant(hf['C1'][:], dtype=dt)
                id1 = int(hf['idx1'][E_idx])
                w1 = tf.expand_dims(tf.gather(C1, id1, axis=1)[:, 0], 1)
                if 'C2' in hf:
                    C2 = tf.constant(hf['C2'][:], dtype=dt)
                    id2 = int(hf['idx2'][E_idx])
                    w2 = tf.expand_dims(tf.gather(C2, id2, axis=1)[:, 0], 1)
                    W_sel = w1 + w2
                else:
                    W_sel = w1
            else:
                # No quantization available, fallback to bias only
                W_sel = tf.zeros((W0.shape[1], 1), dtype=dt)
            b_sel = tf.expand_dims(tf.gather(b_tab_base, E_idx), 0)

    xs = tf.matmul(hidden, W_sel) + b_sel
    return float(xs.numpy().squeeze())


def parse_range(spec, dtype=float):
    if ':' in spec:
        a, b, n = spec.split(':')
        return np.linspace(dtype(a), dtype(b), int(n), dtype=dtype)
    return np.array([dtype(x) for x in spec.split(',')], dtype=dtype)


def batch_xs(table, T_spec, E_spec, device, xla):
    Ts = parse_range(T_spec, float)
    Es = parse_range(E_spec, int)
    with h5py.File(table, 'r') as hf:
        dt = np.float16 if hf['dtype'][()].decode() == 'float16' else np.float32
        W0 = tf.constant(hf['W0'][:], dtype=dt)
        b0 = tf.constant(hf['b0'][:], dtype=dt)
        alpha = float(hf['alpha'][()])
        T_scale = tf.constant(hf['T_scale'][:], dtype=dt)
        T_mean = tf.constant(hf['T_mean'][:], dtype=dt)
        b_tab_base = tf.constant(hf['b_tab_base'][:], dtype=dt)
        fb_idx = hf['fb_idx'][:]
        fb_W = tf.constant(hf['fb_W'][:], dtype=dt)
        fb_b = tf.constant(hf['fb_b'][:], dtype=dt)
        has_C1 = 'C1' in hf
        has_C2 = 'C2' in hf
        if has_C1:
            C1 = tf.constant(hf['C1'][:], dtype=dt)
            idx1 = tf.constant(hf['idx1'][:], dtype=tf.int32)
        if has_C2:
            C2 = tf.constant(hf['C2'][:], dtype=dt)
            idx2 = tf.constant(hf['idx2'][:], dtype=tf.int32)

    with tf.device(f'/{device}:0'):
        if xla:
            tf.config.optimizer.set_jit(True)
        Tt = tf.constant(Ts, dtype=dt)
        Tn = (Tt - T_mean) / T_scale
        hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(Tn, 1), W0) + b0, alpha)

        # Build lookup
        full_indices = []
        quant_indices = []
        for e in Es:
            pos = np.searchsorted(fb_idx, e)
            if pos < fb_idx.size and fb_idx[pos] == e:
                full_indices.append(pos)
            else:
                quant_indices.append(e)

        parts = []
        if full_indices:
            W_full = tf.gather(fb_W, full_indices, axis=1)
            b_full = tf.gather(fb_b, full_indices)
            parts.append((W_full, b_full))
        if quant_indices and has_C1:
            id1s = tf.gather(idx1, quant_indices)
            W1 = tf.gather(C1, id1s, axis=1)
            if has_C2:
                id2s = tf.gather(idx2, quant_indices)
                W2 = tf.gather(C2, id2s, axis=1)
                W_quant = W1 + W2
            else:
                W_quant = W1
            b_quant = tf.gather(b_tab_base, quant_indices)
            parts.append((W_quant, b_quant))

        # Concatenate all parts in the order they were discovered
        W_sel = tf.concat([w for w, _ in parts], axis=1)
        b_sel = tf.concat([tf.expand_dims(b, 0) if len(b.shape)==0 else b for _, b in parts], axis=0)

        start = time.time()
        xs = tf.matmul(hidden, W_sel) + b_sel
        elapsed = (time.time() - start) * 1000
        total = xs.shape[0] * xs.shape[1]
        print(f"[batch] ({xs.shape[0]},{xs.shape[1]}) on /{device}:0 in {elapsed:.1f} ms → {elapsed*1e3/total:.6f} µs/value")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd')

    pb = sp.add_parser('build')
    pb.add_argument('--weights', required=True)
    pb.add_argument('--scaler', required=True)
    pb.add_argument('--window', type=int, required=True)
    pb.add_argument('--step', type=int, required=True)
    pb.add_argument('--out', required=True)
    pb.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
    pb.add_argument('--quantize', type=int, default=0)
    pb.add_argument('--quantize_residual', type=int, default=0)
    pb.add_argument('--fallback_error', type=float, default=0.001,
                    help='relative error cutoff for fallback (0.001=0.1%)')

    pq = sp.add_parser('query')
    pq.add_argument('--table', required=True)
    pq.add_argument('--T', type=float, required=True)
    pq.add_argument('--E_idx', type=int, required=True)

    pbatch = sp.add_parser('batch')
    pbatch.add_argument('--table', required=True)
    pbatch.add_argument('--T', required=True)
    pbatch.add_argument('--E', required=True)
    pbatch.add_argument('--device', choices=['CPU', 'GPU'], default='CPU')
    pbatch.add_argument('--xla', action='store_true')

    args = p.parse_args()
    if args.cmd == 'build':
        dt = np.float16 if args.dtype == 'float16' else np.float32
        build_table(args.weights, args.scaler, args.window, args.step, args.out,
                    dtype=dt,
                    quantize=args.quantize,
                    quantize_residual=args.quantize_residual,
                    fallback_error=args.fallback_error)
    elif args.cmd == 'query':
        print(query_xs(args.table, args.T, args.E_idx))
    elif args.cmd == 'batch':
        batch_xs(args.table, args.T, args.E, args.device, args.xla)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
