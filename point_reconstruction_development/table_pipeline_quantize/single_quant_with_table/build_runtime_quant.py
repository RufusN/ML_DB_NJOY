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
    with h5py.File(table_path, 'r') as hf:
        W_tab   = hf['W_tab'][:]            # (16, n_E)
        b_tab   = hf['b_tab'][:]            # (n_E,)
        W0      = hf['W0'][:]               # (16,)
        b0      = hf['b0'][:]               # (16,)
        alpha   = float(hf['alpha'][()])    # scalar
        T_scale = hf['T_scale'][:]          # scalar or (1,)
        T_mean  = hf['T_mean'][:]           # scalar or (1,)
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean


def build_table(weights_path, scaler_path, window, step, out_path, *, dtype,
                quantize=0, fallback_error=0.001):
    # Load full table and MLP params
    W_tab, b_tab, W0, b0, alpha, T_scale, T_mean = load_weights_and_scaler(weights_path, scaler_path)
    n_E = W_tab.shape[1]

    # Single-stage k-means
    C1 = idx1 = None
    if quantize > 0:
        X = W_tab.T  # (n_E,16)
        km = KMeans(n_clusters=quantize, random_state=0).fit(X)
        C1   = km.cluster_centers_.T   # (16, K)
        idx1 = km.labels_              # (n_E,)

    # Representative hidden at T_mean
    T0 = np.atleast_1d(T_mean)
    Tn = (T0 - T_mean) / T_scale
    h0 = (np.maximum(0, Tn[:,None] * W0 + b0)
          + np.minimum(0, 0.01*(Tn[:,None] * W0 + b0))).reshape(-1)

    # True and quant outputs
    xs_true = h0 @ W_tab + b_tab
    if C1 is not None:
        Wq = C1[:, idx1]
        xs_quant = h0 @ Wq + b_tab
    else:
        xs_quant = xs_true

    # Relative error and fallback mask
    rel_err = np.abs(xs_quant - xs_true) / (np.abs(xs_true) + 1e-12)
    fb_mask = rel_err > fallback_error
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


def query_xs(table_h5, T, E_idx):
    # Runtime: single-point lookup with optional fallback or single-stage codebook
    with h5py.File(table_h5, 'r') as hf:
        dt = np.float16 if hf['dtype'][()].decode()=='float16' else np.float32
        # Load MLP params
        W0_arr = hf['W0'][:]; b0_arr = hf['b0'][:]; alpha = float(hf['alpha'][()])
        T_scale_arr = hf['T_scale'][:]; T_mean_arr  = hf['T_mean'][:]
        # Bias and fallback
        b_tab_np  = hf['b_tab_base'][:]
        fb_idx_np = hf['fb_idx'][:]; fb_W_np = hf['fb_W'][:]; fb_b_np = hf['fb_b'][:]
        # Codebook
        C1_np  = hf['C1'][:]  if 'C1' in hf else None
        idx1_np = hf['idx1'][:] if 'C1' in hf else None

    # Tensors
    W0 = tf.constant(W0_arr, dtype=dt); b0 = tf.constant(b0_arr, dtype=dt)
    Tt = tf.constant([T], dtype=dt); T_scale = tf.constant(T_scale_arr, dtype=dt); T_mean = tf.constant(T_mean_arr, dtype=dt)
    hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims((Tt-T_mean)/T_scale,1), W0) + b0, alpha)
    b_tab = tf.constant(b_tab_np, dtype=dt)

    # Fallback or quant
    pos = int(np.searchsorted(fb_idx_np, E_idx))
    if pos<fb_idx_np.size and fb_idx_np[pos]==E_idx:
        W_sel = tf.expand_dims(tf.constant(fb_W_np[:,pos], dtype=dt),1)
        b_sel = tf.expand_dims(tf.constant(fb_b_np[pos], dtype=dt),0)
    else:
        if C1_np is not None:
            W_sel = tf.expand_dims(tf.constant(C1_np[:, idx1_np[E_idx]], dtype=dt),1)
        else:
            W_sel = tf.zeros((W0.shape[1],1), dtype=dt)
        b_sel = tf.expand_dims(tf.gather(b_tab, E_idx),0)

    xs = tf.matmul(hidden, W_sel) + b_sel
    return float(xs.numpy().squeeze())


def parse_range(spec, dtype=float):
    if ':' in spec:
        a,b,n = spec.split(':'); return np.linspace(dtype(a),dtype(b),int(n),dtype=dtype)
    return np.array([dtype(x) for x in spec.split(',')], dtype=dtype)


def batch_xs(table, T_spec, E_spec, device, xla):
    Ts = parse_range(T_spec, float); Es = parse_range(E_spec, int)
    with h5py.File(table,'r') as hf:
        dt = np.float16 if hf['dtype'][()].decode()=='float16' else np.float32
        W0 = tf.constant(hf['W0'][:],dtype=dt); b0=tf.constant(hf['b0'][:],dtype=dt)
        alpha=float(hf['alpha'][()]); T_scale=tf.constant(hf['T_scale'][:],dtype=dt); T_mean=tf.constant(hf['T_mean'][:],dtype=dt)
        b_tab=tf.constant(hf['b_tab_base'][:],dtype=dt); fb_idx=hf['fb_idx'][:]; fb_W=tf.constant(hf['fb_W'][:],dtype=dt); fb_b=tf.constant(hf['fb_b'][:],dtype=dt)
        has_C1='C1' in hf; C1=tf.constant(hf['C1'][:],dtype=dt) if has_C1 else None; idx1=tf.constant(hf['idx1'][:],dtype=tf.int32) if has_C1 else None
    with tf.device(f'/{device}:0'):
        if xla: tf.config.optimizer.set_jit(True)
        hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims((tf.constant(Ts,dtype=dt)-T_mean)/T_scale,1),W0)+b0,alpha)
        full=[]; quant=[]
        for e in Es:
            pos=int(np.searchsorted(fb_idx,e))
            (full if pos<fb_idx.size and fb_idx[pos]==e else quant).append(e)
        parts=[]
        if full: parts.append((tf.gather(fb_W,full,axis=1),tf.gather(fb_b,full)))
        if quant and has_C1: parts.append((tf.gather(C1,tf.gather(idx1,quant),axis=1),tf.gather(b_tab,quant)))
        W_sel = tf.concat([p[0] for p in parts], axis=1)
        b_sel = tf.concat([tf.expand_dims(p[1], 0) for p in parts], axis=1)
        start=time.time(); xs=tf.matmul(hidden,W_sel)+b_sel; elapsed=(time.time()-start)*1000; tot=xs.shape[0]*xs.shape[1]
        print(f"[batch] ({xs.shape[0]},{xs.shape[1]}) on /{device}:0 in {elapsed:.1f} ms → {elapsed*1e3/tot:.5f} µs/value")


def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest='cmd')
    pb=sp.add_parser('build'); pb.add_argument('--weights',required=True); pb.add_argument('--scaler',required=True)
    pb.add_argument('--window',type=int,required=True); pb.add_argument('--step',type=int,required=True); pb.add_argument('--out',required=True)
    pb.add_argument('--dtype',choices=['float32','float16'],default='float32'); pb.add_argument('--quantize',type=int,default=0)
    pb.add_argument('--fallback_error',type=float,default=0.001,help='relative error cutoff')
    pq=sp.add_parser('query'); pq.add_argument('--table',required=True); pq.add_argument('--T',type=float,required=True); pq.add_argument('--E_idx',type=int,required=True)
    pbatch=sp.add_parser('batch'); pbatch.add_argument('--table',required=True); pbatch.add_argument('--T',required=True); pbatch.add_argument('--E',required=True)
    pbatch.add_argument('--device',choices=['CPU','GPU'],default='CPU'); pbatch.add_argument('--xla',action='store_true')
    args=p.parse_args()
    if args.cmd=='build': dt=np.float16 if args.dtype=='float16' else np.float32; build_table(args.weights,args.scaler,args.window,args.step,args.out,dtype=dt,quantize=args.quantize,fallback_error=args.fallback_error)
    elif args.cmd=='query': print(query_xs(args.table,args.T,args.E_idx))
    elif args.cmd=='batch': batch_xs(args.table,args.T,args.E,args.device,args.xla)
    else: p.print_help()

if __name__=='__main__': main()
