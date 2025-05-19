# import argparse
# import time
# import h5py
# import numpy as np
# import tensorflow as tf
# from sklearn.cluster import KMeans


# def load_weights_and_scaler(table_path: str, _: str):
#     """
#     Load a precomputed latent-major table (16×E_ROWS) and base MLP parameters.
#     Expects an HDF5 produced by build_w_table.py containing datasets:
#       - W_tab (16×E_ROWS), b_tab (E_ROWS), W0, b0, alpha, T_scale, T_mean
#     The second argument is ignored (compatibility stub).
#     """
#     with h5py.File(table_path, 'r') as hf:
#         W_tab   = hf['W_tab'][:]            # [16, E_rows]
#         b_tab   = hf['b_tab'][:]            # [E_rows]
#         W0      = hf['W0'][:]               # [16]
#         b0      = hf['b0'][:]               # [16]
#         alpha   = float(hf['alpha'][()])    # scalar
#         T_scale = hf['T_scale'][:]          # [E_rows] or scalar
#         T_mean  = hf['T_mean'][:]           # [E_rows] or scalar
#     return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean


# def build_table(weights_path, scaler_path, window, step, out_path, *, dtype, quantize=0):
#     # Load the full table & params via build_w_table output
#     W_tab, b_tab, W0, b0, alpha, T_scale, T_mean = load_weights_and_scaler(weights_path, scaler_path)

#     with h5py.File(out_path, "w") as hf:
#         hf.create_dataset("dtype", data=np.string_(dtype.__name__))

#         if quantize > 0:
#             X = W_tab.T  # (E_rows, 16)
#             km = KMeans(n_clusters=quantize, random_state=0)
#             km.fit(X)
#             centers = km.cluster_centers_           # (K,16)
#             idx_arr = km.labels_.astype("uint8")   # (E_rows,)
#             C = centers.T                           # (16,K)

#             hf.create_dataset("C",    data=C,       dtype=dtype)
#             hf.create_dataset("idx",  data=idx_arr, dtype="uint8")
#             hf.create_dataset("b_tab",data=b_tab,   dtype=dtype)
#         else:
#             hf.create_dataset("W_tab", data=W_tab, dtype=dtype)
#             hf.create_dataset("b_tab", data=b_tab, dtype=dtype)

#         hf.create_dataset("W0",      data=W0,      dtype=dtype)
#         hf.create_dataset("b0",      data=b0,      dtype=dtype)
#         hf.create_dataset("alpha",   data=alpha)
#         hf.create_dataset("T_scale", data=T_scale, dtype=dtype)
#         hf.create_dataset("T_mean",  data=T_mean,  dtype=dtype)


# def query_xs(table_h5, T, E_idx):
#     with h5py.File(table_h5, "r") as hf:
#         dt = np.float16 if hf["dtype"][()].decode() == "float16" else np.float32
#         if "C" in hf:
#             C       = tf.constant(hf["C"][:],    dtype=dt)
#             idx_arr = tf.constant(hf["idx"][:],  dtype=tf.int32)
#             b_tab   = tf.constant(hf["b_tab"][:], dtype=dt)
#             E_tensor   = tf.constant([E_idx], dtype=tf.int32)
#             cluster_id = tf.gather(idx_arr, E_tensor)
#             W_sel      = tf.gather(C, cluster_id, axis=1)
#             b_sel      = tf.gather(b_tab, E_tensor)
#         else:
#             W_tab = tf.constant(hf["W_tab"][:], dtype=dt)
#             b_tab = tf.constant(hf["b_tab"][:], dtype=dt)
#             E_tensor = tf.constant([E_idx], dtype=tf.int32)
#             W_sel    = tf.gather(W_tab, E_tensor, axis=1)
#             b_sel    = tf.gather(b_tab, E_tensor)

#         W0      = tf.constant(hf["W0"][:],      dtype=dt)
#         b0      = tf.constant(hf["b0"][:],      dtype=dt)
#         alpha   = float(hf["alpha"][()])
#         T_scale = tf.constant(hf["T_scale"][:], dtype=dt)
#         T_mean  = tf.constant(hf["T_mean"][:],  dtype=dt)

#     T_tensor = tf.constant([T], dt)
#     T_norm   = (T_tensor - T_mean) / T_scale
#     hidden   = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm,1), W0) + b0, alpha)
#     xs_val   = tf.matmul(hidden, W_sel) + b_sel
#     return float(xs_val.numpy().squeeze())


# def parse_range(spec, dtype=float):
#     if ":" in spec:
#         start, stop, count = spec.split(":")
#         return np.linspace(dtype(start), dtype(stop), int(count), dtype=dtype)
#     else:
#         return np.array([dtype(x) for x in spec.split(",")], dtype=dtype)


# def batch_xs(table_h5, T_spec, E_spec, device, xla):
#     Ts = parse_range(T_spec, float)
#     Es = parse_range(E_spec, int)
#     with h5py.File(table_h5, 'r') as hf:
#         dt = np.float16 if hf["dtype"][()].decode() == "float16" else np.float32
#         tf_tbl = {}
#         if "C" in hf:
#             tf_tbl['C']    = tf.constant(hf["C"][:],    dtype=dt)
#             tf_tbl['idx']  = tf.constant(hf["idx"][:],  dtype=tf.int32)
#             tf_tbl['b_tab']= tf.constant(hf["b_tab"][:], dtype=dt)
#         else:
#             tf_tbl['W_tab']= tf.constant(hf["W_tab"][:], dtype=dt)
#             tf_tbl['b_tab']= tf.constant(hf["b_tab"][:], dtype=dt)
#         W0      = tf.constant(hf["W0"][:],      dtype=dt)
#         b0      = tf.constant(hf["b0"][:],      dtype=dt)
#         alpha   = float(hf["alpha"][()])
#         T_scale = tf.constant(hf["T_scale"][:], dtype=dt)
#         T_mean  = tf.constant(hf["T_mean"][:],  dtype=dt)

#     with tf.device(f'/{device}:0'):
#         if xla:
#             tf.config.optimizer.set_jit(True)
#         T_tensor = tf.constant(Ts, dtype=dt)
#         T_norm   = (T_tensor - T_mean) / T_scale
#         hidden   = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(T_norm,1), W0) + b0, alpha)
#         Es_tensor = tf.constant(Es, dtype=tf.int32)
#         if 'C' in tf_tbl:
#             cluster_ids = tf.gather(tf_tbl['idx'], Es_tensor)
#             W_sel = tf.gather(tf_tbl['C'], cluster_ids, axis=1)
#             b_sel = tf.gather(tf_tbl['b_tab'], Es_tensor)
#         else:
#             W_sel = tf.gather(tf_tbl['W_tab'], Es_tensor, axis=1)
#             b_sel = tf.gather(tf_tbl['b_tab'], Es_tensor)
#         start = time.time()
#         xs = tf.matmul(hidden, W_sel) + b_sel
#         elapsed_ms = (time.time() - start) * 1000
#         total = xs.shape[0] * xs.shape[1]
#         print(f"[batch] ({xs.shape[0]},{xs.shape[1]}) on /{device}:0 in {elapsed_ms:.1f} ms → {elapsed_ms*1e3/total:.6f} µs/value")


# def main():
#     p = argparse.ArgumentParser()
#     sp = p.add_subparsers(dest="cmd")

#     # Build
#     pb = sp.add_parser("build")
#     pb.add_argument("--weights", required=True, help="path to w_table.h5 from build_w_table.py")
#     pb.add_argument("--scaler",  required=True, help="ignored stub for compatibility")
#     pb.add_argument("--window",  type=int, required=True)
#     pb.add_argument("--step",    type=int, required=True)
#     pb.add_argument("--out",     required=True)
#     pb.add_argument("--dtype",   choices=["float32","float16"], default="float32")
#     pb.add_argument("--quantize", type=int, default=0,
#                    help="# clusters for k-means; 0 to disable")

#     # Query
#     pq = sp.add_parser("query")
#     pq.add_argument("--table", required=True)
#     pq.add_argument("--T",     type=float, required=True)
#     pq.add_argument("--E_idx", type=int,   required=True)

#     # Batch
#     pbatch = sp.add_parser("batch")
#     pbatch.add_argument("--table", required=True)
#     pbatch.add_argument("--T",     required=True, help="start:stop:count or comma list")
#     pbatch.add_argument("--E",     required=True, help="start:stop:count or comma list")
#     pbatch.add_argument("--device", choices=["CPU","GPU"], default="CPU")
#     pbatch.add_argument("--xla", action="store_true")

#     args = p.parse_args()
#     if args.cmd == "build":
#         dtype = np.float16 if args.dtype == "float16" else np.float32
#         build_table(
#             args.weights, args.scaler, args.window, args.step,
#             args.out, dtype=dtype, quantize=args.quantize
#         )
#     elif args.cmd == "query":
#         val = query_xs(args.table, args.T, args.E_idx)
#         print(val)
#     elif args.cmd == "batch":
#         batch_xs(args.table, args.T, args.E, args.device, args.xla)
#     else:
#         p.print_help()

# if __name__ == "__main__":
#     main()

import argparse
import time
import h5py
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


def load_weights_and_scaler(table_path: str, _: str):
    with h5py.File(table_path, 'r') as hf:
        W_tab   = hf['W_tab'][:]            # [16, E_rows]
        b_tab   = hf['b_tab'][:]            # [E_rows]
        W0      = hf['W0'][:]               # [16]
        b0      = hf['b0'][:]               # [16]
        alpha   = float(hf['alpha'][()])    # scalar
        T_scale = hf['T_scale'][:]          # [E_rows] or scalar
        T_mean  = hf['T_mean'][:]           # [E_rows] or scalar
    return W_tab, b_tab, W0, b0, alpha, T_scale, T_mean


def build_table(weights_path, scaler_path, window, step, out_path, *, dtype, quantize=0, quantize_residual=0):
    # Load full table from build_w_table.py output
    W_tab, b_tab, W0, b0, alpha, T_scale, T_mean = load_weights_and_scaler(weights_path, scaler_path)
    # W_tab shape: [16, E_rows]
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("dtype", data=np.string_(dtype.__name__))
        hf.create_dataset("W0", data=W0, dtype=dtype)
        hf.create_dataset("b0", data=b0, dtype=dtype)
        hf.create_dataset("alpha", data=alpha)
        hf.create_dataset("T_scale", data=T_scale, dtype=dtype)
        hf.create_dataset("T_mean", data=T_mean, dtype=dtype)
        hf.create_dataset("b_tab", data=b_tab, dtype=dtype)

        if quantize > 0:
            # Stage 1: coarse codebook
            X = W_tab.T  # (E_rows, 16)
            km1 = KMeans(n_clusters=quantize, random_state=0)
            km1.fit(X)
            C1 = km1.cluster_centers_.T  # (16, K1)
            idx1 = km1.labels_.astype('uint8')
            hf.create_dataset('C1', data=C1, dtype=dtype)
            hf.create_dataset('idx1', data=idx1, dtype='uint8')

            if quantize_residual > 0:
                # Stage 2: residual codebook
                # compute residuals
                residuals = X - km1.cluster_centers_[km1.labels_]  # (E_rows,16)
                km2 = KMeans(n_clusters=quantize_residual, random_state=0)
                km2.fit(residuals)
                C2 = km2.cluster_centers_.T  # (16, K2)
                idx2 = km2.labels_.astype('uint8')
                hf.create_dataset('C2', data=C2, dtype=dtype)
                hf.create_dataset('idx2', data=idx2, dtype='uint8')
        else:
            hf.create_dataset('W_tab', data=W_tab, dtype=dtype)


def query_xs(table_h5, T, E_idx):
    with h5py.File(table_h5, 'r') as hf:
        dt = np.float16 if hf['dtype'][()].decode()=='float16' else np.float32
        W0 = tf.constant(hf['W0'][:], dtype=dt)
        b0 = tf.constant(hf['b0'][:], dtype=dt)
        alpha = float(hf['alpha'][()])
        T_scale = tf.constant(hf['T_scale'][:], dtype=dt)
        T_mean  = tf.constant(hf['T_mean'][:], dtype=dt)
        b_tab   = tf.constant(hf['b_tab'][:], dtype=dt)

        # Normalize T
        Tt = tf.constant([T], dt)
        Tn = (Tt - T_mean) / T_scale
        hidden = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(Tn,1), W0) + b0, alpha)

        if 'C1' in hf:
            # coarse lookup
            C1 = tf.constant(hf['C1'][:], dtype=dt)   # [16,K1]
            idx1_arr = tf.constant(hf['idx1'][:], dtype=tf.int32)
            id1 = tf.gather(idx1_arr, [E_idx])        # [1]
            w1 = tf.gather(C1, id1, axis=1)           # [16,1]
            # residual stage
            if 'C2' in hf:
                C2 = tf.constant(hf['C2'][:], dtype=dt)   # [16,K2]
                idx2_arr = tf.constant(hf['idx2'][:], dtype=tf.int32)
                id2 = tf.gather(idx2_arr, [E_idx])        # [1]
                w2 = tf.gather(C2, id2, axis=1)           # [16,1]
                W_sel = w1 + w2
            else:
                W_sel = w1
            b_sel = tf.gather(b_tab, [E_idx])
        else:
            W_tab = tf.constant(hf['W_tab'][:], dtype=dt)
            W_sel = tf.gather(W_tab, [E_idx], axis=1)
            b_sel = tf.gather(b_tab, [E_idx])

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
        # load parameters
        W0 = tf.constant(hf['W0'][:], dtype=dt); b0=tf.constant(hf['b0'][:],dtype=dt)
        alpha=float(hf['alpha'][()]); T_scale=tf.constant(hf['T_scale'][:],dtype=dt); T_mean=tf.constant(hf['T_mean'][:],dtype=dt)
        b_tab = tf.constant(hf['b_tab'][:],dtype=dt)
        has_res = 'C1' in hf
        if has_res:
            C1 = tf.constant(hf['C1'][:],dtype=dt); idx1=tf.constant(hf['idx1'][:],dtype=tf.int32)
            has_r2 = 'C2' in hf
            if has_r2:
                C2=tf.constant(hf['C2'][:],dtype=dt); idx2=tf.constant(hf['idx2'][:],dtype=tf.int32)
        else:
            W_tab=tf.constant(hf['W_tab'][:],dtype=dt)
    with tf.device(f'/{device}:0'):
        if xla: tf.config.optimizer.set_jit(True)
        Tt=tf.constant(Ts,dtype=dt); Tn=(Tt-T_mean)/T_scale
        hidden=tf.nn.leaky_relu(tf.matmul(tf.expand_dims(Tn,1),W0)+b0,alpha)
        E_tensor=tf.constant(Es,dtype=tf.int32)
        if has_res:
            id1 = tf.gather(idx1, E_tensor)
            w1 = tf.gather(C1, id1, axis=1)
            if has_r2:
                id2=tf.gather(idx2,E_tensor); w2=tf.gather(C2,id2,axis=1); W_sel=w1+w2
            else: W_sel=w1
            b_sel=tf.gather(b_tab, E_tensor)
        else:
            W_sel=tf.gather(W_tab,E_tensor,axis=1); b_sel=tf.gather(b_tab,E_tensor)
        start=time.time()
        xs=tf.matmul(hidden,W_sel)+b_sel
        elapsed=(time.time()-start)*1000
        tot=xs.shape[0]*xs.shape[1]
        print(f"[batch] ({xs.shape[0]},{xs.shape[1]}) on /{device}:0 in {elapsed:.1f} ms → {elapsed*1e3/tot:.3f} µs/value")


def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest='cmd')
    pb=sp.add_parser('build'); pb.add_argument('--weights',required=True); pb.add_argument('--scaler',required=True)
    pb.add_argument('--window',type=int,required=True); pb.add_argument('--step',type=int,required=True)
    pb.add_argument('--out',required=True); pb.add_argument('--dtype',choices=['float32','float16'],default='float32')
    pb.add_argument('--quantize',type=int,default=0); pb.add_argument('--quantize_residual',type=int,default=0)
    pq=sp.add_parser('query'); pq.add_argument('--table',required=True); pq.add_argument('--T',type=float,required=True); pq.add_argument('--E_idx',type=int,required=True)
    pbatch=sp.add_parser('batch'); pbatch.add_argument('--table',required=True); pbatch.add_argument('--T',required=True);
    pbatch.add_argument('--E',required=True); pbatch.add_argument('--device',choices=['CPU','GPU'],default='CPU'); pbatch.add_argument('--xla',action='store_true')
    args=p.parse_args()
    if args.cmd=='build':
        dt=np.float16 if args.dtype=='float16' else np.float32
        build_table(args.weights,args.scaler,args.window,args.step,args.out,dtype=dt,quantize=args.quantize,quantize_residual=args.quantize_residual)
    elif args.cmd=='query': print(query_xs(args.table,args.T,args.E_idx))
    elif args.cmd=='batch': batch_xs(args.table,args.T,args.E,args.device,args.xla)
    else: p.print_help()

if __name__=='__main__': main()
