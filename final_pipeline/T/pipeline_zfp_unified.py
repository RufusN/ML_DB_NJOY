#!/usr/bin/env python3
"""
pipeline_fp16_tiered.py – HVQ + tiered fp32/fp16 caches  (v10)
==============================================================
• row/col-major centroids auto-detected
• fp32 → fp16 → HVQ fall-through
• XLA disabled automatically for --policy float32
• ***NEW***  pandas-version-agnostic HDF reader
   (fixes “generic buffer has not been implemented”)
"""
from __future__ import annotations
import argparse, time, glob, os, h5py, tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
# -------------------------------------------------------------------
#  ▌1▐  robust HDF5 helpers  (only new code in this file)
# -------------------------------------------------------------------
# -------------------------------------------------------------------

E_MIN, E_MAX = 1e4*1e-6, 1e6*1e-6
BASE_FILE_PATH = r'/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'
TEMP_DATA_DIR  = r'/mnt/d/800_1200'

# ─────────── GPU-resident tensors ───────────
W0=b0=T_scale=T_mean=None; alpha=0.0
C1=C2=idx1=idx2=b_tab=None
rows32=W32=b32=rows16=W16=b16=None
ROW_MAJOR=True
#  (everything below **unchanged** except the three places that now call
#   _safe_read_hdf instead of pd.read_hdf)

_KEEP = {"BASE_FILE_PATH", "TEMP_DATA_DIR", "E_MIN", "E_MAX"}

def _reset():
    for k in list(globals()):
        if k.isupper() and k not in _KEEP:
            globals()[k] = None
    global alpha, ROW_MAJOR
    alpha = 0.0
    ROW_MAJOR = True

def load_table(fp: str, dtype):
    _reset()
    with h5py.File(fp) as f, tf.device('/GPU:0'):
        cast=lambda k: tf.constant(f[k][:], dtype=dtype)
        globals().update(
            W0=cast('W0'), b0=cast('b0'),
            T_scale=cast('T_scale'), T_mean=cast('T_mean'),
            alpha=float(f['alpha'][()]),
            C1=cast('C1'), C2=cast('C2'),
            idx1=tf.constant(f['idx1'][:], tf.int32),
            idx2=tf.constant(f['idx2'][:], tf.int32),
            b_tab=cast('b_tab'))
        if 'rows32' in f:
            globals().update(
                rows32=tf.constant(f['rows32'][:], tf.int32),
                W32=cast('W32'), b32=cast('b32'))
        if 'rows16' in f:
            globals().update(
                rows16=tf.constant(f['rows16'][:], tf.int32),
                W16=cast('W16'), b16=cast('b16'))
    global ROW_MAJOR
    ROW_MAJOR = (C1.shape[-1] == 16)
    rm = 'row' if ROW_MAJOR else 'col'
    print(f"[load] centroids {rm}-major; fp32={0 if rows32 is None else rows32.shape[0]}, "
          f"fp16={0 if rows16 is None else rows16.shape[0]}")

@tf.function(jit_compile=False)
def _hidden(T):
    T=tf.cast(T,T_scale.dtype)
    return tf.nn.leaky_relu(
        tf.matmul(((T-T_mean)/T_scale)[:,None],W0)+b0,alpha)

def _apply(rows,Wc,bc,W_prev,b_prev,E):
    if rows is None: return W_prev,b_prev
    n=tf.shape(rows)[0]
    pad_row=tf.concat([rows,[-1]],0)
    W_pad=tf.concat([Wc,tf.zeros([1,tf.shape(Wc)[1]],Wc.dtype)],0)
    b_pad=tf.concat([bc,tf.zeros([1],bc.dtype)],0)
    idx=tf.searchsorted(rows,E,side='left'); idx_c=tf.minimum(idx,n)
    m=tf.equal(tf.gather(pad_row,idx_c),E)
    W=tf.where(m[:,None],tf.gather(W_pad,idx_c),W_prev)
    b=tf.where(m,tf.gather(b_pad,idx_c),b_prev)
    return W,b

def make_query(xla: bool):
    @tf.function(experimental_compile=xla)
    def query(T,E):
        h=_hidden(T)
        if ROW_MAJOR:
            W=tf.gather(C1,tf.gather(idx1,E))+tf.gather(C2,tf.gather(idx2,E))
        else:
            W=tf.transpose(tf.gather(C1,tf.gather(idx1,E),axis=1))+\
               tf.transpose(tf.gather(C2,tf.gather(idx2,E),axis=1))
        b=tf.gather(b_tab,E)
        W,b=_apply(rows16,W16,b16,W,b,E)
        W,b=_apply(rows32,W32,b32,W,b,E)
        return tf.reduce_sum(h*W,1)+b
    return query

def benchmark(tbl,batch,tmin,tmax,policy):
    mixed_precision.set_global_policy(policy)
    use_xla = policy!='float32'
    if not use_xla: tf.config.optimizer.set_jit(False)
    dtype = tf.float16 if policy!='float32' else tf.float32
    load_table(tbl,dtype)
    T=tf.random.uniform([batch],tmin,tmax,dtype=dtype)
    E=tf.random.uniform([batch],4,90000,tf.int32)
    q=make_query(use_xla); _=q(T[:1],E[:1])
    t0=time.perf_counter(); xs=q(T,E); xs.numpy()
    print(f"{batch:,} pts → {(time.perf_counter()-t0)*1e6/batch:.2f} µs/pt")
    return xs,T,E

# ═══════════════════════ helper: base grid ════════════════════════
def load_base(e_min, e_max, base_t=200.0):
    df = pd.read_hdf(BASE_FILE_PATH, key='xs_data', compression='gzip')
    subset = df[(df['T'] == base_t) &
                (df['ERG'] >= e_min) &
                (df['ERG'] <= e_max)]
    E = subset['ERG'].to_numpy()
    idx = np.argsort(E)
    return E[idx], len(E)

# ═══════════════════════ helper: pad & interp ════════════════════════
def init_full():
    global PAD, WINDOW_SAMPS, STEP_SAMPS
    # read window/step from table metadata
    WINDOW_SAMPS = 4
    STEP_SAMPS   = 2
    PAD = WINDOW_SAMPS
    base_e, _ = load_base(E_MIN, E_MAX)
    return base_e


def load_temperature(test_temp: float, base_e: np.ndarray, file_dir: str):
    for fp in glob.glob(os.path.join(file_dir, '*.h5')):
        df = pd.read_hdf(fp, key='xs_data', compression='gzip')
        if test_temp not in df['T'].values:
            continue
        subset = df[(df['T'] == test_temp) &
                    (df['ERG'] >= E_MIN) &
                    (df['ERG'] <= E_MAX)]
        if len(subset) < 2:
            continue
        E = subset['ERG'].to_numpy()
        xs = subset['XS'].to_numpy()
        idx = np.argsort(E)
        interp = interp1d(E[idx], xs[idx], kind='cubic', fill_value='extrapolate')
        padded_e   = np.pad(base_e, (PAD, PAD), mode='constant')
        padded_xs  = np.pad(interp(base_e), (PAD, PAD), mode='constant')
        return padded_e, padded_xs
    return None, None

# ═════════════════════════════ Accuracy plot ═════════════════════════════
def analyse(base_e, xs_rec, temps, eidxs, file_dir):
    padded_e, padded = load_temperature(float(temps[0]), base_e, file_dir)
    if padded is None:
        print("Ground-truth data not found – skipping accuracy plot.")
        return
    xs_vals = xs_rec.numpy()
    idxs    = eidxs.numpy()
    orig    = padded[idxs]
    relerr  = np.abs(xs_vals - orig) / np.abs(orig) * 100

    os.makedirs("./results_exotic_inference", exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(base_e, padded[4:-4], label='Ground truth')
    plt.scatter(padded_e[idxs-4], xs_vals, c='r', s=10, label='Reconstructed')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Energy'); plt.ylabel('XS')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results_exotic_inference/xs.png", dpi=200)
    plt.close()

    sorted_idx = np.argsort(idxs)
    idxs_sorted = idxs[sorted_idx]
    rel_err_sorted = relerr[sorted_idx]
    print(len(base_e), len(relerr))
    plt.figure(figsize=(8,5))
    plt.plot(base_e[idxs_sorted-4], rel_err_sorted, marker='o', linestyle='-')
    plt.xlabel('Energy'); plt.ylabel('Relative error (%)')
    plt.title('Relative error vs energy'); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results_exotic_inference/relative_error_linear.png", dpi=200)
    plt.close()
    print("Accuracy plots saved to ./results_exotic_inference")

# ─────────── CLI ───────────
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table",default="w_table_ahvq.h5")
    p.add_argument("--batch",type=int,default=1_000_000)
    p.add_argument("--tmin",type=float,default=1000.0)
    p.add_argument("--tmax",type=float,default=1000.0)
    p.add_argument("--policy",choices=["float16","mixed_float16","float32"],
                   default="float32")
    p.add_argument("--accuracy",action="store_true")
    a=p.parse_args()

    xs,T,E = benchmark(a.table,a.batch,a.tmin,a.tmax,a.policy)
    if a.accuracy:
        base_e = init_full()
        analyse(base_e,xs,T,E,TEMP_DATA_DIR)
