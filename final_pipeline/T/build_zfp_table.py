#!/usr/bin/env python3
"""
table_auto_tiered.py  •  fp32 / fp16 / HVQ builder  (v6, memory-safe)
--------------------------------------------------------------------
Greedy algorithm
  1.  make 256+64 HVQ baseline (fp16 centroids, *row-major* → (k,16))
  2.  compute per-row max-error over all temperatures in --T_grid
  3.  iterate rows from worst to best
        • try fp16  → keep if error ≤ target
        • else try fp32 → keep if error ≤ target
        • stop when every row passes OR the byte budget is exhausted
The whole procedure never materialises (n_T × E) matrices; it streams
one row at a time, so RAM ≲ 200 MB for E≈91 k.
"""
from __future__ import annotations
import argparse, time, os, h5py, numpy as np
from sklearn.cluster import MiniBatchKMeans

# ────────────────────────── dense W_tab (unchanged maths) ─────────────────────
def map_segments_and_locals(e,w,s):
    n_over=w//s
    seg0=int(np.ceil((e-w)/s-0.5))
    segs=np.arange(seg0,seg0+n_over,dtype=np.int32)
    local=(e-(segs+1)*s)%w
    return segs,local

def dense_table(weights_h5,scaler_h5,window,step):
    H,C = 3,2
    with h5py.File(weights_h5) as f:
        W_dec=f['W_dec'][:].astype('float32')
        b_dec=f['b_dec'][:].astype('float32')
        alpha=float(f['alpha'][()])
        W0  =f['W0'] [:].astype('float32')
        b0  =f['b0'] [:].astype('float32')
        full_w=W_dec.shape[1]
    with h5py.File(scaler_h5) as f:
        spec_scale=f['spec_scale'][:].astype('float32')
        spec_mean =f['spec_mean'] [:].astype('float32')
        T_scale   =f['T_scale']  [:].astype('float32')
        T_mean    =f['T_mean']   [:].astype('float32')

    N_STFT=full_w//(H*C)
    E_ROWS=2*N_STFT+(window-1)
    LATENT, n_over = 16, window//step
    W_tab=np.zeros((E_ROWS,LATENT),np.float32)
    b_tab=np.zeros(E_ROWS            ,np.float32)

    hann=np.hanning(window).astype('float32')
    scale_fac=np.sum(hann**2)/step
    stride=N_STFT*C

    for e in range(E_ROWS):
        segs,local=map_segments_and_locals(e,window,step)
        if segs.max()>=N_STFT: continue
        idx=np.concatenate([(f*stride+segs[:,None]*C+np.arange(C)).ravel()
                            for f in range(H)])
        W_sub=W_dec[:,idx]*spec_scale[idx]
        b_sub=b_dec[idx]*spec_scale[idx]+spec_mean[idx]
        W_c=W_sub.reshape(LATENT,H,n_over,2)
        b_c=b_sub.reshape(H,n_over,2)
        W_c=(W_c[...,0]+1j*W_c[...,1]).transpose(0,2,1)
        b_c=(b_c[...,0]+1j*b_c[...,1]).transpose(1,0)
        seg_W=np.fft.irfft(W_c,n=window,axis=-1)*hann
        seg_b=np.fft.irfft(b_c,n=window,axis=-1)*hann
        W_tab[e]=seg_W[np.arange(LATENT)[:,None],np.arange(n_over),local].sum(1)/scale_fac
        b_tab[e]=seg_b[np.arange(n_over),local].sum()/scale_fac
    return W_tab,b_tab,W0,b0,alpha,T_scale,T_mean

# ────────────────────────── 256+64 HVQ  (row-major) ───────────────────────────
def hvq(mat,k1=256,k2=64):
    km1=MiniBatchKMeans(k1,random_state=0,batch_size=4096,
                        n_init='auto').fit(mat)
    idx1=km1.labels_.astype(np.uint16)
    resid=mat-km1.cluster_centers_[idx1]
    km2=MiniBatchKMeans(k2,random_state=1,batch_size=4096,
                        n_init='auto').fit(resid)
    idx2=km2.labels_.astype(np.uint16)
    C1=km1.cluster_centers_.astype(np.float16)   # (k1,16)
    C2=km2.cluster_centers_.astype(np.float16)   # (k2,16)
    return C1,C2,idx1,idx2

# ────────────────────────── builder core ───────────────────────────
def build(weights,scaler,window,step,out_h5,
          T_grid:list[float],target_err,budget_mb,k1,k2):

    (W_tab,b_tab,W0,b0,alpha,T_scale,T_mean)=dense_table(
        weights,scaler,window,step)

    # HVQ baseline
    C1,C2,idx1,idx2=hvq(W_tab,k1,k2)
    hvq_W=C1[idx1]+C2[idx2]                       # (E,16)

    # helpers
    BYTES16=16*2+2; BYTES32=16*4+4
    cap=int(budget_mb*1024*1024); used=0
    rows16=[]; rows32=[]

    Ts=np.asarray(T_grid,dtype=np.float32)
    H_mat=[]
    for T in Ts:
        Tn=(T-T_mean)/T_scale
        h=np.maximum(Tn*W0+b0,0)+np.minimum(Tn*W0+b0,0)*0.01
        H_mat.append(h.astype(np.float32))
    H_mat=np.stack(H_mat,0)   # (nT,16)

    # pre-compute baseline error row-by-row to avoid big matrices
    def row_error(e, Wrow):
        xs_ref = H_mat @ Wrow          + b_tab[e]     # (nT,)
        xs_hvq = H_mat @ hvq_W[e]      + b_tab[e]
        return np.max(np.abs(xs_ref-xs_hvq)/(np.abs(xs_ref)+1e-12))

    rel=np.asarray([row_error(e,W_tab[e]) for e in range(W_tab.shape[0])],
                   dtype=np.float32)
    order=np.argsort(-rel)                 # worst → best

    print("[greedy] filling caches …")
    for e in order:
        if rel[e]<=target_err: break
        # ---------- try fp16 ----------
        need=BYTES16
        if used+need<=cap:
            W16=W_tab[e].astype(np.float16).astype(np.float32)
            xs_fp16=H_mat@W16 + b_tab[e]
            err=np.max(np.abs(xs_fp16-(H_mat@W_tab[e]+b_tab[e])) /
                       (np.abs(xs_fp16)+1e-12))
            if err<=target_err:
                rows16.append(e); used+=need; rel[e]=err; continue
        # ---------- try fp32 ----------
        need=BYTES32
        if used+need<=cap:
            rows32.append(e); used+=need; rel[e]=0.0
        else:
            break   # out of memory

    rows16=np.sort(rows16).astype(np.int32)
    rows32=np.sort(rows32).astype(np.int32)
    print(f"[budget] used {used/1024:.1f} kB  "
          f"(fp32 {len(rows32)}, fp16 {len(rows16)})")

    # write file
    with h5py.File(out_h5,'w') as f:
        f['C1'],f['C2'],f['idx1'],f['idx2']=C1,C2,idx1,idx2
        f['b_tab']=b_tab.astype(np.float16)
        if rows16.size:
            f['rows16'],f['W16'],f['b16']=rows16,\
                 W_tab[rows16].astype(np.float16), b_tab[rows16].astype(np.float16)
        if rows32.size:
            f['rows32'],f['W32'],f['b32']=rows32,\
                 W_tab[rows32].astype(np.float32), b_tab[rows32].astype(np.float32)
        f['W0'],f['b0'],f['alpha']=W0,b0,alpha
        f['T_scale'],f['T_mean']=T_scale,T_mean
        f['window'],f['step']=window,step
        f.attrs['tiered']=1
    print(f"[write] {out_h5}  ({os.path.getsize(out_h5)/1024:.1f} kB)")

# ────────────────────────── CLI ──────────────────────────
if __name__=="__main__":
    P=argparse.ArgumentParser()
    P.add_argument('--weights',required=True)
    P.add_argument('--scaler', required=True)
    P.add_argument('--window',type=int,required=True)
    P.add_argument('--step',  type=int,required=True)
    P.add_argument('--out',   default='w_table_ahvq.h5')
    P.add_argument('--T_grid',default='950,1000,1050',
                   help='comma-separated list of temperatures for error check')
    P.add_argument('--target_err',type=float,default=1e-3)
    P.add_argument('--budget_mb', type=float,default=0.3)
    P.add_argument('--k1',type=int,default=256)
    P.add_argument('--k2',type=int,default=64)
    A=P.parse_args()
    T_grid=[float(x) for x in A.T_grid.split(',')]
    build(A.weights,A.scaler,A.window,A.step,
          A.out,T_grid,A.target_err,A.budget_mb,A.k1,A.k2)
