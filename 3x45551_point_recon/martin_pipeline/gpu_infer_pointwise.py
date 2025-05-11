#!/usr/bin/env python3
"""
mlp_stft_pointwise.py – GPU‐accelerated pointwise MLP + inverse‐STFT timing & accuracy

Usage:
  # 1) Pure speed test:
  python mlp_stft_pointwise.py --mode speed --batch 30000000 --chunk 1000000

  # 2) Accuracy over temperature range:
  python mlp_stft_pointwise.py --mode accuracy \
      --tmin 950 --tmax 1050 --tstep 0.1 --npoints 500

  # 3) Both:
  python mlp_stft_pointwise.py --mode both \
      --batch 10000000 --chunk 1000000 \
      --tmin 950 --tmax 1050 --tstep 0.1 --npoints 500
"""
import argparse, time, glob, os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from scipy.fft import irfft

# I/O paths — adjust if needed
MLP_NPZ    = "mlp_pointwise.npz"
BASE_XS_H5 = "/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
SPEC_DIR   = "/mnt/d/800_1200"

# Fixed spectrogram parameters
E_min, E_max = 1e4*1e-6, 1e5*1e-6
WINDOW_SIZE  = 0.000398
STEP_SIZE    = 0.000199

def load_base():
    df = pd.read_hdf(BASE_XS_H5, key="xs_data", mode="r")
    df = df[df["T"]==200.0]
    df = df[(df["ERG"]>=E_min)&(df["ERG"]<=E_max)]
    E = df["ERG"].to_numpy(); xs = df["XS"].to_numpy()
    order = np.argsort(E)
    return E[order], xs[order]

def load_temperature(T, Base_E, pad):
    for fn in glob.glob(os.path.join(SPEC_DIR, "*.h5")):
        df = pd.read_hdf(fn, key="xs_data", mode="r")
        if T not in df["T"].values: continue
        sub = df[df["T"]==T]
        sub = sub[(sub["ERG"]>=E_min)&(sub["ERG"]<=E_max)]
        E = sub["ERG"].to_numpy(); xs = sub["XS"].to_numpy()
        order = np.argsort(E)
        sig = interp1d(E[order], xs[order], kind="cubic",
                       fill_value="extrapolate")(Base_E)
        padded_sig = np.pad(sig, (pad,pad), "constant")
        padded_E   = np.pad(Base_E, (pad,pad), "constant")
        return padded_E, padded_sig
    raise FileNotFoundError(f"No spectrogram for T={T}")

def mapSegIdx(idx, step_samps, window_samps):
    n = window_samps // step_samps
    segs=[]; locs=[]
    seg = int(np.ceil((idx-window_samps)/step_samps -1 + .5))
    for _ in range(n):
        segs.append(seg)
        seg+=1
        start=seg*step_samps
        locs.append(idx-start)
    return np.array(segs,dtype=int), locs

def point_reconstruct_tf(spec, window_samps, step_samps, locs):
    # spec: [h, slice_length, 2] complex channels real+imag
    win = tf.signal.hann_window(window_samps, periodic=False)
    # perform inverse FFT on each of the 2 channels in one go:
    # spec_real+ i spec_imag → complex
    complex_spec = tf.complex(spec[...,0], spec[...,1])  # [h,L]
    time_segs = tf.signal.irfft(complex_spec, fft_length=window_samps)  # [h,L]
    time_segs = time_segs * win[None,:]
    # pick the local points and sum:
    vals = tf.stack([time_segs[:,i][loc] for i,loc in enumerate(locs)], axis=1)  # [h, n_overlaps]
    summed = tf.reduce_sum(vals, axis=1)
    scale = tf.reduce_sum(win**2) / step_samps
    return summed/scale  # [h]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["speed","accuracy","both"], required=True)
    p.add_argument("--batch",  type=int,   default=10_000_000)
    p.add_argument("--chunk",  type=int,   default=1_000_000)
    p.add_argument("--tmin",   type=float, default=1000.0)
    p.add_argument("--tmax",   type=float, default=1000.0)
    p.add_argument("--tstep",  type=float, default=1.0)
    p.add_argument("--npoints",type=int,   default=200)
    args = p.parse_args()

    # load MLP params
    dat = np.load(MLP_NPZ)
    W0 = tf.constant(dat["W0"],tf.float32); b0 = tf.constant(dat["b0"],tf.float32)
    W1 = tf.constant(dat["W1"],tf.float32); b1 = tf.constant(dat["b1"],tf.float32)
    T_scale = tf.constant(dat["T_scale"],tf.float32)
    T_mean  = tf.constant(dat["T_mean"], tf.float32)
    R_scale = dat["R_scale"]; R_mean = dat["R_mean"]
    flat = int(dat["h"])*int(dat["w"])*int(dat["c"])

    # precompute STFT dims
    Base_E, _ = load_base()
    fs = len(Base_E)
    window_samps = int(WINDOW_SIZE*fs)
    step_samps   = int(STEP_SIZE*fs)
    pad = window_samps
    h = int(dat["h"]); w = int(dat["w"]); c = int(dat["c"])

    @tf.function
    def mlp_pt(Tb, idxb):
        Tn = (Tb - T_mean)/T_scale
        hid = tf.nn.leaky_relu(tf.expand_dims(Tn,1)@W0 + b0, alpha=0.1)
        Wsub = tf.gather(tf.transpose(W1), idxb)
        bsub = tf.gather(b1, idxb)
        return tf.reduce_sum(hid*Wsub, axis=1) + bsub

    # warm up
    _ = mlp_pt(tf.constant([args.tmin]), tf.constant([0],tf.int32))

    if args.mode in ("speed","both"):
        N = args.batch
        Ts = tf.random.uniform([N], args.tmin, args.tmax, tf.float32)
        idx = tf.random.uniform([N], 0, flat, tf.int32)
        t0 = time.perf_counter()
        for i in range(0,N,args.chunk):
            j = min(N,i+args.chunk)
            raw = mlp_pt(Ts[i:j], idx[i:j]).numpy()
            ids = idx[i:j].numpy()
            spec_flat = raw * R_scale[ids] + R_mean[ids]
            # reconstruct each
            for k,eid in enumerate(ids):
                segs,locs = mapSegIdx(int(eid), step_samps, window_samps)
                small = spec_flat[k].reshape(h,w,c)[:,segs,:]
                _ = point_reconstruct_tf(small, window_samps, step_samps, locs)
        dt = time.perf_counter()-t0
        print(f"{N:,} pts → {dt/N*1e6:.2f} µs/pt")

    if args.mode in ("accuracy","both"):
        temps = np.arange(args.tmin, args.tmax+1e-9, args.tstep, dtype=np.float32)
        # sample energy indices on log‐uniform grid:
        Emin, Emax = Base_E[0], Base_E[-1]
        logs = np.random.uniform(np.log(Emin),np.log(Emax), args.npoints)
        Es = np.exp(logs); Eidx = np.searchsorted(Base_E,Es).clip(0,flat-1)
        padded_E, padded_sig = load_temperature(temps[0], Base_E, pad)
        for T in temps:
            Ts = tf.constant([T]*len(Eidx),tf.float32)
            Ei = tf.constant(Eidx,tf.int32)
            raw = mlp_pt(Ts,Ei).numpy()
            spec_flat = raw * R_scale[Eidx] + R_mean[Eidx]
            results=[]
            for k,eid in enumerate(Eidx):
                segs,locs = mapSegIdx(int(eid), step_samps, window_samps)
                small = spec_flat[k].reshape(h,w,c)[:,segs,:]
                results.append(point_reconstruct_tf(small, window_samps, step_samps, locs).numpy())
            results = np.array(results)
            true = padded_sig[Eidx]
            rel_err = np.abs(results-true)/np.abs(true)*100
            print(f"T={T:.1f}K  max_err={rel_err.max():.2f}%  mean_err={rel_err.mean():.2f}%")

if __name__=="__main__":
    main()
