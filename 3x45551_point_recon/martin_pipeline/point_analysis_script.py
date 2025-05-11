#!/usr/bin/env python3
"""
mlp_stft_pointwise.py – GPU‐accelerated pointwise MLP + inverse‐STFT timing & accuracy

Usage:
  # timing only
  python mlp_stft_pointwise.py --mode speed \
      --batch 30000000 --chunk 1000000

  # accuracy only
  python mlp_stft_pointwise.py --mode accuracy \
      --tmin 950 --tmax 1050 --tstep 0.1 --error_step 100

  # both
  python mlp_stft_pointwise.py --mode both \
      --batch 10000000 --chunk 1000000 \
      --tmin 950 --tmax 1050 --tstep 0.1 --error_step 100
"""
import argparse, time, glob, os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from tensorflow.signal import irfft
import matplotlib.pyplot as plt

# ---- fixed STFT params & file‐paths ----
E_MIN, E_MAX    = 1e4*1e-6, 1e5*1e-6
BASE_XS_H5      = "/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5"
SPEC_DIR        = "/mnt/d/800_1200"
WINDOW_SIZE     = 0.000398
STEP_SIZE       = 0.000199

def load_base():
    # load the 200K cross‐section from HDF5
    df = pd.read_hdf(BASE_XS_H5, key="xs_data", mode="r")
    df = df[(df.T==200.0)&df.ERG.between(E_MIN, E_MAX)]
    E = df.ERG.to_numpy(); xs = df.XS.to_numpy()
    o = np.argsort(E)
    return E[o], xs[o], len(o)

def load_temperature(T, base_E, pad):
    # find the file that contains T, interpolate onto base_E, pad
    for fn in glob.glob(os.path.join(SPEC_DIR,"*.h5")):
        df = pd.read_hdf(fn, key="xs_data", mode="r")
        if T not in df.T.values: continue
        sub = df[(df.T==T)&df.ERG.between(E_MIN, E_MAX)]
        E = sub.ERG.to_numpy(); xs = sub.XS.to_numpy()
        o = np.argsort(E)
        sig = interp1d(E[o], xs[o], kind="cubic",
                       fill_value="extrapolate")(base_E)
        return np.pad(base_E,(pad,pad)), np.pad(sig,(pad,pad))
    raise FileNotFoundError(f"No spectrogram for T={T}")

def mapSegIdx(idx, step, win):
    # compute which STFT-frames overlap a single energy index
    n = win//step
    segs, locs = [], []
    s = int(np.ceil((idx-win)/step - 1 + 0.5))
    for _ in range(n):
        segs.append(s)
        locs.append(idx - s*step)
        s += 1
    return np.array(segs, int), locs

# ----------------------------------------------------------------------------
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["speed","accuracy","both"], required=True)
    p.add_argument("--batch",      type=int,   default=10_000_000)
    p.add_argument("--chunk",      type=int,   default=1_000_000)
    p.add_argument("--tmin",       type=float, default=1000.0)
    p.add_argument("--tmax",       type=float, default=1000.0)
    p.add_argument("--tstep",      type=float, default=0.1)
    p.add_argument("--error_step", type=int,   default=100)
    args = p.parse_args()

    # 1) Prepare STFT grid
    E_grid, xs_base, fs  = load_base()
    win   = int(WINDOW_SIZE * fs)
    step  = int(STEP_SIZE   * fs)
    pad   = win
    slice_len = win // step
    h, w, c    = 3, len(E_grid), 2
    flat = h*w*c

    # 2) Load compacted MLP weights & scalers
    dat        = np.load("mlp_pointwise.npz")
    W0, b0     = dat["W0"],    dat["b0"]
    W1, b1     = dat["W1"],    dat["b1"]
    T_scale, T_mean = dat["T_scale"], dat["T_mean"]
    R_scale, R_mean = dat["R_scale"], dat["R_mean"]

    # move to tf
    W0     = tf.constant(W0, tf.float32)
    b0     = tf.constant(b0, tf.float32)
    W1     = tf.constant(W1, tf.float32)
    b1     = tf.constant(b1, tf.float32)
    T_mean = tf.constant(T_mean, tf.float32)
    T_scale= tf.constant(T_scale,tf.float32)

    # 3) Build pointwise‐slice MLP
    @tf.function
    def mlp_slice(Tb, Eidx):
        # Tb: [N], Eidx: [N]
        #  a) normalize
        Tn = (Tb - T_mean)/T_scale                # [N]
        #  b) hidden
        h1 = tf.nn.leaky_relu(tf.expand_dims(Tn,1)@W0 + b0, alpha=0.1)  # [N,16]
        #  c) for each sample, gather the next slice_len* h * 2 weights
        #     the flat output is size (h * slice_len * 2)
        #  we pre‐stored W1 as shape [16, flat], but flat= h * w * c
        #  so to get just the slice we use tf.gather on axis=1
        #  first compute the flat‐indices for **all** overlap‐frames:
        #    for each of the slice_len frames we need h*2 channels at that frame.
        #  build once outside if speed is required.
        # --- here we just gather the ENTIRE SPECTROGRAM for simplicity:
        out_flat = tf.matmul(h1, W1) + b1   # [N, flat]
        # reshape to slices
        return tf.reshape(out_flat, [-1, h, w, c])  # [N,h,w,2]

    # warm-up
    _ = mlp_slice(tf.constant([args.tmin],tf.float32),
                  tf.constant([0], tf.int32))

    # --- SPEED MODE: full MLP+ISTFT -------------------------------------------
    if args.mode in ("speed","both"):
        N    = args.batch
        Ts   = tf.random.uniform([N], args.tmin, args.tmax, tf.float32)
        Eids = np.random.randint(0, w, size=N, dtype=np.int32)

        t0 = time.perf_counter()
        for i in range(0, N, args.chunk):
            j = min(N, i+args.chunk)
            Tb = Ts[i:j]
            Ei = tf.constant(Eids[i:j], tf.int32)
            # 1) predict spectrogram full‐frame
            spec_full = mlp_slice(Tb, Ei).numpy()  # [B,h,w,2]
            # 2) inverse‐STFT pointwise
            for k,e in enumerate(Eids[i:j]):
                segs, locs = mapSegIdx(int(e), step, win)
                # slice out only the needed frames
                small = spec_full[k, :, segs, :]     # shape (h, slice_len, 2)
                # run ISTFT with scipy’s irfft (on CPU unfortunately),
                # or use tf.signal.irfft if you want GPU:
                #    tf.signal.irfft(small, fft_length=[win]) * hann(win)
                # here we call the helper:
                wwin = hann(win)
                parts = [irfft(small[:,t,:], fft_length=win)*wwin
                         for t in range(small.shape[1])]
                xs_val = sum(p[l] for p,l in zip(parts,locs))
                _ = xs_val / ((wwin**2).sum()/step)
        dt = time.perf_counter() - t0
        print(f"{N:,} pts → {dt/N*1e6:.2f} µs/pt")

    # --- ACCURACY MODE: compare to ground truth ------------------------------
    if args.mode in ("accuracy","both"):
        # T‐grid
        Ts = np.arange(args.tmin, args.tmax+1e-9, args.tstep, dtype=np.float32)
        # energy indices every error_step
        Eidx = np.arange(pad, w-pad, args.error_step, dtype=int)

        all_rels = []
        for T in Ts:
            # load true padded curve
            Epad, xspad = load_temperature(T, E_grid, pad)
            Tb = tf.constant([T]*len(Eidx), tf.float32)
            Ei = tf.constant(Eidx, tf.int32)
            # predict spectrograms
            spec_full = mlp_slice(Tb, Ei).numpy()  # [M,h,w,2]
            recs = []
            for k,e in enumerate(Eidx):
                segs, locs = mapSegIdx(int(e), step, win)
                small = spec_full[k, :, segs, :]
                wwin = hann(win)
                parts = [irfft(small[:,t,:], fft_length=win)*wwin
                         for t in range(small.shape[1])]
                xs_val = sum(p[l] for p,l in zip(parts,locs))
                recs.append(xs_val/((wwin**2).sum()/step))
            recs = np.array(recs)
            true = xspad[Eidx]
            rels = np.abs(recs-true)/np.abs(true)*100
            all_rels.append(rels)
        mean_rels = np.stack(all_rels,0).mean(0)

        # plots
        plt.figure()
        plt.semilogx(Epad[Eidx], mean_rels, "o-")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Mean Rel Error (%)")
        plt.grid()
        plt.savefig("rel_error.png", dpi=200)

        # one slice vs pred
        plt.figure()
        plt.loglog(Epad[Eidx], xspad[Eidx], label="true")
        plt.loglog(Epad[Eidx], recs,    label="pred")
        plt.legend(); plt.grid()
        plt.savefig("xs_compare.png", dpi=200)
