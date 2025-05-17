#!/usr/bin/env python3
"""
transpose_wtab.py – convert W_tab from [E,16] to fast [16,E].

Run this exactly **one time** right after you’ve generated w_table.h5.
It rewrites the dataset in-place; no copy is kept.
"""

import h5py, argparse, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="w_table.h5")
    args = p.parse_args()

    with h5py.File(args.table, "r+") as hf:
        if hf["W_tab"].shape[0] == 16:
            print("W_tab already transposed – nothing to do.")
            return
        W_fast = hf["W_tab"][:].astype(np.float32).T       # [E,16] → [16,E]
        del hf["W_tab"]
        hf["W_tab"] = W_fast
        print(f"Done.  New shape: {W_fast.shape}")

if __name__ == "__main__":
    main()
