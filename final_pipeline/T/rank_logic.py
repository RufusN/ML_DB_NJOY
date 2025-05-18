#!/usr/bin/env python3
"""
compute_required_rank.py – determine minimal SVD rank for desired energy retention
====================================================================
Loads a full FP32 weight table (W_tab) from HDF5, computes its singular values,
and reports the smallest rank r needed to capture specified energy thresholds.
Usage:
    python compute_required_rank.py --table w_table.h5 --threshold 0.9999
"""
from __future__ import annotations
import argparse
import h5py
import numpy as np

def compute_energy_rank(W_tab: np.ndarray, thresholds: list[float]) -> dict[float,int]:
    # full SVD
    U, S, Vt = np.linalg.svd(W_tab, full_matrices=False)
    # energy spectrum
    cum_energy = np.cumsum(S**2)
    total = cum_energy[-1]
    # find minimal rank for each threshold
    ranks = {}
    for thresh in thresholds:
        # find first index where cumulative energy >= thresh * total
        idx = np.searchsorted(cum_energy, thresh * total)
        ranks[thresh] = idx + 1  # +1 for 1-based rank
    return ranks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table", type=str, required=True,
        help="Path to full-table HDF5 file containing dataset 'W_tab'."
    )
    parser.add_argument(
        "--threshold", type=float, nargs='+', default=[0.90, 0.95, 0.99, 0.999, 0.9999],
        help="List of energy retention thresholds (fractions) to evaluate."
    )
    args = parser.parse_args()

    # load full table
    with h5py.File(args.table, 'r') as hf:
        W_tab = hf['W_tab'][:]  # shape (E,16), float32

    # compute required ranks
    ranks = compute_energy_rank(W_tab, args.threshold)

    # report
    print("Required ranks for energy thresholds:")
    for thresh, r in sorted(ranks.items()):
        print(f"  {thresh*100:.4f}% energy → rank {r}")
