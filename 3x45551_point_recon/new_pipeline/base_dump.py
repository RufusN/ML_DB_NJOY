#!/usr/bin/env python3
"""
export_base_grid.py
"""
import argparse
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

def extract_base_grid(base_file: Path, T_ref: float = 200.0) -> np.ndarray:
    """Return the sorted energy grid (MeV) for the reference temperature."""
    df = pd.read_hdf(base_file, key="xs_data", compression="gzip")
    subset = df[df["T"] == T_ref]["ERG"].to_numpy()
    if subset.size == 0:
        raise ValueError(f"T = {T_ref} not found in {base_file}")
    return np.sort(subset)


def dump_grid(grid: np.ndarray, out_file: Path) -> None:
    """Write the energy grid to a small HDF5 file."""
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("Base_E", data=grid)
    print(f"Saved grid ({len(grid)} points) → {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-file", default = '/mnt/d/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5')
    ap.add_argument("--out-file", default = "./base_energy_grid.h5")
    ap.add_argument("--T-ref", type=float, default=200.0,
                    help="reference temperature in base file (default: 200.0)")
    args = ap.parse_args()

    grid = extract_base_grid(Path(args.base_file), T_ref=args.T_ref)
    dump_grid(grid, Path(args.out_file))


if __name__ == "__main__":
    main()
