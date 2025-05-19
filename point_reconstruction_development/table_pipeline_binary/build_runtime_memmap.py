#!/usr/bin/env python3
"""build_runtime_memmap.py – Binary dump and memmap testing for Doppler-XS lookup

This script extends build_runtime.py with multiple subcommands:

1. build-bin: Dump the precomputed table from an HDF5 file into raw binary files (row-major E_ROWS×hidden).
2. memmap-test: Load the raw binaries via numpy.memmap and benchmark random or batch row lookups.
3. load-test: Load the raw binaries fully into a NumPy array and benchmark lookups (no memmap).
4. compare: Run both memmap-test and load-test back-to-back for direct comparison.
"""
from __future__ import annotations
import argparse
import time
import os
import h5py
import numpy as np
import sys

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def dump_binary(h5_path: str, out_prefix: str) -> None:
    """
    Reads W_tab and b_tab from `h5_path` and writes them as raw binaries:
      <out_prefix>_W_tab.bin (float array of shape [E_ROWS, hidden])
      <out_prefix>_b_tab.bin (float array of length E_ROWS)
    """
    with h5py.File(h5_path, 'r') as hf:
        dtype_str = hf['dtype'][()].decode()
        dtype = np.float16 if dtype_str == 'float16' else np.float32
        W_tab = hf['W_tab'][:].astype(dtype)  # shape (hidden, E_ROWS)
        b_tab = hf['b_tab'][:].astype(dtype)  # shape (E_ROWS,)

    # transpose to row-major [E_ROWS, hidden]
    W_tab = W_tab.T

    W_bin = f"{out_prefix}_W_tab.bin"
    b_bin = f"{out_prefix}_b_tab.bin"
    W_tab.tofile(W_bin)
    b_tab.tofile(b_bin)

    print(f"Dumped W_tab -> {W_bin}: {os.path.getsize(W_bin)/1e6:.3f} MB")
    print(f"Dumped b_tab -> {b_bin}: {os.path.getsize(b_bin)/1e6:.3f} MB")


def memmap_test(prefix: str, trials: int = 1000) -> tuple[float, float]:
    """
    Opens raw binaries via memmap, then samples `trials` row lookups:
      - Random-access per-point timing
      - Batch numpy-index timing
    Returns (avg_loop_us, avg_batch_us).
    """
    # infer dtype and shape from HDF5
    h5_path = f"{prefix}.h5"
    with h5py.File(h5_path, 'r') as hf:
        dtype_str = hf['dtype'][()].decode()
        dtype = np.float16 if dtype_str == 'float16' else np.float32
        hidden, E_ROWS = hf['W_tab'][:].shape

    W_mm = np.memmap(f"{prefix}_W_tab.bin", mode='r', dtype=dtype, shape=(E_ROWS, hidden))
    b_mm = np.memmap(f"{prefix}_b_tab.bin", mode='r', dtype=dtype, shape=(E_ROWS,))

    # warm-up
    for idx in (0, E_ROWS//2, E_ROWS-1):
        _ = W_mm[idx]; _ = b_mm[idx]

    # random indices
    idxs = np.random.randint(0, E_ROWS, size=trials)

    # per-point loop timing
    start = time.perf_counter()
    for i in idxs:
        _ = W_mm[i]
        _ = b_mm[i]
    total = time.perf_counter() - start
    avg_loop = total / trials * 1e6
    print(f"[memmap] loop: total {total:.4f}s, avg {avg_loop:.2f} µs/point")

    # batch fetch timing
    start2 = time.perf_counter()
    _ = W_mm[idxs]
    _ = b_mm[idxs]
    total_batch = time.perf_counter() - start2
    avg_batch = total_batch / trials * 1e6
    print(f"[memmap] batch: total {total_batch:.4f}s, avg {avg_batch:.2f} µs/point")

    return avg_loop, avg_batch


def load_test(prefix: str, trials: int = 1000) -> tuple[float, float]:
    """
    Loads raw binaries into memory, then benchmarks row lookups:
      - Random-access per-point timing
      - Batch numpy-index timing
    Returns (avg_loop_us, avg_batch_us).
    """
    # infer dtype and shape
    h5_path = f"{prefix}.h5"
    with h5py.File(h5_path, 'r') as hf:
        dtype_str = hf['dtype'][()].decode()
        dtype = np.float16 if dtype_str == 'float16' else np.float32
        hidden, E_ROWS = hf['W_tab'][:].shape

    # load entire table into memory
    W = np.fromfile(f"{prefix}_W_tab.bin", dtype=dtype).reshape(E_ROWS, hidden)
    b = np.fromfile(f"{prefix}_b_tab.bin", dtype=dtype)

    # warm-up
    for idx in (0, E_ROWS//2, E_ROWS-1):
        _ = W[idx]; _ = b[idx]

    idxs = np.random.randint(0, E_ROWS, size=trials)

    # per-point loop
    start = time.perf_counter()
    for i in idxs:
        _ = W[i]
        _ = b[i]
    total = time.perf_counter() - start
    avg_loop = total / trials * 1e6
    print(f"[load] loop: total {total:.4f}s, avg {avg_loop:.2f} µs/point")

    # batch fetch
    start2 = time.perf_counter()
    _ = W[idxs]
    _ = b[idxs]
    total_batch = time.perf_counter() - start2
    avg_batch = total_batch / trials * 1e6
    print(f"[load] batch: total {total_batch:.4f}s, avg {avg_batch:.2f} µs/point")

    return avg_loop, avg_batch


def compare_tests(prefix: str, trials: int = 1000) -> None:
    print("Running memmap-test...")
    mem_loop, mem_batch = memmap_test(prefix, trials)
    print("\nRunning load-test...")
    load_loop, load_batch = load_test(prefix, trials)
    print(f"\nSummary (avg µs/point): memloop={mem_loop:.2f}, membatch={mem_batch:.2f}, loop={load_loop:.2f}, batch={load_batch:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Binary dump and lookup benchmarking for Doppler-XS tables"
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_build = sub.add_parser('build-bin', help='Dump HDF5 table to raw binaries')
    p_build.add_argument('--h5', required=True, help='Input HDF5 table (e.g. w_table.h5)')
    p_build.add_argument('--out-prefix', default='w_table', help='Prefix for output binaries')

    p_mem = sub.add_parser('memmap-test', help='Benchmark memmap row lookups')
    p_mem.add_argument('--prefix', default='w_table', help='Prefix of binary files')
    p_mem.add_argument('--trials', type=int, default=1000, help='Number of lookups')

    p_load = sub.add_parser('load-test', help='Benchmark in-memory array row lookups')
    p_load.add_argument('--prefix', default='w_table', help='Prefix of binary files')
    p_load.add_argument('--trials', type=int, default=1000, help='Number of lookups')

    p_cmp = sub.add_parser('compare', help='Run both memmap-test and load-test')
    p_cmp.add_argument('--prefix', default='w_table', help='Prefix of binary files')
    p_cmp.add_argument('--trials', type=int, default=1000, help='Number of lookups')

    args = parser.parse_args()
    if args.cmd == 'build-bin':
        dump_binary(args.h5, args.out_prefix)
    elif args.cmd == 'memmap-test':
        memmap_test(args.prefix, args.trials)
    elif args.cmd == 'load-test':
        load_test(args.prefix, args.trials)
    elif args.cmd == 'compare':
        compare_tests(args.prefix, args.trials)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
