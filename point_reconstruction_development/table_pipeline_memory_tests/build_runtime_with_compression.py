#!/usr/bin/env python3
"""build_runtime_with_compression.py – extend build_runtime.py with a zstd test subcommand

Adds a `test-compression` command to run zstd -3 on a built FP16 table and report sizes and timing."""

import argparse
import os
import subprocess
import time

# --- Compression tester -----------------------------------------------------

def compress_zstd(input_path: str, level: int = 3) -> tuple[str, float]:
    """
    Compress `input_path` using `zstd -<level>` (stdout) and write to `<input_path>.zst`.
    Returns (output_path, duration_seconds).
    """
    output_path = f"{input_path}.zst"
    start = time.perf_counter()
    # call zstd to stdout and capture into file
    with open(output_path, "wb") as out_f:
        proc = subprocess.Popen(
            ["zstd", f"-{level}", "--stdout", input_path],
            stdout=out_f,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate()
    duration = time.perf_counter() - start
    return output_path, duration

# --- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extend build_runtime with zstd compression tests"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # test-compression subcommand
    p_comp = sub.add_parser(
        "test-compression", help="Compress a built FP16 table with zstd -3 and report ratio"
    )
    p_comp.add_argument(
        "--table", required=True,
        help="Path to the FP16 HDF5 table (e.g. w_table_fp16.h5)"
    )
    p_comp.add_argument(
        "--level", type=int, default=3,
        help="zstd compression level to use (default: 3)"
    )

    args = parser.parse_args()

    if args.cmd == "test-compression":
        table = args.table
        level = args.level
        if not os.path.isfile(table):
            print(f"Error: table file '{table}' not found.")
            return

        size_in = os.path.getsize(table)
        print(f"[input]  {table}: {size_in/1e6:.3f} MB")

        out_path, duration = compress_zstd(table, level)
        size_out = os.path.getsize(out_path)

        print(f"[output] {out_path}: {size_out/1e6:.3f} MB")
        print(f"[info]   zstd level={level} in {duration:.2f}s (ratio {size_in/size_out:.2f}x)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
