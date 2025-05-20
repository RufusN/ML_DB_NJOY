#!/usr/bin/env python3
"""
h5_to_bin.py

Read every dataset from an HDF5 (.h5) file and write out a single .bin file
containing a compressed pickle of all arrays. Reconstruction is easy with the
companion bin_to_h5() function below.
"""
import h5py
import pickle
import zlib
import lzma
import argparse
import os

def h5_to_bin(in_path, out_path, method='zlib', level=9):
    # 1) Read in all datasets into a dict
    data = {}
    with h5py.File(in_path, 'r') as h5f:
        def _visitor(name, node):
            if isinstance(node, h5py.Dataset):
                data[name] = node[()]
        h5f.visititems(_visitor)

    # 2) Serialize
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    # 3) Compress
    if method == 'zlib':
        compressed = zlib.compress(payload, level)
    elif method == 'lzma':
        compressed = lzma.compress(payload, preset=level)
    else:
        compressed = payload  # no compression

    # 4) Write out
    with open(out_path, 'wb') as f:
        f.write(compressed)

    in_sz = os.path.getsize(in_path)
    out_sz = os.path.getsize(out_path)
    print(f"Wrote {out_path} ({out_sz:,} bytes), compression ratio: {in_sz/out_sz:.2f}×")

def bin_to_h5(bin_path, h5_path, method='zlib'):
    # Companion to restore .h5 from the .bin
    with open(bin_path, 'rb') as f:
        raw = f.read()

    if method == 'zlib':
        payload = zlib.decompress(raw)
    elif method == 'lzma':
        payload = lzma.decompress(raw)
    else:
        payload = raw

    data = pickle.loads(payload)

    with h5py.File(h5_path, 'w') as h5f:
        for name, arr in data.items():
            h5f.create_dataset(name, data=arr)
    print(f"Reconstructed HDF5 to {h5_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert an HDF5 file to a compact .bin (and back)."
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # h5 → bin
    p1 = sub.add_parser('pack', help='Pack .h5 → .bin')
    p1.add_argument('input',  help='Input .h5 file')
    p1.add_argument('output', help='Output .bin file')
    p1.add_argument('--method', choices=('none','zlib','lzma'), default='zlib',
                    help='Compression method (default: zlib)')
    p1.add_argument('--level', type=int, default=9,
                    help='Compression level (1–9)')

    # bin → h5
    p2 = sub.add_parser('unpack', help='Unpack .bin → .h5')
    p2.add_argument('input',  help='Input .bin file')
    p2.add_argument('output', help='Reconstructed .h5 file')
    p2.add_argument('--method', choices=('none','zlib','lzma'), default='zlib',
                    help='Decompression method (must match pack)')

    args = parser.parse_args()

    if args.cmd == 'pack':
        h5_to_bin(args.input, args.output, method=args.method, level=args.level)
    elif args.cmd == 'unpack':
        bin_to_h5(args.input, args.output, method=args.method)

if __name__ == '__main__':
    main()
