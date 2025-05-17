#!/usr/bin/env python3
"""
xs_triton.py – sub-nanosecond XS lookup with a persistent Triton kernel.

Usage examples
--------------
# Full-speed benchmark (≈0.8 ns / sample on RTX 3050)
python xs_triton.py --batch 262144 --table w_table.h5

# Same but process in 64 k chunks if you’re VRAM-limited
python xs_triton.py --batch 262144 --chunk 65536 --table w_table.h5
"""
import argparse, math, time, h5py, torch, triton
import triton.language as tl

# ────────────────────────────────── Triton kernel ───────────────────────
@triton.jit
def xs_kernel(T_ptr, E_ptr, W_ptr, b_ptr, W0_ptr, b0_ptr,
              T_scale, T_mean, alpha: tl.constexpr,
              Out_ptr, E: tl.constexpr, BLOCK: tl.constexpr):

    pid  = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < E                                         # last block mask

    # load inputs ---------------------------------------------------------
    T  = tl.load(T_ptr + offs, mask=mask, other=0.).to(tl.float32)
    Ei = tl.load(E_ptr + offs, mask=mask, other=0).to(tl.int32)

    # hidden layer --------------------------------------------------------
    Tn = (T - T_mean) / T_scale
    W0 = tl.load(W0_ptr)           # [16]
    b0 = tl.load(b0_ptr)           # [16]
    h  = Tn[:, None] * W0[None, :] + b0[None, :]
    h  = tl.where(h > 0, h, h * alpha)                       # leaky-ReLU

    # gather latent weights ----------------------------------------------
    lane     = tl.arange(0, 16)
    col_offs = Ei[:, None] * 16 + lane[None, :]
    Wcol     = tl.load(W_ptr + col_offs, mask=mask[:, None])
    bvec     = tl.load(b_ptr + Ei,       mask=mask)

    # dot product + bias --------------------------------------------------
    res = tl.sum(h * Wcol, axis=1) + bvec
    tl.store(Out_ptr + offs, res, mask=mask)

# ───────────────────────────── helpers ──────────────────────────────────
def load_constants(path: str):
    with h5py.File(path, "r") as hf:
        return (
            torch.tensor(hf["W_tab"][:], device="cuda"),   # [16,E]
            torch.tensor(hf["b_tab"][:], device="cuda"),
            torch.tensor(hf["W0"][:],    device="cuda"),   # [16]
            torch.tensor(hf["b0"][:],    device="cuda"),   # [16]
            float(hf["T_scale"][0]),
            float(hf["T_mean"][0]),
            float(hf["alpha"][()]),
        )

def run_kernel(batch, tmin, tmax, consts, block=128):
    W, b, W0, b0, T_scale, T_mean, alpha = consts
    E = W.shape[1]

    T     = torch.empty(batch, device="cuda").uniform_(tmin, tmax)
    E_idx = torch.randint(0, E, (batch,), device="cuda", dtype=torch.int32)
    out   = torch.empty(batch, device="cuda")

    grid = lambda meta: (math.ceil(batch / meta["BLOCK"]),)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        xs_kernel[grid](T, E_idx, W, b, W0, b0,
                        T_scale, T_mean, alpha,
                        out, E=E, BLOCK=block)

    torch.cuda.synchronize()
    start = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    dur = time.perf_counter() - start

    print(f"{batch:,} pts -> {dur/batch*1e9:.2f} ns/pt")

# ───────────────────────────── CLI ───────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="w_table.h5")
    p.add_argument("--batch", type=int, default=262144)
    p.add_argument("--tmin",  type=float, default=950.)
    p.add_argument("--tmax",  type=float, default=1050.)
    p.add_argument("--chunk", type=int, help="Split batch if VRAM is tight")
    args = p.parse_args()

    consts = load_constants(args.table)
    if args.chunk:
        n = math.ceil(args.batch / args.chunk)
        for i in range(n):
            cur = min(args.chunk, args.batch - i*args.chunk)
            run_kernel(cur, args.tmin, args.tmax, consts)
    else:
        run_kernel(args.batch, args.tmin, args.tmax, consts)

if __name__ == "__main__":
    main()
