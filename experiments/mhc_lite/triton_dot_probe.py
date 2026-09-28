# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Probe triton-npu capabilities needed by the Tier-1 lite kernel."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu
import triton
import triton.language as tl

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

print('triton backends:', __import__('os').listdir(__import__('os').path.join(__import__('os').path.dirname(triton.__file__), 'backends')))


@triton.jit
def dot_probe_kernel(
    x_ptr, w_ptr, y_ptr,
    BS, K: tl.constexpr, N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask_r = rows < BS
    acc = tl.zeros((GROUP, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=mask_r[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + ks[:, None] * N + ks[None, :] * 0, mask=None, other=0.0).to(tl.float32)  # [K?,N] placeholder
        acc += tl.dot(x, w)
    tl.store(y_ptr + rows[:, None] * N + tl.arange(0, N)[None, :], acc, mask=mask_r[:, None])


@triton.jit
def small_dot_kernel(
    x_ptr, w_ptr, y_ptr,
    BS, K: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
):
    # x: [BS,K] bf16, w: [K,N] fp32 -> y: [BS,N] fp32
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask_r = rows < BS
    acc = tl.zeros((GROUP, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=mask_r[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + ks[:, None] * N + tl.arange(0, N)[None, :]).to(tl.float32)
        acc = tl.dot(x, w, out_dtype=tl.float32)
    tl.store(y_ptr + rows[:, None] * N + tl.arange(0, N)[None, :], acc, mask=mask_r[:, None])


def ref(x, w):
    return x.float() @ w.float()


BS, K, N = 4096, 1024, 32
x = torch.randn(BS, K, device=DEV, dtype=torch.bfloat16)

print('--- bf16 x bf16 -> fp32 ---')
w = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
y = torch.empty(BS, N, device=DEV, dtype=torch.float32)
for group, bk in [(16, 128), (32, 256), (64, 512), (128, 256)]:
    try:
        grid = (triton.cdiv(BS, group),)
        small_dot_kernel[grid](x, w, y, BS, K=K, N=N, BLOCK_K=bk, GROUP=group)
        torch.npu.synchronize()
        err = (y - ref(x, w)).abs().max().item()
        print(f'tl.dot OK  GROUP={group} BLOCK_K={bk}  maxdiff={err:.2e}')
    except Exception as exc:  # noqa: BLE001
        print(f'tl.dot FAIL GROUP={group} BLOCK_K={bk}: {str(exc).splitlines()[0][:100]}')

print('--- fp32 x fp32 (input_precision default) ---')
w32 = torch.randn(K, N, device=DEV, dtype=torch.float32)
y32 = torch.empty(BS, N, device=DEV, dtype=torch.float32)
for group, bk in [(16, 256), (32, 256)]:
    try:
        grid = (triton.cdiv(BS, group),)
        small_dot_kernel[grid](x.float(), w32, y32, BS, K=K, N=N, BLOCK_K=bk, GROUP=group)
        torch.npu.synchronize()
        err = (y32 - ref(x, w32)).abs().max().item()
        print(f'tl.dot OK  GROUP={group} BLOCK_K={bk}  maxdiff={err:.2e}')
    except Exception as exc:  # noqa: BLE001
        print(f'tl.dot FAIL GROUP={group} BLOCK_K={bk}: {str(exc).splitlines()[0][:100]}')

print('--- manual mul+sum reduction (existing repo idiom) ---')


@triton.jit
def manual_gemm_kernel(
    x_ptr, w_ptr, y_ptr,
    BS, K: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
):
    # x: [BS,K] bf16, w: [K,N] bf16 -> y: [BS,N] fp32, one output column block per program
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask_r = rows < BS
    acc = tl.zeros((GROUP, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=mask_r[:, None], other=0.0)
        w = tl.load(w_ptr + ks[:, None] * N + tl.arange(0, N)[None, :])
        acc += tl.sum(x[:, :, None].to(tl.float32) * w[None, :, :].to(tl.float32), axis=1)
    tl.store(y_ptr + rows[:, None] * N + tl.arange(0, N)[None, :], acc, mask=mask_r[:, None])


def bench(fn, iters=20):
    import time
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


w = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
y = torch.empty(BS, N, device=DEV, dtype=torch.float32)
for group, bk in [(8, 512), (16, 512), (32, 1024)]:
    try:
        grid = (triton.cdiv(BS, group),)
        manual_gemm_kernel[grid](x, w, y, BS, K=K, N=N, BLOCK_K=bk, GROUP=group)
        torch.npu.synchronize()
        err = (y - ref(x, w)).abs().max().item()
        t_ms = bench(lambda: manual_gemm_kernel[grid](x, w, y, BS, K=K, N=N, BLOCK_K=bk, GROUP=group))
        print(f'manual OK GROUP={group} BLOCK_K={bk} maxdiff={err:.2e}  {t_ms:.3f} ms')
    except Exception as exc:  # noqa: BLE001
        print(f'manual FAIL GROUP={group} BLOCK_K={bk}: {repr(exc)[:300]}')

t_torch = bench(lambda: torch.nn.functional.linear(x.float(), w.t().float()))
print(f'torch fp32 linear (reference speed): {t_torch:.3f} ms')
