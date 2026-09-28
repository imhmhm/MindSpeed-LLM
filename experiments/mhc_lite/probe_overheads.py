# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Measure per-launch dispatch cost and python autograd overhead on NPU."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu
import triton
import triton.language as tl

torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')


@triton.jit
def tiny_kernel(ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(ptr + off, tl.load(ptr + off, mask=m, other=0.0) + 1.0, mask=m)


buf = torch.zeros(4096, device=DEV, dtype=torch.float32)


def rate(fn, iters=300):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


t = rate(lambda: tiny_kernel[(1,)](buf, N=4096, BLOCK=4096))
print(f'triton tiny kernel launch      : {t * 1e3:7.1f} us')

a = torch.ones(4096, device=DEV, dtype=torch.bfloat16)
b = torch.ones(4096, device=DEV, dtype=torch.bfloat16)
t = rate(lambda: torch.add(a, b))
print(f'torch.add [4096] launch        : {t * 1e3:7.1f} us')

big = torch.ones(4096, 4096, device=DEV, dtype=torch.bfloat16)
big2 = torch.ones(4096, 4096, device=DEV, dtype=torch.bfloat16)
t = rate(lambda: torch.add(big, big2), iters=100)
print(f'torch.add [4096x4096] (32MBx2) : {t * 1e3:7.1f} us')


class _TrivialFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x + 1.0

    @staticmethod
    def backward(ctx, g):
        return g


xa = torch.ones(4096, device=DEV, dtype=torch.bfloat16, requires_grad=True)
t = rate(lambda: _TrivialFn.apply(xa))
print(f'autograd Function fwd (1 op)   : {t * 1e3:7.1f} us')
t = rate(lambda: torch.autograd.grad(_TrivialFn.apply(xa).sum(), xa), iters=100)
print(f'autograd Function fwd+bwd      : {t * 1e3:7.1f} us')
yb = (xa + 1.0).sum()
t = rate(lambda: torch.autograd.grad((xa + 1.0).sum(), xa, retain_graph=True), iters=100)
print(f'plain autograd fwd+bwd         : {t * 1e3:7.1f} us')


def pre_call_launches():
    # launch count of one lite pre fwd+bwd (from the code path)
    return {
        'fwd': 'rms_norm + gemm + K1a + K1b + res(mul,softmax,matmul) = 7',
        'bwd': 'K4 + dcoeff + K2 + reduce + 2 casts + 2 gemms + rms_bwd + K5 ~= 11',
        'full_mhc': 'fused pre op: 1 fwd + 1 bwd = 2 (+wrapper copies)',
    }


for k, v in pre_call_launches().items():
    print(f'{k}: {v}')


# bypass JITFunction.run: pre-bound CompiledKernel launch
compiled = tiny_kernel.warmup(buf, N=4096, BLOCK=4096, grid=(1,))
compiled._init_handles()
try:
    t = rate(lambda: compiled[(1,)](buf))
    print(f'CompiledKernel direct launch   : {t * 1e3:7.1f} us')
except Exception as exc:  # noqa: BLE001
    print(f'CompiledKernel direct launch   : FAIL {repr(exc)[:120]}')

print('debug:', type(compiled).__name__, [a for a in dir(compiled) if not a.startswith('__')][:12])
import inspect  # noqa: E402
for name in ('run', '__getitem__', 'launch'):
    fn = getattr(compiled, name, None)
    if fn is not None:
        try:
            print(f'  {name}{inspect.signature(fn)}')
        except (ValueError, TypeError):
            print(f'  {name}: <no sig>')
for g in ((1, 1, 1), (1,)):
    try:
        t = rate(lambda: compiled[g](buf))
        print(f'direct launch grid={g}         : {t * 1e3:7.1f} us')
        break
    except Exception as exc:  # noqa: BLE001
        print(f'direct launch grid={g}         : FAIL {repr(exc)[:90]}')
