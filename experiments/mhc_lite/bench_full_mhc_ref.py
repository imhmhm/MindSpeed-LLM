# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Time the full-MHC fused aclnnMhcPreSinkhorn forward at the lite bench shape."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 4096, 1, 4, 1024
MIX = (2 + E) * E

x = torch.randn(S, B, E, H, device=DEV, dtype=torch.bfloat16)
phi = (torch.randn(MIX, E * H, device=DEV, dtype=torch.float32) * 0.02).requires_grad_()
alpha = torch.zeros(3, device=DEV, dtype=torch.float32).requires_grad_()
bias = torch.zeros(MIX, device=DEV, dtype=torch.float32).requires_grad_()


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


t = bench(lambda: mhc_pre_sinkhorn_ascend(x, phi, alpha, bias, E, 20, 1e-3, 1e-5))
print(f'full-MHC aclnn pre fwd (sinkhorn iters=4): {t:.3f} ms')

y, post, comb = mhc_pre_sinkhorn_ascend(x, phi, alpha, bias, E, 20, 1e-3, 1e-5)
torch.npu.synchronize()
gy = torch.randn_like(y)
gp = torch.randn_like(post)
gc = torch.randn_like(comb)


def bwd():
    y, post, comb = mhc_pre_sinkhorn_ascend(x, phi, alpha, bias, E, 20, 1e-3, 1e-5)
    torch.autograd.grad((y, post, comb), (x, phi, alpha, bias), (gy, gp, gc))


try:
    t = bench(bwd, iters=10)
    print(f'full-MHC aclnn pre fwd+bwd            : {t:.3f} ms')
except Exception as exc:  # noqa: BLE001
    print(f'full-MHC aclnn pre fwd+bwd            : FAIL {repr(exc)[:100]}')
