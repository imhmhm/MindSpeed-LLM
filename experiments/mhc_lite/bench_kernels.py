# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Time the Tier-2 kernels standalone (heads+y fwd, pre bwd, grad_x, post bwd)."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from mindspeed_llm.ops.triton.mhc_lite_heads import (
    lite_heads_y_forward,
    lite_pre_backward,
    lite_grad_x,
    lite_post_backward,
)

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

BS, E, D, NP = 4096, 4, 1024, 24
N_TOTAL = 8 + NP

logits = torch.randn(BS, N_TOTAL, device=DEV, dtype=torch.float32)
x = torch.randn(BS, E, D, device=DEV, dtype=torch.bfloat16)
scale = torch.tensor([0.011, 0.013, 0.017], device=DEV, dtype=torch.float32)
base = torch.randn(N_TOTAL, device=DEV, dtype=torch.float32) * 0.5
perm_t = torch.rand(16, NP, device=DEV, dtype=torch.float32).softmax(dim=0)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


y, h_pre, h_post, h_res = lite_heads_y_forward(logits, x, scale, base, perm_t)
torch.npu.synchronize()

t = bench(lambda: lite_heads_y_forward(logits, x, scale, base, perm_t))
print(f'K1 heads+y fwd (K1a+K1b)  : {t:.3f} ms')

ghpost = torch.randn(BS, 4, device=DEV, dtype=torch.float32)
ghres = torch.randn(BS, E * E, device=DEV, dtype=torch.float32)
g = torch.randn(BS, D, device=DEV, dtype=torch.bfloat16)
t = bench(lambda: torch.matmul(ghres, perm_t))
print(f'dcoeff matmul (folded)    : {t:.3f} ms')
t = bench(lambda: lite_pre_backward(g, x, ghpost, ghres, perm_t, logits, scale, base))
print(f'megaK pre bwd (+reduce)   : {t:.3f} ms')

d_x_rms = torch.randn(BS, E, D, device=DEV, dtype=torch.bfloat16)
t = bench(lambda: lite_grad_x(g, h_pre, d_x_rms))
print(f'grad_x (recompute)        : {t:.3f} ms')

h_out = torch.randn(BS, D, device=DEV, dtype=torch.bfloat16)
gg = torch.randn(BS, E, D, device=DEV, dtype=torch.bfloat16)
t = bench(lambda: lite_post_backward(gg, h_out, x, h_post, h_res.view(BS, 16)))
print(f'K3 post bwd               : {t:.3f} ms')
