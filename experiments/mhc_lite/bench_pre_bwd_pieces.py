# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Break down the triton pre-path backward cost piece by piece."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from mindspeed_llm.ops.triton.mhc_lite_heads import lite_heads_backward
from mindspeed_llm.ops.triton.mhc_pre_only import hc_pre_bmm_backward

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H, NP = 4096, 1, 4, 1024, 24
SB = S * B
xf = torch.randn(SB, E * H, device=DEV, dtype=torch.bfloat16)
xn, rstd = torch_npu.npu_rms_norm(xf, torch.ones(E * H, device=DEV, dtype=torch.bfloat16), epsilon=1e-5)
weight = torch.randn(8 + NP, E * H, device=DEV, dtype=torch.bfloat16) * 0.02
gamma = torch.ones(E * H, device=DEV, dtype=torch.bfloat16)
logits = torch.randn(SB, 8 + NP, device=DEV, dtype=torch.float32)
h_pre = torch.rand(SB, E, device=DEV, dtype=torch.bfloat16).softmax(-1).to(torch.bfloat16)
grad_y = torch.randn(SB, H, device=DEV, dtype=torch.bfloat16)
grad_h_post = torch.randn(SB, E, device=DEV, dtype=torch.float32)
grad_h_res = torch.randn(SB, E * E, device=DEV, dtype=torch.float32)
perm_t = torch.randn(E * E, NP, device=DEV, dtype=torch.float32)
dcoeff_in = torch.randn(SB, NP, device=DEV, dtype=torch.float32)
scale = torch.tensor([0.01, 0.01, 0.01], device=DEV, dtype=torch.float32)
base = torch.randn(8 + NP, device=DEV, dtype=torch.float32)
d_xn = torch.randn(SB, E * H, device=DEV, dtype=torch.bfloat16)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


t = bench(lambda: hc_pre_bmm_backward(h_pre.view(1, SB, E), xf.view(1, SB, E, H), grad_y.view(1, SB, H)))
print(f'b1 hc_pre_bmm_backward      : {t:.3f} ms')

t = bench(lambda: torch.matmul(grad_h_res, perm_t))
print(f'b2 dcoeff gemm              : {t:.3f} ms')

t = bench(lambda: lite_heads_backward(h_pre.view(SB, E).float(), grad_h_post, dcoeff_in, logits, scale, base)[0])
print(f'b3 K2 heads bwd             : {t:.3f} ms')

dlogits = torch.randn(SB, 8 + NP, device=DEV, dtype=torch.float32)
t = bench(lambda: torch.matmul(dlogits.t(), xn.float()))
print(f'b4 grad_weight gemm         : {t:.3f} ms')

t = bench(lambda: torch.matmul(dlogits, weight.float()))
print(f'b5 d_xn gemm                : {t:.3f} ms')

t = bench(lambda: torch_npu.npu_rms_norm_backward(d_xn, xf, gamma, rstd))
print(f'b6 rms_norm_backward        : {t:.3f} ms')

d_xr = torch.randn(SB, E * H, device=DEV, dtype=torch.float32)
d_xd = torch.randn(SB, E, H, device=DEV, dtype=torch.float32)
t = bench(lambda: (d_xd.view(SB, E, H).float() + d_xr.view(SB, E, H).float()).to(torch.bfloat16))
print(f'b7 add + cast grad_x        : {t:.3f} ms')
