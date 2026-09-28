# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Time the restructured Tier-1 kernels standalone (K1' heads+y, K2 heads-bwd, K3 post-bwd)."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from mindspeed_llm.ops.triton.mhc_lite_heads import (
    lite_heads_y_forward,
    lite_res_head_forward,
    lite_heads_backward,
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
perm_flat = torch.rand(NP, 16, device=DEV, dtype=torch.float32).softmax(dim=0)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


t = bench(lambda: lite_heads_y_forward(logits, x, scale, base))
print(f"K1' heads+y          : {t:.3f} ms")
t = bench(lambda: lite_res_head_forward(logits, scale, base, perm_flat))
print(f"res head (torch)     : {t:.3f} ms")

ghpre = torch.randn(BS, 4, device=DEV, dtype=torch.float32)
ghpost = torch.randn(BS, 4, device=DEV, dtype=torch.float32)
dcoeff = torch.randn(BS, NP, device=DEV, dtype=torch.float32)
t = bench(lambda: lite_heads_backward(ghpre, ghpost, dcoeff, logits, scale, base))
print(f"K2 heads bwd         : {t:.3f} ms")

g = torch.randn(BS, E, D, device=DEV, dtype=torch.bfloat16)
h_out = torch.randn(BS, D, device=DEV, dtype=torch.bfloat16)
h_post = torch.rand(BS, 4, device=DEV, dtype=torch.float32)
h_res = torch.rand(BS, 16, device=DEV, dtype=torch.float32)
t = bench(lambda: lite_post_backward(g, h_out, x, h_post, h_res))
print(f"K3 post bwd          : {t:.3f} ms")

# control: the existing repo kernel, same geometry (GROUP=2, BLOCK_D=D)
from mindspeed_llm.ops.triton.mhc_pre_only import hc_pre_bmm_forward  # noqa: E402

h_pre4 = torch.rand(BS, 4, device=DEV, dtype=torch.float32).softmax(-1)
t = bench(lambda: hc_pre_bmm_forward(h_pre4.view(1, BS, 4), x.view(1, BS, E, D)))
print(f"existing hc_pre_bmm_fwd (G=2): {t:.3f} ms")

t = bench(lambda: lite_heads_y_forward(logits, x, scale, base))
print(f"K1' recheck                    : {t:.3f} ms")
