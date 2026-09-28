# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Sweep GROUP/BLOCK_D for the K1 heads+y kernel and isolate the slow part."""

import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu
import triton

from mindspeed_llm.ops.triton.mhc_lite_heads import lite_heads_y_fwd_kernel

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

BS, E, D = 4096, 4, 1024
NP = 24
N_TOTAL = 8 + NP

logits = torch.randn(BS, N_TOTAL, device=DEV, dtype=torch.float32)
x = torch.randn(BS, E, D, device=DEV, dtype=torch.bfloat16)
scale = torch.tensor([0.011, 0.013, 0.017], device=DEV, dtype=torch.float32)
base = torch.randn(N_TOTAL, device=DEV, dtype=torch.float32) * 0.5
perm_flat = torch.randn(NP, 16, device=DEV, dtype=torch.float32).softmax(dim=0)
perm_cols = perm_flat.t().contiguous()

y_ref = None


def run(group, block_d, check_ref=False):
    global y_ref
    y = torch.empty((BS, D), device=DEV, dtype=torch.bfloat16)
    h_pre = torch.empty((BS, 4), device=DEV, dtype=torch.bfloat16)
    h_post = torch.empty((BS, 4), device=DEV, dtype=torch.float32)
    h_res = torch.empty((BS, 16), device=DEV, dtype=torch.float32)
    grid = (triton.cdiv(BS, group),)
    fn = lambda: lite_heads_y_fwd_kernel[grid](  # noqa: E731
        logits, x, y, h_pre, h_post, h_res, perm_cols, scale, base, BS,
        D=D, NP=NP, GROUP=group, BLOCK_D=block_d,
    )
    fn()
    torch.npu.synchronize()
    if check_ref:
        y_ref = y.clone()
        return None
    err = (y.float() - y_ref.float()).abs().max().item()
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(30):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / 30 * 1e3, err


ok = run(4, 1024, check_ref=True)
for group, block_d in [(1, 1024), (2, 1024), (4, 1024), (8, 1024), (2, 512), (4, 512), (4, 256), (8, 512), (16, 1024)]:
    try:
        t, err = run(group, block_d)
        print(f'GROUP={group:3d} BLOCK_D={block_d:5d}: {t:7.3f} ms  (vs G4D1024 y maxdiff {err:.1e})')
    except Exception as exc:  # noqa: BLE001
        print(f'GROUP={group:3d} BLOCK_D={block_d:5d}: FAIL {repr(exc)[:120]}')

# reference: torch chain on the same inputs
pre_l, post_l, res_l = logits.split([4, 4, NP], dim=-1)
h_pre = torch.sigmoid(pre_l * scale[0] + base[:4])
coeff = torch.softmax(res_l * scale[2] + base[8:], dim=-1)


def torch_chain():
    h_res = coeff @ perm_flat
    y = torch.einsum('be,beh->bh', h_pre, x.float()).to(torch.bfloat16)
    return y, h_res


for _ in range(3):
    torch_chain()
torch.npu.synchronize()
t0 = time.time()
for _ in range(30):
    torch_chain()
torch.npu.synchronize()
print(f'torch heads+y chain (no gemm/norm): {(time.time() - t0) / 30 * 1e3:.3f} ms')
