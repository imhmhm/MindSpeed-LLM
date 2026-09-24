# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Shape matrix for the Ascend mhc_post backward: which (b, s) combos survive."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

import cann_ops_transformer

E, H = 4, 1024

for (b, s) in [(1, 512), (2, 512), (1, 4096), (2, 4096), (2, 1024), (1, 1024), (4, 512)]:
    x = torch.randn(b, s, H, device=DEV, dtype=torch.bfloat16).requires_grad_(True)          # sublayer out
    streams = torch.randn(b, s, E, H, device=DEV, dtype=torch.bfloat16).requires_grad_(True)  # residual
    post = torch.rand(b, s, E, device=DEV, dtype=torch.float32).requires_grad_(True)
    comb = torch.rand(b, s, E, E, device=DEV, dtype=torch.float32).requires_grad_(True)
    try:
        out = cann_ops_transformer.ops.mhc_post(streams, comb, x, post)
        loss = out.float().sum()
        loss.backward()
        ok = all(t.grad is not None and torch.isfinite(t.grad.float()).all() for t in (x, streams, post, comb))
        print(f"[b={b} s={s}] {'OK' if ok else 'GRAD-BAD'} out={tuple(out.shape)} {out.dtype}")
    except Exception as exc:  # noqa: BLE001
        print(f"[b={b} s={s}] FAIL: {str(exc).splitlines()[0][:120]}")
