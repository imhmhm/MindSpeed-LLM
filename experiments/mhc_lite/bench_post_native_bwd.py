# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Time the native aclnnMhcPost backward vs the triton K3 replacement.

B=1 throughout: aclnnMhcPostBackward's first-call cold tiling failure only
bites B>=2 (see post_backward_order_probe.py), so the native backward is
benched the way the mainline full-MHC path uses it.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

import cann_ops_transformer

OPS = cann_ops_transformer.ops

torch.manual_seed(2)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 4096, 1, 4, 1024

h_out = torch.randn(S, B, H, device=DEV, dtype=torch.bfloat16)
x = torch.randn(S, B, E, H, device=DEV, dtype=torch.bfloat16)
h_post = torch.rand(S, B, E, device=DEV, dtype=torch.float32)
h_res = torch.rand(S, B, E, E, device=DEV, dtype=torch.float32)
g = torch.randn(S, B, E, H, device=DEV, dtype=torch.bfloat16)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


def op_call():
    xb = x.permute(1, 0, 2, 3).contiguous()
    hb = h_out.permute(1, 0, 2).contiguous()
    pb = h_post.permute(1, 0, 2).contiguous()
    rb = h_res.permute(1, 0, 2, 3).contiguous()
    return OPS.mhc_post(xb, rb, hb, pb), (xb, hb, pb, rb)


t = bench(lambda: op_call()[0])
print(f'post fwd (aclnn, wrapper permutes) : {t:.3f} ms')


def fwd_bwd_native():
    out, (xb, hb, pb, rb) = op_call()
    torch.autograd.grad(out, (xb, hb, pb, rb), g.permute(1, 0, 2, 3).contiguous())


xb = x.permute(1, 0, 2, 3).contiguous().requires_grad_()
hb = h_out.permute(1, 0, 2).contiguous().requires_grad_()
pb = h_post.permute(1, 0, 2).contiguous().requires_grad_()
rb = h_res.permute(1, 0, 2, 3).contiguous().requires_grad_()
gb = g.permute(1, 0, 2, 3).contiguous()


def fwd_bwd_premade():
    out = OPS.mhc_post(xb, rb, hb, pb)
    torch.autograd.grad(out, (xb, hb, pb, rb), gb)


try:
    t = bench(fwd_bwd_premade, iters=20)
    print(f'post fwd+bwd (aclnn native, pre-made layouts): {t:.3f} ms')
    t = bench(fwd_bwd_native, iters=20)
    print(f'post fwd+bwd (aclnn native, wrapper permutes) : {t:.3f} ms')
except Exception as exc:  # noqa: BLE001
    print(f'post fwd+bwd aclnn native: FAIL {repr(exc)[:120]}')

# current Tier-1 path for reference: fwd op + triton K3 backward
from mindspeed_llm.ops.triton.mhc_lite_heads import lite_post_backward  # noqa: E402

sb = S * B


def fwd_bwd_k3():
    out, _ = op_call()
    lite_post_backward(g.reshape(sb, E, H), h_out.reshape(sb, H), x.reshape(sb, E, H),
                       h_post.reshape(sb, E), h_res.reshape(sb, E * E))


t = bench(fwd_bwd_k3, iters=20)
print(f'post fwd+bwd (aclnn fwd + triton K3 bwd)      : {t:.3f} ms')
