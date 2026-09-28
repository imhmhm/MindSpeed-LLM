# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Time each stage of the Tier-0 MHCLite pre/post forward+backward at train shapes."""

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from megatron.training.global_vars import set_args
from megatron.core.transformer import TransformerConfig

from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.tasks.models.transformer.mhc_lite import MHCLite, MHCLiteSubmodules, _MhcPostFn
from mindspeed_llm.ops.npu_mhc import mhc_post_ascend

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 4096, 1, 4, 1024
EPS = 1e-5


def make_args():
    args = argparse.Namespace()
    args.hc_mult = E
    args.norm_epsilon = EPS
    args.enable_mhc = True
    args.use_triton_mhc = False
    args.use_fused_mhc = False
    args.fp8 = None
    return args


set_args(make_args())
module = MHCLite(
    TransformerConfig(hidden_size=H, num_layers=28, num_attention_heads=16,
                      ffn_hidden_size=4096, params_dtype=torch.bfloat16),
    MHCLiteSubmodules(hc_fn=LinearNoTP),
    mhc_position='attn',
    layer_number=5,
).to(DEV)
with torch.no_grad():
    module.hc_fn.weight.copy_(torch.randn_like(module.hc_fn.weight) * 0.02)

x = (torch.randn(S, B, E, H, device=DEV) * 1.5).to(torch.bfloat16).requires_grad_(True)
sub = torch.randn(S, B, H, device=DEV).to(torch.bfloat16)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


print(f'shapes: x=[{S},{B},{E},{H}] bf16 ({S*B*E*H*2/1e6:.0f} MB per stream tensor)')

# --- component timings (forward) ---
with torch.no_grad():
    xn_t = torch_npu.npu_rms_norm(x.reshape(S, B, -1), module.hc_gamma.type_as(x), epsilon=EPS)[0]
    logits_t = module.hc_fn(xn_t)

t = bench(lambda: torch_npu.npu_rms_norm(x.reshape(S, B, -1), module.hc_gamma.type_as(x), epsilon=EPS))
print(f'f01 rms_norm            : {t:.3f} ms')
t = bench(lambda: module.hc_fn(xn_t))
print(f'f02 logits gemm         : {t:.3f} ms')
t = bench(lambda: module._coefficients(x))
print(f'f03 heads(_coefficients): {t:.3f} ms')

with torch.no_grad():
    _, h_post0, h_res0 = module._coefficients(x)
    h_pre0 = torch.rand(S, B, E, device=DEV, dtype=torch.float32)
t = bench(lambda: torch.matmul(h_pre0.view(S * B, 1, E), x.view(S * B, E, H)))
print(f'f04 y bmm (torch)       : {t:.3f} ms')
t = bench(lambda: module.hc_pre(x))
print(f'f05 hc_pre total        : {t:.3f} ms')
t = bench(lambda: mhc_post_ascend(sub, x, h_post0, h_res0))
print(f'f06 mhc_post fwd (cann) : {t:.3f} ms')

# --- backward timings ---
y0, hp0, hr0 = module.hc_pre(x)
gy = torch.randn_like(y0)
ghp = torch.randn_like(hp0)
ghr = torch.randn_like(hr0)


def pre_backward():
    y, h_post, h_res = module.hc_pre(x)
    (y.float() * gy.float()).sum().add((h_post * ghp).sum()).add((h_res * ghr).sum()).backward()


t = bench(pre_backward)
print(f'b01 hc_pre fwd+bwd      : {t:.3f} ms')

post = sub.clone().requires_grad_(True)
res = x.detach().clone().requires_grad_(True)
p = hp0.detach().clone().requires_grad_(True)
c = hr0.detach().clone().requires_grad_(True)
out = _MhcPostFn.apply(post, res, p, c)
g = torch.randn_like(out)


def post_backward():
    o = _MhcPostFn.apply(post, res, p, c)
    o.backward(g)


t = bench(post_backward)
print(f'b02 hc_post fwd+bwd     : {t:.3f} ms')
