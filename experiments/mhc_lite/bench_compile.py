# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Naive torch.compile on the mhc_lite Tier-0 torch path: gain vs eager baseline.

Compiles the exact torch-op chain of the Tier-0 pre/post stages (the triton
path is hand-written kernels and not a compile target) with the backends
available on this torch_npu build, and times forward and forward+backward.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

SB, E, H, NP = 4096, 4, 1024, 24
N_TOTAL = 8 + NP
EPS = 1e-5


def pre_torch(x, weight, gamma, scale, base, perm_flat):
    xf = x.reshape(SB, E * H)
    xn, _ = torch_npu.npu_rms_norm(xf, gamma, epsilon=EPS)
    logits = torch.matmul(xn, weight.t()).float()
    pre, post, res = logits.split([E, E, NP], -1)
    h_pre = torch.sigmoid(pre * scale[0] + base[:E]).type_as(x)
    h_post = 2.0 * torch.sigmoid(post * scale[1] + base[E:2 * E])
    coeff = torch.softmax(res * scale[2] + base[2 * E:], -1)
    h_res = torch.matmul(coeff, perm_flat)
    y = torch.matmul(h_pre.reshape(SB, 1, E), x).reshape(SB, H)
    return y.type_as(x), h_post, h_res.view(SB, E, E)


def pre_torch_split(x, weight, gamma, s0, s1, s2, b_pre, b_post, b_res, perm_flat):
    # scale/base slices hoisted out of the compiled region: avoids the scalar
    # tl.broadcast_to pattern that the npu inductor codegen cannot compile
    xf = x.reshape(SB, E * H)
    xn, _ = torch_npu.npu_rms_norm(xf, gamma, epsilon=EPS)
    logits = torch.matmul(xn, weight.t()).float()
    pre, post, res = logits.split([E, E, NP], -1)
    h_pre = torch.sigmoid(pre * s0 + b_pre).type_as(x)
    h_post = 2.0 * torch.sigmoid(post * s1 + b_post)
    coeff = torch.softmax(res * s2 + b_res, -1)
    h_res = torch.matmul(coeff, perm_flat)
    y = torch.matmul(h_pre.reshape(SB, 1, E), x).reshape(SB, H)
    return y.type_as(x), h_post, h_res.view(SB, E, E)


def post_torch(x, residual, post, comb):
    y = x.reshape(SB, 1, H) * post.reshape(SB, E, 1).type_as(x)
    y = y + torch.matmul(comb.reshape(SB, E, E).transpose(-1, -2).type_as(x), residual)
    return y.type_as(x)


def make_pre():
    x = torch.randn(SB, E, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(N_TOTAL, E * H, device=DEV, dtype=torch.bfloat16) * 0.02).requires_grad_()
    gamma = torch.ones(E * H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    scale = torch.full((3,), 1e-2, device=DEV, dtype=torch.float32, requires_grad=True)
    base = torch.randn(N_TOTAL, device=DEV, dtype=torch.float32, requires_grad=True)
    perm_flat = torch.rand(NP, E * E, device=DEV, dtype=torch.float32).softmax(0)
    return x, weight, gamma, scale, base, perm_flat


def make_post():
    x = torch.randn(SB, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(SB, E, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    post = torch.rand(SB, E, device=DEV, dtype=torch.float32, requires_grad=True)
    comb = torch.rand(SB, E, E, device=DEV, dtype=torch.float32, requires_grad=True)
    return x, residual, post, comb


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


def run_case(name, fn, args, grads_of, grad_outputs):
    try:
        outs = fn(*args)
        base_out = [o.detach().clone() for o in outs]
        t_fwd = bench(lambda: fn(*args))

        def fwd_bwd():
            outs = fn(*args)
            torch.autograd.grad(outs, grads_of, grad_outputs)

        t_fb = bench(fwd_bwd, iters=20)
        outs = fn(*args)
        torch.autograd.grad(outs, grads_of, grad_outputs)
        out_err = max((a.float() - b.float()).abs().max().item() for a, b in zip(outs, base_out))
        print(f'{name:32s} fwd {t_fwd:7.3f} ms | fwd+bwd {t_fb:7.3f} ms | out maxdiff {out_err:.1e}')
    except Exception as exc:  # noqa: BLE001
        print(f'{name:32s} FAIL {repr(exc)[:160]}')


print(f'shape: x=[{SB},{E},{H}] bf16, mix={N_TOTAL}')

import torch_npu.dynamo.torchair as torchair  # noqa: E402

cfg = torchair.CompilerConfig()
NPU_BACKEND = torchair.get_npu_backend(compiler_config=cfg)

# post first: plain tensor math, no scalar indexing patterns
pargs = make_post()
pg = torch.randn(SB, E, H, device=DEV, dtype=torch.bfloat16)
print('--- post (Tier-0 torch formula) ---')
run_case('eager', post_torch, pargs, pargs, (pg,))
run_case('aot_eager', torch.compile(post_torch, backend='aot_eager'), pargs, pargs, (pg,))
run_case('inductor', torch.compile(post_torch, backend='inductor', dynamic=False), pargs, pargs, (pg,))
run_case('torchair (CANN graph)', torch.compile(post_torch, backend=NPU_BACKEND), pargs, pargs, (pg,))

# pre: full Tier-0 chain
args = make_pre()
x, weight, gamma, scale, base, perm_flat = args
gy = torch.randn(SB, H, device=DEV, dtype=torch.bfloat16)
gp = torch.randn(SB, E, device=DEV, dtype=torch.float32)
gc = torch.randn(SB, E, E, device=DEV, dtype=torch.float32)
inputs_pre = (x, weight, gamma, scale, base)
print('--- pre (Tier-0 torch chain) ---')
run_case('eager', pre_torch, args, inputs_pre, (gy, gp, gc))
run_case('aot_eager', torch.compile(pre_torch, backend='aot_eager'), args, inputs_pre, (gy, gp, gc))
run_case('inductor', torch.compile(pre_torch, backend='inductor', dynamic=False), args, inputs_pre, (gy, gp, gc))
run_case('torchair (CANN graph)', torch.compile(pre_torch, backend=NPU_BACKEND), args, inputs_pre, (gy, gp, gc))

# pre with scale/base slices hoisted out (works around the npu inductor codegen bug)
sargs = (x, weight, gamma, scale[0], scale[1], scale[2], base[:E], base[E:2 * E], base[2 * E:], perm_flat)
inputs_split = (x, weight, gamma, scale[0], scale[1], scale[2], base[:E], base[E:2 * E], base[2 * E:])
print('--- pre (scale/base sliced outside) ---')
run_case('eager', pre_torch_split, sargs, inputs_split, (gy, gp, gc))
run_case('inductor', torch.compile(pre_torch_split, backend='inductor', dynamic=False), sargs, inputs_split, (gy, gp, gc))
run_case('torchair (CANN graph)', torch.compile(pre_torch_split, backend=NPU_BACKEND), sargs, inputs_split, (gy, gp, gc))
