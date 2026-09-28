# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Borrowed-idea experiments vs the mhc_sinkhorn baseline.

Sources: vllm-ascend hc_post MicroAPI (register FMA vector path), tilelang-ascend
mhc_post V0->V10 (AXPY beats broadcast+mul+reduce for tiny-hc contractions),
vllm-ascend hc_pre (fp32 cube GEMM, in-kernel sinkhorn).

Part A (torch level, post formula, S=4096 hc=4 h=1024 bf16):
  A1 mainline broadcast form (deepseek4/mhc.py hc_post)
  A2 bmm form (Tier-0 post_torch)
  A3 AXPY form (addcmul_ accumulation, the tilelang V4 insight at torch level)
  A4 aclnn mhc_post raw (op-native layout, no wrapper copies)
  A5 aclnn mhc_post through the npu_mhc wrapper (adds output clone)

Part B (kernel level, res head): lite single-softmax head vs full 20-iter
sinkhorn head, both as fused triton kernels (the vllm-ascend in-kernel sinkhorn
borrowing), plus the fp32-vs-bf16 logits GEMM comparison.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu
import triton
import triton.language as tl

from mindspeed_llm.ops.npu_mhc import mhc_post_ascend
from mindspeed_llm.ops.triton.mhc_lite_heads import _launch

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

BS, E, H, NP = 4096, 4, 1024, 24
EPS = 1e-6


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


print(f'== Part A: mhc_post forms (S={BS}, hc={E}, h={H}, bf16) ==')

x = torch.randn(BS, H, device=DEV, dtype=torch.bfloat16)
res = torch.randn(BS, E, H, device=DEV, dtype=torch.bfloat16)
post = torch.rand(BS, E, device=DEV, dtype=torch.float32) + 0.5
comb = torch.rand(BS, E, E, device=DEV, dtype=torch.float32).softmax(-1)

ref = post.type_as(res).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2).type_as(res), res)

t = bench(lambda: post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2).type_as(res))
print(f'A1 mainline broadcast form : {t:7.3f} ms')

t = bench(lambda: (x.reshape(BS, 1, H) * post.reshape(BS, E, 1).type_as(x) + torch.matmul(comb.transpose(-1, -2).type_as(x), res)).type_as(res))
print(f'A2 bmm form                : {t:7.3f} ms')


def axpy_form(x, res, post, comb):
    pb = post.to(res.dtype).unsqueeze(-1)
    cb = comb.to(res.dtype)
    out = pb * x.unsqueeze(-2)
    for i in range(E):
        for j in range(E):
            out[:, j].addcmul_(cb[:, i, j].unsqueeze(-1), res[:, i])
    return out


out3 = axpy_form(x, res, post, comb)
err3 = (out3.float() - ref.float()).abs().max().item()
t = bench(lambda: axpy_form(x, res, post, comb))
print(f'A3 AXPY form (addcmul_)    : {t:7.3f} ms   maxdiff {err3:.1e}')

# op-native layout at b=1 is [1, s, ...]; keep persistent buffers so only the
# wrapper's output clone differs between A4 and A5
x_op = x.reshape(1, BS, H).contiguous()
res_op = res.reshape(1, BS, E, H).contiguous()
post_op = post.reshape(1, BS, E).contiguous()
comb_op = comb.reshape(1, BS, E, E).contiguous()

import cann_ops_transformer  # noqa: E402

ops = cann_ops_transformer.ops
out4 = ops.mhc_post(res_op, comb_op, x_op, post_op)
err4 = (out4.reshape(BS, E, H).float() - ref.float()).abs().max().item()
t = bench(lambda: ops.mhc_post(res_op, comb_op, x_op, post_op))
print(f'A4 aclnn mhc_post raw      : {t:7.3f} ms   maxdiff {err4:.1e}')

x_m = x.reshape(BS, 1, H)
res_m = res.reshape(BS, 1, E, H)
post_m = post.reshape(BS, 1, E)
comb_m = comb.reshape(BS, 1, E, E)
t = bench(lambda: mhc_post_ascend(x_m, res_m, post_m, comb_m))
print(f'A5 aclnn via npu_mhc wrap  : {t:7.3f} ms')

print()
print('== Part B: res head, kernel level ==')

logits = torch.randn(BS, 8 + NP, device=DEV, dtype=torch.float32)
scale = torch.tensor([0.011, 0.013, 0.017], device=DEV, dtype=torch.float32)
base = torch.randn(8 + NP, device=DEV, dtype=torch.float32) * 0.5
perm_flat = torch.rand(NP, 16, device=DEV, dtype=torch.float32).softmax(0)
perm_t = perm_flat.t().contiguous()


def torch_lite_head():
    z = logits[:, 8:] * scale[2] + base[8:]
    coeff = torch.softmax(z, -1)
    return torch.matmul(coeff, perm_flat)


ref_lite = torch_lite_head()


logits_sk = torch.randn(BS, 16, device=DEV, dtype=torch.float32)
base_sk = torch.randn(16, device=DEV, dtype=torch.float32) * 0.5


def torch_sinkhorn_head():
    m = logits_sk.view(BS, E, E) * scale[2] + base_sk.view(1, E, E)
    m = m.softmax(-1) + EPS
    m = m / (m.sum(-2, keepdim=True) + EPS)
    for _ in range(19):
        m = m / (m.sum(-1, keepdim=True) + EPS)
        m = m / (m.sum(-2, keepdim=True) + EPS)
    return m.reshape(BS, 16)


ref_sk = torch_sinkhorn_head()


@triton.jit
def lite_head_kernel(
    logits_ptr, out_ptr, perm_ptr, scale_ptr, base_ptr, bs,
    NP: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s2 = tl.load(scale_ptr + 2)
    ar_n = tl.arange(0, NP)
    l = tl.load(logits_ptr + rows[:, None] * (8 + NP) + (8 + ar_n)[None, :], mask=mask[:, None], other=0.0)
    z = l * s2 + tl.load(base_ptr + 8 + ar_n)[None, :]
    zmax = tl.max(z, axis=1)
    e = tl.exp(z - zmax[:, None])
    coeff = e / tl.sum(e, axis=1)[:, None]
    for j in tl.static_range(16):
        col = tl.load(perm_ptr + j * NP + ar_n)
        v = tl.sum(coeff * col[None, :], axis=1)
        tl.store(out_ptr + rows * 16 + j, v, mask=mask)


@triton.jit
def sinkhorn_head_kernel(
    logits_ptr, out_ptr, scale_ptr, base_ptr, bs,
    GROUP: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s2 = tl.load(scale_ptr + 2)
    ar4 = tl.arange(0, 4)

    # four row slices m0..m3 stay in registers; row/col sums are slice-local
    m0 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m1 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m2 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m3 = tl.zeros((GROUP, 4), dtype=tl.float32)
    row_off = rows[:, None] * 16 + ar4[None, :]
    for r in tl.static_range(4):
        l = tl.load(logits_ptr + row_off + r * 4, mask=mask[:, None], other=0.0)
        z = l * s2 + tl.load(base_ptr + r * 4 + ar4)[None, :]
        zmax = tl.max(z, axis=1)
        e = tl.exp(z - zmax[:, None])
        s = e / tl.sum(e, axis=1)[:, None] + EPS
        if r == 0:
            m0 = s
        elif r == 1:
            m1 = s
        elif r == 2:
            m2 = s
        else:
            m3 = s
    colsum = m0 + m1 + m2 + m3
    m0 = m0 / (colsum + EPS)
    m1 = m1 / (colsum + EPS)
    m2 = m2 / (colsum + EPS)
    m3 = m3 / (colsum + EPS)
    for _ in range(ITERS - 1):
        m0 = m0 / (tl.sum(m0, axis=1)[:, None] + EPS)
        m1 = m1 / (tl.sum(m1, axis=1)[:, None] + EPS)
        m2 = m2 / (tl.sum(m2, axis=1)[:, None] + EPS)
        m3 = m3 / (tl.sum(m3, axis=1)[:, None] + EPS)
        colsum = m0 + m1 + m2 + m3
        m0 = m0 / (colsum + EPS)
        m1 = m1 / (colsum + EPS)
        m2 = m2 / (colsum + EPS)
        m3 = m3 / (colsum + EPS)
    tl.store(out_ptr + rows[:, None] * 16 + ar4[None, :], m0, mask=mask[:, None])
    tl.store(out_ptr + rows[:, None] * 16 + 4 + ar4[None, :], m1, mask=mask[:, None])
    tl.store(out_ptr + rows[:, None] * 16 + 8 + ar4[None, :], m2, mask=mask[:, None])
    tl.store(out_ptr + rows[:, None] * 16 + 12 + ar4[None, :], m3, mask=mask[:, None])


def run_lite():
    out = torch.empty(BS, 16, device=DEV, dtype=torch.float32)
    _launch(lite_head_kernel, ('lite_head', BS), (triton.cdiv(BS, 32),),
            (logits, out, perm_t, scale, base, BS), {'NP': NP, 'GROUP': 32})
    return out


def run_sinkhorn():
    out = torch.empty(BS, 16, device=DEV, dtype=torch.float32)
    _launch(sinkhorn_head_kernel, ('sk_head', BS), (triton.cdiv(BS, 32),),
            (logits_sk, out, scale, base_sk, BS), {'GROUP': 32, 'ITERS': 20, 'EPS': EPS})
    return out


o = run_lite()
torch.npu.synchronize()
err = (o - ref_lite).abs().max().item()
t = bench(run_lite)
print(f'B1 lite softmax head kernel: {t:7.3f} ms   maxdiff {err:.1e}')
t = bench(torch_lite_head)
print(f'   torch reference         : {t:7.3f} ms')

o = run_sinkhorn()
torch.npu.synchronize()
err = (o - ref_sk).abs().max().item()
t = bench(run_sinkhorn)
print(f'B2 full sinkhorn-20 kernel : {t:7.3f} ms   maxdiff {err:.1e}')
t = bench(torch_sinkhorn_head)
print(f'   torch reference         : {t:7.3f} ms')

print()
print('== Part C: logits GEMM precision (vllm-ascend casts to fp32 for the cube) ==')

xn = torch.randn(BS, E * H, device=DEV, dtype=torch.bfloat16)
w = (torch.randn(8 + NP, E * H, device=DEV, dtype=torch.bfloat16) * 0.02)

log_bf16 = torch.matmul(xn, w.t()).float()
t = bench(lambda: torch.matmul(xn, w.t()))
print(f'C1 bf16 GEMM               : {t:7.3f} ms')

xn32 = xn.float()
w32 = w.float()
log_fp32 = torch.matmul(xn32, w32.t())
t = bench(lambda: torch.matmul(xn.float(), w.float().t()))
print(f'C2 fp32 GEMM (with casts)  : {t:7.3f} ms')

ref64 = torch.matmul(xn.double(), w.double().t()).float()
e1 = (log_bf16 - ref64).abs().max().item()
e2 = (log_fp32 - ref64).abs().max().item()
print(f'   maxdiff vs fp64: bf16 {e1:.2e} | fp32 {e2:.2e}')
