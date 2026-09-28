# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Bisect the K1' performance cliff by adding K1'-only ingredients one by one.

V1: y-only, weights preloaded from a [bs,4] tensor (existing-kernel analogue)
V2: V1 + inline sigmoid(pre logits) / 2*sigmoid(post logits), no head stores
V3: V2 + h_pre/h_post stores (== K1')
V4: V1 + linear heads (no sigmoid), with head stores
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

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

BS, E, D, NP = 4096, 4, 1024, 24

logits = torch.randn(BS, 8 + NP, device=DEV, dtype=torch.float32)
x = torch.randn(BS, E, D, device=DEV, dtype=torch.bfloat16)
scale = torch.tensor([0.011, 0.013], device=DEV, dtype=torch.float32)
base = torch.randn(8, device=DEV, dtype=torch.float32) * 0.5
w4 = torch.rand(BS, 4, device=DEV, dtype=torch.float32).softmax(-1)

y_out = torch.empty((BS, D), device=DEV, dtype=torch.bfloat16)
h_pre_out = torch.empty((BS, 4), device=DEV, dtype=torch.bfloat16)
h_post_out = torch.empty((BS, 4), device=DEV, dtype=torch.float32)


@triton.jit
def v1_y_only(
        w_ptr, x_ptr, y_ptr, bs,
        D: tl.constexpr, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    w0 = tl.load(w_ptr + rows * 4 + 0, mask=mask, other=0.0)
    w1 = tl.load(w_ptr + rows * 4 + 1, mask=mask, other=0.0)
    w2 = tl.load(w_ptr + rows * 4 + 2, mask=mask, other=0.0)
    w3 = tl.load(w_ptr + rows * 4 + 3, mask=mask, other=0.0)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = mask[:, None]
        x_off = rows[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
        yv = w0[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
        tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)


@triton.jit
def v2_sigmoid_nostore(
        logits_ptr, x_ptr, y_ptr, scale_ptr, base_ptr, bs,
        D: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    row_off = rows * (8 + NP)
    w0 = tl.sigmoid(tl.load(logits_ptr + row_off + 0, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 0))
    w1 = tl.sigmoid(tl.load(logits_ptr + row_off + 1, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 1))
    w2 = tl.sigmoid(tl.load(logits_ptr + row_off + 2, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 2))
    w3 = tl.sigmoid(tl.load(logits_ptr + row_off + 3, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 3))
    p0 = 2.0 * tl.sigmoid(tl.load(logits_ptr + row_off + 4, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 4))
    p1 = 2.0 * tl.sigmoid(tl.load(logits_ptr + row_off + 5, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 5))
    p2 = 2.0 * tl.sigmoid(tl.load(logits_ptr + row_off + 6, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 6))
    p3 = 2.0 * tl.sigmoid(tl.load(logits_ptr + row_off + 7, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 7))
    acc = tl.sum(p0 + p1 + p2 + p3) * 1e-30
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = mask[:, None]
        x_off = rows[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
        yv = (w0 + acc)[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
        tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)


@triton.jit
def v3_sigmoid_store(
        logits_ptr, x_ptr, y_ptr, h_pre_ptr, h_post_ptr, scale_ptr, base_ptr, bs,
        D: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    row_off = rows * (8 + NP)
    pre_off = rows * 4
    w0 = tl.sigmoid(tl.load(logits_ptr + row_off + 0, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 0))
    w1 = tl.sigmoid(tl.load(logits_ptr + row_off + 1, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 1))
    w2 = tl.sigmoid(tl.load(logits_ptr + row_off + 2, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 2))
    w3 = tl.sigmoid(tl.load(logits_ptr + row_off + 3, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 3))
    tl.store(h_pre_ptr + pre_off + 0, w0.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 1, w1.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 2, w2.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 3, w3.to(h_pre_ptr.dtype.element_ty), mask=mask)
    post_off = row_off + 4
    p0 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 0, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 4))
    p1 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 1, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 5))
    p2 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 2, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 6))
    p3 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 3, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 7))
    tl.store(h_post_ptr + pre_off + 0, p0, mask=mask)
    tl.store(h_post_ptr + pre_off + 1, p1, mask=mask)
    tl.store(h_post_ptr + pre_off + 2, p2, mask=mask)
    tl.store(h_post_ptr + pre_off + 3, p3, mask=mask)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = mask[:, None]
        x_off = rows[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
        yv = w0[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
        tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)


@triton.jit
def v4_linear_store(
        logits_ptr, x_ptr, y_ptr, h_pre_ptr, h_post_ptr, scale_ptr, base_ptr, bs,
        D: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    row_off = rows * (8 + NP)
    pre_off = rows * 4
    w0 = tl.load(logits_ptr + row_off + 0, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 0)
    w1 = tl.load(logits_ptr + row_off + 1, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 1)
    w2 = tl.load(logits_ptr + row_off + 2, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 2)
    w3 = tl.load(logits_ptr + row_off + 3, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 3)
    tl.store(h_pre_ptr + pre_off + 0, w0.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 1, w1.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 2, w2.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 3, w3.to(h_pre_ptr.dtype.element_ty), mask=mask)
    post_off = row_off + 4
    p0 = tl.load(logits_ptr + post_off + 0, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 4)
    p1 = tl.load(logits_ptr + post_off + 1, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 5)
    p2 = tl.load(logits_ptr + post_off + 2, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 6)
    p3 = tl.load(logits_ptr + post_off + 3, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 7)
    tl.store(h_post_ptr + pre_off + 0, p0, mask=mask)
    tl.store(h_post_ptr + pre_off + 1, p1, mask=mask)
    tl.store(h_post_ptr + pre_off + 2, p2, mask=mask)
    tl.store(h_post_ptr + pre_off + 3, p3, mask=mask)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = mask[:, None]
        x_off = rows[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
        yv = w0[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
        tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


for group in (2, 1):
    grid = (triton.cdiv(BS, group),)
    t = bench(lambda: v1_y_only[grid](w4, x, y_out, BS, D=D, GROUP=group, BLOCK_D=D))
    print(f'V1 y-only            G={group}: {t:7.3f} ms')
    t = bench(lambda: v2_sigmoid_nostore[grid](logits, x, y_out, scale, base, BS, D=D, NP=NP, GROUP=group, BLOCK_D=D))
    print(f'V2 sigmoid no-store  G={group}: {t:7.3f} ms')
    t = bench(lambda: v3_sigmoid_store[grid](logits, x, y_out, h_pre_out, h_post_out, scale, base, BS, D=D, NP=NP, GROUP=group, BLOCK_D=D))
    print(f'V3 sigmoid +store    G={group}: {t:7.3f} ms')
    t = bench(lambda: v4_linear_store[grid](logits, x, y_out, h_pre_out, h_post_out, scale, base, BS, D=D, NP=NP, GROUP=group, BLOCK_D=D))
    print(f'V4 linear   +store   G={group}: {t:7.3f} ms')



# finer bisect at GROUP=2: which small store (or dtype) triggers the cliff
@triton.jit
def v5_post_store_only(
        logits_ptr, x_ptr, y_ptr, h_post_ptr, scale_ptr, base_ptr, bs,
        D: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    row_off = rows * (8 + NP)
    w0 = tl.sigmoid(tl.load(logits_ptr + row_off + 0, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 0))
    w1 = tl.sigmoid(tl.load(logits_ptr + row_off + 1, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 1))
    w2 = tl.sigmoid(tl.load(logits_ptr + row_off + 2, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 2))
    w3 = tl.sigmoid(tl.load(logits_ptr + row_off + 3, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 3))
    post_off = row_off + 4
    p0 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 0, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 4))
    p1 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 1, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 5))
    p2 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 2, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 6))
    p3 = 2.0 * tl.sigmoid(tl.load(logits_ptr + post_off + 3, mask=mask, other=0.0) * s1 + tl.load(base_ptr + 7))
    pre_off = rows * 4
    tl.store(h_post_ptr + pre_off + 0, p0, mask=mask)
    tl.store(h_post_ptr + pre_off + 1, p1, mask=mask)
    tl.store(h_post_ptr + pre_off + 2, p2, mask=mask)
    tl.store(h_post_ptr + pre_off + 3, p3, mask=mask)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = mask[:, None]
        x_off = rows[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
        yv = w0[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
        tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)


@triton.jit
def v6_pre_store_only(
        logits_ptr, x_ptr, y_ptr, h_pre_ptr, scale_ptr, base_ptr, bs,
        D: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s0 = tl.load(scale_ptr + 0)
    row_off = rows * (8 + NP)
    w0 = tl.sigmoid(tl.load(logits_ptr + row_off + 0, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 0))
    w1 = tl.sigmoid(tl.load(logits_ptr + row_off + 1, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 1))
    w2 = tl.sigmoid(tl.load(logits_ptr + row_off + 2, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 2))
    w3 = tl.sigmoid(tl.load(logits_ptr + row_off + 3, mask=mask, other=0.0) * s0 + tl.load(base_ptr + 3))
    pre_off = rows * 4
    tl.store(h_pre_ptr + pre_off + 0, w0.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 1, w1.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 2, w2.to(h_pre_ptr.dtype.element_ty), mask=mask)
    tl.store(h_pre_ptr + pre_off + 3, w3.to(h_pre_ptr.dtype.element_ty), mask=mask)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = mask[:, None]
        x_off = rows[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
        yv = w0[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
        tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)


h_pre_f32 = torch.empty((BS, 4), device=DEV, dtype=torch.float32)
grid = (triton.cdiv(BS, 2),)
t = bench(lambda: v5_post_store_only[grid](logits, x, y_out, h_post_out, scale, base, BS, D=D, NP=NP, GROUP=2, BLOCK_D=D))
print(f'V5 h_post(fp32) store only G=2: {t:7.3f} ms')
t = bench(lambda: v6_pre_store_only[grid](logits, x, y_out, h_pre_out, scale, base, BS, D=D, NP=NP, GROUP=2, BLOCK_D=D))
print(f'V6 h_pre(bf16)  store only G=2: {t:7.3f} ms')
t = bench(lambda: v6_pre_store_only[grid](logits, x, y_out, h_pre_f32, scale, base, BS, D=D, NP=NP, GROUP=2, BLOCK_D=D))
print(f'V6 h_pre(fp32)  store only G=2: {t:7.3f} ms')
