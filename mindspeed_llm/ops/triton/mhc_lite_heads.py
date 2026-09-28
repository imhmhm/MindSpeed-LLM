# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

"""triton kernels for the mhc_lite pre/post stages (Tier 1).

The ascend triton backend supports neither tl.dot (wrong numerics) nor 3-D
broadcast reductions (MLIR compile failure), so these kernels stick to the
2-D load / elementwise / tl.sum idiom of mhc_pre_only.py.

- lite_heads_y_forward: logits [bs,32] + streams -> h_pre/h_post + y,
  replacing the split/sigmoid chain and the y bmm. Two kernels: a heads-only
  kernel and a y kernel that re-loads h_pre, because a [GROUP]-lane store of a
  value that is also consumed by wide tile math degrades that kernel ~75x on
  the ascend backend (see experiments/mhc_lite/bench_k1_bisect.py); splitting
  the store side from the consume side avoids it;
- lite_heads_backward: heads jacobian for the three lite heads, with the
  deterministic two-stage reduce for the scalar/vector params;
- lite_y_backward: y-path backward (dW_pre reduction + d_x_direct tiles in one
  kernel), plus lite_add_cast for the final d_x_direct + d_x_rms sum;
- lite_post_backward: fused backward of out_j = post_j*h_out + sum_i
  h_res[i,j]*x_i, replacing four einsums and their fp32 upcasts.
"""

import torch
import torch_npu

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    @triton.jit
    def lite_heads_fwd_kernel(
        logits_ptr,  # [bs, 32] fp32, layout [pre 4 | post 4 | res 24]
        h_pre_ptr,  # [bs, 4] bf16
        h_post_ptr,  # [bs, 4] fp32
        scale_ptr,  # [3] fp32
        base_ptr,  # [32] fp32
        bs,
        NP: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * GROUP + tl.arange(0, GROUP)
        mask = rows < bs

        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)

        row_off = rows * (8 + NP)
        pre_off = rows * 4
        for i in tl.static_range(4):
            l = tl.load(logits_ptr + row_off + i, mask=mask, other=0.0)
            h = tl.sigmoid(l * s0 + tl.load(base_ptr + i))
            tl.store(h_pre_ptr + pre_off + i, h.to(h_pre_ptr.dtype.element_ty), mask=mask)
        for i in tl.static_range(4):
            l = tl.load(logits_ptr + row_off + 4 + i, mask=mask, other=0.0)
            h = 2.0 * tl.sigmoid(l * s1 + tl.load(base_ptr + 4 + i))
            tl.store(h_post_ptr + pre_off + i, h, mask=mask)

    @triton.jit
    def lite_y_fwd_kernel(
        h_pre_ptr,  # [bs, 4] bf16
        x_ptr,  # [bs, 4, D] bf16
        y_ptr,  # [bs, D] bf16
        bs,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * GROUP + tl.arange(0, GROUP)
        mask = rows < bs

        pre_off = rows * 4
        w0 = tl.load(h_pre_ptr + pre_off + 0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(h_pre_ptr + pre_off + 1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(h_pre_ptr + pre_off + 2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(h_pre_ptr + pre_off + 3, mask=mask, other=0.0).to(tl.float32)

        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            x_off = rows[:, None] * (4 * D) + d[None, :]
            m2 = mask[:, None]
            x0 = tl.load(x_ptr + x_off + 0 * D, mask=m2, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + x_off + 1 * D, mask=m2, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + x_off + 2 * D, mask=m2, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + x_off + 3 * D, mask=m2, other=0.0).to(tl.float32)
            yv = w0[:, None] * x0 + w1[:, None] * x1 + w2[:, None] * x2 + w3[:, None] * x3
            tl.store(y_ptr + rows[:, None] * D + d[None, :], yv.to(y_ptr.dtype.element_ty), mask=m2)

    @triton.jit
    def lite_heads_bwd_kernel(
        ghpre_ptr,  # [bs, 4] fp32 grad wrt h_pre (from the y path)
        ghpost_ptr,  # [bs, 4] fp32
        dcoeff_ptr,  # [bs, 24] fp32 grad wrt the softmax coefficients
        logits_ptr,  # [bs, 32] fp32 (saved from forward)
        scale_ptr,  # [3] fp32
        base_ptr,  # [32] fp32
        dlogits_ptr,  # [bs, 32] fp32 out
        tmp_dscale_ptr,  # [nprog, 3] fp32 out
        tmp_dbase_ptr,  # [nprog, 32] fp32 out
        bs,
        NP: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * GROUP + tl.arange(0, GROUP)
        mask = rows < bs

        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        ar_n = tl.arange(0, NP)

        row_off = rows * (8 + NP)
        dscale0 = tl.zeros((), dtype=tl.float32)
        dscale1 = tl.zeros((), dtype=tl.float32)

        # pre head: h = sigmoid(s0*l + b)
        for i in tl.static_range(4):
            l = tl.load(logits_ptr + row_off + i, mask=mask, other=0.0)
            sig = tl.sigmoid(l * s0 + tl.load(base_ptr + i))
            gh = tl.load(ghpre_ptr + rows * 4 + i, mask=mask, other=0.0)
            dz = gh * sig * (1.0 - sig)  # grad wrt (s0*l + b)
            tl.store(dlogits_ptr + row_off + i, dz * s0, mask=mask)
            dscale0 += tl.sum(tl.where(mask, dz * l, 0.0))
            tl.store(tmp_dbase_ptr + pid * 32 + i, tl.sum(tl.where(mask, dz, 0.0)))

        # post head: h = 2*sigmoid(s1*l + b)
        for i in tl.static_range(4):
            l = tl.load(logits_ptr + row_off + 4 + i, mask=mask, other=0.0)
            sig = tl.sigmoid(l * s1 + tl.load(base_ptr + 4 + i))
            gh = tl.load(ghpost_ptr + rows * 4 + i, mask=mask, other=0.0)
            dz = gh * 2.0 * sig * (1.0 - sig)
            tl.store(dlogits_ptr + row_off + 4 + i, dz * s1, mask=mask)
            dscale1 += tl.sum(tl.where(mask, dz * l, 0.0))
            tl.store(tmp_dbase_ptr + pid * 32 + 4 + i, tl.sum(tl.where(mask, dz, 0.0)))

        # res head: coeff = softmax(s2*l + b); dzc = coeff*(dcoeff - sum(dcoeff*coeff))
        l_res = tl.load(logits_ptr + row_off[:, None] + (8 + ar_n)[None, :], mask=mask[:, None], other=0.0)
        z = l_res * s2 + tl.load(base_ptr + 8 + ar_n)[None, :]
        zmax = tl.max(z, axis=1)
        e = tl.exp(z - zmax[:, None])
        coeff = e / tl.sum(e, axis=1)[:, None]  # [GROUP, NP]
        dcoeff = tl.load(dcoeff_ptr + rows[:, None] * NP + ar_n[None, :], mask=mask[:, None], other=0.0)
        sdot = tl.sum(dcoeff * coeff, axis=1)  # [GROUP]
        dzc = coeff * (dcoeff - sdot[:, None])  # grad wrt z
        tl.store(dlogits_ptr + row_off[:, None] + (8 + ar_n)[None, :], dzc * s2, mask=mask[:, None])
        dscale2 = tl.sum(tl.where(mask[:, None], dzc * l_res, 0.0))
        tl.store(tmp_dbase_ptr + pid * 32 + 8 + ar_n, tl.sum(tl.where(mask[:, None], dzc, 0.0), axis=0))

        tl.store(tmp_dscale_ptr + pid * 4 + 0, dscale0)
        tl.store(tmp_dscale_ptr + pid * 4 + 1, dscale1)
        tl.store(tmp_dscale_ptr + pid * 4 + 2, dscale2)

    @triton.jit
    def lite_heads_bwd_reduce_kernel(
        tmp_dscale_ptr,
        tmp_dbase_ptr,
        dscale_ptr,  # [3] out
        dbase_ptr,  # [32] out
        nprog,
    ):
        if tl.program_id(0) != 0:
            return
        ar3 = tl.arange(0, 4)
        ar32 = tl.arange(0, 32)
        acc_scale = tl.zeros((4,), dtype=tl.float32)
        acc_base = tl.zeros((32,), dtype=tl.float32)
        for i in range(nprog):
            acc_scale += tl.load(tmp_dscale_ptr + i * 4 + ar3)
            acc_base += tl.load(tmp_dbase_ptr + i * 32 + ar32)
        tl.store(dscale_ptr + ar3, acc_scale)
        tl.store(dbase_ptr + ar32, acc_base)

    @triton.jit
    def lite_post_bwd_kernel(
        g_ptr,  # [bs, 4, D] bf16 grad of the post output
        h_out_ptr,  # [bs, D] bf16
        x_ptr,  # [bs, 4, D] bf16 streams
        h_post_ptr,  # [bs, 4] fp32
        h_res_ptr,  # [bs, 16] fp32 (row-major [i*4+j])
        dh_out_ptr,  # [bs, D] bf16 out
        dx_ptr,  # [bs, 4, D] bf16 out
        dh_post_ptr,  # [bs, 4] fp32 out
        dh_res_ptr,  # [bs, 16] fp32 out
        bs,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * GROUP + tl.arange(0, GROUP)
        mask = rows < bs

        p0 = tl.load(h_post_ptr + rows * 4 + 0, mask=mask, other=0.0)
        p1 = tl.load(h_post_ptr + rows * 4 + 1, mask=mask, other=0.0)
        p2 = tl.load(h_post_ptr + rows * 4 + 2, mask=mask, other=0.0)
        p3 = tl.load(h_post_ptr + rows * 4 + 3, mask=mask, other=0.0)

        # h_res[i, j] = h_res_flat[i*4 + j]
        r00 = tl.load(h_res_ptr + rows * 16 + 0, mask=mask, other=0.0)
        r01 = tl.load(h_res_ptr + rows * 16 + 1, mask=mask, other=0.0)
        r02 = tl.load(h_res_ptr + rows * 16 + 2, mask=mask, other=0.0)
        r03 = tl.load(h_res_ptr + rows * 16 + 3, mask=mask, other=0.0)
        r10 = tl.load(h_res_ptr + rows * 16 + 4, mask=mask, other=0.0)
        r11 = tl.load(h_res_ptr + rows * 16 + 5, mask=mask, other=0.0)
        r12 = tl.load(h_res_ptr + rows * 16 + 6, mask=mask, other=0.0)
        r13 = tl.load(h_res_ptr + rows * 16 + 7, mask=mask, other=0.0)
        r20 = tl.load(h_res_ptr + rows * 16 + 8, mask=mask, other=0.0)
        r21 = tl.load(h_res_ptr + rows * 16 + 9, mask=mask, other=0.0)
        r22 = tl.load(h_res_ptr + rows * 16 + 10, mask=mask, other=0.0)
        r23 = tl.load(h_res_ptr + rows * 16 + 11, mask=mask, other=0.0)
        r30 = tl.load(h_res_ptr + rows * 16 + 12, mask=mask, other=0.0)
        r31 = tl.load(h_res_ptr + rows * 16 + 13, mask=mask, other=0.0)
        r32 = tl.load(h_res_ptr + rows * 16 + 14, mask=mask, other=0.0)
        r33 = tl.load(h_res_ptr + rows * 16 + 15, mask=mask, other=0.0)

        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            m = mask[:, None]
            g_off = rows[:, None] * (4 * D) + d[None, :]
            g0 = tl.load(g_ptr + g_off + 0 * D, mask=m, other=0.0).to(tl.float32)
            g1 = tl.load(g_ptr + g_off + 1 * D, mask=m, other=0.0).to(tl.float32)
            g2 = tl.load(g_ptr + g_off + 2 * D, mask=m, other=0.0).to(tl.float32)
            g3 = tl.load(g_ptr + g_off + 3 * D, mask=m, other=0.0).to(tl.float32)
            ho = tl.load(h_out_ptr + rows[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32)

            # dh_out = sum_i post_i * g_i
            dh = p0[:, None] * g0 + p1[:, None] * g1 + p2[:, None] * g2 + p3[:, None] * g3
            tl.store(dh_out_ptr + rows[:, None] * D + d[None, :], dh.to(dh_out_ptr.dtype.element_ty), mask=m)

            # dx_i = sum_j h_res[i, j] * g_j
            x_off = rows[:, None] * (4 * D) + d[None, :]
            tl.store(dx_ptr + x_off + 0 * D, (r00[:, None] * g0 + r01[:, None] * g1 + r02[:, None] * g2 + r03[:, None] * g3).to(dx_ptr.dtype.element_ty), mask=m)
            tl.store(dx_ptr + x_off + 1 * D, (r10[:, None] * g0 + r11[:, None] * g1 + r12[:, None] * g2 + r13[:, None] * g3).to(dx_ptr.dtype.element_ty), mask=m)
            tl.store(dx_ptr + x_off + 2 * D, (r20[:, None] * g0 + r21[:, None] * g1 + r22[:, None] * g2 + r23[:, None] * g3).to(dx_ptr.dtype.element_ty), mask=m)
            tl.store(dx_ptr + x_off + 3 * D, (r30[:, None] * g0 + r31[:, None] * g1 + r32[:, None] * g2 + r33[:, None] * g3).to(dx_ptr.dtype.element_ty), mask=m)

            # dh_res[i, j] = <g_j, x_i>; dh_post_i = <g_i, h_out>
            # (full-D programs: per-row values, no atomics needed)
            x0 = tl.load(x_ptr + x_off + 0 * D, mask=m, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + x_off + 1 * D, mask=m, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + x_off + 2 * D, mask=m, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + x_off + 3 * D, mask=m, other=0.0).to(tl.float32)

            tl.store(dh_post_ptr + rows * 4 + 0, tl.sum(tl.where(m, g0 * ho, 0.0), axis=1), mask=mask)
            tl.store(dh_post_ptr + rows * 4 + 1, tl.sum(tl.where(m, g1 * ho, 0.0), axis=1), mask=mask)
            tl.store(dh_post_ptr + rows * 4 + 2, tl.sum(tl.where(m, g2 * ho, 0.0), axis=1), mask=mask)
            tl.store(dh_post_ptr + rows * 4 + 3, tl.sum(tl.where(m, g3 * ho, 0.0), axis=1), mask=mask)

            tl.store(dh_res_ptr + rows * 16 + 0, tl.sum(tl.where(m, g0 * x0, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 1, tl.sum(tl.where(m, g1 * x0, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 2, tl.sum(tl.where(m, g2 * x0, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 3, tl.sum(tl.where(m, g3 * x0, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 4, tl.sum(tl.where(m, g0 * x1, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 5, tl.sum(tl.where(m, g1 * x1, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 6, tl.sum(tl.where(m, g2 * x1, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 7, tl.sum(tl.where(m, g3 * x1, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 8, tl.sum(tl.where(m, g0 * x2, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 9, tl.sum(tl.where(m, g1 * x2, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 10, tl.sum(tl.where(m, g2 * x2, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 11, tl.sum(tl.where(m, g3 * x2, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 12, tl.sum(tl.where(m, g0 * x3, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 13, tl.sum(tl.where(m, g1 * x3, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 14, tl.sum(tl.where(m, g2 * x3, 0.0), axis=1), mask=mask)
            tl.store(dh_res_ptr + rows * 16 + 15, tl.sum(tl.where(m, g3 * x3, 0.0), axis=1), mask=mask)

    @triton.jit
    def lite_y_bwd_kernel(
        g_ptr,  # [bs, D] grad of the pre output y
        x_ptr,  # [bs, 4, D] bf16 streams
        h_pre_ptr,  # [bs, 4] bf16
        dw_ptr,  # [bs, 4] fp32 out
        dx_ptr,  # [bs, 4, D] out in x dtype
        bs,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # y = sum_i h_pre_i * x_i, so dx_i = h_pre_i * g and
        # dh_pre_i = <g, x_i>; both outputs are terminal stores
        pid = tl.program_id(0)
        rows = pid * GROUP + tl.arange(0, GROUP)
        mask = rows < bs

        pre_off = rows * 4
        w0 = tl.load(h_pre_ptr + pre_off + 0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(h_pre_ptr + pre_off + 1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(h_pre_ptr + pre_off + 2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(h_pre_ptr + pre_off + 3, mask=mask, other=0.0).to(tl.float32)

        dw0 = tl.zeros((GROUP,), dtype=tl.float32)
        dw1 = tl.zeros((GROUP,), dtype=tl.float32)
        dw2 = tl.zeros((GROUP,), dtype=tl.float32)
        dw3 = tl.zeros((GROUP,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            m = mask[:, None]
            g = tl.load(g_ptr + rows[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32)
            x_off = rows[:, None] * (4 * D) + d[None, :]
            x0 = tl.load(x_ptr + x_off + 0 * D, mask=m, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + x_off + 1 * D, mask=m, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + x_off + 2 * D, mask=m, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + x_off + 3 * D, mask=m, other=0.0).to(tl.float32)
            dw0 += tl.sum(tl.where(m, g * x0, 0.0), axis=1)
            dw1 += tl.sum(tl.where(m, g * x1, 0.0), axis=1)
            dw2 += tl.sum(tl.where(m, g * x2, 0.0), axis=1)
            dw3 += tl.sum(tl.where(m, g * x3, 0.0), axis=1)
            tl.store(dx_ptr + x_off + 0 * D, (w0[:, None] * g).to(dx_ptr.dtype.element_ty), mask=m)
            tl.store(dx_ptr + x_off + 1 * D, (w1[:, None] * g).to(dx_ptr.dtype.element_ty), mask=m)
            tl.store(dx_ptr + x_off + 2 * D, (w2[:, None] * g).to(dx_ptr.dtype.element_ty), mask=m)
            tl.store(dx_ptr + x_off + 3 * D, (w3[:, None] * g).to(dx_ptr.dtype.element_ty), mask=m)

        tl.store(dw_ptr + pre_off + 0, dw0, mask=mask)
        tl.store(dw_ptr + pre_off + 1, dw1, mask=mask)
        tl.store(dw_ptr + pre_off + 2, dw2, mask=mask)
        tl.store(dw_ptr + pre_off + 3, dw3, mask=mask)

    @triton.jit
    def lite_add_cast_kernel(
        a_ptr,  # [n] d_x_direct in x dtype
        b_ptr,  # [n] d_x_rms from the rms backward
        out_ptr,  # [n] out in a dtype
        n,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(a_ptr + off, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + off, (a + b).to(out_ptr.dtype.element_ty), mask=m)


def lite_heads_y_forward(
    logits: torch.Tensor,  # [bs, 32] fp32
    x: torch.Tensor,  # [bs, 4, D] bf16
    scale: torch.Tensor,  # [3] fp32
    base: torch.Tensor,  # [32] fp32
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs, four, d = x.shape
    assert four == 4
    np_ = logits.shape[-1] - 8
    y = torch.empty((bs, d), device=x.device, dtype=x.dtype)
    h_pre = torch.empty((bs, 4), device=x.device, dtype=x.dtype)
    h_post = torch.empty((bs, 4), device=x.device, dtype=torch.float32)
    heads_group = 32
    lite_heads_fwd_kernel[(triton.cdiv(bs, heads_group),)](  # pylint: disable=possibly-used-before-assignment
        logits, h_pre, h_post, scale, base, bs, NP=np_, GROUP=heads_group,
    )
    lite_y_fwd_kernel[(triton.cdiv(bs, 2),)](  # pylint: disable=possibly-used-before-assignment
        h_pre, x, y, bs, D=d, GROUP=2, BLOCK_D=d,
    )
    return y, h_pre, h_post


def lite_res_head_forward(
    logits: torch.Tensor,  # [bs, 32] fp32
    scale: torch.Tensor,  # [3] fp32
    base: torch.Tensor,  # [32] fp32
    perm_flat: torch.Tensor,  # [n_perm, e*e] fp32
):
    """h_res as a convex combination of permutation matrices (torch path)."""
    np_ = perm_flat.shape[0]
    z = logits[:, 8:] * scale[2] + base[8:]
    coeff = torch.softmax(z, dim=-1, dtype=torch.float32)
    return torch.matmul(coeff, perm_flat)


def lite_heads_backward(
    ghpre: torch.Tensor,  # [bs, 4] fp32
    ghpost: torch.Tensor,  # [bs, 4] fp32
    dcoeff: torch.Tensor,  # [bs, 24] fp32
    logits: torch.Tensor,  # [bs, 32] fp32
    scale: torch.Tensor,  # [3] fp32
    base: torch.Tensor,  # [32] fp32
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs = ghpre.shape[0]
    np_ = logits.shape[-1] - 8
    dlogits = torch.empty_like(logits)
    group = 32
    nprog = triton.cdiv(bs, group)
    tmp_dscale = torch.empty((nprog, 4), device=logits.device, dtype=torch.float32)
    tmp_dbase = torch.empty((nprog, 32), device=logits.device, dtype=torch.float32)
    dscale = torch.empty((4,), device=logits.device, dtype=torch.float32)
    dbase = torch.empty((32,), device=logits.device, dtype=torch.float32)
    lite_heads_bwd_kernel[(nprog,)](  # pylint: disable=possibly-used-before-assignment
        ghpre, ghpost, dcoeff, logits, scale, base,
        dlogits, tmp_dscale, tmp_dbase, bs, NP=np_, GROUP=group,
    )
    lite_heads_bwd_reduce_kernel[(1,)](  # pylint: disable=possibly-used-before-assignment
        tmp_dscale, tmp_dbase, dscale, dbase, nprog,
    )
    return dlogits, dscale[:3], dbase


def lite_y_backward(
    g: torch.Tensor,  # [bs, D] grad of y
    x: torch.Tensor,  # [bs, 4, D] bf16
    h_pre: torch.Tensor,  # [bs, 4]
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs, four, d = x.shape
    assert four == 4
    dw = torch.empty((bs, 4), device=x.device, dtype=torch.float32)
    dx = torch.empty_like(x)
    lite_y_bwd_kernel[(triton.cdiv(bs, 2),)](  # pylint: disable=possibly-used-before-assignment
        g, x, h_pre, dw, dx, bs, D=d, GROUP=2, BLOCK_D=d,
    )
    return dw, dx


def lite_add_cast(
    a: torch.Tensor,  # d_x_direct in x dtype
    b: torch.Tensor,  # d_x_rms
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    out = torch.empty_like(a)
    n = a.numel()
    lite_add_cast_kernel[(triton.cdiv(n, 4096),)](  # pylint: disable=possibly-used-before-assignment
        a, b, out, n, BLOCK=4096,
    )
    return out


def lite_post_backward(
    g: torch.Tensor,  # [bs, 4, D] bf16
    h_out: torch.Tensor,  # [bs, D] bf16
    x: torch.Tensor,  # [bs, 4, D] bf16
    h_post: torch.Tensor,  # [bs, 4] fp32
    h_res: torch.Tensor,  # [bs, 16] fp32
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs, four, d = x.shape
    dh_out = torch.empty((bs, d), device=x.device, dtype=x.dtype)
    dx = torch.empty((bs, 4, d), device=x.device, dtype=x.dtype)
    dh_post = torch.empty((bs, 4), device=x.device, dtype=torch.float32)
    dh_res = torch.empty((bs, 16), device=x.device, dtype=torch.float32)
    group = 2
    grid = (triton.cdiv(bs, group),)
    lite_post_bwd_kernel[grid](  # pylint: disable=possibly-used-before-assignment
        g, h_out, x, h_post, h_res, dh_out, dx, dh_post, dh_res,
        bs, D=d, GROUP=group, BLOCK_D=d,
    )
    return dh_out, dx, dh_post, dh_res
