# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

"""triton kernels for the mhc_lite pre/post stages (Tier 2).

The ascend triton backend supports neither tl.dot (wrong numerics) nor 3-D
broadcast reductions (MLIR compile failure), so these kernels stick to the
2-D load / elementwise / tl.sum idiom of mhc_pre_only.py.

Two backend properties shape the design:
- a kernel that stores a [GROUP]-lane vector whose value also feeds wide
  tile math degrades ~75x at GROUP >= 2, so storing and consuming sides live
  in separate kernels (see experiments/mhc_lite/bench_k1_bisect.py);
- JITFunction dispatch costs ~105us per launch (a CompiledKernel direct call
  is ~49us), so every wrapper launches through _launch's compiled cache.

Kernels:
- lite_heads_fwd_kernel: logits [bs,32] -> h_pre/h_post/h_res (all three
  heads, terminal stores) and lite_y_fwd_kernel: y = sum_i h_pre_i*x_i
  re-loading h_pre from memory;
- lite_pre_bwd_kernel: fused backward of the heads+y chain (dW reduction +
  three-head jacobian -> dlogits/dscale/dbase partials), with
  lite_heads_bwd_reduce_kernel finishing the scalar/vector params;
- lite_grad_x_kernel: grad_x = h_pre*g + d_x_rms, recomputing the direct
  term instead of materializing it (saves a 32MB round trip per call);
- lite_post_bwd_kernel: fused backward of out_j = post_j*h_out + sum_i
  h_res[i,j]*x_i.
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

    _COMPILED_KERNELS = {}

    def _launch(jit_fn, key, grid, args, constexprs):
        """Launch through a cached CompiledKernel, falling back to the
        JITFunction path on any API mismatch.

        `key` must cover every value that changes compilation or
        specialization: shapes, dtypes and runtime ints.
        """
        ck = _COMPILED_KERNELS.get(key)
        if ck is None:
            try:
                ck = jit_fn.warmup(*args, grid=grid, **constexprs)
                ck._init_handles()
            except Exception:  # noqa: BLE001  (older triton APIs)
                ck = False
            _COMPILED_KERNELS[key] = ck
        if ck:
            try:
                ck[(grid[0], 1, 1)](*args)
                return
            except Exception:  # noqa: BLE001
                _COMPILED_KERNELS[key] = False
        jit_fn[grid](*args, **constexprs)

    @triton.jit
    def lite_heads_fwd_kernel(
        logits_ptr,  # [bs, 32] fp32, layout [pre 4 | post 4 | res 24]
        h_pre_ptr,  # [bs, 4] bf16
        h_post_ptr,  # [bs, 4] fp32
        h_res_ptr,  # [bs, 16] fp32 out, row-major [i*4+j]
        perm_ptr,  # [16, NP] fp32, permutation matrices flattened and transposed
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
        s2 = tl.load(scale_ptr + 2)

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

        # res head: convex combination of the permutation matrices
        ar_n = tl.arange(0, NP)
        l_res = tl.load(logits_ptr + row_off[:, None] + (8 + ar_n)[None, :], mask=mask[:, None], other=0.0)
        z = l_res * s2 + tl.load(base_ptr + 8 + ar_n)[None, :]
        zmax = tl.max(z, axis=1)
        e = tl.exp(z - zmax[:, None])
        coeff = e / tl.sum(e, axis=1)[:, None]  # [GROUP, NP]
        for j in tl.static_range(16):
            col = tl.load(perm_ptr + j * NP + ar_n)
            v = tl.sum(coeff * col[None, :], axis=1)
            tl.store(h_res_ptr + rows * 16 + j, v, mask=mask)

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
    def lite_pre_bwd_kernel(
        g_ptr,  # [bs, D] grad of the pre output y
        x_ptr,  # [bs, 4, D] bf16 streams
        ghpost_ptr,  # [bs, 4] fp32
        ghres_ptr,  # [bs, EE] fp32 grad wrt the mixed permutation matrices
        perm_t_ptr,  # [EE, NP] fp32 permutation table
        logits_ptr,  # [bs, 32] fp32 saved from forward
        scale_ptr,  # [3] fp32
        base_ptr,  # [32] fp32
        dlogits_ptr,  # [bs, 32] fp32 out
        tmp_dscale_ptr,  # [nprog, 4] fp32 out
        tmp_dbase_ptr,  # [nprog, 32] fp32 out
        bs,
        D: tl.constexpr,
        NP: tl.constexpr,
        EE: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # y-path reduction dh_pre_i = <g, x_i>, then the three-head jacobian
        # for dlogits/dscale/dbase partials; dw stays in registers
        pid = tl.program_id(0)
        rows = pid * GROUP + tl.arange(0, GROUP)
        mask = rows < bs

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

        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        ar_n = tl.arange(0, NP)

        row_off = rows * (8 + NP)
        dscale0 = tl.zeros((), dtype=tl.float32)
        dscale1 = tl.zeros((), dtype=tl.float32)

        # pre head: h = sigmoid(s0*l + b), incoming grad dw_i in registers
        for i in tl.static_range(4):
            l = tl.load(logits_ptr + row_off + i, mask=mask, other=0.0)
            sig = tl.sigmoid(l * s0 + tl.load(base_ptr + i))
            dw = dw0 if i == 0 else (dw1 if i == 1 else (dw2 if i == 2 else dw3))
            dz = dw * sig * (1.0 - sig)
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
        # dcoeff = ghres @ perm_t, the softmax-coefficient grad, folded in here
        # so no separate tiny GEMM kernel is needed
        ar_ee = tl.arange(0, EE)
        ghres = tl.load(ghres_ptr + rows[:, None] * EE + ar_ee[None, :],
                        mask=mask[:, None], other=0.0).to(tl.float32)
        perm_t = tl.load(perm_t_ptr + ar_ee[:, None] * NP + ar_n[None, :])  # [EE, NP]
        dcoeff = tl.sum(ghres[:, :, None] * perm_t[None, :, :], axis=1)  # [GROUP, NP]
        sdot = tl.sum(dcoeff * coeff, axis=1)  # [GROUP]
        dzc = coeff * (dcoeff - sdot[:, None])
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
    def lite_grad_x_kernel(
        g_ptr,  # [bs, D] grad of y
        h_pre_ptr,  # [bs, 4]
        drms_ptr,  # [bs, 4, D] d_x_rms from the rms backward
        out_ptr,  # [bs, 4, D] out, grad_x = h_pre*g + d_x_rms
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
            m = mask[:, None]
            off = rows[:, None] * (4 * D) + d[None, :]
            g = tl.load(g_ptr + rows[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32)
            r0 = tl.load(drms_ptr + off + 0 * D, mask=m, other=0.0).to(tl.float32)
            r1 = tl.load(drms_ptr + off + 1 * D, mask=m, other=0.0).to(tl.float32)
            r2 = tl.load(drms_ptr + off + 2 * D, mask=m, other=0.0).to(tl.float32)
            r3 = tl.load(drms_ptr + off + 3 * D, mask=m, other=0.0).to(tl.float32)
            tl.store(out_ptr + off + 0 * D, (w0[:, None] * g + r0).to(out_ptr.dtype.element_ty), mask=m)
            tl.store(out_ptr + off + 1 * D, (w1[:, None] * g + r1).to(out_ptr.dtype.element_ty), mask=m)
            tl.store(out_ptr + off + 2 * D, (w2[:, None] * g + r2).to(out_ptr.dtype.element_ty), mask=m)
            tl.store(out_ptr + off + 3 * D, (w3[:, None] * g + r3).to(out_ptr.dtype.element_ty), mask=m)

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


def lite_heads_y_forward(
    logits: torch.Tensor,  # [bs, 32] fp32
    x: torch.Tensor,  # [bs, 4, D] bf16
    scale: torch.Tensor,  # [3] fp32
    base: torch.Tensor,  # [32] fp32
    perm_t: torch.Tensor,  # [16, n_perm] fp32 contiguous
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs, four, d = x.shape
    assert four == 4
    np_ = logits.shape[-1] - 8
    y = torch.empty((bs, d), device=x.device, dtype=x.dtype)
    h_pre = torch.empty((bs, 4), device=x.device, dtype=x.dtype)
    h_post = torch.empty((bs, 4), device=x.device, dtype=torch.float32)
    h_res = torch.empty((bs, 16), device=x.device, dtype=torch.float32)
    heads_group = 32
    grid = (triton.cdiv(bs, heads_group),)
    _launch(
        lite_heads_fwd_kernel, ('heads_fwd', bs, d, x.dtype), grid,
        (logits, h_pre, h_post, h_res, perm_t, scale, base, bs),
        {'NP': np_, 'GROUP': heads_group},
    )
    _launch(
        lite_y_fwd_kernel, ('y_fwd', bs, d, x.dtype), (triton.cdiv(bs, 2),),
        (h_pre, x, y, bs), {'D': d, 'GROUP': 2, 'BLOCK_D': d},
    )
    return y, h_pre, h_post, h_res


def lite_pre_backward(
    g: torch.Tensor,  # [bs, D] grad of y
    x: torch.Tensor,  # [bs, 4, D]
    ghpost: torch.Tensor,  # [bs, 4] fp32
    ghres: torch.Tensor,  # [bs, e*e] fp32 grad wrt the mixed permutation matrices
    perm_t: torch.Tensor,  # [e*e, NP] fp32 contiguous
    logits: torch.Tensor,  # [bs, 32] fp32
    scale: torch.Tensor,  # [3] fp32
    base: torch.Tensor,  # [32] fp32
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs = g.shape[0]
    d = g.shape[1]
    np_ = logits.shape[-1] - 8
    dlogits = torch.empty_like(logits)
    group = 2
    nprog = triton.cdiv(bs, group)
    tmp_dscale = torch.empty((nprog, 4), device=logits.device, dtype=torch.float32)
    tmp_dbase = torch.empty((nprog, 32), device=logits.device, dtype=torch.float32)
    dscale = torch.empty((4,), device=logits.device, dtype=torch.float32)
    dbase = torch.empty((32,), device=logits.device, dtype=torch.float32)
    _launch(
        lite_pre_bwd_kernel, ('pre_bwd', bs, d, x.dtype), (nprog,),
        (g, x, ghpost, ghres, perm_t, logits, scale, base, dlogits, tmp_dscale, tmp_dbase, bs),
        {'D': d, 'NP': np_, 'EE': ghres.shape[-1], 'GROUP': group, 'BLOCK_D': d},
    )
    _launch(
        lite_heads_bwd_reduce_kernel, ('bwd_reduce', nprog), (1,),
        (tmp_dscale, tmp_dbase, dscale, dbase, nprog), {},
    )
    return dlogits, dscale[:3], dbase


def lite_grad_x(
    g: torch.Tensor,  # [bs, D] grad of y
    h_pre: torch.Tensor,  # [bs, 4]
    d_x_rms: torch.Tensor,  # [bs, 4, D]
):
    if not TRITON_AVAILABLE:
        raise RuntimeError('triton is not available')
    bs, d = g.shape
    out = torch.empty_like(d_x_rms)
    _launch(
        lite_grad_x_kernel, ('grad_x', bs, d, d_x_rms.dtype), (triton.cdiv(bs, 2),),
        (g, h_pre, d_x_rms, out, bs), {'D': d, 'GROUP': 2, 'BLOCK_D': d},
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
    _launch(
        lite_post_bwd_kernel, ('post_bwd', bs, d, x.dtype), (triton.cdiv(bs, 2),),
        (g, h_out, x, h_post, h_res, dh_out, dx, dh_post, dh_res, bs),
        {'D': d, 'GROUP': 2, 'BLOCK_D': d},
    )
    return dh_out, dx, dh_post, dh_res
