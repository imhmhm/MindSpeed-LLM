# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Scheme C prototype: tilelang (Ascend C codegen) fused mhc_lite pre forward.

The non-GEMM part of the lite pre stage -- RMS square-sum, the three heads,
and the y AXPY -- is folded into ONE two-dimensional [ROWS=8, W] vector-core
kernel, so the per-op launch overhead that dominates a per-token kernel is
amortized over 8 tokens per core. The RMSNorm linearity
    xn = x * rstd * gamma   =>   xn @ W^T = rstd * (x @ (W * gamma)^T)
lets the logits GEMM consume x directly, so the 32 MB xn materialization of
the npu_rms_norm chain disappears; the only extra is W' = W * gamma, one
broadcast elementwise mul over a preallocated padded buffer.

Head groups are padded to 8-lane boundaries in the GEMM output (pre at
columns 0-3, post at 8-11, res at 16-39 of a 48-wide logits tensor): every
UB window the kernel slices is then 32B-aligned, which the vector core
requires. Kernel config follows the validated mhc_post V10 recipe:
pass_configs + clear=True on reduces + explicit broadcast axis.

The per-row y scalars cannot ride T.tile.axpy (it takes one scalar shared by
all rows); they are extracted from the [ROWS, 8] sigmoid result by a masked
reduce (hpre * onehot_i -> row sum) and broadcast-multiplied into x chunks,
the rms_grad idiom. h_res stays as coeff [sb, 24] and is mixed on the host
with one tiny GEMM for the same reason.

Three tilelang-ascend codegen rules this kernel works around, each verified
by reduction probes (experiments/mhc_lite/probe_tilelang_miscompiles.py):
  * a binary elementwise op (mul/add) whose operand is a window VIEW of a
    wider UB buffer miscompiles -- stage the window into a dedicated
    [ROWS, 8] buffer with T.copy first (this also covers the axpy case:
    T.tile.axpy with an element-scalar read is exact for row 0 and silently
    wrong ~3-5% for every other row);
  * a binary elementwise op whose operand is a [ROWS, 1] column buffer
    miscompiles -- column scalars must go through an explicit
    T.tile.broadcast(..., axis=1) first;
  * an in-place add whose destination is a column-slice REGION of a wider
    UB buffer corrupts every row but the first -- accumulate into full-width
    buffers only (which is why the y stage runs at chunk = h = 1024).

Compared at the same shape against the current triton forward chain, the
fp32 torch reference, and the aclnnMhcPreSinkhorn forward (timing bar only:
full-MHC semantics). Forward only -- if this wins, a matching backward is
phase two.

Requires the source-built tilelang-ascend clone:
  PYTHONPATH=<tilelang-ascend clone> LD_LIBRARY_PATH=<conda env>/lib:$LD_LIBRARY_PATH
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402

try:
    import tilelang  # noqa: E402
    from tilelang import language as T
except ImportError as exc:
    raise SystemExit(f'tilelang not importable ({exc}); see module docstring for env')

from mindspeed_llm.ops.triton.mhc_lite_heads import lite_heads_y_forward  # noqa: E402

torch.manual_seed(7)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

SB, E, H, NP = 4096, 4, 1024, 24
N32 = 8 + NP
# padded column layout: [pre 0:4 | pad 4:8 | post 8:12 | pad 12:16 | res 16:40 | pad 40:48]
N48 = 16 + NP + 8
EPS = 1e-5
VEC_NUM = 2
ROWS = 8  # tokens per vector core

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[5, 6, 7, 8], pass_configs=pass_configs)
def lite_pre2d(sb, e=4, h=1024, np_=24, eps=1e-5, dtype='bfloat16'):
    """[ROWS, W] vector kernel: 8 token rows of x + padded raw logits in,
    (y [h], h_pre [8], h_post [8], coeff [np_]) out per token."""
    n48 = 16 + np_ + 8
    eh = e * h
    ch = h  # y-stage chunk = full row width: full-width buffers only
    blk = ROWS * VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor((sb, eh), dtype),
        logits_raw: T.Tensor((sb, n48), dtype),
        scale48: T.Tensor((1, n48), 'float32'),
        base: T.Tensor((1, n48), 'float32'),
        masks: T.Tensor((e, 8), 'float32'),
        y: T.Tensor((sb, h), dtype),
        h_pre8: T.Tensor((sb, 8), 'float32'),
        h_post8: T.Tensor((sb, 8), 'float32'),
        coeff: T.Tensor((sb, np_), 'float32'),
    ):
        with T.Kernel(T.ceildiv(sb, blk), is_npu=True) as (cid, vid):
            r0 = cid * blk + vid * ROWS

            if r0 < sb:
                with T.Scope('V'):
                    x_ub = T.alloc_ub((ROWS, ch), dtype)
                    x32 = T.alloc_ub((ROWS, ch), 'float32')
                    sq = T.alloc_ub((ROWS, ch), 'float32')
                    srow = T.alloc_ub((ROWS, 1), 'float32')
                    ss = T.alloc_ub((ROWS, 1), 'float32')
                    rstd = T.alloc_ub((ROWS, 1), 'float32')

                    y_acc = T.alloc_ub((ROWS, h), 'float32')
                    y_bf = T.alloc_ub((ROWS, h), dtype)
                    colb = T.alloc_ub((ROWS, ch), 'float32')

                    l_bf = T.alloc_ub((ROWS, n48), dtype)
                    l32 = T.alloc_ub((ROWS, n48), 'float32')
                    rstd_b = T.alloc_ub((ROWS, n48), 'float32')
                    scale48_ub = T.alloc_ub((1, n48), 'float32')
                    s48b = T.alloc_ub((ROWS, n48), 'float32')
                    base_ub = T.alloc_ub((1, n48), 'float32')
                    base8 = T.alloc_ub((ROWS, n48), 'float32')

                    # head buffers are 8 lanes wide so every window the heads
                    # slice stays 32B-aligned
                    z8 = T.alloc_ub((ROWS, 8), 'float32')
                    zero8 = T.alloc_ub((ROWS, 8), 'float32')
                    e8 = T.alloc_ub((ROWS, 8), 'float32')
                    one8 = T.alloc_ub((ROWS, 8), 'float32')
                    d8 = T.alloc_ub((ROWS, 8), 'float32')
                    hpre8 = T.alloc_ub((ROWS, 8), 'float32')
                    hpost8 = T.alloc_ub((ROWS, 8), 'float32')
                    two8 = T.alloc_ub((ROWS, 8), 'float32')
                    tmp8 = T.alloc_ub((ROWS, 8), 'float32')
                    mask8 = T.alloc_ub((1, 8), 'float32')
                    mask_b = T.alloc_ub((ROWS, 8), 'float32')

                    s8w = T.alloc_ub((ROWS, 8), 'float32')
                    b8w = T.alloc_ub((ROWS, 8), 'float32')
                    s24w = T.alloc_ub((ROWS, np_), 'float32')
                    b24w = T.alloc_ub((ROWS, np_), 'float32')
                    z24 = T.alloc_ub((ROWS, np_), 'float32')
                    m1 = T.alloc_ub((ROWS, 1), 'float32')
                    m_b = T.alloc_ub((ROWS, np_), 'float32')
                    e24 = T.alloc_ub((ROWS, np_), 'float32')
                    s1t = T.alloc_ub((ROWS, 1), 'float32')
                    s_b = T.alloc_ub((ROWS, np_), 'float32')
                    coeff24 = T.alloc_ub((ROWS, np_), 'float32')

                    # pass 1: chunked RMS square-sum over the [ROWS, eh] tile
                    T.tile.fill(ss, 0.0)
                    for c in T.serial(eh // ch):
                        T.copy(x[r0 : r0 + ROWS, c * ch : (c + 1) * ch], x_ub)
                        T.tile.cast(x32, x_ub, 'CAST_NONE', ROWS * ch)
                        T.tile.mul(sq, x32, x32)
                        T.reduce_sum(sq, srow, dim=-1, clear=True)
                        T.tile.add(ss, ss, srow)
                    inv_n = T.cast(1.0 / eh, 'float32')
                    eps_c = T.cast(eps, 'float32')
                    T.tile.mul(ss, ss, inv_n)
                    T.tile.add(ss, ss, eps_c)
                    T.tile.rsqrt(rstd, ss)

                    # heads: logits = raw * rstd (gamma already folded into W')
                    T.copy(logits_raw[r0 : r0 + ROWS, 0:n48], l_bf)
                    T.tile.cast(l32, l_bf, 'CAST_NONE', ROWS * n48)
                    T.tile.broadcast(rstd_b, rstd, axis=1)
                    T.tile.mul(l32, l32, rstd_b)
                    T.copy(scale48[0:1, 0:n48], scale48_ub)
                    T.copy(base[0:1, 0:n48], base_ub)
                    T.tile.broadcast(s48b, scale48_ub, axis=0)
                    T.tile.broadcast(base8, base_ub, axis=0)

                    # pre head: sigmoid(z), z = s0*l + b; window operands
                    # are staged into dedicated buffers first -- a mul whose
                    # operand is a window VIEW miscompiles
                    T.copy(l32[0:ROWS, 0:8], z8)
                    T.copy(s48b[0:ROWS, 0:8], s8w)
                    T.copy(base8[0:ROWS, 0:8], b8w)
                    T.tile.mul(z8, z8, s8w)
                    T.tile.add(z8, z8, b8w)
                    T.tile.fill(zero8, 0.0)
                    T.tile.fill(one8, 1.0)
                    T.tile.sub(e8, zero8, z8)
                    T.tile.exp(e8, e8)
                    T.tile.add(d8, one8, e8)
                    T.tile.div(hpre8, one8, d8)

                    # post head: 2*sigmoid(z) over the 8:16 window
                    T.copy(l32[0:ROWS, 8:16], hpost8)
                    T.copy(s48b[0:ROWS, 8:16], s8w)
                    T.copy(base8[0:ROWS, 8:16], b8w)
                    T.tile.mul(hpost8, hpost8, s8w)
                    T.tile.add(hpost8, hpost8, b8w)
                    T.tile.sub(e8, zero8, hpost8)
                    T.tile.exp(e8, e8)
                    T.tile.add(d8, one8, e8)
                    T.tile.div(hpost8, one8, d8)
                    T.tile.fill(two8, 2.0)
                    T.tile.mul(hpost8, hpost8, two8)

                    # res head: softmax(z24); the permutation mixture stays
                    # on the host because its weights differ per row
                    T.copy(l32[0:ROWS, 16 : 16 + np_], z24)
                    T.copy(s48b[0:ROWS, 16 : 16 + np_], s24w)
                    T.copy(base8[0:ROWS, 16 : 16 + np_], b24w)
                    T.tile.mul(z24, z24, s24w)
                    T.tile.add(z24, z24, b24w)
                    T.reduce_max(z24, m1, dim=-1, clear=True)
                    T.tile.broadcast(m_b, m1, axis=1)
                    T.tile.sub(e24, z24, m_b)
                    T.tile.exp(e24, e24)
                    T.reduce_sum(e24, s1t, dim=-1, clear=True)
                    T.tile.broadcast(s_b, s1t, axis=1)
                    T.tile.div(coeff24, e24, s_b)

                    # y = sum_i h_pre[:, i] * x[:, i*h:(i+1)*h]: extract each
                    # per-row scalar by masked reduce, broadcast, accumulate
                    T.tile.fill(y_acc, 0.0)
                    for i in T.unroll(e):
                        T.copy(masks[i : i + 1, 0:8], mask8)
                        T.tile.broadcast(mask_b, mask8, axis=0)
                        T.tile.mul(tmp8, hpre8, mask_b)
                        T.reduce_sum(tmp8, srow, dim=-1, clear=True)
                        T.copy(x[r0 : r0 + ROWS, i * h : (i + 1) * h], x_ub)
                        T.tile.cast(x32, x_ub, 'CAST_NONE', ROWS * ch)
                        T.tile.broadcast(colb, srow, axis=1)
                        T.tile.mul(x32, x32, colb)
                        T.tile.add(y_acc, y_acc, x32)

                    T.tile.cast(y_bf, y_acc, 'CAST_RINT', ROWS * h)
                    T.copy(y_bf, y[r0 : r0 + ROWS, 0:h])
                    T.copy(hpre8, h_pre8[r0 : r0 + ROWS, 0:8])
                    T.copy(hpost8, h_post8[r0 : r0 + ROWS, 0:8])
                    T.copy(coeff24, coeff[r0 : r0 + ROWS, 0:np_])

    return main


def main():
    from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend

    x = (torch.randn(SB, E, H, device=DEV) * 1.5).to(torch.bfloat16)
    xf = x.reshape(SB, E * H)
    w = torch.randn(N32, E * H, device=DEV) * 0.02
    gamma = 1.0 + torch.randn(E * H, device=DEV) * 0.05
    gamma_bf = gamma.to(torch.bfloat16)
    scale = torch.tensor([[0.011, 0.013, 0.017]], device=DEV)
    base = (torch.randn(N32, device=DEV) * 0.5).view(1, N32).contiguous()
    perm_flat = torch.eye(E, dtype=torch.float32)[
        torch.tensor(list(__import__('itertools').permutations(range(E))))
    ].flatten(1).to(DEV)  # [24, 16]
    perm_t = perm_flat.t().contiguous()  # [16, 24], layout the triton chain wants

    # W in the padded 48-row layout, placed once; per-step W' is one
    # broadcast mul against gamma into a second preallocated buffer
    w_pad = torch.zeros(N48, E * H, device=DEV, dtype=torch.bfloat16)
    w_pad[0:4] = w[0:4].to(torch.bfloat16)
    w_pad[8:12] = w[4:8].to(torch.bfloat16)
    w_pad[16:40] = w[8:32].to(torch.bfloat16)
    wp48 = torch.empty_like(w_pad)

    def wp():
        torch.mul(w_pad, gamma_bf.view(1, -1), out=wp48)
        return wp48

    # padded runtime scalars; scale uniform over each head window so the
    # kernel can broadcast-multiply it instead of reading an element scalar
    scale48 = torch.zeros(1, N48, device=DEV)
    scale48[0, 0:8] = scale[0, 0]
    scale48[0, 8:16] = scale[0, 1]
    scale48[0, 16:40] = scale[0, 2]
    base48 = torch.zeros(1, N48, device=DEV)
    base48[0, 0:4] = base[0, 0:4]
    base48[0, 8:12] = base[0, 4:8]
    base48[0, 16:40] = base[0, 8:32]
    masks = torch.zeros(E, 8, device=DEV)
    for i in range(E):
        masks[i, i] = 1.0

    # fp32 reference
    xf32 = xf.float()
    rstd = torch.rsqrt(xf32.pow(2).mean(-1, keepdim=True) + EPS)
    logits_r = (xf32 * rstd * gamma) @ w.float().t()
    pre_l, post_l, res_l = torch.split(logits_r, [E, E, NP], -1)
    h_pre_r = torch.sigmoid(pre_l * scale[0, 0] + base[0, :E])
    h_post_r = 2 * torch.sigmoid(post_l * scale[0, 1] + base[0, E:2 * E])
    coeff_r = torch.softmax(res_l * scale[0, 2] + base[0, 2 * E:], -1)
    h_res_r = coeff_r @ perm_flat
    y_r = torch.sum(h_pre_r.unsqueeze(-1) * x.float().view(SB, E, H), dim=1)

    # current triton forward chain (Tier-2 path, gamma in bf16 like the module)
    def chain_cur():
        xn, _ = torch_npu.npu_rms_norm(xf, gamma_bf, epsilon=EPS)
        logits = torch.matmul(xn, w.to(torch.bfloat16).t()).float()
        return lite_heads_y_forward(logits, x, scale.view(3), base.view(N32), perm_t)

    kernel = lite_pre2d(SB)

    def chain_tl():
        logits_raw = torch.matmul(xf, wp().t())
        y, hpre8, hpost8, coeff = kernel(xf, logits_raw, scale48, base48, masks)
        h_res = torch.matmul(coeff, perm_flat)
        return y, hpre8[:, 0:4], hpost8[:, 0:4], h_res

    y_c, hpre_c, hpost_c, hres_c = chain_cur()
    y_t, hpre_t, hpost_t, hres_t = chain_tl()
    torch.npu.synchronize()

    def md(a, b):
        return (a.float() - b.float()).abs().max().item()

    print(f'fp32-ref noise floor (current chain vs fp32 ref): '
          f'y {md(y_c, y_r):.1e} h_pre {md(hpre_c, h_pre_r):.1e} h_post {md(hpost_c, h_post_r):.1e} h_res {md(hres_c, h_res_r):.1e}')
    print(f'tilelang 2-D vs fp32 ref:                        '
          f'y {md(y_t, y_r):.1e} h_pre {md(hpre_t, h_pre_r):.1e} h_post {md(hpost_t, h_post_r):.1e} h_res {md(hres_t, h_res_r):.1e}')
    print(f'tilelang 2-D vs current chain:                   '
          f'y {md(y_t, y_c):.1e} h_pre {md(hpre_t, hpre_c):.1e} h_post {md(hpost_t, hpost_c):.1e} h_res {md(hres_t, hres_c):.1e}')

    # attribution: W' rebuild, GEMM, and the fused kernel separately
    logits_raw = torch.matmul(xf, wp().t())
    t_wp = bench(wp)
    t_gemm = bench(lambda: torch.matmul(xf, wp48.t()))
    t_kern = bench(lambda: kernel(xf, logits_raw, scale48, base48, masks))
    t_cur = bench(chain_cur)
    t_tl = bench(chain_tl)
    print(f'\nW\' = W*gamma broadcast mul     : {t_wp:.3f} ms')
    print(f'GEMM (excl W\')                : {t_gemm:.3f} ms')
    print(f'fused 2-D kernel alone        : {t_kern:.3f} ms')
    print(f'current chain (rms+GEMM+K1ab) : {t_cur:.3f} ms')
    print(f'tilelang (W\'+GEMM+ker+mix)    : {t_tl:.3f} ms')

    # aclnn fused pre forward, the bar (full-MHC semantics -- timing only)
    w_full = torch.randn((2 + E) * E, E * H, device=DEV, dtype=torch.float32) * 0.02
    s_full = torch.randn(3, device=DEV, dtype=torch.float32) * 0.05
    b_full = torch.randn((2 + E) * E, device=DEV, dtype=torch.float32) * 0.1
    xb = x.view(SB, 1, E, H)
    mhc_pre_sinkhorn_ascend(xb, w_full, s_full, b_full, E, 20, 1e-6, 1e-5)
    torch.npu.synchronize()
    t_aclnn = bench(lambda: mhc_pre_sinkhorn_ascend(
        xb, w_full, s_full, b_full, E, 20, 1e-6, 1e-5))
    print(f'aclnn mhc_pre_sinkhorn fwd    : {t_aclnn:.3f} ms  (full-MHC semantics, timing bar)')


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


if __name__ == '__main__':
    main()
