# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Scheme F: fold the logits GEMM into the Ascend C op (ascendc_lite_pre_fused/).

LitePreFused runs the bf16 cube GEMM x @ W^T through the Ascend C Matmul
API with C in VECIN position, so each [baseM, 32] fp32 tile lands in the
cube core's UB and the vector epilogue (RMS pass1, three heads, y) runs
on the same core -- logits never round-trip GM, W' = W*gamma build
stays outside (gamma acts on K and cannot fold into a per-lane scale), and
the GEMM + heads + y run in one aclnn launch.  The op also saves a bf16
copy of l so the scheme-E triton backward can be reused unchanged.

This bench validates against the fp32 torch reference and the two live
chains (current triton chain, scheme-E split op chain), then times:
  fused op alone vs W'+GEMM+LitePreHeads split vs the triton chain,
plus the aclnnMhcPreSinkhorn forward as the timing bar (full-MHC
semantics).  Tail-row shapes (sb not a multiple of the core split) are
checked explicitly: the matmul tiling covers padded rows, the epilogue
clamps them.

Requires:
  bash experiments/mhc_lite/ascendc_lite_pre_fused/sync_and_build.sh
  export ASCEND_CUSTOM_OPP_PATH=<cann-ops clone>/build_out
"""

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

os.environ.setdefault(
    'ASCEND_CUSTOM_OPP_PATH',
    str(Path.home() / 'work/dataset/huashan_zhh_guiyang_turbo/github/cann-ops/build_out'))

import torch  # noqa: E402
import torch_npu  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from mindspeed_llm.ops.triton.mhc_lite_heads import lite_heads_y_forward  # noqa: E402

torch.manual_seed(7)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

SB, E, H, NP = 4096, 4, 1024, 24
N32 = 8 + NP
EPS = 1e-5

TORCH_NPU = Path(torch_npu.__file__).parent
ext_f = load(
    name='ascendc_lite_pre_fused_ext',
    sources=[str(Path(__file__).parent / 'ascendc_lite_pre_fused/extension.cpp')],
    extra_include_paths=[
        str(TORCH_NPU / 'include'),
        str(TORCH_NPU / 'include/third_party/acl/inc'),
        '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
    ],
    extra_ldflags=[f'-L{TORCH_NPU}/lib', '-ltorch_npu'],
    verbose=False,
)
ext_e = load(
    name='ascendc_lite_pre_ext',
    sources=[str(Path(__file__).parent / 'ascendc_lite_pre/extension.cpp')],
    extra_include_paths=[
        str(TORCH_NPU / 'include'),
        str(TORCH_NPU / 'include/third_party/acl/inc'),
        '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
    ],
    extra_ldflags=[f'-L{TORCH_NPU}/lib', '-ltorch_npu'],
    verbose=False,
)


def md(a, b):
    return (a.float() - b.float()).abs().max().item()


def make_inputs(sb, h):
    x = (torch.randn(sb, E, h, device=DEV) * 1.5).to(torch.bfloat16)
    xf = x.reshape(sb, E * h)
    w = torch.randn(N32, E * h, device=DEV) * 0.02
    gamma = 1.0 + torch.randn(E * h, device=DEV) * 0.05
    gamma_bf = gamma.to(torch.bfloat16)
    scale = torch.tensor([[0.011, 0.013, 0.017]], device=DEV)
    base = (torch.randn(N32, device=DEV) * 0.5).view(1, N32).contiguous()
    return x, xf, w, gamma, gamma_bf, scale, base


def build_scale(sb_parts):
    scale32 = torch.empty(N32, device=DEV)
    scale32[0:E] = sb_parts[0]
    scale32[E:2 * E] = sb_parts[1]
    scale32[2 * E:] = sb_parts[2]
    return scale32


def fp32_ref(xf, w, gamma, scale, base, sb, h):
    xf32 = xf.float()
    rstd = torch.rsqrt(xf32.pow(2).mean(-1, keepdim=True) + EPS)
    logits_r = (xf32 * rstd * gamma) @ w.float().t()
    pre_l, post_l, res_l = torch.split(logits_r, [E, E, NP], -1)
    h_pre_r = torch.sigmoid(pre_l * scale[0, 0] + base[0, :E])
    h_post_r = 2 * torch.sigmoid(post_l * scale[0, 1] + base[0, E:2 * E])
    coeff_r = torch.softmax(res_l * scale[0, 2] + base[0, 2 * E:], -1)
    l_r = logits_r  # l before scale/base, what the fused op saves as bf16
    return h_pre_r, h_post_r, coeff_r, l_r


def accuracy():
    print('== accuracy (vs fp32 torch reference) ==')
    for sb, h in [(4096, 1024), (1033, 1024), (512, 1024), (4096, 512)]:
        x, xf, w, gamma, gamma_bf, scale, base = make_inputs(sb, h)
        scale32 = build_scale([scale[0, 0], scale[0, 1], scale[0, 2]])

        h_pre_r, h_post_r, coeff_r, l_r = fp32_ref(xf, w, gamma, scale, base, sb, h)

        wp32 = (w.to(torch.bfloat16) * gamma_bf.view(1, -1))
        y_f, hp_f, ho_f, co_f, rstd_f, lg_f = ext_f.lite_pre_fused(xf, wp32, scale32, base.view(N32))
        hpre_f = hp_f[:, 0:4]
        hpost_f = ho_f[:, E:8]
        torch.npu.synchronize()

        r1 = md(hpre_f, h_pre_r)
        r2 = md(hpost_f, h_post_r)
        r3 = md(co_f[:, 2 * E:], coeff_r)
        # saved l vs fp32 l: bf16 rounding floor expected
        r4 = md(lg_f.float(), l_r) / l_r.abs().max().item()
        print(f'sb={sb:5d} h={h:5d}  h_pre {r1:.1e}  h_post {r2:.1e}  coeff {r3:.1e}  '
              f'l(bf16-saved,rel) {r4:.1e}')
        assert r1 < 5e-2 and r2 < 5e-2 and r3 < 5e-2, f'accuracy gate failed at sb={sb}'


def main():
    from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend

    accuracy()

    x, xf, w, gamma, gamma_bf, scale, base = make_inputs(SB, H)
    w_bf = w.to(torch.bfloat16)
    wp = torch.empty_like(w_bf)
    scale32 = build_scale([scale[0, 0], scale[0, 1], scale[0, 2]])
    perm_flat = torch.eye(E, dtype=torch.float32)[
        torch.tensor(list(__import__('itertools').permutations(range(E))))
    ].flatten(1).to(DEV)
    perm_t = perm_flat.t().contiguous()

    def wprime():
        torch.mul(w_bf, gamma_bf.view(1, -1), out=wp)
        return wp

    def chain_cur():
        xn, _ = torch_npu.npu_rms_norm(xf, gamma_bf, epsilon=EPS)
        logits = torch.matmul(xn, w_bf.t()).float()
        return lite_heads_y_forward(logits, x, scale.view(3), base.view(N32), perm_t)

    def chain_e():
        logits_raw = torch.matmul(xf, wprime().t())
        y, hpre8, hpost8, coeff, rstd = ext_e.lite_pre_heads(xf, logits_raw, scale32, base.view(N32))
        h_res = torch.matmul(coeff[:, 2 * E:N32], perm_flat)
        return y, hpre8[:, 0:4], hpost8[:, E:8], h_res

    def chain_f():
        y, hpre8, hpost8, coeff, rstd, logits = ext_f.lite_pre_fused(
            xf, wprime(), scale32, base.view(N32))
        h_res = torch.matmul(coeff[:, 2 * E:N32], perm_flat)
        return y, hpre8[:, 0:4], hpost8[:, E:8], h_res

    def chain_g():
        y, hpre8, hpost8, coeff, rstd, hres = ext_e.lite_pre_chain(
            xf, w_bf, gamma_bf, scale32, base.view(N32), perm_flat)
        return y, hpre8[:, 0:4], hpost8[:, E:8], hres

    y_c, hpre_c, hpost_c, hres_c = chain_cur()
    y_e, hpre_e, hpost_e, hres_e = chain_e()
    y_f, hpre_f, hpost_f, hres_f = chain_f()
    y_g, hpre_g, hpost_g, hres_g = chain_g()
    torch.npu.synchronize()

    h_pre_r, h_post_r, coeff_r, l_r = fp32_ref(xf, w, gamma, scale, base, SB, H)
    y_r = None  # same reference form as bench_ascendc_lite_pre; y built inline below
    xf32 = xf.float()
    rstd_r = torch.rsqrt(xf32.pow(2).mean(-1, keepdim=True) + EPS)
    y_ref = torch.sum(h_pre_r.unsqueeze(-1) * x.float().view(SB, E, H), dim=1)

    print('\n== accuracy at sb=4096 h=1024 ==')
    print(f'noise floor (triton chain vs fp32 ref):    '
          f'y {md(y_c, y_ref):.1e} h_pre {md(hpre_c, h_pre_r):.1e} h_post {md(hpost_c, h_post_r):.1e}')
    print(f'scheme E split chain vs fp32 ref:          '
          f'y {md(y_e, y_ref):.1e} h_pre {md(hpre_e, h_pre_r):.1e} h_post {md(hpost_e, h_post_r):.1e}')
    print(f'scheme F fused chain vs fp32 ref:          '
          f'y {md(y_f, y_ref):.1e} h_pre {md(hpre_f, h_pre_r):.1e} h_post {md(hpost_f, h_post_r):.1e}')
    print(f'scheme F fused vs scheme E split:          '
          f'y {md(y_f, y_e):.1e} h_pre {md(hpre_f, hpre_e):.1e} h_post {md(hpost_f, hpost_e):.1e}')
    print(f'scheme G one-call chain vs fp32 ref:       '
          f'y {md(y_g, y_ref):.1e} h_pre {md(hpre_g, h_pre_r):.1e} h_post {md(hpost_g, h_post_r):.1e} '
          f'h_res {md(hres_g, hres_e):.1e} (vs E)')

    print('\n== timing (sb=4096 h=1024) ==')
    logits_raw = torch.matmul(xf, wprime().t())
    t_wp = bench(wprime)
    t_gemm = bench(lambda: torch.matmul(xf, wp.t()))
    t_op_e = bench(lambda: ext_e.lite_pre_heads(xf, logits_raw, scale32, base.view(N32)))
    t_op_f = bench(lambda: ext_f.lite_pre_fused(xf, wp, scale32, base.view(N32)))
    t_cur = bench(chain_cur)
    t_e = bench(chain_e)
    t_f = bench(chain_f)
    t_g = bench(chain_g)
    print(f'W\' = W*gamma broadcast mul     : {t_wp:.3f} ms')
    print(f'GEMM (excl W\')                : {t_gemm:.3f} ms')
    print(f'LitePreHeads op alone         : {t_op_e:.3f} ms')
    print(f'LitePreFused op alone         : {t_op_f:.3f} ms')
    print(f'current chain (rms+GEMM+K1ab) : {t_cur:.3f} ms')
    print(f'scheme E split chain          : {t_e:.3f} ms')
    print(f'scheme F fused chain          : {t_f:.3f} ms')
    print(f'scheme G one-call chain       : {t_g:.3f} ms')

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
