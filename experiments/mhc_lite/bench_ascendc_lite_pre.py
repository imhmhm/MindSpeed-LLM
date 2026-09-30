# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Scheme E: hand-written standalone Ascend C operator for the mhc_lite pre
forward (experiments/mhc_lite/ascendc_lite_pre/).

Same decomposition as the tilelang prototype (scheme C): the RMSNorm
linearity folds gamma into the logits GEMM weight,
    xn = x * rstd * gamma   =>   xn @ W^T = rstd * (x @ (W * gamma)^T),
so the GEMM consumes raw x and ONE hand-written vector kernel
(LitePreHeads, aclnn-exposed) folds everything else: per-token RMS
square-sum, the three heads (sigmoid / 2*sigmoid / softmax) and the y
mixture.  Unlike the tilelang version it is a single aclnn launch with
no W'-layout padding, no 48-wide logits and no python-side launch gap;
per-row scalars ride scalar registers via LocalTensor::GetValue + Muls,
which tilelang cannot express (its axpy miscompiles on element scalars).

The op is built outside any CANN installation: sync_and_build.sh drops
the sources into a cann-ops clone and the resulting vendor tree in
<clone>/build_out is picked up at runtime through
ASCEND_CUSTOM_OPP_PATH, so libcust_opapi.so answers the aclnn symbols
and no CANN file is touched.

Compared at the same shape against the current triton forward chain, the
fp32 torch reference, and the aclnnMhcPreSinkhorn forward (timing bar
only: full-MHC semantics).  Forward only -- a matching backward is phase
two.

Requires:
  bash experiments/mhc_lite/ascendc_lite_pre/sync_and_build.sh
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
ext = load(
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

    # op-side runtime scalars: W' one broadcast mul into a preallocated buffer,
    # scale as a per-lane [32] vector so the kernel's head windows stay
    # 32B-aligned (built once, it is a model constant)
    w_bf = w.to(torch.bfloat16)
    wp = torch.empty_like(w_bf)

    def wprime():
        torch.mul(w_bf, gamma_bf.view(1, -1), out=wp)
        return wp

    scale32 = torch.empty(N32, device=DEV)
    scale32[0:E] = scale[0, 0]
    scale32[E:2 * E] = scale[0, 1]
    scale32[2 * E:] = scale[0, 2]

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

    def chain_ac():
        logits_raw = torch.matmul(xf, wprime().t())
        y, hpre8, hpost8, coeff = ext.lite_pre_heads(xf, logits_raw, scale32, base.view(N32))
        h_res = torch.matmul(coeff[:, 2 * E:N32], perm_flat)
        return y, hpre8[:, 0:4], hpost8[:, E:8], h_res

    y_c, hpre_c, hpost_c, hres_c = chain_cur()
    y_a, hpre_a, hpost_a, hres_a = chain_ac()
    torch.npu.synchronize()

    def md(a, b):
        return (a.float() - b.float()).abs().max().item()

    print(f'fp32-ref noise floor (current chain vs fp32 ref): '
          f'y {md(y_c, y_r):.1e} h_pre {md(hpre_c, h_pre_r):.1e} h_post {md(hpost_c, h_post_r):.1e} h_res {md(hres_c, h_res_r):.1e}')
    print(f'ascend-c op chain vs fp32 ref:                   '
          f'y {md(y_a, y_r):.1e} h_pre {md(hpre_a, h_pre_r):.1e} h_post {md(hpost_a, h_post_r):.1e} h_res {md(hres_a, h_res_r):.1e}')
    print(f'ascend-c op chain vs current chain:               '
          f'y {md(y_a, y_c):.1e} h_pre {md(hpre_a, hpre_c):.1e} h_post {md(hpost_a, hpost_c):.1e} h_res {md(hres_a, hres_c):.1e}')

    # attribution: W' rebuild, GEMM, and the fused op separately
    logits_raw = torch.matmul(xf, wprime().t())
    t_wp = bench(wprime)
    t_gemm = bench(lambda: torch.matmul(xf, wp.t()))
    t_op = bench(lambda: ext.lite_pre_heads(xf, logits_raw, scale32, base.view(N32)))
    t_cur = bench(chain_cur)
    t_ac = bench(chain_ac)
    print(f'\nW\' = W*gamma broadcast mul     : {t_wp:.3f} ms')
    print(f'GEMM (excl W\')                : {t_gemm:.3f} ms')
    print(f'LitePreHeads op alone         : {t_op:.3f} ms')
    print(f'current chain (rms+GEMM+K1ab) : {t_cur:.3f} ms')
    print(f'ascend-c (W\'+GEMM+op+mix)     : {t_ac:.3f} ms')

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
