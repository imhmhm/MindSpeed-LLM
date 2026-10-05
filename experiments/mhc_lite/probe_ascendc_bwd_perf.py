# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Attribute the scheme-E pre-backward cost piece by piece at sb=4096 and
compare against the lite-t3 (triton pre) backward pieces it replaces."""
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

from mindspeed_llm.ops.triton.mhc_lite_heads import (  # noqa: E402
    lite_grad_x, lite_heads_y_forward, lite_pre_backward)
from mindspeed_llm.ops.ascendc.mhc_lite_ac import lite_pre_ascendc  # noqa: E402

torch.manual_seed(5)
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


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


x = (torch.randn(SB, E, H, device=DEV) * 1.5).to(torch.bfloat16)
xf = x.view(SB, E * H)
w = torch.randn(N32, E * H, device=DEV) * 0.02
gamma = 1.0 + torch.randn(E * H, device=DEV) * 0.05
gamma_bf = gamma.to(torch.bfloat16)
scale = torch.tensor([0.011, 0.013, 0.017], device=DEV)
base = (torch.randn(N32, device=DEV) * 0.5).contiguous()
perm_flat = torch.eye(E)[torch.tensor(
    list(__import__('itertools').permutations(range(E))))].flatten(1).to(DEV)
perm_t = perm_flat.t().contiguous()
lane = torch.tensor([0] * 4 + [1] * 4 + [2] * 24, device=DEV)
scale32 = scale[lane]
eps8 = torch.full((8,), EPS, device=DEV)

wp = w.to(torch.bfloat16) * gamma_bf.view(1, -1)
logits = torch.matmul(xf, wp.t())
gy = torch.randn(SB, H, device=DEV).to(torch.bfloat16)
ghp = torch.randn(SB, E, device=DEV)
ghr = torch.randn(SB, E * E, device=DEV)

y, hpre8, hpost8, coeff, rstd = ext.lite_pre_heads(xf, logits, scale32, base, eps8)
l = logits.float() * rstd.unsqueeze(-1)

# scheme-E backward pieces
dl, ds3, db = lite_pre_backward(gy, x, ghp.contiguous(), ghr.contiguous(),
                                perm_t, l, scale, base)
coef = (((dl * logits.float()).sum(-1)) * -(rstd.pow(3)) / (E * H)).to(torch.bfloat16)
dxrms = (coef.unsqueeze(-1).unsqueeze(-1) * x).contiguous()
glog = (dl * rstd.unsqueeze(-1)).to(torch.bfloat16)

print('--- scheme-E pre backward pieces (sb=4096)')
print(f'l = logits.float()*rstd          : {bench(lambda: logits.float() * rstd.unsqueeze(-1)):.3f} ms')
print(f'lite_pre_backward (triton)       : {bench(lambda: lite_pre_backward(gy, x, ghp.contiguous(), ghr.contiguous(), perm_t, l, scale, base)):.3f} ms')
print(f'grad_logits cast                 : {bench(lambda: (dl * rstd.unsqueeze(-1)).to(torch.bfloat16)):.3f} ms')
print(f'drstd/coef                       : {bench(lambda: (((dl * logits.float()).sum(-1)) * -(rstd.pow(3)) / (E * H)).to(torch.bfloat16)):.3f} ms')
print(f'coef*x (bf16 broadcast mul)      : {bench(lambda: coef.unsqueeze(-1).unsqueeze(-1) * x):.3f} ms')
print(f'lite_grad_x (triton)             : {bench(lambda: lite_grad_x(gy, hpre8[:, :4].contiguous(), dxrms)):.3f} ms')
print(f'hpre8 slice .contiguous          : {bench(lambda: hpre8[:, :4].contiguous()):.3f} ms')
print(f'grad_wp GEMM (glog.t()@xf)       : {bench(lambda: torch.matmul(glog.t(), xf)):.3f} ms')
print(f'd_xf GEMM (glog@wp)              : {bench(lambda: torch.matmul(glog, wp)):.3f} ms')
print(f'wp vjp elementwise (2x [32,eh])  : {bench(lambda: (torch.matmul(glog.t(), xf) * gamma_bf.view(1, -1), torch.matmul(glog.t(), xf) * w.to(torch.bfloat16))):.3f} ms')
inv_lane = torch.tensor([4.0, 4.0, 24.0], device=DEV).reciprocal()[lane]
print(f'dscale lane gather+mul           : {bench(lambda: ds3[lane] * inv_lane):.3f} ms')

# full Function path
xg = x.detach().view(SB, 1, E, H).requires_grad_()
wg = w.detach().requires_grad_()
gg = gamma.detach().requires_grad_()
sg = scale.detach().requires_grad_()
bg = base.detach().requires_grad_()


def full_fwd():
    with torch.no_grad():
        lite_pre_ascendc(xg, wg, gg, sg, bg, perm_flat, perm_t, EPS)


def full_fwdbwd():
    yv, hpv, hrv = lite_pre_ascendc(xg, wg, gg, sg, bg, perm_flat, perm_t, EPS)
    torch.autograd.grad((yv, hpv, hrv), (xg, wg, gg, sg, bg), (gy.view(SB, 1, H), ghp.view(SB, 1, E), ghr.view(SB, 1, E, E)))


print(f'scheme-E full pre fwd            : {bench(full_fwd):.3f} ms')
print(f'scheme-E full pre fwd+bwd        : {bench(full_fwdbwd):.3f} ms')

# lite-t3 pre backward pieces for comparison
xn, rstd_t3 = torch_npu.npu_rms_norm(xf, gamma_bf, epsilon=EPS)
logits_t3 = torch.matmul(xn, w.to(torch.bfloat16).t()).float()
y_t3, hpre_t3, hpost_t3, hres_t3 = lite_heads_y_forward(logits_t3, x, scale, base, perm_t)
dl_t3, ds_t3, db_t3 = lite_pre_backward(gy, x, ghp.contiguous(), ghr.contiguous(),
                                        perm_t, logits_t3, scale, base)
glog_t3 = dl_t3.to(torch.bfloat16)
dxn_t3 = torch.matmul(glog_t3, w.to(torch.bfloat16))

print('--- lite-t3 pre backward pieces (sb=4096)')
print(f'logits fwd (rms+GEMM+float)      : {bench(lambda: torch.matmul(xn, w.to(torch.bfloat16).t()).float()):.3f} ms')
print(f'grad_weight GEMM (glog.t()@xn)   : {bench(lambda: torch.matmul(glog_t3.t(), xn)):.3f} ms')
print(f'd_xn GEMM (glog@w)               : {bench(lambda: torch.matmul(glog_t3, w.to(torch.bfloat16))):.3f} ms')
dxrms_t3, dgamma_t3 = torch_npu.npu_rms_norm_backward(dxn_t3, xf, gamma_bf, rstd_t3)
print(f'npu_rms_norm_backward            : {bench(lambda: torch_npu.npu_rms_norm_backward(dxn_t3, xf, gamma_bf, rstd_t3)):.3f} ms')
print(f'lite_grad_x (triton)             : {bench(lambda: lite_grad_x(gy, hpre_t3, dxrms_t3.view(SB, E, H))):.3f} ms')
