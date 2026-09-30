# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Validate and time the LitePreGrad Ascend C op (task #26) against the
scheme-E triton backward path it replaces.

Three references per output:
  1. the live triton path (lite_pre_backward + the wrapper glue) -- same
     math, different kernel; agreement bounds implementation divergence,
  2. a pure fp32 recomputation -- decides which side is closer to truth
     when they differ (the op keeps coef/dx in fp32 where the triton path
     rounds through bf16),
  3. timing: op alone vs the sum of the pieces it replaces.
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

from mindspeed_llm.ops.triton.mhc_lite_heads import lite_grad_x, lite_pre_backward  # noqa: E402

torch.manual_seed(11)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')
E, NP = 4, 24
N32 = 32
EPS = 1e-5

TORCH_NPU = Path(torch_npu.__file__).parent
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
ext_g = load(
    name='ascendc_lite_pre_grad_ext',
    sources=[str(Path(__file__).parent / 'ascendc_lite_pre_grad/extension.cpp')],
    extra_include_paths=[
        str(TORCH_NPU / 'include'),
        str(TORCH_NPU / 'include/third_party/acl/inc'),
        '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
    ],
    extra_ldflags=[f'-L{TORCH_NPU}/lib', '-ltorch_npu'],
    verbose=False,
)


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


def run_shape(sb, h, check=True):
    x = (torch.randn(sb, E, h, device=DEV) * 1.5).to(torch.bfloat16)
    xf = x.reshape(sb, E * h)
    w = torch.randn(N32, E * h, device=DEV) * 0.02
    gamma = 1.0 + torch.randn(E * h, device=DEV) * 0.05
    wp = (w * gamma.view(1, -1)).to(torch.bfloat16)
    logits = torch.matmul(xf, wp.t())
    scale32 = torch.empty(N32, device=DEV)
    scale32[0:E] = 0.011
    scale32[E:2 * E] = 0.013
    scale32[2 * E:] = 0.017
    base32 = (torch.randn(N32, device=DEV) * 0.5).contiguous()
    perm_flat = torch.eye(E, dtype=torch.float32)[
        torch.tensor(list(__import__('itertools').permutations(range(E))))].flatten(1).to(DEV)
    perm_t = perm_flat.t().contiguous()

    y, hpre8, hpost8, coeff, rstd = ext_e.lite_pre_heads(xf, logits, scale32, base32)
    h_res = torch.matmul(coeff[:, 2 * E:], perm_flat)
    g = (torch.randn(sb, h, device=DEV) * 0.05).to(torch.bfloat16)
    ghp = torch.randn(sb, E, device=DEV, dtype=torch.float32) * 0.1
    ghr = torch.randn(sb, E * E, device=DEV, dtype=torch.float32) * 0.1

    # live triton path + wrapper glue (what the op replaces)
    l = logits.float() * rstd.unsqueeze(-1)
    dl, dscale3, dbase = lite_pre_backward(
        g, x, ghp.contiguous(), ghr.contiguous(), perm_t, l, scale32[[0, E, 2 * E]], base32)
    gl_t = (dl * rstd.unsqueeze(-1)).to(logits.dtype)
    drstd = (dl * logits.float()).sum(-1)
    coef = (drstd * -(rstd.pow(3)) / (E * h)).to(xf.dtype).unsqueeze(-1).unsqueeze(-1)
    gx_t = lite_grad_x(g, hpre8[:, :E].contiguous(), coef * x).view(sb, E * h)
    inv = torch.tensor([E, E, NP], device=DEV, dtype=torch.float32).reciprocal()
    ds_t = dscale3 * inv

    # Ascend C op
    gl_o, gx_o, ds_o, db_o = ext_g.lite_pre_grad(
        g, xf, ghp, ghr, perm_t, logits, rstd, hpre8, scale32, base32)
    torch.npu.synchronize()

    if check:
        print(f'-- sb={sb} h={h}')
        seg_o = gl_o.float()
        seg_t = gl_t.float()
        print(f'   gl diff pre/post/res: '
              f'{(seg_o[:, :E] - seg_t[:, :E]).abs().max().item():.2e} '
              f'{(seg_o[:, E:2 * E] - seg_t[:, E:2 * E]).abs().max().item():.2e} '
              f'{(seg_o[:, 2 * E:] - seg_t[:, 2 * E:]).abs().max().item():.2e}'
              f'   max|gl| {seg_t.abs().max().item():.2e}')
        print(f'   grad_x    vs triton : {(gx_o.float() - gx_t.float()).abs().max().item():.2e}')

        # fp32 recomputation: ground truth for every output
        lf = logits.float()
        l32 = lf * rstd.unsqueeze(-1)
        dw = torch.einsum('bd,bed->be', g.float(), x.float())
        sigp = lambda z: torch.sigmoid(z) * (1 - torch.sigmoid(z))  # noqa: E731
        dz_pre = dw[:, :E] * sigp(l32[:, :E] * scale32[0] + base32[:E])
        dz_post = ghp * 2 * sigp(l32[:, E:2 * E] * scale32[E] + base32[E:2 * E])
        zr = l32[:, 2 * E:] * scale32[2 * E] + base32[2 * E:]
        cf = torch.softmax(zr, -1)
        dcoef = ghr @ perm_t
        dzc = cf * (dcoef - (dcoef * cf).sum(-1, keepdim=True))
        dl32 = torch.cat([dz_pre * scale32[0], dz_post * scale32[E], dzc * scale32[2 * E]], -1)
        ds_ref = torch.stack([(dz_pre * l32[:, :E]).sum() / E,
                              (dz_post * l32[:, E:2 * E]).sum() / E,
                              (dzc * l32[:, 2 * E:]).sum() / NP])
        db_ref = torch.cat([dz_pre.sum(0), dz_post.sum(0), dzc.sum(0)])
        print(f'   dscale op/triton/fp32: {ds_o.tolist()} | {ds_t.tolist()} | {ds_ref.tolist()}')
        print(f'   dbase  max|op-tri| {(db_o - dbase).abs().max().item():.2e}'
              f'  max|op-fp32| {(db_o - db_ref).abs().max().item():.2e}'
              f'  max|tri-fp32| {(dbase - db_ref).abs().max().item():.2e}')
        gl_ref = (dl32 * rstd.unsqueeze(-1))
        for nm, tt in [('pre', slice(0, E)), ('post', slice(E, 2 * E)), ('res', slice(2 * E, N32))]:
            print(f'   gl[{nm}] vs fp32   : op {(gl_o[:, tt].float() - gl_ref[:, tt]).abs().max().item():.2e}'
                  f'  triton {(gl_t[:, tt].float() - gl_ref[:, tt]).abs().max().item():.2e}')
        coef_ref = -(dl32 * lf).sum(-1) * rstd.pow(3) / (E * h)
        gx_ref = (hpre8[:, :E].unsqueeze(-1) * g.float().unsqueeze(1)
                  + coef_ref.view(sb, 1, 1) * x.float()).view(sb, E * h)
        d = (gx_o.float() - gx_ref).abs().max().item()
        d2 = (gx_t.float() - gx_ref).abs().max().item()
        print(f'   grad_x     vs fp32  : op {d:.2e}  triton {d2:.2e}')
    return dict(g=g, xf=xf, ghp=ghp, ghr=ghr, perm_t=perm_t, logits=logits, rstd=rstd,
                hpre8=hpre8, scale32=scale32, base32=base32, x=x, sb=sb, h=h,
                l=l, scale3=scale32[[0, E, 2 * E]], coef=coef)


def main():
    for sb, h in [(8, 1024), (64, 1024), (4096, 1024), (1033, 1024), (512, 1024), (4096, 512)]:
        run_shape(sb, h, check=True)

    t = run_shape(4096, 1024, check=False)
    sb, h = t['sb'], t['h']
    x = t['x']

    def pieces():
        l = t['logits'].float() * t['rstd'].unsqueeze(-1)
        dl, dscale3, dbase = lite_pre_backward(
            t['g'], x, t['ghp'].contiguous(), t['ghr'].contiguous(), t['perm_t'], l,
            t['scale3'], t['base32'])
        gl = (dl * t['rstd'].unsqueeze(-1)).to(t['logits'].dtype)
        drstd = (dl * t['logits'].float()).sum(-1)
        coef = (drstd * -(t['rstd'].pow(3)) / (E * h)).to(torch.bfloat16).unsqueeze(-1).unsqueeze(-1)
        gx = lite_grad_x(t['g'], t['hpre8'][:, :E].contiguous(), coef * x).view(sb, E * h)

    t_pieces = bench(pieces)
    t_op = bench(lambda: ext_g.lite_pre_grad(
        t['g'], t['xf'], t['ghp'], t['ghr'], t['perm_t'], t['logits'], t['rstd'], t['hpre8'],
        t['scale32'], t['base32']))
    print(f'\ntriton path + glue pieces : {t_pieces:.3f} ms')
    print(f'LitePreGrad op alone      : {t_op:.3f} ms')


if __name__ == '__main__':
    main()
