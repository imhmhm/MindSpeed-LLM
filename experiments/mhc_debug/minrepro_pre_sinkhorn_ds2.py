# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Minimal repro: aclnnMhcPreSinkhorn backward ds2 error vs res-head logit scale.

parity_scale_cliff.py established that the fused pre op's gradient error
explodes once z_res = mix*s2 + b2 reaches saturation magnitude, while the
same math in eager fp32 stays clean. This script shrinks that finding to an
8-token case with the res-head logit magnitude as the only swept variable:
the table below (op vs fp64 side by side) is the repro attachment for an
operator issue - no training, no luck involved.

S=8 tokens, x/W fixed small; scale2 swept so z_res = mix * s2 spans 0..50.
W (hc_fn) is scaled 10x in the second pass so mix magnitude and s2 are
decoupled: if the error tracks z (either way of reaching it), the trigger is
the saturated softmax/sinkhorn vjp, not the scalar s2 itself.

Usage:  python experiments/mhc_debug/minrepro_pre_sinkhorn_ds2.py [--save]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402

from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend  # noqa: E402

E, ITERS, EPS, NORM_EPS = 4, 20, 1e-6, 1e-5
MIX = (2 + E) * E
DEV = torch.device('npu:0')
S, B, H = 8, 1, 128  # backward tiling requires hidden divisible by 128


def reference(x, w, scale, base):
    """fp64 replica of the mhc.py non-fused formulas (see parity_scale_cliff)."""
    shape = x.size()
    xf = x.double().flatten(2)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, w.double()) * rsqrt
    mixes = mixes.permute(1, 0, 2)
    pre = torch.sigmoid(mixes[..., :E] * scale.double()[0] + base.double()[:E]) + EPS
    post = 2 * torch.sigmoid(mixes[..., E:2 * E] * scale.double()[1]
                             + base.double()[E:2 * E])
    comb = (mixes[..., 2 * E:].view(B, S, E, E) * scale.double()[2]
            + base.double()[2 * E:].view(1, 1, E, E))
    comb = comb.softmax(-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    y = torch.sum(pre.permute(1, 0, 2).unsqueeze(-1) * xf.view(shape), dim=2)
    return y, post.permute(1, 0, 2), comb.permute(1, 0, 2, 3)


def grads(fn, tensors, grads_out):
    leaves = [t.detach().requires_grad_() for t in tensors]
    y, post, comb = fn(*leaves)
    return torch.autograd.grad((y, post, comb), leaves, grads_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true',
                    help='save the worst-case tensors to worst_ds2_case.pt')
    args = ap.parse_args()
    torch_npu.npu.set_device(0)

    g = torch.Generator(device='cpu').manual_seed(0)
    x = torch.randn(S, B, E, H, generator=g).to(torch.bfloat16).to(DEV)
    gy = torch.randn(S, B, H, generator=g).to(torch.bfloat16).to(DEV)
    gpost = torch.randn(S, B, E, generator=g).float().to(DEV)
    gcomb = torch.randn(S, B, E, E, generator=g).float().to(DEV)

    print(f'== min repro: ds2 (grad of hc_scale[2]) vs z_res magnitude '
          f'(S={S} B={B} H={H}, seed fixed) ==')
    worst = None
    for w_mult, s2_list in [(1.0, [0.05, 0.5, 1, 2, 3, 5, 10, 20, 50]),
                            (10.0, [0.05, 0.2, 0.5, 1, 2, 5])]:
        w = (torch.randn(MIX, E * H, generator=g).float() * 0.02 * w_mult).to(DEV)
        base = (torch.randn(MIX, generator=g).float() * 0.1).to(DEV)
        print(f'- W x{w_mult:.0f} (mix sigma ~{1.28 * w_mult:.1f}) -')
        print(f'  {"s2":>5s} {"max|z_res|":>10s} {"ds2 op":>12s} {"ds2 fp64":>12s} '
              f'{"ds2 rel":>9s} {"dx rel":>8s} {"dw res rel":>10s}')
        for s2 in s2_list:
            scale = torch.tensor([0.05, 0.05, s2], dtype=torch.float32).to(DEV)

            op_g = grads(lambda *a: mhc_pre_sinkhorn_ascend(
                *a, E, ITERS, EPS, NORM_EPS), (x, w, scale, base),
                (gy, gpost, gcomb))
            ref_g = grads(reference, (t.cpu() for t in (x, w, scale, base)),
                          (t.cpu() for t in (gy, gpost, gcomb)))

            # z_res magnitude the op actually sees (from the fp64 mixes)
            xf = x.double().cpu().flatten(2)
            rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
            mixes = torch.nn.functional.linear(xf, w.double().cpu()) * rsqrt
            zmax = (mixes[..., 2 * E:] * s2
                    + base.double().cpu()[2 * E:]).abs().max().item()

            ds2_op = op_g[2][2].item()
            ds2_ref = ref_g[2][2].item()
            rel = abs(ds2_op - ds2_ref) / (abs(ds2_ref) + 1e-30)
            dx_rel = ((op_g[0].float().cpu() - ref_g[0]).abs().max().item()
                      / (ref_g[0].abs().max().item() + 1e-30))
            dwres_op = op_g[1][2 * E:].float().cpu()
            dwres_ref = ref_g[1][2 * E:]
            dw_rel = ((dwres_op - dwres_ref).abs().max().item()
                      / (dwres_ref.abs().max().item() + 1e-30))
            print(f'  {s2:5.2f} {zmax:10.2f} {ds2_op:12.4e} {ds2_ref:12.4e} '
                  f'{rel:9.2e} {dx_rel:8.1e} {dw_rel:10.1e}')
            if worst is None or rel > worst[0]:
                worst = (rel, s2, w_mult)

    print(f'\nworst: rel {worst[0]:.2e} at s2={worst[1]} W x{worst[2]:.0f}')
    if args.save and worst is not None:
        w_mult = worst[2]
        w = (torch.randn(MIX, E * H, generator=g).float() * 0.02 * w_mult).to(DEV)
        scale = torch.tensor([0.05, 0.05, worst[1]], dtype=torch.float32).to(DEV)
        base = (torch.randn(MIX, generator=g).float() * 0.1).to(DEV)
        path = Path(__file__).parent / 'worst_ds2_case.pt'
        torch.save({'x': x.cpu(), 'w': w.cpu(), 'scale': scale.cpu(),
                    'base': base.cpu(), 'gy': gy.cpu(), 'gpost': gpost.cpu(),
                    'gcomb': gcomb.cpu()}, path)
        print(f'saved {path}')


if __name__ == '__main__':
    main()
