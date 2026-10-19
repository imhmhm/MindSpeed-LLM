# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Discriminate op defect from intrinsic ill-conditioning on the scale cliff.

parity_fused_bwd.py found the fused pre op's GRADIENT error exploding once
hc_scale entries reach ~5 (dscale rel ~1, dx rel ~6e-2). Before blaming the
op, the same fp64 reference must be applied to fp32 implementations of the
identical math: if eager fp32 shows the same growth, the cliff is a property
of the saturated-sigmoid/sinkhorn parameterization (cancellation sums), not
of the aclnn kernel.

Three arms share one fp64 reference and one set of quantized inputs/grads:
  op     aclnnMhcPreSinkhorn via mhc_pre_sinkhorn_ascend (training fused path)
  npu32  mhc.py non-fused branch, fp32 eager on NPU (training fallback path)
  cpu32  the same formulas, fp32 eager on CPU (algorithmic conditioning only)

Saturation is driven by z = mix * scale + base, i.e. the PRODUCT of hc_scale
and the hc_fn logit magnitude, so the matrix sweeps scale alone and W-growth
x scale pairs that hold comparable z ranges. dscale is printed per component
(ds0 sigmoid head, ds1 sigmoid head, ds2 sinkhorn res head).

Usage:  python experiments/mhc_debug/parity_scale_cliff.py [--seeds 3]
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
S, B, H = 4096, 1, 1024


def pre_chain(x, w, scale, base, dtype):
    """The mhc.py non-fused formulas in the requested dtype; differentiable."""
    shape = x.size()
    xf = x.to(dtype).flatten(2)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, w.to(dtype)) * rsqrt
    mixes = mixes.permute(1, 0, 2)

    pre = torch.sigmoid(mixes[..., :E] * scale.to(dtype)[0] + base.to(dtype)[:E]) + EPS
    post = 2 * torch.sigmoid(mixes[..., E:2 * E] * scale.to(dtype)[1]
                             + base.to(dtype)[E:2 * E])
    comb = (mixes[..., 2 * E:].view(B, S, E, E) * scale.to(dtype)[2]
            + base.to(dtype)[2 * E:].view(1, 1, E, E))
    comb = comb.softmax(-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)

    y = torch.sum(pre.permute(1, 0, 2).unsqueeze(-1) * xf.view(shape), dim=2)
    return y, post.permute(1, 0, 2), comb.permute(1, 0, 2, 3)


def grads_of(fn, x, w, scale, base, gy, gpost, gcomb):
    leaves = [t.detach().requires_grad_() for t in (x, w, scale, base)]
    y, post, comb = fn(*leaves)
    return torch.autograd.grad((y, post, comb), leaves,
                               (gy.to(y.dtype), gpost.to(post.dtype), gcomb.to(comb.dtype)))


def stats(op, ref):
    op, ref = op.float().cpu(), ref.float().cpu()
    md = (op - ref).abs().max().item()
    rel = md / (ref.abs().max().item() + 1e-30)
    bias = (op - ref).mean().item() / (ref.abs().mean().item() + 1e-30)
    return rel, bias


def make_inputs(seed, w_mult=1.0, scale_val=None):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(S, B, E, H, generator=g).to(torch.bfloat16)
    w = torch.randn(MIX, E * H, generator=g).float() * (0.02 * w_mult)
    if scale_val is None:
        scale = torch.randn(3, generator=g).float() * 0.05
    else:
        scale = torch.full((3,), scale_val, dtype=torch.float32)
    base = torch.randn(MIX, generator=g).float() * 0.1
    return x.to(DEV), w.to(DEV), scale.to(DEV), base.to(DEV)


CASES = [
    ('baseline', {}),
    ('scale0.5', {'scale_val': 0.5}),
    ('scale1', {'scale_val': 1.0}),
    ('scale2', {'scale_val': 2.0}),
    ('scale3', {'scale_val': 3.0}),
    ('scale5', {'scale_val': 5.0}),
    ('scale20', {'scale_val': 20.0}),
    ('Wx3_s2', {'w_mult': 3.0, 'scale_val': 2.0}),
    ('Wx10_s1', {'w_mult': 10.0, 'scale_val': 1.0}),
    ('Wx30_s0.5', {'w_mult': 30.0, 'scale_val': 0.5}),
    ('Wx10_s3', {'w_mult': 10.0, 'scale_val': 3.0}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)

    print(f'== scale cliff discrimination (S={S} B={B}; rel vs fp64 ref; '
          f'ds0/ds1/ds2 = dscale components) ==')
    for seed in range(args.seeds):
        print(f'- seed {seed} -')
        for label, kw in CASES:
            x, w, scale, base = make_inputs(seed, **kw)
            g = torch.Generator(device='cpu').manual_seed(seed + 9999)
            gy = torch.randn(S, B, H, generator=g).to(torch.bfloat16)
            gpost = torch.randn(S, B, E, generator=g).float()
            gcomb = torch.randn(S, B, E, E, generator=g).float()
            gy, gpost, gcomb = gy.to(DEV), gpost.to(DEV), gcomb.to(DEV)

            # fp64 reference on CPU (same quantized input values)
            rdx, rdw, rds, rdb = grads_of(
                lambda *a: pre_chain(*a, torch.float64),
                *(t.cpu() for t in (x, w, scale, base)),
                gy.cpu(), gpost.cpu(), gcomb.cpu())

            arms = {'op': None, 'npu32': None, 'cpu32': None}
            arms['op'] = grads_of(
                lambda x_, w_, s_, b_: mhc_pre_sinkhorn_ascend(
                    x_, w_, s_, b_, E, ITERS, EPS, NORM_EPS),
                x, w, scale, base, gy, gpost, gcomb)
            arms['npu32'] = grads_of(
                lambda *a: pre_chain(*a, torch.float32),
                x, w, scale, base, gy, gpost, gcomb)
            arms['cpu32'] = grads_of(
                lambda *a: pre_chain(*a, torch.float32),
                *(t.cpu() for t in (x, w, scale, base)),
                gy.cpu(), gpost.cpu(), gcomb.cpu())

            line = [f'  [{label:11s}]']
            for name, (dx, dw, ds, db) in arms.items():
                ds_rel = [stats(ds[i], rds[i].cpu())[0] for i in range(3)]
                ds_bias2 = stats(ds[2], rds[2].cpu())[1]
                dx_rel = stats(dx, rdx)[0]
                dw_rel = stats(dw, rdw)[0]
                line.append(
                    f'{name}: dx {dx_rel:.1e} dw {dw_rel:.1e} '
                    f'ds0/1/2 {ds_rel[0]:.1e}/{ds_rel[1]:.1e}/{ds_rel[2]:.1e} '
                    f'(ds2bias {ds_bias2:+.1e})')
            print('\n            '.join(line))


if __name__ == '__main__':
    main()
