# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Analyze the crashed cloud run's checkpoints: MHC z-saturation + live op parity.

Part 1 (trajectory): for every saved iter, estimate each MHC module's res-head
logit scale  z ~ |s2| * ||W_res_row|| + |b2|  (the op RMS-normalizes x internally,
so the estimate does not depend on activations). The aclnn backward defect
onset is |z| ~ 5, sign flips beyond ~20 (minrepro_pre_sinkhorn_ds2.py).

Part 2 (live parity): take the modules with the largest z from a chosen ckpt,
run aclnnMhcPreSinkhorn fwd+bwd on random bf16 activations with THOSE EXACT
trained parameters, and compare against the fp64 reference. x scale is
irrelevant to the comparison (both arms see the same tensor), so random x is
sufficient. If the production parameters sit in the poisoned zone, ds2 is
wrong here and now - closing the loop from offline repro to the crashed run.

Usage:  python experiments/mhc_debug/ckpt_z_analysis.py <ckpt_root> \
            [--iters 2000 6000 14000 25000] [--parity-iter 2000] [--top 6]
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402


def module_stats(model):
    out = {}
    for k in model:
        m = re.match(r'decoder\.layers\.(\d+)\.(attn_mhc|mlp_mhc)\.hc_scale', k)
        if not m:
            continue
        pre = f'decoder.layers.{m.group(1)}.{m.group(2)}'
        scale, base, W = model[k], model[pre + '.hc_base'], model[pre + '.hc_fn.weight']
        wres = W[8:]
        zstd = abs(scale[2].item()) * wres.norm(dim=1).mean().item()
        zmax = abs(scale[2].item()) * wres.norm(dim=1).max().item() \
            + base[8:].abs().max().item()
        out[pre] = dict(s2=scale[2].item(), zstd=zstd, zmax=zmax)
    return out


def load_model(ckpt_root, it):
    path = Path(ckpt_root) / f'iter_{it:07d}' / 'mp_rank_00' / 'model_optim_rng.pt'
    return torch.load(path, map_location='cpu', weights_only=False)['model']


def z_trajectory(ckpt_root, iters):
    print('== z_sat trajectory (per-module |s2|*max||W_res||+|b2|max; '
          'defect onset ~5, sign-flip ~20) ==')
    print(f'{"iter":>7s} {"median":>7s} {"max":>7s} {"max@":>28s} '
          f'{"n>5":>5s} {"n>10":>5s} {"n>20":>5s}')
    for it in iters:
        st = module_stats(load_model(ckpt_root, it))
        if not st:
            print(f'{it:7d}  no mhc tensors')
            continue
        zs = sorted(st.items(), key=lambda kv: -kv[1]['zmax'])
        vals = [v['zmax'] for v in st.values()]
        med = sorted(vals)[len(vals) // 2]
        print(f'{it:7d} {med:7.2f} {zs[0][1]["zmax"]:7.2f} {zs[0][0]:>28s} '
              f'{sum(v > 5 for v in vals):5d} {sum(v > 10 for v in vals):5d} '
              f'{sum(v > 20 for v in vals):5d}')


def live_parity(ckpt_root, it, top, s_len):
    import torch_npu  # noqa: F401
    from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend

    sys.path.insert(0, str(REPO / 'experiments' / 'mhc_debug'))
    from parity_fused_bwd import pre_ref_grads_fp64, compare

    torch_npu.npu.set_device(0)
    DEV = torch.device('npu:0')
    E, ITERS, EPS, NORM_EPS, H = 4, 20, 1e-6, 1e-5, 1024

    model = load_model(ckpt_root, it)
    st = module_stats(model)
    ranked = sorted(st.items(), key=lambda kv: -kv[1]['zmax'])[:top]
    controls = sorted(st.items(), key=lambda kv: kv[1]['zmax'])[:2]

    g = torch.Generator(device='cpu').manual_seed(0)
    x = torch.randn(s_len, 1, E, H, generator=g).to(torch.bfloat16)
    gy = torch.randn(s_len, 1, H, generator=g).to(torch.bfloat16)
    gpost = torch.randn(s_len, 1, E, generator=g).float()
    gcomb = torch.randn(s_len, 1, E, E, generator=g).float()

    print(f'\n== live parity with iter_{it:07d} trained params '
          f'(S={s_len}; op vs fp64, same tensors) ==')
    for pre, info in ranked + controls:
        scale = model[pre + '.hc_scale'].to(DEV)
        base = model[pre + '.hc_base'].to(DEV)
        w = model[pre + '.hc_fn.weight'].to(DEV)
        xd, gyd = x.to(DEV), gy.to(DEV)

        xg = xd.detach().requires_grad_()
        wg = w.detach().requires_grad_()
        sg = scale.detach().requires_grad_()
        bg = base.detach().requires_grad_()
        y, post, comb = mhc_pre_sinkhorn_ascend(xg, wg, sg, bg, E, ITERS, EPS, NORM_EPS)
        dx, dw, dscale, dbase = torch.autograd.grad(
            (y, post, comb), (xg, wg, sg, bg),
            (gyd.to(y.dtype), gpost.to(DEV), gcomb.to(DEV)))

        rdx, rdw, rds, rdb = pre_ref_grads_fp64(
            xd.cpu(), w.cpu(), scale.cpu(), base.cpu(),
            gyd.to(y.dtype).cpu(), gpost, gcomb)

        print(f'  [{pre}] s2 {info["s2"]:+.3f} zmax_est {info["zmax"]:.2f}')
        compare('ds2', dscale[2], rds[2])
        compare('dscale', dscale, rds)
        compare('dx', dx, rdx)
        compare('dw', dw, rdw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt_root')
    ap.add_argument('--iters', type=int, nargs='+',
                    default=[2000, 6000, 14000, 25000])
    ap.add_argument('--parity-iter', type=int, default=2000)
    ap.add_argument('--top', type=int, default=6)
    ap.add_argument('--s-len', type=int, default=512)
    args = ap.parse_args()
    z_trajectory(args.ckpt_root, args.iters)
    live_parity(args.ckpt_root, args.parity_iter, args.top, args.s_len)


if __name__ == '__main__':
    main()
