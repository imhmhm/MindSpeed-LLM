# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Root-cause probe for the dscale elevation the full-shape matrix found.

probe_finalstack_accuracy.py --quick showed the final stack's dscale ~10x the
Tier-0 module's own fp32 distance at some shapes (1033x1 FAIL: rel 3.5e-2 vs
t0 floor 3.1e-3; 4096x1 1.1e-2 vs 6.8e-4).  This probe isolates the cause at
those shapes: same inputs, side-by-side stacks,

  ac5   final stack (E forward + H grad, chained)
  ac4   same kernels, chain merge off (call-time env toggle)
  t0    Tier-0 torch module
  t2    triton pre (the incumbent production fast path)
  fp32  reference

per scale component (s0 pre / s1 post / s2 res), over several input draws --
dscale is a scalar double-sum whose bf16 noise is heavy-tailed, so one draw
proves nothing about either chain.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault(
    'ASCEND_CUSTOM_OPP_PATH',
    str(Path.home() / 'work/dataset/huashan_zhh_guiyang_turbo/github/cann-ops/build_out'))

import torch  # noqa: E402
import torch_npu  # noqa: E402

import bench_lite_e2e as bench  # noqa: E402

E, H = bench.E, bench.H
STACK_KEYS = ('MHC_LITE_TRITON', 'MHC_LITE_NATIVE_POST_BWD', 'MHC_LITE_POST_DIRECT',
              'MHC_LITE_ASCENDC', 'MHC_LITE_ASCENDC_GRAD', 'MHC_LITE_ASCENDC_CHAIN')


class _EnvSet:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.saved = {k: os.environ.pop(k, None) for k in STACK_KEYS}
        for k, v in self.kv.items():
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is not None:
                os.environ[k] = v
        return False


def build(stack):
    """stack in {'t0','t2','ac'}: build one module under that env."""
    kw = {'MHC_LITE_ASCENDC': '1', 'MHC_LITE_ASCENDC_GRAD': '1',
          'MHC_LITE_ASCENDC_CHAIN': '1'}
    if stack == 't0':
        kw = {}
    elif stack == 't2':
        kw = {'MHC_LITE_TRITON': '1'}
    with _EnvSet(**kw):
        return bench.build_lite()


def copy_weights(dst, src):
    with torch.no_grad():
        dst.hc_fn.weight.copy_(src.hc_fn.weight)
        dst.hc_gamma.copy_(src.hc_gamma)
        dst.hc_scale.copy_(src.hc_scale)
        dst.hc_base.copy_(src.hc_base)


def dscale_of(module, x, gy, gpost, gres, gout):
    _, grads = bench._module_grads(module, x, gy, gpost, gres, gout)
    return grads[3].detach().float()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapes', default='1033x1,4096x1,512x2',
                        help='comma list sxb, the shapes to break down')
    parser.add_argument('--draws', type=int, default=8)
    args = parser.parse_args()
    shapes = [tuple(int(v) for v in sh.split('x')) for sh in args.shapes.split(',')]

    torch.manual_seed(7)
    bench.warm_native_post(torch.bfloat16)

    modules = {}
    modules['ac'] = build('ac')
    modules['t0'] = build('t0')
    modules['t2'] = build('t2')
    for m in modules.values():
        copy_weights(m, modules['ac'])
    perm_flat = bench.perm_mats_flat(E)

    for s, b in shapes:
        print(f'[s={s} b={b}] per-draw dscale rel vs fp32 (s0/s1/s2), |dscale| ref', flush=True)
        for seed in range(args.draws):
            torch.manual_seed(1000 + seed)
            x, gy, gpost, gres, gout = bench.make_inputs(s, b, torch.bfloat16)

            xr = x.detach().float().requires_grad_()
            wr = modules['ac'].hc_fn.weight.detach().float().requires_grad_()
            gr = modules['ac'].hc_gamma.detach().float().requires_grad_()
            sr = modules['ac'].hc_scale.detach().float().requires_grad_()
            br = modules['ac'].hc_base.detach().float().requires_grad_()
            y_r, post_r, res_r, out_r = bench.lite_ref(xr, wr, gr, sr, br, perm_flat, 1e-5)
            refs = torch.autograd.grad((y_r, post_r, res_r, out_r), (xr, wr, gr, sr, br),
                                       (gy.float(), gpost, gres, gout.float()))
            dref = refs[3].detach()
            mag = dref.abs().max().item()

            row = {'ac5': dscale_of(modules['ac'], x, gy, gpost, gres, gout)}
            os.environ['MHC_LITE_ASCENDC_CHAIN'] = '0'
            row['ac4'] = dscale_of(modules['ac'], x, gy, gpost, gres, gout)
            os.environ['MHC_LITE_ASCENDC_CHAIN'] = '1'
            with bench._EnvCleared():
                row['t0'] = dscale_of(modules['t0'], x, gy, gpost, gres, gout)
                row['t2'] = dscale_of(modules['t2'], x, gy, gpost, gres, gout)

            def rel(d):
                return ((d - dref).abs() / mag).tolist()

            print(f'  seed {seed}: '
                  + ' | '.join(f"{k} {rel(d)[0]:.1e}/{rel(d)[1]:.1e}/{rel(d)[2]:.1e}"
                               for k, d in row.items())
                  + f'  |dscale| {mag:.2e}', flush=True)
            del x, gy, gpost, gres, gout, xr, y_r, post_r, res_r, out_r, refs
            torch.npu.empty_cache()


if __name__ == '__main__':
    main()
