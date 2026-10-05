# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""eps passthrough probe for the Ascend C LitePreHeads op.

eps is the RMSNorm epsilon (rstd = rsqrt(mean(x^2) + eps)).  It used to be
a compile-time constant in the kernel; it now rides a 5th fp32 [8]-lane
tensor input (one 32B DataCopy per block, lane 0 read as the scalar), so a
runtime configuration can change it with zero structural cost.

Two levels:

  op      direct ext.lite_pre_heads calls with eps8 = 1e-5 / 1.0 / 1e-1;
          rstd must match the torch formula of ITS OWN eps and move away
          from the formula of the other eps by the amount eps itself
          shifts rstd (orders of magnitude here);
  module  MHCLite.hc_pre on the final stack with module.norm_eps patched,
          vs the fp32 reference built with the same eps -- proves the
          wrapper threads the module's norm_eps into the op on both the
          chain (G/H merge) and non-chain paths.

Gate: the matched-eps rel must stay at the bf16 noise floor of the campaign
(<~3e-3 for these outputs) while the mismatched-eps rel must be far above
it (eps=1.0 changes rstd by ~6% on x~N(0,1.5^2) rows, which the heads and
y then amplify).
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault(
    'ASCEND_CUSTOM_OPP_PATH',
    str(Path.home() / 'work/dataset/huashan_zhh_guiyang_turbo/github/cann-ops/build_out'))

# final-stack env: exercise the chain (G/H) path; the plain path flips the
# env var off per call below
os.environ['MHC_LITE_ASCENDC'] = '1'
os.environ['MHC_LITE_ASCENDC_GRAD'] = '1'
os.environ['MHC_LITE_ASCENDC_CHAIN'] = '1'

import torch  # noqa: E402
import torch_npu  # noqa: E402

import bench_lite_e2e as bench  # noqa: E402
from mindspeed_llm.ops.ascendc.mhc_lite_ac import _ext  # noqa: E402

E, H = bench.E, bench.H
DEV = bench.DEV
N32 = 32


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def main():
    torch.manual_seed(11)
    print('== op level: rstd must follow ITS OWN eps ==', flush=True)

    sb = 1033  # tail-block rows: the eps copy happens per 8-row block too
    xf = (torch.randn(sb, E * H, device=DEV) * 1.5).to(torch.bfloat16)
    logits = (torch.randn(sb, N32, device=DEV) * 0.5).to(torch.bfloat16)
    scale32 = torch.empty(N32, device=DEV)
    scale32[0:E] = 0.011
    scale32[E:2 * E] = 0.013
    scale32[2 * E:] = 0.017
    base = (torch.randn(N32, device=DEV) * 0.5).contiguous()

    ms = xf.float().pow(2).mean(-1)
    rstd_ref = {e: torch.rsqrt(ms + e) for e in (1e-5, 1e-1, 1.0)}
    print(f'eps shift sanity: |rstd(1e-5)-rstd(1.0)| / |rstd| rel = '
          f'{rel(rstd_ref[1e-5], rstd_ref[1.0]):.2e} (eps itself moves rstd by this much)')

    ok = True
    for eps in (1e-5, 1e-1, 1.0):
        eps8 = torch.full((8,), eps, dtype=torch.float32, device=DEV)
        rstd = _ext().lite_pre_heads(xf, logits, scale32, base, eps8)[4]
        matched = rel(rstd, rstd_ref[eps])
        others = [rel(rstd, rstd_ref[o]) for o in rstd_ref if o != eps]
        passed = matched < 1e-6 and min(others) > 1e-3
        ok &= passed
        print(f'  eps={eps:<6g} vs own formula {matched:.2e} | vs other eps '
              f'{min(others):.2e} -> {"PASS" if passed else "FAIL"}', flush=True)

    print('\n== module level: norm_eps threads through the wrapper ==', flush=True)
    bench.warm_native_post(torch.bfloat16)
    module, refmod = _pair()
    x, gy, gpost, gres, gout = bench.make_inputs(512, 2, torch.bfloat16)
    perm_flat = bench.perm_mats_flat(E)
    w = module.hc_fn.weight.detach()
    gamma, scale, base3 = (module.hc_gamma.detach(), module.hc_scale.detach(),
                           module.hc_base.detach())
    wr, gr, sr, br = (t.float().requires_grad_() for t in (w, gamma, scale, base3))
    xr = x.float().requires_grad_()

    for eps in (1e-5, 1.0):
        for chain in (True, False):
            os.environ['MHC_LITE_ASCENDC_CHAIN'] = '1' if chain else ''
            module.norm_eps = eps
            with torch.no_grad():
                y, h_post, h_res = module.hc_pre(x)
            y_r, post_r, res_r, _ = bench.lite_ref(xr, wr, gr, sr, br, perm_flat, eps)
            r_ok = rel(y, y_r) < 3e-3 and rel(h_post, post_r) < 3e-3 and rel(h_res, res_r) < 3e-3
            ok &= r_ok
            tag = 'chain' if chain else 'split'
            print(f'  eps={eps:<6g} {tag:5s}: y {rel(y, y_r):.2e} h_post {rel(h_post, post_r):.2e} '
                  f'h_res {rel(h_res, res_r):.2e} -> {"PASS" if r_ok else "FAIL"}', flush=True)

    # Mismatch control with amplified head scales: at production scale
    # (0.011) the eps-induced shift in the heads (~3e-4) sits below the bf16
    # rounding floor of y (~6e-4), so a wrong-eps output would still "match"
    # a wrong reference.  Scale 0.4/0.5/0.6 pushes the sigmoid arguments far
    # enough that eps=1.0 vs eps=1e-5 separates by an order of magnitude
    # above the floor.
    big_scale = torch.tensor([0.4, 0.5, 0.6], device=DEV)
    with torch.no_grad():
        module.hc_scale.copy_(big_scale)
    sr_big = big_scale.float().requires_grad_()
    os.environ['MHC_LITE_ASCENDC_CHAIN'] = '1'

    module.norm_eps = 1e-5
    with torch.no_grad():
        y_match, hp_match, _ = module.hc_pre(x)
    y_r_big, post_r_big, *_ = bench.lite_ref(xr, wr, gr, sr_big, br, perm_flat, 1e-5)
    m_floor = max(rel(y_match, y_r_big), rel(hp_match, post_r_big))

    module.norm_eps = 1.0
    with torch.no_grad():
        y_big, hp_big, _ = module.hc_pre(x)
    gap_y = rel(y_big, y_r_big)
    gap_p = rel(hp_big, post_r_big)
    gap = max(gap_y, gap_p)
    ok &= gap > 3e-3 and m_floor < 3e-3
    print(f'\nmismatch control (scale 0.4/0.5/0.6): matched-eps floor {m_floor:.2e}; '
          f'hc_pre(eps=1.0) vs ref(eps=1e-5) y {gap_y:.2e} h_post {gap_p:.2e} '
          f'(mismatch must sit far above the 3e-3 floor) '
          f'-> {"PASS" if gap > 3e-3 and m_floor < 3e-3 else "FAIL"}', flush=True)

    try:
        commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        commit = '?'
    print(f'\n[eps passthrough] commit {commit}: {"ALL PASS" if ok else "FAILURES PRESENT"}')
    sys.exit(0 if ok else 1)


def _pair():
    module = bench.build_lite()
    with bench._EnvCleared():
        refmod = bench.build_lite()
    with torch.no_grad():
        refmod.hc_fn.weight.copy_(module.hc_fn.weight)
        refmod.hc_gamma.copy_(module.hc_gamma)
        refmod.hc_scale.copy_(module.hc_scale)
        refmod.hc_base.copy_(module.hc_base)
    return module, refmod


if __name__ == '__main__':
    main()
