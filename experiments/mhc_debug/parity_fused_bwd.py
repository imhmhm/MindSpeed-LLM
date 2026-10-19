# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Offline fwd+bwd parity of the mainline fused MHC ops vs an fp64 reference.

Probes the aclnnMhcPreSinkhorn / aclnnMhcPost path that ailab_slm_mhc runs
with ``use-fused-mhc: true``:

  * gradient parity including SYSTEMATIC BIAS (mean signed error), which a
    maxdiff-only comparison cannot see - a small but consistently signed
    gradient error is the classic "trains fine for N steps then blows up"
    mechanism;
  * saturation sweep: hc_scale / hc_fn magnitude growth during training drives
    sigmoid/softmax saturation and pushes sinkhorn toward a hard permutation;
  * degenerate inputs (tiny x, near-one-hot res head, B>=2 shapes);
  * soak: repeated fwd+bwd on fixed inputs, watching for rare NaN and
    run-to-run nondeterminism;
  * bitwise determinism of every output across two identical calls.

The reference is a CPU float64 replica of the mhc.py non-fused formulas fed
the SAME quantized input values (bf16 x / fp32 params are cast, not resampled),
so every reported difference is attributable to the fused op internals.

Usage:  ASCEND_RT_VISIBLE_DEVICES=<free card> python experiments/mhc_debug/parity_fused_bwd.py
        [--seeds 5] [--soak 2000] [--skip-post] [--save-failing]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402

from mindspeed_llm.ops.npu_mhc import mhc_post_ascend, mhc_pre_sinkhorn_ascend  # noqa: E402

E, ITERS, EPS, NORM_EPS = 4, 20, 1e-6, 1e-5
MIX = (2 + E) * E
DEV = torch.device('npu:0')

# rel maxdiff and |bias|/mean|ref| levels treated as "interesting" (not hard
# failures - this is a probe; anything near these lines deserves a look)
REL_FLAG, BIAS_FLAG = 5e-2, 1e-2


# ---------------------------------------------------------------------------
# fp64 reference (CPU), exact replica of the mhc.py non-fused formulas
# ---------------------------------------------------------------------------

def pre_ref_fp64(x, w, scale, base):
    """hc_pre non-fused branch in float64. Inputs are the same values the op sees."""
    s, b = x.shape[0], x.shape[1]
    shape = x.size()
    xf = x.double().flatten(2)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, w.double()) * rsqrt        # [s,b,24]
    mixes = mixes.permute(1, 0, 2)                                    # [b,s,24]

    pre = torch.sigmoid(mixes[..., :E] * scale.double()[0] + base.double()[:E]) + EPS
    post = 2 * torch.sigmoid(mixes[..., E:2 * E] * scale.double()[1] + base.double()[E:2 * E])
    comb = (mixes[..., 2 * E:].view(b, s, E, E) * scale.double()[2]
            + base.double()[2 * E:].view(1, 1, E, E))
    comb = comb.softmax(-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)

    y = torch.sum(pre.permute(1, 0, 2).unsqueeze(-1) * xf.view(shape), dim=2)
    return y, pre.permute(1, 0, 2), post.permute(1, 0, 2), comb.permute(1, 0, 2, 3)


def pre_ref_grads_fp64(x, w, scale, base, gy, gpost, gcomb):
    xg = x.double().detach().requires_grad_()
    wg = w.double().detach().requires_grad_()
    sg = scale.double().detach().requires_grad_()
    bg = base.double().detach().requires_grad_()
    y, pre, post, comb = pre_ref_fp64(xg, wg, sg, bg)
    grads = torch.autograd.grad((y, post, comb), (xg, wg, sg, bg),
                                (gy.double(), gpost.double(), gcomb.double()))
    return grads  # dx, dw, dscale, dbase


def post_ref_fp64(y, res, post, comb):
    y64, r64 = y.double(), res.double()
    out = (post.double().unsqueeze(-1) * y64.unsqueeze(-2)
           + torch.sum(comb.double().unsqueeze(-1) * r64.unsqueeze(-2), dim=2))
    return out


# ---------------------------------------------------------------------------
# fused-path runners (exactly what training calls, via the same wrapper)
# ---------------------------------------------------------------------------

def pre_fused_with_grads(x, w, scale, base, gy, gpost, gcomb):
    xg = x.detach().requires_grad_()
    wg = w.detach().requires_grad_()
    sg = scale.detach().requires_grad_()
    bg = base.detach().requires_grad_()
    y, post, comb = mhc_pre_sinkhorn_ascend(xg, wg, sg, bg, E, ITERS, EPS, NORM_EPS)
    grads = torch.autograd.grad((y, post, comb), (xg, wg, sg, bg),
                                (gy.to(y.dtype), gpost, gcomb))
    return (y, post, comb), grads


def post_fused_with_grads(y, res, post, comb, gout):
    yg = y.detach().requires_grad_()
    rg = res.detach().requires_grad_()
    pg = post.detach().requires_grad_()
    cg = comb.detach().requires_grad_()
    out = mhc_post_ascend(yg, rg, pg, cg)
    grads = torch.autograd.grad(out, (yg, rg, pg, cg), gout.to(out.dtype))
    return out, grads


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compare(name, op_t, ref_t):
    op, ref = op_t.float().cpu(), ref_t.float().cpu()
    finite = bool(torch.isfinite(op).all())
    md = (op - ref).abs().max().item()
    rel = md / (ref.abs().max().item() + 1e-30)
    bias = (op - ref).mean().item() / (ref.abs().mean().item() + 1e-30)
    flag = ('' if finite and rel < REL_FLAG and abs(bias) < BIAS_FLAG
            else '  <-- FLAG')
    print(f'    {name:10s} maxdiff {md:.3e}  rel {rel:.3e}  bias {bias:+.3e}'
          f'{"" if finite else "  NON-FINITE"}{flag}')
    return finite, rel, bias


# ---------------------------------------------------------------------------
# case matrix
# ---------------------------------------------------------------------------

def make_inputs(S, B, seed, w_mult=1.0, scale_val=None, base_mult=1.0,
                x_mult=1.0, x_tiny=False, res_base_jump=0.0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = (torch.randn(S, B, E, 1024, generator=g) * x_mult).to(torch.bfloat16)
    if x_tiny:
        x = x.float().mul_(1e-4).to(torch.bfloat16)
    w = torch.randn(MIX, E * 1024, generator=g).float() * (0.02 * w_mult)
    if scale_val is None:
        scale = torch.randn(3, generator=g).float() * 0.05
    else:
        scale = torch.full((3,), scale_val, dtype=torch.float32)
    base = torch.randn(MIX, generator=g).float() * (0.1 * base_mult)
    if res_base_jump:
        base[2 * E:] = base[2 * E:].sign() * res_base_jump
    return x.to(DEV), w.to(DEV), scale.to(DEV), base.to(DEV)


def make_grads(S, B, seed):
    g = torch.Generator(device='cpu').manual_seed(seed + 9999)
    gy = torch.randn(S, B, 1024, generator=g).to(torch.bfloat16)
    gpost = torch.randn(S, B, E, generator=g).float()
    gcomb = torch.randn(S, B, E, E, generator=g).float()
    gout = torch.randn(S, B, E, 1024, generator=g).to(torch.bfloat16)
    return (gy.to(DEV), gpost.to(DEV), gcomb.to(DEV), gout.to(DEV))


PRE_CASES = [
    # (label, kwargs) - baseline matches bench_sinkhorn_e2e magnitudes
    ('baseline', {}),
    ('scale1', {'scale_val': 1.0}),
    ('scale5', {'scale_val': 5.0}),
    ('scale20', {'scale_val': 20.0}),
    ('scale100', {'scale_val': 100.0}),
    ('Wx3', {'w_mult': 3.0}),
    ('Wx10', {'w_mult': 10.0}),
    ('Wx30', {'w_mult': 30.0}),
    ('x0.1', {'x_mult': 0.1}),
    ('x10', {'x_mult': 10.0}),
    ('x_tiny', {'x_tiny': True}),
    ('res_base50', {'res_base_jump': 50.0}),
    ('b2_baseline', {}),
    ('b2_scale20', {'scale_val': 20.0}),
]


def run_pre_case(label, S, B, seed, save_failing, **kw):
    print(f'  [{label}] S={S} B={B} seed={seed}')
    x, w, scale, base = make_inputs(S, B, seed, **kw)
    gy, gpost, gcomb, _ = make_grads(S, B, seed)

    (y, post, comb), (dx, dw, dscale, dbase) = pre_fused_with_grads(
        x, w, scale, base, gy, gpost, gcomb)
    print(f'    op output dtypes: y {y.dtype}, post {post.dtype}, comb {comb.dtype}')

    ref_y, ref_pre, ref_post, ref_comb = pre_ref_fp64(x.cpu(), w.cpu(), scale.cpu(), base.cpu())
    rdx, rdw, rdscale, rbase = pre_ref_grads_fp64(
        x.cpu(), w.cpu(), scale.cpu(), base.cpu(),
        gy.to(y.dtype).cpu(), gpost.cpu(), gcomb.cpu())

    ok = True
    ok &= compare('y', y, ref_y)[0]
    ok &= compare('post', post, ref_post)[0]
    ok &= compare('comb', comb, ref_comb)[0]
    ok &= compare('dx', dx, rdx)[0]
    compare('dw', dw, rdw)
    compare('dscale', dscale, rdscale)
    compare('dbase', dbase, rbase)

    # bitwise determinism of a second identical call
    (y2, post2, comb2), (dx2, _, _, _) = pre_fused_with_grads(
        x, w, scale, base, gy, gpost, gcomb)
    det = (torch.equal(y, y2) and torch.equal(post, post2)
           and torch.equal(comb, comb2) and torch.equal(dx, dx2))
    print(f'    deterministic repeat: {"yes" if det else "NO"}')

    if save_failing and not ok:
        path = Path(__file__).parent / 'cases' / f'pre_{label}_s{seed}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save({'x': x.cpu(), 'w': w.cpu(), 'scale': scale.cpu(),
                    'base': base.cpu(), 'gy': gy.cpu(), 'gpost': gpost.cpu(),
                    'gcomb': gcomb.cpu()}, path)
        print(f'    saved failing inputs to {path}')


POST_CASES = [
    ('baseline', {}),
    ('post_sat', {'post_val': 2.0}),
    ('perm_comb', {'perm_comb': True}),
    ('big_res', {'res_mult': 10.0}),
    ('b2', {'B': 2}),
]


def make_post_inputs(S, B, seed, post_val=None, perm_comb=False, res_mult=1.0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    y = torch.randn(S, B, 1024, generator=g).to(torch.bfloat16)
    res = (torch.randn(S, B, E, 1024, generator=g) * res_mult).to(torch.bfloat16)
    if post_val is None:
        post = torch.randn(S, B, E, generator=g).abs().clamp(0, 2).float()
    else:
        post = torch.full((S, B, E), post_val, dtype=torch.float32)
    comb = torch.rand(S, B, E, E, generator=g)
    comb = comb / comb.sum(-1, keepdim=True)  # random row-stochastic
    if perm_comb:
        eye = torch.eye(E).view(1, 1, E, E)
        comb = (1 - 1e-3) * eye + 1e-3 * comb  # near-hard permutation
    return y.to(DEV), res.to(DEV), post.to(DEV), comb.to(DEV)


def run_post_case(label, S, B, seed, **kw):
    print(f'  [{label}] S={S} B={B} seed={seed}')
    y, res, post, comb = make_post_inputs(S, B, seed, **kw)
    _, _, _, gout = make_grads(S, B, seed)
    gout = gout[:S, :B]

    out, (dy, dres, dpost, dcomb) = post_fused_with_grads(y, res, post, comb, gout)
    ref_out = post_ref_fp64(y.cpu(), res.cpu(), post.cpu(), comb.cpu())

    yg = y.double().detach().requires_grad_()
    rg = res.double().detach().requires_grad_()
    pg = post.double().detach().requires_grad_()
    cg = comb.double().detach().requires_grad_()
    ref = post_ref_fp64(yg, rg, pg, cg)
    rdy, rdres, rdpost, rdcomb = torch.autograd.grad(ref, (yg, rg, pg, cg),
                                                     gout.double())

    compare('out', out, ref_out)
    compare('dy', dy, rdy)
    compare('dres', dres, rdres)
    compare('dpost', dpost, rdpost)
    compare('dcomb', dcomb, rdcomb)


def soak(n_iters, S, B, seed):
    """Repeated fused fwd+bwd on fixed inputs: rare NaN / drift / nondeterminism."""
    print(f'  soak: {n_iters} iters, S={S} B={B}')
    x, w, scale, base = make_inputs(S, B, seed)
    gy, gpost, gcomb, _ = make_grads(S, B, seed)
    (y0, post0, comb0), (dx0, _, _, _) = pre_fused_with_grads(
        x, w, scale, base, gy, gpost, gcomb)
    bad = 0
    for i in range(n_iters):
        (y, post, comb), (dx, dw, dscale, dbase) = pre_fused_with_grads(
            x, w, scale, base, gy, gpost, gcomb)
        if not all(torch.isfinite(t).all() for t in (y, post, comb, dx, dw, dscale, dbase)):
            bad += 1
            print(f'    iter {i}: NON-FINITE output')
        if (i + 1) % 500 == 0:
            same = all(torch.equal(a, b) for a, b in zip((y, post, comb, dx),
                                                         (y0, post0, comb0, dx0)))
            print(f'    iter {i + 1}: finite ok, bitwise-vs-first '
                  f'{"identical" if same else "DIFFERENT"}')
    print(f'  soak done: {bad}/{n_iters} non-finite iterations')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--soak', type=int, default=2000)
    ap.add_argument('--soak-seed', type=int, default=0)
    ap.add_argument('--skip-post', action='store_true')
    ap.add_argument('--save-failing', action='store_true')
    args = ap.parse_args()

    torch_npu.npu.set_device(0)
    torch.manual_seed(0)

    print('== pre op (aclnnMhcPreSinkhorn via mhc_pre_sinkhorn_ascend) '
          'vs fp64 reference ==')
    for seed in range(args.seeds):
        print(f'- seed {seed} -')
        for label, kw in PRE_CASES:
            if label.startswith('b2_'):
                run_pre_case(label, 512, 2, seed, args.save_failing, **kw)
            else:
                run_pre_case(label, 4096, 1, seed, args.save_failing, **kw)

    # any backward (the pre backward above qualifies) warms the shared tiling
    # context that aclnnMhcPostBackward's first call can trip on
    if not args.skip_post:
        print('\n== post op (aclnnMhcPost via mhc_post_ascend) '
              'vs fp64 reference ==')
        for seed in range(args.seeds):
            print(f'- seed {seed} -')
            for label, kw in POST_CASES:
                b = kw.pop('B', 1)
                run_post_case(label, 512, b, seed, **kw)

    print('\n== soak ==')
    soak(args.soak, 4096, 1, args.soak_seed)


if __name__ == '__main__':
    main()
