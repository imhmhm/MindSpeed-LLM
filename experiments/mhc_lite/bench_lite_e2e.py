# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""mhc_lite optimization campaign baseline: current tiers vs CANN mhc_sinkhorn.

One --variant per process (env toggles are read at module init and at some
call sites, so variants must not share a process). Every run measures the
COMPLETE module chain at the same shape/dtype discipline,

  pre  : x [s,b,e,h] -> (y, h_post, h_res)
  post : out = h_post (x) y + h_res^T @ x        (residual = input streams)

and prints fwd / fwd+bwd for the pre stage and the e2e chain, plus accuracy
gates: outputs and all five gradients vs an fp32 torch reference with its own
autograd (bf16 tolerances from parity_test.py). fast-but-wrong variants fail
the gate regardless of their timing.

Variants:
  full-cann       aclnnMhcPreSinkhorn + aclnnMhcPost (mainline fused path,
                  the bar to beat; full-MHC semantics, checked against its
                  own torch reference)
  lite-t0         MHCLite torch chain (CANN rms/GEMM/post fwd, torch bwd)
  lite-t2         MHC_LITE_TRITON=1 (triton pre Function + K3 post bwd)
  lite-t2-native  + MHC_LITE_NATIVE_POST_BWD=1 (aclnn native post backward;
                  warmed up first by one B=1 random-gradient call, the shape/
                  grad pattern that passes the first-call cold tiling init)
  lite-t2-direct  + MHC_LITE_POST_DIRECT=1 (scheme D: aclnn post called on
                  native layouts, output returned as a view, no wrapper clone)
  lite-t3         TRITON + NATIVE_POST_BWD + POST_DIRECT (A+D stacked)
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402

S, B, E, H = 4096, 1, 4, 1024
ITERS, EPS, NORM_EPS = 20, 1e-6, 1e-5  # full-MHC sinkhorn constants

torch.manual_seed(2)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


# ---------------------------------------------------------------------------
# full-MHC (mhc_sinkhorn) chains: CANN native ops and their torch reference
# ---------------------------------------------------------------------------

def full_pre_torch(x, w, scale, base, s, b):
    shape = x.size()
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, w) * rsqrt
    mixes = mixes.permute(1, 0, 2)

    pre = torch.sigmoid(mixes[..., :E] * scale[0] + base[:E]) + EPS
    post = 2 * torch.sigmoid(mixes[..., E:2 * E] * scale[1] + base[E:2 * E])
    comb = (mixes[..., 2 * E:].view(b, s, E, E) * scale[2]
            + base[2 * E:].view(1, 1, E, E))
    comb = comb.softmax(-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)

    y = torch.sum(pre.permute(1, 0, 2).unsqueeze(-1) * xf.view(shape), dim=2).to(x.dtype)
    return y, post.permute(1, 0, 2), comb.permute(1, 0, 2, 3)


def full_post_torch(y, res, post, comb):
    return (post.unsqueeze(-1) * y.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)).type_as(y)


# ---------------------------------------------------------------------------
# fp32 torch reference of the mhc_lite semantics (mirrors mhc_lite.py)
# ---------------------------------------------------------------------------

def perm_mats_flat(n):
    import itertools
    perms = list(itertools.permutations(range(n)))
    idx = torch.tensor(perms, dtype=torch.int64)
    return torch.eye(n, dtype=torch.float32)[idx].flatten(1).to(DEV)


def lite_ref(x, w, gamma, scale, base, perm_flat, eps):
    s, b, e, h = x.shape
    xf = x.reshape(s, b, e * h).float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()
    logits = torch.matmul(xn, w.float().t())
    pre_l, post_l, res_l = torch.split(logits, [e, e, perm_flat.shape[0]], -1)
    h_pre = torch.sigmoid(pre_l * scale[0] + base[:e])
    h_post = 2 * torch.sigmoid(post_l * scale[1] + base[e:2 * e])
    p = torch.softmax(res_l * scale[2] + base[2 * e:], -1)
    h_res = torch.matmul(p, perm_flat).view(s, b, e, e)
    y = (h_pre.unsqueeze(-1) * x.reshape(s, b, e, h).float()).sum(2)
    out = h_post.unsqueeze(-1) * y.unsqueeze(2) + torch.matmul(
        h_res.transpose(-1, -2), x.reshape(s, b, e, h).float())
    return y, h_post, h_res, out


# ---------------------------------------------------------------------------
# module construction (probe-style: meaningful non-saturated coefficients)
# ---------------------------------------------------------------------------

def build_lite():
    from megatron.training.global_vars import set_args
    from megatron.core.transformer import TransformerConfig

    from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
    from mindspeed_llm.tasks.models.transformer.mhc_lite import MHCLite, MHCLiteSubmodules

    args = argparse.Namespace()
    args.hc_mult = E
    args.norm_epsilon = 1e-5
    args.enable_mhc = True
    args.use_triton_mhc = False
    args.use_fused_mhc = False
    args.fp8 = None
    set_args(args)

    module = MHCLite(
        TransformerConfig(hidden_size=H, num_layers=28, num_attention_heads=16,
                          ffn_hidden_size=4096, params_dtype=torch.bfloat16),
        MHCLiteSubmodules(hc_fn=LinearNoTP),
        mhc_position='attn',
        layer_number=5,
    ).to(DEV)
    with torch.no_grad():
        module.hc_fn.weight.copy_(torch.randn_like(module.hc_fn.weight) * 0.02)
        module.hc_gamma.copy_(1.0 + torch.randn_like(module.hc_gamma) * 0.05)
        module.hc_base.add_(torch.randn_like(module.hc_base) * 0.5)
        module.hc_scale.copy_(torch.tensor([0.011, 0.013, 0.017],
                                           dtype=module.hc_scale.dtype))
    return module


def make_inputs(s, b, dtype):
    x = (torch.randn(s, b, E, H, device=DEV) * 1.5).to(dtype)
    gy = torch.randn(s, b, H, device=DEV).to(dtype)
    gpost = torch.randn(s, b, E, device=DEV, dtype=torch.float32)
    gres = torch.randn(s, b, E, E, device=DEV, dtype=torch.float32)
    gout = torch.randn(s, b, E, H, device=DEV).to(dtype)
    return x, gy, gpost, gres, gout


def warm_native_post(dtype):
    """The aclnnMhcPost backward tiling context needs one successful backward
    first (first-call cold-init failure, see post_backward_order_probe.py).
    B=1 with a RANDOM gradient is the combination that passes the first call
    and then warms every later shape/pattern."""
    from mindspeed_llm.ops.npu_mhc import mhc_post_ascend

    y = torch.randn(512, 1, H, device=DEV).to(dtype).requires_grad_()
    res = torch.randn(512, 1, E, H, device=DEV).to(dtype).requires_grad_()
    post = torch.rand(512, 1, E, device=DEV).requires_grad_()
    comb = torch.rand(512, 1, E, E, device=DEV).requires_grad_()
    out = mhc_post_ascend(y, res, post, comb)
    out.backward(torch.randn_like(out))
    torch.npu.synchronize()


class _EnvCleared:
    """Temporarily drop the lite env toggles (module init and _MhcPostFn read
    them), so a Tier-0 torch reference module can coexist with a variant."""

    KEYS = ('MHC_LITE_TRITON', 'MHC_LITE_NATIVE_POST_BWD', 'MHC_LITE_POST_DIRECT',
            'MHC_LITE_TORCH_POST')

    def __enter__(self):
        self.saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is not None:
                os.environ[k] = v


def _module_grads(module, x, gy, gpost, gres, gout):
    """Forward+backward through the module with Parameter-rebound leaves."""
    w, gamma, scale, base = (module.hc_fn.weight, module.hc_gamma,
                             module.hc_scale, module.hc_base)
    xm = x.detach().requires_grad_()
    wm = torch.nn.Parameter(w.detach(), requires_grad=True)
    gm = torch.nn.Parameter(gamma.detach(), requires_grad=True)
    sm = torch.nn.Parameter(scale.detach(), requires_grad=True)
    bm = torch.nn.Parameter(base.detach(), requires_grad=True)
    module.hc_fn.weight = wm
    module.hc_gamma = gm
    module.hc_scale = sm
    module.hc_base = bm
    y, post, res = module.hc_pre(xm)
    out = module.hc_post(y, residual=xm, post=post, comb=res)
    grads = torch.autograd.grad((y, post, res, out), (xm, wm, gm, sm, bm),
                                (gy, gpost, gres, gout))
    module.hc_fn.weight = w
    module.hc_gamma = gamma
    module.hc_scale = scale
    module.hc_base = base
    return (y, post, res, out), grads


def check(module, refmod, s, b, tag, gates):
    """Two-tier accuracy gate at one shape.

    tier1: variant vs the Tier-0 torch module on the same bf16 inputs -- both
           share the forward graph, so structural scheme errors (wrong axis,
           layout, missing term) show up at O(1) relative, far above the
           last-bit rounding floor.
    tier2: variant vs an fp32 torch reference -- bf16 forward noise level;
           scalar grads sum over all tokens, hence relative comparison.
    """
    x, gy, gpost, gres, gout = make_inputs(s, b, torch.bfloat16)
    perm_flat = perm_mats_flat(E)
    w, gamma, scale, base = (module.hc_fn.weight, module.hc_gamma,
                             module.hc_scale, module.hc_base)

    xr = x.detach().float().requires_grad_()
    wr = w.detach().float().requires_grad_()
    gr = gamma.detach().float().requires_grad_()
    sr = scale.detach().float().requires_grad_()
    br = base.detach().float().requires_grad_()
    y_r, post_r, res_r, out_r = lite_ref(xr, wr, gr, sr, br, perm_flat, 1e-5)
    refs = torch.autograd.grad((y_r, post_r, res_r, out_r), (xr, wr, gr, sr, br),
                               (gy.float(), gpost, gres, gout.float()))

    outs, mods = _module_grads(module, x, gy, gpost, gres, gout)
    y, post, res, out = outs

    names = ('y', 'h_post', 'h_res', 'out')
    gnames = ('dx', 'dW', 'dgamma', 'dscale', 'dbase')
    ref_outs = (y_r, post_r, res_r, out_r)
    fwd = [f'{n} {(m.float() - r.detach()).abs().max().item():.1e}'
           for n, m, r in zip(names, outs, ref_outs)]
    bwd = [f'{n} {(m.float() - r).abs().max().item():.1e}({r.abs().max().item():.1e})'
           for n, m, r in zip(gnames, mods, refs)]
    rel = [(m.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
           for m, r in zip(mods, refs)]
    rel_f = max((m.float() - r.detach()).abs().max().item()
                / max(r.detach().abs().max().item(), 1e-6)
                for m, r in zip(outs, ref_outs))

    tier1 = 0.0
    if refmod is not None and refmod is not module:
        with _EnvCleared():
            outs0, mods0 = _module_grads(refmod, x, gy, gpost, gres, gout)
        tier1 = max([((m.float() - m0.float()).abs().max().item()
                      / max(m0.float().abs().max().item(), 1e-6))
                     for m, m0 in zip(list(mods) + list(outs), list(mods0) + list(outs0))])

    ok = rel_f <= 2e-2 and max(rel) <= 5e-2 and tier1 <= 2e-2
    gates.append(ok)
    print(f'  [{tag} s={s} b={b}] fwd {" ".join(fwd)}')
    print(f'  [{" " * len(tag)} s={s} b={b}] bwd(abs|ref|) {" ".join(bwd)}')
    t1s = 'n/a' if refmod is None else f'{tier1:.1e}'
    print(f'  [{" " * len(tag)} s={s} b={b}] tier2 relw {max(rel):.1e} relf {rel_f:.1e} '
          f'tier1-vs-t0 {t1s} -> {"PASS" if ok else "FAIL"}', flush=True)


def run_lite(variant, s, b):
    module = build_lite()
    # Tier-0 twin (torch paths) with identical weights for the tier-1 gate
    with _EnvCleared():
        refmod = build_lite()
    with torch.no_grad():
        refmod.hc_fn.weight.copy_(module.hc_fn.weight)
        refmod.hc_gamma.copy_(module.hc_gamma)
        refmod.hc_scale.copy_(module.hc_scale)
        refmod.hc_base.copy_(module.hc_base)
    if variant == 'lite-t0':
        refmod = None  # the variant IS the tier-0 reference

    gates = []
    check(module, refmod, 512, 2, 'acc', gates)
    check(module, refmod, s, b, 'acc', gates)

    x, gy, gpost, gres, gout = make_inputs(s, b, torch.bfloat16)
    w = module.hc_fn.weight
    gamma, scale, base = module.hc_gamma, module.hc_scale, module.hc_base

    def pre_fwd():
        with torch.no_grad():
            module.hc_pre(x)

    def pre_fwdbwd():
        xm = x.detach().requires_grad_()
        wm = torch.nn.Parameter(w.detach(), requires_grad=True)
        module.hc_fn.weight = wm
        y, post, res = module.hc_pre(xm)
        torch.autograd.grad((y, post, res), (xm, wm), (gy, gpost, gres))
        module.hc_fn.weight = w

    def e2e_fwd():
        with torch.no_grad():
            y, post, res = module.hc_pre(x)
            module.hc_post(y, residual=x, post=post, comb=res)

    def e2e_fwdbwd():
        xm = x.detach().requires_grad_()
        wm = torch.nn.Parameter(w.detach(), requires_grad=True)
        module.hc_fn.weight = wm
        y, post, res = module.hc_pre(xm)
        out = module.hc_post(y, residual=xm, post=post, comb=res)
        torch.autograd.grad(out, (xm, wm), gout)
        module.hc_fn.weight = w

    print(f'[{variant}] S={s} B={b} bf16  gates {"PASS" if all(gates) else "FAIL"}')
    print(f'  pre  fwd {bench(pre_fwd):7.3f} ms   pre  fwd+bwd {bench(pre_fwdbwd, 15):7.3f} ms')
    print(f'  e2e  fwd {bench(e2e_fwd):7.3f} ms   e2e  fwd+bwd {bench(e2e_fwdbwd, 15):7.3f} ms',
          flush=True)


def run_full(s, b):
    from mindspeed_llm.ops.npu_mhc import mhc_post_ascend, mhc_pre_sinkhorn_ascend

    mix = (2 + E) * E
    x, gy, gpost, gres, gout = make_inputs(s, b, torch.bfloat16)
    w = torch.randn(mix, E * H, device=DEV, dtype=torch.float32) * 0.02
    scale = torch.randn(3, device=DEV, dtype=torch.float32) * 0.05
    base = torch.randn(mix, device=DEV, dtype=torch.float32) * 0.1

    def full_pre(x, w, scale, base):
        return mhc_pre_sinkhorn_ascend(x, w, scale, base, E, ITERS, EPS, NORM_EPS)

    def chain(x, w, scale, base):
        y, post, comb = full_pre(x, w, scale, base)
        return mhc_post_ascend(y, x, post, comb)

    with torch.no_grad():
        y_r, post_r, comb_r = full_pre_torch(x, w, scale, base, s, b)
        out_r = full_post_torch(y_r, x, post_r, comb_r).float()
        y, post, comb = full_pre(x, w, scale, base)
        out = chain(x, w, scale, base)
    fwd = [(m.float() - r).abs().max().item()
           for m, r in zip((y, post, comb, out), (y_r, post_r, comb_r, out_r))]

    xg = x.detach().requires_grad_()
    wg = w.detach().requires_grad_()
    sg = scale.detach().requires_grad_()
    bg = base.detach().requires_grad_()
    mods = torch.autograd.grad(chain(xg, wg, sg, bg), (xg, wg, sg, bg), gout)

    xr = x.detach().float().requires_grad_()
    y_r2, post_r2, comb_r2 = full_pre_torch(xr, wg, sg, bg, s, b)
    out_r2 = full_post_torch(y_r2, xr, post_r2, comb_r2)
    refs = torch.autograd.grad(out_r2, (xr, wg, sg, bg), gout.float())

    bwd = [(m.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
           for m, r in zip(mods, refs)]
    print(f'[full-cann] S={s} B={b} bf16 (full-MHC semantics, own reference)')
    print(f'  fwd maxdiff y/post/comb/out: '
          f'{" ".join(f"{v:.1e}" for v in fwd)}')
    print(f'  bwd rel diff dx/dW/dscale/dbase: '
          f'{" ".join(f"{v:.1e}" for v in bwd)} (informational bar, not a gate)')

    def pre_fwdbwd():
        torch.autograd.grad(full_pre(xg, wg, sg, bg), (xg, wg, sg, bg), (gy, gpost, gres))

    print(f'  pre  fwd {bench(lambda: full_pre(x, w, scale, base)):7.3f} ms   '
          f'pre  fwd+bwd {bench(pre_fwdbwd, 15):7.3f} ms')
    print(f'  e2e  fwd {bench(lambda: chain(x, w, scale, base)):7.3f} ms   '
          f'e2e  fwd+bwd {bench(lambda: torch.autograd.grad(chain(xg, wg, sg, bg), (xg, wg, sg, bg), gout), 15):7.3f} ms',
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True,
                        choices=['full-cann', 'lite-t0', 'lite-t2', 'lite-t2-native',
                                 'lite-t2-direct', 'lite-t3'])
    parser.add_argument('--shape', default=f'{S}x{B}', help='sxb')
    args = parser.parse_args()

    s, b = (int(v) for v in args.shape.split('x'))
    if args.variant == 'full-cann':
        run_full(s, b)
        return
    if args.variant.startswith('lite-t2') or args.variant == 'lite-t3':
        os.environ['MHC_LITE_TRITON'] = '1'
    if args.variant in ('lite-t2-native', 'lite-t3'):
        os.environ['MHC_LITE_NATIVE_POST_BWD'] = '1'
    if args.variant in ('lite-t2-direct', 'lite-t3'):
        os.environ['MHC_LITE_POST_DIRECT'] = '1'
    if args.variant in ('lite-t2-native', 'lite-t3'):
        warm_native_post(torch.bfloat16)
    run_lite(args.variant, s, b)


if __name__ == '__main__':
    main()
