# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Tier-1 triton path parity: MHCLite with MHC_LITE_TRITON=1 vs the fp32 reference.

Covers the fused heads/y forward kernel, the heads backward kernel (via the
full pre-stage gradient set) and the fused post backward kernel.

Run: MHC_LITE_TRITON=1 python experiments/mhc_lite/triton_parity_test.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('MHC_LITE_TRITON', '1')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

import parity_test as parity  # noqa: E402  (same directory)

DEV = parity.DEV
S, B, E, H, EPS = parity.S, parity.B, parity.E, parity.H, parity.EPS
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_triton_pre(dtype, atol):
    tag = f"triton-{str(dtype).split('.')[-1]}"
    module, perm = parity.build_module(dtype)
    parity.randomize(module)
    assert module.use_triton_pre, 'MHC_LITE_TRITON=1 not picked up'

    x = (torch.randn(S, B, E, H, device=DEV) * 1.0).to(dtype).requires_grad_(True)
    y, h_post, h_res = module.hc_pre(x)

    y_ref, h_pre_ref, h_post_ref, h_res_ref = parity.ref_forward_fp32(x.detach(), module, perm)
    check(f"[{tag}] y vs fp32 ref", torch.allclose(y.float(), y_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={parity.max_diff(y, y_ref):.2e}")
    check(f"[{tag}] h_post vs fp32 ref", torch.allclose(h_post.float(), h_post_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={parity.max_diff(h_post, h_post_ref):.2e}")
    check(f"[{tag}] h_res vs fp32 ref", torch.allclose(h_res.float(), h_res_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={parity.max_diff(h_res, h_res_ref):.2e}")

    wy = torch.randn_like(y)
    wp = torch.randn_like(h_post)
    wr = torch.randn_like(h_res)
    loss = (y.float() * wy.float()).sum() + (h_post.float() * wp.float()).sum() + (h_res.float() * wr.float()).sum()
    loss.backward()

    xr = x.detach().clone().float().requires_grad_(True)
    w_pre, w_post, w_res = parity.split_ref_weights(module)
    base = module.hc_base.float().detach().requires_grad_(True)
    scale = module.hc_scale.float().detach().requires_grad_(True)
    gamma = module.hc_gamma.float().detach().requires_grad_(True)
    wpre = w_pre.detach().clone().requires_grad_(True)
    wpost = w_post.detach().clone().requires_grad_(True)
    wres = w_res.detach().clone().requires_grad_(True)
    s, b, e, h = xr.shape
    xn = xr.reshape(s, b, -1) * torch.rsqrt(xr.reshape(s, b, -1).pow(2).mean(-1, keepdim=True) + EPS) * gamma
    h_pre = torch.sigmoid(scale[0] * (xn @ wpre) + base[:e])
    hp = 2 * torch.sigmoid(scale[1] * (xn @ wpost) + base[e:2 * e])
    coeff = torch.softmax(scale[2] * (xn @ wres) + base[2 * e:], dim=-1)
    hr = (coeff @ perm).view(s, b, e, e)
    yr = torch.einsum('...e,...eh->...h', h_pre, xr)
    ((yr * wy.float()).sum() + (hp * wp.float()).sum() + (hr * wr.float()).sum()).backward()

    gtol = atol * 10
    pairs = [
        ("dx", x.grad, xr.grad),
        ("dgamma", module.hc_gamma.grad, gamma.grad),
        ("dbase", module.hc_base.grad, base.grad),
        ("dscale", module.hc_scale.grad, scale.grad),
        ("dW_pre", module.hc_fn.weight.grad[:E], wpre.grad.t()),
        ("dW_post", module.hc_fn.weight.grad[E:2 * E], wpost.grad.t()),
        ("dW_res", module.hc_fn.weight.grad[2 * E:], wres.grad.t()),
    ]
    for name, a, b in pairs:
        if a is None:
            check(f"[{tag}] grad {name}", False, detail="grad is None")
            continue
        ok = torch.allclose(a.float(), b.float(), atol=gtol, rtol=gtol)
        check(f"[{tag}] grad {name}", ok, detail=f"maxdiff={parity.max_diff(a, b):.2e}")


def test_triton_post(dtype, atol):
    tag = f"triton-{str(dtype).split('.')[-1]}"
    module, perm = parity.build_module(dtype)
    parity.randomize(module)
    with torch.no_grad():
        sub = torch.randn(S, B, H, device=DEV).to(dtype)
        streams = (torch.randn(S, B, E, H, device=DEV) * 1.5).to(dtype)
        _, h_post, h_res = module._coefficients(streams)

    x1 = sub.clone().requires_grad_(True)
    r1 = streams.clone().requires_grad_(True)
    p1 = h_post.clone().requires_grad_(True)
    c1 = h_res.clone().requires_grad_(True)
    out = module.hc_post(x1, residual=r1, post=p1, comb=c1)
    g = torch.randn_like(out)
    out.backward(g)

    x2 = sub.clone().requires_grad_(True)
    r2 = streams.clone().requires_grad_(True)
    p2 = h_post.clone().requires_grad_(True)
    c2 = h_res.clone().requires_grad_(True)
    out2 = x2.float().unsqueeze(2) * p2.unsqueeze(3) + torch.matmul(c2.transpose(-1, -2), r2.float())
    out2.backward(g.float())

    def rel_diff(a, b):
        b = b.float()
        return ((a.float() - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()

    check(f"[{tag}] post fwd cann vs torch", rel_diff(out, out2) < 1e-2, detail=f"reldiff={rel_diff(out, out2):.2e}")
    details = [f"{n}: {rel_diff(a.grad, b.grad):.2e}" for a, b, n in
               [(x1, x2, 'dx'), (r1, r2, 'dres'), (p1, p2, 'dpost'), (c1, c2, 'dcomb')]]
    ok = all(rel_diff(a.grad, b.grad) < 3e-2 for a, b in [(x1, x2), (r1, r2), (p1, p2), (c1, c2)])
    check(f"[{tag}] post grads triton vs torch", ok, detail=" ".join(details))


def main():
    torch.manual_seed(1234)
    print(f"shapes: x=[{S},{B},{E},{H}], MHC_LITE_TRITON={os.environ.get('MHC_LITE_TRITON')}")
    test_triton_pre(torch.float32, atol=1e-3)
    test_triton_post(torch.float32, atol=1e-3)
    test_triton_pre(torch.bfloat16, atol=2e-2)
    test_triton_post(torch.bfloat16, atol=2e-2)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
