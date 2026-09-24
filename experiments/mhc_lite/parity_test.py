# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""MHCLite parity + autograd test against the 2.1.0 mhc_lite V1 reference math.

Checks, on a single NPU device:
1. bias/scale init pattern (h_pre one-hot stream, h_res ~ identity, h_post ~ 1);
2. forward parity of MHCLite vs pure-fp32 reference (fp32 run) and vs the
   2.1.0-style dataflow (npu_rms_norm + three GEMMs + fp32 softmax, bf16 run);
3. hc_post: Ascend fused mhc_post vs the torch formula, forward and grads;
4. gradients of hc_pre (x / fused weight / gamma / base / scale) vs reference;
5. hc_head pre-only reduction.

Run: python experiments/mhc_lite/parity_test.py
"""

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from megatron.training.global_vars import set_args
from megatron.core.transformer import TransformerConfig

from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.tasks.models.transformer.mhc_lite import MHCLite, MHCLiteSubmodules, _permutation_mats_flat
from mindspeed_llm.ops.npu_mhc import mhc_post_ascend

torch.manual_seed(1234)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 512, 2, 4, 1024
NPERM = math.factorial(E)
LAYER_NUMBER = 5
EPS = 1e-5

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def make_args():
    args = argparse.Namespace()
    args.hc_mult = E
    args.norm_epsilon = EPS
    args.enable_mhc = True
    args.use_triton_mhc = False
    args.use_fused_mhc = False
    args.fp8 = None
    return args


def make_config(params_dtype):
    return TransformerConfig(
        hidden_size=H,
        num_layers=28,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        params_dtype=params_dtype,
    )


def build_module(dtype):
    set_args(make_args())
    module = MHCLite(
        make_config(dtype),
        MHCLiteSubmodules(hc_fn=LinearNoTP),
        mhc_position='attn',
        layer_number=LAYER_NUMBER,
    ).to(DEV)
    perm = _permutation_mats_flat(E).to(DEV)
    return module, perm


def randomize(module):
    with torch.no_grad():
        module.hc_fn.weight.copy_(torch.randn_like(module.hc_fn.weight) * 0.02)
        module.hc_gamma.copy_(1.0 + torch.randn_like(module.hc_gamma) * 0.05)
        module.hc_base.add_(torch.randn_like(module.hc_base) * 0.5)
        module.hc_scale.copy_(torch.tensor([0.011, 0.013, 0.017], dtype=module.hc_scale.dtype))


def split_ref_weights(module):
    """Reference [in, out] weights sliced out of the fused [out, in] weight."""
    w = module.hc_fn.weight.float()
    return w[:E].t(), w[E : 2 * E].t(), w[2 * E :].t()


def ref_forward_fp32(x, module, perm):
    """Pure-fp32 reference of the 2.1.0 MHCLiteEngramV1 math."""
    s, b, e, h = x.shape
    w_pre, w_post, w_res = split_ref_weights(module)
    base = module.hc_base.float()
    scale = module.hc_scale.float()
    xn = x.reshape(s, b, -1).float()
    xn = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + EPS) * module.hc_gamma.float()
    h_pre = torch.sigmoid(scale[0] * (xn @ w_pre) + base[:e])
    h_post = 2 * torch.sigmoid(scale[1] * (xn @ w_post) + base[e : 2 * e])
    coeff = torch.softmax(scale[2] * (xn @ w_res) + base[2 * e :], dim=-1)
    h_res = (coeff @ perm).view(s, b, e, e)
    y = torch.einsum('...e,...eh->...h', h_pre, x.float())
    return y, h_pre, h_post, h_res


def ref_forward_v1_dataflow(x, module, perm):
    """2.1.0-style dataflow: npu_rms_norm + three bf16 GEMMs + fp32 softmax."""
    s, b, e, h = x.shape
    dt = x.dtype
    w = module.hc_fn.weight.type_as(x)
    base = module.hc_base.type_as(x)
    scale = module.hc_scale.type_as(x)
    gamma = module.hc_gamma.type_as(x)
    xn = torch_npu.npu_rms_norm(x.reshape(s, b, -1), gamma, epsilon=EPS)[0]
    logits = xn @ w.t()
    pre_l, post_l, res_l = torch.split(logits, [e, e, NPERM], dim=-1)
    h_pre = torch.sigmoid(pre_l * scale[0] + base[:e])
    h_post = 2 * torch.sigmoid(post_l * scale[1] + base[e : 2 * e])
    coeff = torch.softmax(res_l * scale[2] + base[2 * e :], dim=-1, dtype=torch.float32)
    h_res = torch.matmul(coeff, perm).view(s, b, e, e).type_as(x)
    y = torch.einsum('...e,...eh->...h', h_pre, x)
    return y, h_pre, h_post.type_as(x), h_res


def test_init_pattern():
    module, perm = build_module(torch.float32)
    x = torch.randn(S, B, E, H, device=DEV)
    h_pre, h_post, h_res = module._coefficients(x)
    ref_pre = torch.zeros(E, device=DEV); ref_pre[LAYER_NUMBER % E] = 1.0
    check("init h_pre one-hot", torch.allclose(h_pre.mean(0).mean(0), ref_pre, atol=1e-2),
          detail=f"pre={h_pre.mean(0).mean(0).tolist()}")
    check("init h_post ~1", torch.allclose(h_post.mean(0).mean(0), torch.ones(E, device=DEV), atol=1e-2))
    check("init h_res ~identity", torch.allclose(h_res.mean(0).mean(0), torch.eye(E, device=DEV), atol=1e-2))
    check("perm table shape/fp32", perm.shape == (NPERM, E * E) and perm.dtype == torch.float32)


def test_forward(dtype, atol):
    tag = str(dtype).split('.')[-1]
    module, perm = build_module(dtype)
    randomize(module)
    x = (torch.randn(S, B, E, H, device=DEV) * 1.0).to(dtype).requires_grad_(True)

    y, h_post, h_res = module.hc_pre(x)

    y_ref, h_pre_ref, h_post_ref, h_res_ref = ref_forward_fp32(x.detach(), module, perm)
    check(f"[{tag}] y vs fp32 ref", torch.allclose(y.float(), y_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={max_diff(y, y_ref):.2e}")
    check(f"[{tag}] h_post vs fp32 ref", torch.allclose(h_post.float(), h_post_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={max_diff(h_post, h_post_ref):.2e}")
    check(f"[{tag}] h_res vs fp32 ref", torch.allclose(h_res.float(), h_res_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={max_diff(h_res, h_res_ref):.2e}")

    y_v1, _, h_post_v1, h_res_v1 = ref_forward_v1_dataflow(x.detach(), module, perm)
    check(f"[{tag}] y vs v1 dataflow", torch.allclose(y.float(), y_v1.float(), atol=atol, rtol=atol),
          detail=f"maxdiff={max_diff(y, y_v1):.2e}")
    check(f"[{tag}] h_res vs v1 dataflow", torch.allclose(h_res.float(), h_res_v1.float(), atol=atol, rtol=atol),
          detail=f"maxdiff={max_diff(h_res, h_res_v1):.2e}")

    # doubly stochastic: rows and columns of each h_res sum to ~1
    sums = torch.cat([h_res.float().sum(-1), h_res.float().sum(-2)], dim=-1)
    check(f"[{tag}] h_res doubly stochastic", bool((sums - 1.0).abs().max() < 1e-2),
          detail=f"max|sum-1|={(sums - 1.0).abs().max():.2e}")

    return module, perm, x, (y, h_post, h_res), (y_ref, h_pre_ref, h_post_ref, h_res_ref)


def test_head(dtype, atol):
    tag = str(dtype).split('.')[-1]
    set_args(make_args())
    head = MHCLite(
        make_config(dtype),
        MHCLiteSubmodules(hc_fn=LinearNoTP),
        mhc_position='head',
        layer_number=-1,
    ).to(DEV)
    with torch.no_grad():
        head.hc_fn.weight.copy_(torch.randn_like(head.hc_fn.weight) * 0.02)
        head.hc_gamma.copy_(1.0 + torch.randn_like(head.hc_gamma) * 0.05)
    x = torch.randn(S, B, E, H, device=DEV).to(dtype)
    y = head.hc_head(x)
    s, b, e, h = x.shape
    xn = x.reshape(s, b, -1).float()
    xn = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + EPS) * head.hc_gamma.float()
    pre = torch.sigmoid(xn @ head.hc_fn.weight.float().t() * head.hc_scale.float() + head.hc_base.float())
    y_ref = torch.einsum('...e,...eh->...h', pre, x.float())
    check(f"[{tag}] hc_head vs fp32 ref", torch.allclose(y.float(), y_ref, atol=atol, rtol=atol),
          detail=f"maxdiff={max_diff(y, y_ref):.2e}")


def test_post(dtype, atol):
    tag = str(dtype).split('.')[-1]
    module, perm = build_module(dtype)
    randomize(module)
    with torch.no_grad():
        sub = torch.randn(S, B, H, device=DEV).to(dtype)
        streams = (torch.randn(S, B, E, H, device=DEV) * 1.5).to(dtype)
        _, h_post, h_res = module._coefficients(streams)

    x1 = sub.clone().requires_grad_(True)
    r1 = streams.clone().requires_grad_(True)
    p1 = h_post.clone().requires_grad_(True)
    c1 = h_res.clone().requires_grad_(True)
    out_cann = module.hc_post(x1, residual=r1, post=p1, comb=c1)
    g = torch.randn_like(out_cann)
    out_cann.backward(g)

    x2 = sub.clone().requires_grad_(True)
    r2 = streams.clone().requires_grad_(True)
    p2 = h_post.clone().requires_grad_(True)
    c2 = h_res.clone().requires_grad_(True)
    # fp32 coefficients against upcast streams: matches the operator's internal math
    out_torch = x2.float().unsqueeze(2) * p2.unsqueeze(3) + torch.matmul(c2.transpose(-1, -2), r2.float())
    out_torch.backward(g.float())

    def rel_diff(a, b):
        if a is None or b is None:
            return float('inf')
        b = b.float()
        return ((a.float() - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()

    check(f"[{tag}] mhc_post fwd cann vs torch", rel_diff(out_cann, out_torch) < 1e-2,
          detail=f"reldiff={rel_diff(out_cann, out_torch):.2e}")
    ok = all(rel_diff(a.grad, b.grad) < 3e-2
             for a, b in [(x1, x2), (r1, r2), (p1, p2), (c1, c2)])
    details = [f"{n}: {rel_diff(a.grad, b.grad):.2e}" for a, b, n in
               [(x1, x2, 'dx'), (r1, r2, 'dres'), (p1, p2, 'dpost'), (c1, c2, 'dcomb')]]
    check(f"[{tag}] mhc_post grads cann vs torch", ok, detail=" ".join(details))


def test_grads(dtype, atol):
    tag = str(dtype).split('.')[-1]
    module, perm = build_module(dtype)
    randomize(module)
    x = (torch.randn(S, B, E, H, device=DEV) * 1.0).to(dtype).requires_grad_(True)

    y, h_post, h_res = module.hc_pre(x)
    wy = torch.randn_like(y)
    wp = torch.randn_like(h_post)
    wr = torch.randn_like(h_res)
    loss = (y.float() * wy.float()).sum() + (h_post.float() * wp.float()).sum() + (h_res.float() * wr.float()).sum()
    loss.backward()

    # reference: fp32 math on leaf copies
    xr = x.detach().clone().float().requires_grad_(True)
    w_pre, w_post, w_res = split_ref_weights(module)
    base = module.hc_base.float().detach().requires_grad_(True)
    scale = module.hc_scale.float().detach().requires_grad_(True)
    gamma = module.hc_gamma.float().detach().requires_grad_(True)
    wpre = w_pre.detach().clone().requires_grad_(True)
    wpost = w_post.detach().clone().requires_grad_(True)
    wres = w_res.detach().clone().requires_grad_(True)
    s, b, e, h = xr.shape
    xn = xr.reshape(s, b, -1) * torch.rsqrt(xr.reshape(s, b, -1).pow(2).mean(-1, keepdim=True) + EPS) * gamma
    h_pre = torch.sigmoid(scale[0] * (xn @ wpre) + base[:e])
    hp = 2 * torch.sigmoid(scale[1] * (xn @ wpost) + base[e : 2 * e])
    coeff = torch.softmax(scale[2] * (xn @ wres) + base[2 * e :], dim=-1)
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
        ("dW_post", module.hc_fn.weight.grad[E : 2 * E], wpost.grad.t()),
        ("dW_res", module.hc_fn.weight.grad[2 * E :], wres.grad.t()),
    ]
    for name, a, b in pairs:
        ok = torch.allclose(a.float(), b.float(), atol=gtol, rtol=gtol)
        check(f"[{tag}] grad {name}", ok, detail=f"maxdiff={max_diff(a, b):.2e}")


def main():
    torch.manual_seed(1234)
    print(f"shapes: x=[{S},{B},{E},{H}], layer_number={LAYER_NUMBER}, eps={EPS}")
    test_init_pattern()
    test_forward(torch.float32, atol=1e-3)
    test_grads(torch.float32, atol=1e-3)
    test_head(torch.float32, atol=1e-3)
    test_post(torch.bfloat16, atol=2e-2)
    test_forward(torch.bfloat16, atol=2e-2)
    test_grads(torch.bfloat16, atol=2e-2)
    test_head(torch.bfloat16, atol=2e-2)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
