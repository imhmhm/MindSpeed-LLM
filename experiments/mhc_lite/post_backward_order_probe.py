# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""One fresh-process probe of aclnnMhcPostBackward call-order hypotheses.

post_backward_repro.py's "seed 1 fails 100%" always ran the bad case as the
process's FIRST mhc_post backward, and every later mixed case passed; the
seed sweep's first call also raised and its repeat passed. That points at a
cold first-call tiling failure (InitTilingParseCtx), not an input value
domain. Scenarios (one process each, driven by --order):

  bad            seed 1 backward only (the repro's failing first call)
  good,bad       warm with seed 2, then the seed 1 values
  bad,bad        seed 1 twice (does the second call survive the first?)
  seed0,seed0    sweep's first-call failure, repeated
  good,good      control
  fwdN           N forward-only calls (no backward) before the next tag
                 (bench_post_native_bwd.py ran 35 forwards before its first
                 backward and never failed -- does the forward warm it?)
  prewarm        one full mhc_pre_sinkhorn fwd+bwd first (the aclnn op that
                 mainline training runs besides mhc_post; cross-op warm-up)
  prefwd         mhc_pre_sinkhorn forward only first (training's exact state
                 before the first mhc_post backward)
  bad@4096x1     first backward at the training/bench shape instead of 512x2
                 (seed@sxb in general)
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402

from megatron.training.global_vars import set_args
from megatron.core.transformer import TransformerConfig

from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.tasks.models.transformer.mhc_lite import MHCLite, MHCLiteSubmodules

torch.manual_seed(1234)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 512, 2, 4, 1024

import cann_ops_transformer  # noqa: E402


def make_args():
    args = argparse.Namespace()
    args.hc_mult = E
    args.norm_epsilon = 1e-5
    args.enable_mhc = True
    args.use_triton_mhc = False
    args.use_fused_mhc = False
    args.fp8 = None
    return args


def build_module():
    set_args(make_args())
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
        module.hc_scale.copy_(torch.tensor([0.011, 0.013, 0.017], dtype=module.hc_scale.dtype))
    return module


def make_case(module, seed, s=S, b=B):
    torch.manual_seed(seed)
    with torch.no_grad():
        streams = (torch.randn(s, b, E, H, device=DEV) * 1.5).to(torch.bfloat16)
        _, h_post, h_res = module._coefficients(streams)
        sub = torch.randn(s, b, H, device=DEV).to(torch.bfloat16)
        g = torch.randn(s, b, E, H, device=DEV).to(torch.bfloat16)
    return sub, streams, h_post, h_res, g


def probe(module, seed, s=S, b=B):
    try:
        sub, streams, post, comb, g = [t.clone().requires_grad_(True) for t in make_case(module, seed, s, b)]
        out = cann_ops_transformer.ops.mhc_post(
            streams.permute(1, 0, 2, 3).contiguous(), comb.permute(1, 0, 2, 3).contiguous(),
            sub.permute(1, 0, 2).contiguous(), post.permute(1, 0, 2).contiguous())
        out.backward(g.permute(1, 0, 2, 3))
        torch.npu.synchronize()
        ins = (sub, streams, post, comb)
        ok = all(t.grad is not None and torch.isfinite(t.grad.float()).all() for t in ins)
        return 'OK' if ok else 'GRAD-BAD'
    except Exception as exc:  # noqa: BLE001
        return 'RAISE'


def probe_prewarm(module):
    """One full mhc_pre_sinkhorn fwd+bwd (the op mainline runs besides mhc_post)."""
    from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend

    try:
        x = (torch.randn(S, B, E, H, device=DEV) * 1.5).to(torch.bfloat16).requires_grad_()
        w = (torch.randn((2 + E) * E, E * H, device=DEV) * 0.02).requires_grad_()
        scale = (torch.randn(3, device=DEV) * 0.05).requires_grad_()
        base = (torch.randn((2 + E) * E, device=DEV) * 0.1).requires_grad_()
        out = mhc_pre_sinkhorn_ascend(x, w, scale, base, E, 20, 1e-6, 1e-5)
        torch.autograd.grad(out[0], (x, w, scale, base),
                            torch.randn_like(out[0]))
        torch.npu.synchronize()
        return 'WARM'
    except Exception as exc:  # noqa: BLE001
        return f'PRE-RAISE'


def probe_prefwd(module):
    """mhc_pre_sinkhorn forward only -- training's state before the first
    mhc_post backward (backward order runs post before pre within a layer)."""
    from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend

    try:
        with torch.no_grad():
            x = (torch.randn(S, B, E, H, device=DEV) * 1.5).to(torch.bfloat16)
            w = (torch.randn((2 + E) * E, E * H, device=DEV) * 0.02)
            scale = (torch.randn(3, device=DEV) * 0.05)
            base = (torch.randn((2 + E) * E, device=DEV) * 0.1)
            mhc_pre_sinkhorn_ascend(x, w, scale, base, E, 20, 1e-6, 1e-5)
            torch.npu.synchronize()
        return 'WARM'
    except Exception as exc:  # noqa: BLE001
        return f'PRE-RAISE'


def probe_fwd_only(module, seed, calls):
    """Forward-only warm-up: `calls` forward passes with no backward."""
    try:
        with torch.no_grad():
            sub, streams, post, comb, g = make_case(module, seed)
            args = (streams.permute(1, 0, 2, 3).contiguous(), comb.permute(1, 0, 2, 3).contiguous(),
                    sub.permute(1, 0, 2).contiguous(), post.permute(1, 0, 2).contiguous())
            for _ in range(calls):
                cann_ops_transformer.ops.mhc_post(*args)
            torch.npu.synchronize()
        return 'WARM'
    except Exception as exc:  # noqa: BLE001
        return 'FWD-RAISE'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', required=True,
                        help='comma list of: good|bad|seed0|fwdN|prewarm|prefwd|'
                             'seed@sxb (e.g. bad@4096x1)')
    args = parser.parse_args()

    module = build_module()
    seeds = {'good': 2, 'bad': 1, 'seed0': 0}
    results = []
    for tag in args.order.split(','):
        if tag.startswith('fwd'):
            results.append(probe_fwd_only(module, seeds['good'], int(tag[3:])))
        elif tag == 'prewarm':
            results.append(probe_prewarm(module))
        elif tag == 'prefwd':
            results.append(probe_prefwd(module))
        elif '@' in tag:
            name, shape = tag.split('@')
            s, b = (int(v) for v in shape.split('x'))
            results.append(probe(module, seeds[name], s, b))
        else:
            results.append(probe(module, seeds[tag]))
    print(f"{args.order}: {','.join(results)}", flush=True)


if __name__ == '__main__':
    main()
