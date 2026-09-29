# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Seed sweep probing whether aclnnMhcPostBackward failures depend on values.

post_backward_repro.py pinned one failing value combination (seed=1, S=512,
B=2) and seed=2/3 passing, which read as a value-domain defect. This script
sweeps N seeds through the same make_case path to measure how often random
values trip it.

Methodology caveat (see post_backward_order_probe.py for the resolution):
the failure is actually a first-call-of-process tiling cold-init, so in this
single-process sweep only the very first call can fail; after its async
error the context is poisoned (the control case returns GRAD-BAD), which
makes the sweep's own results untrustworthy. Kept as the record of why
in-process sweeps cannot answer this question.
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

E, H = 4, 1024
EPS = 1e-5

import cann_ops_transformer  # noqa: E402


def make_args():
    args = argparse.Namespace()
    args.hc_mult = E
    args.norm_epsilon = EPS
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


def make_case(module, seed, s, b):
    torch.manual_seed(seed)
    with torch.no_grad():
        streams = (torch.randn(s, b, E, H, device=DEV) * 1.5).to(torch.bfloat16)
        _, h_post, h_res = module._coefficients(streams)
        sub = torch.randn(s, b, H, device=DEV).to(torch.bfloat16)
        g = torch.randn(s, b, E, H, device=DEV).to(torch.bfloat16)
    return sub, streams, h_post, h_res, g


def run_backward(case):
    """True when backward completes with finite grads on all four inputs."""
    sub, streams, post, comb, g = [t.clone().requires_grad_(True) for t in case]
    out = cann_ops_transformer.ops.mhc_post(
        streams.permute(1, 0, 2, 3).contiguous(), comb.permute(1, 0, 2, 3).contiguous(),
        sub.permute(1, 0, 2).contiguous(), post.permute(1, 0, 2).contiguous())
    out.backward(g.permute(1, 0, 2, 3))
    torch.npu.synchronize()
    ins = (sub, streams, post, comb)
    return all(t.grad is not None and torch.isfinite(t.grad.float()).all() for t in ins)


def probe(case):
    try:
        return 'OK' if run_backward(case) else 'GRAD-BAD'
    except Exception as exc:  # noqa: BLE001
        return f'RAISE {str(exc).splitlines()[0][:60]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    module = build_module()
    control = make_case(module, 2, 512, 2)

    failures = []
    for seed in range(args.start, args.start + args.seeds):
        status = probe(make_case(module, seed, 512, 2))
        if status == 'OK':
            continue
        # an async device error can poison the context: verify the control
        # case still passes before trusting this seed's result
        ctrl = probe(control)
        again = probe(make_case(module, seed, 512, 2))
        print(f'seed {seed}: {status} | control-after {ctrl} | repeat {again}', flush=True)
        if ctrl == 'OK' and again != 'OK':
            failures.append(seed)

    print(f'\n{len(failures)}/{args.seeds} seeds fail (S=512 B=2): {failures}')

    # do the failing value combinations stay bad at the training shape?
    for seed in failures[:5]:
        for (s, b) in [(4096, 1), (512, 1)]:
            print(f'seed {seed} @ s={s} b={b}: {probe(make_case(module, seed, s, b))}', flush=True)


if __name__ == '__main__':
    main()
