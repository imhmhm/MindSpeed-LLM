# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Mix good/bad value cases through aclnnMhcPostBackward in one process.

Saves the S=512,B=2 seed-1/seed-2 cases to post_backward_case.pt and runs
the bad case, the good case, all pairwise mixes and all leave-one-out
mixes. bad-all is always the process's FIRST backward call, and that
ordering -- not the values -- is what makes it fail: the failure is a
first-call cold tiling init at B>=2, see post_backward_order_probe.py.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch_npu

from megatron.training.global_vars import set_args
from megatron.core.transformer import TransformerConfig

from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.tasks.models.transformer.mhc_lite import MHCLite, MHCLiteSubmodules

torch.manual_seed(1234)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 512, 2, 4, 1024
EPS = 1e-5
LAYER_NUMBER = 5

CACHE = Path(__file__).parent / 'post_backward_case.pt'


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
        layer_number=LAYER_NUMBER,
    ).to(DEV)
    with torch.no_grad():
        module.hc_fn.weight.copy_(torch.randn_like(module.hc_fn.weight) * 0.02)
        module.hc_gamma.copy_(1.0 + torch.randn_like(module.hc_gamma) * 0.05)
        module.hc_base.add_(torch.randn_like(module.hc_base) * 0.5)
        module.hc_scale.copy_(torch.tensor([0.011, 0.013, 0.017], dtype=module.hc_scale.dtype))
    return module


def make_case(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        streams = (torch.randn(S, B, E, H, device=DEV) * 1.5).to(torch.bfloat16)
        _, h_post, h_res = module._coefficients(streams)
        sub = torch.randn(S, B, H, device=DEV).to(torch.bfloat16)
        g = torch.randn(S, B, E, H, device=DEV).to(torch.bfloat16)
    return sub, streams, h_post, h_res, g


module = build_module()
good = make_case(module, 2)
bad = make_case(module, 1)
torch.save({'good': [t.cpu() for t in good], 'bad': [t.cpu() for t in bad]}, CACHE)

import cann_ops_transformer

names = ['sub', 'streams', 'post', 'comb', 'g']


def run(tag, case):
    sub, streams, post, comb, g = [t.clone().to(DEV).requires_grad_(True) for t in case]
    try:
        out = cann_ops_transformer.ops.mhc_post(
            streams.permute(1, 0, 2, 3).contiguous(), comb.permute(1, 0, 2, 3).contiguous(),
            sub.permute(1, 0, 2).contiguous(), post.permute(1, 0, 2).contiguous())
        out.backward(g.permute(1, 0, 2, 3))
        torch.npu.synchronize()
        ok = all(t.grad is not None for t in (sub, streams, post, comb))
        print(f"[{tag}] {'OK' if ok else 'MISSING GRADS'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] RAISE: {str(exc).splitlines()[0][:90]}")


run('bad-all', bad)
run('good-all', good)
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        if j <= i:
            continue
        mix = list(good)
        mix[i], mix[j] = bad[i], bad[j]
        run(f'good+bad-{ni}+{nj}', mix)
print('-- leave-one-out from bad --')
for j, nj in enumerate(names):
    mix = list(bad)
    mix[j] = good[j]
    run(f'bad+good-{nj}', mix)
