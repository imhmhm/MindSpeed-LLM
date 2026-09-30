# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Stage isolation probes for the standalone Ascend C LitePreHeads op.

Each probe degenerates one input so a single kernel stage is exercised:
  P1 scale=0 base=0   -> sigmoid stage + y stage + layouts only
                         (h_pre == 0.5, h_post == 1.0, coeff == 1/24,
                          y == 0.5 * sum_i x[:, i, :])
  P2 scale=1 base=0   -> adds pass1 (rstd) and the per-lane scale mul
  P3 real scale/base  -> full semantics on 8 tokens (one block)
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault(
    'ASCEND_CUSTOM_OPP_PATH',
    str(Path.home() / 'work/dataset/huashan_zhh_guiyang_turbo/github/cann-ops/build_out'))

import torch  # noqa: E402
import torch_npu  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

torch.manual_seed(11)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

SB, E, H, NP = 8, 4, 1024, 24
N32 = 8 + NP
EPS = 1e-5

TORCH_NPU = Path(torch_npu.__file__).parent
ext = load(
    name='ascendc_lite_pre_ext',
    sources=[str(Path(__file__).parent / 'ascendc_lite_pre/extension.cpp')],
    extra_include_paths=[
        str(TORCH_NPU / 'include'),
        str(TORCH_NPU / 'include/third_party/acl/inc'),
        '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
    ],
    extra_ldflags=[f'-L{TORCH_NPU}/lib', '-ltorch_npu'],
    verbose=False,
)

x = (torch.randn(SB, E, H, device=DEV) * 1.5).to(torch.bfloat16)
xf = x.reshape(SB, E * H)
w = torch.randn(N32, E * H, device=DEV) * 0.02
gamma = 1.0 + torch.randn(E * H, device=DEV) * 0.05
wp = (w.to(torch.bfloat16) * gamma.to(torch.bfloat16).view(1, -1))

scale = torch.tensor([0.011, 0.013, 0.017], device=DEV)
base = (torch.randn(N32, device=DEV) * 0.5).contiguous()


def scale_vec(s0, s1, s2):
    s = torch.empty(N32, device=DEV)
    s[0:E] = s0
    s[E:2 * E] = s1
    s[2 * E:] = s2
    return s


def run(sv, bv, tag):
    logits = torch.matmul(xf, wp.t())
    y, hpre, hpost, coeff, rstd = ext.lite_pre_heads(xf, logits, sv, bv)
    torch.npu.synchronize()
    print(f'--- {tag}')
    print('h_pre [0,:8] :', [f'{v:+.4f}' for v in hpre[0].tolist()])
    print('h_post[0,:8] :', [f'{v:+.4f}' for v in hpost[0].tolist()])
    print('coeff [0]    :', [f'{v:+.4f}' for v in coeff[0].tolist()])
    print('y[0,:4]      :', [f'{v:+.4f}' for v in y[0, :4].tolist()])
    return y, hpre, hpost, coeff, rstd


# P1: zero scale and base -> pure sigmoid(0)/softmax(0) constants
y1, hp1, ho1, c1, rstd1 = run(scale_vec(0, 0, 0), torch.zeros(N32, device=DEV), 'P1 scale=0 base=0')
print('expect h_pre=0.5 h_post=1.0 coeff=1/24=%.4f y=0.5*sum(x)' % (1 / 24))
print('y err vs 0.5*sum(x):', (y1.float() - 0.5 * x.float().sum(1)).abs().max().item())

# P2: unit scale, zero base -> sigmoid(l * rstd)
y2, hp2, ho2, c2, rstd2 = run(scale_vec(1, 1, 1), torch.zeros(N32, device=DEV), 'P2 scale=1 base=0')
rstd_ref = torch.rsqrt(xf.float().pow(2).mean(-1) + EPS)
print('rstd md (P2, all rows)     :', (rstd2 - rstd_ref).abs().max().item())
xf32 = xf.float()
rstd = torch.rsqrt(xf32.pow(2).mean(-1, keepdim=True) + EPS)
logits_r = (xf32 * rstd) @ (wp.float()).t()
pre_l, post_l, res_l = torch.split(logits_r, [E, E, NP], -1)
print('expect h_pre=sigmoid(l)  md:',
      (hp2[:, 0:4].float() - torch.sigmoid(pre_l)).abs().max().item())
print('expect h_post=2sig(l)    md:',
      (ho2[:, 4:8].float() - 2 * torch.sigmoid(post_l)).abs().max().item())
print('expect coeff=softmax(l)  md:',
      (c2[:, 2 * E:].float() - torch.softmax(res_l, -1)).abs().max().item())
print('l[0,:8]:', [f'{v:+.4f}' for v in logits_r[0, :8].tolist()])

# P3: real scalars
y3, hp3, ho3, c3, rstd3 = run(scale_vec(scale[0], scale[1], scale[2]), base, 'P3 real')
perm_flat = torch.eye(E)[torch.tensor(
    list(__import__('itertools').permutations(range(E))))].flatten(1).to(DEV)
print('h_pre md :', (hp3[:, 0:4].float() - torch.sigmoid(pre_l * scale[0] + base[0:4])).abs().max().item())
print('h_post md:', (ho3[:, 4:8].float() - 2 * torch.sigmoid(post_l * scale[1] + base[4:8])).abs().max().item())
print('coeff md :', (c3[:, 2 * E:].float() - torch.softmax(res_l * scale[2] + base[8:], -1)).abs().max().item())
h_pre_r = torch.sigmoid(pre_l * scale[0] + base[0:4])
y_r = torch.sum(h_pre_r.unsqueeze(-1) * x.float().view(SB, E, H), dim=1)
print('y md     :', (y3.float() - y_r).abs().max().item())
