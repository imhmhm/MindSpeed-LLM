# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Localize the LitePreFused wrongness: compare the op's saved l (= C*rstd,
dumped before scale/base) and rstd against torch, row by row, to tell a
tile-split mismatch (blocks of wrong rows) from a C-layout mismatch
(transposed / fractal / wrong lanes)."""
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

torch.manual_seed(3)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')
E, NP = 4, 24
N32 = 32
EPS = 1e-5

TORCH_NPU = Path(torch_npu.__file__).parent
ext = load(
    name='ascendc_lite_pre_fused_ext',
    sources=[str(Path(__file__).parent / 'ascendc_lite_pre_fused/extension.cpp')],
    extra_include_paths=[
        str(TORCH_NPU / 'include'),
        str(TORCH_NPU / 'include/third_party/acl/inc'),
        '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
    ],
    extra_ldflags=[f'-L{TORCH_NPU}/lib', '-ltorch_npu'],
    verbose=False,
)

sb, h = 4096, 1024
x = (torch.randn(sb, E, h, device=DEV) * 1.5).to(torch.bfloat16)
xf = x.reshape(sb, E * h)
w = (torch.randn(N32, E * h, device=DEV) * 0.02).to(torch.bfloat16)
gamma = torch.ones(E * h, device=DEV)
scale32 = torch.full((N32,), 0.01, device=DEV)
base = torch.zeros(N32, device=DEV)

y, hp, ho, co, rstd, lg = ext.lite_pre_fused(xf, w, scale32, base)
torch.npu.synchronize()

xf32 = xf.float()
rstd_t = torch.rsqrt(xf32.pow(2).mean(-1, keepdim=True) + EPS)
logits_t = (xf32 @ w.float().t())
l_t = (logits_t * rstd_t)

print('rstd md  :', (rstd.float() - rstd_t.squeeze(-1)).abs().max().item())

# recover the unscaled C the kernel saw: saved l / its own (verified-correct) rstd
c_rec = lg.float() / rstd.float().clamp_min(1e-6).unsqueeze(-1)
cand = {
    'intended  x@Wt': logits_t,
    'b-inverted x@W.reshape(K,32)': (xf32 @ w.float().reshape(-1, 32)) if w.numel() == xf32.shape[1] * 32 else None,
}
for name, ref in cand.items():
    if ref is None:
        continue
    dd = (c_rec - ref).abs().max(-1).values
    print(f'{name}: exact rows {int((dd < 0.05).sum())}/{sb}  best row md {dd.min().item():.3e}')
    j = int(dd.argmin())
    print(f'   closest row {j}: op {c_rec[j, :4].tolist()} ref {ref[j, :4].tolist()}')
# tile-transposed hypothesis on the first 128-row chunk: lanes<->rows swapped
t = c_rec[:128].reshape(32, 128).t()
dd = (t - logits_t[:128]).abs().max(-1).values
print(f'tile-transposed[128x32]: exact rows {int((dd < 0.05).sum())}/128  best md {dd.min().item():.3e}')
