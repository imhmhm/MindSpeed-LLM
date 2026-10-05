# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Separate the LitePreHeads op cost into the eh-proportional part (pass1
RMS + y mixture, both scaling with sb*h) and the sb-proportional heads part,
by timing the existing binary at several (sb, h) shapes and fitting
    t = a * sb*h + b * sb + c
a = per-element pass1/y cost, b = per-token heads/scalar cost, c = launch."""
import os
import sys
import time
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

E, NL = 4, 32
scale32 = torch.empty(NL, device=DEV)
scale32[:4] = 0.011
scale32[4:8] = 0.013
scale32[8:] = 0.017
base = (torch.randn(NL, device=DEV) * 0.5).contiguous()
eps8 = torch.full((8,), 1e-5, device=DEV)


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


shapes = [(4096, 1024), (4096, 512), (4096, 256), (2048, 1024), (8192, 512)]
rows = []
for sb, h in shapes:
    x = (torch.randn(sb, E * h, device=DEV) * 1.5).to(torch.bfloat16)
    logits = (torch.randn(sb, NL, device=DEV) * 0.5).to(torch.bfloat16)
    t = bench(lambda: ext.lite_pre_heads(x, logits, scale32, base, eps8))
    rows.append((sb, h, t))
    print(f'sb={sb:5d} h={h:5d}  op {t:.3f} ms')

# least squares on t = a*sb*h + b*sb + c
import numpy as np  # noqa: E402

A = np.array([[sb * h, sb, 1.0] for sb, h, _ in rows])
t = np.array([r[2] for r in rows])
coef, *_ = np.linalg.lstsq(A, t, rcond=None)
a, b, c = coef
pred = A @ coef
print(f'\nfit: t = {a * 1e9:.4f} ns/(sb*h elem) + {b * 1e6:.2f} us/token + {c:.3f} ms launch')
print(f'residuals (ms): {[f"{p - q:.4f}" for p, q in zip(pred, t)]}')
sb0, h0 = 4096, 1024
tah = a * sb0 * h0
tb = b * sb0
print(f'at sb=4096 h=1024: pass1+y {tah:.3f} ms ({tah / (tah + tb) * 100:.0f}%), '
      f'heads+scalar {tb:.3f} ms ({tb / (tah + tb) * 100:.0f}%)')
