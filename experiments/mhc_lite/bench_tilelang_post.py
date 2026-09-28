# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Measure the tilelang-ascend mhc_post example at mhc_lite shapes.

Runs examples/mhc_post/example_mhc_post.py from the reference clone (V10 pure
Vector kernel) against the eager bmm baseline and the aclnn fused op at the
same shapes/dtypes, so the three post-side implementations are directly
comparable. Note the tilelang repo's own "vs CANN" numbers use the eager
torch baseline, not the aclnn op.

Requires the tilelang-ascend 0.1.1.10 (ubuntu20.4/cann900) wheel; the pypi
0.1.4 wheel needs glibc 2.38. Run with the conda libstdc++ on LD_LIBRARY_PATH.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/ma-user/work/dataset/huashan_zhh_guiyang_turbo/"
                   "github/_ref_mhc/tilelang-ascend/examples/mhc_post")

import torch  # noqa: E402
import torch_npu  # noqa: E402

import tilelang  # noqa: E402
import example_mhc_post as tl_ex  # noqa: E402

torch.manual_seed(42)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

import cann_ops_transformer  # noqa: E402

ops = cann_ops_transformer.ops

print(f'tilelang version: {tilelang.__version__}')


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


for n, h in [(4096, 1024), (4096, 2560)]:
    hc = 4
    print(f'\n== mhc_post at n={n}, h={h}, hc={hc}, bf16 ==')
    data = tl_ex.generate_test_data(n, h, hc)

    out_tl = tl_ex.mhc_post(**data)
    ref = tl_ex.mhc_post_ref(**data)
    err = (out_tl.float() - ref.float()).abs().max().item()

    t_tl = bench(lambda: tl_ex.mhc_post(**data))
    print(f'tilelang V10 kernel        : {t_tl:7.3f} ms   maxdiff {err:.1e}')

    t_pt = bench(lambda: tl_ex.mhc_post_pytorch_baseline(**data))
    print(f'eager bmm baseline         : {t_pt:7.3f} ms')

    # aclnn fused op, native [b, s, ...] layout (b=1 slices are views)
    x_op = data['x'].reshape(1, n, h).contiguous()
    res_op = data['residual'].reshape(1, n, hc, h).contiguous()
    post_op = data['post_layer_mix'].reshape(1, n, hc).contiguous()
    comb_op = data['comb_res_mix'].reshape(1, n, hc, hc).contiguous()
    out_op = ops.mhc_post(res_op, comb_op, x_op, post_op)
    err_op = (out_op.reshape(n, hc, h).float() - ref.float()).abs().max().item()
    t_op = bench(lambda: ops.mhc_post(res_op, comb_op, x_op, post_op))
    print(f'aclnn mhc_post raw         : {t_op:7.3f} ms   maxdiff {err_op:.1e}')
