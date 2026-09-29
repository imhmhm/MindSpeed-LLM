# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""tilelang-ascend codegen miscompile regression probes for [ROWS, W] tiles.

Four silent-wrong-answer patterns hit while building the fused mhc_lite
2-D kernel (bench_tilelang_lite_pre.py); each probe pairs a BROKEN form with
the WORKING form so a tilelang-ascend fix flips the printed verdict from
BAD to ok. All run on the same synthetic x and an exact fp32 host reference,
so a verdict is unambiguous. Layout notes: ROWS = 8 rows per vector-core
tile, every window 32B-aligned, pass_configs as in the mhc_post V10 recipe.

  P1 column-operand binary op: mul(out, x, colv) with colv [ROWS, 1]
     miscompiles (scattered wrong lanes); broadcast colv to width first.
  P2 axpy element-scalar: axpy(dst, src, buf[0, k]) is exact for row 0 and
     wrong ~3-5% for every other row; broadcast+mul+add instead. (The
     working form also stages x through a [ROWS, 48] read: a [ROWS, 8]
     bf16 buffer has 16B rows, under the 32B alignment the vector core
     requires.)
  P3 window-view operands: mul(a, b[i:j], c[i:j]) with sliced UB buffers
     as operands miscompiles (most rows zeroed/garbage); T.copy each window
     into a dedicated buffer first.
  P4 region-destination accumulate: add(acc[:, i:j], acc[:, i:j], t)
     into a column slice of a wider accumulator corrupts rows >= 1;
     accumulate full-width only.

Requires the source-built tilelang-ascend clone:
  PYTHONPATH=<tilelang-ascend clone> LD_LIBRARY_PATH=<conda env>/lib:$LD_LIBRARY_PATH
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402

import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402

torch.manual_seed(7)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

SB, W = 4096, 1024
ROWS, VEC_NUM = 8, 2
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def p1_column_operand(broken, sb, eh=8 * W, w=W, dtype='bfloat16'):
    """P1: mul with a [ROWS, 1] column operand vs broadcast-then-mul."""
    blk = ROWS * VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor((sb, eh), dtype),
        out: T.Tensor((sb, w), 'float32'),
    ):
        with T.Kernel(T.ceildiv(sb, blk), is_npu=True) as (cid, vid):
            r0 = cid * blk + vid * ROWS
            if r0 < sb:
                with T.Scope('V'):
                    x_ub = T.alloc_ub((ROWS, w), dtype)
                    x32 = T.alloc_ub((ROWS, w), 'float32')
                    colv = T.alloc_ub((ROWS, 1), 'float32')
                    big = T.alloc_ub((ROWS, w), 'float32')
                    T.copy(x[r0 : r0 + ROWS, 0:w], x_ub)
                    T.tile.cast(x32, x_ub, 'CAST_NONE', ROWS * w)
                    T.tile.fill(colv, 2.0)
                    if broken:
                        T.tile.mul(big, x32, colv)
                    else:
                        T.tile.broadcast(big, colv, axis=1)
                        T.tile.mul(big, x32, big)
                    T.copy(big, out[r0 : r0 + ROWS, 0:w])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def p2_axpy_scalar(broken, sb, eh=8 * W, n48=48, dtype='bfloat16'):
    """P2: axpy with an element-scalar read vs broadcast+mul+add.

    The scalar rides a [1, 8] buffer whose row is 32B-aligned; the head
    windows of the fused kernel are exactly this shape.
    """
    blk = ROWS * VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor((sb, eh), dtype),
        scale: T.Tensor((1, 8), 'float32'),
        out: T.Tensor((sb, 8), 'float32'),
    ):
        with T.Kernel(T.ceildiv(sb, blk), is_npu=True) as (cid, vid):
            r0 = cid * blk + vid * ROWS
            if r0 < sb:
                with T.Scope('V'):
                    x_ub = T.alloc_ub((ROWS, n48), dtype)
                    x32 = T.alloc_ub((ROWS, n48), 'float32')
                    x8 = T.alloc_ub((ROWS, 8), 'float32')
                    scale_ub = T.alloc_ub((1, 8), 'float32')
                    s8 = T.alloc_ub((ROWS, 8), 'float32')
                    s8c = T.alloc_ub((ROWS, 8), 'float32')
                    z8 = T.alloc_ub((ROWS, 8), 'float32')
                    T.copy(x[r0 : r0 + ROWS, 0:n48], x_ub)
                    T.tile.cast(x32, x_ub, 'CAST_NONE', ROWS * n48)
                    T.copy(x32[0:ROWS, 0:8], x8)
                    T.copy(scale[0:1, 0:8], scale_ub)
                    if broken:
                        T.tile.axpy(z8, x8, scale_ub[0, 0])
                    else:
                        T.tile.broadcast(s8, scale_ub, axis=0)
                        T.copy(s8, s8c)
                        T.tile.mul(z8, x8, s8c)
                    T.copy(z8, out[r0 : r0 + ROWS, 0:8])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def p3_view_operands(broken, sb, eh=8 * W, n48=48, dtype='bfloat16'):
    """P3: mul over window VIEWS of a [ROWS, 48] buffer vs staged copies."""
    blk = ROWS * VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor((sb, eh), dtype),
        dummy: T.Tensor((1, 8), 'float32'),
        out: T.Tensor((sb, 8), 'float32'),
    ):
        with T.Kernel(T.ceildiv(sb, blk), is_npu=True) as (cid, vid):
            r0 = cid * blk + vid * ROWS
            if r0 < sb:
                with T.Scope('V'):
                    x_bf = T.alloc_ub((ROWS, n48), dtype)
                    x32 = T.alloc_ub((ROWS, n48), 'float32')
                    twice = T.alloc_ub((ROWS, n48), 'float32')
                    z8 = T.alloc_ub((ROWS, 8), 'float32')
                    t8 = T.alloc_ub((ROWS, 8), 'float32')
                    two48 = T.alloc_ub((1, n48), 'float32')
                    T.copy(x[r0 : r0 + ROWS, 0:n48], x_bf)
                    T.tile.cast(x32, x_bf, 'CAST_NONE', ROWS * n48)
                    T.tile.fill(two48, 2.0)
                    T.tile.broadcast(twice, two48, axis=0)
                    if broken:
                        T.tile.mul(z8, x32[0:ROWS, 8:16], twice[0:ROWS, 8:16])
                    else:
                        T.copy(x32[0:ROWS, 8:16], z8)
                        T.copy(twice[0:ROWS, 8:16], t8)
                        T.tile.mul(z8, z8, t8)
                    T.copy(z8, out[r0 : r0 + ROWS, 0:8])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def p4_region_accumulate(broken, sb, eh=8 * W, w=W, dtype='bfloat16'):
    """P4: y-style accumulate into column-slice regions vs full width.

    Broken form: each half-width input chunk accumulates into its own
    [ROWS, w/2] column slice of a wider accumulator, the shape the fused
    kernel's y stage used before running at chunk = w. Working form: the
    same two chunks at full width accumulating into a [ROWS, w] buffer.
    """
    w2 = w // 2
    blk = ROWS * VEC_NUM

    @T.prim_func
    def main(
        x: T.Tensor((sb, eh), dtype),
        out: T.Tensor((sb, w), 'float32'),
    ):
        with T.Kernel(T.ceildiv(sb, blk), is_npu=True) as (cid, vid):
            r0 = cid * blk + vid * ROWS
            if r0 < sb:
                with T.Scope('V'):
                    if broken:
                        x_ub = T.alloc_ub((ROWS, w2), dtype)
                        x32 = T.alloc_ub((ROWS, w2), 'float32')
                        y_acc = T.alloc_ub((ROWS, w), 'float32')
                        T.tile.fill(y_acc, 0.0)
                        for c in T.unroll(2):
                            T.copy(x[r0 : r0 + ROWS, c * w2 : (c + 1) * w2], x_ub)
                            T.tile.cast(x32, x_ub, 'CAST_NONE', ROWS * w2)
                            T.tile.add(y_acc[0:ROWS, c * w2 : (c + 1) * w2],
                                       y_acc[0:ROWS, c * w2 : (c + 1) * w2], x32)
                        T.copy(y_acc, out[r0 : r0 + ROWS, 0:w])
                    else:
                        x_ub = T.alloc_ub((ROWS, w), dtype)
                        x32 = T.alloc_ub((ROWS, w), 'float32')
                        y_acc = T.alloc_ub((ROWS, w), 'float32')
                        T.tile.fill(y_acc, 0.0)
                        for c in T.unroll(2):
                            T.copy(x[r0 : r0 + ROWS, c * w : (c + 1) * w], x_ub)
                            T.tile.cast(x32, x_ub, 'CAST_NONE', ROWS * w)
                            T.tile.add(y_acc, y_acc, x32)
                        T.copy(y_acc, out[r0 : r0 + ROWS, 0:w])

    return main


def verdict(got, ref, tol=1e-2):
    bad = ((got - ref).abs() > tol).float().mean().item()
    return f'{"BAD" if bad > 1e-3 else "ok "} (bad frac {bad:.3f}, maxerr {(got - ref).abs().max().item():.1e})'


def main():
    x = (torch.randn(SB, 8 * W, device=DEV) * 1.5).to(torch.bfloat16)
    xf = x.float()
    scale8 = torch.zeros(1, 8, device=DEV)
    scale8[0, 0] = 0.011
    dummy = torch.zeros(1, 8, device=DEV)

    for broken in (True, False):
        tag = 'broken ' if broken else 'working'
        out = p1_column_operand(broken, SB)(x)
        torch.npu.synchronize()
        print(f'P1 column-operand mul   {tag}: {verdict(out, xf[:, :W] * 2)}')
        out = p2_axpy_scalar(broken, SB)(x, scale8)
        torch.npu.synchronize()
        print(f'P2 axpy element-scalar  {tag}: {verdict(out, xf[:, :8] * scale8)}')
        out = p3_view_operands(broken, SB)(x, dummy)
        torch.npu.synchronize()
        print(f'P3 window-view operands {tag}: {verdict(out, xf[:, 8:16] * 2)}')
        out = p4_region_accumulate(broken, SB)(x)
        torch.npu.synchronize()
        if broken:
            ref = torch.cat([xf[:, :W // 2], xf[:, W // 2:W]], dim=1)
        else:
            ref = xf[:, :W] + xf[:, W:2 * W]
        print(f'P4 region-dst accumulate {tag}: {verdict(out, ref)}')


if __name__ == '__main__':
    main()
