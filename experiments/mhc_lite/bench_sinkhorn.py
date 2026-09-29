# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Sinkhorn (full-MHC res head) implementations, head-to-head at S=4096 fp32.

Same computation scope for every variant: logits [S,16] fp32 -> softmax(s*l+b)
+ eps -> initial col norm -> 19 x (row norm, col norm) -> out [S,16]
(exact torch_hc_split_sinkhorn res-head semantics, deepseek4/mhc.py).

  eager      mainline torch loop (the mhc.py fallback form)
  aot_eager  dynamo FX graph, aten ops
  inductor   npu inductor -> triton codegen
  torchair   FX -> CANN GE graph
  triton     single fused kernel, 19 iterations in registers
  aclnn      aclnnMhcPreSinkhorn whole-op forward for context: the sinkhorn
             part is fused inside (norm+GEMM+3 heads+sinkhorn+y, bf16 in)
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch_npu  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from mindspeed_llm.ops.npu_mhc import mhc_pre_sinkhorn_ascend  # noqa: E402
from mindspeed_llm.ops.triton.mhc_lite_heads import _launch  # noqa: E402

torch.manual_seed(0)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

BS, E, ITERS, EPS = 4096, 4, 20, 1e-6

logits = torch.randn(BS, 16, device=DEV, dtype=torch.float32)
s2 = torch.tensor(0.017, device=DEV, dtype=torch.float32)
base = torch.randn(16, device=DEV, dtype=torch.float32) * 0.5


def sinkhorn_eager(logits, s2, base):
    m = (logits.view(BS, E, E) * s2 + base.view(1, E, E)).softmax(-1) + EPS
    m = m / (m.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        m = m / (m.sum(-1, keepdim=True) + EPS)
        m = m / (m.sum(-2, keepdim=True) + EPS)
    return m.reshape(BS, 16)


@triton.jit
def sinkhorn_head_kernel(
    logits_ptr, out_ptr, s2_ptr, base_ptr, bs,
    GROUP: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s2 = tl.load(s2_ptr)
    ar4 = tl.arange(0, 4)

    # four row slices m0..m3 stay in registers; row/col sums are slice-local
    m0 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m1 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m2 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m3 = tl.zeros((GROUP, 4), dtype=tl.float32)
    row_off = rows[:, None] * 16 + ar4[None, :]
    for r in tl.static_range(4):
        l = tl.load(logits_ptr + row_off + r * 4, mask=mask[:, None], other=0.0)
        z = l * s2 + tl.load(base_ptr + r * 4 + ar4)[None, :]
        zmax = tl.max(z, axis=1)
        e = tl.exp(z - zmax[:, None])
        s = e / tl.sum(e, axis=1)[:, None] + EPS
        if r == 0:
            m0 = s
        elif r == 1:
            m1 = s
        elif r == 2:
            m2 = s
        else:
            m3 = s
    colsum = m0 + m1 + m2 + m3
    m0 = m0 / (colsum + EPS)
    m1 = m1 / (colsum + EPS)
    m2 = m2 / (colsum + EPS)
    m3 = m3 / (colsum + EPS)
    for _ in range(ITERS - 1):
        m0 = m0 / (tl.sum(m0, axis=1)[:, None] + EPS)
        m1 = m1 / (tl.sum(m1, axis=1)[:, None] + EPS)
        m2 = m2 / (tl.sum(m2, axis=1)[:, None] + EPS)
        m3 = m3 / (tl.sum(m3, axis=1)[:, None] + EPS)
        colsum = m0 + m1 + m2 + m3
        m0 = m0 / (colsum + EPS)
        m1 = m1 / (colsum + EPS)
        m2 = m2 / (colsum + EPS)
        m3 = m3 / (colsum + EPS)
    tl.store(out_ptr + rows[:, None] * 16 + ar4[None, :], m0, mask=mask[:, None])
    tl.store(out_ptr + rows[:, None] * 16 + 4 + ar4[None, :], m1, mask=mask[:, None])
    tl.store(out_ptr + rows[:, None] * 16 + 8 + ar4[None, :], m2, mask=mask[:, None])
    tl.store(out_ptr + rows[:, None] * 16 + 12 + ar4[None, :], m3, mask=mask[:, None])


def sinkhorn_triton(logits, s2, base):
    out = torch.empty(BS, 16, device=DEV, dtype=torch.float32)
    _launch(sinkhorn_head_kernel, ('sk_head', BS), (triton.cdiv(BS, 32),),
            (logits, out, s2, base, BS), {'GROUP': 32, 'ITERS': ITERS, 'EPS': EPS})
    return out


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


m64 = (logits.double().view(BS, E, E) * s2.double() + base.double().view(1, E, E)).softmax(-1) + EPS
m64 = m64 / (m64.sum(-2, keepdim=True) + EPS)
for _ in range(ITERS - 1):
    m64 = m64 / (m64.sum(-1, keepdim=True) + EPS)
    m64 = m64 / (m64.sum(-2, keepdim=True) + EPS)
ref64 = m64.reshape(BS, 16).float()

print(f'== sinkhorn res head, S={BS} fp32, {ITERS} normalizations (1 init + {ITERS - 1} loops) ==')

variants = [
    ('eager torch loop', sinkhorn_eager),
    ('aot_eager', torch.compile(sinkhorn_eager, backend='aot_eager')),
    ('inductor', torch.compile(sinkhorn_eager, backend='inductor', dynamic=False)),
]

for name, fn in variants:
    try:
        out = fn(logits, s2, base)
        err = (out - ref64).abs().max().item()
        t = bench(lambda: fn(logits, s2, base))
        print(f'{name:24s}: {t:7.3f} ms   maxdiff {err:.1e}')
    except Exception as exc:  # noqa: BLE001
        print(f'{name:24s}: FAIL {repr(exc)[:110]}')

import torch_npu.dynamo.torchair as torchair  # noqa: E402

cfg = torchair.CompilerConfig()
try:
    ge_fn = torch.compile(sinkhorn_eager, backend=torchair.get_npu_backend(compiler_config=cfg))
    out = ge_fn(logits, s2, base)
    err = (out - ref64).abs().max().item()
    t = bench(lambda: ge_fn(logits, s2, base))
    print(f'{"torchair (CANN graph)":24s}: {t:7.3f} ms   maxdiff {err:.1e}')
except Exception as exc:  # noqa: BLE001
    print(f'{"torchair (CANN graph)":24s}: FAIL {repr(exc)[:110]}')

out = sinkhorn_triton(logits, s2, base)
torch.npu.synchronize()
err = (out - ref64).abs().max().item()
t = bench(lambda: sinkhorn_triton(logits, s2, base))
print(f'{"triton fused kernel":24s}: {t:7.3f} ms   maxdiff {err:.1e}')

print()
print('== context: aclnnMhcPreSinkhorn whole-op fwd (norm+GEMM+heads+sinkhorn+y, bf16) ==')
S, B, H = BS, 1, 1024
x = torch.randn(S, B, E, H, device=DEV, dtype=torch.bfloat16)
phi = (torch.randn(24, E * H, device=DEV, dtype=torch.float32) * 0.02)
alpha = torch.zeros(3, device=DEV, dtype=torch.float32)
bias = torch.zeros(24, device=DEV, dtype=torch.float32)
t = bench(lambda: mhc_pre_sinkhorn_ascend(x, phi, alpha, bias, E, 20, 1e-3, 1e-5), iters=30)
print(f'{"aclnn fused pre fwd":24s}: {t:7.3f} ms   (sinkhorn amortized inside)')

# ---------------------------------------------------------------------------
# tilelang variant: needs the source-built tilelang-ascend (see README); the
# pypi 0.1.4 wheel requires glibc 2.38 and cannot load on this host.
# ---------------------------------------------------------------------------
try:
    import tilelang  # noqa: E402
    from tilelang import language as TL  # noqa: E402

    TL_OK = True
except Exception as exc:  # noqa: BLE001
    TL_OK = False
    print(f'\ntilelang unavailable ({repr(exc)[:80]}), skipping tilelang variant')

if TL_OK:
    VEC = 2  # dual vector cores per block, the tilelang-ascend mhc_post pattern
    SUB = 256

    tl_pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }

    # fp32 tile rows must be 32B-aligned (8 lanes); the 4 live lanes stay in
    # columns 0-3 and pad lanes are kept at exact zeros: loads use pad_value=0,
    # s2/base are zero-padded on the host, and a -1e30 mask before exp makes the
    # pad lanes exp() to 0 so row/col sums only see live lanes.
    @tilelang.jit(out_idx=[5], pass_configs=tl_pass_configs)
    def sk_tilelang_kernel(sub=SUB, vec=VEC):
        n = TL.symbolic('n')
        block = sub * vec

        @TL.prim_func
        def main(
            logits: TL.Tensor((n, 16), 'float32'),
            s2: TL.Tensor((8,), 'float32'),
            base: TL.Tensor((4, 8), 'float32'),
            epv: TL.Tensor((8,), 'float32'),
            mask: TL.Tensor((8,), 'float32'),
            out: TL.Tensor((n, 16), 'float32'),
        ):
            with TL.Kernel(TL.ceildiv(n, block), is_npu=True) as (cid, vid):
                row0 = cid * block + vid * sub
                l0 = TL.alloc_ub((sub, 8), 'float32')
                l1 = TL.alloc_ub((sub, 8), 'float32')
                l2 = TL.alloc_ub((sub, 8), 'float32')
                l3 = TL.alloc_ub((sub, 8), 'float32')
                m0 = TL.alloc_ub((sub, 8), 'float32')
                m1 = TL.alloc_ub((sub, 8), 'float32')
                m2 = TL.alloc_ub((sub, 8), 'float32')
                m3 = TL.alloc_ub((sub, 8), 'float32')
                t = TL.alloc_ub((sub, 8), 'float32')
                bb = TL.alloc_ub((sub, 8), 'float32')
                dnm = TL.alloc_ub((sub, 8), 'float32')
                cs = TL.alloc_ub((sub, 8), 'float32')
                s2f = TL.alloc_ub((sub, 8), 'float32')
                epvf = TL.alloc_ub((sub, 8), 'float32')
                maskf = TL.alloc_ub((sub, 8), 'float32')
                ones8 = TL.alloc_ub((sub, 8), 'float32')
                b_r = TL.alloc_ub((8,), 'float32')
                s2r = TL.alloc_ub((1, 8), 'float32')
                zm = TL.alloc_ub((sub, 1), 'float32')
                zmb = TL.alloc_ub((sub, 8), 'float32')
                es = TL.alloc_ub((sub, 1), 'float32')
                rs = TL.alloc_ub((sub, 1), 'float32')
                ones1 = TL.alloc_ub((sub, 1), 'float32')

                TL.copy(s2[0:8], b_r)
                TL.tile.broadcast(s2r, b_r, axis=0)
                TL.tile.broadcast(s2f, s2r, axis=0)
                TL.copy(epv[0:8], b_r)
                TL.tile.broadcast(epvf, b_r, axis=0)
                TL.copy(mask[0:8], b_r)
                TL.tile.broadcast(maskf, b_r, axis=0)
                TL.tile.fill(ones8, 1.0)
                TL.tile.fill(ones1, 1.0)

                TL.copy(logits[row0 : row0 + sub, 0 : 4], l0[0:sub, 0:4], pad_value=0.0)
                TL.copy(base[0, 0:8], b_r)
                TL.tile.broadcast(bb, b_r, axis=0)
                TL.tile.mul(t, l0, s2f)
                TL.tile.axpy(t, bb, 1.0)
                TL.tile.axpy(t, maskf, 1.0)
                TL.reduce_max(t, zm, dim=1, clear=True)
                TL.tile.broadcast(zmb, zm, axis=1)
                TL.tile.sub(t, t, zmb)
                TL.tile.exp(t, t)
                TL.reduce_sum(t, es, dim=1, clear=True)
                TL.tile.broadcast(dnm, es, axis=1)
                TL.tile.div(m0, t, dnm)
                TL.tile.axpy(m0, epvf, 1.0)
                TL.copy(logits[row0 : row0 + sub, 4 : 8], l1[0:sub, 0:4], pad_value=0.0)
                TL.copy(base[1, 0:8], b_r)
                TL.tile.broadcast(bb, b_r, axis=0)
                TL.tile.mul(t, l1, s2f)
                TL.tile.axpy(t, bb, 1.0)
                TL.tile.axpy(t, maskf, 1.0)
                TL.reduce_max(t, zm, dim=1, clear=True)
                TL.tile.broadcast(zmb, zm, axis=1)
                TL.tile.sub(t, t, zmb)
                TL.tile.exp(t, t)
                TL.reduce_sum(t, es, dim=1, clear=True)
                TL.tile.broadcast(dnm, es, axis=1)
                TL.tile.div(m1, t, dnm)
                TL.tile.axpy(m1, epvf, 1.0)
                TL.copy(logits[row0 : row0 + sub, 8 : 12], l2[0:sub, 0:4], pad_value=0.0)
                TL.copy(base[2, 0:8], b_r)
                TL.tile.broadcast(bb, b_r, axis=0)
                TL.tile.mul(t, l2, s2f)
                TL.tile.axpy(t, bb, 1.0)
                TL.tile.axpy(t, maskf, 1.0)
                TL.reduce_max(t, zm, dim=1, clear=True)
                TL.tile.broadcast(zmb, zm, axis=1)
                TL.tile.sub(t, t, zmb)
                TL.tile.exp(t, t)
                TL.reduce_sum(t, es, dim=1, clear=True)
                TL.tile.broadcast(dnm, es, axis=1)
                TL.tile.div(m2, t, dnm)
                TL.tile.axpy(m2, epvf, 1.0)
                TL.copy(logits[row0 : row0 + sub, 12 : 16], l3[0:sub, 0:4], pad_value=0.0)
                TL.copy(base[3, 0:8], b_r)
                TL.tile.broadcast(bb, b_r, axis=0)
                TL.tile.mul(t, l3, s2f)
                TL.tile.axpy(t, bb, 1.0)
                TL.tile.axpy(t, maskf, 1.0)
                TL.reduce_max(t, zm, dim=1, clear=True)
                TL.tile.broadcast(zmb, zm, axis=1)
                TL.tile.sub(t, t, zmb)
                TL.tile.exp(t, t)
                TL.reduce_sum(t, es, dim=1, clear=True)
                TL.tile.broadcast(dnm, es, axis=1)
                TL.tile.div(m3, t, dnm)
                TL.tile.axpy(m3, epvf, 1.0)

                TL.tile.add(cs, m0, m1)
                TL.tile.add(cs, cs, m2)
                TL.tile.add(cs, cs, m3)
                TL.tile.axpy(cs, ones8, EPS)
                TL.tile.div(m0, m0, cs)
                TL.tile.div(m1, m1, cs)
                TL.tile.div(m2, m2, cs)
                TL.tile.div(m3, m3, cs)

                for _ in range(ITERS - 1):
                    TL.reduce_sum(m0, rs, dim=1, clear=True)
                    TL.tile.axpy(rs, ones1, EPS)
                    TL.tile.broadcast(dnm, rs, axis=1)
                    TL.tile.div(m0, m0, dnm)
                    TL.reduce_sum(m1, rs, dim=1, clear=True)
                    TL.tile.axpy(rs, ones1, EPS)
                    TL.tile.broadcast(dnm, rs, axis=1)
                    TL.tile.div(m1, m1, dnm)
                    TL.reduce_sum(m2, rs, dim=1, clear=True)
                    TL.tile.axpy(rs, ones1, EPS)
                    TL.tile.broadcast(dnm, rs, axis=1)
                    TL.tile.div(m2, m2, dnm)
                    TL.reduce_sum(m3, rs, dim=1, clear=True)
                    TL.tile.axpy(rs, ones1, EPS)
                    TL.tile.broadcast(dnm, rs, axis=1)
                    TL.tile.div(m3, m3, dnm)
                    TL.tile.add(cs, m0, m1)
                    TL.tile.add(cs, cs, m2)
                    TL.tile.add(cs, cs, m3)
                    TL.tile.axpy(cs, ones8, EPS)
                    TL.tile.div(m0, m0, cs)
                    TL.tile.div(m1, m1, cs)
                    TL.tile.div(m2, m2, cs)
                    TL.tile.div(m3, m3, cs)

                TL.copy(m0[0:sub, 0:4], out[row0 : row0 + sub, 0 : 4])
                TL.copy(m1[0:sub, 0:4], out[row0 : row0 + sub, 4 : 8])
                TL.copy(m2[0:sub, 0:4], out[row0 : row0 + sub, 8 : 12])
                TL.copy(m3[0:sub, 0:4], out[row0 : row0 + sub, 12 : 16])

        return main

    print()
    print(f'== tilelang variant (source-built, sub={SUB} x vec={VEC}) ==')
    try:
        assert BS % (SUB * VEC) == 0
        s2p = torch.zeros(8, device=DEV, dtype=torch.float32)
        s2p[:4] = s2
        basep = torch.zeros(4, 8, device=DEV, dtype=torch.float32)
        basep[:, :4] = base.view(4, 4)
        epv8 = torch.zeros(8, device=DEV, dtype=torch.float32)
        epv8[:4] = EPS
        mask8 = torch.zeros(8, device=DEV, dtype=torch.float32)
        mask8[4:] = -1e30
        kern = sk_tilelang_kernel()
        args = (logits, s2p, basep, epv8, mask8)
        out = kern(*args)
        err = (out - ref64).abs().max().item()
        t = bench(lambda: kern(*args))
        print(f'{"tilelang UB kernel":24s}: {t:7.3f} ms   maxdiff {err:.1e}')
    except Exception as exc:  # noqa: BLE001
        print(f'{"tilelang UB kernel":24s}: FAIL {repr(exc)[:150]}')
