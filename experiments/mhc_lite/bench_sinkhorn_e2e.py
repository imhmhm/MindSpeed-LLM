# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""End-to-end speed of complete mhc_sinkhorn implementations (full-MHC semantics).

The experiment object is the WHOLE mhc_sinkhorn computation of one MHC module,
exactly the deepseek4/mhc.py semantics (no mhc_lite in the matrix):

  pre  : x [s,b,4,h] bf16 -> fp32 RMS rsqrt -> logits GEMM (mix 24) ->
         h_pre = sigmoid(l*s0+b0)+eps, h_post = 2*sigmoid(l*s1+b1),
         h_res = softmax -> init col norm -> 19x(row norm, col norm) sinkhorn
         -> y = sum_i h_pre_i * x_i (bf16)
  post : out = h_post (x) y + h_res^T @ residual streams

Variants (fwd and fwd+bwd at S=4096, B=1, hc=4, h=1024, bf16 in / fp32 params):
  pre   aclnn fused (mainline fused path, wrapper layout adapt)
        torch fallback (exact mhc.py non-fused branch)
        fallback + triton sinkhorn head (custom fwd/bwd kernels; per-stage
        m checkpoints in GM let the backward replay the 40 normalizations)
  post  aclnn fused (mainline wrapper) / torch broadcast (fallback form) /
        torch bmm / torchair-compiled bmm
  e2e   mainline fused / mainline fallback / fallback+triton head+bmm post /
        fused pre + bmm post / fused pre + torchair post
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

from mindspeed_llm.ops.npu_mhc import mhc_post_ascend, mhc_pre_sinkhorn_ascend  # noqa: E402
from mindspeed_llm.ops.triton.mhc_lite_heads import _launch  # noqa: E402

# seed=2 keeps the value-dependent aclnnMhcPostBackward on its known-good side
# (post_backward_repro.py); the timing is representative of the mainline path.
torch.manual_seed(2)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

S, B, E, H = 4096, 1, 4, 1024
MIX = (2 + E) * E
ITERS, NSTAGES = 20, 40  # 1 softmax + 1 init col norm + 19x(row, col) checkpoints
EPS, NORM_EPS = 1e-6, 1e-5
BS = S * B

X = torch.randn(S, B, E, H, device=DEV, dtype=torch.bfloat16)
W = (torch.randn(MIX, E * H, device=DEV, dtype=torch.float32) * 0.02)
SCALE = (torch.randn(3, device=DEV, dtype=torch.float32) * 0.05)
BASE = (torch.randn(MIX, device=DEV, dtype=torch.float32) * 0.1)


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


# ---------------------------------------------------------------------------
# pre-side implementations
# ---------------------------------------------------------------------------

def pre_torch(x, w, scale, base):
    """Exact mhc.py hc_pre non-fused branch: fp32 math, eager sinkhorn loop."""
    shape = x.size()
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, w) * rsqrt        # [s,b,24] fp32
    mixes = mixes.permute(1, 0, 2)                           # [b,s,24]

    pre = torch.sigmoid(mixes[..., :E] * scale[0] + base[:E]) + EPS
    post = 2 * torch.sigmoid(mixes[..., E:2 * E] * scale[1] + base[E:2 * E])
    comb = (mixes[..., 2 * E:].view(B, S, E, E) * scale[2]
            + base[2 * E:].view(1, 1, E, E))
    comb = comb.softmax(-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)

    pre = pre.permute(1, 0, 2)
    post = post.permute(1, 0, 2)
    comb = comb.permute(1, 0, 2, 3)
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2).to(x.dtype)
    return y, post, comb


def pre_fused(x, w, scale, base):
    """Mainline fused path: aclnnMhcPreSinkhorn through the [s,b]<->[b,s] wrapper."""
    return mhc_pre_sinkhorn_ascend(x, w, scale, base, E, ITERS, EPS, NORM_EPS)


@triton.jit
def sk_fwd_ckpt_kernel(
    logits_ptr, out_ptr, stages_ptr, s2_ptr, base_ptr, bs,
    GROUP: tl.constexpr, ITERS: tl.constexpr, NSTAGES: tl.constexpr, EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s2 = tl.load(s2_ptr)
    ar4 = tl.arange(0, 4)
    roff = rows[:, None] * 16 + ar4[None, :]
    msk = mask[:, None]

    # four row slices stay in registers; every normalization stage is also
    # written to the checkpoint tensor for the backward replay
    m0 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m1 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m2 = tl.zeros((GROUP, 4), dtype=tl.float32)
    m3 = tl.zeros((GROUP, 4), dtype=tl.float32)
    for r in tl.static_range(4):
        l = tl.load(logits_ptr + roff + r * 4, mask=msk, other=0.0)
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
    st = stages_ptr + roff
    tl.store(st + 0 * bs * 16, m0, mask=msk)
    tl.store(st + 0 * bs * 16 + 4, m1, mask=msk)
    tl.store(st + 0 * bs * 16 + 8, m2, mask=msk)
    tl.store(st + 0 * bs * 16 + 12, m3, mask=msk)

    colsum = m0 + m1 + m2 + m3
    m0 = m0 / (colsum + EPS)
    m1 = m1 / (colsum + EPS)
    m2 = m2 / (colsum + EPS)
    m3 = m3 / (colsum + EPS)
    tl.store(st + 1 * bs * 16, m0, mask=msk)
    tl.store(st + 1 * bs * 16 + 4, m1, mask=msk)
    tl.store(st + 1 * bs * 16 + 8, m2, mask=msk)
    tl.store(st + 1 * bs * 16 + 12, m3, mask=msk)

    # runtime loop (like the backward kernel): the fully unrolled store chain
    # makes the ascend backend compile pathologically slow
    for it in range(1, ITERS):
        m0 = m0 / (tl.sum(m0, axis=1)[:, None] + EPS)
        m1 = m1 / (tl.sum(m1, axis=1)[:, None] + EPS)
        m2 = m2 / (tl.sum(m2, axis=1)[:, None] + EPS)
        m3 = m3 / (tl.sum(m3, axis=1)[:, None] + EPS)
        tl.store(st + (2 * it) * bs * 16, m0, mask=msk)
        tl.store(st + (2 * it) * bs * 16 + 4, m1, mask=msk)
        tl.store(st + (2 * it) * bs * 16 + 8, m2, mask=msk)
        tl.store(st + (2 * it) * bs * 16 + 12, m3, mask=msk)
        colsum = m0 + m1 + m2 + m3
        m0 = m0 / (colsum + EPS)
        m1 = m1 / (colsum + EPS)
        m2 = m2 / (colsum + EPS)
        m3 = m3 / (colsum + EPS)
        tl.store(st + (2 * it + 1) * bs * 16, m0, mask=msk)
        tl.store(st + (2 * it + 1) * bs * 16 + 4, m1, mask=msk)
        tl.store(st + (2 * it + 1) * bs * 16 + 8, m2, mask=msk)
        tl.store(st + (2 * it + 1) * bs * 16 + 12, m3, mask=msk)

    tl.store(out_ptr + roff, m0, mask=msk)
    tl.store(out_ptr + roff + 4, m1, mask=msk)
    tl.store(out_ptr + roff + 8, m2, mask=msk)
    tl.store(out_ptr + roff + 12, m3, mask=msk)


@triton.jit
def sk_bwd_kernel(
    g_ptr, logits_ptr, stages_ptr, s2_ptr, base_ptr, dlogits_ptr, gz_ptr, bs,
    GROUP: tl.constexpr, ITERS: tl.constexpr, NSTAGES: tl.constexpr, EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUP + tl.arange(0, GROUP)
    mask = rows < bs
    s2 = tl.load(s2_ptr)
    ar4 = tl.arange(0, 4)
    roff = rows[:, None] * 16 + ar4[None, :]
    msk = mask[:, None]

    g0 = tl.load(g_ptr + roff, mask=msk, other=0.0)
    g1 = tl.load(g_ptr + roff + 4, mask=msk, other=0.0)
    g2 = tl.load(g_ptr + roff + 8, mask=msk, other=0.0)
    g3 = tl.load(g_ptr + roff + 12, mask=msk, other=0.0)

    # each forward stage is m_out = m_in / (sum over an axis + EPS); its vjp
    # is g <- (g - <g, m_out>_axis) / (S + EPS) where the dot keeps the
    # normalized axis as a vector (per matrix row for row steps, per column
    # for col steps). The stages are replayed in reverse; odd stages
    # normalize by col sum, even by row sum, strictly alternating, so the
    # walk runs as (col, row) pairs. A runtime loop keeps the kernel body
    # small: the fully unrolled form makes the ascend backend compile
    # pathologically slow.
    for k in range(0, ITERS - 1):
        j = ITERS - 1 - k
        c_off = (2 * j + 1) * bs * 16 + roff
        r_off = (2 * j) * bs * 16 + roff
        p_off = (2 * j - 1) * bs * 16 + roff
        # col vjp: output stage 2j+1, input stage 2j
        mo0 = tl.load(stages_ptr + c_off, mask=msk, other=0.0)
        mo1 = tl.load(stages_ptr + c_off + 4, mask=msk, other=0.0)
        mo2 = tl.load(stages_ptr + c_off + 8, mask=msk, other=0.0)
        mo3 = tl.load(stages_ptr + c_off + 12, mask=msk, other=0.0)
        mi0 = tl.load(stages_ptr + r_off, mask=msk, other=0.0)
        mi1 = tl.load(stages_ptr + r_off + 4, mask=msk, other=0.0)
        mi2 = tl.load(stages_ptr + r_off + 8, mask=msk, other=0.0)
        mi3 = tl.load(stages_ptr + r_off + 12, mask=msk, other=0.0)
        cv = g0 * mo0 + g1 * mo1 + g2 * mo2 + g3 * mo3
        cs = mi0 + mi1 + mi2 + mi3 + EPS
        g0 = (g0 - cv) / cs
        g1 = (g1 - cv) / cs
        g2 = (g2 - cv) / cs
        g3 = (g3 - cv) / cs
        # row vjp: output stage 2j, input stage 2j-1
        d0 = tl.sum(g0 * mi0, 1)
        d1 = tl.sum(g1 * mi1, 1)
        d2 = tl.sum(g2 * mi2, 1)
        d3 = tl.sum(g3 * mi3, 1)
        pi0 = tl.load(stages_ptr + p_off, mask=msk, other=0.0)
        pi1 = tl.load(stages_ptr + p_off + 4, mask=msk, other=0.0)
        pi2 = tl.load(stages_ptr + p_off + 8, mask=msk, other=0.0)
        pi3 = tl.load(stages_ptr + p_off + 12, mask=msk, other=0.0)
        r0 = tl.sum(pi0, 1) + EPS
        r1 = tl.sum(pi1, 1) + EPS
        r2 = tl.sum(pi2, 1) + EPS
        r3 = tl.sum(pi3, 1) + EPS
        g0 = (g0 - d0[:, None]) / r0[:, None]
        g1 = (g1 - d1[:, None]) / r1[:, None]
        g2 = (g2 - d2[:, None]) / r2[:, None]
        g3 = (g3 - d3[:, None]) / r3[:, None]

    # final col vjp: output stage 1, input stage 0
    mo0 = tl.load(stages_ptr + 1 * bs * 16 + roff, mask=msk, other=0.0)
    mo1 = tl.load(stages_ptr + 1 * bs * 16 + roff + 4, mask=msk, other=0.0)
    mo2 = tl.load(stages_ptr + 1 * bs * 16 + roff + 8, mask=msk, other=0.0)
    mo3 = tl.load(stages_ptr + 1 * bs * 16 + roff + 12, mask=msk, other=0.0)
    cv = g0 * mo0 + g1 * mo1 + g2 * mo2 + g3 * mo3
    cs = (tl.load(stages_ptr + roff, mask=msk, other=0.0)
          + tl.load(stages_ptr + roff + 4, mask=msk, other=0.0)
          + tl.load(stages_ptr + roff + 8, mask=msk, other=0.0)
          + tl.load(stages_ptr + roff + 12, mask=msk, other=0.0) + EPS)
    g0 = (g0 - cv) / cs
    g1 = (g1 - cv) / cs
    g2 = (g2 - cv) / cs
    g3 = (g3 - cv) / cs

    # softmax + EPS head: only the softmax part carries gradient
    sm0 = tl.zeros((GROUP, 4), dtype=tl.float32)
    sm1 = tl.zeros((GROUP, 4), dtype=tl.float32)
    sm2 = tl.zeros((GROUP, 4), dtype=tl.float32)
    sm3 = tl.zeros((GROUP, 4), dtype=tl.float32)
    for r in tl.static_range(4):
        l = tl.load(logits_ptr + roff + r * 4, mask=msk, other=0.0)
        z = l * s2 + tl.load(base_ptr + r * 4 + ar4)[None, :]
        zmax = tl.max(z, axis=1)
        e = tl.exp(z - zmax[:, None])
        s = e / tl.sum(e, axis=1)[:, None]
        if r == 0:
            sm0 = s
        elif r == 1:
            sm1 = s
        elif r == 2:
            sm2 = s
        else:
            sm3 = s
    gz0 = sm0 * (g0 - tl.sum(g0 * sm0, 1)[:, None])
    gz1 = sm1 * (g1 - tl.sum(g1 * sm1, 1)[:, None])
    gz2 = sm2 * (g2 - tl.sum(g2 * sm2, 1)[:, None])
    gz3 = sm3 * (g3 - tl.sum(g3 * sm3, 1)[:, None])
    # z = logits*s2 + base -> dlogits in kernel; dbase/ds2 reduced on host
    tl.store(gz_ptr + roff, gz0, mask=msk)
    tl.store(gz_ptr + roff + 4, gz1, mask=msk)
    tl.store(gz_ptr + roff + 8, gz2, mask=msk)
    tl.store(gz_ptr + roff + 12, gz3, mask=msk)
    tl.store(dlogits_ptr + roff, gz0 * s2, mask=msk)
    tl.store(dlogits_ptr + roff + 4, gz1 * s2, mask=msk)
    tl.store(dlogits_ptr + roff + 8, gz2 * s2, mask=msk)
    tl.store(dlogits_ptr + roff + 12, gz3 * s2, mask=msk)


class SinkhornHead(torch.autograd.Function):
    """Fused triton res head: logits [bs,16] -> comb [bs,16], fp32, with bwd."""

    @staticmethod
    def forward(ctx, logits, s2, base):
        bs = logits.shape[0]
        stages = torch.empty(NSTAGES, bs, 16, device=logits.device, dtype=torch.float32)
        out = torch.empty_like(logits)
        consts = {'GROUP': 32, 'ITERS': ITERS, 'NSTAGES': NSTAGES, 'EPS': EPS}
        _launch(sk_fwd_ckpt_kernel, ('sk_fwd_ckpt', bs), (triton.cdiv(bs, 32),),
                (logits, out, stages, s2, base, bs), consts)
        ctx.save_for_backward(logits, s2, base, stages)
        return out

    @staticmethod
    def backward(ctx, g):
        logits, s2, base, stages = ctx.saved_tensors
        bs = logits.shape[0]
        gz = torch.empty_like(logits)
        dlogits = torch.empty_like(logits)
        consts = {'GROUP': 32, 'ITERS': ITERS, 'NSTAGES': NSTAGES, 'EPS': EPS}
        _launch(sk_bwd_kernel, ('sk_bwd', bs), (triton.cdiv(bs, 32),),
                (g.contiguous(), logits, stages, s2, base, dlogits, gz, bs), consts)
        dbase = gz.sum(0)
        ds2 = (gz * logits).sum()
        return dlogits, ds2, dbase


def pre_triton(x, w, scale, base):
    """Fallback chain with only the res head swapped for the fused triton kernel."""
    shape = x.size()
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, w) * rsqrt
    mixes = mixes.permute(1, 0, 2)

    pre = torch.sigmoid(mixes[..., :E] * scale[0] + base[:E]) + EPS
    post = 2 * torch.sigmoid(mixes[..., E:2 * E] * scale[1] + base[E:2 * E])
    # reshape only squeezes the size-1 batch dim here, leaving row stride 24;
    # the kernel reads a contiguous [bs,16], so materialize it
    logits = mixes[..., 2 * E:].reshape(BS, 16).contiguous()
    comb = SinkhornHead.apply(logits, scale[2], base[2 * E:])

    pre = pre.permute(1, 0, 2)
    post = post.permute(1, 0, 2)
    comb = comb.view(B, S, E, E).permute(1, 0, 2, 3)
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2).to(x.dtype)
    return y, post, comb


# ---------------------------------------------------------------------------
# post-side implementations
# ---------------------------------------------------------------------------

def post_broadcast(y, res, post, comb):
    """Exact mhc.py hc_post non-fused branch (fp32 broadcast chain)."""
    return (post.unsqueeze(-1) * y.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)).type_as(y)


def post_bmm(y, res, post, comb):
    """cube bmm form of the same formula (bf16 contraction)."""
    y2 = y.reshape(BS, H)
    res2 = res.reshape(BS, E, H)
    out = (y2.unsqueeze(1) * post.reshape(BS, E, 1).type_as(y2)
           + torch.matmul(comb.reshape(BS, E, E).transpose(-1, -2).type_as(res2), res2))
    return out.reshape(S, B, E, H)


def post_aclnn(y, res, post, comb):
    return mhc_post_ascend(y, res, post, comb)


# ---------------------------------------------------------------------------
# isolated correctness of the triton sinkhorn head fwd/bwd vs eager autograd
# ---------------------------------------------------------------------------

def sinkhorn_eager_np(logits, s2, base):
    m = (logits.view(-1, E, E) * s2 + base.view(1, E, E)).softmax(-1) + EPS
    m = m / (m.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        m = m / (m.sum(-1, keepdim=True) + EPS)
        m = m / (m.sum(-2, keepdim=True) + EPS)
    return m.reshape(-1, 16)


def check_sinkhorn_head():
    n = 512
    l = torch.randn(n, 16, device=DEV, dtype=torch.float32)
    s2 = (torch.randn((), device=DEV, dtype=torch.float32) * 0.05)
    b16 = torch.randn(16, device=DEV, dtype=torch.float32) * 0.1
    g = torch.randn(n, 16, device=DEV, dtype=torch.float32)

    le = l.detach().requires_grad_()
    s2e = s2.detach().requires_grad_()
    b16e = b16.detach().requires_grad_()
    ref = sinkhorn_eager_np(le, s2e, b16e)
    dle, ds2e, db16e = torch.autograd.grad(ref, (le, s2e, b16e), g)

    lt = l.detach().requires_grad_()
    s2t = s2.detach().requires_grad_()
    b16t = b16.detach().requires_grad_()
    out = SinkhornHead.apply(lt, s2t, b16t)
    dlt, ds2t, dbt = torch.autograd.grad(out, (lt, s2t, b16t), g)

    print('== triton sinkhorn head vs eager autograd (fp32, n=512) ==')
    print(f'  comb    maxdiff {(out - ref.detach()).abs().max().item():.2e}')
    print(f'  dlogits maxdiff {(dlt - dle).abs().max().item():.2e}')
    print(f'  ds2     maxdiff {(ds2t - ds2e).abs().max().item():.2e}')
    print(f'  dbase   maxdiff {(dbt - db16e).abs().max().item():.2e}')


check_sinkhorn_head()

# ---------------------------------------------------------------------------
# pre-side timing + numerics
# ---------------------------------------------------------------------------

with torch.no_grad():
    ref_y, ref_post, ref_comb = pre_torch(X, W, SCALE, BASE)
GY = torch.randn_like(ref_y)
GPOST = torch.randn_like(ref_post)
GCOMB = torch.randn_like(ref_comb)

print(f'\n== pre side (S={S}, B={B}, hc={E}, h={H}, bf16 in, {ITERS} sinkhorn iters) ==')
for name, fn in [('aclnn fused (mainline)', pre_fused),
                 ('torch fallback (mainline)', pre_torch),
                 ('fallback + triton head', pre_triton)]:
    with torch.no_grad():
        y, post, comb = fn(X, W, SCALE, BASE)
    ey = (y.float() - ref_y.float()).abs().max().item()
    ep = (post - ref_post).abs().max().item()
    ec = (comb - ref_comb).abs().max().item()
    tf = bench(lambda: fn(X, W, SCALE, BASE))

    def fwd_bwd():
        xg = X.detach().requires_grad_()
        wg = W.detach().requires_grad_()
        sg = SCALE.detach().requires_grad_()
        bg = BASE.detach().requires_grad_()
        y, post, comb = fn(xg, wg, sg, bg)
        torch.autograd.grad((y, post, comb), (xg, wg, sg, bg), (GY, GPOST, GCOMB))

    try:
        tb = bench(fwd_bwd, iters=15)
        print(f'{name:26s}: fwd {tf:7.3f} ms   fwd+bwd {tb:7.3f} ms   '
              f'maxdiff y/post/comb {ey:.1e}/{ep:.1e}/{ec:.1e}')
    except Exception as exc:  # noqa: BLE001
        print(f'{name:26s}: fwd {tf:7.3f} ms   fwd+bwd FAIL {repr(exc)[:90]}   '
              f'maxdiff {ey:.1e}/{ep:.1e}/{ec:.1e}')

# ---------------------------------------------------------------------------
# post-side timing + numerics
# ---------------------------------------------------------------------------

RES = X.detach()
Y1 = ref_y.detach()
POST1 = ref_post.detach()
COMB1 = ref_comb.detach()
GOUT = torch.randn(S, B, E, H, device=DEV, dtype=torch.bfloat16)

import torch_npu.dynamo.torchair as torchair  # noqa: E402

cfg = torchair.CompilerConfig()
try:
    post_torchair = torch.compile(post_bmm, backend=torchair.get_npu_backend(compiler_config=cfg))
except Exception as exc:  # noqa: BLE001
    post_torchair = None
    print(f'torchair post compile unavailable: {repr(exc)[:90]}')

with torch.no_grad():
    ref_out = post_broadcast(Y1, RES, POST1, COMB1).float()

print(f'\n== post side (same shape/dtype) ==')
post_variants = [('aclnn fused (mainline)', post_aclnn),
                 ('torch broadcast (fallback)', post_broadcast),
                 ('torch bmm', post_bmm)]
if post_torchair is not None:
    post_variants.append(('torchair bmm', post_torchair))

for name, fn in post_variants:
    with torch.no_grad():
        out = fn(Y1, RES, POST1, COMB1)
    eo = (out.float() - ref_out).abs().max().item()
    tf = bench(lambda: fn(Y1, RES, POST1, COMB1))

    def fwd_bwd():
        yg = Y1.detach().requires_grad_()
        rg = RES.detach().requires_grad_()
        pg = POST1.detach().requires_grad_()
        cg = COMB1.detach().requires_grad_()
        out = fn(yg, rg, pg, cg)
        torch.autograd.grad(out, (yg, rg, pg, cg), GOUT)

    try:
        tb = bench(fwd_bwd, iters=15)
        print(f'{name:26s}: fwd {tf:7.3f} ms   fwd+bwd {tb:7.3f} ms   out maxdiff {eo:.1e}')
    except Exception as exc:  # noqa: BLE001
        print(f'{name:26s}: fwd {tf:7.3f} ms   fwd+bwd FAIL {repr(exc)[:90]}   out maxdiff {eo:.1e}')

# ---------------------------------------------------------------------------
# end-to-end chains: pre + post with residual = input streams
# ---------------------------------------------------------------------------

def chain(pre_fn, post_fn):
    def run(x, w, scale, base):
        y, post, comb = pre_fn(x, w, scale, base)
        return post_fn(y, x, post, comb)
    return run


E2E = [
    ('mainline fused', chain(pre_fused, post_aclnn)),
    ('mainline fallback', chain(pre_torch, post_broadcast)),
    ('triton head + bmm post', chain(pre_triton, post_bmm)),
    ('fused pre + bmm post', chain(pre_fused, post_bmm)),
]
if post_torchair is not None:
    E2E.append(('fused pre + torchair post', chain(pre_fused, post_torchair)))

with torch.no_grad():
    ref_chain_out = E2E[1][1](X, W, SCALE, BASE).float()

print(f'\n== end-to-end mhc_sinkhorn chain (pre + post) ==')
for name, fn in E2E:
    with torch.no_grad():
        out = fn(X, W, SCALE, BASE)
    eo = (out.float() - ref_chain_out).abs().max().item()
    tf = bench(lambda: fn(X, W, SCALE, BASE))

    def fwd_bwd():
        xg = X.detach().requires_grad_()
        wg = W.detach().requires_grad_()
        sg = SCALE.detach().requires_grad_()
        bg = BASE.detach().requires_grad_()
        out = fn(xg, wg, sg, bg)
        torch.autograd.grad(out, (xg, wg, sg, bg), GOUT)

    try:
        tb = bench(fwd_bwd, iters=15)
        print(f'{name:26s}: fwd {tf:7.3f} ms   fwd+bwd {tb:7.3f} ms   out maxdiff {eo:.1e}')
    except Exception as exc:  # noqa: BLE001
        print(f'{name:26s}: fwd {tf:7.3f} ms   fwd+bwd FAIL {repr(exc)[:90]}   out maxdiff {eo:.1e}')
