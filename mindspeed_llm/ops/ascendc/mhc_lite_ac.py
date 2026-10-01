# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""mhc_lite pre stage on the standalone Ascend C LitePreHeads op (scheme E).

Forward: W' = weight * gamma folded RMSNorm linearity, one raw-logits GEMM
(x @ W'^T) and the fused Ascend C op (RMS square-sum, three heads, y
mixture -- see experiments/mhc_lite/ascendc_lite_pre/).  Backward has two
switchable implementations:

  MHC_LITE_ASCENDC_GRAD=1 (task #26): one Ascend C LitePreGrad op produces
      grad_logits/grad_x/dscale/dbase from the saved bf16 x/logits and the
      op's fp32 rstd/h_pre; dscale comes back already divided by its group
      lane count.
  +MHC_LITE_ASCENDC_CHAIN=1 (G/H merge): the forward runs as the scheme-G
      one-call chain (W' + logits GEMM + op + h_res, weight/gamma passed
      into the Function instead of precomputed logits) and the whole
      backward chain -- grad op, both GEMM vjps, the W' vjps and the
      dscale lane gather -- issues from one extension call.
  default: the scheme-B triton kernels -- lite_pre_backward consumes the
      normalized logits l = logits_raw * rstd (bitwise the values the op
      used, since the bf16->fp32 cast and the rstd multiply are exact) and
      lite_grad_x adds the y-mixture term to the rstd-path gradient.

The GEMM / W' / h_res-mix vjp's stay outside the autograd.Function and are
derived by autograd itself, so dW/dgamma/dscale flow through ordinary
torch ops.

Activation contract across the fwd/bwd implementation split: neither
backward reads forward intermediates beyond rstd/h_pre -- the sigmoid/
softmax derivatives are recomputed from the bf16 raw logits exactly the
way the forward op computed them, so there is no recompute divergence
between the implementations.
"""

import os
from pathlib import Path

import torch
import torch_npu

from mindspeed_llm.ops.triton.mhc_lite_heads import (
    TRITON_AVAILABLE,
    lite_grad_x,
    lite_pre_backward,
)

_EXT = None
_EXT_GRAD = None
_LANE_IDX = {}


def _load_ext(name, src):
    from torch.utils.cpp_extension import load

    torch_npu_dir = Path(torch_npu.__file__).parent
    return load(
        name=name,
        sources=[str(src)],
        extra_include_paths=[
            str(torch_npu_dir / 'include'),
            str(torch_npu_dir / 'include/third_party/acl/inc'),
            '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
        ],
        extra_ldflags=[f'-L{torch_npu_dir}/lib', '-ltorch_npu'],
        verbose=False,
    )


def _ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not os.environ.get('ASCEND_CUSTOM_OPP_PATH'):
        raise RuntimeError(
            'MHC_LITE_ASCENDC needs ASCEND_CUSTOM_OPP_PATH pointing at the '
            'vendor tree built by '
            'bash experiments/mhc_lite/ascendc_lite_pre/sync_and_build.sh '
            '(set it before importing torch_npu)')
    repo = Path(__file__).resolve().parents[3]
    src = repo / 'experiments/mhc_lite/ascendc_lite_pre/extension.cpp'
    if not src.exists():
        raise RuntimeError(f'LitePreHeads extension source missing: {src}')
    _EXT = _load_ext('ascendc_lite_pre_ext', src)
    return _EXT


def _ext_grad():
    global _EXT_GRAD
    if _EXT_GRAD is not None:
        return _EXT_GRAD
    if not os.environ.get('ASCEND_CUSTOM_OPP_PATH'):
        raise RuntimeError(
            'MHC_LITE_ASCENDC_GRAD needs ASCEND_CUSTOM_OPP_PATH pointing at '
            'the vendor tree built by '
            'bash experiments/mhc_lite/ascendc_lite_pre_grad/sync_and_build.sh '
            '(set it before importing torch_npu)')
    repo = Path(__file__).resolve().parents[3]
    src = repo / 'experiments/mhc_lite/ascendc_lite_pre_grad/extension.cpp'
    if not src.exists():
        raise RuntimeError(f'LitePreGrad extension source missing: {src}')
    _EXT_GRAD = _load_ext('ascendc_lite_pre_grad_ext', src)
    return _EXT_GRAD


def _lane_idx(device):
    # per-logit-lane scale expansion: s0 x4 | s1 x4 | s2 x24 (e = 4, 24 perms)
    if device not in _LANE_IDX:
        _LANE_IDX[device] = torch.tensor([0] * 4 + [1] * 4 + [2] * 24,
                                         device=device)
    return _LANE_IDX[device]


_INV_LANE_COUNT = {}


def _inv_lane_count(device):
    # the backward's dscale entries are per-group totals, but the gather
    # backward sums the [32] lane grads, so each lane returns its group's
    # share (total / group width)
    if device not in _INV_LANE_COUNT:
        counts = torch.tensor([4.0, 4.0, 24.0], device=device)
        _INV_LANE_COUNT[device] = counts.reciprocal()[_lane_idx(device)]
    return _INV_LANE_COUNT[device]


def _chain():
    # G/H merge: one extension call per direction (forward chain + whole
    # backward chain); reads env at call time so a process can toggle it
    return os.environ.get('MHC_LITE_ASCENDC_CHAIN') == '1'


class _LitePreAscendCFn(torch.autograd.Function):
    """Fused Ascend C lite-pre forward with a switchable backward.

    Inputs are the flattened streams, the fp32 scale/base vectors and the
    permutation tables; slot a/b carry either the raw bf16 logits (W' GEMM
    left outside, autograd derives its vjp) or, with
    MHC_LITE_ASCENDC_CHAIN=1, the bf16 weight/gamma pair (the forward
    chain and the whole backward chain each issue as one extension call).
    h_pre stays inside so the y-path dots are the complete dh_pre.
    Returns (y, h_post, h_res).
    """

    @staticmethod
    def forward(ctx, xf, a, b, scale32, base32, perm_flat, perm_t):
        sb = xf.shape[0]
        if _chain():
            y, hpre8, hpost8, coeff, rstd, h_res, logits = _ext().lite_pre_chain(
                xf, a, b, scale32, base32, perm_flat)
            ctx.save_for_backward(xf, a, b, hpre8, rstd, perm_t, logits, scale32, base32)
            return y, hpost8[:, 4:].contiguous(), h_res
        logits = a
        y, hpre8, hpost8, coeff, rstd = _ext().lite_pre_heads(xf, logits, scale32, base32)
        h_res = torch.matmul(coeff[:, 8:], perm_flat)
        ctx.save_for_backward(xf, logits, hpre8, rstd, perm_t, scale32, base32)
        return y, hpost8[:, 4:].contiguous(), h_res

    @staticmethod
    def backward(ctx, grad_y, grad_h_post, grad_h_res):
        if _chain():
            xf, wbf, gbf, hpre8, rstd, perm_t, logits, scale32, base32 = ctx.saved_tensors
            grad_x, grad_w, grad_g, grad_s, grad_b = _ext_grad().lite_pre_train_backward(
                grad_y,
                grad_h_post.contiguous(),
                grad_h_res.contiguous(),
                xf,
                wbf,
                gbf,
                perm_t,
                logits,
                rstd,
                hpre8,
                scale32,
                base32,
            )
            return grad_x, grad_w, grad_g, grad_s, grad_b, None, None

        xf, logits, hpre8, rstd, perm_t, scale32, base32 = ctx.saved_tensors
        sb, eh = xf.shape
        h = eh // 4

        if os.environ.get('MHC_LITE_ASCENDC_GRAD') == '1':
            # whole backward in one op; dscale arrives already divided by
            # its group lane count, so the gather needs no correction
            grad_logits, grad_x, dscale3, dbase = _ext_grad().lite_pre_grad(
                grad_y,
                xf,
                grad_h_post.contiguous(),
                grad_h_res.contiguous(),
                perm_t,
                logits,
                rstd,
                hpre8,
                scale32,
                base32,
            )
            return grad_x, grad_logits, None, dscale3[_lane_idx(xf.device)], dbase, None, None

        # l = logits_raw * rstd: the exact values the op differentiated along
        l = logits.float() * rstd.unsqueeze(-1)
        dl, dscale3, dbase = lite_pre_backward(
            grad_y,
            xf.view(sb, 4, h),
            grad_h_post.contiguous(),
            grad_h_res.contiguous(),
            perm_t,
            l,
            scale32[[0, 4, 8]],
            base32,
        )

        # chain rule through l = L * rstd(x): two x-paths add up, the GEMM
        # path (dL @ W') is produced by autograd outside this Function
        grad_logits = (dl * rstd.unsqueeze(-1)).to(logits.dtype)
        drstd = (dl * logits.float()).sum(-1)
        # per-row coefficient in fp32, one bf16 broadcast mul (the fp32
        # materialization + cast costs two extra full-stream passes)
        coef = (drstd * -(rstd.pow(3)) / eh).to(xf.dtype).unsqueeze(-1).unsqueeze(-1)
        grad_x = lite_grad_x(grad_y, hpre8[:, :4].contiguous(),
                             coef * xf.view(sb, 4, h)).view(sb, eh)

        grad_scale32 = dscale3[_lane_idx(xf.device)] * _inv_lane_count(xf.device)
        return grad_x, grad_logits, None, grad_scale32, dbase, None, None


def lite_pre_ascendc(x, weight, gamma, scale, base, perm_flat, perm_t, eps):
    """x [s,b,e,h] -> (y, h_post, h_res), the MHCLite.hc_pre protocol on the
    standalone Ascend C op.  bf16 streams only; requires the built custom
    op (and, without MHC_LITE_ASCENDC_GRAD=1, triton for its backward)."""
    if not TRITON_AVAILABLE and os.environ.get('MHC_LITE_ASCENDC_GRAD') != '1':
        raise RuntimeError('MHC_LITE_ASCENDC needs triton for its backward'
                           ' (or MHC_LITE_ASCENDC_GRAD=1)')
    s, b, e, h = x.shape
    sb = s * b
    xf = x.reshape(sb, e * h)
    scale32 = scale.float()[_lane_idx(x.device)]
    if _chain():
        y, h_post, h_res = _LitePreAscendCFn.apply(
            xf, weight.type_as(x), gamma.type_as(x), scale32, base.float(),
            perm_flat, perm_t)
    else:
        # W' = W * gamma: 128x smaller than the xn materialization of the
        # triton path, and the gamma vjp is autograd's broadcast-mul backward
        wp = weight.type_as(x) * gamma.type_as(x).view(1, -1)
        logits = torch.matmul(xf, wp.t())
        y, h_post, h_res = _LitePreAscendCFn.apply(
            xf, logits, None, scale32, base.float(), perm_flat, perm_t)
    return y.view(s, b, h), h_post.view(s, b, e), h_res.view(s, b, e, e)
