# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

"""mhc_lite: lightweight hyper-connections with permutation-mixture h_res.

Experimental counterpart of deepseek4/mhc.py. It plugs into the same
attn_mhc/mlp_mhc/hc_head slots and follows the same pre/post/head protocol:
hc_pre(x) -> (y, h_post, h_res); hc_post(x, residual=, post=, comb=) -> streams.

Coefficient parameterization differs from the full MHC:
- RMSNorm over the flattened streams keeps a learned gamma;
- the fused projection emits [pre | post | res] logits in one GEMM
  (2*e + e! columns instead of (2+e)*e);
- h_res is a convex combination of the e! permutation matrices (one softmax,
  no sinkhorn iterations), which spans the same Birkhoff polytope;
- h_post side reuses the Ascend fused mhc_post operator, which only consumes
  the produced (h_post, h_res) and is independent of how they were computed.
"""

import itertools
import math
import os
from dataclasses import dataclass
from typing import Union

import torch
import torch_npu
from torch import nn

from megatron.training import get_args
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module

from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.ops.npu_mhc import mhc_post_ascend
from mindspeed_llm.ops.triton.mhc_lite_heads import (
    TRITON_AVAILABLE,
    lite_heads_y_forward,
    lite_res_head_forward,
    lite_heads_backward,
    lite_y_backward,
    lite_add_cast,
    lite_post_backward,
)


def _triton_lite_enabled():
    return TRITON_AVAILABLE and os.environ.get('MHC_LITE_TRITON') == '1'


@dataclass
class MHCLiteSubmodules:
    hc_fn: Union[ModuleSpec, type] = None


def get_mhc_lite_spec(enable_mhc):
    """Layer/head spec for the lite hyper-connections module."""
    if enable_mhc:
        return ModuleSpec(
            module=MHCLite,
            submodules=MHCLiteSubmodules(hc_fn=LinearNoTP),
        )
    return IdentityOp


def _permutation_mats_flat(n: int) -> torch.Tensor:
    """All n! permutation matrices flattened to [n!, n*n] in fp32."""
    perms = list(itertools.permutations(range(n)))
    indices = torch.tensor(perms, dtype=torch.int64)
    return torch.eye(n, dtype=torch.float32)[indices].flatten(1)


class _MhcPostFn(torch.autograd.Function):
    """Fused Ascend mhc_post forward with an explicit backward.

    out_j = h_post_j * h_out + sum_i h_res[i, j] * x_i. The aclnnMhcPostBackward
    kernel is unstable on some in-range coefficient values (internal launch
    failure), so gradients are recomputed explicitly: fused triton kernel when
    enabled, fp32 torch reductions otherwise.
    """

    @staticmethod
    def forward(ctx, h_out, x, h_post, h_res):
        ctx.save_for_backward(h_out, x, h_post, h_res)
        ctx.use_triton = _triton_lite_enabled()
        return mhc_post_ascend(h_out, x, h_post, h_res)

    @staticmethod
    def backward(ctx, grad):
        h_out, x, h_post, h_res = ctx.saved_tensors
        if ctx.use_triton:
            sb = h_out.shape[0] * h_out.shape[1]
            e, h = x.shape[2], x.shape[3]
            dh_out, dx, dh_post, dh_res = lite_post_backward(
                grad.reshape(sb, e, h),
                h_out.reshape(sb, h),
                x.reshape(sb, e, h),
                h_post.reshape(sb, e),
                h_res.reshape(sb, e * e),
            )
            return dh_out.view_as(h_out), dx.view_as(x), dh_post.view_as(h_post), dh_res.view_as(h_res)
        g = grad.float()
        h_outf = h_out.float()
        grad_h_out = torch.einsum('sbe,sbeh->sbh', h_post, g).to(h_out.dtype)
        grad_post = torch.einsum('sbeh,sbh->sbe', g, h_outf)
        grad_x = torch.matmul(h_res.transpose(-1, -2), g).to(x.dtype)
        grad_res = torch.einsum('sbjh,sbih->sbij', g, x.float())
        return grad_h_out, grad_x, grad_post, grad_res


class _LitePreTritonFn(torch.autograd.Function):
    """Tier-1 lite pre stage: CANN rms_norm + CANN logits GEMM + fused triton
    heads/y kernel, with a backward built from the triton heads jacobian, the
    mhc_pre_only y-path kernel and CANN rms_norm backward."""

    @staticmethod
    def forward(ctx, x, weight, gamma, scale, base, perm_flat, eps):
        s, b, e, h = x.shape
        sb = s * b
        xf = x.reshape(sb, e * h)
        xn, rstd = torch_npu.npu_rms_norm(xf, gamma.type_as(x), epsilon=eps)
        logits = torch.matmul(xn, weight.type_as(x).t()).float()
        y, h_pre, h_post = lite_heads_y_forward(
            logits, xf.view(sb, e, h), scale.float(), base.float()
        )
        h_res = lite_res_head_forward(logits, scale.float(), base.float(), perm_flat)
        ctx.save_for_backward(xf, xn, rstd, logits, h_pre, weight, gamma, scale, base)
        ctx.perm_flat_t = perm_flat.t()
        ctx.x_shape = (s, b, e, h)
        ctx.eps = eps
        return y.view(s, b, h), h_post.view(s, b, e), h_res.view(s, b, e, e)

    @staticmethod
    def backward(ctx, grad_y, grad_h_post, grad_h_res):
        xf, xn, rstd, logits, h_pre, weight, gamma, scale, base = ctx.saved_tensors
        s, b, e, h = ctx.x_shape
        sb = s * b

        d_h_pre, d_x_direct = lite_y_backward(
            grad_y.reshape(sb, h), xf.view(sb, e, h), h_pre
        )
        dcoeff = torch.matmul(grad_h_res.reshape(sb, e * e).float(), ctx.perm_flat_t)
        dlogits, d_scale, d_base = lite_heads_backward(
            d_h_pre,
            grad_h_post.reshape(sb, e).float(),
            dcoeff,
            logits,
            scale.float(),
            base.float(),
        )
        grad_weight = torch.matmul(dlogits.t().to(xn.dtype), xn).to(weight.dtype)
        d_xn = torch.matmul(dlogits.to(weight.dtype), weight)
        d_x_rms, d_gamma = torch_npu.npu_rms_norm_backward(d_xn, xf, gamma.type_as(xf), rstd)
        grad_x = lite_add_cast(d_x_direct, d_x_rms.view(sb, e, h))
        return (
            grad_x.view(s, b, e, h),
            grad_weight,
            d_gamma.to(gamma.dtype),
            d_scale.to(scale.dtype),
            d_base.to(base.dtype),
            None,
            None,
        )


class MHCLite(MegatronModule):
    _perm_mats_cache = {}

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MHCLiteSubmodules,
        mhc_position: str,
        layer_number: int,
    ):
        super().__init__(config=config)
        args = get_args()

        self.mhc_position = mhc_position
        self.hc_mult = hc_mult = args.hc_mult
        self.num_perm_mats = math.factorial(hc_mult)
        self.norm_eps = args.norm_epsilon

        hc_dim = hc_mult * config.hidden_size
        is_head = mhc_position == 'head'
        mix_hc = hc_mult if is_head else 2 * hc_mult + self.num_perm_mats

        self.hc_fn = build_module(
            submodules.hc_fn,
            hc_dim,
            mix_hc,
            config=self.config,
            bias=False,
        )
        # zero weights make the biases alone select the initial stream routing
        with torch.no_grad():
            self.hc_fn.weight.zero_()

        device = torch.device('cpu') if config.use_cpu_initialization else torch.npu.current_device()
        self.hc_gamma = nn.Parameter(torch.ones(hc_dim, device=device))
        self.hc_scale = nn.Parameter(torch.full((1 if is_head else 3,), 1e-2, device=device))

        hc_base = torch.zeros(mix_hc, device=device)
        if not is_head:
            # pre bias selects this layer's residual stream, res bias the identity permutation
            hc_base[:hc_mult] = -8.0
            hc_base[layer_number % hc_mult] = 8.0
            hc_base[2 * hc_mult :] = -8.0
            hc_base[2 * hc_mult] = 0.0
        self.hc_base = nn.Parameter(hc_base)

        for param in (self.hc_gamma, self.hc_scale, self.hc_base):
            setattr(param, 'sequence_parallel', config.sequence_parallel)

        try:
            import cann_ops_transformer  # noqa: F401
            self.use_cann_post = True
        except ImportError:
            self.use_cann_post = False
        if os.environ.get('MHC_LITE_TORCH_POST') == '1':
            self.use_cann_post = False
        self.use_triton_pre = _triton_lite_enabled()

    def _get_perm_mats(self, device) -> torch.Tensor:
        # class-level cache keeps the fp32 tables out of Float16Module casts
        if device not in self._perm_mats_cache:
            self._perm_mats_cache[device] = _permutation_mats_flat(self.hc_mult).to(device)
        return self._perm_mats_cache[device]

    def _coefficients(self, x: torch.Tensor):
        """x: [s,b,e,h] streams -> (h_pre in x.dtype, h_post/h_res in fp32).

        The Ascend mhc_post operator consumes fp32 h_post/h_res, matching the
        full-MHC fused path where both come from the pre operator in fp32.
        """
        s, b, e, h = x.shape
        x_norm = torch_npu.npu_rms_norm(
            x.reshape(s, b, -1), self.hc_gamma.type_as(x), epsilon=self.norm_eps
        )[0]
        logits = self.hc_fn(x_norm).float()

        pre_logits, post_logits, res_logits = torch.split(
            logits, [e, e, self.num_perm_mats], dim=-1
        )
        h_pre = (torch.sigmoid(pre_logits * self.hc_scale[0] + self.hc_base[:e])).type_as(x)
        h_post = 2 * torch.sigmoid(post_logits * self.hc_scale[1] + self.hc_base[e : 2 * e])
        res_logits = res_logits * self.hc_scale[2] + self.hc_base[2 * e :]
        perms_coeff = torch.softmax(res_logits, dim=-1, dtype=torch.float32)
        h_res = torch.matmul(perms_coeff, self._get_perm_mats(x.device)).view(s, b, e, e)
        return h_pre, h_post, h_res

    def hc_pre(self, x: torch.Tensor, *args, **kwargs):
        # x: [s,b,e,h] -> y: [s,b,h]
        s, b, e, h = x.shape
        if self.use_triton_pre:
            return _LitePreTritonFn.apply(
                x, self.hc_fn.weight, self.hc_gamma, self.hc_scale, self.hc_base,
                self._get_perm_mats(x.device), self.norm_eps,
            )
        h_pre, h_post, h_res = self._coefficients(x)
        y = torch.matmul(h_pre.view(s * b, 1, e), x.view(s * b, e, h)).view(s, b, h)
        return y.type_as(x), h_post, h_res

    def hc_post(self, x: torch.Tensor, *args, **kwargs):
        residual, post, comb = kwargs['residual'], kwargs['post'], kwargs['comb']

        # x: [s,b,d], residual: [s,b,e,d], post: [s,b,e] fp32, comb: [s,b,e,e] fp32 -> y: [s,b,e,d]
        # the Ascend op only accepts fp16/bf16 streams
        if self.use_cann_post and x.dtype in (torch.bfloat16, torch.float16):
            y = _MhcPostFn.apply(x, residual, post, comb)
            return y.type_as(x)
        y = x.unsqueeze(2) * post.type_as(x).unsqueeze(3)
        y = y + torch.matmul(comb.type_as(x).transpose(-1, -2), residual)
        return y.type_as(x)

    def hc_head(self, x: torch.Tensor, *args, **kwargs):
        # x: [s,b,e,h] -> y: [s,b,h]; pre-only reduction with the lite parameterization
        s, b, e, h = x.shape
        x_norm = torch_npu.npu_rms_norm(
            x.reshape(s, b, -1), self.hc_gamma.type_as(x), epsilon=self.norm_eps
        )[0]
        logits = self.hc_fn(x_norm)
        h_pre = torch.sigmoid(logits * self.hc_scale[0] + self.hc_base).type_as(x)
        y = torch.matmul(h_pre.view(s * b, 1, e), x.view(s * b, e, h)).view(s, b, h)
        return y.type_as(x)

    def hc_identity(self, x, *args, **kwargs):
        return x

    def forward(self, hidden_states, mhc_stage='identity', *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        if mhc_stage == 'pre':
            return self.hc_pre(hidden_states, *args, **kwargs)
        if mhc_stage == 'post':
            return self.hc_post(hidden_states, *args, **kwargs)
        if mhc_stage == 'head':
            return self.hc_head(hidden_states, *args, **kwargs)
        if mhc_stage == 'identity':
            return self.hc_identity(hidden_states, *args, **kwargs)
        raise AssertionError(
            f"Invalid mhc_stage '{mhc_stage}', only support 'pre' 'post' 'head' 'identity'."
        )
