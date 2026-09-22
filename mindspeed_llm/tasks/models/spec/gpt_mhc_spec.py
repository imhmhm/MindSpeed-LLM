# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

"""
GPT layer specification with MHC slots, for GPT-architecture models reusing hyper-connections.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.training import get_args

from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.core.transformer.transformer_layer import TransformerLayer, CustomTransformerLayerSubmodules
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc import get_mhc_spec, get_add_op_with_bias


args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm, enable_mhc = (
    args.num_experts,
    args.moe_grouped_gemm,
    args.qk_layernorm,
    args.enable_mhc,
)
use_te = args.transformer_impl == "transformer_engine"

layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=CustomTransformerLayerSubmodules(
        attn_mhc=get_mhc_spec(enable_mhc=enable_mhc),
        mlp_mhc=get_mhc_spec(enable_mhc=enable_mhc),
        input_layernorm=IdentityOp if use_te else PTNorm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                core_attention=TEDotProductAttention if use_te else DotProductAttention,
                linear_proj=TERowParallelLinear if use_te else RowParallelLinear,
                q_layernorm=PTNorm if qk_layernorm else IdentityOp,
                k_layernorm=PTNorm if qk_layernorm else IdentityOp,
            ),
        ),
        self_attn_bda=get_add_op_with_bias,
        pre_mlp_layernorm=IdentityOp if (not num_experts and use_te) else PTNorm,
        mlp=get_mlp_module_spec(
            use_te=use_te, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
        ),
        mlp_bda=get_add_op_with_bias,
        sharded_state_dict_keys_map={
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        },
    ),
)
