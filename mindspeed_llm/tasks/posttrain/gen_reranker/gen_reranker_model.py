"""Generative reranker model (Qwen3-Reranker style).

与 ORM 的 RewardModelHead 方案不同, 本实现不新增任何参数: 复用预训练
output_layer(vocab-parallel)中 "yes"/"no" 两个 token 的权重行, 对每个位置输出

    rerank_logit[..., 0] = logit(yes),  rerank_logit[..., 1] = logit(no)

每条序列的打分位置(最后一个有效 token)由 trainer 侧的 loss_func 提取,
分数 = logit(yes) - logit(no), 与 ms-swift 生成式 reranker / Qwen3-Reranker
部署打分语义一致。

TP 处理: output_layer 是 vocab 维切分的 ColumnParallelLinear, yes/no 两个
token id 各归属一个 TP rank。每个 rank 只对自己持有的行做 matmul(未持有
则贡献 0), 再对 [s, b, 2] 做 TP 组 all_reduce(sum), 通信量可忽略。

SP 处理: 参照 orm_model, 先 gather_from_sequence_parallel_region 再取行。

CP 处理: head 是逐位置 matmul, 与序列分片无关, 模型侧无需感知; 序列分片下
每行最后一个有效位置只落在持它的 CP rank 上, trainer 侧 masked-sum 后对
CP 组 all_reduce 还原(见 loss_func)。

checkpoint 兼容: 无新参数, sharded_state_dict 沿用 GPTModel 原生实现,
存/读与普通 SFT ckpt 完全一致(全 vocab output_layer 照常保存)。
"""
import os
from typing import Literal, Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.training import get_tokenizer, get_args


def resolve_reranker_token_ids():
    """yes/no token id 解析优先级: 环境变量(ms-swift 同名, 兼容旧脚本) > args > 默认 yes/no。"""
    args = get_args()
    positive_token = os.environ.get(
        'GENERATIVE_RERANKER_POSITIVE_TOKEN',
        getattr(args, 'reranker_positive_token', None) or 'yes')
    negative_token = os.environ.get(
        'GENERATIVE_RERANKER_NEGATIVE_TOKEN',
        getattr(args, 'reranker_negative_token', None) or 'no')
    tokenizer = get_tokenizer().tokenizer

    def _token_id(token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        return int(token_id)

    positive_id = _token_id(positive_token)
    negative_id = _token_id(negative_token)
    if positive_id == negative_id:
        raise ValueError(f'reranker positive/negative token resolve to the same id: {positive_id}')
    if not getattr(args, 'reranker_token_ids_logged', False):
        args.reranker_token_ids_logged = True
        print(f'[reranker] positive_token={positive_token}(id={positive_id}), '
              f'negative_token={negative_token}(id={negative_id})', flush=True)
    return positive_id, negative_id


class GPTRerankerModel(GPTModel):
    """MCoreGPT-based generative reranker: per-position yes/no logits, no new params."""

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
    ):
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )
        self.sequence_parallel = config.sequence_parallel
        # resolved lazily on last pipeline stage (tokenizer is global once built)
        self._resolved_token_ids = None

    def _reranker_token_ids(self):
        if self._resolved_token_ids is None:
            self._resolved_token_ids = resolve_reranker_token_ids()
        return self._resolved_token_ids

    def _reranker_head(self, hidden_states: Tensor) -> Tensor:
        """hidden_states [s, b, h] -> yes/no logits [s, b, 2], TP-collected."""
        positive_id, negative_id = self._reranker_token_ids()
        # shared_embedding_or_output_weight() 返回本 TP rank 的 output 局部权重
        # [vocab_shard, h]; tie/untie 两种情况都适用(见 gpt_model.py 同名用法)。
        weight = self.shared_embedding_or_output_weight()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        vocab_shard = weight.shape[0]

        parts = []
        for token_id in (positive_id, negative_id):
            local_idx = token_id - tp_rank * vocab_shard
            if 0 <= local_idx < vocab_shard:
                row = weight[local_idx:local_idx + 1].to(hidden_states.dtype)
                parts.append(F.linear(hidden_states, row))  # [s, b, 1]
            else:
                parts.append(hidden_states.new_zeros(hidden_states.shape[:-1] + (1,)))
        logits = torch.cat(parts, dim=-1)  # [s, b, 2]

        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(logits, group=mpu.get_tensor_model_parallel_group())
        return logits

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        inference_params=None,
        **kwargs,
    ):
        # 与 orm_model 相同: 关掉 post_process 跑原生 GPT forward, 拿最终 hidden
        with patch.object(self, "post_process", False):
            hidden_states = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                inference_params=inference_params,
            )

        if self.post_process:
            if self.sequence_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
                # sp_world_size x [s/sp, b, h] -> [s, b, h]
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False)
            # [s, b, 2] -> [b, s, 2]
            return self._reranker_head(hidden_states).transpose(0, 1).contiguous()
        return hidden_states
