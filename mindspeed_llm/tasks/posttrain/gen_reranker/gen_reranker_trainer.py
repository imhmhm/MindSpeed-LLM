"""Generative reranker trainer (listwise), ms-swift `listwise_reranker` 等价实现。

任务组织(与 ms-swift 对齐, 变长组):
  - 一条样本 = 一个 query 组 = 1 个正样本 + 1~N 个负样本(N<=--reranker-max-negatives),
    共 K_i 条独立序列(真实数据 K_i ∈ [2,6]); collator 产出 [ΣK_i, S] 的 batch,
    并携带 group_sizes[G]; 组内顺序固定(0 号为正样本);
  - 模型(GPTRerankerModel)对每条序列每个位置输出 yes/no 两个 logit;
  - loss 只取每条序列最后一个有效位置: score = logit(yes) - logit(no);
  - 逐组 softmax cross-entropy, 目标类别 = 0(正样本位置), 对组取平均
    (ms-swift ListwiseRerankerLoss 的逐组循环在此精确移植);
  - 指标: 组内 argmax 命中正样本的比例(两种 loss 同口径), 跨 DP + 跨 micro-batch
    的组等权全局平均(与 ms-swift RerankerMetrics 的 MRR/NDCG、Qwen3-Reranker
    部署形态同为排序口径, 无绝对阈值); pointwise 另报逐序列二分类 acc 作辅助
    (绝对校准口径 = ms-swift pointwise acc, 基线 (K-1)/K)。
  
  - 变长组下 batch 行数逐 micro-batch 浮动, 与 PP 的固定通信 shape 不兼容,
    pipeline_model_parallel_size > 1 时显式报错; TP/DP/CP 不受影响
    (CP 沿序列维切分, 与 batch 行数浮动无关)。

"""
from functools import partial

import torch
import torch.nn.functional as F

from megatron.core import mpu, tensor_parallel
from megatron.training import get_args, print_rank_0, get_timers
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group,
)
from megatron.training.training import evaluate_and_print_results
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

from mindspeed_llm.training.utils import get_tune_attention_mask
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.gen_reranker.gen_reranker_model import GPTRerankerModel


class RerankerTrainer(BaseTrainer):
    """Listwise generative reranker trainer."""

    def __init__(self):
        if getattr(get_args(), 'pipeline_model_parallel_size', 1) > 1:
            raise NotImplementedError(
                'variable group size is incompatible with the fixed PP communication shape; '
                'use TP/DP only (pipeline-model-parallel-size=1)')
        super().__init__()

    def train(self):
        # --eval-on-start: 训练前先跑一轮 valid eval 作为起点 ckpt 的基线
        # (= ms-swift --eval_on_start true); dataset 侧已同步给 valid 流的
        # epoch 填充需求量加上这一 pass, 不会挤占训练中的 eval。每轮评估
        # (含这次 pre-eval 和训练中的 valid) 后的缓存释放在 megatron
        # evaluate() 末尾统一处理(对所有 stage 生效)
        args = get_args()
        if (getattr(args, 'eval_on_start', False) and args.do_valid
                and args.do_train and not args.skip_train):
            (forward_step_func, model, _, _, _, valid_data_iterator,
             process_non_loss_data_func, config) = self.train_args
            evaluate_and_print_results(
                f'iteration {args.iteration} (pre-train) on validation set',
                forward_step_func, valid_data_iterator, model, args.iteration,
                process_non_loss_data_func, config,
                verbose=True, write_to_tensorboard=True)
        super().train()


    @staticmethod
    def model_provider(pre_process=True, post_process=True):
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building reranker GPT model ...')
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if not args.use_mcore_models:
            raise ValueError("Reranker training currently supports mcore only.")

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        return GPTRerankerModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )


    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch: [ΣK_i, S] tokens + attention_mask + group_sizes(+ last-pos mask)."""
        args = get_args()
        keys = ['input_ids', 'attention_mask', 'group_sizes']
        data_type = torch.int64

        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        group_sizes = data_b.get('group_sizes').long()
        # 每行只留最后一个有效位置: [1,1,...,1,0,..] -> [0,..,1,0,..]
        last_token_position = torch.sum(attention_mask_1d, dim=1) - 1
        last_pos_mask = torch.zeros_like(attention_mask_1d)
        last_pos_mask.scatter_(1, last_token_position.unsqueeze(1), 1)

        batch = {
            'tokens': tokens,
            'last_pos_mask': last_pos_mask,
            'attention_mask': get_tune_attention_mask(attention_mask_1d),
        }
        batch = get_batch_on_this_cp_rank(batch)
        return batch['tokens'], batch['last_pos_mask'], group_sizes, batch['attention_mask'], None


    def loss_func(self, last_pos_mask: torch.Tensor, group_sizes: torch.Tensor,
                  output_tensor: torch.Tensor):
        """output_tensor: [b, s, 2] yes/no logits (CP 下按 s 分片);
        --reranker-loss-type 选择 listwise(组内 CE, 默认) 或 pointwise(逐序列 BCE)。"""
        args = get_args()
        temperature = args.reranker_temperature

        # [b, s, 2] -> 每行最后有效位置的 yes/no logit: [b, 2]
        # CP>1 时 last_pos_mask 只在持有该位置的 rank 上为 1, all_reduce(sum) 还原全局值
        token_logits = (last_pos_mask.unsqueeze(-1).float() * output_tensor.float()).sum(dim=1)
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(token_logits, group=mpu.get_context_parallel_group())

        scores = token_logits[:, 0] - token_logits[:, 1]  # logit(yes) - logit(no), [b]
        if int(group_sizes.sum()) != scores.shape[0]:
            raise ValueError(
                f'group_sizes sum {int(group_sizes.sum())} != batch rows {scores.shape[0]}; '
                f'data pipeline mismatch (collator vs dataset)')

        if args.reranker_loss_type == 'pointwise':
            # 逐序列 BCE: 组内首行(正样本) label=1, 其余 0 —— swift PointwiseRerankerLoss
            # 的精确移植(BCEWithLogits 全序列均值, 无温度), 与 Qwen3-Reranker 官方 SFT
            # 目标同型(含绝对校准项; 见 listwise_vs_pointwise 推导文档)
            labels = torch.zeros_like(scores)
            starts = torch.cumsum(group_sizes, 0) - group_sizes
            labels[starts.long()] = 1
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            with torch.no_grad():
                # 辅助指标: 逐序列二分类 acc(score>0 即 sigmoid>0.5) —— 绝对校准
                # 口径, ms-swift pointwise 的 acc 即此; 基线 (K-1)/K(全预测 no 即
                # 达到, 1:5 失衡下 0.833), 只作校准参考, 排序能力看主指标 reranker acc
                binary_hits = ((scores > 0) == labels.bool()).float().sum()
                binary_count = float(scores.numel())
        else:
            # 逐组 listwise CE(变长组), ms-swift ListwiseRerankerLoss 的精确移植
            group_losses = []
            start = 0
            for k in group_sizes.tolist():
                group_scores = scores[start:start + k] / temperature
                target = torch.zeros(1, dtype=torch.long, device=group_scores.device)
                group_losses.append(F.cross_entropy(group_scores.unsqueeze(0), target))
                start += k
            loss = torch.stack(group_losses).mean()

        # acc 两种 loss 同口径: 组内 argmax 命中正样本(组等权) —— ms-swift
        # RerankerMetrics 与 Qwen3-Reranker 部署形态均为连续分数排序(MRR/NDCG),
        # 无绝对阈值; 温度除法不改变 argmax
        with torch.no_grad():
            hits = 0.0
            start = 0
            for k in group_sizes.tolist():
                hits += float(scores[start:start + k].argmax().item() == 0)
                start += k
            count = float(len(group_sizes))

        averaged_loss = average_losses_across_data_parallel_group([loss])
        acc_stat = torch.tensor([float(hits), count], dtype=torch.float,
                                device=scores.device)
        torch.distributed.all_reduce(acc_stat, group=mpu.get_data_parallel_group())
        metrics = {'reranker loss': averaged_loss[0],
                   'reranker acc': (acc_stat[0], acc_stat[1])}
        if args.reranker_loss_type == 'pointwise':
            binary_stat = torch.tensor([float(binary_hits), binary_count],
                                       dtype=torch.float, device=scores.device)
            torch.distributed.all_reduce(binary_stat, group=mpu.get_data_parallel_group())
            metrics['reranker acc binary'] = (binary_stat[0], binary_stat[1])
        return loss, metrics


    def forward_step(self, data_iterator, model):
        args = get_args()
        timers = get_timers()

        timers('batch-generator', log_level=2).start()
        tokens, last_pos_mask, group_sizes, attention_mask, _ = self.get_batch(data_iterator)
        timers('batch-generator').stop()

        output_tensor = model(tokens, None, attention_mask)

        return output_tensor, partial(self.loss_func, last_pos_mask, group_sizes)
