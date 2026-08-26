"""Reranker dataset + collator + provider (变长组版, 单流).

磁盘格式(由 preprocess_data/preprocess_data_reranker.py 生成, 单流):
    {output_prefix}_packed_input_ids_document.bin/.idx
一个 document = 一个 query 组(多 item): 组内 K_i 条 (query, doc) 的 prompt-only
序列连续存放, 首条为正样本; 组边界即 .idx 的 document_indices, 无需额外流。

RerankerDataset 把一个"组"作为一个样本吐出(变长 K_i 条序列, 经 slice 读取
自动按 item 长度切分); RerankerDataCollator 把一个 micro-batch 的 G 个组 pad 成
[ΣK_i, S] 的 input_ids/attention_mask, 并输出 group_sizes[G](int64) ——
训练侧 loss 按组边界逐组算 listwise CE, 与 ms-swift ListwiseRerankerLoss
(以 labels==1 定组边界)语义一致。
"""
from typing import Dict, List, Optional, Sequence

import os

import numpy as np
import torch

from megatron.core import mpu
from megatron.core.datasets import indexed_dataset
from megatron.training import get_args, get_tokenizer, print_rank_0


def _indexed_dataset_exists(path_prefix: str) -> bool:
    return (os.path.exists(indexed_dataset.get_idx_path(path_prefix))
            and os.path.exists(indexed_dataset.get_bin_path(path_prefix)))


class RerankerDataset(torch.utils.data.Dataset):
    """Group-level view over the single stream: item i = the i-th group (variable size)."""

    def __init__(self, group_indices: Optional[List[int]] = None, ids_ds=None):
        self._ds = ids_ds
        # document g 的 item 范围 = document_indices[g] : document_indices[g+1]
        self._doc_indices = np.asarray(self._ds.index.document_indices, dtype=np.int64)
        if group_indices is None:
            self.group_indices = list(range(len(self._doc_indices) - 1))
        else:
            self.group_indices = group_indices

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx) -> Dict[str, List[List[int]]]:
        gi = self.group_indices[idx]
        start, end = int(self._doc_indices[gi]), int(self._doc_indices[gi + 1])
        # slice 读取: 按 item 长度切分, 返回该组每条序列的数组列表
        return {'input_ids': [seq.tolist() for seq in self._ds[start:end]]}


class RerankerDataCollator:
    """Pad G groups (variable sizes) into one flat [ΣK_i, S] batch + group_sizes[G]."""

    def __init__(self, pad_token_id: int, seq_length: Optional[int] = None,
                 pad_to_seq_length: bool = False):
        self.pad_token_id = pad_token_id
        self.seq_length = seq_length
        self.pad_to_seq_length = pad_to_seq_length

    def __call__(self, features: Sequence[Dict[str, List[List[int]]]]) -> Dict[str, torch.Tensor]:
        flat: List[List[int]] = []
        group_sizes: List[int] = []
        for feature in features:
            flat.extend(feature['input_ids'])
            group_sizes.append(len(feature['input_ids']))
        max_len = max(len(seq) for seq in flat)
        if self.pad_to_seq_length and self.seq_length is not None:
            if max_len > self.seq_length:
                raise ValueError(f'padded len {max_len} > seq_length {self.seq_length}')
            max_len = self.seq_length

        input_ids = torch.full((len(flat), max_len), self.pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((len(flat), max_len), dtype=torch.int64)
        for i, seq in enumerate(flat):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.int64)
            attention_mask[i, :len(seq)] = 1
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'group_sizes': torch.tensor(group_sizes, dtype=torch.int64),
        }


def build_reranker_dataset(data_prefix, splits_string, seq_length, train_valid_test_num_samples, seed):
    """Provider entry: build train/valid/test RerankerDataset from the single stream.

    split 用 seeded permutation 按 splits_string 权重切分, 所有 rank 上确定一致;
    不做 epoch 重复/配额(组数即样本数)。
    """
    args = get_args()
    if len(data_prefix) == 1:
        data_prefix = data_prefix[0].split(',')
    if len(data_prefix) != 1:
        raise ValueError(
            f'reranker stage currently supports a single --data-path prefix, got {data_prefix}. '
            f'Multiple/weighted blends: 分别预处理后训练时逐个传入')
    prefix = data_prefix[0]
    # 文件名约定与 SFT 一致(mtf_dataset.get_packed_indexed_dataset):
    # IndexedDataset(f"{data_prefix}_packed_{field}_document"), data-path 传纯前缀
    stream_prefix = f'{prefix}_packed_input_ids_document'
    if not _indexed_dataset_exists(stream_prefix):
        raise FileNotFoundError(
            f'{indexed_dataset.get_idx_path(stream_prefix)} / .bin not found. '
            f'preprocess_data_reranker.py writes {{output_prefix}}_packed_input_ids_document'
            f'.bin/.idx; pass the plain --output-prefix via --data-path')

    ids_ds = indexed_dataset.IndexedDataset(stream_prefix)
    total_groups = len(ids_ds.index.document_indices) - 1
    # 数据契约校验(计算语义不依赖这两个值 —— loss 只认盘上的 group_sizes):
    # 1) 组 >= 2 条序列是 listwise CE 的数学要求(单条序列 softmax 恒为 1, loss 恒为
    #    0, 白耗算力), 违反说明数据不是用 --min-group-size>=2 预处理的, 硬报错;
    # 2) 盘上最大组 <= 1 + --reranker-max-negatives, 不等说明预处理参数与训练认知
    #    不一致(训练仍按盘上 group_sizes 跑, 仅告警提示核对)。
    group_sizes = np.diff(np.asarray(ids_ds.index.document_indices))
    min_g, max_g = int(group_sizes.min()), int(group_sizes.max())
    if min_g < 2:
        raise ValueError(
            f'found groups with {min_g} sequence(s) in {stream_prefix}: listwise CE is '
            f'degenerate on single-sequence groups. Re-preprocess with '
            f'preprocess_data_reranker.py --min-group-size >= 2.')
    max_neg = getattr(args, 'reranker_max_negatives', 5)
    if max_g > 1 + max_neg:
        print_rank_0(f' > WARNING: max group size on disk ({max_g}) exceeds 1 + '
                     f'--reranker-max-negatives ({1 + max_neg}); training still follows '
                     f'the on-disk group_sizes - check which --max-negatives was used '
                     f'at preprocess time.')
    print_rank_0(f' > building reranker datasets: {total_groups} groups '
                 f'(size min={min_g}, max={max_g}, mean={group_sizes.mean():.2f}) '
                 f'from {stream_prefix}')

    # splits_string: megatron 风格权重 "990,10" / "998,1,1" / "100,0,0"
    # valid/test 取各自比例的向下取整, 取整余数一律归 train —— 保证 valid/test
    # 永远不会被余数撑出"凑不满一个 DP 批"的迷你数据集(那种数据集过了 loader
    # 的空守卫, 却会在 eval 首批 StopIteration)
    parts = [float(x) for x in splits_string.split(',')]
    if len(parts) not in (2, 3):
        raise ValueError(f'unsupported --split for reranker: {splits_string}')
    total = sum(parts)
    n_valid = int(total_groups * parts[1] / total)
    n_test = int(total_groups * parts[2] / total) if len(parts) == 3 else 0
    n_train = total_groups - n_valid - n_test  # 余数归 train

    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_groups)

    def _wrap(indices):
        return RerankerDataset(group_indices=[int(i) for i in indices], ids_ds=ids_ds)

    splits = (
        _wrap(perm[:n_train]) if n_train > 0 else None,
        _wrap(perm[n_train:n_train + n_valid]) if n_valid > 0 else None,
        _wrap(perm[n_train + n_valid:]) if n_test > 0 else None,
    )
    print_rank_0(f' > reranker split: train={n_train}, valid={n_valid}, test={n_test} groups')
    # 防呆: single 型 eval dataloader 拉干即 StopIteration, 需求量必须 <= 对应 split 组数
    dp_world = mpu.get_data_parallel_world_size()
    num_microbatches = max(1, args.global_batch_size // (args.micro_batch_size * dp_world))
    eval_demand = (getattr(args, 'eval_iters', 0) or 0) * num_microbatches \
        * args.micro_batch_size * dp_world
    for name, n_groups in (('valid', n_valid), ('test', n_test)):
        if n_groups and eval_demand > n_groups:
            print_rank_0(f' > WARNING: {name} groups ({n_groups}) < eval demand ({eval_demand} = '
                         f'eval-iters x num_microbatches x micro-batch-size x DP); '
                         f'eval WILL hit StopIteration. Widen the --split fraction or '
                         f'lower --eval-iters.')
    return splits


def build_reranker_collator():
    args = get_args()
    tokenizer = get_tokenizer()
    pad_id = getattr(tokenizer, 'pad', None)
    if pad_id is None:
        pad_id = tokenizer.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos
    return RerankerDataCollator(
        pad_token_id=pad_id,
        seq_length=args.seq_length,
        pad_to_seq_length=not args.no_pad_to_seq_lengths,
    )
