"""Reranker dataset + collator + provider (变长组版, 单流按比例切分或 per-split 多流).

磁盘格式(由 preprocess_data/preprocess_data_reranker.py 生成, 每条流):
    {output_prefix}_packed_input_ids_document.bin/.idx
一个 document = 一个 query 组(多 item): 组内 K_i 条 (query, doc) 的 prompt-only
序列连续存放, 首条为正样本; 组边界即 .idx 的 document_indices, 无需额外流。

数据来源两种(对齐 megatron 的 blend vs blend_per_split, 互斥):
- Mode A: --data-path 单前缀 + --split 比例切分(单流切三段);
- Mode B: --train/valid/test-data-path 各自一条独立流, 整条流属于对应 split,
  --split 不参与; 缺省的 split 返回 None, 由 loader 层跳过该 phase。

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

from megatron.core.datasets import indexed_dataset
from megatron.training import get_args, get_tokenizer, print_rank_0


def _indexed_dataset_exists(path_prefix: str) -> bool:
    return (os.path.exists(indexed_dataset.get_idx_path(path_prefix))
            and os.path.exists(indexed_dataset.get_bin_path(path_prefix)))


class RerankerDataset(torch.utils.data.Dataset):
    """Group-level view over one stream: item i = the i-th group (variable size)."""

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


def _load_reranker_stream(prefix: str, name: str):
    """打开一条预处理流并做数据契约校验, 返回 (IndexedDataset, 组数)。

    文件名约定与 SFT 一致(mtf_dataset.get_packed_indexed_dataset):
    IndexedDataset(f"{prefix}_packed_input_ids_document"), 传纯前缀。
    数据契约校验(计算语义不依赖校验值 —— loss 只认盘上的 group_sizes):
    1) 组 >= 2 条序列是 listwise CE 的数学要求(单条序列 softmax 恒为 1, loss
       恒为 0, 白耗算力), 违反说明数据不是用 --min-group-size>=2 预处理的,
       硬报错;
    2) 盘上最大组 <= 1 + --reranker-max-negatives, 不等说明预处理参数与训练
       认知不一致(训练仍按盘上 group_sizes 跑, 仅告警提示核对)。
    """
    args = get_args()
    stream_prefix = f'{prefix}_packed_input_ids_document'
    if not _indexed_dataset_exists(stream_prefix):
        raise FileNotFoundError(
            f'{indexed_dataset.get_idx_path(stream_prefix)} / .bin not found. '
            f'preprocess_data_reranker.py writes {{output_prefix}}_packed_input_ids_document'
            f'.bin/.idx; pass the plain --output-prefix via --data-path or '
            f'--train/valid/test-data-path')

    ids_ds = indexed_dataset.IndexedDataset(stream_prefix)
    total_groups = len(ids_ds.index.document_indices) - 1
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
    print_rank_0(f' > building reranker {name} dataset: {total_groups} groups '
                 f'(size min={min_g}, max={max_g}, mean={group_sizes.mean():.2f}) '
                 f'from {stream_prefix}')
    return ids_ds, total_groups


def _parse_split_prefix(paths, name):
    """解析 --{name}-data-path; None 返回 None; v1 只接受单个前缀(不支持 blend)。"""
    if paths is None:
        return None
    if len(paths) == 1 and ',' in paths[0]:
        paths = paths[0].split(',')
    if len(paths) != 1:
        raise ValueError(
            f'--{name}-data-path for reranker accepts a single plain prefix; weighted '
            f'or multi-dataset blends are not supported yet, got {paths}')
    return paths[0]


def _fill_epochs(indices, num_samples, rng, shuffle):
    """epoch 填充(对齐 mcore GPTDataset 的 index 语义): megatron 训练循环按
    train_iters x gbs 拉取, single 型迭代器不会回绕, 各 split 必须供满需求量,
    否则数据不足时恰在 epoch 边界 StopIteration。首 epoch 原样使用(调用方决定
    是否预洗牌), 之后每 epoch 重洗牌拼接; valid/test 原序重复(全量评估,
    顺序无关)。num_samples 口径即 get_train_valid_test_num_samples:
    train = train_iters x gbs; valid = 全程累计 eval 消耗; test = 单次 eval 消耗。
    """
    filled = [int(i) for i in indices]
    epochs = 1
    while len(filled) < num_samples and len(indices) > 0:
        epoch = indices.copy()
        if shuffle:
            rng.shuffle(epoch)
        filled.extend(int(i) for i in epoch)
        epochs += 1
    return filled, epochs


def _wrap_split(ids_ds, indices, num_samples, name, rng):
    """按需求量填满一个 split 并包成 RerankerDataset; 空组集返回 None(跳过该 phase)。"""
    if len(indices) == 0:
        return None
    filled, epochs = _fill_epochs(indices, num_samples, rng, shuffle=(name == 'train'))
    if epochs > 1:
        print_rank_0(f' > reranker {name}: repeated {len(indices)} groups x {epochs} '
                     f'epochs to meet the {num_samples}-sample demand')
    return RerankerDataset(group_indices=filled, ids_ds=ids_ds)


def _split_by_fraction(ids_ds, total_groups, splits_string, train_valid_test_num_samples, seed):
    """Mode A: 单流按 --split 权重切三段, seeded permutation 全 rank 确定一致。"""
    parts = [float(x) for x in splits_string.split(',')]
    if len(parts) not in (2, 3):
        raise ValueError(f'unsupported --split for reranker: {splits_string}')
    total = sum(parts)
    n_valid = int(total_groups * parts[1] / total)
    n_test = int(total_groups * parts[2] / total) if len(parts) == 3 else 0
    n_train = total_groups - n_valid - n_test  # 余数归 train

    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_groups)
    print_rank_0(f' > reranker split: train={n_train}, valid={n_valid}, test={n_test} groups')
    return (
        _wrap_split(ids_ds, perm[:n_train], train_valid_test_num_samples[0], 'train', rng),
        _wrap_split(ids_ds, perm[n_train:n_train + n_valid], train_valid_test_num_samples[1], 'valid', rng),
        _wrap_split(ids_ds, perm[n_train + n_valid:], train_valid_test_num_samples[2], 'test', rng),
    )


def build_reranker_dataset(data_prefix, splits_string, seq_length, train_valid_test_num_samples, seed):
    """Provider entry: build train/valid/test RerankerDataset(s) from preprocessed streams.

    两种数据来源(对齐 megatron 的 blend vs blend_per_split, 互斥):
    - Mode A: --data-path 单前缀 + --split 比例切分(单流切三段);
    - Mode B: --train/valid/test-data-path 每条流整体属于对应 split(等价 mcore
      blend_per_split 单前缀的 (0.0, 1.0) 直通), --split 不参与(传了报错),
      缺省的 split 返回 None 由 loader 层跳过该 phase。
    各 split 的组索引均按 train_valid_test_num_samples 需求量做 epoch 重复补满。
    """
    args = get_args()
    per_split_paths = (getattr(args, 'train_data_path', None),
                       getattr(args, 'valid_data_path', None),
                       getattr(args, 'test_data_path', None))
    has_data_path = bool(data_prefix)
    has_per_split = any(p is not None for p in per_split_paths)
    if has_data_path and has_per_split:
        raise ValueError(
            '--data-path and --train/valid/test-data-path are mutually exclusive '
            '(megatron allows a single data source per run)')
    if not has_data_path and not has_per_split:
        raise ValueError('no reranker data source: pass either --data-path or '
                         '--train/valid/test-data-path')
    if has_per_split and splits_string is not None:
        raise ValueError(
            f'--split {splits_string!r} is incompatible with --train/valid/test-data-path: '
            f'each stream belongs entirely to its split (megatron blend_per_split '
            f'semantics), fractions do not apply')

    if has_data_path:
        if len(data_prefix) == 1:
            data_prefix = data_prefix[0].split(',')
        if len(data_prefix) != 1:
            raise ValueError(
                f'reranker stage currently supports a single --data-path prefix, got {data_prefix}. '
                f'Multiple/weighted blends: 分别预处理后训练时逐个传入')
        ids_ds, total_groups = _load_reranker_stream(data_prefix[0], 'train+valid+test')
        return _split_by_fraction(ids_ds, total_groups, splits_string,
                                  train_valid_test_num_samples, seed)

    # Mode B: 每条 per-split 流整体属于对应 split; train 首 epoch 洗牌(盘上是
    # 原始顺序, 对齐 Mode A 的 perm), 之后逐 epoch 重洗牌; valid/test 原序重复
    rng = np.random.RandomState(seed)
    splits = []
    for name, paths, num_samples in zip(('train', 'valid', 'test'),
                                        per_split_paths, train_valid_test_num_samples):
        prefix = _parse_split_prefix(paths, name)
        if prefix is None:
            splits.append(None)
            continue
        ids_ds, total_groups = _load_reranker_stream(prefix, name)
        indices = np.arange(total_groups)
        if name == 'train':
            rng.shuffle(indices)
        splits.append(_wrap_split(ids_ds, indices, num_samples, name, rng))
    return tuple(splits)


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
