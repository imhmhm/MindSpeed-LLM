"""Reranker dataset + collator + provider (变长组版, 单流按比例切分或 per-split 多流).

磁盘格式(由 preprocess_data/preprocess_data_reranker.py 生成, 每条流):
    {output_prefix}_packed_input_ids_document.bin/.idx
一个 document = 一个 query 组(多 item): 组内 K_i 条 (query, doc) 的 prompt-only
序列连续存放, 首条为正样本; 组边界即 .idx 的 document_indices, 无需额外流。

数据来源两种(对齐 megatron 的 blend vs blend_per_split, 互斥), 均支持多流 blend
(组级配比: 权重省略时按各流组数, 即磁盘 document 数):
- Mode A: --data-path 单/多前缀 + --split 比例切分(每流独立按同分数切三段后,
  各段跨流混合);
- Mode B: --train/valid/test-data-path 各自单/多前缀, 每条流整体属于对应
  split, --split 不参与; 缺省的 split 返回 None, 由 loader 层跳过该 phase。
权重格式同 megatron get_blend_from_list: [w1, p1, w2, p2, ...] 交错或纯前缀。

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
    """Group-level view over one or more streams: item i = one group (variable size).

    多流按 argv 顺序串接: 全局组号经累计组偏移定位到流, 再用该流的
    document_indices 取 item 范围。
    """

    def __init__(self, streams: List, group_indices: Optional[List[int]] = None):
        # streams: [(IndexedDataset, 组数), ...]
        self._streams = [
            (ds, np.asarray(ds.index.document_indices, dtype=np.int64)) for ds, _ in streams
        ]
        self._offsets = np.cumsum([0] + [int(n) for _, n in streams])
        if group_indices is None:
            group_indices = list(range(int(self._offsets[-1])))
        self.group_indices = group_indices

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx) -> Dict[str, List[List[int]]]:
        gi = self.group_indices[idx]
        s = int(np.searchsorted(self._offsets, gi, side='right') - 1)
        ds, doc_indices = self._streams[s]
        local = gi - int(self._offsets[s])
        # document local 的 item 范围 = document_indices[local] : [local + 1]
        start, end = int(doc_indices[local]), int(doc_indices[local + 1])
        # slice 读取: 按 item 长度切分, 返回该组每条序列的数组列表
        return {'input_ids': [seq.tolist() for seq in ds[start:end]]}


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
    if total_groups == 0:
        raise ValueError(f'{stream_prefix} contains no query groups')
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


def _parse_blend(paths, name):
    """解析 --data-path(Mode A, name='data')或 --{name}-data-path(Mode B)的多流 blend。

    接受(与 megatron get_blend_from_list / wisemlops _parse_weights 同构):
    ['p'] / ['p1,p2'] / ['p1', 'p2', ...](纯前缀) / [w1, 'p1', w2, 'p2', ...]
    (交错权重)。返回 (prefixes, weights|None); 输入 None 返回 (None, None)。
    """
    if paths is None:
        return None, None
    flag = '--data-path' if name == 'data' else f'--{name}-data-path'
    if len(paths) == 1 and ',' in paths[0]:
        paths = paths[0].split(',')
    paths = [str(p).strip() for p in paths]
    if len(paths) % 2 == 1:
        try:
            float(paths[0])
        except (TypeError, ValueError):
            return paths, None
        raise ValueError(
            f'{flag} for reranker looks like weight/prefix pairs but has '
            f'an odd length; give a weight for every prefix '
            f'```[0.3, "p1", 0.4, "p2"]``` or none ```["p1", "p2"]```, got {paths}')
    weights = []
    for item in paths[0::2]:
        try:
            weights.append(float(item))
        except (TypeError, ValueError):
            weights.append(None)
    if all(w is None for w in weights):
        return paths, None
    if any(w is None for w in weights):
        raise ValueError(
            f'{flag} for reranker mixes weights and plain prefixes; give '
            f'a weight for every prefix or none of them, got {paths}')
    if min(weights) <= 0:
        raise ValueError(f'{flag} weights must be positive, got {paths}')
    return paths[1::2], weights


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


def _build_split(streams, local_indices, num_samples, name, seed, weights=None,
                 pre_shuffle=False, rngs=None):
    """把各流的(局部)组索引按配比混成一个 split 的 RerankerDataset。

    配比口径是组(= query 组, 磁盘上一个 document): weights=None 时按各流组数,
    否则用用户权重作组抽样概率; 份额 = 权重占比 x num_samples(最大余数法取整)。
    多流时每流按份额截取(需求量 < 单 epoch 时配比仍成立)并整体截到需求量;
    单流保持整 epoch 语义。每流独立 rng(缺省 RandomState(seed + 流序号))做
    epoch 填充 —— 增删流不影响其它流的序列; rngs 可显式传入已推进状态的实例
    (单流时与单数据集训练的采样序列逐位一致)。空组集的流跳过; 全空返回
    None(上层跳过该 phase)。pre_shuffle=True 时先整体洗牌各流索引(Mode B 的
    盘上原始序 -> 随机序); Mode A 的切分片已是洗牌结果, 传 False。多流 train
    在拼接后再整体洗牌一次, 保证 batch 内跨流混合。
    """
    kept = [(i, li) for i, li in enumerate(local_indices) if len(li) > 0]
    if not kept:
        return None
    counts = np.asarray([len(li) for _, li in kept], dtype=np.float64)
    if weights is None:
        w = counts
    else:
        w = np.asarray([weights[i] for i, _ in kept], dtype=np.float64)
    exact = num_samples * w / w.sum()
    shares = np.floor(exact).astype(np.int64)
    for k in np.argsort(-(exact - shares), kind='stable')[:int(num_samples - shares.sum())]:
        shares[k] += 1

    offsets = np.cumsum([0] + [int(streams[i][1]) for i, _ in kept])
    filled_global: List[int] = []
    for k, (i, li) in enumerate(kept):
        rng = rngs[i] if rngs is not None else np.random.RandomState(seed + i)
        local = list(map(int, li))
        if pre_shuffle:
            rng.shuffle(local)
        filled, epochs = _fill_epochs(local, int(shares[k]), rng, shuffle=(name == 'train'))
        if len(kept) > 1:
            filled = filled[:int(shares[k])]
        if epochs > 1:
            print_rank_0(f' > reranker {name}: stream {i} repeated {len(local)} groups '
                         f'x {epochs} epochs to meet its {int(shares[k])}-sample share')
        filled_global.extend(g + int(offsets[k]) for g in filled)
    if name == 'train' and len(kept) > 1:
        np.random.RandomState(seed).shuffle(filled_global)
    # 按构造 sum(shares) == num_samples 且每流恰好贡献自己的份额, 该截断恒不
    # 触发; 防御未来份额分配改动时静默超采
    if len(kept) > 1 and len(filled_global) > num_samples:
        filled_global = filled_global[:int(num_samples)]
    mode = 'by group counts' if weights is None else 'by weights'
    print_rank_0(f' > reranker {name}: blended {len(kept)} stream(s) ({mode}), '
                 f'{int(offsets[-1])} groups total, shares '
                 f'{"/".join(f"{int(shares[k])}" for k in range(len(kept)))} '
                 f'of {num_samples}-sample demand')
    return RerankerDataset(
        streams=[streams[i] for i, _ in kept], group_indices=filled_global)


def _split_fractions(splits_string):
    """校验 --split 并返回归一化前的 parts(2 或 3 段, 非负)。"""
    parts = [float(x) for x in splits_string.split(',')]
    if len(parts) not in (2, 3):
        raise ValueError(f'unsupported --split for reranker: {splits_string}')
    if min(parts) < 0:
        raise ValueError(f'--split fractions must be non-negative, got {splits_string}')
    return parts


def build_reranker_dataset(data_prefix, splits_string, seq_length, train_valid_test_num_samples, seed):
    """Provider entry: build train/valid/test RerankerDataset(s) from preprocessed streams.

    两种数据来源(对齐 megatron 的 blend vs blend_per_split, 互斥), 均支持多流:
    - Mode A: --data-path(单/多前缀, 可带交错权重) + --split 比例切分, 每流独立
      按同分数切三段后各段跨流混合;
    - Mode B: --train/valid/test-data-path(单/多前缀) 每条流整体属于对应
      split(等价 mcore blend_per_split 的直通), --split 不参与(传了报错),
      缺省的 split 返回 None 由 loader 层跳过该 phase。
    各 split 的组索引均按 train_valid_test_num_samples 需求量做 epoch 重复补满;
    多流配比口径是组数(权重省略时)或用户权重(组抽样概率)。
    """
    args = get_args()
    # --eval-on-start 会在训练前额外消耗一个 eval pass 的 valid 样本
    # (valid 迭代器是累计消费的), 需求量同步加上, 否则最后一次 eval 拉干
    if getattr(args, 'eval_on_start', False):
        train_valid_test_num_samples = list(train_valid_test_num_samples)
        train_valid_test_num_samples[1] += args.eval_iters * args.global_batch_size
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
        prefixes, weights = _parse_blend(data_prefix, 'data')
        parts = _split_fractions(splits_string)
        total = sum(parts)
        streams, carve, rngs = [], {'train': [], 'valid': [], 'test': []}, []
        for i, prefix in enumerate(prefixes):
            ids_ds, total_groups = _load_reranker_stream(prefix, 'train+valid+test')
            streams.append((ids_ds, total_groups))
            rng = np.random.RandomState(seed + i)
            perm = rng.permutation(total_groups)
            n_valid = int(total_groups * parts[1] / total)
            n_test = int(total_groups * parts[2] / total) if len(parts) == 3 else 0
            n_train = total_groups - n_valid - n_test  # 余数归 train
            carve['train'].append(perm[:n_train])
            carve['valid'].append(perm[n_train:n_train + n_valid])
            carve['test'].append(perm[n_train + n_valid:])
            rngs.append(rng)
            print_rank_0(f' > reranker split (stream {i}, {total_groups} groups): '
                         f'train={n_train}, valid={n_valid}, test={n_test}')
        return tuple(
            _build_split(streams, carve[name], train_valid_test_num_samples[i], name,
                         seed, weights, rngs=rngs)
            for i, name in enumerate(('train', 'valid', 'test')))

    # Mode B: 每条 per-split 流整体属于对应 split; train 首 epoch 洗牌(盘上是
    # 原始顺序, 对齐 Mode A 的 perm), 之后逐 epoch 重洗牌; valid/test 原序重复
    splits = []
    for name, paths, num_samples in zip(('train', 'valid', 'test'),
                                        per_split_paths, train_valid_test_num_samples):
        prefixes, weights = _parse_blend(paths, name)
        if prefixes is None:
            splits.append(None)
            continue
        streams = [_load_reranker_stream(prefix, name) for prefix in prefixes]
        local = [np.arange(total_groups) for _, total_groups in streams]
        splits.append(_build_split(streams, local, num_samples, name, seed, weights,
                                   pre_shuffle=(name == 'train')))
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
