# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import logging
import math
import os
import time

import numpy as np
import torch

from megatron.training import print_rank_0, get_args
from megatron.core.datasets.utils import get_blend_from_list
from mindspeed_llm.core.datasets.blended_megatron_dataset_builder import need_to_build_dataset
from mindspeed_llm.training.tokenizer import build_tokenizer
from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import (
    build_train_valid_test_datasets as build_single_mtf_dataset,
    _build_train_valid_test_datasets,
)
from mindspeed_llm.tasks.preprocess.mtf_dataset import get_packed_indexed_dataset

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ['train', 'valid', 'test']

## 与 megatron BlendedMegatronDatasetBuilder 保持一致: 子集额外多建 0.5% 的样本,
## 用来吸收 blend 路由的取整偏差 -- build_blending_indices 只按权重贪心分配, 不做容量钳位
_BUFFER_MARGIN = 0.5

## dataset_index 以 int16 存储, 与 megatron BlendedDataset 一致
_MAX_NUM_DATASETS = 32767


def _get_size_per_split_per_dataset(normalized_weights, target_size_per_split, margin=0.0):
    """Determine the contribution of each sub dataset to each blended split.

    Mirrors megatron.core.datasets.blended_megatron_dataset_builder._get_size_per_split_per_dataset.

    Args:
        normalized_weights: per sub dataset weights summing to 1.0, e.g. [0.3, 0.7]

        target_size_per_split: the number of samples to target for each blended split

        margin (float): the relative quantity of extra samples to build per split per
            dataset, as a percentage

    Returns:
        The number of samples to request per sub dataset per split, indexed [dataset][split]
    """
    if not np.isclose(sum(normalized_weights), 1.0):
        raise ValueError(f'normalized weights must sum to 1.0, got {sum(normalized_weights)}')

    return [
        [
            int(math.ceil(math.ceil(target_size * weight) * (1 + margin / 100)))
            for target_size in target_size_per_split
        ]
        for weight in normalized_weights
    ]


def _get_num_documents(data_prefix):
    """Number of documents in a packed indexed dataset, used as the default weight."""
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix)
    if not packed_indexed_dataset:
        raise ValueError(f'no packed indexed dataset found for prefix {data_prefix}')
    return len(list(packed_indexed_dataset.values())[0])


def _resolve_paths(data_prefix):
    """Normalize a --*-data-path value into (paths, weights)."""
    if len(data_prefix) == 1:
        data_prefix = data_prefix[0].split(',')
    return get_blend_from_list(data_prefix)


def _get_field_set(data_prefix):
    """Union of the packed field names (input_ids, labels, ...) behind one blend."""
    paths, _ = _resolve_paths(data_prefix)
    fields = set()
    for path in paths:
        packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=path)
        if not packed_indexed_dataset:
            raise ValueError(f'no packed indexed dataset found for prefix {path}')
        fields.update(packed_indexed_dataset.keys())
    return fields


## one-hot --split per phase: get_train_valid_test_split_ then hands every document of
## the stream to that phase and leaves the other two empty, which is what blend_per_split
## means -- a stream belongs entirely to its split
_ONE_HOT_SPLITS = ('1,0,0', '0,1,0', '0,0,1')


def build_blended_mtf_dataset(
    data_prefix,
    splits_string,
    seq_length: int,
    train_valid_test_num_samples,
    seed,
):
    """Build train, valid and test instruction datasets from one or three data sources.

    Two mutually exclusive modes, mirroring megatron blend vs blend_per_split (the
    mutual exclusion is already enforced by GPTDatasetConfig.__post_init__, which runs
    before this provider):
    - Mode A: --data-path, one blend carved into three phases by --split;
    - Mode B: --train/valid/test-data-path, each an independent blend owned entirely by
      its phase; --split does not apply and an omitted phase yields None.
    """
    args = get_args()
    per_split_paths = (args.train_data_path, args.valid_data_path, args.test_data_path)
    if any(paths is not None for paths in per_split_paths):
        return _build_per_split_blends(per_split_paths, seq_length,
                                       train_valid_test_num_samples, seed)
    return _build_blend(data_prefix, splits_string, seq_length,
                        train_valid_test_num_samples, seed)


def _build_per_split_blends(per_split_paths, seq_length, train_valid_test_num_samples, seed):
    """Mode B: build each phase from its own blend, keeping the other two empty."""
    ## resolve every stream before building anything: the streams are preprocessed
    ## independently, so a missing prefix or a field one of them lacks would otherwise
    ## surface far away (an IndexError while indexing, a KeyError in the collator)
    field_sets = {
        _SPLIT_NAMES[index]: _get_field_set(paths)
        for index, paths in enumerate(per_split_paths) if paths is not None
    }
    if len({frozenset(fields) for fields in field_sets.values()}) > 1:
        raise ValueError(
            f'per-split streams carry different packed fields: '
            f'{ {name: sorted(fields) for name, fields in field_sets.items()} }. '
            f'Re-run preprocess_data.py with the same --prompt-type / handler for every split')

    datasets = []
    for index, paths in enumerate(per_split_paths):
        if paths is None:
            datasets.append(None)
            continue
        ## only this phase asks for samples; the other two carve zero documents anyway
        sizes = [0, 0, 0]
        sizes[index] = train_valid_test_num_samples[index]
        splits = _build_blend(paths, _ONE_HOT_SPLITS[index], seq_length, sizes, seed)
        datasets.append(splits[index])

    return tuple(datasets)


def _build_blend(
    data_prefix,
    splits_string,
    seq_length: int,
    train_valid_test_num_samples,
    seed,
):
    """Build blended train, valid and test instruction datasets from one data source.

    Sampling volume is decided here (each sub dataset is built with its own quota) and
    realized by DecoderPackedMTFDataset, which repeats or truncates its document epochs to
    reach the requested number of samples. BlendedMTFDataset itself only routes.
    """
    args = get_args()

    paths, weights = _resolve_paths(data_prefix)

    ## 单数据集且未指定权重时直接走原有单集路径, 索引缓存与行为完全不变
    if len(paths) == 1 and weights is None:
        return build_single_mtf_dataset(
            data_prefix=paths,
            splits_string=splits_string,
            seq_length=seq_length,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed,
        )

    if len(paths) > _MAX_NUM_DATASETS:
        raise ValueError(f'at most {_MAX_NUM_DATASETS} sub datasets are supported, got {len(paths)}')

    ## megatron 在未给权重时只建一个 epoch 并按长度配比; 指令微调的训练量由 num_samples
    ## 决定而非文档数, 所以这里退化成按文档数配比, 再走与显式权重相同的配额链路
    if weights is None:
        weights = [_get_num_documents(path) for path in paths]

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError(f'blended dataset weights must sum to a positive value, got {total_weight}')
    normalized_weights = [weight / total_weight for weight in weights]

    # The number of samples we plan to use per sub dataset
    sizes_target = _get_size_per_split_per_dataset(normalized_weights, train_valid_test_num_samples)
    # The number of samples we plan to build per sub dataset
    sizes_buffer = _get_size_per_split_per_dataset(
        normalized_weights, train_valid_test_num_samples, margin=_BUFFER_MARGIN
    )

    print_rank_0(f' > blending {len(paths)} instruction datasets:')
    for path, weight, sizes in zip(paths, normalized_weights, sizes_buffer):
        print_rank_0(f'    {path}: weight {weight:.6f}, samples to build {sizes}')

    tokenizer = build_tokenizer(args)
    pad_token = tokenizer.pad
    eos_token = tokenizer.eos

    datasets_per_split = [[] for _ in _SPLIT_NAMES]
    for index, path in enumerate(paths):
        splits = _build_train_valid_test_datasets(
            data_prefix=path,
            splits_string=splits_string,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=sizes_buffer[index],
            seed=seed,
        )
        for split_index in range(len(_SPLIT_NAMES)):
            datasets_per_split[split_index].append(splits[split_index])

    path_to_cache = args.data_cache_path or os.path.dirname(paths[0])

    return tuple(
        _build_blended_split(
            datasets=datasets_per_split[split_index],
            normalized_weights=normalized_weights,
            sizes=[sizes_target[index][split_index] for index in range(len(paths))],
            split_name=_SPLIT_NAMES[split_index],
            path_to_cache=path_to_cache,
        )
        for split_index in range(len(_SPLIT_NAMES))
    )


def _build_blended_split(datasets, normalized_weights, sizes, split_name, path_to_cache):
    """Blend one split, dropping the sub datasets whose split is empty."""
    kept = [
        (dataset, weight, size)
        for dataset, weight, size in zip(datasets, normalized_weights, sizes)
        if dataset is not None
    ]
    if not kept:
        return None
    if len(kept) < len(datasets):
        print_rank_0(f' > WARNING: {len(datasets) - len(kept)} of {len(datasets)} sub datasets have '
                     f'an empty {split_name} split and are excluded from the blend')

    total_weight = sum(weight for _, weight, _ in kept)
    return BlendedMTFDataset(
        datasets=[dataset for dataset, _, _ in kept],
        normalized_weights=[weight / total_weight for _, weight, _ in kept],
        size=sum(size for _, _, size in kept),
        split_name=split_name,
        path_to_cache=path_to_cache,
    )


class BlendedMTFDataset(torch.utils.data.Dataset):
    """Route a global sample index to a (sub dataset, sub dataset sample) pair.

    This class decides no quantity: the caller has already sized every sub dataset so that
    its length covers the share implied by its weight.
    """

    def __init__(self, datasets, normalized_weights, size, split_name, path_to_cache):
        if len(datasets) != len(normalized_weights):
            raise ValueError(f'got {len(datasets)} datasets but {len(normalized_weights)} weights')

        self.datasets = datasets
        self.weights = np.array(normalized_weights, dtype=np.float64)
        self.size = int(size)
        self.split_name = split_name
        self.path_to_cache = path_to_cache

        self.unique_description = json.dumps(
            {
                'class': type(self).__name__,
                'split': split_name,
                'datasets': [len(dataset) for dataset in datasets],
                'weights': normalized_weights,
                'size': self.size,
            },
            indent=4,
            default=str,
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode('utf-8')
        ).hexdigest()

        self.dataset_index, self.dataset_sample_index = self._build_indices()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = idx % self.size
        dataset_id = int(self.dataset_index[idx])
        dataset_sample_id = int(self.dataset_sample_index[idx])
        return {
            'dataset_id': dataset_id,
            **self.datasets[dataset_id][dataset_sample_id],
        }

    def _build_indices(self):
        def get_path_to(suffix):
            return os.path.join(
                self.path_to_cache,
                f'{self.unique_description_hash}-{type(self).__name__}-{self.split_name}-{suffix}',
            )

        path_to_description = get_path_to('description.txt')
        path_to_dataset_index = get_path_to('dataset_index.npy')
        path_to_dataset_sample_index = get_path_to('dataset_sample_index.npy')
        cache_hit = all(
            map(
                os.path.isfile,
                [path_to_description, path_to_dataset_index, path_to_dataset_sample_index],
            )
        )

        if not cache_hit and need_to_build_dataset():
            print_rank_0(' > WARNING: could not find blended index map files, building '
                         'the indices ...')
            start_time = time.time()

            from megatron.core.datasets import helpers

            dataset_index = np.zeros(self.size, dtype=np.int16)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)
            helpers.build_blending_indices(
                dataset_index,
                dataset_sample_index,
                self.weights,
                len(self.datasets),
                self.size,
                False,
            )
            self._check_capacity(dataset_index, dataset_sample_index)

            os.makedirs(self.path_to_cache, exist_ok=True)
            with open(path_to_description, 'wt') as writer:
                writer.write(self.unique_description)
            np.save(path_to_dataset_index, dataset_index, allow_pickle=True)
            np.save(path_to_dataset_sample_index, dataset_sample_index, allow_pickle=True)
            print_rank_0(f' > elasped time to build and save blended index mapping'
                         f' (seconds): {time.time() - start_time:4f}')

        torch.distributed.barrier()

        start_time = time.time()
        print_rank_0(f' > loading blended index mapping from {path_to_dataset_index}')
        dataset_index = np.load(path_to_dataset_index, allow_pickle=True, mmap_mode='r')
        dataset_sample_index = np.load(path_to_dataset_sample_index, allow_pickle=True, mmap_mode='r')
        print_rank_0(f'    loaded indexed file in {time.time() - start_time:3.3f} seconds')

        return dataset_index, dataset_sample_index

    def _check_capacity(self, dataset_index, dataset_sample_index):
        """Verify that no sub dataset is asked for more samples than it holds.

        build_blending_indices allocates purely by weight and never clamps to the dataset
        length, so an unnoticed overflow would silently read out of range.
        """
        for index, dataset in enumerate(self.datasets):
            requested = int(dataset_sample_index[dataset_index == index].max(initial=-1)) + 1
            if requested > len(dataset):
                raise IndexError(
                    f'blended {self.split_name} split requests {requested} samples from sub '
                    f'dataset {index} which holds only {len(dataset)}'
                )
