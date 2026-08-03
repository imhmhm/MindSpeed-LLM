"""Shared building blocks for the preprocess_data_jsonl_* scripts.

Each script keeps its own Encoder and a thin Partition subclass; this module
owns the parts that used to be copy-pasted across scripts:
  - the common argparse groups + post-parse fixups
  - the Partition skeleton (open input, worker pool, builders, finalize, merge)
  - the main() skeleton (input path expansion, per-file processing, merge)
"""

import io
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

from megatron.core.datasets import indexed_dataset

from mindspeed_llm.features_manager.tokenizer.build_tokenizer import PROMPT_TYPE_CHOICES
from preprocess_data.utils import get_input_paths


DATA_DTYPE = np.int32
IGNORED_LABEL = -100

cur_file_dir = Path(__file__).absolute().parent
TEMPLATES_DIR = os.path.join(cur_file_dir, os.path.pardir, "configs/finetune/templates.json")


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------

def add_preprocess_common_args(parser):
    """
    common args of preprocessing pretrain/sft/rl data
    """
    # ===== shared args with mindspeed-llm ===== #

    group = parser.add_argument_group(title='features_manager.BuildTokenizerFeature')
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    ## build_tokenizer (tokenizer.py:56) reads args.prompt_type even if it is not required in pretrain
    group.add_argument('--prompt-type', type=str, default=None,
                       choices=PROMPT_TYPE_CHOICES,  # 对齐 features_manager.BuildTokenizerFeature，避免 choices 漂移
                       help='Which template to use for constructing prompts in training.'
                            'e.g., "qwen"')

    group = parser.add_argument_group(title='features_manager.DatasetPreprocessFeature')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach.'
                            'This value must be greater than the initial size of the tokenizer.'
                            ' If this argument is used the value of `make-vocab-size-divisible-by` '
                            'will be ignored.')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))

    # ===== shared args with megatron ===== #

    group = parser.add_argument_group(title='megatron args')
    # --tokenizer-type 不带 choices：实际可选值由 features_manager.BuildTokenizerFeature
    # （megatron core + features 追加）在运行时校验，这里只在 help 里注明来源
    group.add_argument('--tokenizer-type', type=str, default='PretrainedFromHF',
                       help='What type of tokenizer to use. choices: features_manager.BuildTokenizerFeature (megatron core + features append)')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')

    # ===== args specific for preprocess_data ===== #

    group = parser.add_argument_group(title='preprocess_data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--chunk-size', type=int, default=32,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--encode-docs-with-imap-unordered', action='store_true',
                       help='speedup json lines encoding with imap_unordered')
    group.add_argument('--file-idx-start', type=int, default=None,
                       help='select from the list of input files to build a partition of the total')
    group.add_argument('--file-idx-end', type=int, default=None,
                       help='select from the list of input files to build a partition of the total')
    group.add_argument('--clean-unmerged-indexed-dataset', action='store_true',
                       help='if input path contains multiple files, clean each indexed dataset after merging them')

    # ===== args specific for parquet input ===== #

    group = parser.add_argument_group(title='parquet')
    group.add_argument('--parquet-batch-size', type=int, default=4096,
                       help='row batch size when streaming parquet files (bounds read memory; not used for jsonl)')


def add_posttrain_args(parser):
    """
    additional args of preprocessing sft/rl data
    """
    # ===== shared args with mindspeed-llm ===== #

    group = parser.add_argument_group(title='features_manager.BuildTokenizerFeature')
    group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                       help='Path to the json file of templates.')

    group = parser.add_argument_group(title='features_manager.DatasetPreprocessFeature')
    group.add_argument("--map-keys", type=json.loads, default=None,
                       help="Dataset field mapping.")
    group.add_argument('--enable-thinking',
                       type=lambda x: {"true": True, "false": False, "none": None}[x.lower()],
                       default=None,
                       help='enable_thinking in prompt template (true/false/none; only takes effect for ReasoningTemplate)')

    group = parser.add_argument_group(title='features_manager.FinetuneFeature')
    group.add_argument('--dataset-additional-keys',
                       nargs='*',
                       default=[],
                       help='Additional keys need to be add from dataset.')

    # ===== args specific for preprocess_data ===== #

    group = parser.add_argument_group(title='preprocess_data')
    group.add_argument('--convert-method', type=str, required=True,
                       help='method in `convert_methods` for converting dataset to intermediate representation')


def finalize_args(args):
    args.keep_empty = False
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    return args


# ---------------------------------------------------------------------------
# Partition skeleton
# ---------------------------------------------------------------------------

class BasePartition(object):
    """Shared per-file processing + multi-file merge skeleton.

    Subclasses set:
      - PACKED_SUFFIX: appended to per-file output prefix; "_packed" marks a
        multi-column packed dataset (NOT sequence packing). pretrain uses "".
      - encoder_cls: the Encoder class used in the worker pool.
      - _consume_encoded: how to drain the encoded-doc stream into builders.
    """

    PACKED_SUFFIX = "_packed"
    encoder_cls = None

    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    @staticmethod
    def print_processing_stats(count, proc_start, total_bytes_processed, total_bytes_tokenized, log_interval=1000):
        if count % log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024 if total_bytes_processed else "-"
            mbs_token = total_bytes_tokenized / elapsed / 1024 / 1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, [bytes: {mbs} MB/s] [tokens: {mbs_token} MB/s]).",
                  file=sys.stderr)

    def _make_builders(self, output_prefix):
        ## each json-key corresponds to an indexed dataset builder
        output_idx_files = {}
        builders = {}
        level = "document"
        for key in self.args.json_keys:
            output_bin_file = f"{output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_file, dtype=DATA_DTYPE
            )
        return builders, output_idx_files

    def _consume_encoded(self, encoded_docs, builders, proc_start):
        ## implemented by subclasses
        raise NotImplementedError

    def _parquet_columns(self):
        ## map-keys(posttrain) / json-keys(pretrain) / dataset-additional-keys
        cols = set()
        map_keys = getattr(self.args, 'map_keys', None)
        if map_keys:
            cols.update(v for v in map_keys.values() if v)
        else:
            ## no map-keys: pretrain (use json-keys)
            cols.update(self.args.json_keys)
        for k in getattr(self.args, 'dataset_additional_keys', []) or []:
            ## "labels" excluded here for RL
            if k and k != "labels":
                cols.add(k)
        return sorted(cols) if cols else None

    def _parquet_row_iter(self, input_file_path):
        import pyarrow.parquet as pq
        columns = self._parquet_columns()
        pf = pq.ParquetFile(input_file_path)
        for batch in pf.iter_batches(batch_size=self.args.parquet_batch_size, columns=columns):
            ## to_pylist convert a batch to list[dict]
            for sample in batch.to_pylist():
                yield sample

    def process_file(self, input_file_path, output_prefix):
        ## "_packed" marks a multi-column packed dataset (NOT sequence packing)
        output_prefix = output_prefix + self.PACKED_SUFFIX

        print("Opening", input_file_path, flush=True)

        startup_start = time.time()
        encoder = self.encoder_cls(self.args)

        if input_file_path.endswith(('.parquet', '.parq')):
            row_iter = self._parquet_row_iter(input_file_path)
            fin = None
        else:
            fin = open(input_file_path, 'rb')
            ## jsonl.zst: stream_reader -> str(utf-8), io.BufferedReader -> binary
            if input_file_path.endswith('.zst'):
                decompressor = zstd.ZstdDecompressor()
                fin = io.BufferedReader(decompressor.stream_reader(fin))
            row_iter = fin

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        ## imap_unordered can be up to 1.6x faster than imap
        if self.args.encode_docs_with_imap_unordered:
            encoded_docs = pool.imap_unordered(encoder.encode, row_iter, self.args.chunk_size)
        else:
            encoded_docs = pool.imap(encoder.encode, row_iter, self.args.chunk_size)

        ## filter out empty content from non utf-8 data and format errors
        encoded_docs = filter(lambda x: x[-1] > 0, encoded_docs)

        builders, output_idx_files = self._make_builders(output_prefix)

        print("Time to startup:", time.time() - startup_start, flush=True)

        proc_start = time.time()
        self._consume_encoded(encoded_docs, builders, proc_start)

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

        if fin is not None:
            fin.close()
        pool.close()
        pool.join()

    def run(self):
        preprocess_tic = time.time()
        args = self.args
        level = "document"

        input_paths = get_input_paths(args.input)
        if args.file_idx_start is not None and args.file_idx_end is not None:
            input_paths = input_paths[args.file_idx_start:args.file_idx_end]

        output_prefix = args.output_prefix
        if len(input_paths) > 1:
            output_pres = [f"{output_prefix}_{i}" for i in range(len(input_paths))]
        else:
            output_pres = [output_prefix]

        ## multiple data files: sequential processing is stable in webstudio
        for j, _input in enumerate(input_paths):
            self.process_file(_input, output_pres[j])

        if len(input_paths) == 1:
            return

        suffix = self.PACKED_SUFFIX
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.json_keys:
            output_bin_files[key] = f"{output_prefix}{suffix}_{key}_{level}.bin"
            output_idx_files[key] = f"{output_prefix}{suffix}_{key}_{level}.idx"
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key], dtype=DATA_DTYPE
            )
            ## multiple data files: merge all indexed datasets
            for _output in output_pres:
                full_output_prefix = f"{_output}{suffix}_{key}_{level}"
                builders[key].add_index(full_output_prefix)
            builders[key].finalize(output_idx_files[key])

        if args.clean_unmerged_indexed_dataset:
            for key in args.json_keys:
                for _output in output_pres:
                    os.remove(f"{_output}{suffix}_{key}_{level}.bin")
                    os.remove(f"{_output}{suffix}_{key}_{level}.idx")

        print("Time to preprocess:", time.time() - preprocess_tic, flush=True)
