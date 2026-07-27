"""modified based on megatron preprocess_data.py."""

import argparse
import json
import orjson
import os
import sys
import zstandard as zstd
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
from typing import Dict, List
from pathlib import Path

from megatron.core.datasets import indexed_dataset
from mindspeed_llm.training.tokenizer import build_tokenizer


DATA_DTYPE = np.int32
IGNORED_LABEL = -100
cur_file_dir = Path(__file__).absolute().parent
TEMPLATES_DIR = os.path.join(cur_file_dir, os.path.pardir, "configs/finetune/templates.json")


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    # LlamaFactory
    group.add_argument('--prompt-type', type=str, default=None,
                       choices=['default', 'empty', 'trl', 'chatglm2', 'chatglm3', 'chatglm3_system', 'glm4', 'chatml',
                                'chatml_de', 'qwen', 'qwen_r1', "qwen_math_r1", 'llama3', 'llama2', 'mistral', 'mixtral', 'gemma', 'alpaca',
                                'deepseek2', 'deepseek2-lite', 'cpm', 'baichuan2', 'deepseek3', 'intern2', 'hunyuan',
                                'ailab_slm', 'qwen3_hybrid'],
                       help='Which template to use for constructing prompts in training.'
                            'e.g., "qwen"')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default='PretrainedFromHF',
                       choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF', 'PanguSentencePieceTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach.'
                            'This value must be greater than the initial size of the tokenizer.'
                            ' If this argument is used the value of `make-vocab-size-divisible-by` '
                            'will be ignored.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
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
    group.add_argument('--trust-remote-code', action='store_true',
                       help='enable trust-remote-code for transformer to load model')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


class PretrainEncoder(object):


    def __init__(self, args):
        super().__init__()

        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        PretrainEncoder.tokenizer = build_tokenizer(self.args)
        PretrainEncoder.dtype = DATA_DTYPE

    def encode(self, json_line_bin):
        ids = {}
        lens = {}

        try:
            _sample = orjson.loads(json_line_bin)
        except Exception as e:
            return ids, lens, 0

        for key in self.args.json_keys:
            text = _sample[key]

            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []

            for sentence in sentences:
                sentence_ids = PretrainEncoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(PretrainEncoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ## [3]
            _doc_ids_np_bytes = np.array(doc_ids, dtype=PretrainEncoder.dtype).tobytes(order='C')

            ids[key] = _doc_ids_np_bytes
            lens[key] = sentence_lens

        return ids, lens, len(json_line_bin)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers


    @staticmethod
    def print_processing_stats(count, proc_start, total_bytes_processed, log_interval=1000):
        if count % log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def process_json_file(self, file_name):

        input_file_path, output_prefix, ext_name = file_name

        print("Opening", input_file_path, flush=True)
        fin = open(input_file_path, 'rb')
        if ext_name == ".zst":
            decompressor = zstd.ZstdDecompressor()
            stream_reader = decompressor.stream_reader(fin)
            ## ---------------------------------------------------------------------------
            ## decompressor.stream_reader will decode the stream to string, default utf-8
            ## io.TextIOWrapper(stream_reader) return string
            ## io.BufferedReader(stream_reader) return binary
            ## ---------------------------------------------------------------------------
            fin = io.BufferedReader(stream_reader)

        startup_start = time.time()
        encoder = PretrainEncoder(self.args)

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)

        ## doc here means the content of each line in jsonl
        if self.args.encode_docs_with_imap_unordered:
            ## [4] imap_unordered could be up to 1.6 faster than imap
            encoded_docs = pool.imap_unordered(encoder.encode, fin, self.args.chunk_size)
        else:
            encoded_docs = pool.imap(encoder.encode, fin, self.args.chunk_size)

        ## filter out empty content due to skipping non utf-8 data and format error
        encoded_docs = filter(lambda x: x[-1] > 0, encoded_docs)

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"

        for key in self.args.json_keys:
            output_bin_files[key] = f"{output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"

            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=DATA_DTYPE
            )

        startup_end = time.time()
        print("Time to startup:", startup_end - startup_start, flush=True)

        proc_start = time.time()
        total_bytes_processed = 0

        for i, (doc, doc_len, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in self.args.json_keys:
                ## -------------------------------------------------------------------------
                ## [1] transformation to torch.IntTensor and back is 5 times slower than
                ##     direct add_item or add_document
                ## [2] add_item is still 4-5 times slower than add_item_np_bytes
                ## -------------------------------------------------------------------------
                builders[key].add_document_np_bytes(doc[key], doc_len[key])

            self.print_processing_stats(i, proc_start, total_bytes_processed, log_interval=self.args.log_interval)

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

        fin.close()
        pool.close()
        pool.join()


def main():

    preprocess_tic = time.time()
    args = get_args()
    partition = Partition(args, args.workers)

    ####
    level = "document"
    input_paths = sorted(glob.glob(os.path.join(args.input, "*.json*")))
    if args.file_idx_start is not None and args.file_idx_end is not None:
        input_paths = input_paths[args.file_idx_start:args.file_idx_end]
    output_pres = []
    output_prefix = args.output_prefix

    if len(input_paths) > 1:
        for i, _ in enumerate(input_paths):
            output_pres.append(f"{output_prefix}_{i}")
    else:
        output_pres.append(output_prefix)

    ## zhh: sequential processing might be better
    # processes = []
    # for j, _input in enumerate(input_paths):
    #     p = multiprocessing.Process(target=partition.process_json_file,
    #                                 args=((_input,
    #                                        output_pres[j],
    #                                        "json"),)
    #                                 )
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()

    for j, _input in enumerate(input_paths):
        partition.process_json_file((_input, output_pres[j], "json"))
    ##

    if len(input_paths) == 1:
        return

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in args.json_keys:
        output_bin_files[key] = f"{output_prefix}_{key}_{level}.bin"
        output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"

        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=DATA_DTYPE
        )

        for _output in output_pres:
            full_output_prefix = f"{_output}_{key}_{level}"
            builders[key].add_index(full_output_prefix)
        builders[key].finalize(output_idx_files[key])

    if args.clean_unmerged_indexed_dataset:
        for key in args.json_keys:
            for _output in output_pres:
                os.remove(f"{_output}_{key}_{level}.bin")
                os.remove(f"{_output}_{key}_{level}.idx")

    print("Time to preprocess:", time.time() - preprocess_tic, flush=True)


if __name__ == '__main__':

    main()
