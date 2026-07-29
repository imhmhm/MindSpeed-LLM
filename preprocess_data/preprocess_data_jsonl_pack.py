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
from collections import defaultdict

from megatron.core.datasets import indexed_dataset

from mindspeed_llm.tasks.preprocess.parser import InstructionDatasetAttr
from mindspeed_llm.tasks.preprocess.templates import get_model_template
from mindspeed_llm.tasks.preprocess.utils import convert_sharegpt_to_intermediate, greedy_knapsack, random_packing
from mindspeed_llm.training.tokenizer import build_tokenizer

from preprocess_data.convert_methods import InstructionDatasetAttr
from preprocess_data.convert_methods import DATASET_CONVERT_METHODS

DATA_DTYPE = np.int32
IGNORED_LABEL = -100
cur_file_dir = Path(__file__).absolute().parent
TEMPLATES_DIR = os.path.join(cur_file_dir, os.path.pardir, "configs/finetune/templates.json")


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--convert-method', type=str, required=True,
                       help='method in `convert_methods` for converting dataset to intermediate representation')
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
    group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                       help='Path to the json file of templates.')
    group.add_argument("--map-keys", type=json.loads, default=None,
                       help="Dataset field mapping.")
    group.add_argument('--dataset-additional-keys',
                       nargs='*',
                       default=[],
                       help='Additional keys need to be add from dataset.'
                       )
    group.add_argument("--neat-pack", action='store_true',
                       help="Use a zigzag attention mask.")
    group.add_argument("--knapsack", action='store_true',
                       help="Use greedy_knapsack to pack sequences.")

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default='PretrainedFromHF',
                       choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF', 'AILabSentencePieceTokenizer'],
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
    group.add_argument('--enable-thinking', action='store_true',
                       help='enable_thinking in prompt template')
    group.add_argument('--trust-remote-code', action='store_true',
                       help='enable trust-remote-code for transformer to load model')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


class LlamaFactoryInstructionEncoder(object):


    def __init__(self, args):
        super().__init__()

        self.args = args

        self.llama_factory_template = get_model_template(
            args.prompt_type.strip(), args.prompt_type_path.strip(),
            enable_thinking=args.enable_thinking
        )

        self.ignored_label = IGNORED_LABEL
        self.train_on_inputs = False

        self.dataset_attr = InstructionDatasetAttr("file", dataset_name="SharegptStyleInstructionHandler")
        self.dataset_attr.dataset_additional_keys = self.args.dataset_additional_keys
        self.dataset_attr.formatting = "sharegpt"
        # attr_names = [
        #     "messages", "system", "tools", "chosen", "rejected",
        #     "role_tag", "content_tag", "user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"
        # ]
        ## zhh: keys are not verified
        if args.map_keys is not None:
            for column_name, target_name in args.map_keys.items():
                setattr(self.dataset_attr, column_name, target_name)
        self.convert_dataset_to_intermediate = DATASET_CONVERT_METHODS[args.convert_method]

    def initializer(self):
        # Use Encoder class as a container for global data
        # Redirect the stdout to avoid printing logs during loading tokenizer
        ori_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        LlamaFactoryInstructionEncoder.tokenizer = build_tokenizer(self.args)
        sys.stdout.close()
        sys.stdout = ori_stdout
        LlamaFactoryInstructionEncoder.dtype = DATA_DTYPE

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        for source_ids, target_ids in self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0]
        ):
            if self.train_on_inputs:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [self.ignored_label] * (len(source_ids) - 1)
            else:
                source_mask = [self.ignored_label] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)

        model_inputs["input_ids"] = input_ids

        if input_ids[0] == 0:
            model_inputs["attention_mask"] = [1] * total_length
        else:
            model_inputs["attention_mask"] = [input_ids[0] // input_ids[0]] * total_length
        model_inputs["labels"] = labels
        return model_inputs

    def encode(self, json_line_bin):
        ids = {}
        lens = {}

        try:
            _sample = orjson.loads(json_line_bin)
        except Exception as e:
            return ids, lens, 0, 0

        # converted_sample = convert_sharegpt_to_intermediate(_sample, self.dataset_attr)
        converted_sample = self.convert_dataset_to_intermediate(_sample, self.dataset_attr)
        if len(converted_sample["prompt"]) == 0 or len(converted_sample["response"]) == 0:
            return ids, lens, 0, 0

        content_ids = self._tokenize_prompt(
            converted_sample, self.llama_factory_template, self.tokenizer.tokenizer)

        ## length of "input_ids", "attention_mask", and "labels" should be the same
        content_lens = [len(content_ids["input_ids"]),]
        if self.args.append_eod:
            content_ids["input_ids"].append(self.tokenizer.eod)
            content_ids["attention_mask"].append(1)
            content_ids["labels"].append(self.tokenizer.eod)
            content_lens[-1] += 1

        for key in self.args.json_keys:
            ids[key] = np.array(content_ids[key], dtype=DATA_DTYPE).tobytes(order='C')
            # ids[key] = content_ids[key]
            lens[key] = content_lens

        return ids, lens, len(json_line_bin), content_lens[-1] * DATA_DTYPE().itemsize


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers
        self.tokenizer = build_tokenizer(self.args)
        self.ignored_label = IGNORED_LABEL

        # self.args.json_keys = ["input_ids", "attention_mask", "labels"]

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
        ## [mindspeed] use 'packed' string to mark that this is a packed dataset
        ## [hh] "_packed" means pack multiple columns/keys [NOT sequence packing]
        output_prefix = output_prefix + "_packed"

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
        encoder = LlamaFactoryInstructionEncoder(self.args)

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
        total_bytes_tokenized = 0
        skip_num = 0

        valid_num = 0
        key_data_dict = {key: [] for key in self.args.json_keys}
        key_data_len_dict = {key: [] for key in self.args.json_keys}
        length2indexes = defaultdict(list)
        lengths = []
        for i, (doc, doc_len, bytes_processed, bytes_tokenized) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_bytes_tokenized += bytes_tokenized

            # sentences = doc["input_ids"]  # json_keys
            sentence_lens = doc_len["input_ids"]  # json_keys
            if self.args.seq_length is not None and sentence_lens[0] > self.args.seq_length:
                skip_num = skip_num + 1
                # print(f"Dropped lengthy example with length {sentence_lens[0]} > {self.args.seq_length}.", flush=True)
            else:
                lengths.append(sentence_lens[0])
                length2indexes[sentence_lens[0]].append(valid_num)
                for key in self.args.json_keys:
                    key_data_dict[key].append(doc[key])
                    key_data_len_dict[key].extend(doc_len[key])
                valid_num += 1
        print(f"Skip {skip_num} samples exceeded seq-length({self.args.seq_length})", flush=True)

        if self.args.knapsack:
            knapsacks = greedy_knapsack(lengths, self.args.seq_length)
        else:
            knapsacks = random_packing(lengths, self.args.seq_length)

        print(f"new samples num : {len(knapsacks)}",  flush=True)

        for k, knapsack in enumerate(knapsacks):

            packed_data_dict = {key: bytearray() for key in self.args.json_keys}

            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                for key in self.args.json_keys:
                    key_data = key_data_dict[key][index]
                    key_data_len = key_data_len_dict[key][index]
                    if self.args.neat_pack and "attention_mask" in key:
                        packed_data_dict[key].extend(np.array([i + 1] * key_data_len, dtype=DATA_DTYPE).tobytes(
                            order='C'))
                    else:
                        packed_data_dict[key].extend(key_data)

            if k % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                print(f"Processed {k} documents ({self.args.log_interval / elapsed} docs/s).")

            pad_length = self.args.seq_length - len(packed_data_dict['input_ids']) // DATA_DTYPE().itemsize
            if hasattr(self.tokenizer, "pad_token_id"):
                pad_token_id = self.tokenizer.pad_token_id
            elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "pad_token_id"):
                pad_token_id = self.tokenizer.tokenizer.pad_token_id
            else:
                raise ValueError("The pad_token_id attribute is missing for this tokenizer.")
            pad_token_dict = {
                "input_ids": pad_token_id,
                ## zhh: mask out [0] in neat_pack mask (varlen mask), else take all [1] in conventional mask
                "attention_mask": 0 if self.args.neat_pack else 1,
                "labels": self.ignored_label
            }
            for key in self.args.json_keys:
                packed_data_dict[key].extend(
                    np.array([pad_token_dict[key]] * pad_length, dtype=DATA_DTYPE).tobytes(order='C'))

            ## [[x,xx,...]]
            for key in self.args.json_keys:
                _sentence_length = len(packed_data_dict[key]) // DATA_DTYPE().itemsize
                if _sentence_length != self.args.seq_length:
                    raise ValueError("The length of packed example should be identical to the seq_length.")

                builders[key].add_item_np_bytes(bytes(packed_data_dict[key]), _sentence_length)
                builders[key].end_document()

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

    ## [mindspeed] use 'packed' string to mark that this is a packed dataset
    ## [hh] "_packed" means pack multiple columns/keys [NOT sequence packing]
    for key in args.json_keys:
        output_bin_files[key] = f"{output_prefix}_packed_{key}_{level}.bin"
        output_idx_files[key] = f"{output_prefix}_packed_{key}_{level}.idx"

        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=DATA_DTYPE
        )

        for _output in output_pres:
            full_output_prefix = f"{_output}_packed_{key}_{level}"
            builders[key].add_index(full_output_prefix)
        builders[key].finalize(output_idx_files[key])

    if args.clean_unmerged_indexed_dataset:
        for key in args.json_keys:
            for _output in output_pres:
                os.remove(f"{_output}_packed_{key}_{level}.bin")
                os.remove(f"{_output}_packed_{key}_{level}.idx")

    print("Time to preprocess:", time.time() - preprocess_tic, flush=True)


if __name__ == '__main__':
    s = time.time()
    main()
    print("total used time", time.time() - s)

