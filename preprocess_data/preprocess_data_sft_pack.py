"""modified based on megatron preprocess_data.py."""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import time
from collections import defaultdict

from mindspeed_llm.tasks.preprocess.utils import greedy_knapsack, random_packing
from mindspeed_llm.training.tokenizer import build_tokenizer

from preprocess_data.preprocess_data_sft_no_pack import LlamaFactoryInstructionEncoder
from preprocess_data.base import (
    DATA_DTYPE,
    IGNORED_LABEL,
    BasePartition,
    add_preprocess_common_args,
    add_posttrain_args,
    finalize_args,
)


def get_args():
    parser = argparse.ArgumentParser()
    add_preprocess_common_args(parser)
    add_posttrain_args(parser)

    group = parser.add_argument_group(title='packing')
    group.add_argument("--neat-pack", action='store_true',
                       help="Use a zigzag attention mask.")
    group.add_argument("--packing-method", type=str, default=None,
                       choices=['knapsack', 'random'],
                       help="method of packing samples into a sequence")

    args = parser.parse_args()
    return finalize_args(args)


class SFTPackPartition(BasePartition):
    encoder_cls = LlamaFactoryInstructionEncoder

    def __init__(self, args, workers):
        super().__init__(args, workers)
        self.tokenizer = build_tokenizer(self.args)
        self.ignored_label = IGNORED_LABEL

    def _consume_encoded(self, encoded_docs, builders, proc_start):
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
        print(f"Skip {skip_num} samples exceeded seq-length({self.args.seq_length})",  flush=True)

        if self.args.packing_method == "knapsack":
            knapsacks = greedy_knapsack(lengths, self.args.seq_length)
        elif self.args.packing_method == "random":
            knapsacks = random_packing(lengths, self.args.seq_length)
        else:
            raise ValueError(f"unsupported packing-method: {args.packing_method}")
        
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


def main():
    args = get_args()
    SFTPackPartition(args, args.workers).run()


if __name__ == '__main__':
    s = time.time()
    main()
    print("total used time", time.time() - s)
