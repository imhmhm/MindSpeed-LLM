"""modified based on megatron preprocess_data.py."""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import orjson
import numpy as np
from typing import Dict, List

from mindspeed_llm.tasks.preprocess.data_format_llamafactory import get_model_template
from mindspeed_llm.training.tokenizer import build_tokenizer

from preprocess_data.convert_methods import InstructionDatasetAttr
from preprocess_data.convert_methods import DATASET_CONVERT_METHODS
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
    args = parser.parse_args()
    return finalize_args(args)


class RLInstructionEncoder(object):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.llama_factory_template = get_model_template(
            args.prompt_type.strip(), args.prompt_type_path.strip(),
            enable_thinking=args.enable_thinking
        )

        self.ignored_label = IGNORED_LABEL
        self.train_on_inputs = False

        self.dataset_attr = InstructionDatasetAttr("file")
        self.dataset_attr.dataset_additional_keys = self.args.dataset_additional_keys
        self.dataset_attr.formatting = "alpaca"
        """
        attr_names = ["prompt", "query", "response", "history", "system", "chosen", "rejected"]
        ## zhh: keys are not verified
        """
        if args.map_keys is not None:
            for column_name, target_name in args.map_keys.items():
                setattr(self.dataset_attr, column_name, target_name)
        self.convert_dataset_to_intermediate = DATASET_CONVERT_METHODS[args.convert_method]

    def initializer(self):
        # Use Encoder class as a container for global data
        RLInstructionEncoder.tokenizer = build_tokenizer(self.args)
        RLInstructionEncoder.dtype = DATA_DTYPE

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:

        model_inputs = {"input_ids": [], "attention_mask": []}
        input_ids = []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        for source_ids, target_ids in self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0]
        ):
            input_ids += source_ids

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(input_ids)

        for add_key in self.args.dataset_additional_keys:
            if add_key == "labels":
                model_inputs["labels"] = self.tokenizer.tokenizer.encode(
                    example["response"][-1]["content"], padding=False, add_special_tokens=False)
            else:
                model_inputs[add_key] = self.tokenizer.tokenizer.encode(
                    example[add_key], padding=False, add_special_tokens=False)

        return model_inputs

    def encode(self, x):
        ## jsonl: bytes, parquet: dict
        if isinstance(x, bytes):
            try:
                sample = orjson.loads(x)
            except Exception:
                return {}, {}, 0, 0
        else:
            sample = x

        ids = {}
        lens = {}

        converted_sample = self.convert_dataset_to_intermediate(sample, self.dataset_attr)

        if len(converted_sample["prompt"]) == 0 or len(converted_sample["response"]) == 0:
            return ids, lens, 0, 0

        content_ids = self._tokenize_prompt(
            converted_sample, self.llama_factory_template, self.tokenizer.tokenizer)

        ## length of "labels" is different from "input_ids" and "attention_mask"
        for key in self.args.json_keys:
            ids[key] = np.array(content_ids[key], dtype=DATA_DTYPE).tobytes(order='C')
            lens[key] = [len(content_ids[key]),]

        bytes_tokenized = len(content_ids["input_ids"]) * DATA_DTYPE().itemsize
        bytes_processed = len(x) if isinstance(x, bytes) else 0
        return ids, lens, bytes_processed, bytes_tokenized


class RlPartition(BasePartition):
    encoder_cls = RLInstructionEncoder

    def _consume_encoded(self, encoded_docs, builders, proc_start):
        total_bytes_processed = 0
        total_bytes_tokenized = 0
        skip_num = 0
        for i, (doc, doc_len, bytes_processed, bytes_tokenized) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_bytes_tokenized += bytes_tokenized
            ## [[x,xx,...]]
            max_len = max(doc_len.values())[0]
            for key in self.args.json_keys:
                # [1] transformation to torch.IntTensor and back is 5 times slower than
                #     direct add_item or add_document
                # [2] add_item is still 4-5 times slower than add_item_np_bytes
                sentences = doc[key]
                sentence_lens = doc_len[key]

                # In post-training stage, we need to drop the data if any key exceeds set sequence-length
                if self.args.seq_length is not None and max_len > self.args.seq_length:
                    skip_num = skip_num + 1
                    continue
                builders[key].add_document_np_bytes(sentences, sentence_lens)

                # for j, _sent in enumerate(sentences):
                #     if self.args.seq_length is not None and sentence_lens[j] >= self.args.seq_length:
                #         skip_num = skip_num + 1
                #         continue
                #     builders[key].add_item_np_bytes(_sent, sentence_lens[j])
                # builders[key].end_document()

            self.print_processing_stats(i, proc_start, total_bytes_processed, total_bytes_tokenized, log_interval=self.args.log_interval)

        print(f"Skip {skip_num / len(self.args.json_keys)} samples exceeded seq-length({self.args.seq_length})",
              flush=True)


def main():
    args = get_args()
    RlPartition(args, args.workers).run()


if __name__ == '__main__':
    main()
