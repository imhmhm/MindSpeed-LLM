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
        """
        attr_names = [
            "messages", "system", "tools", "chosen", "rejected",
            "role_tag", "content_tag", "user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"
        ]
        ## zhh: keys are not verified
        """
        if args.map_keys is not None:
            for column_name, target_name in args.map_keys.items():
                setattr(self.dataset_attr, column_name, target_name)
        self.convert_dataset_to_intermediate = DATASET_CONVERT_METHODS[args.convert_method]

    def initializer(self):
        # Use Encoder class as a container for global data
        LlamaFactoryInstructionEncoder.tokenizer = build_tokenizer(self.args)
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

        ## length of "input_ids", "attention_mask", and "labels" should be the same
        content_lens = [len(content_ids["input_ids"]),]
        if self.args.append_eod:
            content_ids["input_ids"].append(self.tokenizer.eod)
            content_ids["attention_mask"].append(1)
            content_ids["labels"].append(self.tokenizer.eod)
            content_lens[-1] += 1

        for key in self.args.json_keys:
            ids[key] = np.array(content_ids[key], dtype=DATA_DTYPE).tobytes(order='C')
            lens[key] = content_lens

        bytes_tokenized = content_lens[-1] * DATA_DTYPE().itemsize
        bytes_processed = len(x) if isinstance(x, bytes) else 0
        return ids, lens, bytes_processed, bytes_tokenized


class SFTNoPackPartition(BasePartition):
    encoder_cls = LlamaFactoryInstructionEncoder

    def _consume_encoded(self, encoded_docs, builders, proc_start):
        total_bytes_processed = 0
        total_bytes_tokenized = 0
        skip_num = 0
        for i, (doc, doc_len, bytes_processed, bytes_tokenized) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_bytes_tokenized += bytes_tokenized
            for key in self.args.json_keys:
                sentences = doc[key]
                sentence_lens = doc_len[key]
                ## In post-training stage, we need to drop the data if any key exceeds set sequence-length
                ## length of "input_ids", "attention_mask", and "labels" should be the same
                if self.args.seq_length is not None and sentence_lens[0] > self.args.seq_length:
                    skip_num = skip_num + 1
                    continue
                builders[key].add_document_np_bytes(sentences, sentence_lens)

            self.print_processing_stats(i, proc_start, total_bytes_processed, total_bytes_tokenized, log_interval=self.args.log_interval)

        print(f"Skip {skip_num / len(self.args.json_keys)} samples exceeded seq-length({self.args.seq_length})",
              flush=True)


def main():
    args = get_args()
    SFTNoPackPartition(args, args.workers).run()


if __name__ == '__main__':
    main()
