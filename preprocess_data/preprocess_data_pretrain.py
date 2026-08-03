"""modified based on megatron preprocess_data.py."""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import orjson
import numpy as np

from mindspeed_llm.training.tokenizer import build_tokenizer

from preprocess_data.base import (
    DATA_DTYPE,
    BasePartition,
    add_preprocess_common_args,
    finalize_args,
)


def get_args():
    parser = argparse.ArgumentParser()
    add_preprocess_common_args(parser)
    args = parser.parse_args()
    return finalize_args(args)


class PretrainEncoder(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        PretrainEncoder.tokenizer = build_tokenizer(self.args)
        PretrainEncoder.dtype = DATA_DTYPE

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

        for key in self.args.json_keys:
            text = sample[key]

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
            _doc_ids_np_bytes = np.array(doc_ids, dtype=PretrainEncoder.dtype).tobytes(order='C')

            ids[key] = _doc_ids_np_bytes
            lens[key] = sentence_lens

        bytes_processed = len(x) if isinstance(x, bytes) else 0
        if not sentence_lens:
            ## debug
            print(f"debug **** {sentences}", flush=True)
            ##
            return ids, lens, bytes_processed, 0
        bytes_tokenized = sentence_lens[-1] * DATA_DTYPE().itemsize
        return ids, lens, bytes_processed, bytes_tokenized


class PretrainPartition(BasePartition):
    # pretrain 不需要 packing，输出不加 "_packed" 后缀
    PACKED_SUFFIX = ""
    encoder_cls = PretrainEncoder

    def _consume_encoded(self, encoded_docs, builders, proc_start):
        total_bytes_processed = 0
        total_bytes_tokenized = 0
        for i, (doc, doc_len, bytes_processed, bytes_tokenized) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_bytes_tokenized += bytes_tokenized
            for key in self.args.json_keys:
                # [1] transformation to torch.IntTensor and back is 5 times slower than
                #     direct add_item or add_document
                # [2] add_item is still 4-5 times slower than add_item_np_bytes
                builders[key].add_document_np_bytes(doc[key], doc_len[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed, total_bytes_tokenized, log_interval=self.args.log_interval)


def main():
    args = get_args()
    PretrainPartition(args, args.workers).run()


if __name__ == '__main__':
    main()
