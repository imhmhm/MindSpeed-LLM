"""Preprocess generative-reranker JSONL into a group-structured indexed dataset.

搜索团队数据样例：
{
    "channel": "default",
    "messages": [{"role": "user", "content": "你是一个严苛且精准的搜索相关性判定专家。能通过相关性、权威与可靠性、时效敏感性判断Document对回答或理解Query是否有实质性帮助。\n\n## Query: \n帮我生成一张沈阳局高铁形式照片\n[当前时间]：2025年11月12日\n[query分类]：铁路运输|交通摄影|高速铁路|城市\n[时效性]：弱时效性意图\n\n## Document: \n{doc_message}\n\n## 输出\n禁止任何解释！只允许输出yes或no。"}],
    "positive_messages": [
        [{"role": "assistant", "content": "[标题]：沈佳高铁沈白段全线铺轨完成-图片信息-沈阳市交通运输局\n[标题语义匹配度]：低\n[标题字面匹配度]：中\n[站点名]：jtj.shenyang.gov.cn 沈阳市交通运输局\n[站点分类]：交通\n[页面属性]：索引页\n[页面分类]：铁路建设|高铁|交通运输|基础设施建设|区域经济发展\n[来源]：普通网页\n[发布时间]：2025年05月08日\n[发布时间距离当前日期的天数]：180\n[内容]：5月4日，随着最后一对长钢轨在东山隧道内稳稳落下，与已铺轨道精准对接，标志着沈佳高铁沈白段（简称“沈白高铁”）全线铺轨施工圆满收尾，为全线开通奠定了坚实基础。沈白高铁正线全长430公里，设计时速350公里，正线起自沈阳北站，经辽宁省沈阳市、抚顺市、通化市、白山市、延边朝鲜族自治州等7个地市（州），终至长白山站。全线铺轨完成后，沈白高铁项目将加速推进无缝焊轨和轨道精调等工作，为联调联试和试运行做好准备。"}]
    ],
    "negative_messages": [
        [{"role": "assistant", "content": "[标题]：高铁乘务员手绘沈阳打卡地图送游客_腾讯新闻\n[标题语义匹配度]：中\n[标题字面匹配度]：低\n[站点名]：news.qq.com 腾讯网\n[站点分类]：新闻\n[页面属性]：新闻页\n[页面分类]：旅游|铁路|乘务员|文化传播|地方文化|旅游攻略\n[来源]：普通网页\n[发布时间]：2024年04月02日\n[发布时间距离当前日期的天数]：580\n[内容]：图为沈阳南至北京朝阳G976次列车乘务员霍宣洁(左)为乘车旅客发放手绘沈阳打卡地图。中新网记者 于海洋 摄图为沈阳南至北京朝阳G976次列车乘务员霍宣洁(左)为旅客讲解沈阳文化旅游图册。中新网记者 于海洋 摄图为沈阳南至北京朝阳G976次列车乘务员霍宣洁展示手绘沈阳打卡地图，为家乡代言。中新网记者 于海洋 摄图为沈阳南至北京朝阳G976次列车二组列车长谭莉敏(右)为乘务员霍宣洁的手绘沈阳打卡地图提出意见建议。中新网记者 于海洋 摄"}],
        [...]
    ]
}
"""
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import orjson

sys.path.append(str(Path(__file__).absolute().parent.parent))

from mindspeed_llm.tasks.preprocess.data_format_llamafactory import get_model_template
from mindspeed_llm.training.tokenizer import build_tokenizer

from preprocess_data.convert_methods import InstructionDatasetAttr
from preprocess_data.convert_methods import DATASET_CONVERT_METHODS
from preprocess_data.base import (
    DATA_DTYPE,
    BasePartition,
    add_preprocess_common_args,
    add_posttrain_args,
    finalize_args,
)


def get_args():
    parser = argparse.ArgumentParser()
    add_preprocess_common_args(parser)
    add_posttrain_args(parser)

    group = parser.add_argument_group(title='reranker')
    group.add_argument('--max-negatives', type=int, default=5,
                       help='Max negative docs sampled per record (= swift MAX_NEGATIVE_SAMPLES); '
                            'actual group size = 1 + min(available, this), variable')
    group.add_argument('--min-group-size', type=int, default=2,
                       help='Min sequences per group (= swift LISTWISE_RERANKER_MIN_GROUP_SIZE); '
                            'records with fewer are skipped')
    group.add_argument('--doc-placeholder', type=str, default='{doc_message}',
                       help='Placeholder in the query template, replaced by each doc text')
    group.add_argument('--positive-token', type=str, default='yes',
                       help='Label token filled for positive docs (must match '
                            '--reranker-positive-token on the training side)')
    group.add_argument('--negative-token', type=str, default='no',
                       help='Label token filled for negative docs (must match '
                            '--reranker-negative-token on the training side)')

    args = parser.parse_args()
    return finalize_args(args)


class RerankerEncoder(object):
    """
    搜索团队reranker数据
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.llama_factory_template = get_model_template(
            args.prompt_type.strip(), args.prompt_type_path.strip(),
            enable_thinking=args.enable_thinking
        )

        self.max_negatives = args.max_negatives
        self.min_group_size = args.min_group_size
        self.doc_placeholder = args.doc_placeholder
        self.positive_token = args.positive_token
        self.negative_token = args.negative_token
        self.seq_length = args.seq_length

        self.dataset_attr = InstructionDatasetAttr("file", dataset_name="RerankerGroupHandler")
        self.dataset_attr.dataset_additional_keys = self.args.dataset_additional_keys
        self.dataset_attr.messages = "messages"
        ## --map-keys: {"messages": "", "positive": "", "negative": "", "system": "",}
        if args.map_keys is not None:
            for attr_name, column_name in args.map_keys.items():
                setattr(self.dataset_attr, attr_name, column_name)
        self.convert_dataset_to_intermediate = DATASET_CONVERT_METHODS[args.convert_method]

    def initializer(self):
        # Use Encoder class as a container for global data
        RerankerEncoder.tokenizer = build_tokenizer(self.args)
        RerankerEncoder.dtype = DATA_DTYPE

    def _encode_sample(self, query_tpl: str, doc_text: str, system, label) -> List[int]:
        """single sample -> prompt-only
        1) replace doc_placeholder with single positive/negative doc
        2) chat template; the assistant slot carries the doc's label token
           (positive/negative token)
        """
        content = query_tpl.replace(self.doc_placeholder, doc_text)
        hf_tokenizer = RerankerEncoder.tokenizer.tokenizer
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": label},
        ]
        prompt_ids, _ = self.llama_factory_template.encode_oneturn(hf_tokenizer, messages, system, None)
        return prompt_ids

    @staticmethod
    def _drop(reason, x=None):
        """丢弃记录的哨兵返回: 原因放 ids['_drop'](写盘循环只遍历 json_keys, 自动忽略);
        bytes_tokenized 填 1 以穿过 BasePartition 的 filter(x[-1] > 0) 到达统计侧。"""
        return {'_drop': reason}, {}, (len(x) if isinstance(x, bytes) else 0), 1

    def encode(self, x):
        ## jsonl: bytes, parquet: dict
        if isinstance(x, bytes):
            try:
                sample = orjson.loads(x)
            except Exception:
                return self._drop('bad_line', x)
        else:
            sample = x

        ids = {}
        lens = {}

        converted = self.convert_dataset_to_intermediate(sample, self.dataset_attr)
        prompt_msgs = converted["prompt"]
        if not prompt_msgs:
            return self._drop('no_query', x)
        query_tpl = prompt_msgs[0].get('content') or ''
        if self.doc_placeholder not in query_tpl:
            return self._drop('no_placeholder', x)
        system = converted["system"][0] or None
        positives = converted["positive"]
        negatives = converted["negative"]
        if not positives:
            return self._drop('no_positive', x)

        ## zhh: 搜索团队数据: 每组 1 正 + min(可用, max) 负, 组大小可变, 不足 min-group-size 跳过
        n_neg = min(len(negatives), self.max_negatives)
        if n_neg + 1 < self.min_group_size:
            return self._drop('few_negs', x)
        chosen_negs = list(negatives) if n_neg == len(negatives) else random.sample(negatives, n_neg)
        chosen_pos = positives[0] if len(positives) == 1 else random.choice(positives)

        pos_ids = self._encode_sample(query_tpl, chosen_pos, system, self.positive_token)
        if self.seq_length is not None and len(pos_ids) > self.seq_length:
            return self._drop('pos_overlen', x)

        group_ids = [pos_ids]
        n_neg_dropped = 0
        for doc_text in chosen_negs:
            seq_ids = self._encode_sample(query_tpl, doc_text, system, self.negative_token)
            if self.seq_length is not None and len(seq_ids) > self.seq_length:
                n_neg_dropped += 1
                continue
            group_ids.append(seq_ids)
        if len(group_ids) < self.min_group_size:
            return self._drop('group_small', x)

        ids['input_ids'] = np.concatenate(
            [np.asarray(seq, dtype=DATA_DTYPE) for seq in group_ids]).tobytes(order='C')
        lens['input_ids'] = [len(seq) for seq in group_ids]
        lens['_negs_dropped'] = n_neg_dropped  # 额外键, 写盘循环只遍历 json_keys, 仅供统计
        bytes_tokenized = sum(lens['input_ids']) * DATA_DTYPE().itemsize
        bytes_processed = len(x) if isinstance(x, bytes) else 0
        return ids, lens, bytes_processed, bytes_tokenized


class RerankerPartition(BasePartition):

    PACKED_SUFFIX = "_packed"
    encoder_cls = RerankerEncoder

    def _consume_encoded(self, encoded_docs, builders, proc_start):
        total_bytes_processed = 0
        total_bytes_tokenized = 0
        drop_counts = {}       # 记录级丢弃原因 -> 数量(哨兵经 ids['_drop'] 带回)
        negs_dropped = 0       # 序列级: 组内被长度过滤掉的负样本数(lens['_negs_dropped'])
        written = 0
        for doc, doc_len, bytes_processed, bytes_tokenized in encoded_docs:
            total_bytes_processed += bytes_processed
            total_bytes_tokenized += bytes_tokenized
            if '_drop' in doc:
                drop_counts[doc['_drop']] = drop_counts.get(doc['_drop'], 0) + 1
                continue
            negs_dropped += doc_len.get('_negs_dropped', 0)
            for key in self.args.json_keys:
                sentences = doc[key]
                sentence_lens = doc_len[key]
                ## safety guard: encode 侧已保证 <= seq_length, 此处若触发说明有 bug
                if self.args.seq_length is not None and max(sentence_lens) > self.args.seq_length:
                    drop_counts['guard_overlen'] = drop_counts.get('guard_overlen', 0) + 1
                    continue
                ## one document = one query group (multi-item), group boundary = document_indices
                builders[key].add_document_np_bytes(sentences, sentence_lens)
                written += 1

            self.print_processing_stats(written, proc_start, total_bytes_processed, total_bytes_tokenized,
                                        log_interval=self.args.log_interval)

        total = written + sum(drop_counts.values())
        print(f"[reranker-preprocess] records: {total}, groups written: {written}, "
              f"dropped: {drop_counts or 0}, negatives dropped by length: {negs_dropped}", flush=True)


def main():
    args = get_args()
    args.json_keys = ['input_ids']
    if args.seq_length is None:
        raise ValueError('--seq-length is required for reranker preprocessing')
    if not args.prompt_type:
        raise ValueError('--prompt-type is required (e.g. qwen3 / ailab_slm)')
    partition = RerankerPartition(args, args.workers)
    partition.run()


if __name__ == '__main__':
    main()
