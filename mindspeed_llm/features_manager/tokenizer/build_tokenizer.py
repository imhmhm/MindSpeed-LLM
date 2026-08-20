# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from pathlib import Path

from mindspeed.features_manager.tokenizer.build_tokenizer import BuildTokenizerFeature as MindSpeedBuildTokenizerFeature

TEMPLATES_DIR = str(
    Path(__file__).resolve().parent.parents[2]
    / "configs/finetune/templates.json"
)

# Shared prompt-type choices. Imported by standalone preprocess scripts
# (preprocess_data/preprocess_data_{pretrain,sft_no_pack,sft_pack,rl}.py) so their --prompt-type stays
# in sync with features_manager instead of drifting.
PROMPT_TYPE_CHOICES = [
    'default', 'empty', 'trl', 'chatglm2', 'chatglm3', 'chatglm3_system', 'glm4', 'glm4_moe', 'chatml', 'bailing_mini',
    'chatml_de', 'qwen', 'qwen_r1', "qwen_math_r1", 'llama3', 'llama2', 'mistral', 'mixtral', 'gemma', 'alpaca',
    'deepseek2', 'deepseek2-lite', 'minicpm3', 'cpm', 'baichuan2', 'deepseek3', 'intern2', 'hunyuan', 'qwen3', 'magistral', 'plm', 'qwen_lf', 'gpt_oss', 
    'ailab_slm',
]


class BuildTokenizerFeature(MindSpeedBuildTokenizerFeature):

    def register_args(self, parser):
        self.add_parser_argument_choices_value(parser, "--tokenizer-type", 'PretrainedFromHF')
        self.add_parser_argument_choices_value(parser, "--tokenizer-type", 'MagistralTokenizer')
        self.add_parser_argument_choices_value(parser, "--tokenizer-type", 'AILabSentencePieceTokenizer')
        ## zhh: 旧名, 供改名前训练的 ckpt 使用; build_tokenizer 会归一化到 AILabSentencePieceTokenizer
        self.add_parser_argument_choices_value(parser, "--tokenizer-type", 'PanguSentencePieceTokenizer')

        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                            help="Name or path of the huggingface tokenizer.")
        group.add_argument("--tokenizer-not-use-fast", action='store_false', dest='tokenizer_use_fast',
                            help="HuggingFace tokenizer not use the fast version.")
        group.add_argument('--padded-vocab-size', type=int, default=None,
                            help='set padded vocab size')
        group.add_argument('--prompt-type', type=str, default=None,
                            choices=PROMPT_TYPE_CHOICES,
                            help='Which template to use for constructing prompts in training/inference.'  'e.g., "qwen"')
        group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                            help='Path to the json file of templates.')
        group.add_argument('--tokenizer-padding-side', type=str, default='right',
                            help="tokenizer padding side")

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.tokenizer import build_tokenizer
        patch_manager.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer)