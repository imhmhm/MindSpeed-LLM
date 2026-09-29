import pytest

from mindspeed_llm.tasks.preprocess.data_format_llamafactory import (
    ReasoningTemplate,
    get_model_template,
)
from mindspeed_llm.tasks.preprocess.data_format_llamafactory import template as lf_template
from mindspeed_llm.tasks.preprocess.data_format_llamafactory.template import TEMPLATES_DIR
from mindspeed_llm.tasks.preprocess.parser import InstructionDatasetAttr
from mindspeed_llm.tasks.preprocess.utils import convert_sharegpt_to_intermediate


def make_attr(reasoning_tag="thought"):
    attr = InstructionDatasetAttr("file", dataset_name="dummy")
    attr.formatting = "sharegpt"
    attr.reasoning_tag = reasoning_tag
    attr.dataset_additional_keys = []
    return attr


def qwen3_template():
    template = get_model_template("qwen3", str(TEMPLATES_DIR), True)
    assert isinstance(template, ReasoningTemplate)
    return template


SAMPLE = {
    "conversations": [
        {"from": "human", "value": "1+1=?"},
        {"from": "gpt", "value": "2", "thought": "one plus one is two"},
    ]
}


class TestConverterReasoningTag:
    def test_field_attached_to_intermediate(self):
        out = convert_sharegpt_to_intermediate(SAMPLE, make_attr())
        assert out["response"][0]["reasoning_content"] == "one plus one is two"
        assert out["response"][0]["content"] == "2"
        # every message carries the key so the arrow schema stays homogeneous
        assert out["prompt"][0]["reasoning_content"] == ""

    def test_missing_field_becomes_empty(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello"},
            ]
        }
        out = convert_sharegpt_to_intermediate(sample, make_attr())
        assert out["response"][0]["reasoning_content"] == ""

    def test_no_tag_keeps_legacy_output(self):
        out = convert_sharegpt_to_intermediate(SAMPLE, make_attr(reasoning_tag=None))
        assert "reasoning_content" not in out["response"][0]
        assert "reasoning_content" not in out["prompt"][0]


class TestMergeReasoningFields:
    def test_field_wrapped_into_content(self):
        template = qwen3_template()
        messages = [
            {"role": "user", "content": "q", "reasoning_content": ""},
            {"role": "assistant", "content": "a", "reasoning_content": "think"},
        ]
        template._merge_reasoning_fields(messages)
        assert messages[1]["content"] == "<think>\nthink\n</think>\n\na"
        assert messages[0]["content"] == "q"

    def test_inline_thinking_wins(self, caplog):
        template = qwen3_template()
        lf_template._reasoning_conflict_logged = False
        inline = "<think>\ninline\n</think>\n\na"
        messages = [{"role": "assistant", "content": inline, "reasoning_content": "field"}]
        with caplog.at_level("WARNING"):
            template._merge_reasoning_fields(messages)
        assert messages[0]["content"] == inline
        assert any("takes precedence" in record.message for record in caplog.records)

    def test_no_field_is_noop(self):
        template = qwen3_template()
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        template._merge_reasoning_fields(messages)
        assert messages[1]["content"] == "a"


class MockTokenizer:
    """Char-level stand-in: enough of the HF interface for encode_multiturn."""

    bos_token_id = None
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 10 for char in text]

    def convert_tokens_to_ids(self, token):
        return ord(token) + 10 if len(token) == 1 else 999


class TestEncodeMultiturnFold:
    def test_folded_think_lands_in_target(self):
        template = qwen3_template()
        messages = [
            {"role": "user", "content": "1+1?", "reasoning_content": ""},
            {"role": "assistant", "content": "2", "reasoning_content": "one plus one"},
        ]
        (source_ids, target_ids), = template.encode_multiturn(
            MockTokenizer(), messages, "sys", ""
        )
        target_text = "".join(chr(token - 10) for token in target_ids if token != 999)
        source_text = "".join(chr(token - 10) for token in source_ids if token != 999)
        assert "<think>\none plus one\n</think>\n\n2" in target_text
        assert "<think>" not in source_text
