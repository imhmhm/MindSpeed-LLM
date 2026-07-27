# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpeed-LLM preprocess formatting layer (LlamaFactory-extracted design).

Keeps the MindSpeed-LLM Template class hierarchy (Template / Llama2Template /
ReasoningTemplate; LFDefaultTemplate is unified into Template as a backward-compat
alias) and JSON-driven registration (configs/finetune/templates.json), but:
  * swaps the formatter layer to the LlamaFactory-extracted formatter/tool_utils
    (no ForThink mess);
  * threads ``thought_words`` / ``tool_call_words`` through ``format_function.apply``
    (no hardcoded <think> regex);
  * adopts no-truncation (no ``_make_pairs``; ``cutoff_len`` / ``reserved_label_len``
    are accepted only for handler-contract compatibility);
  * fixes the tool_format key bug (``_format_custom_template`` no longer corrupts strings);
  * drops the name-keyed ForThink selection (tool_format drives ToolUtils via the registry).
"""

from __future__ import annotations

import os
import sys
import re
import json
import logging
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from .data_utils import Role
from .formatter import EmptyFormatter, FunctionFormatter, ObservationFormatter, StringFormatter, ToolFormatter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatter import SLOTS, Formatter
    from .tool_utils import FunctionCall

logger = logging.getLogger(__name__)

cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent.parent, "configs/finetune/templates.json")


@dataclass
class Template:
    r"""Chat template aligned with LlamaFactory latest structure.

    ``_encode`` returns a flat list of per-message token-id lists; ``encode_oneturn`` /
    ``encode_multiturn`` build the (prompt, response) pair(s) inline. No ``_make_pairs`` /
    truncation -- ``cutoff_len`` and ``reserved_label_len`` are accepted only for signature
    compatibility with the handler contract. ``format_separator`` is a MindSpeed-LLM
    extension (LlamaFactory latest has none); templates that do not define it get an
    EmptyFormatter (no-op).
    """

    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    thought_words: tuple[str, str]
    tool_call_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    force_system: bool
    enable_thinking: Optional[bool]
    preserve_thinking: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000_000,
        reserved_label_len: int = 1,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids
        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000_000,
        reserved_label_len: int = 1,
        discarding_history_cot: bool = False,
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract tool message."""
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Return stop token ids."""
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))
        return list(stop_token_ids)

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}{self.thought_words[1]}" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to a flat list of per-message token-id lists.

        Turn 0: prefix + system + query        resp
        Turn t: sep + query                     resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message["content"], thought_words=self.thought_words, tool_call_words=self.tool_call_words
                )
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


@dataclass
class Llama2Template(Template):
    r"""A template that fuses the system message into the first user message."""

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message["content"], thought_words=self.thought_words, tool_call_words=self.tool_call_words
                )
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


@dataclass
class ReasoningTemplate(Template):
    r"""A template that adds thought to the assistant message."""

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000_000,
        reserved_label_len: int = 1,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        if not self.preserve_thinking:
            for i in range(1, len(messages) - 2, 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0].strip() not in messages[-1]["content"]
            and self.thought_words[1].strip() not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000_000,
        reserved_label_len: int = 1,
        discarding_history_cot: bool = False,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)

        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if discarding_history_cot:
            for i in range(1, len(messages) - 2, 2):  # preserve the last cot
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)

        if discarding_history_cot:
            turn_indices = [len(messages) - 2]
        else:
            turn_indices = range(0, len(messages), 2)

        for i in turn_indices:
            if (
                self.thought_words[0].strip() not in messages[i + 1]["content"]
                and self.thought_words[1].strip() not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


# Backward-compat: LFDefaultTemplate (0.9.4) is unified into Template (LlamaFactory latest).
LFDefaultTemplate = Template


templates: dict[str, Template] = {}


def get_templates() -> dict[str, Template]:
    return templates


def get_model_template(name, prompt_type_path, enable_thinking):
    name = register_custom_template(name, prompt_type_path, enable_thinking)
    if name is None:
        template = templates["empty"]  # placeholder
    else:
        template = get_templates().get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))
    return template


def fix_model_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
    prompt_type_path: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
):
    template = get_model_template(name, prompt_type_path, enable_thinking)

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Optional[list[str]] = None,
    thought_words: Optional[tuple[str, str]] = None,
    tool_call_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    force_system: bool = False,
    enable_thinking: Optional[bool] = True,
    preserve_thinking: bool = False,
    template_class: type["Template"] = Template,
) -> None:
    r"""Register a chat template."""
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        tool_call_words=tool_call_words or ("<tool_call>", "</tool_call>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def register_custom_template(name, json_file_path=TEMPLATES_DIR, enable_thinking=False) -> str:
    if name in templates:
        return name

    if not bool(re.match(r'(?:(?:/|\.{1,2}/|[^/\0]+/)(?:[^/\0]+/)*[^/\0]*|\.{1,2})', json_file_path)):
        raise ValueError(f"Invalid Path: {json_file_path}, please provide a valid custom template path.")

    with open(json_file_path, 'r') as file:
        config = json.load(file)

    templates_dict = {template['name']: template for template in config}
    config = templates_dict.get(name, None)

    if not config:
        raise ValueError(f"Can't find the template. Please provide a valid prompt type template in the {json_file_path}.")

    format_user = _format_custom_template(config.get("format_user", None))
    format_assistant = _format_custom_template(config.get("format_assistant", None))
    format_system = _format_custom_template(config.get("format_system", None))
    format_function = _format_custom_template(config.get("format_function", None))
    format_observation = _format_custom_template(config.get("format_observation", None))
    format_tools = _format_custom_template(config.get("format_tools", None))
    format_separator = _format_custom_template(config.get("format_separator", None))
    format_prefix = _format_custom_template(config.get("format_prefix", None))
    default_system = _format_custom_template(config.get("default_system", ""))
    stop_words = _format_custom_template(config.get("stop_words", []))
    efficient_eos = _format_custom_template(config.get("efficient_eos", False))
    replace_eos = _format_custom_template(config.get("replace_eos", False))
    force_system = _format_custom_template(config.get("force_system", False))
    template_class = _format_custom_template(config.get("template_class", None))
    thought_words = _format_custom_template(config.get("thought_words", None))
    tool_call_words = _format_custom_template(config.get("tool_call_words", None))
    preserve_thinking = _format_custom_template(config.get("preserve_thinking", False))

    if isinstance(default_system, list):
        default_system = "".join(default_system) if all(isinstance(sentence, str) for sentence in default_system) else default_system
    if isinstance(thought_words, list):
        thought_words = tuple(thought_words)
    if isinstance(tool_call_words, list):
        tool_call_words = tuple(tool_call_words)
    format_user = StringFormatter(**format_user) if format_user else None
    format_assistant = StringFormatter(**format_assistant) if format_assistant else None
    format_system = StringFormatter(**format_system) if format_system else None
    format_observation = ObservationFormatter(**format_observation) if format_observation else None
    format_separator = EmptyFormatter(**format_separator) if format_separator else None
    format_prefix = EmptyFormatter(**format_prefix) if format_prefix else None
    template_class = _get_template_class(template_class) if template_class else Template
    format_function = FunctionFormatter(**format_function) if format_function else None
    format_tools = ToolFormatter(**format_tools) if format_tools else None

    _register_template(
        name=name,
        format_user=format_user,
        format_assistant=format_assistant,
        format_system=format_system,
        format_function=format_function,
        format_observation=format_observation,
        format_tools=format_tools,
        format_separator=format_separator,
        format_prefix=format_prefix,
        default_system=default_system,
        stop_words=stop_words,
        thought_words=thought_words,
        tool_call_words=tool_call_words,
        preserve_thinking=preserve_thinking,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
        enable_thinking=enable_thinking,
        template_class=template_class,
    )

    return name


def _format_custom_template(slots: dict) -> dict:
    r"""Normalise a format_* section from templates.json.

    Only the ``slots`` value may contain list-typed items (e.g. ``["bos_token"]``) that
    must be converted to sets; other keys (notably ``tool_format``) are strings and must
    be left untouched so ``get_tool_utils`` receives the real registry key.
    """
    if slots and isinstance(slots, dict):
        for key, value in slots.items():
            if key == "slots" and isinstance(value, list):
                slots[key] = [set(item) if isinstance(item, list) else item for item in value]
    return slots


def _get_template_class(template_name: str):
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("curent module not found")
    template_class = getattr(current_module, template_name, None)
    if template_class is None:
        template_class = Template
    logger.info("template will use %s to format dataset", template_class.__name__)

    return template_class
