# Copyright 2025 the LlamaFactory team.
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
"""LlamaFactory-extracted formatting layer for MindSpeed-LLM preprocess.

Drop-in replacement for the ``mindspeed_llm.tasks.preprocess.templates`` entry points
(``get_model_template`` / ``fix_model_tokenizer``), backed by the LlamaFactory-extracted
``formatter`` + ``tool_utils`` + the MindSpeed-LLM Template class hierarchy, reading
``configs/finetune/templates.json``.
"""

from .template import (
    TEMPLATES_DIR,
    Template,
    LFDefaultTemplate,
    Llama2Template,
    ReasoningTemplate,
    get_model_template,
    fix_model_tokenizer,
    register_custom_template,
    get_templates,
)
from .formatter import (
    Formatter,
    EmptyFormatter,
    StringFormatter,
    FunctionFormatter,
    ToolFormatter,
)
from .tool_utils import TOOLS, FunctionCall, ToolUtils, get_tool_utils

__all__ = [
    "TEMPLATES_DIR",
    "Template",
    "LFDefaultTemplate",
    "Llama2Template",
    "ReasoningTemplate",
    "get_model_template",
    "fix_model_tokenizer",
    "register_custom_template",
    "get_templates",
    "Formatter",
    "EmptyFormatter",
    "StringFormatter",
    "FunctionFormatter",
    "ToolFormatter",
    "TOOLS",
    "FunctionCall",
    "ToolUtils",
    "get_tool_utils",
]
