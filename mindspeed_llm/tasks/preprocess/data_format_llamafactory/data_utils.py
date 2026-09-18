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
"""Trimmed from LlamaFactory data_utils.py: only the SLOTS type alias and Role enum
are needed by the formatting layer (formatter/template). Dataset-loading helpers dropped."""

from __future__ import annotations

from enum import Enum, unique
from typing import Union

SLOTS = list[Union[str, set, dict]]


@unique
class Role(str, Enum):
    r"""Role of a message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"
