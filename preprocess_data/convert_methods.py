import json
import bisect
from functools import partial
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Sequence, Dict, Any, Literal, Optional, Union
from mindspeed_llm.tasks.preprocess.utils import convert_sharegpt_to_intermediate, convert_alpaca_to_intermediate
from preprocess_data.convert_methods_vehicle import (
    convert_vehicle_labeled_action_to_intermediate,
    convert_vehicle_labeled_ori_to_intermediate,
    convert_vehicle_labeled_multiturn_to_intermediate,
    convert_vehicle_labeled_synthesized_ori_to_intermediate
)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class InstructionDatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["file"]
    dataset_name: Optional[str] = None
    """ extra configs """
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    dataset_additional_keys = ""
    """ columns """
    system: Optional[str] = None
    images: Optional[str] = None
    """ columns for the alpaca format """
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    """ columns for the sharegpt format """
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    """ columns for the pairwise dataset """
    chosen: Optional[str] = "chosen"
    rejected: Optional[str] = "rejected"
    """ tags for the sharegpt format """
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


    """
    `intermediate`
    {
        'prompt': [
            {'role': 'user', 'content': '回答的非常好'},
            {'role': 'assistant', 'content': '感谢你的认可！还有什么需要我帮助的吗？'},
            {'role': 'user', 'content': '我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？'}
        ],
        'response': [
            {'role': 'assistant', 'content': '中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。'}
        ],
        'system': [''],
        'tools': ['']
    }
    """


def convert_dapo_math_17k_to_intermediate(sample: Dict[str, Union[List[Any], Dict]],
                                          dataset_attr: "InstructionDatasetAttr"):
    """
    https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    response = []

    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        prompt.append(
            {
                "role": Role.USER.value,
                "content": sample[dataset_attr.prompt][0]["content"]
            }
        )

    if dataset_attr.response and isinstance(sample[dataset_attr.response], dict):
        response.append(
            {
                "role": Role.ASSISTANT.value,
                "content": sample[dataset_attr.response]['ground_truth']
            }
        )

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def convert_dapo_math_17k_processed_to_intermediate(sample: Dict[str, Union[List[Any], Dict]],
                                                    dataset_attr: "InstructionDatasetAttr"):
    """
    https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    response = []
    ## qwen qwq example:
    suffix_instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        prompt.append(
            {
                "role": Role.USER.value,
                "content": sample[dataset_attr.prompt] + suffix_instruction
            }
        )

    if dataset_attr.response and sample[dataset_attr.response]:
        response.append(
            {
                "role": Role.ASSISTANT.value,
                "content": sample[dataset_attr.response]
            }
        )

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def _convert_alpaca_to_intermediate(sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr"):
    """
    https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []

    if dataset_attr.history and (isinstance(sample[dataset_attr.history], list)):
        for old_prompt, old_response in sample[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    content = []
    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        content.append(sample[dataset_attr.prompt])

    if dataset_attr.query and sample[dataset_attr.query]:
        content.append(sample[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

    if dataset_attr.ranking:
        if dataset_attr.chosen and isinstance(sample[dataset_attr.chosen], list):
            ## zhh: [chosen, rejected]
            response = [
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.chosen][0]},
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.rejected][1]},
            ]
        elif dataset_attr.chosen and isinstance(sample[dataset_attr.chosen], str):
            response = [
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.rejected]},
            ]
        else:
            response = []
    else:
        if dataset_attr.response and isinstance(sample[dataset_attr.response], list):
            ## zhh: multiple assistant?
            response = [
                {"role": Role.ASSISTANT.value, "content": content} for content in sample[dataset_attr.response]
            ]
        elif dataset_attr.response and isinstance(sample[dataset_attr.response], str):
            response = [{"role": Role.ASSISTANT.value, "content": sample[dataset_attr.response]}]
        else:
            response = []

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


DATASET_CONVERT_METHODS = {
    "dapo_math_17k": convert_dapo_math_17k_to_intermediate,
    "dapo_math_17k_processed": convert_dapo_math_17k_processed_to_intermediate,
    "alpaca": convert_alpaca_to_intermediate,
    "sharegpt": convert_sharegpt_to_intermediate,
    ##
    "vehicle_labeled_action": convert_vehicle_labeled_action_to_intermediate,
    "vehicle_labeled_ori": convert_vehicle_labeled_ori_to_intermediate,
    "vehicle_labeled_multiturn": convert_vehicle_labeled_multiturn_to_intermediate,
    "vehicle_labeled_synthesized_ori":convert_vehicle_labeled_synthesized_ori_to_intermediate,
    ##
    # "big_math_distilled": convert_big_math_distilled_to_intermediate,
}