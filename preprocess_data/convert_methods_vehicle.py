import json
import bisect
from functools import partial
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Sequence, Dict, Any, Literal, Optional, Union


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


def convert_vehicle_labeled_action_to_intermediate(
    sample: Dict[str, Union[List[Any], Dict]],
    dataset_attr: "InstructionDatasetAttr"
):
    """

    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    response = []
    system = '''你的名字叫小艺，是由华为开发的智能助手。\n以下是车机操控使用的函数名列表，请根据`user`输入的意图从列表中选择正确的函数名，函数名列表在[extra_id_240]和[extra_id_241]之间：\n[extra_id_240]\nCarAccident, ChangeCarDeviceFragranceDensity, CheckCarFault, CheckCarFuelLeft, CheckCarFuelUsed, CheckCarReminderRange, CheckCarTirePressure, CheckDrivenMileage, CheckInsideOutSideCarTemperture, CloseCarSettingPage, EngageCarGear, FlameOutCar, FoldCarDevice, LayFlatCarDevice, LockCarDevice, MoveCarDeviceHigh, MoveCarDeviceLow, MoveCarDeviceStraight, PauseCarDevice, PowerOffCar, RecoverCarDeviceStatus, RelativeLayFlatCarDevice, ResetCarMileage, SaveCarDeviceStatus, SetCarAccelerationMode, SetCarDevice, SetCarDeviceAngle, SetCarDeviceColor, SetCarDeviceDuration, SetCarDeviceFragranceDensity, SetCarDeviceHardness, SetCarDeviceHeatGear, SetCarDeviceHeight, SetCarDeviceHorizontal, SetCarDeviceLength, SetCarDeviceLuminance, SetCarDeviceMassageGear, SetCarDeviceOpener, SetCarDevicePrivacyLevel, SetCarDeviceScreenSize, SetCarDeviceSpeed, SetCarDeviceSyncTemperature, SetCarDeviceTemperature, SetCarDeviceTransmittance, SetCarDeviceVentilationGear, SetCarDeviceVolume, SetCarDeviceWinddirection, SetCarDeviceWindspeed, SetCarElectricityLimit, SetCarEnergyRecoverGear, SetCarFuelRechargeGear, SetCarMileage, SetCarMode, SetCarReversedChargingLimit, SetCarSeatMassageMode, SetCarSetting, SetCarSteeringMode, SetCarSunroofStatus, SetCarWashMode, SetDriveMode, SetSuspensionStatus, TiltCarSunroof, TurnDownCarDeviceAngle, TurnDownCarDeviceDuration, TurnDownCarDeviceFragranceDensity, TurnDownCarDeviceHardness, TurnDownCarDeviceHeatGear, TurnDownCarDeviceHeight, TurnDownCarDeviceHorizontal, TurnDownCarDeviceLength, TurnDownCarDeviceLuminance, TurnDownCarDeviceMassageGear, TurnDownCarDeviceOpener, TurnDownCarDevicePrivacyLevel, TurnDownCarDeviceScreenSize, TurnDownCarDeviceSpeed, TurnDownCarDeviceTemperature, TurnDownCarDeviceTransmittance, TurnDownCarDeviceVentilationGear, TurnDownCarDeviceVolume, TurnDownCarDeviceWindspeed, TurnDownCarElectricityLimit, TurnDownCarEnergyRecoverGear, TurnDownCarFuelRechargeGear, TurnDownCarReversedChargingLimit, TurnOffCarDevice, TurnOffCarDeviceFunction, TurnOffCarSetting, TurnOnCarDevice, TurnOnCarDeviceFunction, TurnOnCarSetting, TurnUpCarDeviceAngle, TurnUpCarDeviceDuration, TurnUpCarDeviceFragranceDensity, TurnUpCarDeviceHardness, TurnUpCarDeviceHeatGear, TurnUpCarDeviceHeight, TurnUpCarDeviceHorizontal, TurnUpCarDeviceLength, TurnUpCarDeviceLuminance, TurnUpCarDeviceMassageGear, TurnUpCarDeviceOpener, TurnUpCarDevicePrivacyLevel, TurnUpCarDeviceScreenSize, TurnUpCarDeviceSpeed, TurnUpCarDeviceTemperature, TurnUpCarDeviceTransmittance, TurnUpCarDeviceVentilationGear, TurnUpCarDeviceVolume, TurnUpCarDeviceWindspeed, TurnUpCarElectricityLimit, TurnUpCarEnergyRecoverGear, TurnUpCarFuelRechargeGear, TurnUpCarReversedChargingLimit, UnFoldCarDevice, UnSetCarDevice, UnlockCarDevice, UnsetCarAccelerationMode, UnsetCarDeviceSyncTemperature, UnsetCarDeviceWinddirection, UnsetCarMode, UnsetCarSteeringMode, UnsetCarSunroofStatus, UnsetCarWashMode, UnsetDriveMode\n[extra_id_241]\n请直接回复正确的函数名，不要附加任何多余内容。'''

    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        prompt.append(
            {
                "role": Role.USER.value,
                "content": sample[dataset_attr.prompt]
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
    outputs["system"].append(system)
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def convert_vehicle_labeled_ori_to_intermediate(
    sample: Dict[str, Union[List[Any], Dict]],
    dataset_attr: "InstructionDatasetAttr"
):
    """

    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    response = []
    # system = '''你的名字叫小艺，是由华为开发的智能助手，可以理解复杂的用户问题，并遵照用户的所有要求。你具有较强的理解能力，可以根据【历史对话】中的信息，分析用户的问题并给出规划步骤。当你判断用户的问题是一个可以执行的任务时，能够遵从【规划策略】，输出工具调用指令。如果该任务需要多步完成，请将指令进行拆解，制定合理的、手机助手可执行的多步计划；如果该任务只需要单步完成，则只需要输出一步计划。用户的【历史对话】可能包含用户的普通文本指令，或用户想要处理的结构化对象，结构化对象包括文本对象、图片对象、文档对象等。【澄清】：如果用户的意图不明确，则询问用户更多信息以完成工具选择，询问用户时使用AskUser工具，该工具使用方法为：AskUser(text='xxx')，参数text的输入为澄清语句。【No Instruction】：如果没有具体的工具能实现用户诉求，则输出No Instruction。【规划策略】：\n1、计划中的每个API的输入参数只有以下三个来源: 1)用户query中提及；2)用户画像中查询；3)前置步骤API调用获得的结果；其中用户query的优先级最高。\n2、用户画像中能获取的信息，尽量从用户画像中获取。\n3、严禁虚构参数或者假设条件。\n4、使用AskUser工具时，必须要给出你具体想要询问的问题，并假设用户会直接清晰的回答你的问题，默认该工具会获取到你想要的信息。\n【输出格式】：<工具名称>(param1=<param1Value>, param2=<param2Value>)|<工具名称>(param1=<param1Value>, param2=<param2Value>)...\n'''
    system = '''你的名字叫小艺，是由华为开发的智能助手。\n请你根据用户输入的意图输出函数调用指令，具体要求如下：\n1、请根据你的记忆选择正确的函数名<functionName>\n2、如果你选择的函数不包含参数，请按以下格式输出：\n<functionName>()\n3、如果你选择的函数包含一个或多个参数，请按以下格式输出：\n<functionName>(<param1>=<param1Value>, <param2>=<param2Value>, ...)\n请注意，其中函数的参数名<param1>, <param2>, ...需要根据你的记忆正确填写；而函数的参数值<param1Value>, <param2Value>, ...则必须根据用户输入内容提取。\n\n示例1\n##用户输入##\n打开全部位置出风口\n##你的输出##\nTurnOnCarDevice(rangeType='全部', deviceType='出风口')\n\n示例2\n##用户输入##\n能量回收调大\n##你的输出##\nTurnUpCarEnergyRecoverGear()\n\n现在请你根据以下用户输入，直接输出函数调用指令，不要附加任何多余内容。'''

    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        prompt.append(
            {
                "role": Role.USER.value,
                "content": sample[dataset_attr.prompt]
            }
        )

    action = sample["action"]
    inputs = sample.get("input", sample.get("inputs"))

    ## function format
    """
    input: [{"name": "rangeType", "value": "全部"}, {"name": "deviceType", "value": "出风口"}]
    response_content: "TurnOffCarDevice(rangeType='全部', deviceType='出风口')"
    input: null
    response_content: "EngageCarGear()"
    """

    if inputs is None:
        response_content = f"{action}()"
    else:
        param_strs = []
        for param_dict in inputs:
            param_strs.append(f"{param_dict['name']}='{param_dict['value']}'")
        response_content = f"{action}({', '.join(param_strs)})"

    response.append(
        {
            "role": Role.ASSISTANT.value,
            "content": response_content
        }
    )

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(system)
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def convert_vehicle_labeled_multiturn_to_intermediate(
    sample: Dict[str, Union[List[Any], Dict]],
    dataset_attr: "InstructionDatasetAttr"
):
    """

    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    response = []
    system = '''你的名字叫小艺，是由华为开发的智能助手。'''

    action = sample["action"]

    inputs = sample.get("input", sample.get("inputs"))

    #### [prompt] add function names list to prompt
    user_instruction = f"以下是车机操控使用的函数名列表，请根据用户输入的意图从列表中选择正确的函数名，函数名列表在<functions>和</functions>之间：\n<functions>\nCarAccident, ChangeCarDeviceFragranceDensity, CheckCarFault, CheckCarFuelLeft, CheckCarFuelUsed, CheckCarReminderRange, CheckCarTirePressure, CheckDrivenMileage, CheckInsideOutSideCarTemperture, CloseCarSettingPage, EngageCarGear, FlameOutCar, FoldCarDevice, LayFlatCarDevice, LockCarDevice, MoveCarDeviceHigh, MoveCarDeviceLow, MoveCarDeviceStraight, PauseCarDevice, PowerOffCar, RecoverCarDeviceStatus, RelativeLayFlatCarDevice, ResetCarMileage, SaveCarDeviceStatus, SetCarAccelerationMode, SetCarDevice, SetCarDeviceAngle, SetCarDeviceColor, SetCarDeviceDuration, SetCarDeviceFragranceDensity, SetCarDeviceHardness, SetCarDeviceHeatGear, SetCarDeviceHeight, SetCarDeviceHorizontal, SetCarDeviceLength, SetCarDeviceLuminance, SetCarDeviceMassageGear, SetCarDeviceOpener, SetCarDevicePrivacyLevel, SetCarDeviceScreenSize, SetCarDeviceSpeed, SetCarDeviceSyncTemperature, SetCarDeviceTemperature, SetCarDeviceTransmittance, SetCarDeviceVentilationGear, SetCarDeviceVolume, SetCarDeviceWinddirection, SetCarDeviceWindspeed, SetCarElectricityLimit, SetCarEnergyRecoverGear, SetCarFuelRechargeGear, SetCarMileage, SetCarMode, SetCarReversedChargingLimit, SetCarSeatMassageMode, SetCarSetting, SetCarSteeringMode, SetCarSunroofStatus, SetCarWashMode, SetDriveMode, SetSuspensionStatus, TiltCarSunroof, TurnDownCarDeviceAngle, TurnDownCarDeviceDuration, TurnDownCarDeviceFragranceDensity, TurnDownCarDeviceHardness, TurnDownCarDeviceHeatGear, TurnDownCarDeviceHeight, TurnDownCarDeviceHorizontal, TurnDownCarDeviceLength, TurnDownCarDeviceLuminance, TurnDownCarDeviceMassageGear, TurnDownCarDeviceOpener, TurnDownCarDevicePrivacyLevel, TurnDownCarDeviceScreenSize, TurnDownCarDeviceSpeed, TurnDownCarDeviceTemperature, TurnDownCarDeviceTransmittance, TurnDownCarDeviceVentilationGear, TurnDownCarDeviceVolume, TurnDownCarDeviceWindspeed, TurnDownCarElectricityLimit, TurnDownCarEnergyRecoverGear, TurnDownCarFuelRechargeGear, TurnDownCarReversedChargingLimit, TurnOffCarDevice, TurnOffCarDeviceFunction, TurnOffCarSetting, TurnOnCarDevice, TurnOnCarDeviceFunction, TurnOnCarSetting, TurnUpCarDeviceAngle, TurnUpCarDeviceDuration, TurnUpCarDeviceFragranceDensity, TurnUpCarDeviceHardness, TurnUpCarDeviceHeatGear, TurnUpCarDeviceHeight, TurnUpCarDeviceHorizontal, TurnUpCarDeviceLength, TurnUpCarDeviceLuminance, TurnUpCarDeviceMassageGear, TurnUpCarDeviceOpener, TurnUpCarDevicePrivacyLevel, TurnUpCarDeviceScreenSize, TurnUpCarDeviceSpeed, TurnUpCarDeviceTemperature, TurnUpCarDeviceTransmittance, TurnUpCarDeviceVentilationGear, TurnUpCarDeviceVolume, TurnUpCarDeviceWindspeed, TurnUpCarElectricityLimit, TurnUpCarEnergyRecoverGear, TurnUpCarFuelRechargeGear, TurnUpCarReversedChargingLimit, UnFoldCarDevice, UnSetCarDevice, UnlockCarDevice, UnsetCarAccelerationMode, UnsetCarDeviceSyncTemperature, UnsetCarDeviceWinddirection, UnsetCarMode, UnsetCarSteeringMode, UnsetCarSunroofStatus, UnsetCarWashMode, UnsetDriveMode\n</functions>\n请根据以下用户输入直接回复正确的函数名，不要附加任何多余内容。\n\n##用户输入##\n{sample[dataset_attr.prompt]}"

    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        prompt.append(
            {
                "role": Role.USER.value,
                "content": user_instruction
            }
        )
    #### [prompt] output function name
    prompt.append(
        {
            "role": Role.ASSISTANT.value,
            "content": action
        }
    )

    #### [prompt] load function definition
    with open("/home/ma-user/work/dataset/mtp_zhh_wulan_efs/hf_data/vehicle_data/action_HivoiceIntent_车控.json") as jf:
        tools_dict = json.load(jf)
    function_desc = f"以下是函数{action}的信息：\n```{tools_dict[action]}```\n（`inputs`字段是该函数的参数列表）。\n\n现在请你根据上一轮的用户输入和上述函数信息，直接输出函数调用指令，函数调用指令格式如下：\n<functionName>(<param1>=<param1Value>, <param2>=<param2Value>, ...)\n请注意，务必根据用户输入内容和函数参数列表填写涉及的参数名<param1>, <param2>和对应的参数值<param1Value>, <param2Value>; 不要附加任何多余内容。"
    prompt.append(
        {
            "role": Role.USER.value,
            "content": function_desc
        }
    )

    #### [response] output function call
    ## function format
    """
    input: [{"name": "rangeType", "value": "全部"}, {"name": "deviceType", "value": "出风口"}]
    response_content: "TurnOffCarDevice(rangeType='全部', deviceType='出风口')"
    input: null
    response_content: "EngageCarGear()"
    """
    if inputs is None:
        response_content = f"{action}()"
    else:
        param_strs = []
        for param_dict in inputs:
            param_strs.append(f"{param_dict['name']}='{param_dict['value']}'")
        response_content = f"{action}({', '.join(param_strs)})"
    response.append(
        {
            "role": Role.ASSISTANT.value,
            "content": response_content
        }
    )

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(system)
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def convert_vehicle_labeled_synthesized_ori_to_intermediate(
    sample: Dict[str, Union[List[Any], Dict]],
    dataset_attr: "InstructionDatasetAttr"
):
    """

    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    response = []
    # system = '''你的名字叫小艺，是由华为开发的智能助手，可以理解复杂的用户问题，并遵照用户的所有要求。你具有较强的理解能力，可以根据【历史对话】中的信息，分析用户的问题并给出规划步骤。当你判断用户的问题是一个可以执行的任务时，能够遵从【规划策略】，输出工具调用指令。如果该任务需要多步完成，请将指令进行拆解，制定合理的、手机助手可执行的多步计划；如果该任务只需要单步完成，则只需要输出一步计划。用户的【历史对话】可能包含用户的普通文本指令，或用户想要处理的结构化对象，结构化对象包括文本对象、图片对象、文档对象等。【澄清】：如果用户的意图不明确，则询问用户更多信息以完成工具选择，询问用户时使用AskUser工具，该工具使用方法为：AskUser(text='xxx')，参数text的输入为澄清语句。【No Instruction】：如果没有具体的工具能实现用户诉求，则输出No Instruction。【规划策略】：\n1、计划中的每个API的输入参数只有以下三个来源: 1)用户query中提及；2)用户画像中查询；3)前置步骤API调用获得的结果；其中用户query的优先级最高。\n2、用户画像中能获取的信息，尽量从用户画像中获取。\n3、严禁虚构参数或者假设条件。\n4、使用AskUser工具时，必须要给出你具体想要询问的问题，并假设用户会直接清晰的回答你的问题，默认该工具会获取到你想要的信息。\n【输出格式】：<工具名称>(param1=<param1Value>, param2=<param2Value>)|<工具名称>(param1=<param1Value>, param2=<param2Value>)...\n'''
    system = '''你的名字叫小艺，是由华为开发的智能助手。\n请你根据用户输入的意图输出函数调用指令，具体要求如下：\n1、请根据你的记忆选择正确的函数名<functionName>\n2、如果你选择的函数不包含参数，请按以下格式输出：\n<functionName>()\n3、如果你选择的函数包含一个或多个参数，请按以下格式输出：\n<functionName>(<param1>=<param1Value>, <param2>=<param2Value>, ...)\n请注意，其中函数的参数名<param1>, <param2>, ...需要根据你的记忆正确填写；而函数的参数值<param1Value>, <param2Value>, ...则必须根据用户输入内容提取。\n\n示例1\n##用户输入##\n打开全部位置出风口\n##你的输出##\nTurnOnCarDevice(rangeType='全部', deviceType='出风口')\n\n示例2\n##用户输入##\n能量回收调大\n##你的输出##\nTurnUpCarEnergyRecoverGear()\n\n现在请你根据以下用户输入，直接输出函数调用指令，不要附加任何多余内容。'''

    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        prompt.append(
            {
                "role": Role.USER.value,
                "content": sample[dataset_attr.prompt]
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
    outputs["system"].append(system)
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs
