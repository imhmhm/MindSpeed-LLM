# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

"""GPTModel wiring the mhc_lite head; decoder layer slots come from the lite layer spec."""

from mindspeed_llm.core.models.gpt.gpt_model_mhc import GPTModelMHC
from mindspeed_llm.tasks.models.transformer.mhc_lite import get_mhc_lite_spec


class GPTModelMHCLite(GPTModelMHC):

    def __init__(self, *args, hc_head_spec=None, **kwargs):
        if hc_head_spec is None:
            hc_head_spec = get_mhc_lite_spec(True)
        super().__init__(*args, hc_head_spec=hc_head_spec, **kwargs)
