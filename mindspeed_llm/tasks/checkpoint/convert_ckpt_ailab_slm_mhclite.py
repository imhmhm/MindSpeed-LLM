#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
from .convert_ckpt_ailab_slm_mhc import AILabSLMMHCConverter


class AILabSLMMHCLiteConverter(AILabSLMMHCConverter):
    """mcore GPTModelMHCLite (ailab_slm_mhclite) -> HF ailab_slm_mhclite.

    Same layout as the mhc converter (module names and the model-level
    hc_head/final_layernorm placement are identical); the lite module adds the
    learned RMSNorm gain hc_gamma alongside hc_fn/hc_scale/hc_base.
    """

    MHC_PARAMS = ("hc_fn.weight", "hc_gamma", "hc_scale", "hc_base")
