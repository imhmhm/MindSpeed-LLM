#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
from .convert_mg2hf import Mg2HfConvert


class AILabSLMMHCConverter(Mg2HfConvert):
    """mcore GPTModelMHC (ailab_slm_mhc) -> HF ailab_slm_mhc.

    Standard llama-style weights follow the llama2 template (model_cfg.json
    "ailab_slm_mhc"); MHC params keep their mcore names (attn_mhc / mlp_mhc /
    hc_head with hc_fn.weight / hc_scale / hc_base), so only the decoder-layer
    prefix is rewritten. MHC params are mcore-replicated (LinearNoTP, no TP
    shard), so the rank-0 copy is exported as-is.
    In GPTModelMHC the final layernorm applies after hc_head collapses the
    streams, so it lives on the model (final_layernorm.weight) instead of the
    transformer block (decoder.final_layernorm.weight).
    """

    MHC_PARAMS = ("hc_fn.weight", "hc_scale", "hc_base")

    def _copy_mhc_module(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, module_name):
        src = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])]
        for param in self.MHC_PARAMS:
            mg_key = f"decoder.layers.{local_layer_idx}.{module_name}.{param}"
            hf_key = f"model.layers.{hf_layer_idx}.{module_name}.{param}"
            hf_weight[hf_key] = src.pop(mg_key).clone()

    def set_model_layer_attn(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        super().set_model_layer_attn(hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag)
        if not mtp_layer_flag:
            self._copy_mhc_module(hf_weight, mg_weight, hf_layer_idx, local_layer_idx, "attn_mhc")

    def set_model_layer_mlp(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        super().set_model_layer_mlp(hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag)
        if not mtp_layer_flag:
            self._copy_mhc_module(hf_weight, mg_weight, hf_layer_idx, local_layer_idx, "mlp_mhc")

    def set_model_postprocess(self, hf_weight, mg_weight):
        src = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])]
        src["decoder.final_layernorm.weight"] = src.pop("final_layernorm.weight")
        super().set_model_postprocess(hf_weight, mg_weight)
        for param in self.MHC_PARAMS:
            hf_weight[f"model.hc_head.{param}"] = src.pop(f"hc_head.{param}").clone()
