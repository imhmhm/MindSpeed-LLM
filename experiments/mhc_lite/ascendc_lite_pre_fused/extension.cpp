// Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
/** torch binding for the standalone LitePreFused Ascend C op (scheme F:
GEMM folded in).  Same deployment form as ascendc_lite_pre: the op binary
is built into a cann-ops clone via sync_and_build.sh and resolved at
runtime from ASCEND_CUSTOM_OPP_PATH. */

#include <torch/extension.h>
#include <vector>

#include "aclnn_common.h"

std::vector<at::Tensor> lite_pre_fused(const at::Tensor &x, const at::Tensor &w,
    const at::Tensor &scale, const at::Tensor &base)
{
    const int64_t sb = x.size(0);
    const int64_t h = x.size(1) / 4;
    TORCH_CHECK(h % 16 == 0, "h must be a multiple of 16 (32B tile rows)");
    auto y = at::empty({sb, h}, x.options());
    auto hPre = at::empty({sb, 8}, scale.options());
    auto hPost = at::empty({sb, 8}, scale.options());
    auto coeff = at::empty({sb, 32}, scale.options());
    // the kernel always copies whole 8-token rows, so tail-block lanes past
    // sb need slack; the returned view hides it
    auto rstdPad = at::empty({(sb + 7) / 8 * 8}, scale.options());
    auto logits = at::empty({sb, 32}, x.options());
    ACLNN_CMD(aclnnLitePreFused, x, w, scale, base, y, hPre, hPost, coeff, rstdPad, logits);
    return {y, hPre, hPost, coeff, rstdPad.narrow(0, 0, sb), logits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lite_pre_fused", &lite_pre_fused, "mhc_lite pre fused GEMM+heads op (standalone Ascend C)");
}
