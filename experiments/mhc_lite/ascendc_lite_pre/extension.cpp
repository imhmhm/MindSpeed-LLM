// Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
/** torch binding for the standalone LitePreHeads Ascend C op.

The op binary lives outside CANN (built from the cann-ops project tree,
see sync_and_build.sh); the aclnn symbols are resolved at runtime from
libcust_opapi.so via ASCEND_CUSTOM_OPP_PATH, which is exactly how the
cann_ops_transformer ACLNN_CMD loader prefers custom ops over the CANN
built-ins.
 */

#include <torch/extension.h>
#include <vector>

#include "aclnn_common.h"

std::vector<at::Tensor> lite_pre_heads(const at::Tensor &x, const at::Tensor &logits,
    const at::Tensor &scale, const at::Tensor &base)
{
    const int64_t sb = x.size(0);
    const int64_t h = x.size(1) / 4;
    auto y = at::empty({sb, h}, x.options());
    auto hPre = at::empty({sb, 8}, scale.options());
    auto hPost = at::empty({sb, 8}, scale.options());
    auto coeff = at::empty({sb, 32}, scale.options());
    // the kernel always copies whole 8-token rows, so tail-block lanes past
    // sb need slack; the returned view hides it
    auto rstdPad = at::empty({(sb + 7) / 8 * 8}, scale.options());
    ACLNN_CMD(aclnnLitePreHeads, x, logits, scale, base, y, hPre, hPost, coeff, rstdPad);
    return {y, hPre, hPost, coeff, rstdPad.narrow(0, 0, sb)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lite_pre_heads", &lite_pre_heads, "mhc_lite pre heads fused op (standalone Ascend C)");
}
