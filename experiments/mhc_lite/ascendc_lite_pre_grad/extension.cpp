// Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
/** torch binding for the standalone LitePreGrad Ascend C op (task #26:
the whole scheme-E pre backward except the two GEMMs).  Same deployment
form as ascendc_lite_pre: the op binary is built into a cann-ops clone via
sync_and_build.sh and resolved at runtime from ASCEND_CUSTOM_OPP_PATH. */

#include <torch/extension.h>
#include <vector>

#include "aclnn_common.h"

std::vector<at::Tensor> lite_pre_grad(const at::Tensor &g, const at::Tensor &x,
    const at::Tensor &ghpost, const at::Tensor &ghres, const at::Tensor &permT,
    const at::Tensor &logits, const at::Tensor &rstd, const at::Tensor &hpre8,
    const at::Tensor &scale, const at::Tensor &base)
{
    const int64_t sb = x.size(0);
    const int64_t h = x.size(1) / 4;
    TORCH_CHECK(h % 16 == 0, "h must be a multiple of 16 (32B tile rows)");
    TORCH_CHECK(ghpost.size(1) == 4 && ghres.size(1) == 16 && permT.numel() == 16 * 24,
        "mhc_lite shapes expected: ghpost [sb,4], ghres [sb,16], perm_t [16,24]");
    auto dlogits = at::empty({sb, 32}, x.options());
    auto dx = at::empty({sb, 4 * h}, x.options());
    // atomic targets must start at zero; dscale gets lane slack so the
    // kernel's 8-float copy stays on 32B granularity
    auto dscale = at::zeros({8}, scale.options());
    auto dbase = at::zeros({32}, scale.options());
    // ghpost is 4 floats/row (16B), below the copy granularity on tail
    // groups; pad it and rstd (read a full 8-float block per group) to
    // whole 8-row groups
    const int64_t sb8 = (sb + 7) / 8 * 8;
    at::Tensor ghp = ghpost;
    at::Tensor rst = rstd;
    if (sb8 != sb) {
        ghp = at::constant_pad_nd(ghpost, {0, 0, 0, sb8 - sb});
        rst = at::constant_pad_nd(rstd, {0, sb8 - sb});
    } else {
        ghp = ghpost.contiguous();
    }
    at::Tensor ghr = ghres.is_contiguous() ? ghres : ghres.contiguous();
    ACLNN_CMD(aclnnLitePreGrad, g, x, ghp, ghr, permT, logits, rst, hpre8, scale, base,
        dlogits, dx, dscale, dbase);
    return {dlogits, dx, dscale.narrow(0, 0, 3), dbase};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lite_pre_grad", &lite_pre_grad, "mhc_lite pre backward op (standalone Ascend C)");
}
