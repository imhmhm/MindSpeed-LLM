/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Host side of the LitePreGrad custom operator (task #26).
 *
 * One AIV launch replaces the scheme-E triton backward
 * (lite_pre_backward + lite_grad_x) and all the torch glue around it:
 * the l recompute (logits*rstd), the grad_logits cast, the drstd/coef
 * scalar chain, the coef*x broadcast and the dscale lane-count fix all
 * happen inside, so the wrapper's backward is this op plus the two GEMMs
 * autograd already runs.
 *
 * Shapes (mhc_lite, hc = 4, 32 logit lanes, 24 res lanes):
 *   g       [sb, h]    bf16   grad of y
 *   x       [sb, 4h]   bf16   raw RMSNorm input
 *   ghpost  [sb, 4]    fp32   grad of h_post
 *   ghres   [sb, 16]   fp32   grad of h_res (permutation-mixed)
 *   perm_t  [16, 24]   fp32   permutation table (res coefficient basis)
 *   logits  [sb, 32]   bf16   raw GEMM output saved by the forward
 *   rstd    [sb]       fp32   from LitePreHeads (padded storage)
 *   hpre8   [sb, 8]    fp32   from LitePreHeads, lanes 0:4 valid
 *   scale   [32]       fp32   lane scales (reads 0 / 4 / 8)
 *   base    [32]       fp32
 *   dlogits [sb, 32]   bf16   grad wrt the raw logits (dl*rstd, cast once)
 *   dx      [sb, 4h]   bf16   grad wrt x (h_pre*g + coef*x)
 *   dscale  [3]        fp32   lane-count corrected (1/4, 1/4, 1/24)
 *   dbase   [32]       fp32   zero-initialized, atomically accumulated
 */
#include "lite_pre_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
constexpr uint32_t ROWS_PER_BLOCK = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t sb = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    const uint64_t eh = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    const uint64_t h = eh / 4;

    uint64_t totalBlocks = (sb + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    uint64_t blockDim = totalBlocks < aivNum ? totalBlocks : aivNum;

    LitePreGradTilingData tiling;
    tiling.set_sb(static_cast<uint32_t>(sb));
    tiling.set_h(static_cast<uint32_t>(h));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

IMPL_OP_OPTILING(LitePreGrad).Tiling(optiling::TilingFunc);
