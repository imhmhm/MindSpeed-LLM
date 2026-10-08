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
#include "register/op_impl_registry.h"

namespace ge {
static ge::graphStatus InferShape4LitePreGrad(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(1);
    *context->GetOutputShape(0) = *context->GetInputShape(5);  // [sb, 32]
    *context->GetOutputShape(1) = *xShape;                     // [sb, 4h]
    *context->GetOutputShape(2) = gert::Shape({3});
    *context->GetOutputShape(3) = gert::Shape({32});
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4LitePreGrad(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

IMPL_OP_INFERSHAPE(LitePreGrad).InferShape(ge::InferShape4LitePreGrad).InferDataType(ge::InferDataType4LitePreGrad);
