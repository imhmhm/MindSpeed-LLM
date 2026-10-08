/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Host side of the LitePreFused custom operator (scheme F).
 *
 * One call runs the ENTIRE mhc_lite pre stage: the logits GEMM
 *     logits = x @ (W*gamma)^T    (bf16 cube, caller builds W' once)
 * lands tile by tile in the cube core's UB (C position VECIN), and the
 * vector epilogue on the same core turns each fp32 tile into
 *     l     = logits * rstd            (rstd from the x row itself)
 *     h_pre / h_post / coeff               (three heads)
 *     y = sum_i h_pre[i] * x_i,  bf16 out
 * plus a bf16 copy of l for the backward's recompute path.
 *
 * Compared with the split scheme E (torch GEMM + LitePreHeads) the
 * logits never round-trip GM and one aclnn launch covers GEMM + heads +
 * y; only the small W' = W*gamma broadcast stays outside.
 *
 * Shapes (mhc_lite, hc = 4, 32 logit lanes):
 *   x      [sb, 4h]   bf16   raw RMSNorm input
 *   w      [32, 4h]   bf16   W' = W*gamma, [N, K] row-major, B-side of
 *                          the matmul with transpose
 *   scale  [32]       fp32   per logit lane: s0 x4 | s1 x4 | s2 x24
 *   base   [32]       fp32   per logit lane
 *   y      [sb, h]    bf16
 *   h_pre  [sb, 8]    fp32   lanes 0:4 valid
 *   h_post [sb, 8]    fp32   lanes 4:8 valid
 *   coeff  [sb, 32]   fp32   lanes 8:32 valid
 *   rstd   [sb, 1]    fp32
 *   logits [sb, 32]   bf16   l before scale/base, saved for backward
 */
#include "register/op_impl_registry.h"

namespace ge {
static ge::graphStatus InferShape4LitePreFused(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const int64_t h = xShape->GetDim(1) / 4;
    *context->GetOutputShape(0) = *xShape;
    context->GetOutputShape(0)->SetDim(1, h);
    *context->GetOutputShape(1) = *xShape;
    context->GetOutputShape(1)->SetDim(1, 8);
    *context->GetOutputShape(2) = *xShape;
    context->GetOutputShape(2)->SetDim(1, 8);
    *context->GetOutputShape(3) = *xShape;
    context->GetOutputShape(3)->SetDim(1, 32);
    *context->GetOutputShape(4) = *xShape;
    context->GetOutputShape(4)->SetDim(1, 1);
    *context->GetOutputShape(5) = *xShape;
    context->GetOutputShape(5)->SetDim(1, 32);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4LitePreFused(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    context->SetOutputDataType(4, ge::DT_FLOAT);
    context->SetOutputDataType(5, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

IMPL_OP_INFERSHAPE(LitePreFused).InferShape(ge::InferShape4LitePreFused).InferDataType(ge::InferDataType4LitePreFused);
