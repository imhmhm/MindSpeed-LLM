/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Host side of the LitePreHeads custom operator.
 *
 * One call folds the whole non-GEMM part of the mhc_lite pre stage:
 * per-token RMS square-sum, the three heads (sigmoid / 2*sigmoid /
 * softmax) on the raw bf16 logits, and the y mixture
 *     y = sum_i h_pre[i] * x[i*h : (i+1)*h],
 * so the caller only runs the logits GEMM (x @ (W*gamma)^T) around it.
 *
 * Shapes (mhc_lite, hc = 4):
 *   x      [sb, 4h]      bf16   raw RMSNorm input, gamma folded into W'
 *   logits [sb, 32]      bf16   pre 0:4 | post 4:8 | res 8:32
 *   scale  [32]          fp32   per logit lane: s0 x4 | s1 x4 | s2 x24
 *                             (a model constant the caller builds once)
 *   base   [32]          fp32   per logit lane
 *   eps    [8]           fp32   RMSNorm epsilon broadcast to 8 lanes (the
 *                             32B copy granularity); lane 0 is the value
 *   y      [sb, h]       bf16
 *   h_pre  [sb, 8]       fp32   lanes 0:4 valid
 *   h_post [sb, 8]       fp32   lanes 4:8 valid
 *   coeff  [sb, 32]      fp32   lanes 8:32 valid
 *   rstd   [sb, 1]       fp32   rsqrt(mean(x^2) + eps), consumed by the
 *                             backward's chain rule through l = logits*rstd
 * Padded 8/32-wide outputs keep every GM access 32B-aligned; the caller
 * slices the valid lanes.
 */
#include "register/op_impl_registry.h"

namespace ge {
static ge::graphStatus InferShape4LitePreHeads(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
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
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4LitePreHeads(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    context->SetOutputDataType(4, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}

IMPL_OP_INFERSHAPE(LitePreHeads).InferShape(ge::InferShape4LitePreHeads).InferDataType(ge::InferDataType4LitePreHeads);
