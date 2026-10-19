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
#include "register/op_def_registry.h"

namespace ops {
class LitePreFused : public OpDef {
public:
    explicit LitePreFused(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("base")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h_pre")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h_post")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("coeff")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

    this->AICore().AddConfig("ascend910b");
}
};
OP_ADD(LitePreFused);
}  // namespace ops
