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
#include "register/op_def_registry.h"

namespace ops {
class LitePreHeads : public OpDef {
public:
    explicit LitePreHeads(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("logits")
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
        this->Input("eps")
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

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(LitePreHeads);
}
