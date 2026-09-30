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
 *   y      [sb, h]       bf16
 *   h_pre  [sb, 8]       fp32   lanes 0:4 valid
 *   h_post [sb, 8]       fp32   lanes 4:8 valid
 *   coeff  [sb, 32]      fp32   lanes 8:32 valid
 * Padded 8/32-wide outputs keep every GM access 32B-aligned; the caller
 * slices the valid lanes.
 */

#include "lite_pre_heads_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint32_t ROWS_PER_BLOCK = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t sb = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    const uint64_t eh = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    const uint64_t h = eh / 4;

    uint64_t totalBlocks = (sb + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    uint64_t blockDim = totalBlocks < aivNum ? totalBlocks : aivNum;

    LitePreHeadsTilingData tiling;
    tiling.set_sb(static_cast<uint32_t>(sb));
    tiling.set_h(static_cast<uint32_t>(h));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
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
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}

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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(LitePreHeads);
}
