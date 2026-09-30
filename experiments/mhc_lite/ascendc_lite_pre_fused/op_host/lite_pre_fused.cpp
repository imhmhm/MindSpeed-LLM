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

#include "lite_pre_fused_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const int32_t M = static_cast<int32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(0));
    const int32_t eh = static_cast<int32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(1));
    const int32_t h = eh / 4;
    const int32_t N = 32;
    const int32_t K = eh;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    // N = 32 never splits, so the parallelism available to the tiling lib is
    // the M fractal-block count; asking for more cores than blocks makes
    // GetTiling reject small M
    const uint32_t mBlocks = static_cast<uint32_t>((M + 127) / 128);
    uint32_t dim = aicNum;
    if (mBlocks < dim) {
        dim = mBlocks > 0 ? mBlocks : 1;
    }
    cubeTiling.SetDim(dim);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16, true);
    cubeTiling.SetCType(TPosition::VECIN, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetFixSplit(128, 32, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    LitePreFusedTilingData tiling;
    tiling.set_sb(static_cast<uint32_t>(M));
    tiling.set_h(static_cast<uint32_t>(h));
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        // baseN = 32 may be rejected for the tiny N; let the tiling lib
        // choose its own bases and retry once
        printf("[LitePreFused] fixsplit(128,32) tiling failed, retrying auto tiling\n");
        cubeTiling.SetFixSplit(-1, -1, -1);
        if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
            printf("[LitePreFused] auto tiling failed too (M=%d N=%d K=%d dim=%u)\n", M, N, K, aicNum);
            return ge::GRAPH_FAILED;
        }
    }
    tiling.cubeTilingData.set_stepM(1);
    tiling.cubeTilingData.set_stepN(1);
    // The lib may pick singleCoreM = ceil(M / dim) (e.g. 205 for M=4096,
    // dim=20), which is not a multiple of baseM: the per-core Iterate then
    // covers only singleCoreM/baseM full tiles and the rest of the core's
    // rows are never written.  Force a baseM-multiple that still fits dim,
    // and keep all N=32 lanes on one core so each Iterate tile is one
    // baseM-row slice of this core's M range.
    const int32_t baseMB = tiling.cubeTilingData.get_baseM();
    const int32_t desiredM = baseMB * static_cast<int32_t>((mBlocks + dim - 1) / dim);
    tiling.cubeTilingData.set_singleCoreM(desiredM);
    tiling.cubeTilingData.set_singleCoreN(N);
    if (tiling.cubeTilingData.get_baseN() != N) {
        printf("[LitePreFused] baseN=%d != N=%d, tile mapping assumes single N block\n",
               tiling.cubeTilingData.get_baseN(), N);
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(dim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
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

static graphStatus InferDataType(gert::InferDataTypeContext *context)
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

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore()
        .SetTiling(optiling::TilingFunc)
        .AddConfig("ascend910b");
}
};
OP_ADD(LitePreFused);
}  // namespace ops
