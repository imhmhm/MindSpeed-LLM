/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Device side of the LitePreFused custom operator (mhc_lite pre, hc = 4).
 *
 * Scheme F: the logits GEMM x @ W^T runs on the cube pipe of each AI
 * Cube core through the Ascend C Matmul API (C position VECIN, so each
 * [baseM, 32] fp32 tile lands in this core's UB), and the vector
 * epilogue consumes the tile in place:
 *   l   = tile * rstd          (rstd from the x rows, pass1 chunk loop)
 *   heads sigmoid / 2*sigmoid / softmax on l, fp32
 *   y   = sum_i h_pre[i] * x_i, cast bf16
 *   bf16 copy of l saved as `logits` for the backward recompute
 * so the logits never touch GM; the caller's W' = W*gamma stays as the
 * B-side weight (gamma acts on K and cannot fold into a per-lane scale).
 *
 * The M split over cores follows the matmul tiling (singleCoreM per
 * core, mirrored the way matmul_leakyrelu's CalcOffset does it); within
 * a core the epilogue walks the C tile in ROWS = 8 token groups, the
 * structure of the validated LitePreHeads kernel: per-row scalars ride
 * scalar registers (GetValue + Muls) and every vector op is the simple
 * count form over a 32B-aligned window.  The tile window at row r sits
 * at r*32 fp32 = 128 B, so the head passes keep their alignment.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr int32_t ROWS = 8;   // tokens per epilogue group
constexpr int32_t HC = 4;     // mhc head count
constexpr int32_t NL = 32;    // logits width: pre 0:4 | post 4:8 | res 8:32
constexpr int32_t NRES = 24;
constexpr float EPS = 1e-5f;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline void SyncVS()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(id);
    WaitFlag<HardEvent::V_S>(id);
}

__aicore__ inline void SyncSV()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(id);
    WaitFlag<HardEvent::S_V>(id);
}
}  // namespace

class KernelLitePreFused {
public:
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>,
           MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>>
        matmulObj_;

    __aicore__ inline KernelLitePreFused() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR base,
        GM_ADDR y, GM_ADDR hPre, GM_ADDR hPost, GM_ADDR coeff, GM_ADDR rstd, GM_ADDR logits,
        LitePreFusedTilingData *tiling, TPipe *pipe)
    {
        sb_ = tiling->sb;
        h_ = tiling->h;
        eh_ = HC * h_;
        ct_ = tiling->cubeTilingData;
        pipe_ = pipe;
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)x);
        wGm_.SetGlobalBuffer((__gm__ bfloat16_t *)w);
        sGm_.SetGlobalBuffer((__gm__ float *)scale);
        bGm_.SetGlobalBuffer((__gm__ float *)base);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t *)y);
        hpGm_.SetGlobalBuffer((__gm__ float *)hPre);
        hoGm_.SetGlobalBuffer((__gm__ float *)hPost);
        coGm_.SetGlobalBuffer((__gm__ float *)coeff);
        rGm_.SetGlobalBuffer((__gm__ float *)rstd);
        lgGm_.SetGlobalBuffer((__gm__ bfloat16_t *)logits);

        // this core's share of M, mirroring the matmul tiling split (N never
        // splits: singleCoreN == N == 32)
        mSingleBlocks_ = Ceiling(static_cast<uint32_t>(ct_.M), static_cast<uint32_t>(ct_.singleCoreM));
        const int32_t mCoreIndx = static_cast<int32_t>(GetBlockIdx()) % static_cast<int32_t>(mSingleBlocks_);
        rowStart_ = static_cast<uint32_t>(mCoreIndx * ct_.singleCoreM);
        rowsCore_ = (sb_ < rowStart_) ? 0 : ((sb_ - rowStart_ < static_cast<uint32_t>(ct_.singleCoreM))
            ? (sb_ - rowStart_) : static_cast<uint32_t>(ct_.singleCoreM));

        const uint32_t cBytes = static_cast<uint32_t>(ct_.baseM) * static_cast<uint32_t>(ct_.baseN) * sizeof(float);
        const uint32_t xBfBytes = ROWS * eh_ * sizeof(bfloat16_t);
        const uint32_t x32Bytes = ROWS * h_ * sizeof(float);
        const uint32_t yBfBytes = xBfBytes / 2;
        uint32_t off = 0;
        offXbf_ = off;    off += xBfBytes;
        offX32_ = off;    off += x32Bytes;
        offY32_ = off;    off += x32Bytes;
        offYbf_ = off;    off += yBfBytes;
        offL32_ = off;    off += ROWS * NL * sizeof(float);
        offC32_ = off;    off += ROWS * NL * sizeof(float);
        offLgBf_ = off;   off += ROWS * NL * sizeof(bfloat16_t);
        offScale_ = off;  off += NL * sizeof(float);
        offBase_ = off;   off += NL * sizeof(float);
        offZ8_ = off;     off += ROWS * 8 * sizeof(float);
        offE8_ = off;     off += ROWS * 8 * sizeof(float);
        offD8_ = off;     off += ROWS * 8 * sizeof(float);
        offOne8_ = off;   off += ROWS * 8 * sizeof(float);
        offHpre_ = off;   off += ROWS * 8 * sizeof(float);
        offHpost_ = off;  off += ROWS * 8 * sizeof(float);
        offRsum_ = off;   off += ROWS * 8 * sizeof(float);
        offRstd_ = off;   off += ROWS * sizeof(float);
        offRtmp_ = off;   off += 512;
        pipe_->InitBuffer(cBuf_, cBytes);
        pipe_->InitBuffer(ub_, off);
    }

    __aicore__ inline void Process()
    {
        // cores past the M split would wrap to an already-owned block (the
        // grid launches aicNum cores, the split may use fewer)
        if (rowsCore_ == 0 || GetBlockIdx() >= mSingleBlocks_) {
            return;
        }
        LoadConstants();
        aGm_ = xGm_[static_cast<int64_t>(rowStart_) * eh_];
        matmulObj_.SetTensorA(aGm_);
        matmulObj_.SetTensorB(wGm_, true);

        uint32_t tile = 0;
        while (matmulObj_.template Iterate<true>()) {
            auto cTile = cBuf_.Get<float>();
            matmulObj_.template GetTensorC<true>(cTile, false, true);
            const uint32_t done = tile * static_cast<uint32_t>(ct_.baseM);
            if (done < rowsCore_) {
                const uint32_t rows = (rowsCore_ - done < static_cast<uint32_t>(ct_.baseM))
                    ? (rowsCore_ - done) : static_cast<uint32_t>(ct_.baseM);
                ProcessTile(cTile, rowStart_ + done, rows);
            }
            tile++;
        }
        matmulObj_.End();
    }

private:
    __aicore__ inline void LoadConstants()
    {
        auto scale = ub_.GetWithOffset<float>(NL, offScale_);
        auto base = ub_.GetWithOffset<float>(NL, offBase_);
        DataCopy(scale, sGm_, NL);
        DataCopy(base, bGm_, NL);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    // sigmoid over one 8-lane window: dst = 1 / (1 + exp(-z))
    __aicore__ inline void Sigmoid8(const LocalTensor<float> &z, const LocalTensor<float> &e,
        const LocalTensor<float> &d, const LocalTensor<float> &one, const LocalTensor<float> &dst)
    {
        Muls(e, z, -1.0f, 8);
        Exp(e, e, 8);
        Add(d, one, e, 8);
        Div(dst, one, d, 8);
    }

    __aicore__ inline void ProcessTile(const LocalTensor<float> &c, const uint32_t r0, const uint32_t rows)
    {
        for (uint32_t g = 0; g < rows; g += ROWS) {
            const uint32_t rr = (rows - g < static_cast<uint32_t>(ROWS)) ? (rows - g) : ROWS;
            ProcessGroup(c[g * NL], r0 + g, rr);
        }
    }

    __aicore__ inline void ProcessGroup(const LocalTensor<float> &cGrp, const uint32_t r0, const uint32_t rows)
    {
        auto xBf = ub_.GetWithOffset<bfloat16_t>(ROWS * eh_, offXbf_);
        auto x32 = ub_.GetWithOffset<float>(ROWS * h_, offX32_);
        auto y32 = ub_.GetWithOffset<float>(ROWS * h_, offY32_);
        auto yBf = ub_.GetWithOffset<bfloat16_t>(ROWS * h_, offYbf_);
        auto l32 = ub_.GetWithOffset<float>(ROWS * NL, offL32_);
        auto c32 = ub_.GetWithOffset<float>(ROWS * NL, offC32_);
        auto lgBf = ub_.GetWithOffset<bfloat16_t>(ROWS * NL, offLgBf_);
        auto scale = ub_.GetWithOffset<float>(NL, offScale_);
        auto base = ub_.GetWithOffset<float>(NL, offBase_);
        auto z8 = ub_.GetWithOffset<float>(ROWS * 8, offZ8_);
        auto e8 = ub_.GetWithOffset<float>(ROWS * 8, offE8_);
        auto d8 = ub_.GetWithOffset<float>(ROWS * 8, offD8_);
        auto one8 = ub_.GetWithOffset<float>(ROWS * 8, offOne8_);
        auto hpre8 = ub_.GetWithOffset<float>(ROWS * 8, offHpre_);
        auto hpost8 = ub_.GetWithOffset<float>(ROWS * 8, offHpost_);
        auto rsum = ub_.GetWithOffset<float>(ROWS * 8, offRsum_);
        auto rstd = ub_.GetWithOffset<float>(ROWS, offRstd_);
        auto rtmp = ub_.GetWithOffset<float>(128, offRtmp_);

        // the matmul C tile for this group: 8 rows x 32 lanes fp32
        DataCopy(l32, cGrp, ROWS * NL);
        DataCopy(xBf, xGm_[r0 * eh_], rows * eh_);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);


        // pass1: rstd per row, accumulated chunk by chunk in scalar registers
        float sumSq[ROWS];
        for (int32_t r = 0; r < ROWS; r++) {
            sumSq[r] = 0.0f;
        }
        for (int32_t c = 0; c < HC; c++) {
            for (int32_t r = 0; r < ROWS; r++) {
                auto src = xBf[r * eh_ + c * h_];
                auto dst = x32[r * h_];
                Cast(dst, src, RoundMode::CAST_NONE, h_);
                Mul(dst, dst, dst, h_);
                ReduceSum(rsum[r * 8], dst, rtmp, h_);
            }
            SyncVS();
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                sumSq[r] += rsum.GetValue(r * 8);
            }
        }

        // re-route scale through the vector pipe so scalars can read it
        Muls(scale, scale, 1.0f, NL);
        SyncVS();
        const float s0 = scale.GetValue(0);
        const float s1 = scale.GetValue(E_ROWS);
        const float s2 = scale.GetValue(2 * E_ROWS);

        // l *= rstd per row
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            const float rstdV = 1.0f / sqrt(sumSq[r] / eh_ + EPS);
            rstd.SetValue(r, rstdV);
            Muls(l32[r * NL], l32[r * NL], rstdV, NL);
        }

        Duplicate(one8, 1.0f, ROWS * 8);

        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            // pass A over l32[0:8]: h_pre valid at lanes 0:4
            Muls(z8[r * 8], l32[r * NL], s0, 8);
            Add(z8[r * 8], z8[r * 8], base, 8);
            Sigmoid8(z8[r * 8], e8[r * 8], d8[r * 8], one8[r * 8], hpre8[r * 8]);

            // pass B over the same window with s1: h_post valid at lanes 4:8
            Muls(z8[r * 8], l32[r * NL], s1, 8);
            Add(z8[r * 8], z8[r * 8], base, 8);
            Sigmoid8(z8[r * 8], e8[r * 8], d8[r * 8], one8[r * 8], hpost8[r * 8]);
            Muls(hpost8[r * 8], hpost8[r * 8], 2.0f, 8);

            // res group at l32[8:32]
            Muls(c32[r * NL + 8], l32[r * NL + 8], s2, NRES);
            Add(c32[r * NL + 8], c32[r * NL + 8], base[8], NRES);
        }
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            ReduceMax(rsum[r * 8], c32[r * NL + 8], rtmp, NRES);
        }
        SyncVS();
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            Adds(c32[r * NL + 8], c32[r * NL + 8], -rsum.GetValue(r * 8), NRES);
            Exp(c32[r * NL + 8], c32[r * NL + 8], NRES);
        }
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            ReduceSum(rsum[r * 8], c32[r * NL + 8], rtmp, NRES);
        }
        SyncVS();
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            Muls(c32[r * NL + 8], c32[r * NL + 8], 1.0f / rsum.GetValue(r * 8), NRES);
        }

        // y = sum_i h_pre[:, i] * x[:, i*h : (i+1)*h]
        SyncVS();
        for (int32_t c = 0; c < HC; c++) {
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                auto src = xBf[r * eh_ + c * h_];
                auto castDst = x32[r * h_];
                Cast(castDst, src, RoundMode::CAST_NONE, h_);
                const float s = hpre8.GetValue(r * 8 + c);
                if (c == 0) {
                    Muls(y32[r * h_], castDst, s, h_);
                } else {
                    Muls(castDst, castDst, s, h_);
                    Add(y32[r * h_], y32[r * h_], castDst, h_);
                }
            }
        }

        Cast(yBf, y32, RoundMode::CAST_RINT, rows * h_);
        Cast(lgBf, l32, RoundMode::CAST_RINT, rows * NL);
        // route the scalar-written rstd lane through the vector pipe so the
        // MTE3 copy below is ordered after it; full ROWS width keeps the GM
        // burst aligned, tail-block rows beyond sb are sliced off by the host
        SyncSV();
        Muls(rstd, rstd, 1.0f, ROWS);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yGm_[r0 * h_], yBf, rows * h_);
        DataCopy(hpGm_[r0 * 8], hpre8, rows * 8);
        DataCopy(hoGm_[r0 * 8], hpost8, rows * 8);
        DataCopy(coGm_[r0 * NL], c32, rows * NL);
        DataCopy(lgGm_[r0 * NL], lgBf, rows * NL);
        DataCopy(rGm_[r0], rstd, ROWS);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    static constexpr int32_t E_ROWS = 4;  // lanes per head group

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> cBuf_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> aGm_;
    GlobalTensor<float> sGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<bfloat16_t> yGm_;
    GlobalTensor<float> hpGm_;
    GlobalTensor<float> hoGm_;
    GlobalTensor<float> coGm_;
    GlobalTensor<float> rGm_;
    GlobalTensor<bfloat16_t> lgGm_;
    TCubeTiling ct_;
    uint32_t sb_ = 0;
    uint32_t h_ = 0;
    uint32_t eh_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowsCore_ = 0;
    uint32_t mSingleBlocks_ = 1;
    uint32_t offXbf_ = 0;
    uint32_t offX32_ = 0;
    uint32_t offY32_ = 0;
    uint32_t offYbf_ = 0;
    uint32_t offL32_ = 0;
    uint32_t offC32_ = 0;
    uint32_t offLgBf_ = 0;
    uint32_t offScale_ = 0;
    uint32_t offBase_ = 0;
    uint32_t offZ8_ = 0;
    uint32_t offE8_ = 0;
    uint32_t offD8_ = 0;
    uint32_t offOne8_ = 0;
    uint32_t offHpre_ = 0;
    uint32_t offHpost_ = 0;
    uint32_t offRsum_ = 0;
    uint32_t offRstd_ = 0;
    uint32_t offRtmp_ = 0;
};

extern "C" __global__ __aicore__ void lite_pre_fused(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR base,
    GM_ADDR y, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR coeff, GM_ADDR rstd, GM_ADDR logits,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelLitePreFused op;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj_, &tilingData.cubeTilingData);
    op.Init(x, w, scale, base, y, h_pre, h_post, coeff, rstd, logits, &tilingData, &pipe);
    op.Process();
}

#ifndef __CCE_KT_TEST__
extern "C" void lite_pre_fused_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *w,
    uint8_t *scale, uint8_t *base, uint8_t *y, uint8_t *h_pre, uint8_t *h_post,
    uint8_t *coeff, uint8_t *rstd, uint8_t *logits, uint8_t *workspace, uint8_t *tiling)
{
    lite_pre_fused<<<blockDim, l2ctrl, stream>>>(x, w, scale, base, y, h_pre, h_post, coeff,
        rstd, logits, workspace, tiling);
}
#endif
