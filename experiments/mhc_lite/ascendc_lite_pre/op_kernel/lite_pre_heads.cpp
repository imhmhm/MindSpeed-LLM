/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Device side of the LitePreHeads custom operator (mhc_lite pre, hc = 4).
 *
 * One vector-core block handles ROWS = 8 tokens end to end:
 *   pass1  rstd = rsqrt(mean(x^2) + eps) over the [4h] row
 *   heads  l = logits * rstd (fp32);  h_pre = sigmoid(s0*l_pre + b_pre);
 *          h_post = 2*sigmoid(s1*l_post + b_post);
 *          coeff = softmax(s2*l_res + b_res)
 *   y      y = sum_i h_pre[i] * x[i*h : (i+1)*h], cast to bf16
 * Per-row scalars ride scalar registers (GetValue + Muls), so no
 * broadcast staging buffers are materialized.
 *
 * Every vector op is the simple count form over one row's window, the
 * emission style the validated tilelang prototype used: window bases sit
 * on 8-lane (32B) boundaries and no repeat-stride forms appear.  The pre
 * and post groups share the aligned l32[0:8] window, evaluated by two
 * scalar passes -- pass A (scale s0) leaves h_pre at lanes 0:4, pass B
 * (scale s1) leaves 2*sigmoid at lanes 4:8; the res group reads
 * l32[8:32].  Reduce results land in 32B-aligned rsum slots.
 *
 * The tiling struct arrives via the auto-generated kernel_tiling.h that
 * the opc tool derives from REGISTER_TILING_DATA_CLASS in op_host.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t ROWS = 8;   // tokens per block
constexpr int32_t HC = 4;     // mhc head count
constexpr int32_t NL = 32;    // logits width: pre 0:4 | post 4:8 | res 8:32
constexpr int32_t NRES = 24;

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
}

class KernelLitePreHeads {
public:
    __aicore__ inline KernelLitePreHeads() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR logits, GM_ADDR scale, GM_ADDR base,
        GM_ADDR eps, GM_ADDR y, GM_ADDR hPre, GM_ADDR hPost, GM_ADDR coeff, GM_ADDR rstd,
        LitePreHeadsTilingData *tiling)
    {
        sb_ = tiling->sb;
        h_ = tiling->h;
        eh_ = HC * h_;
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)x);
        lGm_.SetGlobalBuffer((__gm__ bfloat16_t *)logits);
        sGm_.SetGlobalBuffer((__gm__ float *)scale);
        bGm_.SetGlobalBuffer((__gm__ float *)base);
        eGm_.SetGlobalBuffer((__gm__ float *)eps);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t *)y);
        hpGm_.SetGlobalBuffer((__gm__ float *)hPre);
        hoGm_.SetGlobalBuffer((__gm__ float *)hPost);
        coGm_.SetGlobalBuffer((__gm__ float *)coeff);
        rGm_.SetGlobalBuffer((__gm__ float *)rstd);

        const uint32_t xBfBytes = ROWS * eh_ * sizeof(bfloat16_t);
        const uint32_t x32Bytes = ROWS * h_ * sizeof(float);
        const uint32_t yBfBytes = xBfBytes / 2;
        uint32_t off = 0;
        offXbf_ = off;    off += xBfBytes;
        offX32_ = off;    off += x32Bytes;
        offY32_ = off;    off += x32Bytes;
        offYbf_ = off;    off += yBfBytes;
        offLbf_ = off;    off += ROWS * NL * sizeof(bfloat16_t);
        offL32_ = off;    off += ROWS * NL * sizeof(float);
        offC32_ = off;    off += ROWS * NL * sizeof(float);
        offScale_ = off;  off += NL * sizeof(float);
        offBase_ = off;   off += NL * sizeof(float);
        offEps_ = off;    off += 8 * sizeof(float);
        offZ8_ = off;     off += ROWS * 8 * sizeof(float);
        offE8_ = off;     off += ROWS * 8 * sizeof(float);
        offD8_ = off;     off += ROWS * 8 * sizeof(float);
        offOne8_ = off;   off += ROWS * 8 * sizeof(float);
        offHpre_ = off;   off += ROWS * 8 * sizeof(float);
        offHpost_ = off;  off += ROWS * 8 * sizeof(float);
        offRsum_ = off;   off += ROWS * 8 * sizeof(float);
        offRstd_ = off;   off += ROWS * sizeof(float);
        offRtmp_ = off;   off += 512;
        pipe_.InitBuffer(ub_, off);
    }

    __aicore__ inline void Process()
    {
        const uint32_t nBlocks = (sb_ + ROWS - 1) / ROWS;
        const uint32_t blockDim = GetBlockNum();
        for (uint32_t b = GetBlockIdx(); b < nBlocks; b += blockDim) {
            const uint32_t r0 = b * ROWS;
            const uint32_t rows = (sb_ - r0 < ROWS) ? (sb_ - r0) : ROWS;
            ProcessBlock(r0, rows);
        }
    }

private:
    // sigmoid over one 8-lane window: dst = 1 / (1 + exp(-z))
    __aicore__ inline void Sigmoid8(const LocalTensor<float> &z, const LocalTensor<float> &e,
        const LocalTensor<float> &d, const LocalTensor<float> &one, const LocalTensor<float> &dst)
    {
        Muls(e, z, -1.0f, 8);
        Exp(e, e, 8);
        Add(d, one, e, 8);
        Div(dst, one, d, 8);
    }

    __aicore__ inline void ProcessBlock(const uint32_t r0, const uint32_t rows)
    {
        auto xBf = ub_.GetWithOffset<bfloat16_t>(ROWS * eh_, offXbf_);
        auto x32 = ub_.GetWithOffset<float>(ROWS * h_, offX32_);
        auto y32 = ub_.GetWithOffset<float>(ROWS * h_, offY32_);
        auto yBf = ub_.GetWithOffset<bfloat16_t>(ROWS * h_, offYbf_);
        auto lBf = ub_.GetWithOffset<bfloat16_t>(ROWS * NL, offLbf_);
        auto l32 = ub_.GetWithOffset<float>(ROWS * NL, offL32_);
        auto c32 = ub_.GetWithOffset<float>(ROWS * NL, offC32_);
        auto scale = ub_.GetWithOffset<float>(NL, offScale_);
        auto base = ub_.GetWithOffset<float>(NL, offBase_);
        auto eps8 = ub_.GetWithOffset<float>(8, offEps_);
        auto z8 = ub_.GetWithOffset<float>(ROWS * 8, offZ8_);
        auto e8 = ub_.GetWithOffset<float>(ROWS * 8, offE8_);
        auto d8 = ub_.GetWithOffset<float>(ROWS * 8, offD8_);
        auto one8 = ub_.GetWithOffset<float>(ROWS * 8, offOne8_);
        auto hpre8 = ub_.GetWithOffset<float>(ROWS * 8, offHpre_);
        auto hpost8 = ub_.GetWithOffset<float>(ROWS * 8, offHpost_);
        auto rsum = ub_.GetWithOffset<float>(ROWS * 8, offRsum_);
        auto rstd = ub_.GetWithOffset<float>(ROWS, offRstd_);
        auto rtmp = ub_.GetWithOffset<float>(128, offRtmp_);

        DataCopy(xBf, xGm_[r0 * eh_], rows * eh_);
        DataCopy(lBf, lGm_[r0 * NL], rows * NL);
        DataCopy(scale, sGm_, NL);
        DataCopy(base, bGm_, NL);
        DataCopy(eps8, eGm_, 8);  // 8 lanes = 32B, the copy granularity
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
            for (int32_t r = 0; r < rows; r++) {
                sumSq[r] += rsum.GetValue(r * 8);
            }
        }

        // re-route scale/eps through the vector pipe so scalars can read them
        Muls(scale, scale, 1.0f, NL);
        Muls(eps8, eps8, 1.0f, 8);
        SyncVS();
        const float s0 = scale.GetValue(0);
        const float s1 = scale.GetValue(E_ROWS);
        const float s2 = scale.GetValue(2 * E_ROWS);
        const float epsV = eps8.GetValue(0);

        // heads: l = logits * rstd
        Cast(l32, lBf, RoundMode::CAST_NONE, ROWS * NL);
        for (int32_t r = 0; r < rows; r++) {
            const float rstdV = 1.0f / sqrt(sumSq[r] / eh_ + epsV);
            rstd.SetValue(r, rstdV);
            Muls(l32[r * NL], l32[r * NL], rstdV, NL);
        }

        Duplicate(one8, 1.0f, ROWS * 8);

        for (int32_t r = 0; r < rows; r++) {
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
        for (int32_t r = 0; r < rows; r++) {
            ReduceMax(rsum[r * 8], c32[r * NL + 8], rtmp, NRES);
        }
        SyncVS();
        for (int32_t r = 0; r < rows; r++) {
            Adds(c32[r * NL + 8], c32[r * NL + 8], -rsum.GetValue(r * 8), NRES);
            Exp(c32[r * NL + 8], c32[r * NL + 8], NRES);
        }
        for (int32_t r = 0; r < rows; r++) {
            ReduceSum(rsum[r * 8], c32[r * NL + 8], rtmp, NRES);
        }
        SyncVS();
        for (int32_t r = 0; r < rows; r++) {
            Muls(c32[r * NL + 8], c32[r * NL + 8], 1.0f / rsum.GetValue(r * 8), NRES);
        }

        // y = sum_i h_pre[:, i] * x[:, i*h : (i+1)*h]
        SyncVS();
        for (int32_t c = 0; c < HC; c++) {
            for (int32_t r = 0; r < rows; r++) {
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
        DataCopy(rGm_[r0], rstd, ROWS);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

private:
    static constexpr int32_t E_ROWS = 4;  // lanes per head group

    TPipe pipe_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> lGm_;
    GlobalTensor<float> sGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> eGm_;
    GlobalTensor<bfloat16_t> yGm_;
    GlobalTensor<float> hpGm_;
    GlobalTensor<float> hoGm_;
    GlobalTensor<float> coGm_;
    GlobalTensor<float> rGm_;
    uint32_t sb_ = 0;
    uint32_t h_ = 0;
    uint32_t eh_ = 0;
    uint32_t offXbf_ = 0;
    uint32_t offX32_ = 0;
    uint32_t offY32_ = 0;
    uint32_t offYbf_ = 0;
    uint32_t offLbf_ = 0;
    uint32_t offL32_ = 0;
    uint32_t offC32_ = 0;
    uint32_t offScale_ = 0;
    uint32_t offBase_ = 0;
    uint32_t offEps_ = 0;
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

extern "C" __global__ __aicore__ void lite_pre_heads(
    GM_ADDR x, GM_ADDR logits, GM_ADDR scale, GM_ADDR base, GM_ADDR eps,
    GM_ADDR y, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR coeff, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelLitePreHeads op;
    op.Init(x, logits, scale, base, eps, y, h_pre, h_post, coeff, rstd, &tilingData);
    op.Process();
}

#ifndef __CCE_KT_TEST__
extern "C" void lite_pre_heads_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *logits,
    uint8_t *scale, uint8_t *base, uint8_t *eps, uint8_t *y, uint8_t *h_pre, uint8_t *h_post, uint8_t *coeff, uint8_t *rstd,
    uint8_t *workspace, uint8_t *tiling)
{
    lite_pre_heads<<<blockDim, l2ctrl, stream>>>(x, logits, scale, base, eps, y, h_pre, h_post, coeff, rstd, workspace, tiling);
}
#endif
