/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Device side of the LitePreGrad custom operator (mhc_lite pre, hc = 4).
 *
 * Single AIV kernel producing the whole scheme-E pre backward except the
 * two GEMMs autograd runs outside (grad_wp, d_xf):
 *   dw_i  = <g, x_i>                        (y path, 4 chunk loops)
 *   pre   : dz = dw_i * sig'(l*s0 + b_i);   dl = dz*s0
 *   post  : dz = ghpost_i * 2sig'(l*s1 + b_{4+i});  dl = dz*s1
 *   res   : dzc = coeff*(dcoeff - <dcoeff,coeff>);  dl = dzc*s2,
 *           with dcoeff = ghres @ perm_t folded in (Axpy per basis row)
 *   drstd = <dl, raw logits>;  coef = -drstd * rstd^3 / (4h)
 *   dx    = h_pre_i * g + coef * x_i         (fp32 math, one bf16 rounding)
 *   dlogits = dl * rstd                       (single bf16 cast, straight out)
 *   dscale / dbase: per-core accumulators, one atomic add per core; the
 *           wrapper's lane-count fix (1/4, 1/4, 1/24) is folded in here
 *
 * l = raw logits * rstd is recomputed exactly like the forward op, so the
 * epilogue differentiates along the values the forward produced.  The
 * pre/post passes share the 8-lane window over l[0:8] (inert lanes carry
 * zeros: dw is zero past lane 4, ghpost is placed at lanes 4:8), and
 * every 24-lane res buffer is strided at 32 lanes because a 24-float row
 * stride (96B) would put vector windows off 32B alignment.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t ROWS = 8;       // tokens per group
constexpr int32_t HC = 4;         // mhc head count
constexpr int32_t NL = 32;        // logit lanes: pre 0:4 | post 4:8 | res 8:32
constexpr int32_t NRES = 24;
constexpr int32_t NRES_PAD = 32;  // 24-lane rows padded to keep windows aligned
constexpr int32_t EE = 16;        // permutation basis rows (e*e)
constexpr int32_t RES_OFF = 8;    // first res lane inside the 32-lane row

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

class KernelLitePreGrad {
public:
    __aicore__ inline KernelLitePreGrad() {}

    __aicore__ inline void Init(GM_ADDR g, GM_ADDR x, GM_ADDR ghpost, GM_ADDR ghres, GM_ADDR perm,
        GM_ADDR logits, GM_ADDR rstd, GM_ADDR hpre, GM_ADDR scale, GM_ADDR base,
        GM_ADDR dlogits, GM_ADDR dx, GM_ADDR dscale, GM_ADDR dbase, LitePreGradTilingData *tiling,
        TPipe *pipe)
    {
        sb_ = tiling->sb;
        h_ = tiling->h;
        eh_ = HC * h_;
        pipe_ = pipe;
        gGm_.SetGlobalBuffer((__gm__ bfloat16_t *)g);
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)x);
        ghpGm_.SetGlobalBuffer((__gm__ float *)ghpost);
        ghrGm_.SetGlobalBuffer((__gm__ float *)ghres);
        pmGm_.SetGlobalBuffer((__gm__ float *)perm);
        lgGm_.SetGlobalBuffer((__gm__ bfloat16_t *)logits);
        rGm_.SetGlobalBuffer((__gm__ float *)rstd);
        hpGm_.SetGlobalBuffer((__gm__ float *)hpre);
        sGm_.SetGlobalBuffer((__gm__ float *)scale);
        bGm_.SetGlobalBuffer((__gm__ float *)base);
        dlGm_.SetGlobalBuffer((__gm__ bfloat16_t *)dlogits);
        dxGm_.SetGlobalBuffer((__gm__ bfloat16_t *)dx);
        dsGm_.SetGlobalBuffer((__gm__ float *)dscale);
        dbGm_.SetGlobalBuffer((__gm__ float *)dbase);

        const uint32_t gBfBytes = ROWS * h_ * sizeof(bfloat16_t);
        const uint32_t g32Bytes = ROWS * h_ * sizeof(float);
        uint32_t off = 0;
        offGbf_ = off;    off += gBfBytes;
        offXbf_ = off;    off += gBfBytes;
        offG32_ = off;    off += g32Bytes;
        offX32_ = off;    off += g32Bytes;
        offGx32_ = off;   off += g32Bytes;
        offGxbf_ = off;   off += gBfBytes;
        offLgBf_ = off;   off += ROWS * NL * sizeof(bfloat16_t);
        offRaw_ = off;    off += ROWS * NL * sizeof(float);
        offL32_ = off;    off += ROWS * NL * sizeof(float);
        offDlA_ = off;    off += ROWS * 8 * sizeof(float);
        offDlB_ = off;    off += ROWS * 8 * sizeof(float);
        offDl32_ = off;   off += ROWS * NL * sizeof(float);
        offDlBf_ = off;   off += ROWS * NL * sizeof(bfloat16_t);
        offZ8_ = off;     off += ROWS * 8 * sizeof(float);
        offE8_ = off;     off += ROWS * 8 * sizeof(float);
        offD8_ = off;     off += ROWS * 8 * sizeof(float);
        offOne8_ = off;   off += ROWS * 8 * sizeof(float);
        offSig8_ = off;   off += ROWS * 8 * sizeof(float);
        offT8A_ = off;    off += ROWS * 8 * sizeof(float);
        offT8B_ = off;    off += ROWS * 8 * sizeof(float);
        offDsA_ = off;    off += ROWS * 8 * sizeof(float);
        offDbA_ = off;    off += ROWS * 8 * sizeof(float);
        offDsB_ = off;    off += ROWS * 8 * sizeof(float);
        offDbB_ = off;    off += ROWS * 8 * sizeof(float);
        offDw8_ = off;    off += ROWS * 8 * sizeof(float);
        offGh8_ = off;    off += ROWS * 8 * sizeof(float);
        offGhp4_ = off;   off += ROWS * 4 * sizeof(float);
        offHpre_ = off;   off += ROWS * 8 * sizeof(float);
        offRstd_ = off;   off += ROWS * sizeof(float);
        offCoef_ = off;   off += ROWS * 4 * sizeof(float);
        offCoefP_ = off;  off += ROWS * NRES_PAD * sizeof(float);
        offDco_ = off;    off += ROWS * NRES_PAD * sizeof(float);
        offDzc_ = off;    off += ROWS * NRES_PAD * sizeof(float);
        offT24_ = off;    off += ROWS * NRES_PAD * sizeof(float);
        offGhr_ = off;    off += ROWS * EE * sizeof(float);
        offPerm_ = off;   off += EE * NRES_PAD * sizeof(float);
        offScale_ = off;  off += NL * sizeof(float);
        offBase_ = off;   off += NL * sizeof(float);
        offDs_ = off;     off += 8 * sizeof(float);
        offDb_ = off;     off += NL * sizeof(float);
        offRsum_ = off;   off += ROWS * NL * sizeof(float);
        offRtmp_ = off;   off += 512;
        pipe_->InitBuffer(ub_, off);
    }

    __aicore__ inline void Process()
    {
        LoadConstants();
        auto dsAcc = ub_.GetWithOffset<float>(8, offDs_);
        auto dbAcc = ub_.GetWithOffset<float>(NL, offDb_);
        Duplicate(dsAcc, 0.0f, 8);
        Duplicate(dbAcc, 0.0f, NL);

        const uint32_t nBlocks = Ceiling(sb_, ROWS);
        const uint32_t blockDim = GetBlockNum();
        for (uint32_t blk = GetBlockIdx(); blk < nBlocks; blk += blockDim) {
            const uint32_t r0 = blk * ROWS;
            const uint32_t rows = (sb_ - r0 < static_cast<uint32_t>(ROWS)) ? (sb_ - r0) : ROWS;
            ProcessGroup(r0, rows, dsAcc, dbAcc);
        }

        // one atomic add per core into the zero-initialized outputs; the
        // identity Muls order the scalar-written accumulator lanes too
        SyncSV();
        Muls(dsAcc, dsAcc, 1.0f, 8);
        Muls(dbAcc, dbAcc, 1.0f, NL);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        SetAtomicAdd<float>();
        DataCopy(dsGm_, dsAcc, 8);
        DataCopy(dbGm_, dbAcc, NL);
        SetAtomicNone();
    }

private:
    __aicore__ inline void LoadConstants()
    {
        auto scale = ub_.GetWithOffset<float>(NL, offScale_);
        auto base = ub_.GetWithOffset<float>(NL, offBase_);
        auto perm = ub_.GetWithOffset<float>(EE * NRES_PAD, offPerm_);
        DataCopy(scale, sGm_, NL);
        DataCopy(base, bGm_, NL);
        for (int32_t k = 0; k < EE; k++) {
            DataCopy(perm[k * NRES_PAD], pmGm_[k * NRES], NRES);
        }
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

    __aicore__ inline void ProcessGroup(const uint32_t r0, const uint32_t rows,
        const LocalTensor<float> &dsAcc, const LocalTensor<float> &dbAcc)
    {
        auto gBf = ub_.GetWithOffset<bfloat16_t>(ROWS * h_, offGbf_);
        auto xBf = ub_.GetWithOffset<bfloat16_t>(ROWS * h_, offXbf_);
        auto g32 = ub_.GetWithOffset<float>(ROWS * h_, offG32_);
        auto x32 = ub_.GetWithOffset<float>(ROWS * h_, offX32_);
        auto gx32 = ub_.GetWithOffset<float>(ROWS * h_, offGx32_);
        auto gxBf = ub_.GetWithOffset<bfloat16_t>(ROWS * h_, offGxbf_);
        auto lgBf = ub_.GetWithOffset<bfloat16_t>(ROWS * NL, offLgBf_);
        auto rawF = ub_.GetWithOffset<float>(ROWS * NL, offRaw_);
        auto l32 = ub_.GetWithOffset<float>(ROWS * NL, offL32_);
        auto dlA = ub_.GetWithOffset<float>(ROWS * 8, offDlA_);
        auto dlB = ub_.GetWithOffset<float>(ROWS * 8, offDlB_);
        auto dl32 = ub_.GetWithOffset<float>(ROWS * NL, offDl32_);
        auto dlBf = ub_.GetWithOffset<bfloat16_t>(ROWS * NL, offDlBf_);
        auto z8 = ub_.GetWithOffset<float>(ROWS * 8, offZ8_);
        auto e8 = ub_.GetWithOffset<float>(ROWS * 8, offE8_);
        auto d8 = ub_.GetWithOffset<float>(ROWS * 8, offD8_);
        auto one8 = ub_.GetWithOffset<float>(ROWS * 8, offOne8_);
        auto sig8 = ub_.GetWithOffset<float>(ROWS * 8, offSig8_);
        auto t8a = ub_.GetWithOffset<float>(ROWS * 8, offT8A_);
        auto t8b = ub_.GetWithOffset<float>(ROWS * 8, offT8B_);
        auto dsA = ub_.GetWithOffset<float>(ROWS * 8, offDsA_);
        auto dbA = ub_.GetWithOffset<float>(ROWS * 8, offDbA_);
        auto dsB = ub_.GetWithOffset<float>(ROWS * 8, offDsB_);
        auto dbB = ub_.GetWithOffset<float>(ROWS * 8, offDbB_);
        auto dw8 = ub_.GetWithOffset<float>(ROWS * 8, offDw8_);
        auto gh8 = ub_.GetWithOffset<float>(ROWS * 8, offGh8_);
        auto ghp4 = ub_.GetWithOffset<float>(ROWS * 4, offGhp4_);
        auto hpre8 = ub_.GetWithOffset<float>(ROWS * 8, offHpre_);
        auto rstd = ub_.GetWithOffset<float>(ROWS, offRstd_);
        auto coef4 = ub_.GetWithOffset<float>(ROWS * 4, offCoef_);
        auto coefP = ub_.GetWithOffset<float>(ROWS * NRES_PAD, offCoefP_);
        auto dco = ub_.GetWithOffset<float>(ROWS * NRES_PAD, offDco_);
        auto dzc = ub_.GetWithOffset<float>(ROWS * NRES_PAD, offDzc_);
        auto t24 = ub_.GetWithOffset<float>(ROWS * NRES_PAD, offT24_);
        auto ghr = ub_.GetWithOffset<float>(ROWS * EE, offGhr_);
        auto perm = ub_.GetWithOffset<float>(EE * NRES_PAD, offPerm_);
        auto scale = ub_.GetWithOffset<float>(NL, offScale_);
        auto base = ub_.GetWithOffset<float>(NL, offBase_);
        auto rsum = ub_.GetWithOffset<float>(ROWS * NL, offRsum_);
        auto rtmp = ub_.GetWithOffset<float>(128, offRtmp_);

        DataCopy(lgBf, lgGm_[r0 * NL], rows * NL);
        DataCopy(gBf, gGm_[r0 * h_], rows * h_);
        // ghpost is 4 floats/row; a tail group would copy just 16B, under
        // the copy granularity -- the wrapper pads ghpost to whole groups,
        // so always read a full ROWS*4 block
        DataCopy(ghp4, ghpGm_[r0 * HC], ROWS * HC);
        DataCopy(ghr, ghrGm_[r0 * EE], rows * EE);
        DataCopy(hpre8, hpGm_[r0 * 8], rows * 8);
        DataCopy(rstd, rGm_[r0], ROWS);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Cast(rawF, lgBf, RoundMode::CAST_NONE, rows * NL);
        Cast(g32, gBf, RoundMode::CAST_NONE, rows * h_);
        // route the MTE2-loaded scalars through the vector pipe so scalar
        // reads (rstd, ghres, hpre) are ordered after the copy
        Muls(rstd, rstd, 1.0f, ROWS);
        Muls(ghr, ghr, 1.0f, rows * EE);
        Muls(hpre8, hpre8, 1.0f, rows * 8);
        Duplicate(one8, 1.0f, ROWS * 8);
        Duplicate(dw8, 0.0f, ROWS * 8);
        Duplicate(gh8, 0.0f, ROWS * 8);
        Duplicate(dsA, 0.0f, ROWS * 8);
        Duplicate(dbA, 0.0f, ROWS * 8);
        Duplicate(dsB, 0.0f, ROWS * 8);
        Duplicate(dbB, 0.0f, ROWS * 8);

        // ghpost rides lanes 4:8 so the post pass shares the pre window
        SyncVS();
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            for (int32_t i = 0; i < HC; i++) {
                gh8.SetValue(r * 8 + HC + i, ghp4.GetValue(r * HC + i));
            }
        }
        SyncSV();

        // scale scalars: re-route through the vector pipe
        Muls(scale, scale, 1.0f, NL);
        SyncVS();
        const float s0 = scale.GetValue(0);
        const float s1 = scale.GetValue(HC);
        const float s2 = scale.GetValue(2 * HC);

        // l = raw * rstd per row (the exact forward values)
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            Muls(l32[r * NL], rawF[r * NL], rstd.GetValue(r), NL);
        }

        // dw[r, c] = <g, x_c> per row, chunk by chunk; chunk c of one row is
        // eh away from chunk c of the next, so the x tiles are copied per
        // row.  Each reduction lands in its own aligned 8-lane window (a
        // strided single-lane dst would be written at vector width and
        // clobber its neighbours), then one scalar pass packs lane 0 of
        // each window into dw8
        for (int32_t c = 0; c < HC; c++) {
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                DataCopy(xBf[r * h_], xGm_[(r0 + r) * eh_ + c * h_], h_);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                Cast(x32[r * h_], xBf[r * h_], RoundMode::CAST_NONE, h_);
                Mul(x32[r * h_], x32[r * h_], g32[r * h_], h_);
                ReduceSum(rsum[r * NL + c * 8], x32[r * h_], rtmp, h_);
            }
        }
        SyncVS();
        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            for (int32_t c = 0; c < HC; c++) {
                dw8.SetValue(r * 8 + c, rsum.GetValue(r * NL + c * 8));
            }
        }
        SyncSV();

        for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
            // pass A (s0): dz lives in lanes 0:4 (dw is zero past lane 4)
            Muls(z8[r * 8], l32[r * NL], s0, 8);
            Add(z8[r * 8], z8[r * 8], base, 8);
            Sigmoid8(z8[r * 8], e8[r * 8], d8[r * 8], one8[r * 8], sig8[r * 8]);
            Sub(d8[r * 8], one8[r * 8], sig8[r * 8], 8);
            Mul(d8[r * 8], d8[r * 8], sig8[r * 8], 8);    // sig'
            Mul(d8[r * 8], d8[r * 8], dw8[r * 8], 8);     // dz
            Muls(dlA[r * 8], d8[r * 8], s0, 8);
            Mul(t8a[r * 8], d8[r * 8], l32[r * NL], 8);   // dz * l
            Add(dsA[r * 8], dsA[r * 8], t8a[r * 8], 8);
            Add(dbA[r * 8], dbA[r * 8], d8[r * 8], 8);

            // pass B (s1): dz lives in lanes 4:8 (ghpost placed there)
            Muls(z8[r * 8], l32[r * NL], s1, 8);
            Add(z8[r * 8], z8[r * 8], base, 8);
            Sigmoid8(z8[r * 8], e8[r * 8], d8[r * 8], one8[r * 8], sig8[r * 8]);
            Sub(e8[r * 8], one8[r * 8], sig8[r * 8], 8);
            Mul(e8[r * 8], e8[r * 8], sig8[r * 8], 8);
            Muls(e8[r * 8], e8[r * 8], 2.0f, 8);          // 2*sig'
            Mul(d8[r * 8], e8[r * 8], gh8[r * 8], 8);     // dz
            Muls(dlB[r * 8], d8[r * 8], s1, 8);
            Mul(t8b[r * 8], d8[r * 8], l32[r * NL], 8);   // dz * l
            Add(dsB[r * 8], dsB[r * 8], t8b[r * 8], 8);
            Add(dbB[r * 8], dbB[r * 8], d8[r * 8], 8);

            // res head: softmax jacobian, dcoeff = ghres @ perm folded in
            Muls(coefP[r * NRES_PAD], l32[r * NL + RES_OFF], s2, NRES);
            Add(coefP[r * NRES_PAD], coefP[r * NRES_PAD], base[RES_OFF], NRES);
            ReduceMax(rsum[r * 8], coefP[r * NRES_PAD], rtmp, NRES);
            SyncVS();
            Adds(coefP[r * NRES_PAD], coefP[r * NRES_PAD], -rsum.GetValue(r * 8), NRES);
            Exp(coefP[r * NRES_PAD], coefP[r * NRES_PAD], NRES);
            ReduceSum(rsum[r * 8], coefP[r * NRES_PAD], rtmp, NRES);
            SyncVS();
            Muls(coefP[r * NRES_PAD], coefP[r * NRES_PAD], 1.0f / rsum.GetValue(r * 8), NRES);

            Duplicate(dco[r * NRES_PAD], 0.0f, NRES_PAD);
            for (int32_t k = 0; k < EE; k++) {
                Axpy(dco[r * NRES_PAD], perm[k * NRES_PAD], ghr.GetValue(r * EE + k), NRES);
            }
            Mul(t24[r * NRES_PAD], dco[r * NRES_PAD], coefP[r * NRES_PAD], NRES);
            ReduceSum(rsum[r * 8], t24[r * NRES_PAD], rtmp, NRES);
            SyncVS();
            Adds(dco[r * NRES_PAD], dco[r * NRES_PAD], -rsum.GetValue(r * 8), NRES);
            Mul(dzc[r * NRES_PAD], coefP[r * NRES_PAD], dco[r * NRES_PAD], NRES);
            Muls(dl32[r * NL + RES_OFF], dzc[r * NRES_PAD], s2, NRES);
            Mul(t24[r * NRES_PAD], dzc[r * NRES_PAD], l32[r * NL + RES_OFF], NRES);
            ReduceSum(rsum[r * 8], t24[r * NRES_PAD], rtmp, NRES);
            SyncVS();
            Add(dbAcc[RES_OFF], dbAcc[RES_OFF], dzc[r * NRES_PAD], NRES);
            const float dscale2 = rsum.GetValue(r * 8);

            // drstd = <dl, raw>; coef = -drstd * rstd^3 / eh
            Add(dl32[r * NL], dlA[r * 8], dlB[r * 8], 8);
            Mul(t24[0], dl32[r * NL], rawF[r * NL], NL);
            ReduceSum(rsum[r * 8], t24[0], rtmp, NL);
            SyncVS();
            const float drstd = rsum.GetValue(r * 8);
            const float rv = rstd.GetValue(r);
            coef4.SetValue(r, -drstd * rv * rv * rv / static_cast<float>(static_cast<int32_t>(eh_)));
            // res group spans 24 lanes: the per-group total folds the
            // 1/24 lane-count correction the wrapper used to apply
            dsAcc.SetValue(2, dsAcc.GetValue(2) + dscale2 * (1.0f / NRES));

            // grad_logits = dl * rstd (single bf16 rounding, straight out)
            Muls(dl32[r * NL], dl32[r * NL], rv, NL);
        }

        // pre/post dscale/dbase: lane sums over the group (inert lanes are
        // zero, so plain 8-wide adds are safe)
        SyncVS();
        for (int32_t i = 0; i < HC; i++) {
            float sa = 0.0f;
            float ba = 0.0f;
            float sb = 0.0f;
            float bb = 0.0f;
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                sa += dsA.GetValue(r * 8 + i);
                ba += dbA.GetValue(r * 8 + i);
                sb += dsB.GetValue(r * 8 + HC + i);
                bb += dbB.GetValue(r * 8 + HC + i);
            }
            // pre/post groups span 4 lanes each
            dsAcc.SetValue(0, dsAcc.GetValue(0) + sa * 0.25f);
            dbAcc.SetValue(i, dbAcc.GetValue(i) + ba);
            dsAcc.SetValue(1, dsAcc.GetValue(1) + sb * 0.25f);
            dbAcc.SetValue(HC + i, dbAcc.GetValue(HC + i) + bb);
        }
        SyncSV();

        // dx_i = h_pre_i * g + coef * x_i, streamed chunk by chunk (x is
        // copied per row: chunk c rows are eh apart, not contiguous)
        for (int32_t c = 0; c < HC; c++) {
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                DataCopy(xBf[r * h_], xGm_[(r0 + r) * eh_ + c * h_], h_);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                const float cf = coef4.GetValue(r);
                const float hp = hpre8.GetValue(r * 8 + c);
                Cast(x32[r * h_], xBf[r * h_], RoundMode::CAST_NONE, h_);
                Muls(x32[r * h_], x32[r * h_], cf, h_);
                Muls(gx32[r * h_], g32[r * h_], hp, h_);
                Add(gx32[r * h_], gx32[r * h_], x32[r * h_], h_);
            }
            Cast(gxBf, gx32, RoundMode::CAST_RINT, rows * h_);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            for (int32_t r = 0; r < static_cast<int32_t>(rows); r++) {
                DataCopy(dxGm_[(r0 + r) * eh_ + c * h_], gxBf[r * h_], h_);
            }
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }

        Cast(dlBf, dl32, RoundMode::CAST_RINT, rows * NL);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(dlGm_[r0 * NL], dlBf, rows * NL);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<bfloat16_t> gGm_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<float> ghpGm_;
    GlobalTensor<float> ghrGm_;
    GlobalTensor<float> pmGm_;
    GlobalTensor<bfloat16_t> lgGm_;
    GlobalTensor<float> rGm_;
    GlobalTensor<float> hpGm_;
    GlobalTensor<float> sGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<bfloat16_t> dlGm_;
    GlobalTensor<bfloat16_t> dxGm_;
    GlobalTensor<float> dsGm_;
    GlobalTensor<float> dbGm_;
    uint32_t sb_ = 0;
    uint32_t h_ = 0;
    uint32_t eh_ = 0;
    uint32_t offGbf_ = 0;
    uint32_t offXbf_ = 0;
    uint32_t offG32_ = 0;
    uint32_t offX32_ = 0;
    uint32_t offGx32_ = 0;
    uint32_t offGxbf_ = 0;
    uint32_t offLgBf_ = 0;
    uint32_t offRaw_ = 0;
    uint32_t offL32_ = 0;
    uint32_t offDlA_ = 0;
    uint32_t offDlB_ = 0;
    uint32_t offDl32_ = 0;
    uint32_t offDlBf_ = 0;
    uint32_t offZ8_ = 0;
    uint32_t offE8_ = 0;
    uint32_t offD8_ = 0;
    uint32_t offOne8_ = 0;
    uint32_t offSig8_ = 0;
    uint32_t offT8A_ = 0;
    uint32_t offT8B_ = 0;
    uint32_t offDsA_ = 0;
    uint32_t offDbA_ = 0;
    uint32_t offDsB_ = 0;
    uint32_t offDbB_ = 0;
    uint32_t offDw8_ = 0;
    uint32_t offGh8_ = 0;
    uint32_t offGhp4_ = 0;
    uint32_t offHpre_ = 0;
    uint32_t offRstd_ = 0;
    uint32_t offCoef_ = 0;
    uint32_t offCoefP_ = 0;
    uint32_t offDco_ = 0;
    uint32_t offDzc_ = 0;
    uint32_t offT24_ = 0;
    uint32_t offGhr_ = 0;
    uint32_t offPerm_ = 0;
    uint32_t offScale_ = 0;
    uint32_t offBase_ = 0;
    uint32_t offDs_ = 0;
    uint32_t offDb_ = 0;
    uint32_t offRsum_ = 0;
    uint32_t offRtmp_ = 0;
};

extern "C" __global__ __aicore__ void lite_pre_grad(
    GM_ADDR g, GM_ADDR x, GM_ADDR ghpost, GM_ADDR ghres, GM_ADDR perm, GM_ADDR logits, GM_ADDR rstd,
    GM_ADDR hpre, GM_ADDR scale, GM_ADDR base, GM_ADDR dlogits, GM_ADDR dx, GM_ADDR dscale,
    GM_ADDR dbase, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelLitePreGrad op;
    TPipe pipe;
    op.Init(g, x, ghpost, ghres, perm, logits, rstd, hpre, scale, base,
        dlogits, dx, dscale, dbase, &tilingData, &pipe);
    op.Process();
}

#ifndef __CCE_KT_TEST__
extern "C" void lite_pre_grad_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *g, uint8_t *x,
    uint8_t *ghpost, uint8_t *ghres, uint8_t *perm, uint8_t *logits, uint8_t *rstd, uint8_t *hpre,
    uint8_t *scale, uint8_t *base, uint8_t *dlogits, uint8_t *dx, uint8_t *dscale, uint8_t *dbase,
    uint8_t *workspace, uint8_t *tiling)
{
    lite_pre_grad<<<blockDim, l2ctrl, stream>>>(g, x, ghpost, ghres, perm, logits, rstd, hpre, scale,
        base, dlogits, dx, dscale, dbase, workspace, tiling);
}
#endif
