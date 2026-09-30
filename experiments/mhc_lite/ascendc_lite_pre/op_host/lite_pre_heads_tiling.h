/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 * Tiling parameters of the LitePreHeads custom operator.
 */
#ifndef LITE_PRE_HEADS_TILING_H
#define LITE_PRE_HEADS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LitePreHeadsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, sb);
  TILING_DATA_FIELD_DEF(uint32_t, h);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LitePreHeads, LitePreHeadsTilingData)
}
#endif // LITE_PRE_HEADS_TILING_H
