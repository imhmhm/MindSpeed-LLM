/**
 * Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
 */

#ifndef LITE_PRE_FUSED_TILING_H
#define LITE_PRE_FUSED_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LitePreFusedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, sb);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LitePreFused, LitePreFusedTilingData)
}  // namespace optiling

#endif
