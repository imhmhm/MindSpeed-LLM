#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Sync LitePreGrad (and LitePreHeads + LitePreFused, which must live in
# the same libcust_opapi.so) into a gitcode ops-transformer clone and
# build one standalone custom-op package holding all three ops. The
# upstream mhc sinkhorn family ships in the same mhc/ category of the
# clone, so it joins the --ops list: one build_out then serves both the
# mhc-lite and mhc-sinkhorn inference paths.
#
# Usage: bash sync_and_build.sh [ops-transformer clone dir]
#
# The clone must sit on tag v9.1.1 (matches a CANN 9.1.1 install); master
# tracks newer CANN releases and may fail to build:
#   git clone --branch v9.1.1 https://gitcode.com/cann/ops-transformer.git
#
# PIP_NO_BUILD_ISOLATION=1 keeps the es_math wheel step out of pip's isolated
# build environment when the configured index does not serve setuptools; the
# interpreter environment already provides it.
#
# After a successful build set, in the runtime environment:
#   export ASCEND_CUSTOM_OPP_PATH=<clone>/build_out
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
HEADS_DIR=$(cd "$HERE/../ascendc_lite_pre" && pwd)
FUSED_DIR=$(cd "$HERE/../ascendc_lite_pre_fused" && pwd)
CLONE=${1:-$HOME/work/dataset/huashan_zhh_guiyang_turbo/github/ops-transformer}

if [ ! -d "$CLONE/mhc" ]; then
    echo "ops-transformer clone not found at $CLONE (git clone --branch v9.1.1 https://gitcode.com/cann/ops-transformer.git)" >&2
    exit 1
fi

for d in lite_pre_heads lite_pre_fused lite_pre_grad; do
    mkdir -p "$CLONE/mhc/$d"
done
cp -r "$HEADS_DIR"/op_host "$HEADS_DIR"/op_kernel "$CLONE/mhc/lite_pre_heads/"
cp -r "$FUSED_DIR"/op_host "$FUSED_DIR"/op_kernel "$CLONE/mhc/lite_pre_fused/"
cp -r "$HERE"/op_host "$HERE"/op_kernel "$CLONE/mhc/lite_pre_grad/"

cd "$CLONE"
PIP_NO_BUILD_ISOLATION=1 bash build.sh --pkg --soc=ascend910b \
    --ops=lite_pre_heads,lite_pre_fused,lite_pre_grad,mhc_pre,mhc_pre_sinkhorn,mhc_sinkhorn,mhc_post

# ASCEND_CUSTOM_OPP_PATH entries must directly contain op_api/lib/, so
# unpack the vendor tree the CPack staging area holds into build_out
# (build.sh itself only leaves the .run self-extractor there)
STAGING="$CLONE"/build/_CPack_Packages/Linux/External/*.run/packages/vendors/custom_transformer
rm -rf "$CLONE/build_out"
mkdir -p "$CLONE/build_out"
cp -r "$STAGING"/. "$CLONE/build_out/"

echo
echo "package root: $CLONE/build_out"
echo "runtime env:  export ASCEND_CUSTOM_OPP_PATH=$CLONE/build_out"
