#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Sync the LitePreHeads op into a cann-ops clone and build the standalone
# custom-op package (no CANN installation is touched).
#
# Usage: bash sync_and_build.sh [cann-ops clone dir]
# Env:   ASCEND_HOME_PATH (default /usr/local/Ascend/cann-9.1.1)
#
# After a successful build set, in the runtime environment:
#   export ASCEND_CUSTOM_OPP_PATH=<clone>/build_out
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
CLONE=${1:-$HOME/work/dataset/huashan_zhh_guiyang_turbo/github/cann-ops}

if [ ! -d "$CLONE/src/contrib" ]; then
    echo "cann-ops clone not found at $CLONE (git clone https://gitee.com/ascend/cann-ops.git)" >&2
    exit 1
fi

mkdir -p "$CLONE/src/contrib/norm/lite_pre_heads"
cp -r "$HERE"/CMakeLists.txt "$HERE"/op_host "$HERE"/op_kernel \
    "$CLONE/src/contrib/norm/lite_pre_heads/"

cd "$CLONE"
bash build.sh -n lite_pre_heads -c ascend910b

# ASCEND_CUSTOM_OPP_PATH entries must directly contain op_api/lib/, so
# unpack the vendor tree the CPack staging area holds into build_out
STAGING="$CLONE"/build/_CPack_Packages/Linux/External/*.run/packages/vendors/*
rm -rf "$CLONE/build_out"
mkdir -p "$CLONE/build_out"
cp -r $STAGING/. "$CLONE/build_out/"

echo
echo "package root: $CLONE/build_out"
echo "runtime env:  export ASCEND_CUSTOM_OPP_PATH=$CLONE/build_out"
