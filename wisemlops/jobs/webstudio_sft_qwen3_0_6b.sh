#!/bin/bash
set -e
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

## ====== code dir ====== ##
MTP_CODE_DIR=/home/ma-user/work/dataset/dataset_zhh_guiyang/github/MindSpeed-LLM
# cp -r /opt/huawei/schedule-train/algorithm $MTP_CODE_DIR
cd $MTP_CODE_DIR
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH

export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export USE_OBS=0
MTP_DATASET_HOME=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs

## ====== launch ====== ##
python wisemlops/launch.py \
    --config wisemlops/configs/qwen3_0_6b_sft.yaml \
    mount_dataset_mtp_task=$MTP_DATASET_HOME \
    mount_dataset_ml_data=$MTP_DATASET_HOME/ml_data \
    dataset_name=MegaScience \
    data_prefixes_and_weights='["MegaScience"]' \
    pack_varlen_data=false \
    load_name=Qwen3-0.6B-Base/tp1pp1 \
    load_iter=1 \
    lr=1e-5 \
    train-iters=10 \
    save-interval=5 \
    rotary-base=1000000 \
    context-parallel-size=1 \
    tensor-model-parallel-size=1 \
    pipeline-model-parallel-size=1 \
    seq-length=4096 \
    micro-batch-size=1 \
    global-batch-size=32 \
    sync_ckpt.enable=true \
    mcore2hf_after_training.iters='[5,10]'
