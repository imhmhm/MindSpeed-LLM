#!/bin/bash
set -e
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

## ====== code dir ====== ##
MTP_CODE_DIR=/cache/algorithm
cp -r /opt/huawei/schedule-train/algorithm $MTP_CODE_DIR
cd $MTP_CODE_DIR
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH

export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

## 50M/100M to avoid the number of partitions exceeding the OBS limit, default 10M. (50M: 52428800, 100M: 104857600)
export MOX_FILE_LARGE_FILE_PART_SIZE=104857600
# export USE_OBS=0
HUASHAN_OBS_MTP_TASK="xxx"
MTP_DATASET_HOME="/opt/huawei/dataset/data_sfs"

## ====== launch ====== ##
python wisemlops/launch.py \
    --config wisemlops/configs/qwen3_0_6b_sft.yaml \
    huashan_obs_mtp_task=$HUASHAN_OBS_MTP_TASK \
    huashan_obs_ml_data=$HUASHAN_OBS_MTP_TASK/ml_data \
    mount_dataset_mtp_task=$MTP_DATASET_HOME \
    mount_dataset_ml_data=$MTP_DATASET_HOME/ml_data \
    dataset_name=MegaScience \
    data_prefixes_and_weights='["MegaScience__knapsack"]' \
    is_pack_varlen_data=true \
    load_name=Qwen3-0.6B-Base \
    load_iter=1 \
    lr=1e-5 \
    train-iters=10 \
    save-interval=5 \
    rotary-base=1000000 \
    context-parallel-size=2 \
    tensor-model-parallel-size=1 \
    pipeline-model-parallel-size=1 \
    seq-length=32768 \
    micro-batch-size=1 \
    global-batch-size=32 \
    sync_ckpt.enable=true \
    mcore2hf_after_training.iters='[5,10]'
