#!/bin/bash
set -e
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

## ====== code dir ====== ##
MTP_CODE_DIR=/cache/algorithm
cp -r /opt/huawei/schedule-train/algorithm/MindSpeed-LLM $MTP_CODE_DIR
cd $MTP_CODE_DIR
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH

export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

## 50M/100M to avoid the number of partitions exceeding the OBS limit, default 10M. (50M: 52428800, 100M: 104857600)
export MOX_FILE_LARGE_FILE_PART_SIZE=104857600
# export USE_OBS=0
# HUASHAN_OBS_MTP_TASK="xxx"
MTP_DATASET_HOME="/opt/huawei/dataset/data_sfs"

## ====== launch ====== ##
python wisemlops/launch.py \
    --config wisemlops/configs/qwen3_0_6b_reranker.yaml \
    mount_dataset_mtp_task=$MTP_DATASET_HOME \
    mount_dataset_ml_data=$MTP_DATASET_HOME/ml_data \
    model_name=Qwen3-0.6B-Base \
    dataset_name=train_l2 \
    data_prefixes_and_weights="['train_filter_v5_v3_4_2']" \
    is_pack_varlen_data=false \
    load_name=Qwen3-0.6B/tp1pp1 \
    load_iter=1 \
    lr=2e-6 \
    train-iters=4600 \
    save-interval=100 \
    tensor-model-parallel-size=1 \
    pipeline-model-parallel-size=1 \
    seq-length=4096 \
    micro-batch-size=1 \
    global-batch-size=256 \
    sync_ckpt.enable=true \
    reranker-loss-type="listwise" \
    enable_swanlab=true \
    enable_tensorboard=false \
    swanlab-mode="local" \
    mcore2hf_after_training.iters='[900,1800,2700,3600,4600]'
