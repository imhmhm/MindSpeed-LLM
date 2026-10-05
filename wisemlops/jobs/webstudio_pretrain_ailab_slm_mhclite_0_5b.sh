#!/bin/bash
set -e
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

## ====== code dir ====== ##
MTP_CODE_DIR=/home/ma-user/work/dataset/huashan_zhh_guiyang_turbo/github/MindSpeed-LLM
cd $MTP_CODE_DIR
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH

export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export USE_OBS=0
MTP_DATASET_HOME=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
export MIND_SPEED_LOG_LEVEL=ERROR

## ====== launch ====== ##
python wisemlops/launch.py \
    --config wisemlops/configs/ailab_slm_mhclite_0_5b_pretrain.yaml \
    mount_dataset_mtp_task=$MTP_DATASET_HOME \
    mount_dataset_ml_data=$MTP_DATASET_HOME/ml_data \
    data_prefixes_and_weights="['data_contamination_strict/siye_exams_zh_new_clean_text']" \
    copy_data_to_cache=true \
    tensor-model-parallel-size=1 \
    pipeline-model-parallel-size=1 \
    context-parallel-size=1 \
    seq-length=4096 \
    micro-batch-size=1 \
    global-batch-size=32 \
    lr=2e-3 \
    min-lr=5e-5 \
    train-iters=30 \
    save-interval=2000 \
    rotary-base=100000 \
    sync_ckpt.enable=true \
    enable_swanlab=true \
    enable_tensorboard=true \
    swanlab-mode="local"
