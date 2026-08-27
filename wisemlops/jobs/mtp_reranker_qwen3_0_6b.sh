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
export USE_OBS=0
HUASHAN_OBS_MTP_TASK="xxx"
MTP_DATASET_HOME="/opt/huawei/dataset/data_sfs"


## ====== offline pre-steps (run once, before the job) ====== ##
## 1) hf(swift ckpt) -> mcore ckpt, placed under the task ckpt dir as <load_name>:
##    python convert_ckpt_v2.py \
##        --load-model-type hf --save-model-type mg \
##        --load-dir <swift_hf_ckpt> --save-dir <task_dir>/ckpt/mcore/<load_name> \
##        --model-type-hf qwen3 \
##        --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 \
##        --hf-cfg-dir <swift_hf_ckpt>
## 2) reranker data preprocess -> ml_data tree (seq-length MUST match the yaml):
##    python preprocess_data/preprocess_data_reranker.py \
##        --input train_filter_v5_v3_4_2.jsonl \
##        --output-prefix <ml_data_root>/ml_data__qwen3_4096/rerank_filter_v5/rerank_v5 \
##        --tokenizer-name-or-path <Qwen3-0.6B dir> --tokenizer-type PretrainedFromHF \
##        --prompt-type qwen3 --convert-method reranker --enable-thinking false \
##        --seq-length 4096 --max-negatives 5 --workers 16
##    => <ml_data_root>/ml_data__qwen3_4096/rerank_filter_v5/rerank_v5_packed_input_ids_document.bin/.idx
##    (trainer opens {data-path}_packed_input_ids_document, so data_prefixes_and_weights
##     below uses the PLAIN prefix "rerank_v5", same rule as SFT)

## ====== launch ====== ##
python wisemlops/launch.py \
    --config wisemlops/configs/qwen3_0_6b_reranker.yaml \
    huashan_obs_mtp_task=$HUASHAN_OBS_MTP_TASK \
    huashan_obs_ml_data=$HUASHAN_OBS_MTP_TASK/ml_data \
    mount_dataset_mtp_task=$MTP_DATASET_HOME \
    mount_dataset_ml_data=$MTP_DATASET_HOME/ml_data \
    dataset_name=rerank_filter_v5 \
    data_prefixes_and_weights='["rerank_v5"]' \
    is_pack_varlen_data=false \
    load_name=<converted_mcore_ckpt_name> \
    load_iter=1 \
    lr=2e-6 \
    train-iters=100 \
    save-interval=50 \
    eval-interval=50 \
    tensor-model-parallel-size=1 \
    pipeline-model-parallel-size=1 \
    seq-length=4096 \
    micro-batch-size=4 \
    global-batch-size=64 \
    sync_ckpt.enable=true \
    mcore2hf_after_training.iters='[100]'
