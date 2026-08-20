SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

export CUDA_DEVICE_MAX_CONNECTIONS=1

PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
MODEL=ailab_slm_0_5b____v2
TASK=slm_archv2_muon_pretrain_8t_v12_lr5e3_muon_cooldown_mtp01_v6v4_320b
TARGET_TP=4
TARGET_PP=1
LOAD_DIR="${PATH_ROOT}/ckpt/mcore/${TASK}/mg2hf_v2/iter_0040000/"
SAVE_DIR="${PATH_ROOT}/ckpt/mcore/${TASK}/debug/tp${TARGET_TP}pp${TARGET_PP}"

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size $TARGET_TP \
    --target-pipeline-parallel-size $TARGET_PP \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --model-type-hf llama2