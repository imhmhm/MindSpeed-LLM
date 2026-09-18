SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

export CUDA_DEVICE_MAX_CONNECTIONS=1

PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
MODEL=ailab_slm_0_5b____v2
TASK=slm_archv2_muon_pretrain_8t_v12_lr5e3_muon_cooldown_mtp01_v6v4_320b
ITER=40000
TARGET_TP=4
TARGET_PP=1
HF_CFG_DIR="${PATH_ROOT}/hf_model/${MODEL}"
LOAD_DIR="${PATH_ROOT}/ckpt/mcore/${TASK}"
SAVE_DIR="${PATH_ROOT}/ckpt/mcore/${TASK}/tp${TARGET_TP}pp${TARGET_PP}"

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --params-dtype bf16 \
    --load-model-type mg \
    --save-model-type mg \
    --ckpt-iter $ITER \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --target-tensor-parallel-size $TARGET_TP \
    --target-pipeline-parallel-size $TARGET_PP 
