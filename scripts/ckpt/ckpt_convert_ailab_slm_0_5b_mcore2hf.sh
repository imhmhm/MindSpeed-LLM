SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

export CUDA_DEVICE_MAX_CONNECTIONS=1

PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
MODEL=ailab_slm_0_5b____v2
TASK=slm_archv2_muon_pretrain_8t_v12_lr5e3_muon_cooldown_mtp01_v6v4_320b
TP=1
PP=1
HF_CFG_DIR="${PATH_ROOT}/hf_model/${MODEL}"
LOAD_DIR="${PATH_ROOT}/ckpt/mcore/${TASK}"
ITER=80000
SAVE_DIR="${PATH_ROOT}/ckpt/mcore/${TASK}/mg2hf"

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir $LOAD_DIR \
    --ckpt-iter $ITER \
    --save-dir $SAVE_DIR \
    --hf-cfg-dir $HF_CFG_DIR \
    --merge-layers-safetensors \
    --model-type-hf llama2