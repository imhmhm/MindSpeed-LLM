SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

export CUDA_DEVICE_MAX_CONNECTIONS=1

PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
MODEL=Qwen3-0.6B-Base
ITER=1
TP=1
PP=1
HF_CFG_DIR="${PATH_ROOT}/hf_model/${MODEL}"
LOAD_DIR="${PATH_ROOT}/ckpt/mcore/${MODEL}/tp${TP}pp${PP}"
SAVE_DIR="${PATH_ROOT}/ckpt/mcore/${MODEL}/tp${TP}pp${PP}/mg2hf_v2"


python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir $LOAD_DIR \
    --ckpt-iter $ITER \
    --save-dir $SAVE_DIR \
    --hf-cfg-dir $HF_CFG_DIR \
    --merge-layers-safetensors \
    --model-type-hf qwen3