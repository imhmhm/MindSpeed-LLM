SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

export CUDA_DEVICE_MAX_CONNECTIONS=1

PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
MODEL=ailab_slm_0_5b____v2
TP=1
PP=1
LOAD_DIR="${PATH_ROOT}/hf_model/${MODEL}"
SAVE_DIR="${PATH_ROOT}/ckpt/mcore/${MODEL}/tp${TP}pp${PP}"

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size $TP \
    --target-pipeline-parallel-size $PP \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --model-type-hf llama2