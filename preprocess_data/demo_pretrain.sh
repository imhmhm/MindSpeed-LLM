SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

## ========  model and input dir  ======== ##
PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
# MODEL=Qwen3-0.6B
# PROMPT_TYPE=qwen3
MODEL=ailab_slm_0_5b
#### processing in /cache is much faster
COPY_TO_CACHE=true

INPUT_SUBDIR=corpus/encyclopedia_zh/wiki_ybk_dbk_others.jsonl
## ======================================== ##

MODEL_DIR=${PATH_ROOT}/hf_model/${MODEL}

INPUT_DIR=${PATH_ROOT}/${INPUT_SUBDIR}
if [ "$COPY_TO_CACHE" = true ]; then
    mkdir -p /cache/$(dirname "$INPUT_SUBDIR")
    cp -r ${PATH_ROOT}/${INPUT_SUBDIR} /cache/$(dirname "$INPUT_SUBDIR")
    INPUT_DIR=/cache/${INPUT_SUBDIR}
fi

## ========  output dir  ======== ##
OUTPUT_SUBDIR=ml_data/ml_data__${MODEL}/encyclopedia_zh/wiki_ybk_dbk_others
## ============================== ##

OUTPUT_DIR=${PATH_ROOT}/${OUTPUT_SUBDIR}
if [ "$COPY_TO_CACHE" = true ]; then
    OUTPUT_DIR=/cache/${OUTPUT_SUBDIR}
fi
mkdir -p $OUTPUT_DIR

## ========  output prefix  ======== ##
OUTPUT_PREFIX=${OUTPUT_DIR}/wiki_ybk_dbk_others
## ================================= ##


python preprocess_data/preprocess_data_pretrain.py \
  --input $INPUT_DIR \
  --json-keys "text" \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path $MODEL_DIR \
  --output-prefix $OUTPUT_PREFIX \
  --append-eod \
  --workers 64

if [ "$COPY_TO_CACHE" = true ]; then
    DEST_OUTPUT_DIR=${PATH_ROOT}/${OUTPUT_SUBDIR}
    mkdir -p $(dirname "$DEST_OUTPUT_DIR")
    cp -r $OUTPUT_DIR $DEST_OUTPUT_DIR
fi