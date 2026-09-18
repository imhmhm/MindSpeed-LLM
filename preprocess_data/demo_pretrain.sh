SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

## ========  model/input/output  ======== ##

PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
# MODEL=Qwen3-0.6B
# TOKENIZER_TAG=qwen3
MODEL=ailab_slm_0_5b
TOKENIZER_TAG=ailab_slm
#### processing in /cache is much faster
COPY_TO_CACHE=true

INPUT_SUBDIR=corpus/encyclopedia_zh/wiki_ybk_dbk_others.jsonl
OUTPUT_SUBDIR_PREFIX=ml_data/ml_data__${TOKENIZER_TAG}/encyclopedia_zh/wiki_ybk_dbk_others/wiki_ybk_dbk_others

## ======================================== ##

MODEL_DIR=${PATH_ROOT}/hf_model/${MODEL}

INPUT_DIR=${PATH_ROOT}/${INPUT_SUBDIR}
if [ "$COPY_TO_CACHE" = true ]; then
    mkdir -p /cache/$(dirname "$INPUT_SUBDIR")
    cp -r ${PATH_ROOT}/${INPUT_SUBDIR} /cache/$(dirname "$INPUT_SUBDIR")
    INPUT_DIR=/cache/${INPUT_SUBDIR}
fi

OUTPUT_PREFIX=${PATH_ROOT}/${OUTPUT_SUBDIR_PREFIX}
if [ "$COPY_TO_CACHE" = true ]; then
    OUTPUT_PREFIX=/cache/${OUTPUT_SUBDIR_PREFIX}
fi
mkdir -p $(dirname "$OUTPUT_PREFIX")

## ======================================== ##

python preprocess_data/preprocess_data_pretrain.py \
  --input $INPUT_DIR \
  --json-keys "text" \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-not-use-fast \
  --tokenizer-name-or-path $MODEL_DIR \
  --output-prefix $OUTPUT_PREFIX \
  --append-eod \
  --workers 64

## ======================================== ##

if [ "$COPY_TO_CACHE" = true ]; then
    DEST_OUTPUT_DIR=$(dirname "${PATH_ROOT}/${OUTPUT_SUBDIR_PREFIX}")
    mkdir -p $DEST_OUTPUT_DIR
    cp -r ${OUTPUT_PREFIX}*.* $DEST_OUTPUT_DIR
fi
