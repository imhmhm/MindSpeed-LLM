SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
cd $PROJECT_ROOT

## ========  model and input dir  ======== ##
PATH_ROOT=/home/ma-user/work/dataset/huashan_zhh_guiyang_pfs
# MODEL=Qwen3-0.6B
# PROMPT_TYPE=qwen3
MODEL=ailab_slm_0_5b
PROMPT_TYPE=ailab_slm
SEQ_LEN=32768
PACKING_METHOD=random  ## knapsack / random
#### processing in /cache is much faster
COPY_TO_CACHE=true

# INPUT_SUBDIR=hf_data/MegaScience/jsonl/MegaScience_nemotron_science.jsonl
INPUT_SUBDIR=hf_data/MegaScience/data/*.parquet
# INPUT_SUBDIR=hf_data/MegaScience/data
## ======================================== ##

MODEL_DIR=${PATH_ROOT}/hf_model/${MODEL}

INPUT_DIR=${PATH_ROOT}/${INPUT_SUBDIR}
if [ "$COPY_TO_CACHE" = true ]; then
    mkdir -p /cache/$(dirname "$INPUT_SUBDIR")
    cp -r ${PATH_ROOT}/${INPUT_SUBDIR} /cache/$(dirname "$INPUT_SUBDIR")
    INPUT_DIR=/cache/${INPUT_SUBDIR}
fi

## ========  dataset attr and output dir ======== ##
CONVERT_METHOD="alpaca"
MAP_KEYS='{"prompt":"question", "query":"", "response":"answer", "system":""}'
# CONVERT_METHOD="sharegpt"
# MAP_KEYS='{"messages":"data", "system":"meta_prompt", "role_tag":"role", "content_tag":"content", "user_tag":"user", "assistant_tag":"assistant", "system_tag": "system", "observation_tag":"tool"}'

OUTPUT_SUBDIR=ml_data/ml_data__${MODEL}_${SEQ_LEN}_pack/MegaScience
## ============================================== ##

OUTPUT_DIR=${PATH_ROOT}/${OUTPUT_SUBDIR}
if [ "$COPY_TO_CACHE" = true ]; then
    OUTPUT_DIR=/cache/${OUTPUT_SUBDIR}
fi
mkdir -p $OUTPUT_DIR

## ========  output prefix  ======== ##
OUTPUT_PREFIX=${OUTPUT_DIR}/MegaScience
## ================================= ##


python preprocess_data/preprocess_data_sft_pack.py \
  --input $INPUT_DIR \
  --packing-method $PACKING_METHOD \
  --neat-pack \
  --convert-method $CONVERT_METHOD \
  --map-keys "$MAP_KEYS" \
  --json-keys "input_ids" "attention_mask" "labels" \
  --tokenizer-name-or-path $MODEL_DIR \
  --tokenizer-type PretrainedFromHF \
  --prompt-type $PROMPT_TYPE \
  --seq-length $SEQ_LEN \
  --output-prefix $OUTPUT_PREFIX \
  --clean-unmerged-indexed-dataset \
  --workers 8


if [ "$COPY_TO_CACHE" = true ]; then
    DEST_OUTPUT_DIR=${PATH_ROOT}/${OUTPUT_SUBDIR}
    mkdir -p $(dirname "$DEST_OUTPUT_DIR")
    cp -r $OUTPUT_DIR $DEST_OUTPUT_DIR
fi