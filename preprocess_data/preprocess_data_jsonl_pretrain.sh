PATH_ROOT=/home/ma-user/work/dataset/dataset_zhh_guiyang
MODEL=ailab_slm_0_5b

PROMPT_TYPE=ailab_slm

MODEL_DIR=${PATH_ROOT}/hf_model/${MODEL}/
INPUT_DIR=${PATH_ROOT}/hf_data/MegaScience/jsonl
OUTPUT_DIR=${PATH_ROOT}/ml_data/ml_data__${MODEL}/MegaScience
OUTPUT_PREFIX=${OUTPUT_DIR}/MegaScience
mkdir -p $OUTPUT_DIR


python preprocess_data/preprocess_data_jsonl_pretrain.py \
  --trust-remote-code \
  --input $INPUT_DIR \
  --json-keys "question" \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path $MODEL_DIR \
  --output-prefix $OUTPUT_PREFIX \
  --append-eod \
  --workers 32