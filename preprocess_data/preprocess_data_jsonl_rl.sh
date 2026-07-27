PATH_ROOT=/home/ma-user/work/dataset/ailab_nlp_zhh_guiyang
MODEL=ailab_slm_0_5b
SEQ_LEN=2048
MODEL_DIR=${PATH_ROOT}/hf_model/${MODEL}/
OUTPUT_DIR=${PATH_ROOT}/ml_data/ml_data__${MODEL}_${SEQ_LEN}/dapo-math-17k

mkdir -p $OUTPUT_DIR

python preprocess_data/preprocess_data_jsonl_rl.py \
  --trust-remote-code \
  --input ${PATH_ROOT}/hf_data/DAPO-Math-17k/data/ \
  --convert-method "dapo_math_17k" \
  --json-keys "input_ids" "attention_mask" "labels" \
  --prompt-type empty \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path $MODEL_DIR \
  --seq-length $SEQ_LEN \
  --output-prefix ${OUTPUT_DIR}/dapo-math-17k \
  --map-keys '{"prompt":"prompt", "query":"", "response":"reward_model", "system":""}' \
  --dataset-additional-keys "labels" \
  --workers 32