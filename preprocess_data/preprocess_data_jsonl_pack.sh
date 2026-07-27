PATH_ROOT=/home/ma-user/work/dataset/ailab_nlp_zhh_guiyang
# MODEL=Qwen3-0.6B
MODEL=ailab_slm_0_5b
PROMPT_TYPE=ailab_slm
SEQ_LEN=32768


MODEL_DIR=${PATH_ROOT}/hf_model/${MODEL}
INPUT_DIR=${PATH_ROOT}/hf_data/MegaScience/jsonl
OUTPUT_DIR=${PATH_ROOT}/ml_data/ml_data__${MODEL}_${SEQ_LEN}_varlen/MegaScience__random
OUTPUT_PREFIX=${OUTPUT_DIR}/MegaScience
mkdir -p $OUTPUT_DIR


python preprocess_data/preprocess_data_jsonl_pack.py \
  --trust-remote-code \
  --input $INPUT_DIR \
  --convert-method "alpaca" \
  --json-keys "input_ids" "attention_mask" "labels" \
  --prompt-type $PROMPT_TYPE \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path $MODEL_DIR \
  --seq-length $SEQ_LEN \
  --output-prefix $OUTPUT_PREFIX \
  --map-keys '{"prompt":"question", "query":"", "response":"answer", "system":""}' \
  --neat-pack \
  --clean-unmerged-indexed-dataset \
  --workers 8

  # --knapsack \

  # --map-keys '{"messages":"data", "system":"meta_prompt", "role_tag":"role", "content_tag":"content", "user_tag":"user", "assistant_tag":"assistant", "system_tag": "system", "observation_tag":"tool"}' \