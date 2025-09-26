#!/bin/bash

DATA_ROOT="../LLM-MCQ-Bias_data"

EVAL_NAMES=("arc,0" "csqa,0") 

MODELS=(
  "K-intelligence/Midm-2.0-Mini-Instruct"
  "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
  "kakaocorp/kanana-1.5-2.1b-instruct-2505"
  "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
  "skt/A.X-4.0-Light"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "google/gemma-3-1b-it"
)

for MODEL in "${MODELS[@]}"; do
  echo "Running evaluation for model: $MODEL"
  python code/eval_clm.py \
    --pretrained_model_path "$MODEL" \
    --eval_names "${EVAL_NAMES[@]}" \
    --data_root "$DATA_ROOT" \
    --ko \
    --prompt_lang ko \
    --option_ids4 "가,나,다,라" \
    --option_ids5 "가,나,다,라,마"
done