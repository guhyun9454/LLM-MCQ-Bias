#!/bin/bash


EVAL_NAMES=("arc,0,cyclic" "csqa,0,cyclic")

# Map each task to its setting suffix (e.g., _cyclic) for result path alignment
declare -A TASK_SUFFIX=()
for eval in "${EVAL_NAMES[@]}"; do
  IFS=',' read -r task shots setting <<< "$eval"
  if [[ -n "$setting" ]]; then
    TASK_SUFFIX["$task"]="_${setting}"
  else
    TASK_SUFFIX["$task"]=""
  fi
done

MODELS=(
  "K-intelligence/Midm-2.0-Mini-Instruct"
  "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
  "skt/A.X-4.0-Light"
  "kakaocorp/kanana-1.5-2.1b-instruct-2505"
  "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "google/gemma-3-1b-it"
)

lang_flags() {
  local lang="$1"
  if [[ "$lang" == "ko" ]]; then
    echo "--ko --prompt_lang ko"
  else
    echo "--prompt_lang en"
  fi
}

declare -A OPT4=(
  ["ABCD"]="A,B,C,D"
  ["abcd"]="a,b,c,d"
  ["1234"]="1,2,3,4"
)
declare -A OPT5=(
  ["ABCD"]="A,B,C,D,E"
  ["abcd"]="a,b,c,d,e"
  ["12345"]="1,2,3,4,5"
)

move_variant_results() {
  local model_name="$1"
  local lang="$2"
  local variant="$3"
  local fewshot_tag="0s_${model_name}"

  for task in arc csqa; do
    local setting_suffix="${TASK_SUFFIX[$task]}"
    local src="results/${task}/${fewshot_tag}/${task}${setting_suffix}"
    local dst="results/${task}/${fewshot_tag}/${lang}/labels-${variant}/${task}${setting_suffix}"
    if [[ -d "$src" ]]; then
      mkdir -p "$(dirname "$dst")"
      if [[ -d "$dst" ]]; then
        echo "[SKIP] $dst already exists."
      else
        mv "$src" "$dst"
        echo "[MOVED] $src -> $dst"
      fi
    fi
  done
}

run_one() {
  local model="$1"
  local lang="$2"
  local variant="$3"
  local opt4="$4"
  local opt5="$5"

  echo "=== Running: model=$model | lang=$lang | variant=$variant ==="
  python ../code/eval_clm.py \
    --pretrained_model_path "$model" \
    --eval_names "${EVAL_NAMES[@]}" \
    $(lang_flags "$lang") \
    --option_ids4 "$opt4" \
    --option_ids5 "$opt5"

  local model_name
  model_name="$(basename "$model")"
  move_variant_results "$model_name" "$lang" "$variant"
}

for MODEL in "${MODELS[@]}"; do
  for LANG in ko en; do
    run_one "$MODEL" "$LANG" "ABCD"  "${OPT4[ABCD]}"  "${OPT5[ABCD]}"
    run_one "$MODEL" "$LANG" "abcd"  "${OPT4[abcd]}"  "${OPT5[abcd]}"
    run_one "$MODEL" "$LANG" "1234"  "${OPT4[1234]}"  "${OPT5[12345]}"
  done
done