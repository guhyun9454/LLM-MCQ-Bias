#!/bin/bash
# ARC fleet for the stage-2 projected identify arm (branch jh/projection_eps).
#
# Idempotent: a model whose result dir already carries three_curves_points.json
# is skipped, so this can be re-run to pick up jobs that were refused at the
# QOS submit cap.  Pins -w to standard ariel-v nodes only (m-class is blocked
# under the student QOS).
#
# Usage:
#   submit_eps_proj_arc.sh canary     # just the canary model
#   submit_eps_proj_arc.sh rest       # the other 14
#   submit_eps_proj_arc.sh all        # all 15
set -euo pipefail
MODE="${1:-canary}"
WT=/ceph_data/jihye4118/LLM-MCQ-Bias-projeps
JOB="$WT/experiments/eps_proj_job.sh"
mkdir -p "$WT/logs"

CANARY="Qwen/Qwen2.5-7B-Instruct"
REST=(
  "allenai/Olmo-3-7B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.2-3B-Instruct"
  "mistralai/Ministral-8B-Instruct-2410"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "microsoft/Phi-3-mini-4k-instruct"
  "microsoft/Phi-4-mini-instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen3-4B-Instruct-2507"
  "google/gemma-3-4b-it"
)
NODES=(v6 v7 v8 v9 v10 v11 v12)

submit () {
  local model="$1" idx="$2"
  local base="${model##*/}"
  local done_marker="$WT/code/results_arc/0s_$base/arc_full_id-ABCD__eps_proj_0913/arc_three_curves_points.json"
  if [ -f "$done_marker" ]; then
    echo "[skip] $base (already complete)"
    return 0
  fi
  if squeue -u "$USER" -h -o '%j' | grep -qx "epsproj-$base"; then
    echo "[skip] $base (already queued)"
    return 0
  fi
  local node="${NODES[$((idx % ${#NODES[@]}))]}"
  sbatch --job-name="epsproj-$base" --nodelist="ariel-$node" --gres=gpu:1 \
         --cpus-per-gpu=8 --mem-per-gpu=32G --time=6-0 --partition=batch_ugrad \
         -o "$WT/logs/%j.out" "$JOB" "$model" arc ABCD \
    || echo "[defer] $base refused (likely QOS cap) -- re-run this script later"
}

case "$MODE" in
  canary) submit "$CANARY" 0 ;;
  rest)   i=1; for m in "${REST[@]}"; do submit "$m" "$i"; i=$((i+1)); done ;;
  all)    submit "$CANARY" 0; i=1; for m in "${REST[@]}"; do submit "$m" "$i"; i=$((i+1)); done ;;
  *) echo "usage: $0 {canary|rest|all}" >&2; exit 1 ;;
esac
echo "--- queue ---"
squeue -u "$USER" -h -o '%T' | sort | uniq -c
