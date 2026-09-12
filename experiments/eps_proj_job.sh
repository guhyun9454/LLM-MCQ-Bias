#!/bin/bash
# Stage-2 PROJECTED identify run (branch jh/projection_eps).
#
# Same estimator as eps_ident_0903, plus --empirical_ident_project: at 2 views
# the difference system is rank k-2, so the min-norm pinv leaks content along
# ker(M).  That leakage lies exactly along ker(M), so projecting it out yields
# the identifiable (k-2)-dim part of eps(q) cleanly.  At >=3 views the
# projection is a mathematical no-op, so every stage >= 3 is bit-identical to
# plain identify -- any delta vs eps_ident_0903 is stage-2 contribution alone.
#
# Reuses eps0_latin_0827 caches (stage-1 selection is mu-only => same
# schedules => full stage-cache hits, no new inference expected).
# Usage: eps_proj_job.sh <hf_model_path> [task] [option_ids]
set -eo pipefail
MODEL="$1"
TASK="${2:-arc}"
OPTS="${3:-ABCD}"
BASE="${MODEL##*/}"
source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
conda activate llm
export HF_HOME=/nas2/data/jihye4118/hf_cache
export HF_HUB_CACHE=/nas2/data/jihye4118/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
CODE=/ceph_data/jihye4118/LLM-MCQ-Bias-projeps/code
SRC_ROOT=/ceph_data/jihye4118/LLM-MCQ-Bias-eps0/code
RES="$CODE/results_${TASK}/0s_$BASE"
SRC0="$SRC_ROOT/results_${TASK}/0s_$BASE/${TASK}_full_id-${OPTS}__eps0_latin_0827"
DST="$RES/${TASK}_full_id-${OPTS}__eps_proj_0913"
echo "[env] node=$(hostname) job=${SLURM_JOB_ID:-none} model=$MODEL task=$TASK"
if [ -f "$DST/${TASK}_three_curves_points.json" ]; then
  echo "[skip] $BASE already done"
  exit 0
fi
if [ ! -d "$SRC0" ]; then
  echo "[fail] missing baseline cache: $SRC0" >&2
  exit 2
fi
mkdir -p "$DST/empirical_analysis"
cp "$SRC0"/*_run*.jsonl "$DST"/
cp "$SRC0"/empirical_analysis/*_stage_cache.jsonl "$DST/empirical_analysis/"
cd "$CODE"
python -u eval_clm.py \
  --pretrained_model_path "$MODEL" \
  --cache_dir /nas2/data/jihye4118/hf_small \
  --eval_names "${TASK},0,full" \
  --option_id_set "$OPTS" \
  --wandb \
  --pride_mix \
  --skip_full \
  --n_runs 3 \
  --wandb_project "3_${TASK}_eps_proj" \
  --empirical_pride \
  --empirical_residual_model identify \
  --empirical_ident_project \
  --plot_empirical_prefix_fractions 2 \
  --empirical_sweep_mode percentile \
  --empirical_stage_schedule flat \
  --result_tag "eps_proj_0913" \
  --empirical_transition_mode latin
echo "[done] $BASE $TASK"
