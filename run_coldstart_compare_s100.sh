#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xudong/work/self_play"
RUN_ID="${1:-coldstart_abs_vs_psro_s100_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${ROOT}/checkpoints/roll_abs_benchmark_coldstart_compare"

mkdir -p "${ROOT}/logs" "${LOCAL_OUT}"

export PYTHONUNBUFFERED=1
export ABS_TRAIN_GPU="${ABS_TRAIN_GPU:-A100-40GB:4}"
export ABS_EVAL_GPU="${ABS_EVAL_GPU:-A100-40GB:2}"
export ABS_PAYOFF_GPU="${ABS_PAYOFF_GPU:-A100-40GB:4}"
export ABS_RM_GPU="${ABS_RM_GPU:-A10G}"
export ABS_RM_MAX_CONTAINERS="${ABS_RM_MAX_CONTAINERS:-1}"

cd "${ROOT}/ROLL"

modal run modal_abs_benchmark.py \
  --mode coldstart-compare-full \
  --run-suffix "${RUN_ID}" \
  --eval-suffix "${RUN_ID}" \
  --max-steps 100 \
  --psro-warmup-steps 20 \
  --asym-role-steps 10 \
  --payoff-episodes-per-pair 12 \
  --payoff-max-concurrent 4 \
  --local-output-dir "${LOCAL_OUT}"
