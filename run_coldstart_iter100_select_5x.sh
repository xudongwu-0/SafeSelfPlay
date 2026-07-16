#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xudong/work/self_play"
ITERATIONS="${ITERATIONS:-5}"
ROLE_STEPS="${ROLE_STEPS:-50}"
ITER_STEPS="$((2 * ROLE_STEPS))"
PAYOFF_EPISODES_PER_PAIR="${PAYOFF_EPISODES_PER_PAIR:-12}"
PAYOFF_MAX_CONCURRENT="${PAYOFF_MAX_CONCURRENT:-4}"
RUN_ID="${1:-abs3b_cs_iter${ITER_STEPS}x${ITERATIONS}_select_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${ROOT}/checkpoints/roll_abs_benchmark_coldstart_iter100_select"
RESUME_STATE_PATH="${RESUME_STATE_PATH:-}"
MODE="${MODE:-coldstart-iter100-select-full}"

mkdir -p "${ROOT}/logs" "${LOCAL_OUT}"

export PYTHONUNBUFFERED=1
export ABS_TRAIN_GPU="${ABS_TRAIN_GPU:-A100-40GB:4}"
export ABS_EVAL_GPU="${ABS_EVAL_GPU:-A100-40GB:2}"
export ABS_PAYOFF_GPU="${ABS_PAYOFF_GPU:-A100-40GB:4}"
export ABS_RM_GPU="${ABS_RM_GPU:-A10G}"
export ABS_RM_MAX_CONTAINERS="${ABS_RM_MAX_CONTAINERS:-1}"
SAFE_RM_LABEL="$(printf '%s' "${RUN_ID}" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-' | cut -c1-45)"
export ABS_RM_LABEL="${ABS_RM_LABEL:-wg-${SAFE_RM_LABEL:-abs}-$(date +%H%M%S)}"

EXTRA_ARGS=()
if [[ -n "${RESUME_STATE_PATH}" ]]; then
  EXTRA_ARGS+=(--resume-state-path "${RESUME_STATE_PATH}")
fi

cd "${ROOT}/ROLL"

modal run modal_abs_benchmark.py \
  --mode "${MODE}" \
  --run-suffix "${RUN_ID}" \
  --eval-suffix "${RUN_ID}" \
  --max-steps "${ITER_STEPS}" \
  --asym-role-steps "${ROLE_STEPS}" \
  --asym-iterations "${ITERATIONS}" \
  --payoff-episodes-per-pair "${PAYOFF_EPISODES_PER_PAIR}" \
  --payoff-max-concurrent "${PAYOFF_MAX_CONCURRENT}" \
  --local-output-dir "${LOCAL_OUT}" \
  "${EXTRA_ARGS[@]}"
