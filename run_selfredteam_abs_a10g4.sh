#!/usr/bin/env bash
set -euo pipefail

# Canonical Modal entrypoint for the Self-RedTeam training pipeline with the
# ABS-style change: attacker and defender are trained as separate LoRA policies.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ITERATIONS="${ITERATIONS:-1}"
ROLE_STEPS="${ROLE_STEPS:-50}"
PAYOFF_EPISODES_PER_PAIR="${PAYOFF_EPISODES_PER_PAIR:-12}"
PAYOFF_MAX_CONCURRENT="${PAYOFF_MAX_CONCURRENT:-4}"
AUX_SFT_COEF="${AUX_SFT_COEF:-0}"
RESPONSE_LOG_STEPS="${RESPONSE_LOG_STEPS:-5}"
RUN_SUFFIX="${RUN_SUFFIX:-selfredteam_repp_abs_two_lora_a10g4_i${ITERATIONS}_a${ROLE_STEPS}d${ROLE_STEPS}_$(date +%Y%m%d_%H%M%S)}"

# These are the A10Gx4 settings that have already completed successfully.
ROLLOUT_BATCH_SIZE=24
TRAIN_ENV_GROUPS=3
TRAIN_GROUP_SIZE=8
TRAIN_MICRO_BATCH=1
GRAD_ACCUM=8
SEQUENCE_LENGTH=4096
MAX_NEW_TOKENS=1024
VLLM_MAX_NUM_BATCHED_TOKENS=8192
ACTOR_INFER_MAX_CONCURRENCY=64

EXTRA_ARGS=()
if [[ -n "${FIXED_SEED_PROMPT:-}" ]]; then
  if [[ "${FIXED_SEED_LABEL:-}" != "harmful" && "${FIXED_SEED_LABEL:-}" != "benign" ]]; then
    echo "FIXED_SEED_LABEL must be harmful or benign when FIXED_SEED_PROMPT is set." >&2
    exit 2
  fi
  EXTRA_ARGS+=(
    --fixed-seed-prompt "${FIXED_SEED_PROMPT}"
    --fixed-seed-label "${FIXED_SEED_LABEL}"
  )
fi

echo "Run: ${RUN_SUFFIX}"
echo "Method: Self-RedTeam REINFORCE++ pipeline + separate ABS attacker/defender LoRAs"
echo "Schedule: ${ITERATIONS} x (A${ROLE_STEPS} + D${ROLE_STEPS} + payoff)"
echo "Modal: train=A10G:4, reward=L4:1, rollout=${ROLLOUT_BATCH_SIZE}, groups=${TRAIN_ENV_GROUPS}x${TRAIN_GROUP_SIZE}, mb=${TRAIN_MICRO_BATCH}, ga=${GRAD_ACCUM}"

cd "${ROOT}"
export ABS_TRAIN_GPU="A10G:4"
export ABS_RM_GPU="L4"
export ABS_RM_MAX_CONTAINERS="1"

modal run --detach --timestamps \
  modal_sft_base_psro_once.py::sft_base_psro \
  --run-suffix "${RUN_SUFFIX}" \
  --iterations "${ITERATIONS}" \
  --role-steps "${ROLE_STEPS}" \
  --save-steps "${ROLE_STEPS}" \
  --payoff-episodes-per-pair "${PAYOFF_EPISODES_PER_PAIR}" \
  --payoff-max-concurrent "${PAYOFF_MAX_CONCURRENT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --train-env-groups "${TRAIN_ENV_GROUPS}" \
  --train-group-size "${TRAIN_GROUP_SIZE}" \
  --val-env-groups 4 \
  --train-micro-batch "${TRAIN_MICRO_BATCH}" \
  --grad-accum "${GRAD_ACCUM}" \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --actor-infer-max-concurrency "${ACTOR_INFER_MAX_CONCURRENCY}" \
  --response-log-steps "${RESPONSE_LOG_STEPS}" \
  --optimizer-profile selfredteam_repp \
  --filter-zero-variance-groups \
  --attacker-on-topic-weight 0 \
  --aux-sft-coef "${AUX_SFT_COEF}" \
  "${EXTRA_ARGS[@]}"
