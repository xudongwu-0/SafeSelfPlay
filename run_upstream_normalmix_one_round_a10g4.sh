#!/usr/bin/env bash
set -euo pipefail

# One independently trained attacker/defender round on the normal upstream
# harmful+benign prompt mixture. The Modal function owns all four A10G GPUs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ATTACKER_STEPS="${ATTACKER_STEPS:-${STEPS:-5}}"
DEFENDER_STEPS="${DEFENDER_STEPS:-${STEPS:-2}}"
ATTACKER_PROMPT_POOL_SIZE="${ATTACKER_PROMPT_POOL_SIZE:-32}"
DEFENDER_PROMPT_POOL_SIZE="${DEFENDER_PROMPT_POOL_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-8}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ATTACKER_SAVE_STEPS="${ATTACKER_SAVE_STEPS:-${ATTACKER_STEPS}}"
DEFENDER_SAVE_STEPS="${DEFENDER_SAVE_STEPS:-${DEFENDER_STEPS}}"
ATTACKER_LR="${ATTACKER_LR:-1e-5}"
DEFENDER_LR="${DEFENDER_LR:-2e-5}"
ATTACKER_LR_TAG="$(printf '%.0e' "${ATTACKER_LR}" | sed 's/e-0/e-/; s/e+0/e+/')"

export UPSTREAM_ROLE_LORA_GPU="A10G:4"

A_SUFFIX="normalmix_one_round_${RUN_TAG}_A1"
D_SUFFIX="normalmix_one_round_${RUN_TAG}_D1_vs_A1"
A_RUN_NAME="upstream_selfredteam_attacker_lora_r32_fromSFT_vs_base_normalmix_harmful_p${ATTACKER_PROMPT_POOL_SIZE}_s${ATTACKER_STEPS}_rb${ROLLOUT_BATCH_SIZE}_mb${MICRO_TRAIN_BATCH_SIZE}_tb${TRAIN_BATCH_SIZE}_lr${ATTACKER_LR_TAG}_const_nosft_${A_SUFFIX}"
A_ADAPTER="/output/upstream_selfredteam_role_lora/${A_RUN_NAME}/ckpt/global_step${ATTACKER_STEPS}_hf"

LOG_DIR="${ROOT_DIR}/logs/upstream_normalmix_one_round"
mkdir -p "${LOG_DIR}"
DRIVER_LOG="${LOG_DIR}/${RUN_TAG}.log"

{
  echo "run_tag=${RUN_TAG}"
  echo "attacker_suffix=${A_SUFFIX}"
  echo "defender_suffix=${D_SUFFIX}"
  echo "attacker_adapter=${A_ADAPTER}"
  echo "attacker_steps=${ATTACKER_STEPS}"
  echo "defender_steps=${DEFENDER_STEPS}"
  echo "attacker_save_steps=${ATTACKER_SAVE_STEPS}"
  echo "defender_save_steps=${DEFENDER_SAVE_STEPS}"
  echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
  echo "micro_rollout_batch_size=${MICRO_ROLLOUT_BATCH_SIZE}"
  echo "Starting A1 at $(date --iso-8601=seconds)"

  modal run --detach \
    modal_upstream_selfredteam_role_lora.py::upstream_attacker_lora_normal_mix \
    --steps "${ATTACKER_STEPS}" \
    --normal-prompt-pool-size "${ATTACKER_PROMPT_POOL_SIZE}" \
    --normal-prompt-pool-profile harmful \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --micro-rollout-batch-size "${MICRO_ROLLOUT_BATCH_SIZE}" \
    --micro-train-batch-size "${MICRO_TRAIN_BATCH_SIZE}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --save-steps "${ATTACKER_SAVE_STEPS}" \
    --actor-learning-rate "${ATTACKER_LR}" \
    --actor-lr-scheduler constant \
    --run-suffix "${A_SUFFIX}"

  echo "A1 completed at $(date --iso-8601=seconds)"
  echo "Starting D1 at $(date --iso-8601=seconds)"

  modal run --detach \
    modal_upstream_selfredteam_role_lora.py::upstream_defender_lora_normal_mix \
    --fixed-attacker-adapter "${A_ADAPTER}" \
    --steps "${DEFENDER_STEPS}" \
    --normal-prompt-pool-size "${DEFENDER_PROMPT_POOL_SIZE}" \
    --normal-prompt-pool-profile balanced \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --micro-rollout-batch-size "${MICRO_ROLLOUT_BATCH_SIZE}" \
    --micro-train-batch-size "${MICRO_TRAIN_BATCH_SIZE}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --save-steps "${DEFENDER_SAVE_STEPS}" \
    --actor-learning-rate "${DEFENDER_LR}" \
    --actor-lr-scheduler constant \
    --no-enable-aux-sft \
    --defender-prompt-profile role_specific \
    --balance-defender-refusal-replay \
    --run-suffix "${D_SUFFIX}"

  echo "D1 completed at $(date --iso-8601=seconds)"
} 2>&1 | tee -a "${DRIVER_LOG}"
