#!/usr/bin/env bash
set -euo pipefail

cd /home/xudong/work/self_play/ROLL

export UPSTREAM_ROLE_LORA_GPU="A10G:4"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-50}"
ATTACKER_POOL_SIZE="${ATTACKER_POOL_SIZE:-16}"
DEFENDER_POOL_SIZE="${DEFENDER_POOL_SIZE:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
SAVE_STEPS="${SAVE_STEPS:-10}"
ATTACKER_LR="${ATTACKER_LR:-1e-6}"
DEFENDER_LR="${DEFENDER_LR:-5e-6}"
ATTACKER_LR_TAG="$(
  python -c 'import sys; print(f"{float(sys.argv[1]):.0e}".replace("e-0", "e-"))' \
    "${ATTACKER_LR}"
)"

A_SUFFIX="harmful_p${ATTACKER_POOL_SIZE}_one_round_${RUN_TAG}_A1"
D_SUFFIX="balanced_p${DEFENDER_POOL_SIZE}_roleprompt_one_round_${RUN_TAG}_D1_vs_A1"
A_RUN_NAME="upstream_selfredteam_attacker_lora_r32_fromSFT_vs_base_normalmix_harmful_p${ATTACKER_POOL_SIZE}_s${STEPS}_rb${ROLLOUT_BATCH_SIZE}_mb${MICRO_TRAIN_BATCH_SIZE}_tb${TRAIN_BATCH_SIZE}_lr${ATTACKER_LR_TAG}_cosmin_nosft_${A_SUFFIX}"
A_ADAPTER="/output/upstream_selfredteam_role_lora/${A_RUN_NAME}/ckpt/global_step${STEPS}_hf"
DRIVER_LOG="/home/xudong/work/self_play/checkpoints/upstream_role_curriculum_one_round_${RUN_TAG}.driver.log"

{
  echo "run_tag=${RUN_TAG}"
  echo "attacker_pool=harmful:${ATTACKER_POOL_SIZE}"
  echo "defender_pool=balanced:${DEFENDER_POOL_SIZE}"
  echo "attacker_suffix=${A_SUFFIX}"
  echo "defender_suffix=${D_SUFFIX}"
  echo "attacker_adapter=${A_ADAPTER}"
  echo "Starting A1 at $(date --iso-8601=seconds)"

  modal run \
    modal_upstream_selfredteam_role_lora.py::upstream_attacker_lora_normal_mix \
    --steps "${STEPS}" \
    --normal-prompt-pool-size "${ATTACKER_POOL_SIZE}" \
    --normal-prompt-pool-profile harmful \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --micro-train-batch-size "${MICRO_TRAIN_BATCH_SIZE}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --save-steps "${SAVE_STEPS}" \
    --actor-learning-rate "${ATTACKER_LR}" \
    --actor-lr-scheduler cosine_with_min_lr \
    --run-suffix "${A_SUFFIX}"

  echo "A1 completed at $(date --iso-8601=seconds)"
  echo "Starting D1 at $(date --iso-8601=seconds)"

  modal run \
    modal_upstream_selfredteam_role_lora.py::upstream_defender_lora_normal_mix \
    --fixed-attacker-adapter "${A_ADAPTER}" \
    --steps "${STEPS}" \
    --normal-prompt-pool-size "${DEFENDER_POOL_SIZE}" \
    --normal-prompt-pool-profile balanced \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --micro-train-batch-size "${MICRO_TRAIN_BATCH_SIZE}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --save-steps "${SAVE_STEPS}" \
    --actor-learning-rate "${DEFENDER_LR}" \
    --actor-lr-scheduler constant \
    --no-enable-aux-sft \
    --defender-prompt-profile role_specific \
    --run-suffix "${D_SUFFIX}"

  echo "D1 completed at $(date --iso-8601=seconds)"
} 2>&1 | tee -a "${DRIVER_LOG}"
