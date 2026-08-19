#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Durable warm-start latest-opponent chain. It reads verified A1/D1 from the
# completed cold-run state, advances one role at a time, and persists state
# before spawning the next controller.
: "${SOURCE_RUN_SUFFIX:?Set SOURCE_RUN_SUFFIX to the completed cold A1/D1 run}"
LAST_GENERATION="${LAST_GENERATION:-5}"
STEPS_PER_ROLE="${STEPS_PER_ROLE:-100}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ATTACKER_LR="${ATTACKER_LR:-1e-5}"
DEFENDER_LR="${DEFENDER_LR:-4e-5}"
ATTACKER_SFT_STOP_AFTER_STEP="${ATTACKER_SFT_STOP_AFTER_STEP:-30}"
DEFENDER_SFT_STOP_AFTER_STEP="${DEFENDER_SFT_STOP_AFTER_STEP:-10}"
SFT_BATCHES_PER_STEP="${SFT_BATCHES_PER_STEP:-1}"
SAVE_STEPS="${SAVE_STEPS:-10}"
ACTOR_LR_SCHEDULER="${ACTOR_LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.05}"
RUN_SUFFIX="${RUN_SUFFIX:-naive_latest_A2_to_D${LAST_GENERATION}_s${STEPS_PER_ROLE}_$(date +%Y%m%d_%H%M%S)}"

export ROLE_LORA_PSRO_GPU="${ROLE_LORA_PSRO_GPU:-H200:4}"

exec modal run --detach \
  modal_role_lora_zero_sum_psro.py::naive_selfplay_train \
  --source-run-suffix "$SOURCE_RUN_SUFFIX" \
  --continuation-suffix "$RUN_SUFFIX" \
  --last-generation "$LAST_GENERATION" \
  --steps-per-role "$STEPS_PER_ROLE" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --attacker-learning-rate "$ATTACKER_LR" \
  --defender-learning-rate "$DEFENDER_LR" \
  --attacker-sft-stop-after-step "$ATTACKER_SFT_STOP_AFTER_STEP" \
  --defender-sft-stop-after-step "$DEFENDER_SFT_STOP_AFTER_STEP" \
  --sft-batches-per-step "$SFT_BATCHES_PER_STEP" \
  --save-steps "$SAVE_STEPS" \
  --actor-lr-scheduler "$ACTOR_LR_SCHEDULER" \
  --lr-warmup-ratio "$LR_WARMUP_RATIO"
