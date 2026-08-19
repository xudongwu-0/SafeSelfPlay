#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Canonical cold-start PSRO LoRA run. Both adapters start independently from
# the same base model; training keeps the general-sum reward and applies the
# asymmetric generated-label-drift policy before PPO replay.
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
RUN_SUFFIX="${RUN_SUFFIX:-lora_A${STEPS_PER_ROLE}D${STEPS_PER_ROLE}_r${LORA_RANK}a${LORA_ALPHA}_A_lr${ATTACKER_LR}_D_lr${DEFENDER_LR}_$(date +%Y%m%d_%H%M%S)}"

# Modal --detach keeps the remote A1 -> D1 call alive after SSH disconnects.
export ROLE_LORA_PSRO_GPU="${ROLE_LORA_PSRO_GPU:-H200:4}"

exec modal run --detach \
  modal_role_lora_zero_sum_psro.py::cold_start_train \
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
  --lr-warmup-ratio "$LR_WARMUP_RATIO" \
  --run-suffix "$RUN_SUFFIX"
