#!/usr/bin/env bash
set -euo pipefail

: "${ATTACKER_ADAPTER:?Set ATTACKER_ADAPTER to the frozen A<N> adapter}"
: "${DEFENDER_START_ADAPTER:?Set DEFENDER_START_ADAPTER to D<N-1>}"

TARGET_GENERATION="${TARGET_GENERATION:-3}"
STEPS="${STEPS:-80}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
SFT_STOP_AFTER_STEP="${SFT_STOP_AFTER_STEP:-10}"
SFT_BATCHES_PER_STEP="${SFT_BATCHES_PER_STEP:-1}"
SAVE_STEPS="${SAVE_STEPS:-10}"
ACTOR_LR_SCHEDULER="${ACTOR_LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.05}"
RUN_SUFFIX="${RUN_SUFFIX:-formal_D${TARGET_GENERATION}_recovery_$(date +%Y%m%d_%H%M%S)}"

export UPSTREAM_ROLE_LORA_V2_GPU="${UPSTREAM_ROLE_LORA_V2_GPU:-H200:4}"

exec modal run --detach \
  modal_upstream_selfredteam_role_lora_v2.py::lora_v2_app.train_lora_v2_defender_only_and_eval \
  --attacker-adapter "${ATTACKER_ADAPTER}" \
  --defender-start-adapter "${DEFENDER_START_ADAPTER}" \
  --target-generation "${TARGET_GENERATION}" \
  --steps "${STEPS}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --learning-rate "${LEARNING_RATE}" \
  --sft-stop-after-step "${SFT_STOP_AFTER_STEP}" \
  --sft-batches-per-step "${SFT_BATCHES_PER_STEP}" \
  --save-steps "${SAVE_STEPS}" \
  --actor-lr-scheduler "${ACTOR_LR_SCHEDULER}" \
  --lr-warmup-ratio "${LR_WARMUP_RATIO}" \
  --run-suffix "${RUN_SUFFIX}"
