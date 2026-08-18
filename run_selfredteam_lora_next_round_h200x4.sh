#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Formal continued self-play iteration. With SOURCE_GENERATION=2:
#   A3 = continue A2 for 80 steps against frozen D2.
#   D3 = continue D2 for 80 steps against frozen A3.
# The CPU orchestration task keeps running remotely after SSH disconnects and
# starts the released five-benchmark defender evaluation after the new D model.
: "${ATTACKER_START_ADAPTER:?Set ATTACKER_START_ADAPTER to the source attacker adapter path}"
: "${DEFENDER_START_ADAPTER:?Set DEFENDER_START_ADAPTER to the source defender adapter path}"
SOURCE_GENERATION="${SOURCE_GENERATION:-1}"
TARGET_GENERATION="$((SOURCE_GENERATION + 1))"
STEPS_PER_ROLE="${STEPS_PER_ROLE:-80}"
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
RUN_SUFFIX="${RUN_SUFFIX:-formal_selfplay_A${TARGET_GENERATION}D${TARGET_GENERATION}_A${STEPS_PER_ROLE}D${STEPS_PER_ROLE}_A_lr${ATTACKER_LR}_D_lr${DEFENDER_LR}_$(date +%Y%m%d_%H%M%S)}"

export UPSTREAM_ROLE_LORA_V2_GPU="${UPSTREAM_ROLE_LORA_V2_GPU:-H200:4}"

exec modal run --detach \
  modal_upstream_selfredteam_role_lora_v2.py::lora_v2_app.train_lora_v2_a2_d2_and_eval \
  --attacker-start-adapter "$ATTACKER_START_ADAPTER" \
  --defender-start-adapter "$DEFENDER_START_ADAPTER" \
  --source-generation "$SOURCE_GENERATION" \
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
