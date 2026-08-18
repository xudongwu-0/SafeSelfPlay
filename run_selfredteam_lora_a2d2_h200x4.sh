#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Formal second self-play iteration:
#   A2 = continue A1 for 80 steps against frozen D1.
#   D2 = continue D1 for 80 steps against frozen A2.
# The CPU orchestration task keeps running remotely after SSH disconnects and
# starts the released five-benchmark defender evaluation when D2 is complete.
ATTACKER_START_ADAPTER="${ATTACKER_START_ADAPTER:-/output/upstream_selfredteam_role_lora_v2/attacker_r64a64_s100_lr1e-05_A1_r64_lr1e5_s100_warm5_const_sft30_20260808_080708/ckpt/global_step100_hf}"
DEFENDER_START_ADAPTER="${DEFENDER_START_ADAPTER:-/output/upstream_selfredteam_role_lora_v2/dual_lora_A100D100_r64a64_lora_lr2x_A2e5_D4e5_20260809_150103/D1_lora_s100_vs_A1_s100/ckpt/global_step100_hf}"
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
RUN_SUFFIX="${RUN_SUFFIX:-formal_selfplay_A2D2_A${STEPS_PER_ROLE}D${STEPS_PER_ROLE}_A_lr${ATTACKER_LR}_D_lr${DEFENDER_LR}_$(date +%Y%m%d_%H%M%S)}"

export UPSTREAM_ROLE_LORA_V2_GPU="${UPSTREAM_ROLE_LORA_V2_GPU:-H200:4}"

exec modal run --detach \
  modal_upstream_selfredteam_role_lora_v2.py::lora_v2_app.train_lora_v2_a2_d2_and_eval \
  --attacker-start-adapter "$ATTACKER_START_ADAPTER" \
  --defender-start-adapter "$DEFENDER_START_ADAPTER" \
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
