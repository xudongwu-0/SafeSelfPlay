#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

STEPS="${STEPS:-50}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SFT_STOP_AFTER_STEP="${SFT_STOP_AFTER_STEP:-30}"
SAVE_STEPS="${SAVE_STEPS:-10}"
RUN_SUFFIX="${RUN_SUFFIX:-lora_v2_$(date +%Y%m%d_%H%M%S)}"

# This entrypoint is independent from both the full-parameter command and the
# legacy dynamic-vLLM LoRA adapter. It uses PEFT plus dense merged rollout sync.
exec modal run --detach \
  modal_upstream_selfredteam_role_lora_v2.py::lora_v2_app.lora_v2_attacker_probe \
  --steps "$STEPS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --learning-rate "$LEARNING_RATE" \
  --sft-stop-after-step "$SFT_STOP_AFTER_STEP" \
  --save-steps "$SAVE_STEPS" \
  --run-suffix "$RUN_SUFFIX" \
  --detach
