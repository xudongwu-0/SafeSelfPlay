#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Canonical sequential full-parameter run. A1 and D1 are separate 8B models:
# A1 starts from base and plays base D; D1 starts from base and plays frozen A1.
STEPS_PER_ROLE="${STEPS_PER_ROLE:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ATTACKER_SFT_STOP_AFTER_STEP="${ATTACKER_SFT_STOP_AFTER_STEP:-30}"
DEFENDER_SFT_STOP_AFTER_STEP="${DEFENDER_SFT_STOP_AFTER_STEP:-10}"
RUN_SUFFIX="${RUN_SUFFIX:-full_A${STEPS_PER_ROLE}D${STEPS_PER_ROLE}_lr${LEARNING_RATE}_$(date +%Y%m%d_%H%M%S)}"

# Keep the tested four-H200 allocation unless explicitly overridden.
export UPSTREAM_ROLE_FULL_GPU="${UPSTREAM_ROLE_FULL_GPU:-H200:4}"

exec modal run --detach \
  modal_upstream_selfredteam_role_full.py::app.train_dynamic_sft_dual_full_round \
  --steps-per-role "$STEPS_PER_ROLE" \
  --attacker-sft-stop-after-step "$ATTACKER_SFT_STOP_AFTER_STEP" \
  --defender-sft-stop-after-step "$DEFENDER_SFT_STOP_AFTER_STEP" \
  --learning-rate "$LEARNING_RATE" \
  --run-suffix "$RUN_SUFFIX"
