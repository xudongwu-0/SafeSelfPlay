#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

STEPS_PER_ROLE="${STEPS_PER_ROLE:-200}"
ATTACKER_SFT_STOP="${ATTACKER_SFT_STOP:-30}"
DEFENDER_SFT_STOP="${DEFENDER_SFT_STOP:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
RUN_SUFFIX="${RUN_SUFFIX:-duallora_r64_$(date +%Y%m%d_%H%M%S)}"

export UPSTREAM_ROLE_LORA_GPU="${UPSTREAM_ROLE_LORA_GPU:-H200:4}"

exec modal run --detach \
  modal_upstream_selfredteam_role_lora.py::dynamic_sft_dual_lora_round \
  --steps-per-role "${STEPS_PER_ROLE}" \
  --attacker-sft-stop-after-step "${ATTACKER_SFT_STOP}" \
  --defender-sft-stop-after-step "${DEFENDER_SFT_STOP}" \
  --learning-rate "${LEARNING_RATE}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --run-suffix "${RUN_SUFFIX}"
