#!/usr/bin/env bash
set -euo pipefail

# Strict Self-RedTeam control: preserve the official game and optimization
# settings, but optimize independent attacker/defender LoRA adapters in A50/D50
# phases. The remote coordinator survives local disconnects.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export UPSTREAM_ROLE_LORA_GPU="H200:4"

modal run --detach \
  modal_upstream_selfredteam_role_lora.py::strict_upstream_aligned_role_round \
  --steps-per-role 50 \
  --prompt-pool-size 0 \
  --rollout-batch-size 128 \
  --micro-rollout-batch-size 8 \
  --micro-train-batch-size 8 \
  --train-batch-size 32 \
  --save-steps 50 \
  --attacker-learning-rate 5e-7 \
  --defender-learning-rate 5e-7 \
  --init-kl-coef 0.01 \
  --attacker-enable-aux-sft \
  --defender-enable-aux-sft \
  --base-model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
  --attacker-prompt-profile upstream \
  --run-suffix "${RUN_SUFFIX}"

