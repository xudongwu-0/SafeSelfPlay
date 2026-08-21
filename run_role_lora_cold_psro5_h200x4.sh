#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Canonical cold, base-inclusive sequential double-oracle run:
# A0/D0 are the frozen base; every A1-A5/D1-D5 oracle is a fresh rank-64
# adapter and optimizer; every matrix cell retains exactly 4000 games.
GENERATIONS="${GENERATIONS:-5}"
MATRIX_EPISODES="${MATRIX_EPISODES:-4000}"
STEPS_PER_ROLE="${STEPS_PER_ROLE:-100}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ATTACKER_LR="${ATTACKER_LR:-1e-5}"
DEFENDER_LR="${DEFENDER_LR:-4e-5}"
ATTACKER_SFT_STOP_AFTER_STEP="${ATTACKER_SFT_STOP_AFTER_STEP:-30}"
DEFENDER_SFT_STOP_AFTER_STEP="${DEFENDER_SFT_STOP_AFTER_STEP:-10}"
SFT_BATCHES_PER_STEP="${SFT_BATCHES_PER_STEP:-1}"
# The formal workflow retains only the terminal LoRA for each role.  A custom
# lower interval is still accepted for debugging, but completed intermediates
# are pruned by the trainer before it returns.
SAVE_STEPS="${SAVE_STEPS:-$STEPS_PER_ROLE}"
ACTOR_LR_SCHEDULER="${ACTOR_LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.05}"
TRAINING_SEED="${TRAINING_SEED:-8888}"
SEED_BASE="${SEED_BASE:-8888}"
MAX_CANDIDATE_MULTIPLIER="${MAX_CANDIDATE_MULTIPLIER:-4}"
CANDIDATE_WAVE_PAIRS="${CANDIDATE_WAVE_PAIRS:-64}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-64}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
RUN_SUFFIX="${RUN_SUFFIX:-cold_psro${GENERATIONS}_base_n${MATRIX_EPISODES}_s${STEPS_PER_ROLE}_$(date +%Y%m%d_%H%M%S)}"
export ROLE_LORA_PSRO_GPU="${ROLE_LORA_PSRO_GPU:-H200:4}"
export ROLE_LORA_PSRO_PAYOFF_GPU="${ROLE_LORA_PSRO_PAYOFF_GPU:-H200}"

MODAL_BIN="${MODAL_BIN:-}"
if [[ -z "$MODAL_BIN" ]]; then
  if command -v modal >/dev/null 2>&1; then
    MODAL_BIN="$(command -v modal)"
  else
    MODAL_BIN="/export/pgs/wuxudong/.cache/uv/archive-v0/CmbsW0bkgE7_0P2w/bin/modal"
  fi
fi
if [[ ! -x "$MODAL_BIN" ]]; then
  echo "Modal CLI not found; set MODAL_BIN to its executable path" >&2
  exit 1
fi

# The local proxy configuration cannot reach Modal's control plane.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

echo "PSRO_RUN_SUFFIX=$RUN_SUFFIX"
echo "PSRO_STATE=/output/role_lora_zero_sum_psro/$RUN_SUFFIX/state.json"
echo "PSRO_CHECKPOINT_INVENTORY=/output/role_lora_zero_sum_psro/$RUN_SUFFIX/checkpoint_inventory.json"

exec "$MODAL_BIN" run --detach \
  modal_role_lora_zero_sum_psro.py::cold_psro_train_and_eval \
  --run-suffix "$RUN_SUFFIX" \
  --generations "$GENERATIONS" \
  --matrix-episodes "$MATRIX_EPISODES" \
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
  --training-seed "$TRAINING_SEED" \
  --seed-base "$SEED_BASE" \
  --max-candidate-multiplier "$MAX_CANDIDATE_MULTIPLIER" \
  --candidate-wave-pairs "$CANDIDATE_WAVE_PAIRS" \
  --generation-batch-size "$GENERATION_BATCH_SIZE" \
  --judge-batch-size "$JUDGE_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS"
