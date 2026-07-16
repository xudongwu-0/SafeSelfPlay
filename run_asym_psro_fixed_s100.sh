#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xudong/work/self_play"
STATE="${ROOT}/checkpoints/roll_abs_benchmark/asym_psro_state.json"
OUT_DIR="${ROOT}/checkpoints/roll_abs_benchmark"
RUN_SUFFIX="asympsro_s50_to_s100_fixed_pool_kl0p3_v5"
EVAL_SUFFIX="asympsro_s100_fixed_pool_v5_full"

export ABS_RM_GPU="${ABS_RM_GPU:-A100-40GB}"
export ABS_RM_MAX_CONTAINERS="${ABS_RM_MAX_CONTAINERS:-1}"
export ABS_RM_BATCH_SIZE="${ABS_RM_BATCH_SIZE:-2}"
export ABS_RM_USE_VLLM="${ABS_RM_USE_VLLM:-0}"
export ABS_SEQUENCE_LENGTH="${ABS_SEQUENCE_LENGTH:-2048}"
export ABS_MAX_TOKENS_PER_STEP="${ABS_MAX_TOKENS_PER_STEP:-512}"
export ABS_MAX_NEW_TOKENS="${ABS_MAX_NEW_TOKENS:-512}"
export ABS_VLLM_GPU_MEMORY_UTILIZATION="${ABS_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export ABS_VLLM_MAX_NUM_BATCHED_TOKENS="${ABS_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export ABS_VLLM_ENFORCE_EAGER="${ABS_VLLM_ENFORCE_EAGER:-true}"
export ROLL_ACTOR_INFER_MAX_CONCURRENCY="${ROLL_ACTOR_INFER_MAX_CONCURRENCY:-64}"

if [[ ! -f "${STATE}" ]]; then
  echo "Missing state file: ${STATE}" >&2
  exit 1
fi

ATTACKER_CKPT="$(
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["attacker"]["remote_checkpoint"])' "${STATE}"
)"
DEFENDER_CKPT="$(
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["defender"]["remote_checkpoint"])' "${STATE}"
)"

echo "Using attacker init: ${ATTACKER_CKPT}"
echo "Using defender init: ${DEFENDER_CKPT}"
echo "Run suffix: ${RUN_SUFFIX}"

cd "${ROOT}"

modal run ROLL/modal_abs_benchmark.py \
  --mode asym-psro-continue \
  --attacker-checkpoint-path "${ATTACKER_CKPT}" \
  --defender-checkpoint-path "${DEFENDER_CKPT}" \
  --output-step-offset 50 \
  --asym-role-steps 50 \
  --run-suffix "${RUN_SUFFIX}" \
  --rollout-batch-size 48 \
  --train-env-groups 12 \
  --train-group-size 4 \
  --max-env-num-per-worker 12 \
  --val-env-groups 1 \
  --val-group-size 1 \
  --psro-max-concurrent 1 \
  --train-micro-batch 1 \
  --grad-accum 16 \
  --train-infer-batch 1 \
  --save-steps 10 \
  --actor-lr 2.0e-6 \
  --init-kl-coef 0.3 \
  --kl-loss-coef 0.3 \
  --use-kl-loss true

CONTINUE_STATE="${OUT_DIR}/asym_psro_continue_s100_state.json"
FIXED_STATE="${OUT_DIR}/asym_psro_continue_s100_fixed_pool_v5_state.json"
cp "${CONTINUE_STATE}" "${FIXED_STATE}"

FIXED_DEFENDER_CKPT="$(
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["defender_checkpoint"])' "${FIXED_STATE}"
)"

echo "Fixed defender checkpoint: ${FIXED_DEFENDER_CKPT}"

modal run ROLL/modal_abs_benchmark.py \
  --mode eval \
  --checkpoint-path "${FIXED_DEFENDER_CKPT}" \
  --eval-suffix "${EVAL_SUFFIX}"

modal run ROLL/modal_abs_benchmark.py \
  --mode upload-wandb-results || true

echo "Done. Fixed state: ${FIXED_STATE}"
