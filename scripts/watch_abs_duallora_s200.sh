#!/usr/bin/env bash
set -euo pipefail

TRAIN_CALL_ID="fc-01KZ4CW4TCZP8BGQFK86XJHZCG"
S100_EVAL_CALL_ID="fc-01KZ4D0DV5PKQ045E0CT92QPGZ"
S200_EVAL_CALL_ID="fc-01KZ4DE5FYT722HHC20FRJSA96"
RUN_NAME="abs_qwen25_3b_duallora_r32_simultaneous_from_s100_to_s200_rb128_tb32_mb8_aLR1e-6_dLR3e-6_kl0p3_rolesft_ourprompts_20260804_020616_spawn"
VOLUME="roll-abs-benchmark-output"
LOCAL_ROOT="/home/xudong/work/self_play/checkpoints/abs_duallora_s200_20260804"

wait_for_call() {
    local call_id="$1"
    python - "$call_id" <<'PY'
import modal
import sys

call_id = sys.argv[1]
result = modal.functions.FunctionCall.from_id(call_id).get(timeout=24 * 60 * 60)
print(result)
PY
}

mkdir -p "$LOCAL_ROOT"
echo "[$(date --iso-8601=seconds)] Waiting for training $TRAIN_CALL_ID"
wait_for_call "$TRAIN_CALL_ID"

echo "[$(date --iso-8601=seconds)] Downloading step200 checkpoint"
modal volume get --force "$VOLUME" \
    "/abs_bipolicy_h200/$RUN_NAME/ckpt/global_step200_hf" \
    "$LOCAL_ROOT/global_step200_hf"

echo "[$(date --iso-8601=seconds)] Waiting for step100 released-pipeline evaluation"
wait_for_call "$S100_EVAL_CALL_ID"
modal volume get --force "$VOLUME" \
    "/abs_bipolicy_official_eval/qwen25_3b_duallora_s100_defender" \
    "$LOCAL_ROOT/official_eval_s100"

echo "[$(date --iso-8601=seconds)] Waiting for step200 released-pipeline evaluation"
wait_for_call "$S200_EVAL_CALL_ID"
modal volume get --force "$VOLUME" \
    "/abs_bipolicy_official_eval/qwen25_3b_duallora_s200_defender_20260804" \
    "$LOCAL_ROOT/official_eval_s200"

echo "[$(date --iso-8601=seconds)] All downloads completed"
