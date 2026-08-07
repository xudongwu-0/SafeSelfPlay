#!/usr/bin/env bash
set -euo pipefail

TRAIN_CALL_ID="fc-01KZ57W7E6KBEHWFRKHHW837ZJ"
RUN_NAME="abs_qwen25_3b_duallora_r32_simultaneous_s100_rb128_tb32_mb8_aLR2e-6_dLR3e-6_kl0p3_rolesft_ourprompts_20260804_095810_spawn"
VOLUME="roll-abs-benchmark-output"
LOCAL_ROOT="/home/xudong/work/self_play/checkpoints/abs_duallora_s100_bugfix_lr2e6_20260804"
EVAL_SLUG="qwen25_3b_duallora_s100_bugfix_lr2e6_20260804"

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

echo "[$(date --iso-8601=seconds)] Downloading step100 checkpoint"
modal volume get --force "$VOLUME" \
    "/abs_bipolicy_h200/$RUN_NAME/ckpt/global_step100_hf" \
    "$LOCAL_ROOT/global_step100_hf"

echo "[$(date --iso-8601=seconds)] Starting released-pipeline defender evaluation"
EVAL_CALL_ID="$(python - "$RUN_NAME" "$EVAL_SLUG" <<'PY'
import modal
import sys

run_name, output_slug = sys.argv[1:]
function = modal.Function.from_name(
    "abs-bipolicy-official-eval", "evaluate_defender"
)
call = function.spawn(
    source_checkpoint=(
        f"/output/abs_bipolicy_h200/{run_name}/ckpt/global_step100_hf"
    ),
    output_slug=output_slug,
    model_label="Our dual-LoRA step100 bugfix LR2e-6 (Qwen2.5-3B)",
)
print(call.object_id)
PY
)"
echo "$EVAL_CALL_ID" > "$LOCAL_ROOT/eval_call_id.txt"
echo "[$(date --iso-8601=seconds)] Waiting for evaluation $EVAL_CALL_ID"
wait_for_call "$EVAL_CALL_ID"

modal volume get --force "$VOLUME" \
    "/abs_bipolicy_official_eval/$EVAL_SLUG" \
    "$LOCAL_ROOT/official_eval"
echo "[$(date --iso-8601=seconds)] All downloads completed"
