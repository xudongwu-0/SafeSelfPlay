#!/usr/bin/env bash
set -euo pipefail

TRAIN_CALL_ID="fc-01KZ5HX0RMZMMSP2DSB5T58KQS"
RUN_NAME="abs_qwen25_3b_duallora_r32_phased_A50_D50_s100_rb128_tb32_mb8_aLR2e-6_dLR3e-6_kl0p3_rolesft_ourprompts_20260804_125321_spawn"
VOLUME="roll-abs-benchmark-output"
LOCAL_ROOT="/home/xudong/work/self_play/checkpoints/abs_duallora_phased_a50d50_retry1_20260804"
EVAL_SLUG="qwen25_3b_duallora_phased_a50d50_retry1_20260804"

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
echo "[$(date --iso-8601=seconds)] Waiting for phased retry $TRAIN_CALL_ID"
wait_for_call "$TRAIN_CALL_ID"

for step in 50 100; do
    echo "[$(date --iso-8601=seconds)] Downloading step${step} checkpoint"
    modal volume get --force "$VOLUME" \
        "/abs_bipolicy_h200/$RUN_NAME/ckpt/global_step${step}_hf" \
        "$LOCAL_ROOT"
done

echo "[$(date --iso-8601=seconds)] Starting final defender evaluation"
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
    model_label="Our phased dual-LoRA A50-D50 retry1 (Qwen2.5-3B)",
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
echo "[$(date --iso-8601=seconds)] Training, evaluation, and downloads completed"
