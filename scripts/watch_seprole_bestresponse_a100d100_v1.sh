#!/usr/bin/env bash
set -euo pipefail

TRAIN_CALL_ID="fc-01KZ5YFNQXMV2YYYCXMF9QHRMN"
RUN_NAME="seprole_qwen25_3b_duallora_r32_sftA_baseD_phased_A100_D100_s200_rb128_tb32_mb8_aLR2e-6_dLR3e-6_kl0p01_aux0p1every4_ourprompts_20260804_163315_bestresponse_v1"
VOLUME="roll-abs-benchmark-output"
LOCAL_ROOT="/home/xudong/work/self_play/checkpoints/seprole_bestresponse_a100d100_v1_20260804"

mkdir -p "$LOCAL_ROOT"
echo "[$(date --iso-8601=seconds)] Waiting for $TRAIN_CALL_ID"
python - "$TRAIN_CALL_ID" <<'PY'
import modal
import sys

call = modal.functions.FunctionCall.from_id(sys.argv[1])
print(call.get(timeout=12 * 60 * 60))
PY

for step in 50 100 150 200; do
    echo "[$(date --iso-8601=seconds)] Downloading step ${step}"
    modal volume get --force "$VOLUME" \
        "/abs_bipolicy_h200/$RUN_NAME/ckpt/global_step${step}_hf" \
        "$LOCAL_ROOT"
done

modal volume get --force "$VOLUME" \
    "/abs_bipolicy_h200/$RUN_NAME/manifest.json" \
    "$LOCAL_ROOT"
modal volume get --force "$VOLUME" \
    "/abs_bipolicy_h200/$RUN_NAME/run_status.json" \
    "$LOCAL_ROOT"
echo "[$(date --iso-8601=seconds)] Training and checkpoint downloads completed"
