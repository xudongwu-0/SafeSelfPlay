#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full checkpoint or PEFT adapter on the Modal output volume}"
OUTPUT_SLUG="${OUTPUT_SLUG:-$(basename "$(dirname "${CHECKPOINT_PATH}")")_eval}"
TRAINED_LABEL="${TRAINED_LABEL:-trained_defender}"
EVALUATE_BASE="${EVALUATE_BASE:-false}"

python - "${CHECKPOINT_PATH}" "${OUTPUT_SLUG}" "${TRAINED_LABEL}" "${EVALUATE_BASE}" <<'PY'
import modal
import sys

checkpoint, output_slug, label, evaluate_base = sys.argv[1:]
fn = modal.Function.from_name(
    "selfredteam-official-eval", "evaluate_full_checkpoint_vs_base"
)
call = fn.spawn(
    trained_checkpoint=checkpoint,
    output_slug=output_slug,
    trained_label=label,
    evaluate_base=evaluate_base.lower() in {"1", "true", "yes"},
)
print(f"FUNCTION_CALL_ID={call.object_id}")
PY
