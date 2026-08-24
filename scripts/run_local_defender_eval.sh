#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full checkpoint or PEFT adapter}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a durable result directory}"
SHARED_ENV="${SHARED_ENV:-/export/pgs/wuxudong/.venvs/llm-safety-eval-vllm082/activate-shared.sh}"
GPU_ID="${GPU_ID:-4}"
TASKS="${TASKS:-wildguardtest,wildjailbreak:harmful,do_anything_now,harmbench_precompute,or_bench:toxic,xstest,wildjailbreak:benign,or_bench:hard-1k}"
UPSTREAM_COMMIT="0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
fork_root="${SELFREDTEAM_EVAL_FORK:-$repo_root/../selfplay-redteaming/eval/benchmarks/safety-eval-fork}"
upstream_root="$(cd "$fork_root/../../.." && pwd)"

if [[ -f "$SHARED_ENV" ]]; then
  source "$SHARED_ENV"
fi
for command_name in python hf flock git; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 1
  fi
done
if [[ ! -f "$CHECKPOINT_PATH/adapter_config.json" && ! -f "$CHECKPOINT_PATH/config.json" ]]; then
  echo "Checkpoint is neither a PEFT adapter nor a full HF model: $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -f "$fork_root/evaluation/eval.py" ]]; then
  echo "Pinned safety-eval fork is missing: $fork_root" >&2
  exit 1
fi
actual_commit="$(git -C "$upstream_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$UPSTREAM_COMMIT" ]]; then
  echo "selfplay-redteaming commit mismatch: expected $UPSTREAM_COMMIT, got $actual_commit" >&2
  exit 1
fi

# Serialize downloads so parallel D1/D2 jobs share one copy of every model.
mkdir -p "${HF_HOME:?HF_HOME must be set by the shared environment}"
exec 9>"$HF_HOME/.safeselfplay-official-eval-assets.lock"
flock 9
python - <<'PY'
from huggingface_hub import HfApi
identity = HfApi().whoami()
print("Authenticated Hugging Face user:", identity.get("name", "<unknown>"))
PY
hf download mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
hf download cais/HarmBench-Llama-2-13b-cls
hf download allenai/wildguard --include '*.json' '*.model' '*.safetensors'
hf download allenai/wildguardmix --repo-type dataset
flock -u 9

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$fork_root:$fork_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

exec python "$fork_root/evaluation/eval.py" generators \
  --model_name_or_path "$CHECKPOINT_PATH" \
  --model_input_template_path_or_name llama3_cot \
  --tasks "$TASKS" \
  --report_output_path "$OUTPUT_DIR/metrics.json" \
  --save_individual_results_path "$OUTPUT_DIR/individual.json" \
  --use_vllm=True
