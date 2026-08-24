#!/usr/bin/env bash
set -euo pipefail

: "${LOCAL_RUN_ROOT:?Set LOCAL_RUN_ROOT to the downloaded cold-PSRO run directory}"
HF_REPO="${HF_REPO:-xudongwu/SafeSelfPlay-checkpoints}"
RUN_NAME="$(basename "$LOCAL_RUN_ROOT")"
HF_PATH="${HF_PATH:-cold-psro/$RUN_NAME}"
SHARED_ENV="${SHARED_ENV:-/export/pgs/wuxudong/.venvs/llm-safety-eval-vllm082/activate-shared.sh}"

if [[ -f "$SHARED_ENV" ]]; then
  source "$SHARED_ENV"
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI not found; activate an environment containing huggingface-hub" >&2
  exit 1
fi
if [[ ! -f "$LOCAL_RUN_ROOT/state.json" || ! -f "$LOCAL_RUN_ROOT/checkpoint_inventory.json" ]]; then
  echo "Not a durable cold-PSRO run root: $LOCAL_RUN_ROOT" >&2
  exit 1
fi
if [[ -f "$LOCAL_RUN_ROOT/SHA256SUMS" ]]; then
  (cd "$LOCAL_RUN_ROOT" && sha256sum --check --quiet SHA256SUMS)
  echo "Verified full-run SHA256SUMS"
fi

python - "$LOCAL_RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve()
inventory = json.loads((root / "checkpoint_inventory.json").read_text())
state = json.loads((root / "state.json").read_text())
if inventory.get("schema_version") != "role-lora-cold-psro-checkpoints-v1":
    raise SystemExit("unsupported checkpoint inventory schema")
if state.get("contract", {}).get("schema_version") != "role-lora-cold-zero-sum-psro-v1":
    raise SystemExit("unsupported cold-PSRO state schema")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

verified = []
for label, record in inventory.get("checkpoints", {}).items():
    if label in {"A0", "D0"}:
        continue
    checkpoint_name = PurePosixPath(str(record.get("path", ""))).name
    if not checkpoint_name.startswith("global_step") or not checkpoint_name.endswith("_hf"):
        raise SystemExit(f"invalid checkpoint path in inventory for {label}")
    checkpoint = root / "training" / label / "ckpt" / checkpoint_name
    weights = checkpoint / str(record.get("weights_file", ""))
    config = checkpoint / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise SystemExit(f"missing final adapter files for {label}: {checkpoint}")
    if sha256(weights) != record.get("adapter_sha256"):
        raise SystemExit(f"adapter SHA-256 mismatch for {label}")
    if sha256(config) != record.get("config_sha256"):
        raise SystemExit(f"config SHA-256 mismatch for {label}")
    verified.append(label)

if not verified:
    raise SystemExit("inventory has no learned checkpoints")
print("Verified checkpoint identities:", ", ".join(verified))
print("Run stage:", state.get("stage"), "completed:", state.get("completed"))
PY

python - <<'PY'
from huggingface_hub import HfApi
identity = HfApi().whoami()
print("Authenticated Hugging Face user:", identity.get("name", "<unknown>"))
PY
echo "Uploading $LOCAL_RUN_ROOT to https://huggingface.co/$HF_REPO/tree/main/$HF_PATH"
hf upload \
  "$HF_REPO" \
  "$LOCAL_RUN_ROOT" \
  "$HF_PATH" \
  --repo-type model \
  --exclude '*.lock' '__pycache__/*' '*.pyc' \
  --commit-message "Upload durable cold PSRO run $RUN_NAME"
