"""FSP checkpoint helpers: convert DeepSpeed-saved LoRA checkpoint into a
minimal PEFT adapter that vLLM's `from_local_checkpoint` can load, and clean
up LoRA weight directories at end of run.
"""

import logging
import os
import shutil
import time
from collections import OrderedDict
from typing import Iterable

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def _load_when_ready(ds_path: str, max_wait_s: int = 120):
    """torch.load with retries on partial-file errors from an in-flight async upload.

    Yields cleanly once the file reads without PytorchStreamReader errors.
    """
    last_err = None
    deadline = time.monotonic() + max_wait_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            return torch.load(ds_path, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError) as e:
            msg = str(e)
            if ("PytorchStreamReader" in msg
                    or "central directory" in msg
                    or "unexpected EOF" in msg
                    or "No such file" in msg):
                last_err = e
                time.sleep(min(2 ** min(attempt, 4), 10))
                continue
            raise
    raise RuntimeError(f"torch.load never succeeded for {ds_path}: {last_err}")


def export_peft_adapter(ckpt_dir: str) -> str:
    """Convert DeepSpeed `checkpoint/mp_rank_00_model_states.pt` into
    `<ckpt_dir>/adapter_model.safetensors` in PEFT format, then remove the
    DeepSpeed subdirectory (optimizer states + model state pt).

    Retries torch.load to ride out an in-flight async upload.
    Returns the path to the written safetensors file.
    """
    ds_path = os.path.join(ckpt_dir, "checkpoint", "mp_rank_00_model_states.pt")
    raw = _load_when_ready(ds_path)
    if isinstance(raw, dict) and "module" in raw:
        state_dict = raw["module"]
    else:
        state_dict = raw

    lora = OrderedDict()
    for k, v in state_dict.items():
        if "lora_" not in k:
            continue
        new_k = k.replace(".default.weight", ".weight").replace(".default.bias", ".bias")
        if not isinstance(v, torch.Tensor):
            continue
        lora[new_k] = v.detach().contiguous().cpu()

    if not lora:
        raise RuntimeError(f"no LoRA tensors found under {ds_path}")

    out_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    save_file(lora, out_path)
    logger.info(f"FSP: wrote {len(lora)} LoRA tensors to {out_path}")

    shutil.rmtree(os.path.join(ckpt_dir, "checkpoint"), ignore_errors=True)
    return out_path


def cleanup_fsp_weights(paths: Iterable[str]) -> None:
    """Remove FSP checkpoint directories (LoRA weight folders).

    Logs + debug messages outside `checkpoint-*` are untouched.
    Best-effort: missing / already-deleted paths are logged and skipped.
    """
    for p in paths:
        if not p:
            continue
        try:
            shutil.rmtree(p)
            logger.info(f"FSP cleanup: removed {p}")
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning(f"FSP cleanup: failed to remove {p}: {e}")
