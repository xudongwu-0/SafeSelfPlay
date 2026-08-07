#!/usr/bin/env python3
"""Stable WildGuard service for the official Self-RedTeam reproduction.

The public training script issues one reward request per actor rank.  The old
A10G service used HF batch size 16 and intermittently OOMed on long generated
prompts, turning a single reward request into a fatal training error.  This
dedicated service uses the paper-equivalent L40-class reward GPU, serializes
requests within each container, and adaptively lowers the classifier batch size
if an unusually long batch still exhausts memory.
"""
from __future__ import annotations

import gc
import sys
import threading
from typing import Any

import modal

if "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_abs_benchmark import (
    _classify_wildguard_payload,
    _get_wildguard_model,
    hf_cache,
    image as base_image,
)


image = base_image
runtime_config = modal.Secret.from_dict(
    {
        "ABS_RM_BATCH_SIZE": "4",
        "ABS_RM_USE_VLLM": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)
app = modal.App("selfredteam-wildguard", image=image)
_CLASSIFY_LOCK = threading.Lock()


def _classify_with_oom_backoff(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    with _CLASSIFY_LOCK:
        classifier = _get_wildguard_model()
        original_batch_size = classifier.batch_size
        batch_sizes = []
        size = max(1, original_batch_size)
        while size not in batch_sizes:
            batch_sizes.append(size)
            if size == 1:
                break
            size = max(1, size // 2)

        last_error: BaseException | None = None
        for batch_size in batch_sizes:
            classifier.batch_size = batch_size
            try:
                return _classify_wildguard_payload(payload)
            except torch.OutOfMemoryError as error:
                last_error = error
                print(
                    f"WildGuard OOM at batch_size={batch_size}; retrying smaller batch",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
        assert last_error is not None
        raise last_error


@app.function(
    gpu="L40S",
    cpu=8,
    memory=65536,
    timeout=43200,
    max_containers=4,
    scaledown_window=1200,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("roll-secrets"), runtime_config],
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app(label="selfredteam-wildguard")
def wildguard_reward_app():
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse

    web_app = FastAPI()

    @web_app.get("/health")
    async def health():
        return {
            "ok": True,
            "model": "allenai/wildguard",
            "gpu": "L40S",
            "batch_size": 4,
        }

    @web_app.post("/classify")
    def classify(payload: dict[str, Any] = Body(...)):
        return JSONResponse(_classify_with_oom_backoff(payload))

    return web_app
