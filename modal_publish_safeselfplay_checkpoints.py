#!/usr/bin/env python3
"""Publish canonical SafeSelfPlay and reproduced Self-RedTeam checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal


REPO_ID = "xudongwu/SafeSelfPlay-checkpoints"
LORA_VOLUME = modal.Volume.from_name("roll-abs-benchmark-output")
OFFICIAL_VOLUME = modal.Volume.from_name("selfredteam-official-output")
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_transfer]>=0.34,<1"
)
app = modal.App("safeselfplay-checkpoint-publisher", image=IMAGE)

A1 = "/lora_output/upstream_selfredteam_role_lora_v2/attacker_r64a64_s100_lr1e-05_A1_r64_lr1e5_s100_warm5_const_sft30_20260808_080708/ckpt/global_step100_hf"
D1 = "/lora_output/upstream_selfredteam_role_lora_v2/dual_lora_A100D100_r64a64_lora_lr2x_A2e5_D4e5_20260809_150103/D1_lora_s100_vs_A1_s100/ckpt/global_step100_hf"
A2 = "/lora_output/upstream_selfredteam_role_lora_v2/selfplay_A2D2_A80D80_r64a64_formal_selfplay_A2D2_A80D80_A_lr1e-5_D_lr4e-5_20260810_155707/A2_from_A1_s80_vs_D1/ckpt/global_step80_hf"
D2 = "/lora_output/upstream_selfredteam_role_lora_v2/selfplay_A2D2_A80D80_r64a64_formal_selfplay_A2D2_A80D80_A_lr1e-5_D_lr4e-5_20260810_155707/D2_from_D1_s80_vs_A2/ckpt/global_step80_hf"
A3 = "/lora_output/upstream_selfredteam_role_lora_v2/selfplay_A3D3_A80D80_r64a64_formal_selfplay_A3D3_A80D80_A_lr1e-5_D_lr4e-5_20260818_183356/A3_from_A2_s80_vs_D2/ckpt/global_step80_hf"
SELFREDTEAM_STEP200 = "/official_output/selfredteam_official/selfredteam_official_repp_fullptx_sft_meta_llama_31_8b_instruct_abliterated_h200x4_s200_20260803_101623/ckpt/global_step200_hf"


def _require_checkpoint(path: str, adapter: bool) -> Path:
    checkpoint = Path(path)
    required = "adapter_config.json" if adapter else "config.json"
    if not checkpoint.is_dir() or not (checkpoint / required).is_file():
        raise FileNotFoundError(f"Invalid checkpoint {checkpoint}: missing {required}")
    return checkpoint


@app.function(secrets=[modal.Secret.from_name("roll-secrets")])
def initialize_checkpoint_repo() -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("roll-secrets does not contain HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    model_card = """---
library_name: peft
base_model: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
tags:
- reinforcement-learning
- lora
- safety
---

# SafeSelfPlay checkpoints

`lora/A1` through `lora/D3` are the canonical role-specific PEFT adapters.
`self-redteam-reproduction/step200` is our reproduction of public Self-RedTeam
commit `0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123`; it is not an
author-released checkpoint. The authors' weights are available in the
[official collection](https://huggingface.co/collections/mickelliu/self-redteam-68f72b48c4beea864617fe4c).

Training and loading commands are documented in
[SafeSelfPlay](https://github.com/xudongwu-0/SafeSelfPlay).
"""
    api.upload_file(
        repo_id=REPO_ID,
        repo_type="model",
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        commit_message="Initialize checkpoint repository",
    )
    return f"https://huggingface.co/{REPO_ID}"


@app.function(
    cpu=8,
    memory=32768,
    timeout=43200,
    volumes={
        "/lora_output": LORA_VOLUME,
        "/official_output": OFFICIAL_VOLUME,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def publish_checkpoint_bundle(
    d3_checkpoint: str = "",
    include_static: bool = True,
) -> dict[str, object]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("roll-secrets does not contain HF_TOKEN")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    checkpoints: dict[str, tuple[Path, str]] = {}
    if include_static:
        checkpoints.update({
            "lora/A1": (_require_checkpoint(A1, True), "adapter"),
            "lora/D1": (_require_checkpoint(D1, True), "adapter"),
            "lora/A2": (_require_checkpoint(A2, True), "adapter"),
            "lora/D2": (_require_checkpoint(D2, True), "adapter"),
            "lora/A3": (_require_checkpoint(A3, True), "adapter"),
            "self-redteam-reproduction/step200": (
                _require_checkpoint(SELFREDTEAM_STEP200, False),
                "full_model",
            ),
        })
    if d3_checkpoint:
        checkpoints["lora/D3"] = (
            _require_checkpoint(d3_checkpoint.replace("/output/", "/lora_output/", 1), True),
            "adapter",
        )
    if not checkpoints:
        raise ValueError("Select static checkpoints and/or provide D3")
    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)

    published: dict[str, str] = {}
    for destination, (source, checkpoint_type) in checkpoints.items():
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="model",
            folder_path=str(source),
            path_in_repo=destination,
            commit_message=f"Publish {destination}",
        )
        published[destination] = checkpoint_type

    manifest = {
        "base_model": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        "upstream_selfredteam_commit": "0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123",
        "checkpoints_uploaded_in_this_call": published,
        "canonical_layout": {
            "lora": ["A1", "D1", "A2", "D2", "A3", "D3"],
            "self-redteam-reproduction": ["step200"],
        },
        "note": "self-redteam-reproduction is our reproduced run, not an author-released checkpoint.",
    }
    manifest_path = Path("/tmp/checkpoint_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    api.upload_file(
        repo_id=REPO_ID,
        repo_type="model",
        path_or_fileobj=str(manifest_path),
        path_in_repo="checkpoint_manifest.json",
        commit_message="Add checkpoint manifest",
    )
    return {"repo": f"https://huggingface.co/{REPO_ID}", **manifest}


@app.local_entrypoint()
def main(
    d3_checkpoint: str = "",
    include_static: bool = True,
    detach: bool = True,
) -> None:
    if detach:
        call = publish_checkpoint_bundle.spawn(
            d3_checkpoint=d3_checkpoint,
            include_static=include_static,
        )
        print(f"FUNCTION_CALL_ID={call.object_id}")
    else:
        print(json.dumps(
            publish_checkpoint_bundle.remote(
                d3_checkpoint=d3_checkpoint,
                include_static=include_static,
            ),
            indent=2,
        ))
