#!/usr/bin/env python3
"""Modal LoRA SFT for the ABS attacker rewrite policy.

This trains a small Qwen2.5-3B attacker adapter on the cleaned teacher rewrite
data generated under checkpoints/abs_attacker_sft_data. It intentionally uses a
plain HuggingFace Transformers + PEFT stack rather than the ROLL RL pipeline.

Setup:
    modal secret create roll-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>

Run:
    modal run modal_abs_attacker_sft.py

Optional:
    ABS_SFT_GPU=A10G modal run modal_abs_attacker_sft.py --epochs 1

Outputs are written to the Modal volume:
    /output/abs_attacker_sft/<run_id>

and downloaded locally to:
    /home/xudong/work/self_play/checkpoints/abs_attacker_sft_runs
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


OUTPUT_VOLUME_NAME = "roll-abs-benchmark-output"
LOCAL_OUTPUT_DIR = "/home/xudong/work/self_play/checkpoints/abs_attacker_sft_runs"
LOCAL_SFT_DATA = (
    Path(__file__).resolve().parent.parent
    / "checkpoints"
    / "abs_attacker_sft_data"
    / "attacker_sft_selected_hb_1200_h60_b40_synthtest_t25_20260714_221622"
    / "sft_train.cleaned.jsonl"
)
LOCAL_SFT_REPORT = LOCAL_SFT_DATA.with_name("sft_train.cleaned_report.md")


hf_cache = modal.Volume.from_name("roll-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(["git", "gcc", "g++"])
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate>=1.1.0",
        "datasets>=2.20.0",
        "peft>=0.17.0,<0.20.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "transformers>=4.51.0,<5.0.0",
        "wandb",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0.0",
    )
    .add_local_file(str(LOCAL_SFT_DATA), "/data/sft_train.cleaned.jsonl", copy=True)
    .add_local_file(str(LOCAL_SFT_REPORT), "/data/sft_train.cleaned_report.md", copy=True)
)

app = modal.App("roll-abs-attacker-sft", image=image)


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


def _prepare_env() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", "self-play")
    token = _hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(token)


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _format_prompt(tokenizer: Any, messages: list[dict[str, str]], *, assistant: bool) -> str:
    if assistant:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_train_features(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, list[int]]]:
    features: list[dict[str, list[int]]] = []
    skipped = 0
    for row in rows:
        messages = row["messages"]
        full_text = _format_prompt(tokenizer, messages, assistant=True)
        prompt_text = _format_prompt(tokenizer, messages[:-1], assistant=False)
        full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
        prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        if len(input_ids) < 8:
            skipped += 1
            continue
        labels = list(input_ids)
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            skipped += 1
            continue
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    if skipped:
        print(f"[data] skipped {skipped} rows after tokenization", flush=True)
    return features


@dataclass
class CausalLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        input_features = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features
        ]
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            row = list(f["labels"])
            row.extend([-100] * (max_len - len(row)))
            labels.append(row)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


@app.function(
    gpu=os.environ.get("ABS_SFT_GPU", "A100-80GB"),
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("roll-secrets")],
    volumes={"/root/.cache/huggingface": hf_cache, "/output": output_vol},
)
def train_attacker_sft(
    run_id: str = "",
    base_model: str = BASE_MODEL,
    epochs: float = 2.0,
    learning_rate: float = 5e-5,
    per_device_batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 1024,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _prepare_env()
    set_seed(seed)
    if not run_id:
        run_id = "abs_attacker_sft_qwen25_3b_lora_r32_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_dir = Path("/output/abs_attacker_sft") / run_id
    remote_run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = remote_run_dir / "final_adapter"
    trainer_output_dir = remote_run_dir / "trainer"

    rows = _load_rows("/data/sft_train.cleaned.jsonl")
    label_counts: dict[str, int] = {}
    for row in rows:
        label = row.get("metadata", {}).get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    print(
        json.dumps(
            {
                "event": "start_sft",
                "run_id": run_id,
                "base_model": base_model,
                "rows": len(rows),
                "label_counts": label_counts,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": per_device_batch_size,
                "grad_accum": grad_accum,
                "max_length": max_length,
                "lora_r": lora_r,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=_hf_token() or None, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    features = _build_train_features(rows, tokenizer, max_length=max_length)
    dataset = Dataset.from_list(features)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=_hf_token() or None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    wandb_url = ""
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "self-play"),
            name=run_id,
            config={
                "base_model": base_model,
                "sft_rows": len(rows),
                "label_counts": label_counts,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "per_device_batch_size": per_device_batch_size,
                "grad_accum": grad_accum,
                "max_length": max_length,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "data_file": "sft_train.cleaned.jsonl",
            },
        )
        wandb_url = wandb.run.url if wandb.run is not None else ""

    args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=True,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=run_id,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    train_result = trainer.train()
    trainer.save_state()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    shutil.copy2("/data/sft_train.cleaned.jsonl", remote_run_dir / "sft_train.cleaned.jsonl")
    shutil.copy2("/data/sft_train.cleaned_report.md", remote_run_dir / "sft_train.cleaned_report.md")
    summary = {
        "run_id": run_id,
        "base_model": base_model,
        "remote_run_dir": str(remote_run_dir),
        "adapter_dir": str(adapter_dir),
        "trainer_output_dir": str(trainer_output_dir),
        "wandb_url": wandb_url,
        "rows": len(rows),
        "label_counts": label_counts,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_batch_size": per_device_batch_size,
        "grad_accum": grad_accum,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_metrics": train_result.metrics,
    }
    (remote_run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    output_vol.commit()
    if wandb.run is not None:
        wandb.finish()
    print(json.dumps({"event": "done_sft", **summary}, ensure_ascii=False), flush=True)
    return summary


def _modal_volume_path(remote_path: str) -> str:
    if remote_path.startswith("/output/"):
        return remote_path[len("/output") :]
    if remote_path == "/output":
        return "/"
    return remote_path


def _download_run(remote_run_dir: str, local_output_dir: str = LOCAL_OUTPUT_DIR) -> None:
    volume_path = _modal_volume_path(remote_run_dir)
    os.makedirs(local_output_dir, exist_ok=True)
    cmd = ["modal", "volume", "get", "--force", OUTPUT_VOLUME_NAME, volume_path, local_output_dir]
    print("Downloading", volume_path, "to", local_output_dir, flush=True)
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(
    run_id: str = "",
    epochs: float = 2.0,
    learning_rate: float = 5e-5,
    per_device_batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 1024,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    download: bool = True,
) -> None:
    summary = train_attacker_sft.remote(
        run_id=run_id,
        epochs=epochs,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
        grad_accum=grad_accum,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if download:
        _download_run(summary["remote_run_dir"])
