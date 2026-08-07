#!/usr/bin/env python3
"""Evaluate the ABS dual-LoRA defender with Self-RedTeam's released tasks."""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from modal_selfredteam_official_eval import (
    UPSTREAM_REMOTE,
    _extract_metrics,
    hf_cache,
    image,
)


SOURCE_CHECKPOINT = (
    "/output/abs_bipolicy_h200/"
    "abs_qwen25_3b_duallora_r32_simultaneous_s100_rb128_tb32_mb8_"
    "aLR1e-6_dLR3e-6_kl0p3_rolesft_ourprompts_20260803_194406_spawn/"
    "ckpt/global_step100_hf"
)
EVAL_ROOT = "/output/abs_bipolicy_official_eval/qwen25_3b_duallora_s100_defender"
TASKS = (
    "wildguardtest",
    "wildjailbreak:harmful",
    "do_anything_now",
    "harmbench_precompute",
)
TABLE_COLUMNS = (
    "WG adv ASR",
    "WG vanilla ASR",
    "WJB harmful ASR",
    "DAN ASR",
    "HarmBench adv ASR",
)
REFERENCE_ROWS = {
    "Paper Base (Llama-3.1-8B)": (0.478, 0.553, 0.991, 0.937, 0.654),
    "Paper Self-RedTeam+SFT (Llama-3.1-8B)": (0.138, 0.019, 0.240, 0.396, 0.221),
    "Measured Base (Llama-3.1-8B)": (0.424, 0.454, 0.938, 0.910, 0.575),
    "Reproduced Self-RedTeam step200 (Llama-3.1-8B)": (
        0.095,
        0.019,
        0.159,
        0.400,
        0.196,
    ),
}

output_vol = modal.Volume.from_name("roll-abs-benchmark-output", create_if_missing=True)
app = modal.App("abs-bipolicy-official-eval", image=image)


def _stage_adapter(source: Path, destination: Path) -> Path:
    """Give the nested role adapter the parent tokenizer expected by the fork."""
    role_source = source / "defender"
    if not (role_source / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Defender adapter not found: {role_source}")
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(role_source, destination)
    for item in source.iterdir():
        if item.is_file() and item.name not in {
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            shutil.copy2(item, destination / item.name)
    return destination


def _write_table(output_root: Path, metrics: dict[str, float | None]) -> Path:
    row = tuple(metrics[column] for column in TABLE_COLUMNS)
    rows = {**REFERENCE_ROWS, "Our dual-LoRA defender step100 (Qwen2.5-3B)": row}
    lines = [
        "# Self-RedTeam Released-Evaluator Comparison",
        "",
        "All columns are lower-is-better. The final row uses the same released tasks "
        "and scorers but a Qwen2.5-3B backbone with its native Qwen chat template; "
        "the Llama and Qwen rows are therefore not a same-backbone comparison.",
        "",
        "| Model | " + " | ".join(f"{column} ↓" for column in TABLE_COLUMNS) + " |",
        "|---|" + "---:|" * len(TABLE_COLUMNS),
    ]
    for name, values in rows.items():
        lines.append(
            "| " + name + " | " + " | ".join(
                "-" if value is None else f"{value:.3f}" for value in values
            ) + " |"
        )
    path = output_root / "comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@app.function(
    gpu="L4",
    cpu=24,
    memory=131072,
    timeout=86400,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def evaluate_defender(
    source_checkpoint: str = SOURCE_CHECKPOINT,
    output_slug: str = "qwen25_3b_duallora_s100_defender",
    model_label: str = "Our dual-LoRA defender step100 (Qwen2.5-3B)",
) -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("A Hugging Face token is required")
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fork_root = Path(UPSTREAM_REMOTE) / "eval/benchmarks/safety-eval-fork"
    os.environ["PYTHONPATH"] = ":".join(
        (str(fork_root), str(fork_root / "src"), os.environ.get("PYTHONPATH", ""))
    )
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    output_vol.reload()
    source = Path(source_checkpoint)
    wait_deadline = time.monotonic() + 12 * 60 * 60
    while not (source / "defender" / "adapter_model.safetensors").exists():
        if time.monotonic() >= wait_deadline:
            raise TimeoutError(f"Checkpoint did not appear: {source}")
        print(f"Waiting for checkpoint: {source}", flush=True)
        time.sleep(60)
        output_vol.reload()
    staged_adapter = _stage_adapter(source, Path("/tmp/abs_defender_adapter"))
    output_root = Path("/output/abs_bipolicy_official_eval") / output_slug
    output_root.mkdir(parents=True, exist_ok=True)
    eval_script = fork_root / "evaluation/eval.py"

    gpu_ids: queue.Queue[int] = queue.Queue()
    for gpu_id in range(1):
        gpu_ids.put(gpu_id)
    commit_lock = threading.Lock()

    def run_task(task: str) -> str:
        gpu_id = gpu_ids.get()
        try:
            report_path = output_root / f"metrics.{task}.json"
            individual_path = output_root / f"individual.{task}.json"
            log_path = output_root / f"eval.{task}.log"
            command = [
                sys.executable,
                str(eval_script),
                "generators",
                "--model_name_or_path",
                str(staged_adapter),
                "--model_input_template_path_or_name",
                "qwen",
                "--tasks",
                task,
                "--report_output_path",
                str(report_path),
                "--save_individual_results_path",
                str(individual_path),
                "--use_vllm=True",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["VLLM_USE_V1"] = "0"
            with log_path.open("w", encoding="utf-8", buffering=1) as handle:
                process = subprocess.Popen(
                    command,
                    cwd=fork_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(f"[{task}] {line}", end="", flush=True)
                    handle.write(line)
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Task {task} failed with status {return_code}")
            with commit_lock:
                output_vol.commit()
            return task
        finally:
            gpu_ids.put(gpu_id)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_task, task) for task in TASKS]
        for future in as_completed(futures):
            future.result()

    report: dict = {}
    for task in TASKS:
        report.update(json.loads((output_root / f"metrics.{task}.json").read_text()))
    metrics = _extract_metrics(report)
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    row = tuple(metrics[column] for column in TABLE_COLUMNS)
    rows = {**REFERENCE_ROWS, model_label: row}
    lines = [
        "# Self-RedTeam Released-Evaluator Comparison",
        "",
        "All columns are lower-is-better. The final row uses the same released "
        "tasks and scorers but a Qwen2.5-3B backbone with its native Qwen chat "
        "template; the Llama and Qwen rows are therefore not a same-backbone "
        "comparison.",
        "",
        "| Model | " + " | ".join(
            f"{column} ↓" for column in TABLE_COLUMNS
        ) + " |",
        "|---|" + "---:|" * len(TABLE_COLUMNS),
    ]
    for name, values in rows.items():
        lines.append(
            "| " + name + " | " + " | ".join(
                "-" if value is None else f"{value:.3f}" for value in values
            ) + " |"
        )
    comparison = output_root / "comparison.md"
    comparison.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_vol.commit()
    return {
        "comparison": str(comparison),
        "metrics": str(output_root / "metrics.json"),
        "source_checkpoint": source_checkpoint,
    }


@app.local_entrypoint()
def main(detach: bool = True) -> None:
    if detach:
        call = evaluate_defender.spawn()
        print(f"FUNCTION_CALL_ID={call.object_id}")
    else:
        print(json.dumps(evaluate_defender.remote(), indent=2))
