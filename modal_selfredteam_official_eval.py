#!/usr/bin/env python3
"""Evaluate the official Self-RedTeam reproduction against its base model.

This wrapper runs the public repository's ``safety-eval-fork`` directly.  It
mounts the full-parameter checkpoints produced by
``modal_selfredteam_official_h200.py`` and evaluates the trained and base
models with the same defender-side CoT template so their scores are directly
comparable.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal


UPSTREAM_LOCAL = Path(__file__).resolve().parent.parent / "selfplay-redteaming"
UPSTREAM_REMOTE = "/selfplay-redteaming"
UPSTREAM_COMMIT = "0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123"
BASE_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
TRAINED_CHECKPOINT = (
    "/official_output/selfredteam_official/"
    "selfredteam_official_repp_fullptx_sft_meta_llama_31_8b_instruct_abliterated_"
    "h200x4_s200_20260803_101623/ckpt/global_step200_hf"
)
EVAL_ROOT = "/official_output/selfredteam_official_eval/official_s200_vs_base"

# These are the tasks used by the public repository's
# evaluation/run_all_generation_benchmarks.py.  They cover every reproducible
# defender metric from Table 9 except StrongREJECT, whose task implementation
# is not included in the released fork.
TASKS = (
    "wildguardtest",
    "harmbench_precompute",
    "harmbench",
    "wildjailbreak:harmful",
    "do_anything_now",
    "or_bench:toxic",
    "xstest",
    "wildjailbreak:benign",
    "or_bench:hard-1k",
)

PAPER_TABLE9 = {
    "Base (paper)": {
        "WG adv ASR": 0.478,
        "WG vanilla ASR": 0.553,
        "WJB harmful ASR": 0.991,
        "DAN ASR": 0.937,
        "HarmBench adv ASR": 0.654,
        "HarmBench vanilla ASR": 0.747,
        "OR-Bench RTA": 0.014,
        "XSTest RTA": 0.290,
        "WJB benign comply": 0.992,
        "XSTest benign comply": 0.988,
    },
    "Self-RedTeam + SFT (paper)": {
        "WG adv ASR": 0.138,
        "WG vanilla ASR": 0.019,
        "WJB harmful ASR": 0.240,
        "DAN ASR": 0.396,
        "HarmBench adv ASR": 0.221,
        "HarmBench vanilla ASR": 0.048,
        "OR-Bench RTA": 0.846,
        "XSTest RTA": 0.814,
        "WJB benign comply": 0.806,
        "XSTest benign comply": 0.920,
    },
}

hf_cache = modal.Volume.from_name("roll-hf-cache", create_if_missing=True)
official_output = modal.Volume.from_name(
    "selfredteam-official-output", create_if_missing=True
)
roll_output = modal.Volume.from_name(
    "roll-abs-benchmark-output", create_if_missing=True
)

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.8.2")
    .entrypoint([])
    .apt_install("git")
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
    .pip_install(
        "datasets>=2.19.0",
        "fire>=0.6.0",
        "fschat>=0.2.36",
        "joblib>=1.4.0",
        "pandas>=2.2.2",
        "peft>=0.11.1",
        "scikit-learn>=1.4.2",
        "scipy>=1.13.0",
        "termcolor>=2.4.0",
        "tenacity>=8.4.1",
        "transformers==4.50.0",
    )
    .add_local_dir(
        str(UPSTREAM_LOCAL),
        UPSTREAM_REMOTE,
        copy=False,
        ignore=[".git", "__pycache__", "**/*.pyc", "**/*.egg-info", "logs/", "wandb/"],
    )
)

app = modal.App("selfredteam-official-eval", image=image)


def _metric(report: dict[str, Any], task: str, key: str) -> float | None:
    value = report.get(task, {}).get(key)
    return None if value is None else float(value)


def _extract_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    return {
        "WG adv ASR": _metric(report, "wildguardtest", "adversarial harm (lower)"),
        "WG vanilla ASR": _metric(report, "wildguardtest", "vanilla harm (lower)"),
        "WJB harmful ASR": _metric(report, "wildjailbreak:harmful", "macro ASR"),
        "DAN ASR": _metric(report, "do_anything_now", "macro ASR"),
        "HarmBench adv ASR": _metric(
            report, "harmbench_precompute", "micro ASR (lower)"
        ),
        "HarmBench vanilla ASR": _metric(report, "harmbench", "micro ASR (lower)"),
        "OR-Bench RTA": _metric(report, "or_bench:toxic", "refusal_rate_mean"),
        "XSTest RTA": _metric(report, "xstest", "rta_all_contrast"),
        "WJB benign comply": _metric(report, "wildjailbreak:benign", "macro ASR"),
        "XSTest benign comply": _metric(report, "xstest", "inverted_rta_all_safe"),
        "OR-Bench benign comply": _metric(
            report, "or_bench:hard-1k", "compliance_rate_mean"
        ),
    }


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _write_comparison(
    output_root: Path, reports: dict[str, dict[str, Any]]
) -> tuple[Path, Path]:
    measured = {name: _extract_metrics(report) for name, report in reports.items()}
    rows = {**PAPER_TABLE9, **measured}
    columns = list(next(iter(PAPER_TABLE9.values())).keys())
    columns.append("OR-Bench benign comply")

    payload = {
        "upstream_commit": UPSTREAM_COMMIT,
        "chat_template": "llama3_cot",
        "paper_table9": PAPER_TABLE9,
        "measured": measured,
        "raw_reports": {name: str(output_root / name / "metrics.json") for name in reports},
    }
    json_path = output_root / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        "# Official Self-RedTeam Step-200 Defender Evaluation",
        "",
        f"- Upstream commit: `{UPSTREAM_COMMIT}`",
        "- Evaluation code: released `eval/benchmarks/safety-eval-fork`",
        "- Both locally measured models use `llama3_cot` and deterministic decoding.",
        "- ASR columns are lower-is-better; RTA/compliance columns are higher-is-better.",
        "- StrongREJECT is omitted because its generation task is absent from the released fork.",
        "",
        "| Method | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name, metrics in rows.items():
        lines.append(
            "| " + name + " | " + " | ".join(_format(metrics.get(c)) for c in columns) + " |"
        )
    md_path = output_root / "comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


@app.function(
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=43200,
    scaledown_window=300,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=15.0),
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/official_output": official_output,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def evaluate_step200_and_base() -> dict[str, str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError("roll-secrets does not contain a Hugging Face token")
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONPATH"] = (
        "/selfplay-redteaming/eval/benchmarks/safety-eval-fork:"
        "/selfplay-redteaming/eval/benchmarks/safety-eval-fork/src:"
        + os.environ.get("PYTHONPATH", "")
    )
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    trained = Path(TRAINED_CHECKPOINT)
    if not (trained / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"Full HF checkpoint not found: {trained}")

    fork_root = Path("/selfplay-redteaming/eval/benchmarks/safety-eval-fork")
    eval_script = fork_root / "evaluation/eval.py"
    output_root = Path(EVAL_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    models = {
        "trained_step200": TRAINED_CHECKPOINT,
        "base_same_template": BASE_MODEL,
    }
    jobs = [
        # Put the highest-signal paired metrics first so partial progress is useful.
        (model_name, models[model_name], task)
        for task in TASKS
        for model_name in ("trained_step200", "base_same_template")
    ]
    gpu_ids: queue.Queue[int] = queue.Queue()
    for gpu_id in range(4):
        gpu_ids.put(gpu_id)
    commit_lock = threading.Lock()

    def run_job(model_name: str, model_path: str, task: str) -> tuple[str, str]:
        gpu_id = gpu_ids.get()
        try:
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            report_path = model_dir / f"metrics.{task}.json"
            individual_path = model_dir / f"individual.{task}.json"
            log_path = model_dir / f"eval.{task}.log"
            command = [
                sys.executable,
                str(eval_script),
                "generators",
                "--model_name_or_path",
                model_path,
                "--model_input_template_path_or_name",
                "llama3_cot",
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
            # vLLM 0.8.2 defaults to the V1 engine, which spends several minutes
            # compiling a separate graph in every benchmark subprocess.  V0 uses
            # the same model and decoding semantics without that startup cost.
            env["VLLM_USE_V1"] = "0"
            print(f"EVAL_START model={model_name} task={task} gpu={gpu_id}", flush=True)
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(fork_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(f"[{model_name}/{task}] {line}", end="", flush=True)
                    log_file.write(line)
                returncode = process.wait()
            if returncode:
                raise RuntimeError(
                    f"Evaluation failed: model={model_name}, task={task}, status={returncode}"
                )
            print(f"EVAL_DONE model={model_name} task={task}", flush=True)
            with commit_lock:
                official_output.commit()
            return model_name, task
        finally:
            gpu_ids.put(gpu_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_job, *job) for job in jobs]
        for future in as_completed(futures):
            future.result()

    reports: dict[str, dict[str, Any]] = {}
    for model_name in models:
        merged: dict[str, Any] = {}
        model_dir = output_root / model_name
        for task in TASKS:
            with (model_dir / f"metrics.{task}.json").open() as report_file:
                merged.update(json.load(report_file))
        (model_dir / "metrics.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False)
        )
        reports[model_name] = merged

    json_path, md_path = _write_comparison(output_root, reports)
    official_output.commit()
    return {
        "comparison_json": str(json_path),
        "comparison_md": str(md_path),
        "trained_checkpoint": TRAINED_CHECKPOINT,
        "base_model": BASE_MODEL,
    }


CORE_TASKS = (
    "wildguardtest",
    "wildjailbreak:harmful",
    "do_anything_now",
    "harmbench_precompute",
)


@app.function(
    gpu="A100-40GB:2",
    cpu=32,
    memory=131072,
    timeout=43200,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": roll_output,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def evaluate_full_checkpoint_vs_base(
    trained_checkpoint: str,
    output_slug: str,
    trained_label: str = "trained_defender",
    evaluate_base: bool = True,
) -> dict[str, str]:
    """Paired core defender evaluation for a full checkpoint or PEFT adapter."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("roll-secrets does not contain a Hugging Face token")
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fork_root = Path(UPSTREAM_REMOTE) / "eval/benchmarks/safety-eval-fork"
    os.environ["PYTHONPATH"] = ":".join(
        (str(fork_root), str(fork_root / "src"), os.environ.get("PYTHONPATH", ""))
    )
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    output_root = Path("/output/upstream_selfredteam_role_full_eval") / output_slug
    output_root.mkdir(parents=True, exist_ok=True)
    trained = Path(trained_checkpoint)
    full_checkpoint = trained
    full_model_files = (
        trained / "model.safetensors.index.json",
        trained / "model.safetensors",
        trained / "pytorch_model.bin.index.json",
        trained / "pytorch_model.bin",
    )
    if not any(path.exists() for path in full_model_files):
        adapter_files = (
            trained / "adapter_model.safetensors",
            trained / "adapter_model.bin",
        )
        if not (trained / "adapter_config.json").exists() or not any(
            path.exists() for path in adapter_files
        ):
            raise FileNotFoundError(
                f"Neither a full HF checkpoint nor a PEFT adapter was found: {trained}"
            )

        import gc

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        full_checkpoint = output_root / "merged_defender"
        if not any((full_checkpoint / name).exists() for name in (
            "model.safetensors.index.json",
            "model.safetensors",
        )):
            print(f"MERGE_START adapter={trained} output={full_checkpoint}", flush=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map={"": "cpu"},
            )
            peft_model = PeftModel.from_pretrained(
                base, str(trained), is_trainable=False
            )
            merged = peft_model.merge_and_unload(safe_merge=True)
            full_checkpoint.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(
                full_checkpoint, safe_serialization=True, max_shard_size="5GB"
            )
            AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(full_checkpoint)
            del merged, peft_model, base
            gc.collect()
            torch.cuda.empty_cache()
            roll_output.commit()
            print(f"MERGE_DONE output={full_checkpoint}", flush=True)

    eval_script = fork_root / "evaluation/eval.py"
    models = {trained_label: str(full_checkpoint)}
    if evaluate_base:
        models["base_same_template"] = BASE_MODEL

    gpu_ids: queue.Queue[int] = queue.Queue()
    for gpu_id in range(2):
        gpu_ids.put(gpu_id)
    commit_lock = threading.Lock()

    def run_job(model_name: str, model_path: str, task: str) -> tuple[str, str]:
        gpu_id = gpu_ids.get()
        try:
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            report_path = model_dir / f"metrics.{task}.json"
            individual_path = model_dir / f"individual.{task}.json"
            log_path = model_dir / f"eval.{task}.log"
            command = [
                sys.executable,
                str(eval_script),
                "generators",
                "--model_name_or_path", model_path,
                "--model_input_template_path_or_name", "llama3_cot",
                "--tasks", task,
                "--report_output_path", str(report_path),
                "--save_individual_results_path", str(individual_path),
                "--use_vllm=True",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["VLLM_USE_V1"] = "0"
            print(f"EVAL_START model={model_name} task={task} gpu={gpu_id}", flush=True)
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(fork_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(f"[{model_name}/{task}] {line}", end="", flush=True)
                    log_file.write(line)
                returncode = process.wait()
            if returncode:
                raise RuntimeError(
                    f"Evaluation failed: model={model_name}, task={task}, status={returncode}"
                )
            with commit_lock:
                roll_output.commit()
            return model_name, task
        finally:
            gpu_ids.put(gpu_id)

    jobs = [
        (model_name, model_path, task)
        for task in CORE_TASKS
        for model_name, model_path in models.items()
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_job, *job) for job in jobs]
        for future in as_completed(futures):
            future.result()

    reports: dict[str, dict[str, Any]] = {}
    for model_name in models:
        merged: dict[str, Any] = {}
        model_dir = output_root / model_name
        for task in CORE_TASKS:
            merged.update(json.loads((model_dir / f"metrics.{task}.json").read_text()))
        (model_dir / "metrics.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False)
        )
        reports[model_name] = merged

    measured = {name: _extract_metrics(report) for name, report in reports.items()}
    measured_base = {
        "WG adv ASR": 0.424,
        "WG vanilla ASR": 0.454,
        "WJB harmful ASR": 0.938,
        "DAN ASR": 0.910,
        "HarmBench adv ASR": 0.575,
    }
    columns = (
        "WG adv ASR",
        "WG vanilla ASR",
        "WJB harmful ASR",
        "DAN ASR",
        "HarmBench adv ASR",
    )
    comparison_json = output_root / "comparison.json"
    comparison_json.write_text(
        json.dumps(
            {
                "trained_checkpoint": trained_checkpoint,
                "evaluated_checkpoint": str(full_checkpoint),
                "chat_template": "llama3_cot",
                "base_source": (
                    "paired_current_run" if evaluate_base
                    else "previous_full_released_evaluator_run"
                ),
                "measured_base": (
                    measured.get("base_same_template", measured_base)
                ),
                "measured": measured,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    lines = [
        f"# Full Defender Evaluation: {trained_label}",
        "",
        "Released-evaluator run. All columns are lower-is-better.",
        "",
        "| Model | " + " | ".join(f"{column} ↓" for column in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    table_rows = [
        ("base_same_template", measured.get("base_same_template", measured_base)),
        (trained_label, measured[trained_label]),
    ]
    for model_name, metrics in table_rows:
        lines.append(
            "| " + model_name + " | "
            + " | ".join(_format(metrics.get(column)) for column in columns)
            + " |"
        )
    comparison_md = output_root / "comparison.md"
    comparison_md.write_text("\n".join(lines) + "\n")
    roll_output.commit()
    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "trained_checkpoint": trained_checkpoint,
        "evaluated_checkpoint": str(full_checkpoint),
        "base_model": BASE_MODEL,
    }


@app.local_entrypoint()
def main(detach: bool = True):
    if detach:
        call = evaluate_step200_and_base.spawn()
        print(f"FUNCTION_CALL_ID={call.object_id}")
    else:
        print(json.dumps(evaluate_step200_and_base.remote(), indent=2))
