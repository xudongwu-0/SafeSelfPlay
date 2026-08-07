#!/usr/bin/env python3
"""Run the public Self-RedTeam training script on four Modal H200 GPUs.

This wrapper changes only deployment concerns:

- the official external WildGuard URL is supplied by an existing Modal service;
- checkpoints are written to a persistent Modal Volume;
- the public W&B entity/project are filled in;
- the public shell command is launched without shell tracing so secrets are not
  printed.

The model, full-parameter shared bipolicy, Re++/REINFORCE optimization, online
SFT, reward settings, batches, and stopping step match
``scripts/red_team_game_reinforce_8b.sh`` at upstream commit
``0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import modal


UPSTREAM_LOCAL = Path(__file__).resolve().parent.parent / "selfplay-redteaming"
UPSTREAM_REMOTE = "/selfplay-redteaming"
UPSTREAM_COMMIT = "0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123"

DEFAULT_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
DEFAULT_REWARD_FUNCTION_APP = "selfredteam-wildguard"
DEFAULT_REWARD_FUNCTION_NAME = "wildguard_reward_app"
WANDB_ENTITY = "2373025856w-the-university-of-hong-kong"
WANDB_PROJECT = "self-play"
WANDB_GROUP = "selfredteam-official-reproduction"

hf_cache = modal.Volume.from_name("roll-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("selfredteam-official-output", create_if_missing=True)

# The upstream repository pins torch 2.6, vLLM 0.8.2, Ray 2.44, and
# DeepSpeed 0.16.5. Starting from the matching vLLM image keeps flash-attn and
# the CUDA extension stack internally consistent.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.8.2")
    .entrypoint([])
    .apt_install("git", "libaio-dev")
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets",
        "deepspeed==0.16.5",
        "einops",
        "isort",
        "jsonlines",
        "loralib",
        "optimum",
        "optree>=0.13.0",
        "peft",
        "pynvml>=12.0.0",
        "ray[default]==2.44.0",
        "sacrebleu",
        "sentence_transformers",
        "tensorboard",
        "torchmetrics",
        "transformers==4.50.0",
        "transformers_stream_generator",
        "wandb",
    )
    .run_commands(
        "python -m pip install flash-attn==2.7.4.post1 --no-build-isolation"
    )
    .add_local_dir(
        str(UPSTREAM_LOCAL),
        UPSTREAM_REMOTE,
        copy=False,
        ignore=[".git", "__pycache__", "**/*.pyc", "**/*.egg-info", "logs/", "wandb/"],
    )
)

app = modal.App("selfredteam-official-h200", image=image)


def _run_checked(
    command: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode:
        # Do not include argv here: the upstream CLI accepts the W&B credential
        # as an argument and subprocess.CalledProcessError would expose it.
        raise RuntimeError(f"Child process exited with status {completed.returncode}")


def _run_logged(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    log_path: Path,
) -> None:
    """Mirror child output to Modal logs and a persistent diagnostic file."""
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"\n===== launch {datetime.now().isoformat()} =====\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        returncode = process.wait()
        log_file.write(f"===== child exit status {returncode} =====\n")
    if returncode:
        raise RuntimeError(f"Child process exited with status {returncode}")


@app.function(
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=43200,
    scaledown_window=300,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=15.0,
    ),
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_volume,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def run_official_selfredteam(
    run_name: str,
    wandb_run_id: str,
    remote_rm_url: str,
    model_path: str = DEFAULT_MODEL,
    stop_at_step: int = 200,
) -> dict[str, str | int]:
    if stop_at_step != 200:
        raise ValueError(
            "The public Self-RedTeam launcher only defines --stop_at_step_200; "
            "use 200 for an exact script run."
        )
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("roll-secrets does not contain WANDB_API_KEY")
    if not (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        raise RuntimeError("roll-secrets does not contain a Hugging Face token")

    output_dir = Path("/output/selfredteam_official") / run_name
    table_dir = output_dir / "run_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
            "RAY_TMPDIR": "/tmp/ray",
            "TRITON_CACHE_DIR": "/tmp/triton",
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TOKENIZERS_PARALLELISM": "true",
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_RESUME": "allow",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    Path("/tmp/ray").mkdir(parents=True, exist_ok=True)
    Path("/tmp/triton").mkdir(parents=True, exist_ok=True)

    _run_checked(
        [sys.executable, "-m", "pip", "install", "-e", UPSTREAM_REMOTE, "--no-deps", "-q"]
    )
    _run_checked(
        [
            sys.executable,
            "-c",
            (
                "import os, wandb; "
                "wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True, verify=True)"
            ),
        ]
    )
    _run_checked(["ray", "stop", "--force"])
    # Ray workers inherit the head process working directory. The public shell
    # script starts Ray from the repository and uses relative red_team/data
    # paths, so preserve that assumption inside Modal.
    _run_checked(
        ["ray", "start", "--head", "--disable-usage-stats"],
        cwd=UPSTREAM_REMOTE,
        env=env,
    )

    versions = {}
    for package in ("torch", "vllm", "ray", "deepspeed", "transformers"):
        module = __import__(package)
        versions[package] = getattr(module, "__version__", "unknown")
    metadata = {
        "run_name": run_name,
        "wandb_run_id": wandb_run_id,
        "model": model_path,
        "upstream_commit": UPSTREAM_COMMIT,
        "gpu": "H200:4",
        "stop_at_step": stop_at_step,
        "reward_url_host": remote_rm_url.split("/classify", 1)[0],
        "versions": versions,
    }
    (output_dir / "launch_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    output_volume.commit()
    print("SELFREDTEAM_LAUNCH=" + json.dumps(metadata, sort_keys=True), flush=True)

    custom_configs = '{"max_turns":2,"reward_type":"general_sum","remove_ties":true}'
    command = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--actor_num_nodes",
        "1",
        "--actor_num_gpus_per_node",
        "4",
        "--ref_num_nodes",
        "1",
        "--ref_num_gpus_per_node",
        "4",
        "--remote_rm_url",
        remote_rm_url,
        "--vllm_num_engines",
        "4",
        "--vllm_tensor_parallel_size",
        "1",
        "--colocate_all_models",
        "--vllm_gpu_memory_utilization",
        "0.7",
        "--pretrain",
        model_path,
        "--save_path",
        str(output_dir),
        "--ckpt_path",
        str(output_dir / "ckpt"),
        "--save_steps",
        "100",
        "--save_hf_ckpt",
        "--disable_ds_ckpt",
        "--micro_train_batch_size",
        "8",
        "--train_batch_size",
        "32",
        "--micro_rollout_batch_size",
        "8",
        "--rollout_batch_size",
        "128",
        "--prompt_data",
        "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl",
        "--prompt_data_probs",
        "0.5, 0.5",
        "--eval_data",
        "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl",
        "--sft_data",
        (
            "red_team/data/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl, "
            "red_team/data/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl"
        ),
        "--sft_data_probs",
        "0.5, 0.5",
        "--sft_input_key",
        "vanilla",
        "--sft_output_key",
        "completion",
        "--sft_steps",
        "1",
        "--sft_batches_per_step",
        "1",
        "--max_samples",
        "40000",
        "--max_epochs",
        "1",
        "--prompt_max_len",
        "2048",
        "--generate_max_len",
        "2048",
        "--flash_attn",
        "--zero_stage",
        "3",
        "--num_episodes",
        "1",
        "--bf16",
        "--seed",
        "8888",
        "--top_p",
        "1.0",
        "--actor_learning_rate",
        "5e-7",
        "--init_kl_coef",
        "0.01",
        "--normalize_reward",
        "--packing_samples",
        "--gradient_checkpointing",
        "--advantage_estimator",
        "reinforce",
        "--custom_configs",
        custom_configs,
        "--actor_loss_coef",
        "1.0",
        "--postfill_cot_loss_coef",
        "1.0",
        "--eval_steps",
        "10",
        "--eval_start_steps",
        "50",
        "--diversity_score_steps",
        "5",
        "--vllm_sync_backend",
        "nccl",
        "--enforce_eager",
        "--vllm_enable_sleep",
        "--deepspeed_enable_sleep",
        "--stop_at_step_200",
        "--use_wandb",
        "enabled",
        "--wandb_org",
        WANDB_ENTITY,
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_group",
        WANDB_GROUP,
        "--wandb_run_name",
        run_name,
        "--wandb_max_log",
        "10000",
        "--wandb_table_log_interval",
        "1",
        "--wandb_table_csv_path",
        str(table_dir),
    ]
    commit_stop = threading.Event()

    def commit_periodically() -> None:
        while not commit_stop.wait(30):
            try:
                output_volume.commit()
            except Exception as error:
                print(f"Periodic Volume commit failed: {error}", flush=True)

    commit_thread = threading.Thread(target=commit_periodically, daemon=True)
    commit_thread.start()
    try:
        _run_logged(
            command,
            cwd=UPSTREAM_REMOTE,
            env=env,
            log_path=output_dir / "training.log",
        )
    except BaseException as error:
        failure = {
            "time": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise
    finally:
        commit_stop.set()
        commit_thread.join(timeout=35)
        subprocess.run(["ray", "stop", "--force"], check=False)
        output_volume.commit()

    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "wandb_project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        "wandb_run_id": wandb_run_id,
        "stop_at_step": stop_at_step,
    }


def _warm_reward_endpoint(remote_rm_url: str) -> None:
    import requests

    payload = {
        "queries": [
            {
                "game_idx": 0,
                "prompt": "How can I stay safe while hiking?",
                "response": "Tell someone your route, carry water, and check the weather.",
            }
        ]
    }
    response = requests.post(remote_rm_url, json=payload, timeout=600)
    response.raise_for_status()


@app.local_entrypoint()
def main(
    stop_at_step: int = 200,
    model_path: str = DEFAULT_MODEL,
) -> None:
    local_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=UPSTREAM_LOCAL,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected upstream commit {UPSTREAM_COMMIT}, found {local_commit}"
        )

    reward_function = modal.Function.from_name(
        DEFAULT_REWARD_FUNCTION_APP,
        DEFAULT_REWARD_FUNCTION_NAME,
    )
    reward_url = reward_function.get_web_url()
    if not reward_url:
        raise RuntimeError("The deployed WildGuard reward function has no web URL")
    remote_rm_url = reward_url.rstrip("/") + "/classify"
    _warm_reward_endpoint(remote_rm_url)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_path.rsplit("/", 1)[-1].lower().replace(".", "").replace("-", "_")
    run_name = (
        f"selfredteam_official_repp_fullptx_sft_{model_slug}_"
        f"h200x4_s{stop_at_step}_{timestamp}"
    )
    wandb_run_id = uuid.uuid4().hex[:8]
    print(f"RUN_NAME={run_name}", flush=True)
    print(f"OUTPUT_VOLUME=selfredteam-official-output:{run_name}", flush=True)
    result = run_official_selfredteam.remote(
        run_name=run_name,
        wandb_run_id=wandb_run_id,
        remote_rm_url=remote_rm_url,
        model_path=model_path,
        stop_at_step=stop_at_step,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
