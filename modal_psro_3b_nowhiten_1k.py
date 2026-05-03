#!/usr/bin/env python3
"""
Modal deployment: ROLL PSRO Kuhn Poker 3B nowhiten 1k steps.

Setup:
    pip install modal && python -m modal setup
    modal secret create roll-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>

Run:
    modal run modal_psro_3b_nowhiten_1k.py
"""
import os
import modal

ROLL_LOCAL = os.path.dirname(os.path.abspath(__file__))

hf_cache = modal.Volume.from_name("roll-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("roll-psro-3b-nowhiten-output", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(["git", "gcc", "g++", "libgomp1", "libaio-dev"])
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("ray[default,cgraph]==2.48.0")
    .run_commands("CC=gcc pip install pycosat==0.6.6")
    .pip_install(
        "numpy>=1.25,<2.0a0",
        "tensordict",
        "sympy",
        "datasets==3.1.0",
        "tqdm",
        "peft==0.12.0",
        "tyro>=0.5.7",
        "accelerate==0.34.2",
        "pydantic",
        "pytest",
        "loralib",
        "einops",
        "isort",
        "jsonlines",
        "deprecated",
        "trl==0.9.6",
        "dacite",
        "codetiming",
        "more_itertools",
        "pybase64",
        "wandb",
        "swanlab",
        "openai",
        "langdetect",
        "nltk>=3.8",
        "gymnasium[toy-text]",
        "hydra-core",
        "omegaconf",
        "mcp",
        "antlr4-python3-runtime==4.9.3",
        "latex2sympy2==1.5.4",
        "latex2sympy2_extended==1.10.1",
        "gem-llm==0.0.4",
    )
    .pip_install("deepspeed==0.16.4")
    .pip_install("vllm==0.10.2")
    .run_commands("pip install wheel packaging && pip install flash-attn --no-build-isolation")
    .run_commands("pip install 'click==8.1.7' imageio 'accelerate>=1.1.0'")
    .add_local_file(
        os.path.join(ROLL_LOCAL, "_modal_patches/patch_vllm_tokenizer.py"),
        "/tmp/patch_vllm_tokenizer.py",
        copy=True,
    )
    .run_commands("python3 /tmp/patch_vllm_tokenizer.py")
    .add_local_dir(
        ROLL_LOCAL,
        "/roll",
        copy=False,
        ignore=[
            ".git",
            "__pycache__",
            "**/*.pyc",
            "**/*.egg-info",
            "logs/",
            "wandb/",
            "data/",
            "output/",
            "**/*.out",
            "**/*.err",
        ],
    )
)

app = modal.App("roll-psro-3b-nowhiten-1k", image=image)

CONFIG_NAME = "agent_kuhn_poker_psro_3b_nowhiten_1k"
MODAL_TAGS = "kuhn_poker,psro,qwen2_5_3b,cold_start,async,modal,ev_payoff,gs24,score0p15,nowhiten"


@app.function(
    gpu="A100-40GB:4",
    cpu=48,
    timeout=50400,  # 14 h ceiling for 1k-step run
    memory=131072,  # 128 GB RAM
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def run_full(max_steps: int = 1000, extra_overrides: list[str] | None = None) -> None:
    """Full 1 000-step PSRO nowhiten run on A100-40GB x4, 48 CPU cores."""
    import subprocess
    import sys

    for d in ["/tmp/triton_cache", "/tmp/ray_tmp", "/output/logs", "/output/render"]:
        os.makedirs(d, exist_ok=True)

    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray_tmp")
    os.environ["MODEL_DOWNLOAD_TYPE"] = "HUGGINGFACE_HUB"

    for pkg in ["/roll", "/roll/mcore_adapter"]:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", pkg, "--no-deps", "-q"], check=True)

    overrides = [
        "logging_dir=/output/logs",
        "output_dir=/output",
        "checkpoint_config.output_dir=/output/render",
        f"tracker_kwargs.tags=[{MODAL_TAGS}]",
    ]
    if max_steps != 1000:
        overrides.append(f"max_steps={max_steps}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [
        sys.executable,
        "examples/start_agentic_pipeline.py",
        "--config_path", "agentic_demo",
        "--config_name", CONFIG_NAME,
        *overrides,
    ]
    subprocess.run(cmd, cwd="/roll", check=True)


@app.local_entrypoint()
def main() -> None:
    print("Launching PSRO 3B nowhiten 1k on A100-40GB x4, 48 CPU cores…")
    run_full.remote()
