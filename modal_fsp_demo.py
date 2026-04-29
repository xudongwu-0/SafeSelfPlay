#!/usr/bin/env python3
"""
Modal deployment: ROLL FSP + async Kuhn Poker smoke test.

Setup:
    pip install modal && python -m modal setup
    modal secret create roll-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>
    modal run modal_fsp_demo.py

Optional overrides via env:
    FSP_CONFIG_NAME   -- hydra config name (default: agent_kuhn_poker_fsp_async_smoke)
    FSP_MAX_STEPS     -- override max_steps
    FSP_GPU           -- Modal GPU spec (default: A10G:4)
"""
import os
import modal

ROLL_LOCAL = os.path.dirname(os.path.abspath(__file__))

hf_cache = modal.Volume.from_name("roll-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("roll-fsp-output", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: cuda 12.8 + Python 3.10 + all ROLL deps
# First build is slow (flash-attn ~30 min); subsequent runs use Modal's cache.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(["git", "gcc", "g++", "libgomp1", "libaio-dev"])
    # Torch first (large, separate layer so cache is preserved on dep changes)
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("ray[default,cgraph]==2.48.0")
    # pycosat (dep of gem-llm→reasoning-gym) has no prebuilt wheel; force gcc
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
    # flash-attn must come after torch; reinstall wheel first (vllm drops it)
    .run_commands("pip install wheel packaging && pip install flash-attn --no-build-isolation")
    # Fixes applied after vllm (to avoid busting flash-attn cache):
    # - click==8.1.7: vllm upgrades click to 8.2+ breaking ray CLI (deepcopy/Sentinel bug)
    # - imageio: imported by agentic utils, not pulled in transitively
    # - accelerate>=1.1.0: transformers requires this; we pinned 0.34.2 earlier
    # No transformers pin: allow vllm's preferred version (needs qwen2_5_omni from >=4.50)
    .run_commands("pip install 'click==8.1.7' imageio 'accelerate>=1.1.0'")
    # transformers>=4.50 loads Qwen2 with the fast tokenizer which lacks all_special_tokens_extended;
    # vllm 0.10.2 accesses that property — patch PreTrainedTokenizerFast in-place so all
    # Ray worker processes see the fix without needing a code change in vllm or ROLL.
    .add_local_file(
        os.path.join(ROLL_LOCAL, "_modal_patches/patch_vllm_tokenizer.py"),
        "/tmp/patch_vllm_tokenizer.py",
        copy=True,
    )
    .run_commands("python3 /tmp/patch_vllm_tokenizer.py")
    # Add ROLL source at container startup (copy=False = fast iteration, no image rebuild on code change)
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

app = modal.App("roll-fsp-demo", image=image)


@app.function(
    gpu=os.environ.get("FSP_GPU", "A10G:4"),
    timeout=7200,  # 2 h ceiling; smoke test finishes in ~20 min
    memory=65536,  # 64 GB RAM
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def run_fsp_async_demo(
    config_name: str = "agent_kuhn_poker_fsp_async_smoke",
    extra_overrides: list[str] | None = None,
) -> None:
    import subprocess
    import sys

    for d in ["/tmp/triton_cache", "/tmp/ray_tmp", "/output/logs", "/output/render"]:
        os.makedirs(d, exist_ok=True)

    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray_tmp")
    os.environ["MODEL_DOWNLOAD_TYPE"] = "HUGGINGFACE_HUB"  # modelscope not installed

    # Install local packages (no-deps: all pip deps already in image)
    for pkg in ["/roll", "/roll/mcore_adapter"]:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", pkg, "--no-deps", "-q"], check=True)

    overrides = [
        "logging_dir=/output/logs",
        "output_dir=/output",
        "checkpoint_config.output_dir=/output/render",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    max_steps = os.environ.get("FSP_MAX_STEPS")
    if max_steps:
        overrides.append(f"max_steps={max_steps}")

    cmd = [
        sys.executable,
        "examples/start_agentic_pipeline.py",
        "--config_path", "agentic_demo",
        "--config_name", config_name,
        *overrides,
    ]
    subprocess.run(cmd, cwd="/roll", check=True)


@app.local_entrypoint()
def main() -> None:
    config_name = os.environ.get("FSP_CONFIG_NAME", "agent_kuhn_poker_fsp_async_smoke")
    run_fsp_async_demo.remote(config_name=config_name)
