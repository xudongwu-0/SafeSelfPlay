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
    FSP_GPU           -- Modal GPU spec (default: A100-40GB:4)
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
    # - Reassert a stable ROLL/role-LoRA stack before mounting local sources.
    #   An unconstrained rebuild resolved transformers 5.x and numpy 2.x;
    #   installing sentence-transformers later then rewrote transformers on
    #   disk after the training process had imported the 5.x modules.
    # - Preinstall replay-buffer dependencies so the existing runtime install
    #   is an idempotent no-op rather than a site-packages mutation.
    .run_commands(
        "pip install 'click==8.1.7' imageio "
        "'numpy==1.26.4' 'transformers==4.57.6' "
        "'accelerate==0.34.2' 'peft==0.12.0' 'trl==0.9.6' "
        "'sacrebleu==2.5.1' 'sentence-transformers==3.4.1' "
        "'cupy-cuda12x==13.6.0' 'opencv-python-headless==4.11.0.86' "
        "'typer==0.16.1'"
    )
    # vLLM 0.10.2 accesses all_special_tokens_extended. Preserve the native
    # Transformers descriptor; the compatibility script only adds a
    # non-recursive fallback for older backends that truly lack it.
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
    gpu=os.environ.get("FSP_GPU", "A100-40GB:4"),
    timeout=86400,  # 24 h ceiling for full 1k-step runs
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
    extra_raw = os.environ.get("FSP_EXTRA_OVERRIDES", "")
    # Prefix each override with ++ so Hydra allows keys not already in the YAML schema.
    extra_overrides = [f"++{o.strip()}" for o in extra_raw.split(",") if o.strip()] or None
    run_fsp_async_demo.remote(config_name=config_name, extra_overrides=extra_overrides)
