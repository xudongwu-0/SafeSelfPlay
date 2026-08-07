#!/usr/bin/env python3
"""Run upstream Self-RedTeam with two independent full-parameter policies.

The game, reward, optimizer, prompt mixture, and batch settings stay on the
upstream path.  A phase optimizes a full attacker against a frozen base
defender.  D phase optimizes a separately initialized full defender against
the frozen A-phase checkpoint.  The frozen role is served by a second vLLM
engine colocated on the same four H200 GPUs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_abs_benchmark import app, hf_cache, output_vol
from modal_upstream_selfredteam_role_lora import (
    LLAMA_ABLITERATED_MODEL,
    UPSTREAM_WORK,
    _replace_once,
    _stable_wildguard_rm_url,
    _prepare_role_lora_upstream,
)


OUTPUT_ROOT = "/output/upstream_selfredteam_role_full"
ATTACKER_SFT_DATA = "/aux_sft/attacker_rewrite_1180.jsonl"
DEFENDER_SFT_FILES = (
    "red_team/data/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl",
    "red_team/data/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl",
)
DEFAULT_VALID_A50_CHECKPOINT = (
    "/output/upstream_selfredteam_role_full/"
    "strict_dualfull_A50D50_strict_dualfull_matchedbudget_scopefix_20260806_1952/"
    "A_full_s50_vs_baseD/ckpt/global_step50_hf"
)


def _patch_full_opponent_vllm() -> None:
    """Colocate one immutable full-policy vLLM fleet with the trainable one."""
    cli_path = UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py"
    _replace_once(
        cli_path,
        '    parser.add_argument("--fixed_opponent_lora_path", type=str, default=None)\n',
        '    parser.add_argument("--fixed_opponent_lora_path", type=str, default=None)\n'
        '    parser.add_argument("--fixed_opponent_pretrain", type=str, default=None)\n'
        '    parser.add_argument(\n'
        '        "--fixed_opponent_vllm_gpu_memory_utilization",\n'
        '        type=float,\n'
        '        default=0.20,\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--monitor_reference_kl",\n'
        '        action="store_true",\n'
        '        help="Compute role-start KL without requiring a KL penalty.",\n'
        '    )\n',
        "full opponent CLI arguments",
    )

    _replace_once(
        cli_path,
        """        if args.init_kl_coef > 0:
""",
        """        if args.init_kl_coef > 0 or args.monitor_reference_kl:
""",
        "reference placement validation for KL monitoring",
    )
    _replace_once(
        cli_path,
        """    if args.init_kl_coef == 0:
        ref_model = None
    else:
""",
        """    if args.init_kl_coef == 0 and not args.monitor_reference_kl:
        ref_model = None
    else:
""",
        "keep reference model for unpenalized KL monitoring",
    )

    _replace_once(
        cli_path,
        """    defender_vllm_engines = None
    if (
        args.custom_configs.get("no_defender_turn", False)
        and not args.custom_configs.get("base_defender_from_actor_vllm", False)
    ):
""",
        """    defender_vllm_engines = None
    if args.fixed_opponent_pretrain:
        if pg is None:
            raise ValueError(
                "fixed_opponent_pretrain requires --colocate_all_models"
            )
        defender_vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.fixed_opponent_pretrain,
            args.seed + 10000,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            args.actor_num_nodes * args.actor_num_gpus_per_node
            // args.ring_attn_size,
            pg,
            args.fixed_opponent_vllm_gpu_memory_utilization,
            False,
            0,
            None,
        )
    elif (
        args.custom_configs.get("no_defender_turn", False)
        and not args.custom_configs.get("base_defender_from_actor_vllm", False)
    ):
""",
        "colocated immutable full opponent vLLM",
    )

    # A second colocated vLLM actor needs 0.2 of each placement-group bundle.
    # These values are Ray scheduling resources only; they do not partition GPU
    # memory or alter the DeepSpeed world size.
    _replace_once(
        cli_path,
        """        num_gpus_per_actor=0.4 if pg else 1,
        num_resources_per_node=args.actor_num_gpus_per_node,
""",
        """        num_gpus_per_actor=(
            0.3 if pg and args.fixed_opponent_pretrain else 0.4 if pg else 1
        ),
        num_resources_per_node=args.actor_num_gpus_per_node,
""",
        "actor Ray resource share for full opponent",
    )
    _replace_once(
        cli_path,
        """            num_gpus_per_actor=0.4 if pg else 1,
            num_resources_per_node=args.ref_num_gpus_per_node,
""",
        """            num_gpus_per_actor=(
                0.3 if pg and args.fixed_opponent_pretrain else 0.4 if pg else 1
            ),
            num_resources_per_node=args.ref_num_gpus_per_node,
""",
        "reference Ray resource share for full opponent",
    )
    _replace_once(
        cli_path,
        """        defender_vllm_engines=defender_vllm_engines if args.custom_configs.get("no_defender_turn", False) else None
""",
        """        defender_vllm_engines=defender_vllm_engines
""",
        "pass immutable full opponent engines",
    )

    experience_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/experience_maker.py"
    )
    _replace_once(
        experience_path,
        """        if custom_configs.get(
            "fixed_attacker_lora_from_actor_vllm", False
        ):
""",
        """        if custom_configs.get(
            "fixed_full_attacker_from_opponent_vllm", False
        ):
            if self.defender_vllm_engines is None:
                raise RuntimeError("Frozen full attacker vLLM is missing")
            def attacker_llm_generator(
                batch_chat_messages, all_labels, **gen_kwargs
            ):
                return self._generate_vllm(
                    self.defender_vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
        elif custom_configs.get(
            "fixed_attacker_lora_from_actor_vllm", False
        ):
""",
        "route frozen full attacker",
    )
    _replace_once(
        experience_path,
        """        if custom_configs.get(
            "fixed_attacker_lora_from_actor_vllm", False
        ):
            # The attacker uses the frozen opponent adapter, while the
""",
        """        if custom_configs.get(
            "fixed_full_defender_from_opponent_vllm", False
        ):
            if self.defender_vllm_engines is None:
                raise RuntimeError("Frozen full defender vLLM is missing")
            def defender_llm_generator(
                batch_chat_messages, all_labels, **gen_kwargs
            ):
                return self._generate_vllm(
                    self.defender_vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
        elif custom_configs.get(
            "fixed_full_attacker_from_opponent_vllm", False
        ):
            # The attacker uses the frozen full opponent, while the defender
            # must sample from the current trainable full policy.
            def defender_llm_generator(
                batch_chat_messages, all_labels, **gen_kwargs
            ):
                return self._generate_vllm(
                    self.vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
        elif custom_configs.get(
            "fixed_attacker_lora_from_actor_vllm", False
        ):
            # The attacker uses the frozen opponent adapter, while the
""",
        "route frozen full defender",
    )


def _patch_role_specific_online_sft() -> None:
    """Use rewrite SFT only for A and answer SFT only for D."""
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """            sft_strategy.args.apply_chat_template = True
            sft_strategy.args.prompt_input_template = DEFENDER_INSTRUCTION_COT_PROMPT
            
            sft_data = blending_datasets(
""",
        """            sft_strategy.args.apply_chat_template = True
            optimizer_train_role = args.custom_configs.get(
                "optimizer_train_role"
            )
            attacker_role_sft = optimizer_train_role == "attacker"
            if attacker_role_sft:
                sft_strategy.args.sft_input_key = "messages"
                sft_strategy.args.sft_output_key = None
                sft_strategy.args.prompt_input_template = None
            else:
                sft_strategy.args.prompt_input_template = (
                    DEFENDER_INSTRUCTION_COT_PROMPT
                )

            sft_data = blending_datasets(
""",
        "select role-specific online SFT schema",
    )
    _replace_once(
        actor_path,
        """                pretrain_mode=False,
                prompt_input_template=DEFENDER_INSTRUCTION_COT_PROMPT,
            )
""",
        """                pretrain_mode=False,
                multiturn=attacker_role_sft,
                prompt_input_template=(
                    None
                    if attacker_role_sft
                    else DEFENDER_INSTRUCTION_COT_PROMPT
                ),
            )
""",
        "construct role-specific online SFT dataset",
    )
    _replace_once(
        actor_path,
        """        sft_samples_this_step = 0 # Counter for SFT samples processed in this step on this rank
        if self.postfill_cot_loss:
            latest_postfill_cot_loss = None # Initialize here to ensure definition
""",
        """        sft_samples_this_step = 0 # Counter for SFT samples processed in this step on this rank
        postfill_cot_stop_after_step = self.args.custom_configs.get(
            "postfill_cot_stop_after_step"
        )
        effective_postfill_cot_loss_coef = float(
            self.args.postfill_cot_loss_coef
        )
        if (
            postfill_cot_stop_after_step is not None
            and global_steps > int(postfill_cot_stop_after_step)
        ):
            effective_postfill_cot_loss_coef = 0.0
        latest_postfill_cot_loss = None
        if self.postfill_cot_loss and effective_postfill_cot_loss_coef > 0:
""",
        "schedule role-specific online SFT coefficient",
    )
    actor_text = actor_path.read_text()
    sft_backward = (
        "self.strategy.backward(self.args.postfill_cot_loss_coef * "
        "postfill_cot_loss_val, self.actor, self.actor_optim)"
    )
    scheduled_sft_backward = (
        "self.strategy.backward(effective_postfill_cot_loss_coef * "
        "postfill_cot_loss_val, self.actor, self.actor_optim)"
    )
    if actor_text.count(sft_backward) != 2:
        raise RuntimeError(
            "Expected exactly two online-SFT backward calls, found "
            f"{actor_text.count(sft_backward)}"
        )
    actor_path.write_text(actor_text.replace(sft_backward, scheduled_sft_backward))
    _replace_once(
        actor_path,
        """        if self.postfill_cot_loss:
            # Log the SFT loss if it was computed in this step
""",
        """        if self.postfill_cot_loss:
            status["postfill_cot_loss_coef_effective"] = (
                effective_postfill_cot_loss_coef
            )
            # Log the SFT loss if it was computed in this step
""",
        "log effective online SFT coefficient",
    )


def _patch_full_parameter_vllm_broadcast_scope() -> None:
    """Keep the upstream module-level GPU helper visible to the full path.

    The LoRA synchronization compatibility patch imports the helper inside a
    conditional branch of ``_broadcast_to_vllm``.  Python then treats that
    name as local throughout the function, so the preceding full-parameter
    branch raises ``UnboundLocalError``.  Upstream already imports the helper
    at module scope; removing the redundant local import restores its original
    full-parameter behavior without changing either synchronization path.
    """
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """                        from torch.multiprocessing.reductions import reduce_tensor
                        from openrlhf.trainer.ray.utils import get_physical_gpu_id

                        weight = param.data.clone()
""",
        """                        from torch.multiprocessing.reductions import reduce_tensor

                        weight = param.data.clone()
""",
        "full-parameter vLLM broadcast helper scope",
    )


def _prepare_dual_full_upstream() -> None:
    # strict=True preserves the upstream prompts/reward/optimizer settings and
    # skips all experimental replay balancing or direct-chat behavior.
    _prepare_role_lora_upstream(
        attacker_prompt_profile="upstream",
        strict_upstream_alignment=True,
    )
    _patch_full_parameter_vllm_broadcast_scope()
    _patch_full_opponent_vllm()
    _patch_role_specific_online_sft()
    experience_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/experience_maker.py"
    ).read_text()
    required_defender_route = '''elif custom_configs.get(
            "fixed_full_attacker_from_opponent_vllm", False
        ):
            # The attacker uses the frozen full opponent, while the defender
            # must sample from the current trainable full policy.'''
    if required_defender_route not in experience_source:
        raise RuntimeError(
            "Patched full-policy D phase does not route defender generation "
            "to the trainable actor"
        )


def _install_upstream_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "sacrebleu==2.5.1",
            "sentence-transformers==3.4.1",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(UPSTREAM_WORK), "--no-deps", "-q"],
        check=True,
    )


def _full_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def _run_role(
    *,
    role: str,
    fixed_opponent: str,
    remote_rm_url: str,
    run_dir: Path,
    steps: int,
    learning_rate: float,
    init_kl_coef: float = 0.01,
    monitor_reference_kl: bool = False,
    postfill_cot_stop_after_step: int | None = None,
) -> Path:
    if role not in {"attacker", "defender"}:
        raise ValueError(role)
    if not fixed_opponent:
        raise ValueError("A frozen full opponent is required")

    subprocess.run(["ray", "stop", "--force"], check=False)
    _prepare_dual_full_upstream()
    _install_upstream_runtime()
    python_paths = [str(UPSTREAM_WORK)]
    if Path("/roll").is_dir():
        python_paths.append("/roll")
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    if inherited_pythonpath:
        python_paths.append(inherited_pythonpath)
    os.environ["PYTHONPATH"] = ":".join(python_paths)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    subprocess.run(
        [
            "ray", "start", "--head", "--num-gpus", "4", "--num-cpus", "8",
            "--disable-usage-stats",
        ],
        check=True,
    )

    prompt_data = ",".join(
        str(UPSTREAM_WORK / f"red_team/data/vanilla_{label}_dataset.jsonl")
        for label in ("harmful", "benign")
    )
    if role == "attacker":
        sft_data = ATTACKER_SFT_DATA
        sft_probs = "1.0"
        sft_keys: list[str] = []
    else:
        sft_data = ",".join(str(UPSTREAM_WORK / item) for item in DEFENDER_SFT_FILES)
        sft_probs = "0.5,0.5"
        sft_keys = ["--sft_input_key", "vanilla", "--sft_output_key", "completion"]

    custom_configs = {
        "max_turns": 2,
        "reward_type": "general_sum",
        "remove_ties": True,
        "optimizer_train_role": role,
        "fixed_full_defender_from_opponent_vllm": role == "attacker",
        "fixed_full_attacker_from_opponent_vllm": role == "defender",
        "actor_lr_scheduler": "cosine_with_min_lr",
    }
    if postfill_cot_stop_after_step is not None:
        custom_configs["postfill_cot_stop_after_step"] = int(
            postfill_cot_stop_after_step
        )
    run_name = f"{run_dir.parent.name}__{run_dir.name}"
    os.environ["WANDB_RUN_ID"] = hashlib.sha1(run_name.encode()).hexdigest()[:8]
    os.environ["WANDB_RESUME"] = "allow"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY is missing from roll-secrets")

    checkpoint_dir = run_dir / "ckpt"
    reference_kl_args = ["--monitor_reference_kl"] if monitor_reference_kl else []
    command = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--actor_num_nodes", "1",
        "--actor_num_gpus_per_node", "4",
        "--ref_num_nodes", "1",
        "--ref_num_gpus_per_node", "4",
        "--remote_rm_url", remote_rm_url,
        "--vllm_num_engines", "4",
        "--vllm_tensor_parallel_size", "1",
        "--colocate_all_models",
        "--vllm_gpu_memory_utilization", "0.35",
        # The frozen 8B engine needs about 35 GiB/model-GPU before allocating
        # KV blocks.  On H200, 0.30 leaves roughly 7 GiB for its KV cache while
        # still fitting beside the current-policy vLLM and ZeRO-3 actor/ref.
        "--fixed_opponent_vllm_gpu_memory_utilization", "0.30",
        "--pretrain", LLAMA_ABLITERATED_MODEL,
        "--fixed_opponent_pretrain", fixed_opponent,
        "--lora_rank", "0",
        "--save_path", str(run_dir),
        "--ckpt_path", str(checkpoint_dir),
        "--save_steps", str(steps),
        "--save_hf_ckpt",
        "--disable_ds_ckpt",
        "--micro_train_batch_size", "8",
        "--train_batch_size", "32",
        "--micro_rollout_batch_size", "8",
        "--rollout_batch_size", "128",
        "--prompt_data", prompt_data,
        "--prompt_data_probs", "0.5,0.5",
        "--sft_data", sft_data,
        "--sft_data_probs", sft_probs,
        *sft_keys,
        "--sft_steps", "1",
        "--sft_batches_per_step", "1",
        "--max_samples", str(128 * steps),
        "--max_epochs", "1",
        "--prompt_max_len", "2048",
        "--generate_max_len", "2048",
        "--flash_attn",
        "--zero_stage", "3",
        "--num_episodes", "1",
        "--bf16",
        "--seed", "8888",
        "--top_p", "1.0",
        "--temperature", "1.0",
        "--actor_learning_rate", str(learning_rate),
        "--init_kl_coef", str(init_kl_coef),
        *reference_kl_args,
        "--normalize_reward",
        "--packing_samples",
        "--gradient_checkpointing",
        "--advantage_estimator", "reinforce",
        "--custom_configs", json.dumps(custom_configs),
        "--actor_loss_coef", "1.0",
        "--postfill_cot_loss_coef", "1.0",
        "--eval_data",
        str(UPSTREAM_WORK / "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl"),
        "--eval_steps", "10",
        "--eval_start_steps", "50",
        "--diversity_score_steps", "5",
        "--vllm_sync_backend", "nccl",
        "--enforce_eager",
        "--vllm_enable_sleep",
        "--deepspeed_enable_sleep",
        "--use_wandb", wandb_key,
        "--wandb_org", "2373025856w-the-university-of-hong-kong",
        "--wandb_project", "self-play",
        "--wandb_group", "upstream-selfredteam-role-full",
        "--wandb_run_name", run_name,
        "--wandb_max_log", "10000",
        "--wandb_table_log_interval", "1",
        "--wandb_table_csv_path", str(run_dir / "run_tables"),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    redacted_command = list(command)
    wandb_key_index = redacted_command.index("--use_wandb") + 1
    redacted_command[wandb_key_index] = "<redacted>"
    (run_dir / "command.json").write_text(
        json.dumps(redacted_command, indent=2)
    )
    log_path = run_dir / "training.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=UPSTREAM_WORK,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    subprocess.run(["ray", "stop", "--force"], check=False)
    output_vol.commit()
    if return_code:
        raise RuntimeError(f"{role} phase exited with code {return_code}")
    checkpoint = checkpoint_dir / f"global_step{steps}_hf"
    if not _full_checkpoint(checkpoint):
        raise RuntimeError(f"Missing full {role} checkpoint: {checkpoint}")
    return checkpoint


@app.function(
    gpu=os.environ.get("UPSTREAM_ROLE_FULL_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_strict_upstream_aligned_dual_full_round(
    steps_per_role: int = 50,
    run_suffix: str = "",
) -> dict[str, object]:
    """Train full A then full D while keeping upstream sample hyperparameters."""
    if steps_per_role < 1:
        raise ValueError("steps_per_role must be positive")
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / f"strict_dualfull_A{steps_per_role}D{steps_per_role}_{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    rm_url = _stable_wildguard_rm_url()
    manifest: dict[str, object] = {
        "method": "upstream Self-RedTeam, two independent full 8B policies",
        "base_model": LLAMA_ABLITERATED_MODEL,
        "attacker_start": LLAMA_ABLITERATED_MODEL,
        "defender_start": LLAMA_ABLITERATED_MODEL,
        "steps_per_role": steps_per_role,
        "rollout_batch_size": 128,
        "micro_rollout_batch_size": 8,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "prompt_budget_per_role": 128 * steps_per_role,
        "total_prompt_budget": 2 * 128 * steps_per_role,
        "attacker_sft": ATTACKER_SFT_DATA,
        "attacker_sft_unique_rows": 1180,
        "defender_sft": list(DEFENDER_SFT_FILES),
        "defender_sft_unique_rows": 29996,
        "sft_steps_per_rl_step": 1,
        "sft_batches_per_step": 1,
        "status": "attacker",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    attacker_checkpoint = _run_role(
        role="attacker",
        fixed_opponent=LLAMA_ABLITERATED_MODEL,
        remote_rm_url=rm_url,
        run_dir=root / f"A_full_s{steps_per_role}_vs_baseD",
        steps=steps_per_role,
        learning_rate=5e-7,
    )
    manifest.update(
        status="defender",
        attacker_checkpoint=str(attacker_checkpoint),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    defender_checkpoint = _run_role(
        role="defender",
        fixed_opponent=str(attacker_checkpoint),
        remote_rm_url=rm_url,
        run_dir=root / f"D_full_s{steps_per_role}_vs_A{steps_per_role}",
        steps=steps_per_role,
        learning_rate=5e-7,
    )
    manifest.update(
        status="completed",
        defender_checkpoint=str(defender_checkpoint),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()
    return manifest


@app.function(
    gpu=os.environ.get("UPSTREAM_ROLE_FULL_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_strict_upstream_aligned_full_defender(
    attacker_checkpoint: str = DEFAULT_VALID_A50_CHECKPOINT,
    steps: int = 50,
    run_suffix: str = "",
) -> dict[str, object]:
    """Rerun only D against an existing valid full-parameter attacker."""
    if steps < 1:
        raise ValueError("steps must be positive")
    output_vol.reload()
    attacker_path = Path(attacker_checkpoint)
    if not _full_checkpoint(attacker_path):
        raise FileNotFoundError(
            f"Missing full attacker checkpoint: {attacker_checkpoint}"
        )
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / f"strict_dualfull_D{steps}_routefix_{suffix}"
    run_dir = root / (
        f"D_full_s{steps}_vs_A50_klpen0_lr1e6_refkl_sft1to10_then0"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "method": "upstream Self-RedTeam defender-only full-policy rerun",
        "routing": {
            "attacker": "frozen full A50 checkpoint",
            "defender": "current trainable full base policy",
        },
        "attacker_checkpoint": attacker_checkpoint,
        "defender_start": LLAMA_ABLITERATED_MODEL,
        "steps": steps,
        "rollout_batch_size": 128,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "actor_learning_rate": 1e-6,
        "init_kl_coef": 0.0,
        "reference_kl_monitoring": True,
        "reference_kl_baseline": LLAMA_ABLITERATED_MODEL,
        "postfill_cot_loss_schedule": {
            "steps_1_to_10": 1.0,
            "steps_11_onward": 0.0,
        },
        "current_vllm_gpu_memory_utilization": 0.35,
        "fixed_opponent_vllm_gpu_memory_utilization": 0.30,
        "status": "defender",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    defender_checkpoint = _run_role(
        role="defender",
        fixed_opponent=attacker_checkpoint,
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=run_dir,
        steps=steps,
        learning_rate=1e-6,
        init_kl_coef=0.0,
        monitor_reference_kl=True,
        postfill_cot_stop_after_step=10,
    )
    manifest.update(
        status="completed",
        defender_checkpoint=str(defender_checkpoint),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()
    return manifest


@app.function(
    gpu=os.environ.get("UPSTREAM_ROLE_FULL_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_dynamic_sft_dual_full_round(
    steps_per_role: int = 200,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    learning_rate: float = 1e-6,
    run_suffix: str = "",
) -> dict[str, object]:
    """Train A then D with unpenalized reference KL and early SFT only."""
    if steps_per_role < 1:
        raise ValueError("steps_per_role must be positive")
    if attacker_sft_stop_after_step < 0:
        raise ValueError("attacker_sft_stop_after_step must be non-negative")
    if defender_sft_stop_after_step < 0:
        raise ValueError("defender_sft_stop_after_step must be non-negative")

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"dynamic_sft_A{steps_per_role}D{steps_per_role}_lr{learning_rate:g}_"
        f"klpen0_A_sft{int(attacker_sft_stop_after_step)}to0_"
        f"D_sft{int(defender_sft_stop_after_step)}to0_{suffix}"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest: dict[str, object] = {
        "method": "two independent full policies; sequential attacker then defender",
        "base_model": LLAMA_ABLITERATED_MODEL,
        "attacker_start": LLAMA_ABLITERATED_MODEL,
        "defender_start": LLAMA_ABLITERATED_MODEL,
        "steps_per_role": steps_per_role,
        "rollout_batch_size": 128,
        "micro_rollout_batch_size": 8,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "actor_learning_rate": learning_rate,
        "init_kl_coef": 0.0,
        "reference_kl_monitoring": True,
        "postfill_cot_loss_schedule": {
            "attacker": {
                f"steps_1_to_{attacker_sft_stop_after_step}": 1.0,
                f"steps_{attacker_sft_stop_after_step + 1}_onward": 0.0,
            },
            "defender": {
                f"steps_1_to_{defender_sft_stop_after_step}": 1.0,
                f"steps_{defender_sft_stop_after_step + 1}_onward": 0.0,
            },
        },
        "attacker_sft": ATTACKER_SFT_DATA,
        "defender_sft": list(DEFENDER_SFT_FILES),
        "status": "attacker",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    attacker_checkpoint = _run_role(
        role="attacker",
        fixed_opponent=LLAMA_ABLITERATED_MODEL,
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=root / (
            f"A1_full_s{steps_per_role}_vs_baseD_lr{learning_rate:g}_"
            f"klpen0_refkl_sft{attacker_sft_stop_after_step}to0"
        ),
        steps=steps_per_role,
        learning_rate=learning_rate,
        init_kl_coef=0.0,
        monitor_reference_kl=True,
        postfill_cot_stop_after_step=attacker_sft_stop_after_step,
    )
    manifest.update(status="defender", attacker_checkpoint=str(attacker_checkpoint))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    defender_checkpoint = _run_role(
        role="defender",
        fixed_opponent=str(attacker_checkpoint),
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=root / (
            f"D1_full_s{steps_per_role}_vs_A1_s{steps_per_role}_lr{learning_rate:g}_"
            f"klpen0_refkl_sft{defender_sft_stop_after_step}to0"
        ),
        steps=steps_per_role,
        learning_rate=learning_rate,
        init_kl_coef=0.0,
        monitor_reference_kl=True,
        postfill_cot_stop_after_step=defender_sft_stop_after_step,
    )
    manifest.update(status="completed", defender_checkpoint=str(defender_checkpoint))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()
    return manifest


@app.function(cpu=2, timeout=1800, memory=8192)
def validate_strict_dual_full_configuration() -> dict[str, object]:
    """Patch a fresh upstream tree and validate schemas without allocating GPUs."""
    _prepare_dual_full_upstream()
    _install_upstream_runtime()
    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--num-cpus",
            "2",
            "--disable-usage-stats",
        ],
        check=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UPSTREAM_WORK)
    help_result = subprocess.run(
        [sys.executable, "-m", "openrlhf.cli.train_ppo_ray", "--help"],
        cwd=UPSTREAM_WORK,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if help_result.returncode:
        raise RuntimeError(
            "Patched upstream CLI failed to import:\n"
            + help_result.stdout
            + help_result.stderr
        )
    required_flags = (
        "--fixed_opponent_pretrain",
        "--fixed_opponent_vllm_gpu_memory_utilization",
        "--monitor_reference_kl",
        "--sft_data",
        "--lora_rank",
    )
    missing = [flag for flag in required_flags if flag not in help_result.stdout]
    if missing:
        raise RuntimeError(f"Patched parser is missing flags: {missing}")

    attacker_path = Path(ATTACKER_SFT_DATA)
    if not attacker_path.is_file():
        raise FileNotFoundError(attacker_path)
    attacker_rows = 0
    with attacker_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("Attacker SFT row lacks messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError("Attacker SFT row does not end in an assistant rewrite")
            attacker_rows += 1

    defender_rows = 0
    nonempty_adversarial = 0
    for relative_path in DEFENDER_SFT_FILES:
        with (UPSTREAM_WORK / relative_path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("vanilla") or not row.get("completion"):
                    raise ValueError(f"Malformed defender SFT row in {relative_path}")
                nonempty_adversarial += bool(row.get("adversarial"))
                defender_rows += 1
    if nonempty_adversarial:
        raise ValueError(
            "Official answer SFT unexpectedly contains attacker rewrites; "
            "re-audit the role split"
        )
    return {
        "status": "validated",
        "base_model": LLAMA_ABLITERATED_MODEL,
        "attacker_sft_rows": attacker_rows,
        "defender_sft_rows": defender_rows,
        "official_sft_nonempty_adversarial_rows": nonempty_adversarial,
        "fixed_opponent_flags": list(required_flags[:2]),
        "lora_rank": 0,
    }


@app.local_entrypoint(name="strict_upstream_aligned_dual_full_round")
def strict_upstream_aligned_dual_full_round(
    steps_per_role: int = 50,
    run_suffix: str = "",
    detach: bool = True,
) -> None:
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    if detach:
        call = train_strict_upstream_aligned_dual_full_round.spawn(
            steps_per_role=steps_per_role,
            run_suffix=suffix,
        )
        print(f"CALL_ID={call.object_id}")
        print(f"RUN_SUFFIX={suffix}")
        return
    result = train_strict_upstream_aligned_dual_full_round.remote(
        steps_per_role=steps_per_role,
        run_suffix=suffix,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint(name="strict_upstream_aligned_full_defender")
def strict_upstream_aligned_full_defender(
    attacker_checkpoint: str = DEFAULT_VALID_A50_CHECKPOINT,
    steps: int = 50,
    run_suffix: str = "",
    detach: bool = True,
) -> None:
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    if detach:
        call = train_strict_upstream_aligned_full_defender.spawn(
            attacker_checkpoint=attacker_checkpoint,
            steps=steps,
            run_suffix=suffix,
        )
        print(f"CALL_ID={call.object_id}")
        print(f"RUN_SUFFIX={suffix}")
        return
    result = train_strict_upstream_aligned_full_defender.remote(
        attacker_checkpoint=attacker_checkpoint,
        steps=steps,
        run_suffix=suffix,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint(name="validate_strict_dual_full_configuration")
def validate_strict_dual_full_configuration_entrypoint() -> None:
    result = validate_strict_dual_full_configuration.remote()
    print(json.dumps(result, indent=2))
