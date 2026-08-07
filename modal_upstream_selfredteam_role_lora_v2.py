#!/usr/bin/env python3
"""Train a role-specific LLaMA-Factory LoRA in Self-RedTeam.

This is intentionally independent from ``modal_upstream_selfredteam_role_lora``.
LLaMA-Factory owns adapter construction, trainable parameter selection, dtype
handling under ZeRO-3, and PEFT checkpointing. Self-RedTeam continues to own
the online RL loop, reward, prompts, batches, and optimizer updates.
For rollout synchronization, each LoRA layer is merged into its frozen base
weight and sent through the same dense vLLM weight-update path used by the
working full-parameter experiment.  No dynamic vLLM LoRA worker is involved.

The existing full-parameter entrypoint is not imported or modified at runtime.
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

from modal_upstream_selfredteam_fixed_seed import (
    UPSTREAM_WORK,
    _patch_upstream_attacker_only_sampling,
    _patch_upstream_release_rl_logits_before_sft,
    _patch_upstream_replay_buffer_diagnostics,
    _patch_upstream_sft_chat_template,
    _patch_upstream_sft_micro_batch_floor,
    _patch_upstream_vllm_version_check,
    _patch_upstream_zero3_sync_active_params,
    _prepare_upstream_source,
)
from modal_upstream_selfredteam_role_lora import (
    _patch_upstream_comprehensive_wandb_logging,
    _patch_upstream_defender_metric_keys,
    _patch_upstream_remote_rm_retry,
    _patch_upstream_role_lr_scheduler,
    _stable_wildguard_rm_url,
)
from modal_upstream_selfredteam_role_full import (
    _install_upstream_runtime,
    _patch_role_specific_online_sft,
)


BASE_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
OUTPUT_ROOT = "/output/upstream_selfredteam_role_lora_v2"
ATTACKER_SFT_DATA = "/aux_sft/attacker_rewrite_1180.jsonl"
SELFPLAY_LOCAL = Path(__file__).resolve().parent.parent / "selfplay-redteaming"
ATTACKER_SFT_LOCAL = (
    Path(__file__).resolve().parent.parent
    / "checkpoints"
    / "abs_attacker_sft_runs"
    / "abs_attacker_sft_qwen25_3b_lora_r32_1180_20260715_010540_retry1"
    / "sft_train.cleaned.jsonl"
)
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Match the official Self-RedTeam H200 dependency stack, while mounting only
# the source directories required for training. This avoids uploading ROLL's
# virtualenv/docs/tests and Self-RedTeam's large evaluation submodules.
llamafactory_lora_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.8.2")
    .entrypoint([])
    .apt_install("git", "libaio-dev")
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets==3.6.0",
        "deepspeed==0.16.5",
        "einops",
        "isort",
        "jsonlines",
        "loralib",
        "optimum",
        "optree>=0.13.0",
        "omegaconf",
        "peft==0.15.2",
        "pynvml>=12.0.0",
        "ray[default]==2.44.0",
        "sacrebleu",
        "sentence_transformers",
        "tensorboard",
        "torchmetrics",
        "transformers==4.50.0",
        "transformers_stream_generator",
        "trl==0.9.6",
        "wandb",
    )
    .run_commands(
        "python -m pip install flash-attn==2.7.4.post1 --no-build-isolation"
    )
)
for root_file in ("README.md", "setup.py", "pyproject.toml", "version.txt", "requirements.txt"):
    llamafactory_lora_image = llamafactory_lora_image.add_local_file(
        str(SELFPLAY_LOCAL / root_file),
        f"/selfplay-redteaming/{root_file}",
        copy=False,
    )
for source_dir in ("openrlhf", "red_team", "wildguard"):
    llamafactory_lora_image = llamafactory_lora_image.add_local_dir(
        str(SELFPLAY_LOCAL / source_dir),
        f"/selfplay-redteaming/{source_dir}",
        copy=False,
        ignore=["__pycache__", "**/*.pyc", "**/*.egg-info", "logs/", "wandb/"],
    )
for helper_file in (
    "modal_fsp_demo.py",
    "modal_abs_benchmark.py",
    "modal_upstream_selfredteam_fixed_seed.py",
    "modal_upstream_selfredteam_role_lora.py",
    "modal_upstream_selfredteam_role_full.py",
):
    llamafactory_lora_image = llamafactory_lora_image.add_local_file(
        str(Path(__file__).resolve().parent / helper_file),
        f"/roll/{helper_file}",
        copy=False,
    )
llamafactory_lora_image = llamafactory_lora_image.add_local_file(
    str(Path(__file__).resolve().parent / "_modal_patches/patch_vllm_tokenizer.py"),
    "/roll/_modal_patches/patch_vllm_tokenizer.py",
    copy=False,
)
llamafactory_lora_image = llamafactory_lora_image.add_local_file(
    str(ATTACKER_SFT_LOCAL),
    ATTACKER_SFT_DATA,
    copy=False,
)
hf_cache = modal.Volume.from_name("roll-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("roll-abs-benchmark-output", create_if_missing=True)
lora_v2_app = modal.App(
    "selfredteam-llamafactory-lora",
    image=llamafactory_lora_image,
)


def _replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    if old not in source:
        raise RuntimeError(f"Expected upstream block was not found: {label}")
    path.write_text(source.replace(old, new, 1))


def _patch_wandb_run_identity() -> None:
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )
""",
        """            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                id=os.environ.get("WANDB_RUN_ID"),
                resume=os.environ.get("WANDB_RESUME", "allow"),
                config=self.strategy.args.__dict__,
                reinit=True,
            )
""",
        "stable W&B identity",
    )


def _patch_linear_cot_format_parser() -> None:
    """Avoid pathological regex backtracking on malformed long generations."""
    utils_path = UPSTREAM_WORK / "red_team/utils.py"
    source = utils_path.read_text()
    start = source.index("def cot_format_check_and_extract(response: str)")
    end = source.index("\ndef get_redteaming_game_reward_zero_sum", start)
    replacement = '''def cot_format_check_and_extract(response: str) -> Tuple[str, bool]:
    """Extract Self-RedTeam tags in linear time.

    This preserves the upstream parser's return values and validation rules.
    The upstream regex has overlapping ``\\s*``/``.*?`` quantifiers and can
    backtrack for minutes when a long generation omits ``</think>``.
    """
    think_close = "</think>"
    answer_open = "<answer>"
    answer_close = "</answer>"

    think_end_pos = response.find(think_close)
    answer_start_pos = response.find(answer_open)
    answer_end_pos = (
        response.find(answer_close, answer_start_pos + len(answer_open))
        if answer_start_pos >= 0
        else -1
    )
    answer = (
        response[answer_start_pos + len(answer_open):answer_end_pos].strip()
        if answer_start_pos >= 0 and answer_end_pos >= 0
        else None
    )

    # Match the upstream fallback: retain a parseable answer, otherwise the
    # original response, while marking malformed output as invalid.
    if think_end_pos < 0 or answer is None:
        return (None, answer if answer is not None else response), True

    thinking = response[:think_end_pos].strip()
    if "<think>" in response:
        return (None, answer), True
    if not thinking or not answer:
        return (None, answer), True
    if (
        response.count(think_close) != 1
        or response.count(answer_open) != 1
        or response.count(answer_close) != 1
    ):
        return (None, answer), True
    if not (0 < think_end_pos < answer_start_pos < answer_end_pos):
        return (None, answer), True
    if response[think_end_pos + len(think_close):answer_start_pos].strip():
        return (None, answer), True
    if not response.strip().endswith(answer_close):
        return (None, answer), True
    if response[answer_end_pos + len(answer_close):].strip():
        return (thinking, answer), True
    return (thinking, answer), False

'''
    utils_path.write_text(source[:start] + replacement + source[end + 1:])


def _patch_reference_kl_monitoring() -> None:
    """Keep the base reference for metrics while its loss coefficient is zero."""
    cli_path = UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py"
    _replace_once(
        cli_path,
        '    parser.add_argument("--lora_dropout", type=float, default=0)\n',
        '    parser.add_argument("--lora_dropout", type=float, default=0)\n'
        '    parser.add_argument(\n'
        '        "--monitor_reference_kl",\n'
        '        action="store_true",\n'
        '        help="Compute reference KL without adding it to the loss.",\n'
        '    )\n',
        "reference KL monitoring CLI argument",
    )
    _replace_once(
        cli_path,
        """        if args.init_kl_coef > 0:
""",
        """        if args.init_kl_coef > 0 or args.monitor_reference_kl:
""",
        "reference placement validation for monitoring",
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
        "retain unpenalized reference model",
    )


def _patch_colocated_base_defender() -> None:
    """Place the immutable base defender beside actor/ref/current vLLM."""
    cli_path = UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py"
    _replace_once(
        cli_path,
        """    if args.custom_configs.get("no_defender_turn", False):
        # create a new vllm engine for defender, because we are red-teaming against a fixed defender
        defender_vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
            None, # non-colocated will occupy 1 GPU per engine, so it will get push to another node
            gpu_memory_utilization=0.95,
        )
        batch_vllm_engine_call(defender_vllm_engines, "wake_up", rank_0_only=False)
""",
        """    defender_vllm_engines = None
    if args.custom_configs.get("no_defender_turn", False):
        # Keep the base defender immutable, but colocate it on the same H200
        # node. Ray resource fractions are scheduling declarations only.
        if pg is None:
            raise ValueError("LoRA v2 base defender requires --colocate_all_models")
        defender_vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed + 10000,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
            pg,
            gpu_memory_utilization=args.fixed_defender_vllm_gpu_memory_utilization,
            vllm_enable_sleep=False,
        )
""",
        "colocated immutable base defender",
    )
    _replace_once(
        cli_path,
        """        num_gpus_per_actor=0.4 if pg else 1,
        num_resources_per_node=args.actor_num_gpus_per_node,
""",
        """        num_gpus_per_actor=(
            0.3
            if pg and args.custom_configs.get("no_defender_turn", False)
            else 0.4 if pg else 1
        ),
        num_resources_per_node=args.actor_num_gpus_per_node,
""",
        "actor Ray share with base defender",
    )
    _replace_once(
        cli_path,
        """            num_gpus_per_actor=0.4 if pg else 1,
            num_resources_per_node=args.ref_num_gpus_per_node,
""",
        """            num_gpus_per_actor=(
                0.3
                if pg and args.custom_configs.get("no_defender_turn", False)
                else 0.4 if pg else 1
            ),
            num_resources_per_node=args.ref_num_gpus_per_node,
""",
        "reference Ray share with base defender",
    )
    _replace_once(
        cli_path,
        """    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
""",
        """    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--fixed_defender_vllm_gpu_memory_utilization",
        type=float,
        default=0.30,
        help="GPU memory fraction for the immutable base defender",
    )
""",
        "base defender memory CLI argument",
    )


def _patch_single_role_advantage_normalization() -> None:
    """Avoid the upstream attacker-only double-normalization branch."""
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """                if self.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
                    if not self.args.custom_configs.get('no_attacker_turn', False):
                        self.replay_buffer.normalize(strategy=self.strategy, attribute="advantages", role="attacker")
                    if not self.args.custom_configs.get('no_defender_turn', False):
                        self.replay_buffer.normalize(strategy=self.strategy, attribute="advantages", role="defender")
                    else:
                        self.replay_buffer.normalize(strategy=self.strategy, attribute="advantages", divide_by_std=not self.args.no_advantage_std_norm)
""",
        """                if self.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
                    if self.args.custom_configs.get('no_defender_turn', False):
                        self.replay_buffer.normalize(
                            strategy=self.strategy,
                            attribute="advantages",
                            role="attacker",
                            divide_by_std=not self.args.no_advantage_std_norm,
                        )
                    elif self.args.custom_configs.get('no_attacker_turn', False):
                        self.replay_buffer.normalize(
                            strategy=self.strategy,
                            attribute="advantages",
                            role="defender",
                            divide_by_std=not self.args.no_advantage_std_norm,
                        )
                    else:
                        self.replay_buffer.normalize(
                            strategy=self.strategy,
                            attribute="advantages",
                            role="attacker",
                        )
                        self.replay_buffer.normalize(
                            strategy=self.strategy,
                            attribute="advantages",
                            role="defender",
                        )
""",
        "single role advantage normalization",
    )


def _patch_dense_merged_lora_sync() -> None:
    """Synchronize merged LoRA weights through upstream dense vLLM updates."""
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    anchor = """        count, num_params = 0, len(list(model.named_parameters()))
"""
    replacement = """        if self.strategy.args.lora_rank > 0:
            from peft.tuners.lora import LoraLayer

            lora_layers = [
                (name, module)
                for name, module in model.named_modules()
                if isinstance(module, LoraLayer)
            ]
            if not lora_layers:
                raise RuntimeError("LoRA v2 found no trainable LoraLayer modules")

            for count, (module_name, module) in enumerate(lora_layers, start=1):
                active_adapters = module.active_adapters
                if isinstance(active_adapters, str):
                    active_adapters = [active_adapters]
                else:
                    active_adapters = list(active_adapters)
                if len(active_adapters) != 1:
                    raise RuntimeError(
                        f"Expected one active adapter for {module_name}, "
                        f"found {active_adapters}"
                    )
                adapter_name = active_adapters[0]
                if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                    raise RuntimeError(
                        f"Unsupported non-linear LoRA target: {module_name}"
                    )
                base_weight = module.get_base_layer().weight
                adapter_a = module.lora_A[adapter_name].weight
                adapter_b = module.lora_B[adapter_name].weight
                gathered = [base_weight, adapter_a, adapter_b]

                with deepspeed.zero.GatheredParameters(
                    gathered,
                    enabled=self.strategy.args.zero_stage == 3,
                ):
                    with torch.no_grad():
                        merged_weight = (
                            base_weight.detach()
                            + module.get_delta_weight(adapter_name).to(
                                dtype=base_weight.dtype
                            )
                        ).detach().contiguous()
                    vllm_name = module_name
                    if vllm_name.startswith("base_model.model."):
                        vllm_name = vllm_name[len("base_model.model."):]
                    vllm_name = vllm_name + ".weight"

                    if self.use_cuda_ipc:
                        from torch.multiprocessing.reductions import reduce_tensor
                        from openrlhf.trainer.ray.utils import get_physical_gpu_id

                        local_handle = {
                            get_physical_gpu_id(): reduce_tensor(merged_weight)
                        }
                        handle_list = [None] * torch.distributed.get_world_size()
                        torch.distributed.all_gather_object(
                            handle_list,
                            local_handle,
                        )
                        if torch.distributed.get_rank() == 0:
                            ipc_handles = {}
                            for handle_by_device in handle_list:
                                ipc_handles.update(handle_by_device)
                            refs = [
                                engine.update_weight_cuda_ipc.remote(
                                    vllm_name,
                                    dtype=merged_weight.dtype,
                                    shape=merged_weight.shape,
                                    ipc_handles=ipc_handles,
                                    empty_cache=count == len(lora_layers),
                                )
                                for engine in self.vllm_engines
                            ]
                            ray.get(refs)
                    else:
                        if torch.distributed.get_rank() == 0:
                            refs = [
                                engine.update_weight.remote(
                                    vllm_name,
                                    dtype=merged_weight.dtype,
                                    shape=merged_weight.shape,
                                    empty_cache=count == len(lora_layers),
                                )
                                for engine in self.vllm_engines
                            ]
                            use_ray = getattr(
                                self.strategy.args,
                                "vllm_sync_with_ray",
                                False,
                            )
                            if use_ray:
                                import ray.util.collective as collective
                                collective.broadcast(
                                    merged_weight,
                                    0,
                                    group_name=self._model_update_group,
                                )
                            else:
                                torch.distributed.broadcast(
                                    merged_weight,
                                    0,
                                    group=self._model_update_group,
                                )
                            ray.get(refs)

                    torch.distributed.barrier()
                    torch.cuda.synchronize()
                    del merged_weight

            if self.strategy.is_rank_0():
                self.strategy.print(
                    f"LoRA v2 synchronized {len(lora_layers)} merged dense weights"
                )
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            return

        count, num_params = 0, len(list(model.named_parameters()))
"""
    _replace_once(actor_path, anchor, replacement, "dense merged LoRA sync")

    _replace_once(
        actor_path,
        """        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
""",
        """        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        needs_initial_sync = (
            args.load_checkpoint and os.path.exists(ckpt_path)
        ) or args.lora_rank > 0
        if needs_initial_sync and vllm_engines is not None:
""",
        "initial merged LoRA sync",
    )


def _patch_llamafactory_lora_initialization() -> None:
    """Delegate adapter construction to pinned LLaMA-Factory v0.9.3."""
    actor_path = UPSTREAM_WORK / "openrlhf/models/actor.py"
    _replace_once(
        actor_path,
        "from peft import LoraConfig, TaskType, get_peft_model\n",
        "from llamafactory.hparams import FinetuningArguments, ModelArguments\n"
        "from llamafactory.model.adapter import init_adapter as llamafactory_init_adapter\n",
        "LLaMA-Factory adapter imports",
    )
    _replace_once(
        actor_path,
        """            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
""",
        """            # LLaMA-Factory owns LoRA injection, freezing, and
            # ZeRO-3-aware trainable parameter dtype handling.
            if lora_rank > 0:
                model_args = ModelArguments(model_name_or_path=pretrain_or_model)
                finetuning_args = FinetuningArguments(
                    stage="sft",
                    finetuning_type="lora",
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_target=(
                        ",".join(target_modules)
                        if target_modules
                        else "all"
                    ),
                    pure_bf16=False,
                )
                self.model = llamafactory_init_adapter(
                    self.model.config,
                    self.model,
                    model_args,
                    finetuning_args,
                    is_trainable=True,
                )
                self.model.enable_input_require_grads()
                self.model.print_trainable_parameters()

                if load_in_4bit:
""",
        "LLaMA-Factory LoRA construction",
    )


def _prepare_lora_v2_upstream() -> None:
    """Prepare a clean upstream tree without the legacy LoRA interface."""
    _prepare_upstream_source()
    _patch_linear_cot_format_parser()
    _patch_upstream_vllm_version_check()
    _patch_upstream_sft_chat_template()
    _patch_upstream_sft_micro_batch_floor()
    _patch_upstream_release_rl_logits_before_sft()
    _patch_upstream_zero3_sync_active_params()
    _patch_llamafactory_lora_initialization()
    _patch_upstream_replay_buffer_diagnostics()
    _patch_upstream_attacker_only_sampling()
    _patch_upstream_role_lr_scheduler()
    _patch_single_role_advantage_normalization()
    _patch_upstream_remote_rm_retry()
    _patch_upstream_comprehensive_wandb_logging()
    _patch_upstream_defender_metric_keys()
    _patch_reference_kl_monitoring()
    _patch_role_specific_online_sft()
    _patch_wandb_run_identity()
    _patch_colocated_base_defender()
    _patch_dense_merged_lora_sync()


def _adapter_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
    )


def _install_llamafactory_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "llamafactory==0.9.3",
            "peft==0.15.2",
            "trl==0.9.6",
        ],
        check=True,
    )


def _run_attacker(
    *,
    remote_rm_url: str,
    run_dir: Path,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    sft_stop_after_step: int,
    save_steps: int,
) -> Path:
    subprocess.run(["ray", "stop", "--force"], check=False)
    _prepare_lora_v2_upstream()
    _install_upstream_runtime()
    _install_llamafactory_runtime()
    os.environ["PYTHONPATH"] = ":".join(
        [str(UPSTREAM_WORK), "/roll", os.environ.get("PYTHONPATH", "")]
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--num-gpus",
            "4",
            "--num-cpus",
            "8",
            "--disable-usage-stats",
        ],
        check=True,
    )

    prompt_data = ",".join(
        str(UPSTREAM_WORK / f"red_team/data/vanilla_{label}_dataset.jsonl")
        for label in ("harmful", "benign")
    )
    custom_configs = {
        "max_turns": 2,
        "reward_type": "general_sum",
        "remove_ties": True,
        "no_defender_turn": True,
        "optimizer_train_role": "attacker",
        "actor_lr_scheduler": "cosine_with_min_lr",
        "postfill_cot_stop_after_step": sft_stop_after_step,
    }
    run_name = f"{run_dir.parent.name}__{run_dir.name}"
    run_id = hashlib.sha1(run_name.encode()).hexdigest()[:8]
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY is missing from roll-secrets")

    ckpt_dir = run_dir / "ckpt"
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
        "0.35",
        "--fixed_defender_vllm_gpu_memory_utilization",
        "0.30",
        "--pretrain",
        BASE_MODEL,
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--target_modules",
        *LORA_TARGET_MODULES,
        "--save_path",
        str(run_dir),
        "--ckpt_path",
        str(ckpt_dir),
        "--save_steps",
        str(save_steps),
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
        prompt_data,
        "--prompt_data_probs",
        "0.5,0.5",
        "--sft_data",
        ATTACKER_SFT_DATA,
        "--sft_data_probs",
        "1.0",
        "--sft_steps",
        "1",
        "--sft_batches_per_step",
        "1",
        "--max_samples",
        str(128 * steps),
        "--max_epochs",
        "1",
        "--prompt_max_len",
        "2048",
        "--generate_max_len",
        "2048",
        "--flash_attn",
        "--zero_stage",
        "3",
        "--gradient_checkpointing_use_reentrant",
        "--num_episodes",
        "1",
        "--bf16",
        "--seed",
        "8888",
        "--top_p",
        "1.0",
        "--temperature",
        "1.0",
        "--actor_learning_rate",
        str(learning_rate),
        "--init_kl_coef",
        "0.0",
        "--monitor_reference_kl",
        "--normalize_reward",
        "--packing_samples",
        "--gradient_checkpointing",
        "--advantage_estimator",
        "reinforce",
        "--custom_configs",
        json.dumps(custom_configs),
        "--actor_loss_coef",
        "1.0",
        "--postfill_cot_loss_coef",
        "1.0",
        "--eval_data",
        str(UPSTREAM_WORK / "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl"),
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
        "--use_wandb",
        wandb_key,
        "--wandb_org",
        "2373025856w-the-university-of-hong-kong",
        "--wandb_project",
        "self-play",
        "--wandb_group",
        "upstream-selfredteam-role-lora-v2",
        "--wandb_run_name",
        run_name,
        "--wandb_max_log",
        "10000",
        "--wandb_table_log_interval",
        "1",
        "--wandb_table_csv_path",
        str(run_dir / "run_tables"),
    ]

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "Self-RedTeam attacker-only PEFT LoRA v2",
        "base_model": BASE_MODEL,
        "fixed_defender": BASE_MODEL,
        "legacy_lora_interface": False,
        "rollout_sync": "dense merged LoRA weights via upstream full-weight vLLM update",
        "steps": steps,
        "rollout_batch_size": 128,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(LORA_TARGET_MODULES),
        "learning_rate": learning_rate,
        "kl_penalty": 0.0,
        "reference_kl_monitored": True,
        "attacker_sft_stop_after_step": sft_stop_after_step,
        "wandb_run_id": run_id,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    redacted = list(command)
    redacted[redacted.index("--use_wandb") + 1] = "<redacted>"
    (run_dir / "command.json").write_text(json.dumps(redacted, indent=2))
    print(f"WANDB_RUN_ID={run_id}", flush=True)
    print(
        "WANDB_URL=https://wandb.ai/2373025856w-the-university-of-hong-kong/"
        f"self-play/runs/{run_id}",
        flush=True,
    )

    log_path = run_dir / "training.log"
    try:
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
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)
        output_vol.commit()
    if return_code:
        raise RuntimeError(f"LoRA v2 attacker phase exited with code {return_code}")
    checkpoint = ckpt_dir / f"global_step{steps}_hf"
    if not _adapter_checkpoint(checkpoint):
        raise RuntimeError(f"Missing LoRA v2 checkpoint: {checkpoint}")
    return checkpoint


@lora_v2_app.function(
    image=llamafactory_lora_image,
    gpu=os.environ.get("UPSTREAM_ROLE_LORA_V2_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_lora_v2_attacker_probe(
    steps: int = 50,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 5e-6,
    sft_stop_after_step: int = 30,
    save_steps: int = 10,
    run_suffix: str = "",
) -> dict[str, str]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= sft_stop_after_step <= steps:
        raise ValueError("sft_stop_after_step must be in [0, steps]")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"attacker_r{lora_rank}a{lora_alpha}_s{steps}_lr{learning_rate:g}_{suffix}"
    )
    checkpoint = _run_attacker(
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=root,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        save_steps=save_steps,
    )
    return {"run_dir": str(root), "checkpoint": str(checkpoint)}


@lora_v2_app.function(cpu=2, timeout=1800)
def validate_lora_v2_configuration() -> dict[str, object]:
    _prepare_lora_v2_upstream()
    actor_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    ).read_text()
    actor_model_source = (UPSTREAM_WORK / "openrlhf/models/actor.py").read_text()
    cli_source = (UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py").read_text()
    red_team_utils_source = (UPSTREAM_WORK / "red_team/utils.py").read_text()
    required = (
        "llamafactory_init_adapter",
        "FinetuningArguments(",
        "Extract Self-RedTeam tags in linear time",
        "LoRA v2 synchronized",
        "module.get_delta_weight(adapter_name)",
        "needs_initial_sync",
        "fixed_defender_vllm_gpu_memory_utilization",
        "postfill_cot_stop_after_step",
    )
    combined = (
        actor_source
        + actor_model_source
        + cli_source
        + red_team_utils_source
    )
    missing = [item for item in required if item not in combined]
    forbidden = (
        "update_lora_weight_cuda_ipc",
        "roll_training_lora_v1",
        "fixed_opponent_lora_path",
    )
    legacy = [item for item in forbidden if item in combined]
    if missing or legacy:
        raise RuntimeError(
            f"LoRA v2 validation failed: missing={missing}, legacy={legacy}"
        )
    return {
        "valid": True,
        "legacy_lora_sync_present": False,
        "rank": 64,
        "alpha": 64,
        "learning_rate": 5e-6,
        "target_modules": list(LORA_TARGET_MODULES),
    }


@lora_v2_app.local_entrypoint(name="lora_v2_attacker_probe")
def lora_v2_attacker_probe(
    steps: int = 50,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 5e-6,
    sft_stop_after_step: int = 30,
    save_steps: int = 10,
    run_suffix: str = "",
    detach: bool = True,
) -> None:
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    kwargs = {
        "steps": steps,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "sft_stop_after_step": sft_stop_after_step,
        "save_steps": save_steps,
        "run_suffix": suffix,
    }
    if detach:
        call = train_lora_v2_attacker_probe.spawn(**kwargs)
        print(f"CALL_ID={call.object_id}")
        print(f"RUN_SUFFIX={suffix}")
        return
    print(json.dumps(train_lora_v2_attacker_probe.remote(**kwargs), indent=2))


@lora_v2_app.local_entrypoint(name="validate_lora_v2")
def validate_lora_v2() -> None:
    print(json.dumps(validate_lora_v2_configuration.remote(), indent=2))
