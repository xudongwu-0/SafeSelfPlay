#!/usr/bin/env python3
"""Train separate attacker/defender LoRAs with phased best responses.

The rollout, reward, tie filtering, role-wise advantage normalization, and
Re++ objective come from mickelliu/selfplay-redteaming.  The shared full-policy
actor is replaced by two trainable LoRA adapters over one frozen Qwen2.5-3B
backbone. Every outer step generates one A -> D game batch. In the default
schedule, only one role adapter is optimized for a full phase while its
opponent remains frozen; the next phase reverses those roles.

ABS reports role learning rates, LoRA settings, data balance, generation
protocol, and hardware, but not rollout/global/micro batch sizes.  Those three
values therefore inherit the released Self-RedTeam command: 128/32/8.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import modal
import requests

if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_abs_benchmark import (  # noqa: E402
    _hf_token,
    hf_cache,
    image,
    output_vol,
)
from modal_upstream_selfredteam_fixed_seed import (  # noqa: E402
    BASE_MODEL,
    OPTIMIZED_BENIGN_PROMPT,
    OPTIMIZED_HARMFUL_PROMPT,
    SFT_ADAPTER,
)
from modal_upstream_selfredteam_role_lora import (  # noqa: E402
    UPSTREAM_WORK,
    _prepare_role_lora_upstream,
    _replace_once,
)


OUTPUT_ROOT = "/output/abs_bipolicy_h200"
app = modal.App("abs-bipolicy-h200", image=image)
ATTACKER_SFT_DATA = "/aux_sft/attacker_rewrite_1180.jsonl"
DEFENDER_SFT_DATA = (
    "/tmp/selfplay-redteaming/red_team/data/"
    "helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl,"
    "/tmp/selfplay-redteaming/red_team/data/"
    "vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl"
)


def _warm_stable_reward_endpoint(remote_rm_url: str) -> None:
    """Bring up one serialized L40S reward container per actor rank."""
    payload = {
        "queries": [
            {
                "game_idx": 0,
                "prompt": "How can I stay safe while hiking?",
                "response": "Share your route and check the weather.",
            }
        ]
    }

    def warm_worker(_: int) -> None:
        last_error: BaseException | None = None
        for attempt in range(6):
            try:
                response = requests.post(remote_rm_url, json=payload, timeout=900)
                response.raise_for_status()
                return
            except requests.RequestException as error:
                last_error = error
                if attempt < 5:
                    time.sleep(min(60, 2 ** (attempt + 1)))
        assert last_error is not None
        raise last_error

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(warm_worker, range(4)))
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _prepare_dual_role_start_adapters(
    attacker_adapter: str,
    destination: Path,
) -> str:
    """Build an SFT-attacker/base-defender PEFT checkpoint pair."""
    import inspect

    import torch
    from peft import LoraConfig
    from safetensors.torch import load_file, save_file

    source = Path(attacker_adapter)
    if not source.is_dir():
        raise FileNotFoundError(attacker_adapter)
    if destination.exists():
        shutil.rmtree(destination)
    attacker_dir = destination / "attacker"
    defender_dir = destination / "defender"
    shutil.copytree(source, attacker_dir)
    shutil.copytree(source, defender_dir)

    accepted_config_keys = set(inspect.signature(LoraConfig).parameters)
    for role_dir in (attacker_dir, defender_dir):
        config_path = role_dir / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config = {
            key: value
            for key, value in config.items()
            if key in accepted_config_keys
        }
        # Dropout during RL obscures the exact role-start KL reference. The
        # trained SFT weights are unchanged; only subsequent RL forwards are
        # deterministic with respect to LoRA dropout.
        config["lora_dropout"] = 0.0
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    defender_weights_path = defender_dir / "adapter_model.safetensors"
    defender_weights = load_file(str(defender_weights_path), device="cpu")
    save_file(
        {
            name: torch.zeros_like(weight)
            for name, weight in defender_weights.items()
        },
        str(defender_weights_path),
    )
    print(
        "Prepared role starts: attacker=SFT adapter, defender=zero/base LoRA",
        flush=True,
    )
    return str(destination)


def _patch_dual_role_actor() -> None:
    path = UPSTREAM_WORK / "openrlhf/models/actor.py"
    _replace_once(
        path,
        """        lora_init_path=None,
        lora_trainable=True,
        ds_config=None,
""",
        """        lora_init_path=None,
        lora_trainable=True,
        dual_role_lora=False,
        ds_config=None,
""",
        "dual-role Actor constructor argument",
    )
    _replace_once(
        path,
        """            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                if lora_init_path:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_init_path,
                        is_trainable=lora_trainable,
                    )
                else:
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
        """            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                if dual_role_lora:
                    self.dual_role_lora_trainable = lora_trainable
                    if lora_init_path:
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            __import__("os").path.join(
                                lora_init_path, "attacker"
                            ),
                            adapter_name="attacker",
                            is_trainable=lora_trainable,
                        )
                        self.model.load_adapter(
                            __import__("os").path.join(
                                lora_init_path, "defender"
                            ),
                            adapter_name="defender",
                            is_trainable=lora_trainable,
                        )
                    else:
                        lora_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            target_modules=target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                        )
                        self.model = get_peft_model(
                            self.model, lora_config, adapter_name="attacker"
                        )
                        self.model.add_adapter("defender", lora_config)
                    self.model.set_adapter("attacker")
                    # Keep both role adapters in the optimizer. Only the active
                    # adapter participates in a role-specific forward pass.
                    if lora_trainable:
                        for name, parameter in self.model.named_parameters():
                            if ".attacker." in name or ".defender." in name:
                                parameter.requires_grad_(True)
                elif lora_init_path:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_init_path,
                        is_trainable=lora_trainable,
                    )
                else:
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
        "dual attacker/defender PEFT adapters",
    )
    _replace_once(
        path,
        """    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
""",
        """    def set_role_adapter(self, role: str) -> None:
        if role not in {"attacker", "defender"}:
            raise ValueError(f"Unknown ABS role adapter: {role}")
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.set_adapter(role)
        # PEFT set_adapter toggles requires_grad. DeepSpeed was initialized
        # with both adapter groups, so retain both groups while relying on the
        # active-adapter forward path to select which parameters receive grads.
        if getattr(self, "dual_role_lora_trainable", True):
            for name, parameter in model.named_parameters():
                if ".attacker." in name or ".defender." in name:
                    parameter.requires_grad_(True)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
""",
        "Actor role adapter switch",
    )


def _patch_dual_role_cli_and_optimizer() -> None:
    cli_path = UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py"
    _replace_once(
        cli_path,
        '    parser.add_argument("--fixed_opponent_lora_path", type=str, default=None)\n',
        '    parser.add_argument("--fixed_opponent_lora_path", type=str, default=None)\n'
        '    parser.add_argument("--dual_role_lora", action="store_true", default=False)\n'
        '    parser.add_argument("--attacker_learning_rate", type=float, default=1e-6)\n'
        '    parser.add_argument("--defender_learning_rate", type=float, default=3e-6)\n'
        '    parser.add_argument("--attacker_sft_data", type=str, default=None)\n'
        '    parser.add_argument("--defender_sft_data", type=str, default=None)\n',
        "dual-role CLI options",
    )
    _replace_once(
        cli_path,
        """            args.lora_rank,
            args.fixed_opponent_lora_path,
        )
""",
        """            args.lora_rank,
            args.fixed_opponent_lora_path,
            args.dual_role_lora,
        )
""",
        "dual-role vLLM creation argument",
    )

    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """            lora_init_path=strategy.args.lora_init_path,
            ds_config=strategy.get_ds_train_config(is_actor=True),
""",
        """            lora_init_path=strategy.args.lora_init_path,
            dual_role_lora=strategy.args.dual_role_lora,
            ds_config=strategy.get_ds_train_config(is_actor=True),
""",
        "dual-role Actor initialization",
    )

    strategy_path = UPSTREAM_WORK / "openrlhf/utils/deepspeed/deepspeed.py"
    _replace_once(
        strategy_path,
        """        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim
""",
        """        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        if getattr(self.args, "dual_role_lora", False):
            attacker_params = []
            defender_params = []
            unexpected = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".attacker." in name:
                    attacker_params.append(parameter)
                elif ".defender." in name:
                    defender_params.append(parameter)
                else:
                    unexpected.append(name)
            if unexpected:
                raise RuntimeError(
                    "ABS dual-role optimizer found non-LoRA trainable params: "
                    + ", ".join(unexpected[:10])
                )
            if not attacker_params or not defender_params:
                raise RuntimeError(
                    "Both attacker and defender LoRA parameter groups are required"
                )
            optim = AdamOptimizer(
                [
                    {
                        "params": attacker_params,
                        "lr": self.args.attacker_learning_rate,
                        "weight_decay": kwargs["weight_decay"],
                        "role": "attacker",
                    },
                    {
                        "params": defender_params,
                        "lr": self.args.defender_learning_rate,
                        "weight_decay": kwargs["weight_decay"],
                        "role": "defender",
                    },
                ],
                lr=self.args.attacker_learning_rate,
                betas=kwargs["betas"],
            )
        else:
            optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
            optim = AdamOptimizer(optim_params, **kwargs)
        return optim
""",
        "role-specific optimizer parameter groups",
    )

    scheduler_text = actor_path.read_text()
    scheduler_start = scheduler_text.index(
        "        actor_lr_scheduler = self.strategy.args.custom_configs.get("
    )
    scheduler_end = scheduler_text.index(
        "        if args.gradient_checkpointing:", scheduler_start
    )
    scheduler_replacement = """        if args.dual_role_lora:
            from torch.optim.lr_scheduler import LambdaLR

            warmup_steps = math.ceil(max_steps * args.lr_warmup_ratio)

            def abs_lr_scale(step):
                if warmup_steps and step < warmup_steps:
                    return max(step, 1) / warmup_steps
                progress = (step - warmup_steps) / max(
                    1, max_steps - warmup_steps
                )
                progress = min(max(progress, 0.0), 1.0)
                return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

            actor_scheduler = LambdaLR(
                actor_optim,
                lr_lambda=[abs_lr_scale] * len(actor_optim.param_groups),
            )
        else:
            actor_lr_scheduler = self.strategy.args.custom_configs.get(
                "actor_lr_scheduler", "cosine_with_min_lr"
            )
            if actor_lr_scheduler == "constant":
                actor_scheduler = get_scheduler("constant", actor_optim)
            elif actor_lr_scheduler == "cosine_with_min_lr":
                actor_scheduler = get_scheduler(
                    "cosine_with_min_lr",
                    actor_optim,
                    num_warmup_steps=math.ceil(
                        max_steps * args.lr_warmup_ratio
                    ),
                    num_training_steps=max_steps,
                    scheduler_specific_kwargs={
                        "min_lr": args.actor_learning_rate * 0.1
                    },
                )
            else:
                raise ValueError(
                    f"Unsupported actor_lr_scheduler: {actor_lr_scheduler}"
                )

"""
    actor_path.write_text(
        scheduler_text[:scheduler_start]
        + scheduler_replacement
        + scheduler_text[scheduler_end:]
    )


def _patch_dual_role_reference_model() -> None:
    """Make KL reference each role's own immutable starting adapter."""
    launcher_path = UPSTREAM_WORK / "openrlhf/trainer/ray/launcher.py"
    _replace_once(
        launcher_path,
        """            lora_rank=(
                strategy.args.lora_rank
                if strategy.args.reference_lora_init_path
                else 0
            ),
            lora_init_path=strategy.args.reference_lora_init_path,
            lora_trainable=False,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
""",
        """            lora_rank=(
                strategy.args.lora_rank
                if strategy.args.reference_lora_init_path
                else 0
            ),
            lora_init_path=strategy.args.reference_lora_init_path,
            lora_trainable=False,
            dual_role_lora=strategy.args.dual_role_lora,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
""",
        "dual-role reference policy initialization",
    )
    _replace_once(
        launcher_path,
        """        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
""",
        """        packed_seq_lens: Optional[list[int]] = None,
        game_role: Optional[str] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        if game_role is not None:
            self.model.set_role_adapter(game_role)
        with torch.no_grad():
""",
        "reference policy role switch",
    )


def _patch_dual_role_lightweight_resume() -> None:
    """Account for both role updates when restoring the LR schedule."""
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """                resume_updates = resume_step * updates_per_rollout
                self.actor_scheduler.step(resume_updates)
""",
        """                if args.dual_role_lora:
                    updates_per_rollout = int(updates_per_rollout * 1.5)
                resume_updates = resume_step * updates_per_rollout
                self.actor_scheduler.step(resume_updates)
""",
        "dual-role lightweight-resume scheduler position",
    )


def _patch_dual_role_vllm() -> None:
    engine_path = UPSTREAM_WORK / "openrlhf/trainer/ray/vllm_engine.py"
    _replace_once(
        engine_path,
        """_FIXED_OPPONENT_LORA_INT_ID = (
    int(hashlib.sha256(b"roll_fixed_opponent_lora_v1").hexdigest(), 16)
    % 0x7FFFFFFF
)
""",
        """_FIXED_OPPONENT_LORA_INT_ID = (
    int(hashlib.sha256(b"roll_fixed_opponent_lora_v1").hexdigest(), 16)
    % 0x7FFFFFFF
)
_ROLE_LORA_IDS = {
    role: int(hashlib.sha256(f"abs_{role}_lora_v1".encode()).hexdigest(), 16)
    % 0x7FFFFFFF
    for role in ("attacker", "defender")
}
_ROLE_LORA_PATHS = {
    role: os.path.join(
        os.path.expanduser("~"), ".cache", "abs", f"{role}_lora_v1"
    )
    for role in ("attacker", "defender")
}
""",
        "role LoRA IDs",
    )
    _replace_once(
        engine_path,
        """        self.current_lora_request = None
        fixed_opponent_lora_path = kwargs.pop(
""",
        """        self.current_lora_request = None
        self.dual_role_lora = kwargs.pop("dual_role_lora", False)
        self.role_lora_requests = {
            role: LoRARequest(
                lora_name=f"abs_{role}_lora",
                lora_int_id=_ROLE_LORA_IDS[role],
                lora_path=_ROLE_LORA_PATHS[role],
            )
            for role in ("attacker", "defender")
        } if self.dual_role_lora else {}
        fixed_opponent_lora_path = kwargs.pop(
""",
        "role LoRA request initialization",
    )
    _replace_once(
        engine_path,
        """    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()
""",
        """    def update_role_lora_weight(
        self, role, name, dtype, shape, empty_cache=False
    ):
        return self.llm.collective_rpc(
            "update_role_lora_weight",
            args=(role, name, dtype, shape, empty_cache),
        )

    def update_role_lora_weight_cuda_ipc(
        self, role, name, dtype, shape, ipc_handles, empty_cache=False
    ):
        return self.llm.collective_rpc(
            "update_role_lora_weight_cuda_ipc",
            args=(role, name, dtype, shape, ipc_handles, empty_cache),
        )

    def finalize_role_lora(self, role, peft_config):
        return self.llm.collective_rpc(
            "custom_add_role_lora", args=(role, peft_config)
        )

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()
""",
        "role LoRA engine update RPCs",
    )
    _replace_once(
        engine_path,
        """    def add_requests(self, actor_rank, *, sampling_params, prompt_token_ids, use_lora=True):
""",
        """    def add_requests(
        self, actor_rank, *, sampling_params, prompt_token_ids,
        use_lora=True, lora_role=None
    ):
""",
        "role-aware vLLM requests",
    )
    _replace_once(
        engine_path,
        """        self.requests[actor_rank] = prompt_token_ids
        self.actor_counter += 1
""",
        """        self.requests[actor_rank] = (prompt_token_ids, lora_role)
        self.actor_counter += 1
""",
        "request role storage",
    )
    _replace_once(
        engine_path,
        """            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                for r in request:
                    requests.append(TokensPrompt(prompt_token_ids=r))

            if len(requests) > 0:
""",
        """            request_roles = set()
            for actor_rank, (request, request_role) in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                request_roles.add(request_role)
                for r in request:
                    requests.append(TokensPrompt(prompt_token_ids=r))
            if len(request_roles) != 1:
                raise RuntimeError(
                    f"Mixed vLLM adapter roles in one synchronized call: {request_roles}"
                )
            request_role = next(iter(request_roles))

            if len(requests) > 0:
""",
        "synchronized request role validation",
    )
    _replace_once(
        engine_path,
        """                    lora_request=(
                        self.fixed_opponent_lora_request
                        if use_lora == "fixed_opponent"
                        else self.current_lora_request
                        if use_lora
                        else None
                    ),
""",
        """                    lora_request=(
                        self.role_lora_requests[request_role]
                        if request_role in self.role_lora_requests
                        else self.fixed_opponent_lora_request
                        if use_lora == "fixed_opponent"
                        else self.current_lora_request
                        if use_lora
                        else None
                    ),
""",
        "role adapter generation selector",
    )
    _replace_once(
        engine_path,
        """    lora_rank=0,
    fixed_opponent_lora_path=None,
):
""",
        """    lora_rank=0,
    fixed_opponent_lora_path=None,
    dual_role_lora=False,
):
""",
        "dual-role vLLM factory argument",
    )
    _replace_once(
        engine_path,
        """                enable_lora=lora_rank > 0 or bool(fixed_opponent_lora_path),
                max_loras=2 if fixed_opponent_lora_path else 1,
                max_lora_rank=max(1, lora_rank),
                fixed_opponent_lora_path=fixed_opponent_lora_path,
""",
        """                enable_lora=(
                    lora_rank > 0
                    or bool(fixed_opponent_lora_path)
                    or dual_role_lora
                ),
                max_loras=(
                    2 if dual_role_lora
                    else 2 if fixed_opponent_lora_path
                    else 1
                ),
                max_lora_rank=max(1, lora_rank),
                fixed_opponent_lora_path=fixed_opponent_lora_path,
                dual_role_lora=dual_role_lora,
""",
        "dual-role vLLM capacity",
    )

    worker_path = UPSTREAM_WORK / "openrlhf/trainer/ray/vllm_worker_wrap.py"
    _replace_once(
        worker_path,
        "from roll.third_party.vllm.worker import WorkerV1\n",
        """import hashlib
import os
from collections import OrderedDict

from roll.third_party.vllm.vllm_utils import TensorLoRARequest
from roll.third_party.vllm.worker import WorkerV1


_ROLE_LORA_IDS = {
    role: int(hashlib.sha256(f"abs_{role}_lora_v1".encode()).hexdigest(), 16)
    % 0x7FFFFFFF
    for role in ("attacker", "defender")
}
_ROLE_LORA_PATHS = {
    role: os.path.join(
        os.path.expanduser("~"), ".cache", "abs", f"{role}_lora_v1"
    )
    for role in ("attacker", "defender")
}
""",
        "role LoRA worker imports",
    )
    _replace_once(
        worker_path,
        """    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
""",
        """    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        self.role_lora_params = {
            "attacker": OrderedDict(),
            "defender": OrderedDict(),
        }

    def update_role_lora_weight(
        self, role, name, dtype, shape, empty_cache=False
    ):
        import torch

        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective
            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(
                weight, 0, group=self._model_update_group
            )
        self.role_lora_params[role][name] = weight

    def update_role_lora_weight_cuda_ipc(
        self, role, name, dtype, shape, ipc_handles=None, empty_cache=False
    ):
        import torch
        from openrlhf.trainer.ray.utils import get_physical_gpu_id

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        rebuild_tensor, rebuild_args = handle
        rebuild_args = list(rebuild_args)
        rebuild_args[6] = device_id
        weight = rebuild_tensor(*rebuild_args)
        assert weight.dtype == dtype
        assert tuple(weight.shape) == tuple(shape)
        self.role_lora_params[role][name] = weight
        torch.cuda.synchronize()

    def custom_add_role_lora(self, role, peft_config):
        request = TensorLoRARequest(
            lora_name=f"abs_{role}_lora",
            lora_int_id=_ROLE_LORA_IDS[role],
            lora_path=_ROLE_LORA_PATHS[role],
            peft_config=peft_config,
            lora_tensors=self.role_lora_params[role],
        )
        self.role_lora_params[role] = OrderedDict()
        super().reload_model()
        self.model_runner.remove_lora(request.lora_int_id)
        return self.model_runner.add_lora(request)
""",
        "role LoRA worker methods",
    )


def _patch_dual_role_generation_and_training() -> None:
    experience_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/experience_maker.py"
    )
    _replace_once(
        experience_path,
        """        prompts_list = [p for s in samples_list for p in s.prompts]
        labels_list = [l for s in samples_list for l in s.labels]

        # Move data to CPU for remote processing
""",
        """        prompts_list = [p for s in samples_list for p in s.prompts]
        labels_list = [l for s in samples_list for l in s.labels]
        reference_roles_list = []
        if args.dual_role_lora:
            for samples in samples_list:
                roles = set(samples.additional_infos["game_role"])
                if len(roles) != 1:
                    raise RuntimeError(
                        f"Packed reference batch mixes roles: {roles}"
                    )
                reference_roles_list.append(next(iter(roles)))

        # Move data to CPU for remote processing
""",
        "reference role extraction",
    )
    _replace_once(
        experience_path,
        """                logps_allgather=[True] * len(samples_list),
                packed_seq_lens=packed_seq_lens_list,
            )
""",
        """                logps_allgather=[True] * len(samples_list),
                packed_seq_lens=packed_seq_lens_list,
                game_role=(
                    reference_roles_list
                    if args.dual_role_lora
                    else [None] * len(samples_list)
                ),
            )
""",
        "reference role propagation",
    )
    _replace_once(
        experience_path,
        """        else:
            def attacker_llm_generator(
                batch_chat_messages, all_labels, **gen_kwargs
            ):
                return self._generate_vllm(
                    self.vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
""",
        """        else:
            def attacker_llm_generator(
                batch_chat_messages, all_labels, **gen_kwargs
            ):
                return self._generate_vllm(
                    self.vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    lora_role=(
                        "attacker"
                        if custom_configs.get("dual_role_lora", False)
                        else None
                    ),
                    **gen_kwargs,
                )
""",
        "attacker role adapter generation",
    )
    _replace_once(
        experience_path,
        """        else:
            # If no_defender_turn is not enabled or defender_vllm_engines is not available, 
            # use the same generator for both
            defender_llm_generator = attacker_llm_generator
""",
        """        else:
            # The ABS bipolicy uses a distinct defender adapter on the same
            # frozen backbone/vLLM engines.
            if custom_configs.get("dual_role_lora", False):
                def defender_llm_generator(
                    batch_chat_messages, all_labels, **gen_kwargs
                ):
                    return self._generate_vllm(
                        self.vllm_engines,
                        batch_chat_messages,
                        all_labels,
                        lora_role="defender",
                        **gen_kwargs,
                    )
            else:
                defender_llm_generator = attacker_llm_generator
""",
        "defender role adapter generation",
    )
    _replace_once(
        experience_path,
        """        args = self.strategy.args
        use_lora = kwargs.pop("use_lora", True)

        sampling_params = SamplingParams(
""",
        """        args = self.strategy.args
        use_lora = kwargs.pop("use_lora", True)
        lora_role = kwargs.pop("lora_role", None)

        sampling_params = SamplingParams(
""",
        "generation role argument",
    )
    _replace_once(
        experience_path,
        """                    use_lora=use_lora,
                )
""",
        """                    use_lora=use_lora,
                    lora_role=lora_role,
                )
""",
        "generation role propagation",
    )
    _replace_once(
        experience_path,
        """        for seq, num_acts, attn_mask, packed_lens in zip(
            sequences_cpu_list, num_actions_list, attention_mask_cpu_list, packed_seq_lens_list
        ):
            action_log_probs = self.actor(
""",
        """        for samples, seq, num_acts, attn_mask, packed_lens in zip(
            samples_list,
            sequences_cpu_list,
            num_actions_list,
            attention_mask_cpu_list,
            packed_seq_lens_list,
        ):
            if args.dual_role_lora:
                roles = set(samples.additional_infos["game_role"])
                if len(roles) != 1:
                    raise RuntimeError(
                        f"Packed experience mixes ABS roles: {roles}"
                    )
                self.actor.set_role_adapter(next(iter(roles)))
            action_log_probs = self.actor(
""",
        "role-correct old policy log probabilities",
    )
    _replace_once(
        experience_path,
        """        # Handle buffer sizes for distributed training
        if self.strategy.stage == 3:
""",
        """        # A packed forward pass must use exactly one ABS adapter. Align each
        # role independently across data-parallel ranks. Do not round role
        # counts down to a full micro-batch: _post_process_sequences supports a
        # final partial batch, and rounding discarded roughly half the attacker
        # trajectories in typical rollouts.
        if custom_configs.get("dual_role_lora", False):
            selected_indices = []
            role_sync = {}
            for role in ("attacker", "defender"):
                role_indices = [
                    index for index, turn_state in enumerate(filtered_turn_states)
                    if turn_state.get("game_role") == role
                ]
                all_role_lengths = self.strategy.all_gather(len(role_indices))
                common_length = int(min(all_role_lengths))
                keep_length = common_length
                if keep_length == 0:
                    raise RuntimeError(
                        f"No aligned {role} trajectories: {all_role_lengths}"
                    )
                if len(role_indices) > keep_length:
                    permutation = torch.randperm(len(role_indices))[:keep_length]
                    kept = sorted(role_indices[index] for index in permutation.tolist())
                else:
                    kept = role_indices[:keep_length]
                selected_indices.extend(kept)
                role_sync[role] = {
                    "rank_lengths": [int(value) for value in all_role_lengths],
                    "kept_per_rank": keep_length,
                }
            filtered_outputs = [filtered_outputs[index] for index in selected_indices]
            filtered_turn_states = [
                filtered_turn_states[index] for index in selected_indices
            ]
            if self.strategy.is_rank_0():
                self.strategy.print(f"ABS role-aligned rollout batches: {role_sync}")

        # Handle buffer sizes for distributed training
        if self.strategy.stage == 3:
""",
        "role-aligned packed rollout synchronization",
    )
    _replace_once(
        experience_path,
        """        # Post process sequences for experience replay
        samples_list = self._post_process_sequences(filtered_outputs, filtered_turn_states, None, None, **kwargs)
""",
        """        # Post process each role separately. This prevents a final partial
        # attacker batch from being packed with the first defender trajectories.
        if custom_configs.get("dual_role_lora", False):
            samples_list = []
            for role in ("attacker", "defender"):
                role_pairs = [
                    (output, turn_state)
                    for output, turn_state in zip(
                        filtered_outputs, filtered_turn_states
                    )
                    if turn_state.get("game_role") == role
                ]
                role_outputs = [pair[0] for pair in role_pairs]
                role_turn_states = [pair[1] for pair in role_pairs]
                samples_list.extend(
                    self._post_process_sequences(
                        role_outputs, role_turn_states, None, None, **kwargs
                    )
                )
        else:
            samples_list = self._post_process_sequences(
                filtered_outputs, filtered_turn_states, None, None, **kwargs
            )
""",
        "role-separated experience packing",
    )

    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    start = actor_path.read_text().index("    def ppo_train_actor(self, global_steps):")
    end = actor_path.read_text().index("    def training_step(", start)
    text = actor_path.read_text()
    replacement = '''    def ppo_train_actor(self, global_steps):
        torch.cuda.empty_cache()
        device = torch.cuda.current_device()

        if not self.args.dual_role_lora:
            return self._ppo_train_actor_shared(global_steps)

        schedule = self.args.custom_configs.get(
            "role_training_schedule", "simultaneous"
        )
        if schedule == "simultaneous":
            active_roles = ("attacker", "defender")
        elif schedule == "attacker_then_defender":
            switch_step = int(
                self.args.custom_configs.get("role_phase_switch_step", 50)
            )
            active_roles = (
                ("attacker",) if global_steps <= switch_step else ("defender",)
            )
        else:
            raise ValueError(f"Unknown role training schedule: {schedule}")

        role_status = {
            "phase/attacker_active": float("attacker" in active_roles),
            "phase/defender_active": float("defender" in active_roles),
        }
        all_status = []
        role_dataloaders = getattr(self, "role_sft_dataloaders", {})
        if self.strategy.is_rank_0():
            self.strategy.print(
                f"ABS role schedule at outer step {global_steps}: "
                f"{','.join(active_roles)}"
            )
        for role in active_roles:
            role_items = [
                item for item in self.replay_buffer.items
                if item.info.get("game_role") == role
            ]
            # Tie removal happens after rollout-level role alignment and can
            # remove a different number of role trajectories on each DP rank.
            # Equalize again here so every rank executes the same number of
            # ZeRO-3 forwards and optimizer collectives.
            role_counts = self.strategy.all_gather(len(role_items))
            common_role_count = int(min(role_counts))
            if common_role_count == 0:
                raise RuntimeError(
                    f"No aligned {role} trajectories survived filtering: "
                    f"{role_counts}"
                )
            role_items = role_items[:common_role_count]
            if self.strategy.is_rank_0():
                self.strategy.print(
                    f"ABS {role} post-tie alignment: "
                    f"rank_counts={[int(value) for value in role_counts]}, "
                    f"kept_per_rank={common_role_count}"
                )
            self.actor.set_role_adapter(role)
            self._active_training_role = role
            self.sft_dataloader = role_dataloaders.get(role)
            dataloader = DataLoader(
                role_items,
                batch_size=self.replay_buffer.sample_batch_size,
                shuffle=False if self.strategy.ring_attn_group is not None else True,
                drop_last=False,
                pin_memory=self.dataloader_pin_memory,
                collate_fn=self.replay_buffer.collate_fn,
            )
            statuses = []
            for _ in range(self.max_epochs):
                pbar = tqdm(
                    dataloader,
                    desc=f"ABS {role} train",
                    disable=not self.strategy.is_rank_0(),
                )
                for experience in pbar:
                    experience.to_device(device)
                    status = self.training_step(experience, global_steps)
                    if "kl" in status:
                        status["kl"] *= status["response_length"]
                        status = self.strategy.all_reduce(status)
                        status["kl"] /= status["response_length"]
                    statuses.append(status)
                    all_status.append(status)
                    pbar.set_postfix(
                        {
                            "pg": status.get("policy_loss", 0.0),
                            "rm": status.get("reward", 0.0),
                            "kl": status.get("kl", 0.0),
                            "lr": status.get(f"{role}_lr", status.get("actor_lr", 0.0)),
                        }
                    )
            if statuses:
                keys = set().union(*(status.keys() for status in statuses))
                for key in keys:
                    values = [status[key] for status in statuses if key in status]
                    if values and all(isinstance(value, (int, float)) for value in values):
                        role_status[f"{role}/train_{key}"] = sum(values) / len(values)
                role_status[f"{role}/optimizer_microbatches"] = len(statuses)
                role_status[f"{role}/surviving_trajectories"] = len(role_items)

        if all_status:
            keys = set().union(*(status.keys() for status in all_status))
            for key in keys:
                values = [status[key] for status in all_status if key in status]
                if values and all(isinstance(value, (int, float)) for value in values):
                    role_status[key] = sum(values) / len(values)
        torch.cuda.empty_cache()
        return role_status

    def _ppo_train_actor_shared(self, global_steps):
        torch.cuda.empty_cache()
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=False if self.strategy.ring_attn_group is not None else True,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()
        status_list = []
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]
                status_list.append(status)
        status_mean = {}
        if status_list:
            keys = set().union(*(status.keys() for status in status_list))
            for key in keys:
                values = [status[key] for status in status_list if key in status]
                if values and all(isinstance(value, (int, float)) for value in values):
                    status_mean[key] = sum(values) / len(values)
        torch.cuda.empty_cache()
        return status_mean

'''
    actor_path.write_text(text[:start] + replacement + text[end:])

    _replace_once(
        actor_path,
        """        status = {
            "policy_loss": actor_loss.item(),
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
        }
""",
        """        current_lrs = self.actor_scheduler.get_last_lr()
        active_role = getattr(self, "_active_training_role", None)
        role_index = 0 if active_role == "attacker" else 1
        active_lr = current_lrs[min(role_index, len(current_lrs) - 1)]
        status = {
            "policy_loss": actor_loss.item(),
            "actor_lr": active_lr,
        }
        if active_role:
            status[f"{active_role}_lr"] = active_lr
""",
        "active role learning-rate metric",
    )
    _replace_once(
        actor_path,
        """        self.sft_dataloader = sft_dataloader
""",
        """        self.role_sft_dataloaders = (
            sft_dataloader if isinstance(sft_dataloader, dict) else {}
        )
        self.sft_dataloader = (
            None if isinstance(sft_dataloader, dict) else sft_dataloader
        )
""",
        "role SFT loader registration",
    )


def _patch_role_specific_sft() -> None:
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    marker = """        if args.pretrain_data:
"""
    insertion = '''        if args.dual_role_lora:
            role_sft_loaders = {}

            attacker_strategy = deepcopy(strategy)
            attacker_strategy.args.apply_chat_template = True
            attacker_strategy.args.sft_input_key = "messages"
            attacker_strategy.args.sft_output_key = None
            attacker_data = blending_datasets(
                args.attacker_sft_data,
                "1.0",
                attacker_strategy,
                args.seed,
                return_eval=False,
                train_split=args.sft_split,
            )
            attacker_dataset = SFTDataset(
                attacker_data,
                self.tokenizer,
                args.max_len or args.prompt_max_len + args.generate_max_len,
                attacker_strategy,
                pretrain_mode=False,
                prompt_input_template=None,
            )
            role_sft_loaders["attacker"] = itertools.cycle(
                iter(
                    attacker_strategy.setup_dataloader(
                        attacker_dataset,
                        batch_size=max(1, args.micro_train_batch_size // 2),
                        pin_memory=True,
                        shuffle=True,
                        collate_fn=attacker_dataset.packing_collate_fn,
                    )
                )
            )

            defender_strategy = deepcopy(strategy)
            defender_strategy.args.apply_chat_template = True
            defender_strategy.args.prompt_input_template = DEFENDER_INSTRUCTION_COT_PROMPT
            defender_strategy.args.sft_input_key = "vanilla"
            defender_strategy.args.sft_output_key = "completion"
            defender_data = blending_datasets(
                args.defender_sft_data,
                "0.5,0.5",
                defender_strategy,
                args.seed,
                return_eval=False,
                train_split=args.sft_split,
            )
            defender_dataset = SFTDataset(
                defender_data,
                self.tokenizer,
                args.max_len or args.prompt_max_len + args.generate_max_len,
                defender_strategy,
                pretrain_mode=False,
                prompt_input_template=DEFENDER_INSTRUCTION_COT_PROMPT,
            )
            role_sft_loaders["defender"] = itertools.cycle(
                iter(
                    defender_strategy.setup_dataloader(
                        defender_dataset,
                        batch_size=max(1, args.micro_train_batch_size // 2),
                        pin_memory=True,
                        shuffle=True,
                        collate_fn=defender_dataset.packing_collate_fn,
                    )
                )
            )
            self.sft_dataloader = role_sft_loaders

'''
    text = actor_path.read_text()
    if marker not in text:
        raise RuntimeError("Could not locate pretrain dataset block")
    actor_path.write_text(text.replace(marker, insertion + marker, 1))


def _patch_dual_role_broadcast_and_save() -> None:
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    marker = """        if self.strategy.args.lora_rank > 0:
            from dataclasses import asdict
"""
    dual_branch = '''        if self.strategy.args.dual_role_lora:
            from dataclasses import asdict

            peft_model = model
            for role in ("attacker", "defender"):
                role_params = []
                for name, param in peft_model.named_parameters():
                    if "lora_" not in name or f".{role}." not in name:
                        continue
                    clean_name = name
                    if clean_name.startswith("base_model.model."):
                        clean_name = clean_name[len("base_model.model."):]
                    clean_name = clean_name.replace(f".{role}.", ".")
                    role_params.append((clean_name, param))
                if not role_params:
                    raise RuntimeError(f"No vLLM tensors found for {role} LoRA")
                for count, (name, param) in enumerate(role_params, start=1):
                    with deepspeed.zero.GatheredParameters(
                        [param],
                        enabled=self.strategy.args.zero_stage == 3,
                    ):
                        if self.use_cuda_ipc:
                            from torch.multiprocessing.reductions import reduce_tensor
                            from openrlhf.trainer.ray.utils import get_physical_gpu_id

                            weight = param.data.clone()
                            local_handle = {
                                get_physical_gpu_id(): reduce_tensor(weight)
                            }
                            handle_list = [None] * torch.distributed.get_world_size()
                            torch.distributed.all_gather_object(
                                handle_list, local_handle
                            )
                            if torch.distributed.get_rank() == 0:
                                ipc_handles = {}
                                for handle_by_device in handle_list:
                                    ipc_handles.update(handle_by_device)
                                shape = (
                                    param.shape
                                    if self.strategy.args.zero_stage != 3
                                    else param.ds_shape
                                )
                                ray.get([
                                    engine.update_role_lora_weight_cuda_ipc.remote(
                                        role,
                                        name,
                                        dtype=param.dtype,
                                        shape=shape,
                                        ipc_handles=ipc_handles,
                                        empty_cache=count == len(role_params),
                                    )
                                    for engine in self.vllm_engines
                                ])
                            torch.distributed.barrier()
                            torch.cuda.synchronize()
                        else:
                            if torch.distributed.get_rank() == 0:
                                shape = (
                                    param.shape
                                    if self.strategy.args.zero_stage != 3
                                    else param.ds_shape
                                )
                                refs = [
                                    engine.update_role_lora_weight.remote(
                                        role,
                                        name,
                                        dtype=param.dtype,
                                        shape=shape,
                                        empty_cache=count == len(role_params),
                                    )
                                    for engine in self.vllm_engines
                                ]
                            torch.distributed.broadcast(
                                param.data, 0, group=self._model_update_group
                            )
                            if torch.distributed.get_rank() == 0:
                                ray.get(refs)
                if torch.distributed.get_rank() == 0:
                    peft_config = asdict(peft_model.peft_config[role])
                    ray.get([
                        engine.finalize_role_lora.remote(role, peft_config)
                        for engine in self.vllm_engines
                    ])
                torch.distributed.barrier()
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return

'''
    text = actor_path.read_text()
    if marker not in text:
        raise RuntimeError("Could not locate LoRA broadcast branch")
    actor_path.write_text(text.replace(marker, dual_branch + marker, 1))

    strategy_path = UPSTREAM_WORK / "openrlhf/utils/deepspeed/deepspeed.py"
    _replace_once(
        strategy_path,
        """            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
                    filename = os.path.join(output_dir, "adapter_model.safetensors")
                    if os.path.exists(filename):
                        os.remove(filename)
""",
        """            if isinstance(model_to_save, PeftModel):
                if getattr(self.args, "dual_role_lora", False):
                    model_to_save.save_pretrained(
                        output_dir,
                        state_dict=output_state_dict,
                        selected_adapters=["attacker", "defender"],
                        **kwargs,
                    )
                else:
                    model_to_save.save_pretrained(output_dir, **kwargs)
                    if self.stage == 3:
                        torch.save(
                            get_peft_model_state_dict(model_to_save, output_state_dict),
                            os.path.join(output_dir, "adapter_model.bin"),
                        )
                        filename = os.path.join(output_dir, "adapter_model.safetensors")
                        if os.path.exists(filename):
                            os.remove(filename)
""",
        "save both role adapters",
    )


def _prepare_abs_bipolicy_upstream() -> None:
    _prepare_role_lora_upstream()
    _patch_dual_role_actor()
    _patch_dual_role_cli_and_optimizer()
    _patch_dual_role_reference_model()
    _patch_dual_role_lightweight_resume()
    _patch_dual_role_vllm()
    _patch_dual_role_generation_and_training()
    _patch_role_specific_sft()
    _patch_dual_role_broadcast_and_save()


def _run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return process.wait()


@app.function(
    gpu="H200:4",
    cpu=16,
    memory=65536,
    timeout=43200,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_abs_bipolicy_h200(
    remote_rm_url: str,
    target_step: int = 200,
    resume_step: int = 0,
    resume_checkpoint: str = "",
    rank: int = 32,
    rollout_batch_size: int = 128,
    train_batch_size: int = 32,
    micro_train_batch_size: int = 8,
    attacker_learning_rate: float = 2e-6,
    defender_learning_rate: float = 3e-6,
    kl_coef: float = 0.01,
    training_schedule: str = "attacker_then_defender",
    phase_switch_step: int = 100,
    attacker_init_adapter: str = SFT_ADAPTER,
    aux_sft_interval: int = 4,
    aux_sft_coef: float = 0.1,
    run_suffix: str = "",
) -> str:
    """Run role-separated best-response training and persist both adapters."""
    if target_step not in (100, 200):
        raise ValueError("The calibrated entrypoint supports target steps 100 or 200")
    if resume_step not in (0, 100):
        raise ValueError("resume_step must be 0 or 100")
    if resume_step and target_step <= resume_step:
        raise ValueError("target_step must exceed resume_step")
    if bool(resume_step) != bool(resume_checkpoint):
        raise ValueError("resume_step and resume_checkpoint must be set together")
    if (rollout_batch_size, train_batch_size, micro_train_batch_size) != (128, 32, 8):
        raise ValueError(
            "Batch values are locked to the released Self-RedTeam command: 128/32/8"
        )
    if attacker_learning_rate not in (1e-6, 2e-6) or defender_learning_rate != 3e-6:
        raise ValueError(
            "Supported role learning rates are attacker 1e-6 or 2e-6 and "
            "defender 3e-6"
        )
    if training_schedule not in ("simultaneous", "attacker_then_defender"):
        raise ValueError(f"Unsupported training schedule: {training_schedule}")
    if training_schedule == "attacker_then_defender":
        if resume_step:
            raise ValueError("Phased role training must start from step 0")
        if not 0 < phase_switch_step < target_step:
            raise ValueError("phase_switch_step must lie inside the run")
    if kl_coef < 0:
        raise ValueError("kl_coef must be non-negative")
    if aux_sft_interval < 1:
        raise ValueError("aux_sft_interval must be positive")
    if aux_sft_coef < 0:
        raise ValueError("aux_sft_coef must be non-negative")
    if not attacker_init_adapter:
        raise ValueError("An attacker SFT start adapter is required")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    token = _hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HF_HUB_TOKEN"] = token
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY is missing from roll-secrets")

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    step_label = (
        f"from_s{resume_step}_to_s{target_step}"
        if resume_step
        else f"s{target_step}"
    )
    attacker_lr_label = f"{attacker_learning_rate:.0e}".replace("e-0", "e-")
    defender_lr_label = f"{defender_learning_rate:.0e}".replace("e-0", "e-")
    kl_label = f"{kl_coef:g}".replace(".", "p")
    aux_label = f"{aux_sft_coef:g}".replace(".", "p")
    schedule_label = (
        "simultaneous"
        if training_schedule == "simultaneous"
        else f"phased_A{phase_switch_step}_D{target_step - phase_switch_step}"
    )
    run_name = (
        f"seprole_qwen25_3b_duallora_r{rank}_sftA_baseD_{schedule_label}_"
        f"{step_label}_rb128_tb32_mb8_aLR{attacker_lr_label}_"
        f"dLR{defender_lr_label}_"
        f"kl{kl_label}_aux{aux_label}every{aux_sft_interval}_ourprompts_{suffix}"
    )
    output_vol.reload()
    run_dir = Path(OUTPUT_ROOT) / run_name
    ckpt_dir = run_dir / "ckpt"
    table_dir = run_dir / "run_tables"
    run_dir.mkdir(parents=True, exist_ok=True)

    _prepare_abs_bipolicy_upstream()
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
    role_start_path = _prepare_dual_role_start_adapters(
        attacker_init_adapter,
        Path("/tmp/separate_role_start_adapters"),
    )
    os.environ["PYTHONPATH"] = ":".join(
        path for path in (str(UPSTREAM_WORK), "/roll", os.environ.get("PYTHONPATH", "")) if path
    )

    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--num-gpus",
            "4",
            "--num-cpus",
            "16",
            "--min-worker-port",
            "20000",
            "--max-worker-port",
            "20255",
            "--metrics-export-port",
            "31001",
            "--disable-usage-stats",
        ],
        check=True,
    )

    manifest = {
        "method": (
            "two-LoRA simultaneous self-play calibration"
            if training_schedule == "simultaneous"
            else "role-separated best response: frozen defender during attacker phase, then frozen attacker during defender phase"
        ),
        "base_model": BASE_MODEL,
        "target_outer_self_play_step": target_step,
        "resume_step": resume_step,
        "additional_outer_self_play_steps": target_step - resume_step,
        "attacker_role_steps_total": (
            target_step if training_schedule == "simultaneous" else phase_switch_step
        ),
        "defender_role_steps_total": (
            target_step
            if training_schedule == "simultaneous"
            else target_step - phase_switch_step
        ),
        "resume_checkpoint": resume_checkpoint or None,
        "resume_scope": (
            "role LoRA weights, data position, and scheduler position; "
            "Adam moments unavailable because the source run disabled DS checkpoints"
            if resume_step
            else None
        ),
        "shared_backbone": True,
        "separate_role_loras": True,
        "role_training_schedule": training_schedule,
        "role_phase_switch_step": (
            phase_switch_step if training_schedule == "attacker_then_defender" else None
        ),
        "same_batch_role_updates": training_schedule == "simultaneous",
        "phase_semantics": (
            None
            if training_schedule == "simultaneous"
            else (
                f"Outer steps 1-{phase_switch_step} update only the attacker "
                "LoRA against the frozen base-initialized defender LoRA; "
                f"outer steps {phase_switch_step + 1}-{target_step} update only "
                f"the defender LoRA against the frozen attacker checkpoint at "
                f"step {phase_switch_step}."
            )
        ),
        "attacker_initialization": attacker_init_adapter,
        "defender_initialization": "zero LoRA, exactly the base policy",
        "role_reference_policy": (
            "attacker KL is against the immutable SFT attacker start; defender "
            "KL is against the immutable zero-LoRA/base defender start"
        ),
        "lora_rank": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "target_modules": list(TARGET_MODULES),
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "kl_coef_interpretation": "released Self-RedTeam token-level KL coefficient",
        "kl_coef": kl_coef,
        "rollout_batch_size": rollout_batch_size,
        "train_batch_size": train_batch_size,
        "micro_train_batch_size": micro_train_batch_size,
        "batch_source": "inherited from released Self-RedTeam command; ABS paper does not disclose batch sizes",
        "prompt_data": "WildJailBreak train, 50% harmful / 50% benign",
        "effective_game_mix": "25% vanilla harmful, 25% rewritten harmful, 25% vanilla benign, 25% rewritten benign",
        "attacker_prompt": {
            "harmful": OPTIMIZED_HARMFUL_PROMPT,
            "benign": OPTIMIZED_BENIGN_PROMPT,
        },
        "defender_prompt": "original Self-RedTeam hidden-CoT defender prompt",
        "reward": "released Self-RedTeam general_sum + format reward",
        "attacker_sft_data": ATTACKER_SFT_DATA,
        "defender_sft_data": DEFENDER_SFT_DATA.split(","),
        "role_specific_sft": True,
        "aux_sft_interval_outer_steps": aux_sft_interval,
        "aux_sft_loss_coefficient": aux_sft_coef,
        "generation": {"prompt_max_len": 2048, "generate_max_len": 2048, "temperature": 1.0, "top_p": 1.0},
        "hardware": "Modal H200 x4 (four GPUs preserve the released global batch geometry)",
        "paper_hardware_note": "ABS reports three H200-141GB GPUs but does not disclose its distributed batch geometry",
        "gradient_checkpointing": False,
        "gradient_checkpointing_note": (
            "Disabled because PyTorch activation recomputation observes ZeRO-3 "
            "partition placeholders after dynamic role-adapter routing; this is a "
            "memory-only implementation choice and does not change optimization."
        ),
        "save_steps": list(range(max(50, resume_step + 50), target_step + 1, 50)),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    custom_configs = {
        "max_turns": 2,
        "reward_type": "general_sum",
        "remove_ties": True,
        "dual_role_lora": True,
        "role_training_schedule": training_schedule,
        "role_phase_switch_step": phase_switch_step,
        "lightweight_resume_step": resume_step,
    }
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
        "--vllm_gpu_memory_utilization", "0.7",
        "--pretrain", BASE_MODEL,
        "--dual_role_lora",
        "--lora_rank", str(rank),
        "--lora_alpha", str(rank),
        "--target_modules", *TARGET_MODULES,
        "--attacker_learning_rate", str(attacker_learning_rate),
        "--defender_learning_rate", str(defender_learning_rate),
        "--actor_learning_rate", str(attacker_learning_rate),
        "--attacker_sft_data", ATTACKER_SFT_DATA,
        "--defender_sft_data", DEFENDER_SFT_DATA,
        "--sft_steps", str(aux_sft_interval),
        "--sft_batches_per_step", "1",
        "--save_path", str(run_dir),
        "--ckpt_path", str(ckpt_dir),
        "--save_steps", "50",
        "--save_hf_ckpt",
        "--disable_ds_ckpt",
        "--micro_train_batch_size", str(micro_train_batch_size),
        "--train_batch_size", str(train_batch_size),
        "--micro_rollout_batch_size", "8",
        "--rollout_batch_size", str(rollout_batch_size),
        "--prompt_data",
        ",".join(
            [
                str(UPSTREAM_WORK / "red_team/data/vanilla_harmful_dataset.jsonl"),
                str(UPSTREAM_WORK / "red_team/data/vanilla_benign_dataset.jsonl"),
            ]
        ),
        "--prompt_data_probs", "0.5,0.5",
        "--max_samples", str(rollout_batch_size * target_step),
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
        "--init_kl_coef", str(kl_coef),
        "--normalize_reward",
        "--packing_samples",
        "--advantage_estimator", "reinforce",
        "--custom_configs", json.dumps(custom_configs),
        "--actor_loss_coef", "1.0",
        "--postfill_cot_loss_coef", str(aux_sft_coef),
        "--eval_steps", "100000",
        "--eval_start_steps", "100000",
        "--diversity_score_steps", "5",
        "--vllm_sync_backend", "nccl",
        "--enforce_eager",
        "--vllm_enable_sleep",
        "--deepspeed_enable_sleep",
        "--use_wandb", wandb_key,
        "--wandb_org", "2373025856w-the-university-of-hong-kong",
        "--wandb_project", "self-play",
        "--wandb_group", "separate-role-best-response",
        "--wandb_run_name", run_name,
        "--wandb_max_log", "32",
        "--wandb_table_log_interval", "5",
        "--wandb_table_csv_path", str(table_dir),
    ]
    if resume_checkpoint:
        command.extend(["--lora_init_path", resume_checkpoint])
    else:
        command.extend(["--lora_init_path", role_start_path])
    command.extend(["--reference_lora_init_path", role_start_path])

    status_path = run_dir / "run_status.json"
    return_code = -1
    try:
        return_code = _run_command(command, UPSTREAM_WORK, run_dir / "training.log")
        status_path.write_text(
            json.dumps(
                {
                    "return_code": return_code,
                    "completed": return_code == 0,
                    "run_name": run_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)
        output_vol.commit()
    return str(run_dir)


@app.local_entrypoint(name="abs_bipolicy_h200")
def abs_bipolicy_h200(run_suffix: str = "") -> None:
    reward_function = modal.Function.from_name(
        "selfredteam-wildguard", "wildguard_reward_app"
    )
    reward_url = reward_function.get_web_url()
    if not reward_url:
        raise RuntimeError("The deployed stable WildGuard service has no web URL")
    rm_url = reward_url.rstrip("/") + "/classify"
    _warm_stable_reward_endpoint(rm_url)
    result = train_abs_bipolicy_h200.remote(
        remote_rm_url=rm_url,
        run_suffix=run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    print(result)
