#!/usr/bin/env python3
"""Train a role-specific LLaMA-Factory LoRA in Self-RedTeam.

This is intentionally independent from ``modal_upstream_selfredteam_role_lora``.
LLaMA-Factory owns adapter construction, trainable parameter selection, dtype
handling under ZeRO-3, and PEFT checkpointing. Self-RedTeam continues to own
the online RL loop, reward, prompts, batches, and optimizer updates. Rollout
synchronization sends the native LoRA A/B tensors to vLLM's LoRA manager. This
keeps rollout generation numerically aligned with PEFT instead of quantizing a
small LoRA delta into the frozen BF16 base weights.

The existing full-parameter entrypoint is not imported or modified at runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_upstream_selfredteam_fixed_seed import (
    UPSTREAM_WORK,
    _patch_only_attacker_instruction,
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
    DEFENDER_SFT_FILES,
    _install_upstream_runtime,
    _patch_role_specific_online_sft,
)
from roll.utils.role_lora_training_reward import (
    TRAINING_LABEL_DRIFT_POLICY,
)


BASE_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
OUTPUT_ROOT = "/output/upstream_selfredteam_role_lora_v2"
ATTACKER_SFT_DATA = "/aux_sft/attacker_rewrite_1180.jsonl"
ATTACKER_RL_SFT_DATA = "/tmp/attacker_rewrite_1180_rl_continuation.jsonl"
DEFENDER_RL_SFT_ROOT = Path("/tmp/defender_rl_sft")
SELFPLAY_LOCAL = Path(__file__).resolve().parent.parent / "selfplay-redteaming"
ATTACKER_SFT_LOCAL = (
    Path(__file__).resolve().parent
    / "data"
    / "safety_selfplay"
    / "attacker_rewrite_1180.jsonl"
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
        "pybase64",
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
llamafactory_lora_image = llamafactory_lora_image.add_local_dir(
    str(Path(__file__).resolve().parent / "roll"),
    "/roll/roll",
    copy=False,
    ignore=["__pycache__", "**/*.pyc", "tests/", "docs/"],
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
    The upstream regex has overlapping whitespace and lazy wildcard
    quantifiers and can
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

    # Keep malformed hidden reasoning out of the opponent-visible move.  A
    # missing closing answer tag is common early in training; preserve only
    # the intended answer payload so the opponent still receives the attack,
    # while keeping the format violation signal for the trainable policy.
    if think_end_pos < 0 or answer is None:
        if answer is not None:
            fallback_answer = answer
        elif answer_start_pos >= 0:
            fallback_answer = response[
                answer_start_pos + len(answer_open):
            ].strip()
        elif think_end_pos >= 0:
            fallback_answer = response[
                think_end_pos + len(think_close):
            ].strip()
        else:
            fallback_answer = ""
        return (None, fallback_answer), True

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


def _patch_exact_prompt_label_balance() -> None:
    """Make finite PSRO role-training prefixes exactly 50/50 H/B seeds."""

    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        '''        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
''',
        '''        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        if args.custom_configs.get("exact_prompt_label_balance", False):
            if args.max_samples % 2:
                raise ValueError("exact H/B balance requires an even max_samples")
            harmful_indices = [
                index
                for index, data_type in enumerate(prompts_data["data_type"])
                if "harmful" in data_type
            ]
            benign_indices = [
                index
                for index, data_type in enumerate(prompts_data["data_type"])
                if "benign" in data_type
            ]
            quota = args.max_samples // 2
            if len(harmful_indices) < quota or len(benign_indices) < quota:
                raise ValueError(
                    "blended prompt sources cannot fill exact H/B quotas: "
                    f"harmful={len(harmful_indices)}, benign={len(benign_indices)}, "
                    f"quota={quota}"
                )
            selected_indices = harmful_indices[:quota] + benign_indices[:quota]
            prompts_data = prompts_data.select(selected_indices).shuffle(
                seed=args.seed
            )
        else:
            prompts_data = prompts_data.select(
                range(min(args.max_samples, len(prompts_data)))
            )
''',
        "exact finite H/B prompt balance",
    )


def _patch_asymmetric_label_drift_training() -> None:
    """Cap attacker drift rewards and omit drifted defender games."""

    game_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/language_game.py"
    )
    _replace_once(
        game_path,
        "from openrlhf.utils.remote_rm_utils import remote_rm_fn\n",
        "from openrlhf.utils.remote_rm_utils import remote_rm_fn\n"
        "from roll.utils.role_lora_training_reward import (\n"
        "    cap_label_drift_attacker_reward,\n"
        "    label_drift_training_action,\n"
        ")\n",
        "role label-drift policy import",
    )
    _replace_once(
        game_path,
        (
            "            # Skip if wildguard cannot parse the response\n"
            "            if labels.get('is_parsing_error', False):\n"
            "                continue\n"
            "                \n"
            "            for turn_idx, turn in enumerate("
            "game[\"processed_output_history\"]):\n"
        ),
        """            # Skip if wildguard cannot parse the response
            if labels.get('is_parsing_error', False):
                continue

            label_drift_action = label_drift_training_action(
                optimizer_train_role=self.custom_configs.get(
                    "optimizer_train_role"
                ),
                prompt_type=game["prompt_type"],
                prompt_harmfulness=labels.get("prompt_harmfulness"),
                policy=self.custom_configs.get(
                    "label_drift_training_policy"
                ),
            )
            if label_drift_action == "drop":
                # The frozen attacker changed the seed label. Do not let that
                # interaction update the defender or enter its replay buffer.
                continue

            for turn_idx, turn in enumerate(game["processed_output_history"]):
""",
        "drop defender games whose generated attack changes label",
    )
    _replace_once(
        game_path,
        """                turn_states['reward'] = reward
                turn_states['game_outcomes'] = outcome
""",
        """                if (
                    label_drift_action == "cap_zero"
                    and turn_states["game_role"] == "attacker"
                ):
                    turn_states['reward_before_label_drift_cap'] = reward
                    reward = cap_label_drift_attacker_reward(reward)
                turn_states['reward'] = reward
                turn_states['game_outcomes'] = outcome
                turn_states['label_drift_action'] = label_drift_action
""",
        "cap generated label-drift attacker reward at zero",
    )

    experience_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/experience_maker.py"
    )
    _replace_once(
        experience_path,
        """            'actual_prompt_harmfulness', 'actual_prompt_stratum',
            'prompt_origin',
""",
        """            'actual_prompt_harmfulness', 'actual_prompt_stratum',
            'prompt_origin', 'label_drift_action',
            'reward_before_label_drift_cap',
""",
        "label-drift diagnostics in replay metadata",
    )


def _patch_role_aware_malformed_output_fallback() -> None:
    """Recover malformed defender answers without exposing attacker CoT.

    The parser is shared by both roles. Returning an empty string is the safe
    fallback for a malformed attacker generation because the alternative can
    expose its private reasoning to the defender. The same fallback is wrong
    for a defender: an empty response makes WildGuard return no label, turns
    the game into a tie, and removes the attacker sample from RL.
    """
    game_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/language_game.py"
    )
    _replace_once(
        game_path,
        (
            "                (parsed_thinking, parsed_response), "
            "illgel_response_flag = cot_format_check_and_extract(response)\n"
            "                \n"
            "                # Compute length of parsed_response after "
            "tokenization\n"
        ),
        """                (parsed_thinking, parsed_response), illgel_response_flag = cot_format_check_and_extract(response)

                if (
                    illgel_response_flag
                    and player_role == "defender"
                    and not parsed_response.strip()
                ):
                    # This is the terminal role, so recover its best visible
                    # answer for reward classification. Keep the violation
                    # flag unchanged so a trainable defender remains
                    # penalized for malformed output.
                    if "<answer>" in response:
                        defender_visible_response = response.split(
                            "<answer>", 1
                        )[1]
                        defender_visible_response = defender_visible_response.split(
                            "</answer>", 1
                        )[0]
                    elif "</think>" in response:
                        defender_visible_response = response.split(
                            "</think>", 1
                        )[1]
                    else:
                        defender_visible_response = response
                    parsed_response = defender_visible_response.strip()

                # Compute length of parsed_response after tokenization
""",
        "role-aware malformed defender fallback",
    )


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


def _patch_colocated_fixed_opponent() -> None:
    """Place an immutable opponent mixture beside actor/ref/current vLLM."""
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
    if args.fixed_opponent_pretrain:
        # Keep the opposite role immutable on the same H200 node. Ray resource
        # fractions are scheduling declarations only.
        if pg is None:
            raise ValueError("LoRA v2 fixed opponent requires --colocate_all_models")
        defender_vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.fixed_opponent_pretrain,
            args.seed + 10000,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
            pg,
            gpu_memory_utilization=args.fixed_opponent_vllm_gpu_memory_utilization,
            vllm_enable_sleep=False,
            lora_rank=args.fixed_opponent_lora_rank,
            initial_lora_path=args.fixed_opponent_lora_path,
            opponent_pool_json=args.fixed_opponent_pool_json,
            tensor_lora_sync=False,
        )
""",
        "colocated immutable role opponent",
    )
    _replace_once(
        cli_path,
        """        num_gpus_per_actor=0.4 if pg else 1,
        num_resources_per_node=args.actor_num_gpus_per_node,
""",
        """        num_gpus_per_actor=(
            0.3
            if pg and args.fixed_opponent_pretrain
            else 0.4 if pg else 1
        ),
        num_resources_per_node=args.actor_num_gpus_per_node,
""",
        "actor Ray share with fixed opponent",
    )
    _replace_once(
        cli_path,
        """            num_gpus_per_actor=0.4 if pg else 1,
            num_resources_per_node=args.ref_num_gpus_per_node,
""",
        """            num_gpus_per_actor=(
                0.3
                if pg and args.fixed_opponent_pretrain
                else 0.4 if pg else 1
            ),
            num_resources_per_node=args.ref_num_gpus_per_node,
""",
        "reference Ray share with fixed opponent",
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
    parser.add_argument("--fixed_opponent_pretrain", type=str, default=None)
    parser.add_argument("--fixed_opponent_lora_path", type=str, default=None)
    parser.add_argument("--fixed_opponent_lora_rank", type=int, default=0)
    parser.add_argument(
        "--fixed_opponent_pool_json",
        type=str,
        default=None,
        help=(
            "Frozen PSRO opponent entries [{id,adapter,sha256,probability}]; "
            "a null adapter selects the base model"
        ),
    )
    parser.add_argument(
        "--fixed_opponent_vllm_gpu_memory_utilization",
        type=float,
        default=0.30,
        help="GPU memory fraction for the immutable opposite role",
    )
""",
        "fixed opponent CLI arguments",
    )
    _replace_once(
        cli_path,
        """        defender_vllm_engines=defender_vllm_engines if args.custom_configs.get("no_defender_turn", False) else None
""",
        """        defender_vllm_engines=defender_vllm_engines
""",
        "pass fixed opponent engines to experience maker",
    )

    experience_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/experience_maker.py"
    )
    upstream_generator_block = (
        "        # Create generator functions for attacker and defender\n"
        "        def attacker_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):\n"
        "            return self._generate_vllm(self.vllm_engines, batch_chat_messages, all_labels, **gen_kwargs)\n"
        "        " + "\n"
        "        # If no_defender_turn is enabled, use defender_vllm_engines for defender_llm_generator\n"
        "        if custom_configs.get(\"no_defender_turn\", False) and self.defender_vllm_engines is not None:"
        + "            " + "\n"
        "            def defender_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):\n"
        "                return self._generate_vllm(self.defender_vllm_engines, batch_chat_messages, all_labels, **gen_kwargs)\n"
        "        else:\n"
        "            # If no_defender_turn is not enabled or defender_vllm_engines is not available, "
        + "\n"
        "            # use the same generator for both\n"
        "            defender_llm_generator = attacker_llm_generator\n"
    )
    _replace_once(
        experience_path,
        upstream_generator_block,
        """        # Route only the optimized role through the current LoRA policy.
        if custom_configs.get("fixed_attacker_from_opponent_vllm", False):
            if self.defender_vllm_engines is None:
                raise RuntimeError("Frozen attacker vLLM is missing")
            def attacker_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):
                return self._generate_vllm(
                    self.defender_vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
        else:
            def attacker_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):
                return self._generate_vllm(
                    self.vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )

        if custom_configs.get("fixed_defender_from_opponent_vllm", False):
            if self.defender_vllm_engines is None:
                raise RuntimeError("Frozen defender vLLM is missing")
            def defender_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):
                return self._generate_vllm(
                    self.defender_vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
        else:
            def defender_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):
                return self._generate_vllm(
                    self.vllm_engines,
                    batch_chat_messages,
                    all_labels,
                    **gen_kwargs,
                )
""",
        "route frozen attacker or defender independently",
    )


def _patch_single_role_advantage_normalization() -> None:
    """Normalize only the role selected for optimization, exactly once."""
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
                    optimizer_train_role = self.args.custom_configs.get(
                        'optimizer_train_role'
                    )
                    no_attacker_turn = self.args.custom_configs.get(
                        'no_attacker_turn', False
                    )
                    no_defender_turn = self.args.custom_configs.get(
                        'no_defender_turn', False
                    )
                    if optimizer_train_role == 'attacker' or no_defender_turn:
                        self.replay_buffer.normalize(
                            strategy=self.strategy,
                            attribute="advantages",
                            role="attacker",
                            divide_by_std=not self.args.no_advantage_std_norm,
                        )
                    elif optimizer_train_role == 'defender' or no_attacker_turn:
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


def _patch_optimizer_role_replay_filtering() -> None:
    """Keep the frozen opponent in the game but out of PPO optimization."""
    replay_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/replay_buffer.py"
    )
    _replace_once(
        replay_path,
        "    def remove_defender_turn(self, strategy):\n"
        "        if strategy.args.custom_configs.get('no_defender_turn', False):\n"
        "            self.items = [item for item in self.items if item.info['game_role'] == 'attacker']\n"
        "    \n",
        '''    def filter_optimizer_role(self, strategy):
        role = strategy.args.custom_configs.get('optimizer_train_role')
        if role is None:
            return {}
        if role not in ('attacker', 'defender'):
            raise ValueError(f"Invalid optimizer_train_role: {role}")

        before_len = len(self.items)
        self.items = [
            item for item in self.items
            if item.info.get('game_role') == role
        ]
        local_attacker = sum(
            item.info.get('game_role') == 'attacker' for item in self.items
        )
        local_defender = sum(
            item.info.get('game_role') == 'defender' for item in self.items
        )
        all_before = [int(value) for value in strategy.all_gather(before_len)]
        all_after = [int(value) for value in strategy.all_gather(len(self.items))]
        all_attacker = [
            int(value) for value in strategy.all_gather(local_attacker)
        ]
        all_defender = [
            int(value) for value in strategy.all_gather(local_defender)
        ]
        global_after = sum(all_after)
        if global_after == 0:
            raise RuntimeError(
                f"Role filter removed every {role} replay item: "
                f"before={all_before}, after={all_after}"
            )
        unexpected = (
            sum(all_defender) if role == 'attacker' else sum(all_attacker)
        )
        if unexpected:
            raise RuntimeError(
                f"Role-only replay leaked {unexpected} opponent items while "
                f"optimizing {role}"
            )
        if strategy.is_rank_0():
            strategy.print(
                f"Optimizer role filter kept {global_after}/"
                f"{sum(all_before)} {role} replay items"
            )
        return {
            'debug/replay_role_filter_before': sum(all_before),
            'debug/replay_role_filter_after': global_after,
            'debug/replay_attacker_after_role_filter': sum(all_attacker),
            'debug/replay_defender_after_role_filter': sum(all_defender),
        }

    def remove_defender_turn(self, strategy):
        if strategy.args.custom_configs.get('no_defender_turn', False):
            self.items = [
                item for item in self.items
                if item.info['game_role'] == 'attacker'
            ]

''',
        "optimizer role replay filter",
    )

    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        "                if self.args.custom_configs.get(\"no_defender_turn\", False):\n"
        "                    self.replay_buffer.remove_defender_turn(self.strategy)\n"
        "                                                            \n"
        "                # truncate to same length between different actor's buffers\n",
        '''                optimizer_train_role = self.args.custom_configs.get(
                    "optimizer_train_role"
                )
                if optimizer_train_role is not None:
                    status.update(
                        self.replay_buffer.filter_optimizer_role(self.strategy)
                    )
                elif self.args.custom_configs.get("no_defender_turn", False):
                    self.replay_buffer.remove_defender_turn(self.strategy)

                # truncate to same length between different actor's buffers
''',
        "apply optimizer role filter before replay redistribution",
    )


def _patch_single_role_update_budget() -> None:
    """Do not apply the shared-bipolicy 1.5x schedule to a role-only replay."""
    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        '''        if not self.strategy.args.custom_configs.get("no_attacker_turn", False) and not self.strategy.args.custom_configs.get("no_defender_turn", False):
            max_steps *= 1.5 # TODO: we are assuming there will be 1.5x more update because the attacker generation is an additional 50% of training experiences.
''',
        '''        optimizer_train_role = self.strategy.args.custom_configs.get(
            "optimizer_train_role"
        )
        if (
            optimizer_train_role is None
            and not self.strategy.args.custom_configs.get(
                "no_attacker_turn", False
            )
            and not self.strategy.args.custom_configs.get(
                "no_defender_turn", False
            )
        ):
            max_steps *= 1.5
''',
        "single-role scheduler update budget",
    )


def _patch_sparse_tie_redistribution() -> None:
    """Keep data parallel training alive when informative groups are sparse."""
    replay_path = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/replay_buffer.py"
    )
    _replace_once(
        replay_path,
        '''    def remove_ties(self, strategy):
        before_len = len(self.items)
        self.items = [item for item in self.items if GameOutcome.TIE not in item.info['game_outcomes']]
        after_len = len(self.items)
        n_removed_ties = before_len - after_len

        all_removed = strategy.all_gather(n_removed_ties)
        all_after_len = strategy.all_gather(after_len)
        if strategy.is_rank_0():
            strategy.print(f"Removed Ties: {all_removed}")
            strategy.print(f"After Ties: {all_after_len}")
''',
        '''    def remove_ties(self, strategy):
        original_items = self.items
        before_len = len(original_items)
        self.items = [
            item
            for item in original_items
            if GameOutcome.TIE not in item.info['game_outcomes']
        ]
        after_len = len(self.items)
        n_removed_ties = before_len - after_len

        all_removed = strategy.all_gather(n_removed_ties)
        all_after_len = strategy.all_gather(after_len)
        globally_empty = int(sum(all_after_len)) == 0
        if globally_empty:
            # There is no RL preference signal in this step. Retain the
            # zero-advantage samples only as structural padding so every rank
            # can execute the online-SFT update and collective operations.
            self.items = original_items
        if strategy.is_rank_0():
            strategy.print(f"Removed Ties: {all_removed}")
            strategy.print(f"After Ties: {all_after_len}")
            if globally_empty:
                strategy.print(
                    "All groups tied globally; retaining zero-advantage "
                    "padding for this optimizer step"
                )
''',
        "globally empty tie fallback",
    )

    _replace_once(
        replay_path,
        '''                per_rank = (
                    len(global_items)
                    // world_size
                    // micro_train_batch_size
                    * micro_train_batch_size
                )
                assert per_rank > 0, (
                    "No complete replay batch after global tie removal: "
                    f"total={len(global_items)}, world_size={world_size}, "
                    f"micro_batch={micro_train_batch_size}"
                )

                redistribution_step = getattr(self, "_redistribution_step", 0)
                rng = random.Random(
                    int(strategy.args.seed) + redistribution_step
                )
                rng.shuffle(global_items)
                self._redistribution_step = redistribution_step + 1
                rank = strategy.get_rank()
                start = rank * per_rank
                self.items = global_items[start : start + per_rank]

                used = per_rank * world_size
                data_loss = len(global_items) - used
                if strategy.is_rank_0():
                    self.lost_samples = getattr(self, "lost_samples", 0)
                    self.lost_samples += data_loss
                    strategy.print(
                        "Globally redistributed replay buffer after ties: "
                        f"total={len(global_items)}, per_rank={per_rank}, "
                        f"used={used}, dropped={data_loss}"
                    )
                return
''',
        '''                if not global_items:
                    raise RuntimeError(
                        "Replay buffer is globally empty after tie handling"
                    )

                redistribution_step = getattr(self, "_redistribution_step", 0)
                rng = random.Random(
                    int(strategy.args.seed) + redistribution_step
                )
                rng.shuffle(global_items)
                self._redistribution_step = redistribution_step + 1

                # Preserve every complete global micro-batch. The rollout
                # contains several optimizer batches; truncating it to one
                # gradient-accumulation window silently discards most of the
                # RL budget and makes PPO diagnostics degenerate to ratio=1.
                global_micro_batch = world_size * micro_train_batch_size
                lower_total = (
                    len(global_items) // global_micro_batch
                ) * global_micro_batch
                upper_total = (
                    (len(global_items) + global_micro_batch - 1)
                    // global_micro_batch
                ) * global_micro_batch
                if lower_total > 0 and (
                    len(global_items) - lower_total
                    <= upper_total - len(global_items)
                ):
                    required_total = lower_total
                else:
                    required_total = upper_total
                per_rank = required_total // world_size
                if required_total <= len(global_items):
                    selected_items = global_items[:required_total]
                    replicated = 0
                    dropped = len(global_items) - required_total
                else:
                    repeats = (
                        required_total + len(global_items) - 1
                    ) // len(global_items)
                    selected_items = (global_items * repeats)[:required_total]
                    replicated = required_total - len(global_items)
                    dropped = 0

                rank = strategy.get_rank()
                start = rank * per_rank
                self.items = selected_items[start : start + per_rank]

                if strategy.is_rank_0():
                    self.lost_samples = getattr(self, "lost_samples", 0)
                    self.lost_samples += dropped
                    strategy.print(
                        "Globally redistributed replay buffer after ties: "
                        f"informative={len(global_items)}, "
                        f"per_rank={per_rank}, used={required_total}, "
                        f"replicated={replicated}, dropped={dropped}"
                    )
                return
''',
        "sparse informative replay redistribution",
    )


def _patch_native_lora_sync() -> None:
    """Synchronize PEFT A/B tensors into vLLM's native LoRA runtime."""
    worker_path = UPSTREAM_WORK / "openrlhf/trainer/ray/vllm_worker_wrap.py"
    _replace_once(
        worker_path,
        "class WorkerWrap:\n",
        """from collections import OrderedDict

from roll.third_party.vllm.vllm_utils import TensorLoRARequest
from roll.third_party.vllm.worker import (
    WorkerV1,
    _TRAINING_LORA_INT_ID,
    _training_lora_path,
)


class WorkerWrap(WorkerV1):
    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        self._latest_training_lora_config = None
        self._latest_training_lora_tensors = None

    def _install_latest_training_lora(self):
        if self._latest_training_lora_tensors is None:
            return False
        import torch

        request = TensorLoRARequest(
            lora_name="training_lora",
            lora_int_id=_TRAINING_LORA_INT_ID,
            lora_path=_training_lora_path(),
            peft_config=self._latest_training_lora_config,
            lora_tensors=self._latest_training_lora_tensors,
        )
        super().reload_model()
        self.model_runner.remove_lora(request.lora_int_id)
        installed = self.model_runner.add_lora(request)
        # vLLM copies the adapter from the registered CPU model into its GPU
        # LoRA slot during activation.  Synchronize before returning from the
        # RPC so the first rollout after every optimizer update cannot race
        # that transfer and sample from stale/partially copied weights.
        torch.cuda.synchronize()
        return installed

    def custom_add_lora(self, peft_config):
        # vLLM sleep mode can release dynamically loaded adapter weights. Keep
        # one host copy so wake_up can reconstruct the current training slot.
        self._latest_training_lora_config = peft_config
        self._latest_training_lora_tensors = OrderedDict(
            (
                name,
                weight.detach().to(device="cpu", copy=True),
            )
            for name, weight in self.tensor_lora_manager.lora_params.items()
        )
        self.tensor_lora_manager.lora_params = OrderedDict()
        return self._install_latest_training_lora()

    def custom_reload_lora(self):
        return self._install_latest_training_lora()

    def update_lora_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective
            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        self.tensor_lora_manager.add_weight(name, weight)

    def update_lora_weight_cuda_ipc(
        self,
        name,
        dtype,
        shape,
        ipc_handles=None,
        empty_cache=False,
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
        self.tensor_lora_manager.add_weight(name, weight)
        torch.cuda.synchronize()

""",
        "native vLLM LoRA worker",
    )

    engine_path = UPSTREAM_WORK / "openrlhf/trainer/ray/vllm_engine.py"
    _replace_once(
        engine_path,
        "import os\n",
        "import hashlib\nimport json\nimport os\n",
        "stable training LoRA identifier import",
    )
    _replace_once(
        engine_path,
        "from vllm.inputs import TokensPrompt\n",
        "from vllm.inputs import TokensPrompt\n"
        "from vllm.lora.request import LoRARequest\n\n"
        "_TRAINING_LORA_INT_ID = (\n"
        "    int(hashlib.sha256(b\"roll_training_lora_v1\").hexdigest(), 16)\n"
        "    % 0x7FFFFFFF\n"
        ")\n"
        "_FIXED_OPPONENT_LORA_INT_ID = (\n"
        "    int(hashlib.sha256(b\"fixed_opponent_lora_v1\").hexdigest(), 16)\n"
        "    % 0x7FFFFFFF\n"
        ")\n",
        "vLLM LoRA request import",
    )
    _replace_once(
        engine_path,
        """        self.requests = {}
        self.response_queues = defaultdict(queue.Queue)
""",
        """        self.requests = {}
        self.response_queues = defaultdict(queue.Queue)
        self.current_lora_request = None
        self.tensor_lora_sync = kwargs.pop("tensor_lora_sync", False)
        self.initial_lora_path = kwargs.pop("initial_lora_path", None)
        raw_opponent_pool = kwargs.pop("opponent_pool_json", None)
        self.opponent_pool = (
            json.loads(raw_opponent_pool) if raw_opponent_pool else []
        )
        self.opponent_draw_counts = defaultdict(int)
        if self.tensor_lora_sync and (
            self.initial_lora_path or self.opponent_pool
        ):
            raise ValueError(
                "A vLLM engine cannot use tensor sync and frozen LoRAs together"
            )
        if self.initial_lora_path and self.opponent_pool:
            raise ValueError(
                "Use either initial_lora_path or opponent_pool_json, not both"
            )
""",
        "vLLM current LoRA request",
    )
    _replace_once(
        engine_path,
        """        self.llm = vllm.LLM(*args, **kwargs)
""",
        """        self.llm = vllm.LLM(*args, **kwargs)
        if self.tensor_lora_sync:
            # TensorLoRARequest needs ROLL's in-memory adapter loader.
            self.llm.collective_rpc("custom_init_worker")
        elif self.initial_lora_path:
            # Frozen opponents use vLLM's ordinary PEFT file loader.  Do not
            # patch that loader with the tensor-only manager.
            self.current_lora_request = LoRARequest(
                lora_name="fixed_opponent_lora",
                lora_int_id=_FIXED_OPPONENT_LORA_INT_ID,
                lora_path=self.initial_lora_path,
            )
        self.opponent_lora_requests = []
        cumulative = 0.0
        for pool_index, entry in enumerate(self.opponent_pool):
            probability = float(entry["probability"])
            if probability <= 0:
                raise ValueError("Opponent pool probabilities must be positive")
            cumulative += probability
            adapter_path = entry.get("adapter")
            request = (
                LoRARequest(
                    lora_name=f"psro_opponent_{pool_index}_{entry['id']}",
                    lora_int_id=1000 + pool_index,
                    lora_path=adapter_path,
                )
                if adapter_path
                else None
            )
            self.opponent_lora_requests.append((cumulative, request, entry["id"]))
        if self.opponent_lora_requests:
            if abs(cumulative - 1.0) > 1e-8:
                raise ValueError(
                    f"Opponent pool probabilities sum to {cumulative}, not one"
                )
            self.opponent_lora_requests[-1] = (
                1.0,
                self.opponent_lora_requests[-1][1],
                self.opponent_lora_requests[-1][2],
            )
""",
        "initialize vLLM tensor LoRA worker",
    )
    _replace_once(
        engine_path,
        """    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()
""",
        """    def update_lora_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc(
            "update_lora_weight",
            args=(name, dtype, shape, empty_cache),
        )

    def update_lora_weight_cuda_ipc(
        self,
        name,
        dtype,
        shape,
        ipc_handles,
        empty_cache=False,
    ):
        return self.llm.collective_rpc(
            "update_lora_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    def finalize_lora(self, peft_config):
        result = self.llm.collective_rpc("custom_add_lora", args=(peft_config,))
        self.current_lora_request = LoRARequest(
            lora_name="training_lora",
            lora_int_id=_TRAINING_LORA_INT_ID,
            lora_path=os.path.join(
                os.path.expanduser("~"), ".cache", "roll", "training_lora_v1"
            ),
        )
        return result

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def _opponent_selector(self, actor_rank, prompt_token_ids, ordinal):
        if not self.opponent_lora_requests:
            return None
        digest = hashlib.sha256()
        digest.update(b"role_lora_psro_opponent_draw_v1")
        digest.update(str(actor_rank).encode())
        digest.update(str(ordinal).encode())
        for token_id in prompt_token_ids:
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
        draw = int.from_bytes(digest.digest()[:8], "big") / 2**64
        for pool_index, (cumulative, _request, _label) in enumerate(
            self.opponent_lora_requests
        ):
            if draw < cumulative:
                return pool_index
        raise RuntimeError("Opponent mixture CDF did not select a strategy")
""",
        "vLLM tensor LoRA update methods",
    )
    _replace_once(
        engine_path,
        """    def wake_up(self):
        self.llm.wake_up()
""",
        """    def wake_up(self):
        self.llm.wake_up()
        if self.tensor_lora_sync and self.current_lora_request is not None:
            result = self.llm.collective_rpc("custom_reload_lora")
            if not all(result):
                raise RuntimeError(
                    "Failed to reinstall the current LoRA after vLLM wake_up"
                )
            print("Reloaded current LoRA after vLLM wake_up", flush=True)
""",
        "reload dynamic LoRA after vLLM wake-up",
    )
    _replace_once(
        engine_path,
        """        self.requests[actor_rank] = prompt_token_ids
        self.actor_counter += 1
""",
        """        draw_start = self.opponent_draw_counts[actor_rank]
        selectors = [
            self._opponent_selector(
                actor_rank,
                prompt,
                draw_start + local_index,
            )
            for local_index, prompt in enumerate(prompt_token_ids)
        ]
        self.opponent_draw_counts[actor_rank] += len(prompt_token_ids)
        self.requests[actor_rank] = (prompt_token_ids, selectors)
        self.actor_counter += 1
""",
        "deterministic per-episode PSRO opponent selection",
    )
    _replace_once(
        engine_path,
        """            num_requests = []
            requests: list[TokensPrompt] = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                for r in request:
                    requests.append(TokensPrompt(prompt_token_ids=r))
""",
        """            num_requests = []
            requests: list[TokensPrompt] = []
            selectors = []
            for actor_rank, (request, request_selectors) in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                for prompt_token_ids, selector in zip(
                    request, request_selectors, strict=True
                ):
                    requests.append(
                        TokensPrompt(prompt_token_ids=prompt_token_ids)
                    )
                    selectors.append(selector)
""",
        "preserve opponent selectors while gathering actor requests",
    )
    _replace_once(
        engine_path,
        """                responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
""",
        """                if self.opponent_lora_requests:
                    responses = [None] * len(requests)
                    grouped = defaultdict(list)
                    for request_index, selector in enumerate(selectors):
                        grouped[selector].append(request_index)
                    for selector, indices in sorted(grouped.items()):
                        lora_request = self.opponent_lora_requests[selector][1]
                        group_responses = self.llm.generate(
                            prompts=[requests[index] for index in indices],
                            sampling_params=sampling_params,
                            lora_request=lora_request,
                        )
                        for index, response in zip(
                            indices, group_responses, strict=True
                        ):
                            responses[index] = response
                    if any(response is None for response in responses):
                        raise RuntimeError("Opponent response scatter is incomplete")
                else:
                    responses = self.llm.generate(
                        prompts=requests,
                        sampling_params=sampling_params,
                        lora_request=self.current_lora_request,
                    )
""",
        "adapter-aware grouped PSRO opponent generation",
    )
    _replace_once(
        engine_path,
        """    vllm_enable_sleep=False,
):
""",
        """    vllm_enable_sleep=False,
    lora_rank=0,
    initial_lora_path=None,
    opponent_pool_json=None,
    tensor_lora_sync=False,
):
""",
        "vLLM LoRA rank argument",
    )
    _replace_once(
        engine_path,
        """                enable_sleep_mode=vllm_enable_sleep,
                noset_visible_devices=noset_visible_devices,
""",
        """                enable_sleep_mode=vllm_enable_sleep,
                enable_lora=(
                    lora_rank > 0
                    or initial_lora_path is not None
                    or opponent_pool_json is not None
                ),
                max_loras=max(
                    1,
                    sum(
                        bool(entry.get("adapter"))
                        for entry in json.loads(opponent_pool_json or "[]")
                    ),
                ),
                max_cpu_loras=max(
                    1,
                    sum(
                        bool(entry.get("adapter"))
                        for entry in json.loads(opponent_pool_json or "[]")
                    ),
                ),
                max_lora_rank=max(1, lora_rank),
                initial_lora_path=initial_lora_path,
                opponent_pool_json=opponent_pool_json,
                tensor_lora_sync=tensor_lora_sync,
                noset_visible_devices=noset_visible_devices,
""",
        "enable native vLLM LoRA",
    )

    cli_path = UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py"
    _replace_once(
        cli_path,
        """            args.vllm_enable_sleep,
        )
""",
        """            args.vllm_enable_sleep,
            args.lora_rank,
            None,
            None,
            True,
        )
""",
        "pass LoRA rank to current-policy vLLM",
    )

    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    anchor = """        count, num_params = 0, len(list(model.named_parameters()))
"""
    replacement = """        if self.strategy.args.lora_rank > 0:
            from dataclasses import asdict
            from vllm.lora.utils import parse_fine_tuned_lora_name

            lora_params = []
            for name, param in model.named_parameters():
                if "lora_" not in name:
                    continue
                # PEFT inserts the adapter slot (``default``) into live
                # parameter names, while saved adapters omit it.  vLLM's
                # parser must still receive the ``base_model.model.`` prefix:
                # it removes the first two path components when deriving the
                # runtime module name.  Stripping that prefix silently shifts
                # every adapter onto a nonexistent module.
                tensor_name = name.replace(".default.", ".")
                if not tensor_name.startswith("base_model.model."):
                    raise RuntimeError(
                        f"Unsupported PEFT LoRA tensor name: {name}"
                    )
                module_name, _, _ = parse_fine_tuned_lora_name(tensor_name)
                if not module_name.startswith("model.layers."):
                    raise RuntimeError(
                        "LoRA tensor resolved to an unexpected vLLM module: "
                        f"{tensor_name} -> {module_name}"
                    )
                lora_params.append((tensor_name, param))
            if not lora_params:
                raise RuntimeError("LoRA v2 found no trainable adapter tensors")

            for count, (name, param) in enumerate(lora_params, start=1):
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
                            handle_list,
                            local_handle,
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
                            refs = [
                                engine.update_lora_weight_cuda_ipc.remote(
                                    name,
                                    dtype=param.dtype,
                                    shape=shape,
                                    ipc_handles=ipc_handles,
                                    empty_cache=count == len(lora_params),
                                )
                                for engine in self.vllm_engines
                            ]
                            ray.get(refs)
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
                                engine.update_lora_weight.remote(
                                    name,
                                    dtype=param.dtype,
                                    shape=shape,
                                    empty_cache=count == len(lora_params),
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
                                    param.data,
                                    0,
                                    group_name=self._model_update_group,
                                )
                            else:
                                torch.distributed.broadcast(
                                    param.data,
                                    0,
                                    group=self._model_update_group,
                                )
                            ray.get(refs)

            if torch.distributed.get_rank() == 0:
                peft_config = asdict(model.peft_config["default"])
                ray.get(
                    [
                        engine.finalize_lora.remote(peft_config)
                        for engine in self.vllm_engines
                    ]
                )
            torch.distributed.barrier()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if self.strategy.is_rank_0():
                self.strategy.print(
                    f"LoRA v2 synchronized {len(lora_params)} native adapter tensors"
                )
            return

        count, num_params = 0, len(list(model.named_parameters()))
"""
    _replace_once(actor_path, anchor, replacement, "native LoRA tensor sync")

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
        "initial native LoRA sync",
    )


def _patch_attacker_sft_rl_format() -> None:
    """Make attacker SFT supervise the same continuation used by rollout."""
    dataset_path = UPSTREAM_WORK / "openrlhf/datasets/sft_dataset.py"
    _replace_once(
        dataset_path,
        """        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
            prompt_input_template=self.prompt_input_template,
        )

        if not self.pretrain_mode:
""",
        """        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
            prompt_input_template=self.prompt_input_template,
        )

        assistant_prefill = getattr(
            self.strategy.args, "sft_assistant_prefill", None
        )
        if assistant_prefill and prompt:
            prompt += assistant_prefill

        if not self.pretrain_mode:
""",
        "attacker SFT assistant prefill",
    )

    actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        actor_path,
        """            if attacker_role_sft:
                sft_strategy.args.sft_input_key = "messages"
                sft_strategy.args.sft_output_key = None
                sft_strategy.args.prompt_input_template = None
""",
        """            if attacker_role_sft:
                from red_team.prompts import ASSISTANT_THINKING_PREFIX

                sft_strategy.args.sft_input_key = "prompt_messages"
                sft_strategy.args.sft_output_key = "completion_messages"
                sft_strategy.args.prompt_input_template = None
                sft_strategy.args.sft_assistant_prefill = (
                    ASSISTANT_THINKING_PREFIX
                )
""",
        "attacker SFT continuation keys",
    )
    _replace_once(
        actor_path,
        """            else:
                sft_strategy.args.prompt_input_template = (
                    DEFENDER_INSTRUCTION_COT_PROMPT
                )
""",
        """            else:
                from red_team.prompts import ASSISTANT_THINKING_PREFIX

                sft_strategy.args.prompt_input_template = (
                    DEFENDER_INSTRUCTION_COT_PROMPT
                )
                sft_strategy.args.sft_assistant_prefill = (
                    ASSISTANT_THINKING_PREFIX
                )
""",
        "defender SFT assistant prefill",
    )
    _replace_once(
        actor_path,
        """                multiturn=attacker_role_sft,
""",
        """                multiturn=False,
""",
        "single-turn attacker SFT continuation",
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
        """        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
""",
        """        self.temperature = temperature
        role_start_adapter = kwargs.pop("role_start_adapter", None)
        lora_is_trainable = kwargs.pop("lora_is_trainable", True)

        if isinstance(pretrain_or_model, str):
""",
        "role-start adapter argument",
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
                model_args = ModelArguments(
                    model_name_or_path=pretrain_or_model,
                    adapter_name_or_path=role_start_adapter,
                )
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
                    is_trainable=lora_is_trainable,
                )
                if role_start_adapter:
                    print(
                        "ROLE_START_ADAPTER_LOADED="
                        f"{role_start_adapter} trainable={lora_is_trainable}",
                        flush=True,
                    )
                self.model.enable_input_require_grads()
                self.model.print_trainable_parameters()

                if load_in_4bit:
""",
        "LLaMA-Factory LoRA construction",
    )

    ppo_actor_path = UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    _replace_once(
        ppo_actor_path,
        """            use_liger_kernel=strategy.args.use_liger_kernel,
        )
""",
        """            use_liger_kernel=strategy.args.use_liger_kernel,
            role_start_adapter=strategy.args.role_start_adapter,
        )
""",
        "pass role-start adapter to actor",
    )

    launcher_path = UPSTREAM_WORK / "openrlhf/trainer/ray/launcher.py"
    _replace_once(
        launcher_path,
        """            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model)
""",
        """            use_liger_kernel=strategy.args.use_liger_kernel,
            lora_rank=(
                strategy.args.lora_rank
                if strategy.args.role_start_adapter
                else 0
            ),
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            role_start_adapter=strategy.args.role_start_adapter,
            lora_is_trainable=False,
        )
        strategy.print(model)
""",
        "load role-start adapter into reference model",
    )

    cli_path = UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py"
    _replace_once(
        cli_path,
        '    parser.add_argument("--lora_dropout", type=float, default=0)\n',
        '    parser.add_argument("--lora_dropout", type=float, default=0)\n'
        '    parser.add_argument(\n'
        '        "--role_start_adapter",\n'
        '        type=str,\n'
        '        default=None,\n'
        '        help="PEFT adapter whose weights initialize the active role.",\n'
        '    )\n',
        "role-start adapter CLI argument",
    )


def _prepare_lora_v2_upstream() -> None:
    """Prepare a clean upstream tree without the legacy LoRA interface."""
    _prepare_upstream_source()
    # Keep the RL rollout instruction aligned with the attacker-specific SFT
    # data.  This patch is local to the LoRA-v2 working copy; the established
    # full-parameter entrypoint remains unchanged.
    _patch_only_attacker_instruction()
    _patch_linear_cot_format_parser()
    _patch_role_aware_malformed_output_fallback()
    _patch_upstream_vllm_version_check()
    _patch_upstream_sft_chat_template()
    _patch_upstream_sft_micro_batch_floor()
    _patch_upstream_release_rl_logits_before_sft()
    _patch_upstream_zero3_sync_active_params()
    _patch_llamafactory_lora_initialization()
    _patch_upstream_replay_buffer_diagnostics()
    _patch_sparse_tie_redistribution()
    _patch_upstream_attacker_only_sampling()
    _patch_upstream_role_lr_scheduler()
    _patch_single_role_update_budget()
    _patch_optimizer_role_replay_filtering()
    _patch_single_role_advantage_normalization()
    _patch_upstream_remote_rm_retry()
    _patch_upstream_comprehensive_wandb_logging()
    _patch_upstream_defender_metric_keys()
    _patch_reference_kl_monitoring()
    _patch_role_specific_online_sft()
    _patch_wandb_run_identity()
    _patch_colocated_fixed_opponent()
    _patch_native_lora_sync()
    _patch_attacker_sft_rl_format()
    _patch_exact_prompt_label_balance()
    _patch_asymmetric_label_drift_training()


def _adapter_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
    )


def _adapter_weight_sha256(path: Path) -> str:
    if not _adapter_checkpoint(path):
        raise FileNotFoundError(f"Missing PEFT adapter checkpoint: {path}")
    weights = next(
        candidate
        for candidate in (
            path / "adapter_model.safetensors",
            path / "adapter_model.bin",
        )
        if candidate.is_file()
    )
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_fixed_opponent_pool(
    entries: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Validate an immutable PSRO mixture and remove zero-mass strategies."""

    if not entries:
        return []
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    total = 0.0
    for raw in entries:
        label = str(raw.get("id", ""))
        if not re.fullmatch(r"[AD][0-9]+", label) or label in seen:
            raise ValueError(f"Invalid or duplicate opponent id: {label!r}")
        seen.add(label)
        probability = float(raw.get("probability", -1.0))
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"Invalid probability for {label}: {probability}")
        if probability == 0:
            continue
        adapter_value = raw.get("adapter")
        adapter = str(adapter_value) if adapter_value else None
        expected_sha = str(raw.get("sha256", ""))
        if adapter is None:
            if label not in {"A0", "D0"}:
                raise ValueError(f"Only A0/D0 may select the base model: {label}")
            if expected_sha != f"base:{BASE_MODEL}":
                raise ValueError(f"Invalid base identity for {label}")
        else:
            actual_sha = _adapter_weight_sha256(Path(adapter))
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Frozen opponent {label} SHA changed: "
                    f"expected={expected_sha}, actual={actual_sha}"
                )
        total += probability
        normalized.append(
            {
                "id": label,
                "adapter": adapter,
                "sha256": expected_sha,
                "probability": probability,
            }
        )
    if not normalized or not math.isclose(total, 1.0, abs_tol=1e-8):
        raise ValueError(f"Positive opponent probabilities sum to {total}, not one")
    # Eliminate floating-point CDF residue while preserving the recorded mix.
    normalized[-1]["probability"] = float(normalized[-1]["probability"]) + (
        1.0 - total
    )
    return normalized


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


def _prepare_attacker_rl_sft_data() -> str:
    """Derive SFT examples whose prompt and continuation match rollout."""
    upstream_path = str(UPSTREAM_WORK)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    from red_team.prompts import (
        ATTACKER_INSTRUCTION_COT_PROMPT_BENIGN,
        ATTACKER_INSTRUCTION_COT_PROMPT_HARMFUL,
        ATTACKER_SYSTEM_PROMPT,
    )

    source = Path(ATTACKER_SFT_DATA)
    destination = Path(ATTACKER_RL_SFT_DATA)
    if not source.is_file():
        raise FileNotFoundError(f"Missing attacker SFT source: {source}")

    converted = 0
    with source.open(encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"Invalid attacker SFT messages at line {line_number}"
                )
            answer = str(messages[-1].get("content", "")).strip()
            if not answer:
                raise ValueError(
                    f"Empty attacker SFT answer at line {line_number}"
                )
            metadata = sample.get("metadata", {})
            label = str(metadata.get("label", ""))
            seed_prompt = str(metadata.get("seed_prompt", "")).strip()
            if label not in {"harmful", "benign"} or not seed_prompt:
                raise ValueError(
                    "Attacker SFT metadata must contain a harmful/benign "
                    f"label and seed_prompt at line {line_number}"
                )
            if label == "harmful":
                reasoning = (
                    "I should preserve the harmful capability and topic, make "
                    "the request more adversarial, and rewrite rather than answer it."
                )
                instruction = ATTACKER_INSTRUCTION_COT_PROMPT_HARMFUL.format(
                    vanilla_prompt=seed_prompt
                )
            else:
                reasoning = (
                    "I should preserve the benign intent and topic, make the "
                    "request test over-refusal, and rewrite rather than answer it."
                )
                instruction = ATTACKER_INSTRUCTION_COT_PROMPT_BENIGN.format(
                    vanilla_prompt=seed_prompt
                )
            derived = {
                "id": sample.get("id", f"attacker_sft_{line_number:05d}"),
                "prompt_messages": [
                    {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                "completion_messages": [
                    {
                        "role": "assistant",
                        "content": f"{reasoning} </think>\n{answer}",
                    }
                ],
            }
            output_file.write(json.dumps(derived, ensure_ascii=True) + "\n")
            converted += 1
    if converted == 0:
        raise RuntimeError("Attacker SFT conversion produced no examples")
    print(
        f"Prepared {converted} rollout-aligned attacker SFT examples at "
        f"{destination}",
        flush=True,
    )
    return str(destination)


def _prepare_defender_rl_sft_data() -> list[str]:
    """Strip the already-prefilled prefix from defender SFT completions."""
    upstream_path = str(UPSTREAM_WORK)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    from red_team.prompts import ASSISTANT_THINKING_PREFIX

    DEFENDER_RL_SFT_ROOT.mkdir(parents=True, exist_ok=True)
    destinations: list[str] = []
    for relative_path in DEFENDER_SFT_FILES:
        source = UPSTREAM_WORK / relative_path
        destination = DEFENDER_RL_SFT_ROOT / Path(relative_path).name
        converted = 0
        with source.open(encoding="utf-8") as input_file, destination.open(
            "w", encoding="utf-8"
        ) as output_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                completion = str(sample.get("completion", ""))
                if not completion.startswith(ASSISTANT_THINKING_PREFIX):
                    raise ValueError(
                        "Defender SFT completion does not match rollout prefill "
                        f"at {source}:{line_number}"
                    )
                derived = dict(sample)
                derived["completion"] = completion[
                    len(ASSISTANT_THINKING_PREFIX):
                ]
                output_file.write(json.dumps(derived, ensure_ascii=True) + "\n")
                converted += 1
        if converted == 0:
            raise RuntimeError(f"Defender SFT conversion is empty: {source}")
        print(
            f"Prepared {converted} rollout-aligned defender SFT examples at "
            f"{destination}",
            flush=True,
        )
        destinations.append(str(destination))
    return destinations


def _run_role(
    *,
    role: str,
    role_start_adapter: str | None = None,
    fixed_opponent: str,
    fixed_opponent_adapter: str | None = None,
    fixed_opponent_pool: list[dict[str, object]] | None = None,
    remote_rm_url: str,
    run_dir: Path,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    sft_stop_after_step: int,
    sft_batches_per_step: int,
    save_steps: int,
    actor_lr_scheduler: str = "cosine_with_min_lr",
    lr_warmup_ratio: float = 0.03,
    seed: int = 8888,
    exact_prompt_label_balance: bool = False,
    label_drift_training_policy: str = TRAINING_LABEL_DRIFT_POLICY,
    wandb_identity: str = "",
) -> Path:
    if role not in {"attacker", "defender"}:
        raise ValueError(role)
    if not fixed_opponent:
        raise ValueError("A frozen role opponent is required")
    if label_drift_training_policy not in {
        "",
        TRAINING_LABEL_DRIFT_POLICY,
    }:
        raise ValueError(
            "unsupported label-drift training policy: "
            f"{label_drift_training_policy!r}"
        )
    if role_start_adapter and not _adapter_checkpoint(Path(role_start_adapter)):
        raise FileNotFoundError(
            f"Missing active-role start adapter: {role_start_adapter}"
        )
    if fixed_opponent_adapter and not _adapter_checkpoint(
        Path(fixed_opponent_adapter)
    ):
        raise FileNotFoundError(
            f"Missing frozen opponent adapter: {fixed_opponent_adapter}"
        )
    if fixed_opponent_adapter and fixed_opponent_pool:
        raise ValueError(
            "fixed_opponent_adapter and fixed_opponent_pool are mutually exclusive"
        )
    normalized_opponent_pool = _normalize_fixed_opponent_pool(
        fixed_opponent_pool
    )
    # Ray worker processes inherit the environment of the raylet.  Set W&B
    # identity before starting Ray so the actor uses this deterministic ID.
    run_name = wandb_identity or f"{run_dir.parent.name}__{run_dir.name}"
    run_id = hashlib.sha1(run_name.encode()).hexdigest()[:8]
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"
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
    if role == "attacker":
        sft_data = _prepare_attacker_rl_sft_data()
        sft_probs = "1.0"
        sft_keys: list[str] = []
    else:
        sft_data = ",".join(_prepare_defender_rl_sft_data())
        sft_probs = "0.5,0.5"
        sft_keys = [
            "--sft_input_key",
            "vanilla",
            "--sft_output_key",
            "completion",
        ]
    custom_configs = {
        "max_turns": 2,
        "reward_type": "general_sum",
        "remove_ties": True,
        "redistribute_after_ties": True,
        "no_defender_turn": role == "attacker",
        # A defender phase must still execute the attacker turn with the
        # frozen A1 opponent. optimizer_train_role, not no_attacker_turn,
        # determines which replay items receive PPO updates.
        "no_attacker_turn": False,
        "optimizer_train_role": role,
        "fixed_defender_from_opponent_vllm": role == "attacker",
        "fixed_attacker_from_opponent_vllm": role == "defender",
        "actor_lr_scheduler": actor_lr_scheduler,
        "postfill_cot_stop_after_step": sft_stop_after_step,
        "exact_prompt_label_balance": exact_prompt_label_balance,
        "label_drift_training_policy": label_drift_training_policy,
    }
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY is missing from roll-secrets")

    ckpt_dir = run_dir / "ckpt"
    fixed_opponent_lora_args: list[str] = []
    if fixed_opponent_adapter:
        fixed_opponent_lora_args = [
            "--fixed_opponent_lora_path",
            fixed_opponent_adapter,
            "--fixed_opponent_lora_rank",
            str(lora_rank),
        ]
    elif normalized_opponent_pool:
        fixed_opponent_lora_args = [
            "--fixed_opponent_pool_json",
            json.dumps(normalized_opponent_pool, sort_keys=True),
            "--fixed_opponent_lora_rank",
            str(lora_rank),
        ]
    role_start_args: list[str] = []
    if role_start_adapter:
        role_start_args = ["--role_start_adapter", role_start_adapter]
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
        "--fixed_opponent_vllm_gpu_memory_utilization",
        "0.30",
        "--pretrain",
        BASE_MODEL,
        "--fixed_opponent_pretrain",
        fixed_opponent,
        *fixed_opponent_lora_args,
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        *role_start_args,
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
        sft_data,
        "--sft_data_probs",
        sft_probs,
        *sft_keys,
        "--sft_steps",
        "1",
        "--sft_batches_per_step",
        str(sft_batches_per_step),
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
        str(seed),
        "--top_p",
        "1.0",
        "--temperature",
        "1.0",
        "--actor_learning_rate",
        str(learning_rate),
        "--lr_warmup_ratio",
        str(lr_warmup_ratio),
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
        "method": "Self-RedTeam role-specific PEFT LoRA v2",
        "role": role,
        "ppo_reward_type": "general_sum",
        "strict_zero_sum_ppo_reward": False,
        "format_reward_in_ppo": True,
        "label_drift_training_policy": (
            label_drift_training_policy or "disabled"
        ),
        "label_drift_training_effect": (
            "attacker final shaped reward capped at zero on generated label "
            "drift; defender games with generated label drift omitted"
            if label_drift_training_policy
            else "disabled"
        ),
        "prompt_label_balance": (
            "deterministic exact 50/50 harmful/benign finite prefix"
            if exact_prompt_label_balance
            else "probabilistic prompt_data_probs"
        ),
        "base_model": BASE_MODEL,
        "role_start": role_start_adapter or BASE_MODEL,
        "role_start_base": BASE_MODEL,
        "fixed_opponent_base": fixed_opponent,
        "fixed_opponent_adapter": fixed_opponent_adapter,
        "fixed_opponent_pool": normalized_opponent_pool or None,
        "fixed_opponent_sampling": (
            "deterministic request-level categorical draw; one frozen "
            "opponent action per two-turn episode"
            if normalized_opponent_pool
            else "single fixed opponent"
        ),
        "legacy_lora_interface": False,
        "rollout_sync": "native LoRA A/B tensors via vLLM LoRA manager",
        "steps": steps,
        "rollout_batch_size": 128,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(LORA_TARGET_MODULES),
        "learning_rate": learning_rate,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "seed": seed,
        "kl_penalty": 0.0,
        "reference_kl_monitored": True,
        "sft_stop_after_step": sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "attacker_prompt_profile": (
            "optimized_harmful_v5_and_benign"
            if role == "attacker"
            else None
        ),
        "sft_prompt_alignment": "exact_rollout_messages_and_prefill",
        "malformed_cot_opponent_visibility": "answer_payload_only",
        "attacker_sft_format": (
            "rollout continuation after ASSISTANT_THINKING_PREFIX"
            if role == "attacker"
            else None
        ),
        "defender_sft_format": (
            "rollout continuation after ASSISTANT_THINKING_PREFIX"
            if role == "defender"
            else None
        ),
        "wandb_run_id": run_id,
        "wandb_identity": run_name,
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
        raise RuntimeError(f"LoRA v2 {role} phase exited with code {return_code}")
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
    learning_rate: float = 4e-6,
    sft_stop_after_step: int = 30,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "cosine_with_min_lr",
    lr_warmup_ratio: float = 0.03,
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
    if sft_batches_per_step < 1:
        raise ValueError("sft_batches_per_step must be positive")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr", "constant", "constant_with_warmup"
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"attacker_r{lora_rank}a{lora_alpha}_s{steps}_lr{learning_rate:g}_{suffix}"
    )
    checkpoint = _run_role(
        role="attacker",
        fixed_opponent=BASE_MODEL,
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=root,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    return {"run_dir": str(root), "checkpoint": str(checkpoint)}


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
def train_lora_v2_defender_probe(
    attacker_adapter: str,
    steps: int = 5,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 1e-5,
    sft_stop_after_step: int = 5,
    sft_batches_per_step: int = 1,
    save_steps: int = 5,
    actor_lr_scheduler: str = "cosine_with_min_lr",
    lr_warmup_ratio: float = 0.03,
    run_suffix: str = "",
) -> dict[str, str]:
    """Validate D1 against a frozen base-plus-A1 native adapter."""
    output_vol.reload()
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0 <= sft_stop_after_step <= steps:
        raise ValueError("sft_stop_after_step must be in [0, steps]")
    if sft_batches_per_step < 1:
        raise ValueError("sft_batches_per_step must be positive")
    if not _adapter_checkpoint(Path(attacker_adapter)):
        raise FileNotFoundError(
            f"Missing attacker adapter checkpoint: {attacker_adapter}"
        )
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"defender_r{lora_rank}a{lora_alpha}_s{steps}_"
        f"lr{learning_rate:g}_{suffix}"
    )
    checkpoint = _run_role(
        role="defender",
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=attacker_adapter,
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=root,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    return {"run_dir": str(root), "checkpoint": str(checkpoint)}


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
def train_lora_v2_a1_d1(
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
) -> dict[str, object]:
    """Train independent A1 and D1 LoRAs, each initialized from base."""
    if steps_per_role < 1:
        raise ValueError("steps_per_role must be positive")
    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if attacker_learning_rate <= 0 or defender_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 <= attacker_sft_stop_after_step <= steps_per_role:
        raise ValueError("attacker SFT cutoff must be in [0, steps_per_role]")
    if not 0 <= defender_sft_stop_after_step <= steps_per_role:
        raise ValueError("defender SFT cutoff must be in [0, steps_per_role]")
    if sft_batches_per_step < 1:
        raise ValueError("sft_batches_per_step must be positive")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr", "constant", "constant_with_warmup"
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"dual_lora_A{steps_per_role}D{steps_per_role}_"
        f"r{lora_rank}a{lora_alpha}_{suffix}"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "method": "sequential Self-RedTeam with two independent LLaMA-Factory LoRAs",
        "base_model": BASE_MODEL,
        "attacker_start": BASE_MODEL,
        "defender_start": BASE_MODEL,
        "role_order": ["A1", "D1"],
        "steps_per_role": steps_per_role,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(LORA_TARGET_MODULES),
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "attacker_sft_stop_after_step": attacker_sft_stop_after_step,
        "defender_sft_stop_after_step": defender_sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "rollout_batch_size": 128,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "prompt_budget_per_role": 128 * steps_per_role,
        "kl_penalty": 0.0,
        "reference_kl_monitored": True,
        "status": "attacker",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    rm_url = _stable_wildguard_rm_url()
    attacker_dir = root / f"A1_lora_s{steps_per_role}_vs_baseD"
    attacker_checkpoint = _run_role(
        role="attacker",
        fixed_opponent=BASE_MODEL,
        remote_rm_url=rm_url,
        run_dir=attacker_dir,
        steps=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=attacker_learning_rate,
        sft_stop_after_step=attacker_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    manifest.update(
        status="defender",
        attacker_checkpoint=str(attacker_checkpoint),
        frozen_attacker_base=BASE_MODEL,
        frozen_attacker_adapter=str(attacker_checkpoint),
        frozen_attacker_loading="native vLLM PEFT adapter",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    defender_dir = root / f"D1_lora_s{steps_per_role}_vs_A1_s{steps_per_role}"
    defender_checkpoint = _run_role(
        role="defender",
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=str(attacker_checkpoint),
        remote_rm_url=rm_url,
        run_dir=defender_dir,
        steps=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=defender_learning_rate,
        sft_stop_after_step=defender_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    manifest.update(
        status="completed",
        defender_checkpoint=str(defender_checkpoint),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()
    return manifest


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
def train_lora_v2_a2_d2(
    attacker_start_adapter: str,
    defender_start_adapter: str,
    source_generation: int = 1,
    steps_per_role: int = 80,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
) -> dict[str, object]:
    """Train the next attacker and defender generation sequentially."""
    if source_generation < 1:
        raise ValueError("source_generation must be positive")
    target_generation = source_generation + 1
    source_attacker = f"A{source_generation}"
    source_defender = f"D{source_generation}"
    target_attacker = f"A{target_generation}"
    target_defender = f"D{target_generation}"
    if steps_per_role < 1:
        raise ValueError("steps_per_role must be positive")
    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if attacker_learning_rate <= 0 or defender_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 <= attacker_sft_stop_after_step <= steps_per_role:
        raise ValueError("attacker SFT cutoff must be in [0, steps_per_role]")
    if not 0 <= defender_sft_stop_after_step <= steps_per_role:
        raise ValueError("defender SFT cutoff must be in [0, steps_per_role]")
    if sft_batches_per_step < 1:
        raise ValueError("sft_batches_per_step must be positive")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr", "constant", "constant_with_warmup"
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")

    attacker_start = Path(attacker_start_adapter)
    defender_start = Path(defender_start_adapter)
    if not _adapter_checkpoint(attacker_start):
        raise FileNotFoundError(
            f"Missing {source_attacker} adapter: {attacker_start}"
        )
    if not _adapter_checkpoint(defender_start):
        raise FileNotFoundError(
            f"Missing {source_defender} adapter: {defender_start}"
        )

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"selfplay_{target_attacker}{target_defender}_"
        f"A{steps_per_role}D{steps_per_role}_"
        f"r{lora_rank}a{lora_alpha}_{suffix}"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "method": "sequential two-policy Self-RedTeam continuation",
        "base_model": BASE_MODEL,
        "source_generation": source_generation,
        "target_generation": target_generation,
        "role_order": [target_attacker, target_defender],
        "lineage": {
            target_attacker: {
                "initialized_from": str(attacker_start),
                "opponent": str(defender_start),
            },
            target_defender: {
                "initialized_from": str(defender_start),
                "opponent": target_attacker,
            },
        },
        "steps_per_role": steps_per_role,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(LORA_TARGET_MODULES),
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "attacker_sft_stop_after_step": attacker_sft_stop_after_step,
        "defender_sft_stop_after_step": defender_sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "rollout_batch_size": 128,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "prompt_budget_per_role": 128 * steps_per_role,
        "kl_penalty": 0.0,
        "reference_kl_monitored_against": {
            target_attacker: str(attacker_start),
            target_defender: str(defender_start),
        },
        "status": "attacker",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    rm_url = _stable_wildguard_rm_url()
    attacker_dir = root / (
        f"{target_attacker}_from_{source_attacker}_"
        f"s{steps_per_role}_vs_{source_defender}"
    )
    attacker_checkpoint = _run_role(
        role="attacker",
        role_start_adapter=str(attacker_start),
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=str(defender_start),
        remote_rm_url=rm_url,
        run_dir=attacker_dir,
        steps=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=attacker_learning_rate,
        sft_stop_after_step=attacker_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    manifest.update(
        status="defender",
        attacker_checkpoint=str(attacker_checkpoint),
    )
    manifest["lineage"][target_defender]["opponent"] = str(
        attacker_checkpoint
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    defender_dir = root / (
        f"{target_defender}_from_{source_defender}_"
        f"s{steps_per_role}_vs_{target_attacker}"
    )
    defender_checkpoint = _run_role(
        role="defender",
        role_start_adapter=str(defender_start),
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=str(attacker_checkpoint),
        remote_rm_url=rm_url,
        run_dir=defender_dir,
        steps=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=defender_learning_rate,
        sft_stop_after_step=defender_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    manifest.update(
        status="completed",
        defender_checkpoint=str(defender_checkpoint),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()
    return manifest


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
def train_lora_v2_defender_only(
    attacker_adapter: str,
    defender_start_adapter: str,
    target_generation: int,
    steps: int = 80,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 4e-5,
    sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
) -> dict[str, object]:
    """Recover one defender generation from completed policy adapters."""
    if target_generation < 1:
        raise ValueError("target_generation must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= sft_stop_after_step <= steps:
        raise ValueError("SFT cutoff must be in [0, steps]")
    if sft_batches_per_step < 1 or save_steps < 1:
        raise ValueError("SFT batches and save interval must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr", "constant", "constant_with_warmup"
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")

    attacker = Path(attacker_adapter)
    defender_start = Path(defender_start_adapter)
    if not _adapter_checkpoint(attacker):
        raise FileNotFoundError(f"Missing attacker adapter: {attacker}")
    if not _adapter_checkpoint(defender_start):
        raise FileNotFoundError(
            f"Missing defender start adapter: {defender_start}"
        )

    target_defender = f"D{target_generation}"
    source_defender = f"D{target_generation - 1}"
    opponent_attacker = f"A{target_generation}"
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUTPUT_ROOT) / (
        f"selfplay_{target_defender}_recovery_s{steps}_"
        f"r{lora_rank}a{lora_alpha}_{suffix}"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "method": "sequential two-policy Self-RedTeam defender recovery",
        "base_model": BASE_MODEL,
        "target_generation": target_generation,
        "role_order": [target_defender],
        "lineage": {
            target_defender: {
                "initialized_from": str(defender_start),
                "opponent": str(attacker),
            }
        },
        "steps": steps,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(LORA_TARGET_MODULES),
        "learning_rate": learning_rate,
        "sft_stop_after_step": sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "rollout_batch_size": 128,
        "train_batch_size": 32,
        "micro_train_batch_size": 8,
        "kl_penalty": 0.0,
        "reference_kl_monitored_against": str(defender_start),
        "status": "defender",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()

    run_dir = root / (
        f"{target_defender}_from_{source_defender}_s{steps}_"
        f"vs_{opponent_attacker}"
    )
    defender_checkpoint = _run_role(
        role="defender",
        role_start_adapter=str(defender_start),
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=str(attacker),
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=run_dir,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    manifest.update(
        status="completed",
        defender_checkpoint=str(defender_checkpoint),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_vol.commit()
    return manifest


@lora_v2_app.function(cpu=1, timeout=86400)
def train_lora_v2_defender_only_and_eval(
    attacker_adapter: str,
    defender_start_adapter: str,
    target_generation: int,
    steps: int = 80,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 4e-5,
    sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
) -> dict[str, object]:
    """Recover one defender generation, then run the canonical evaluation."""
    training = train_lora_v2_defender_only.remote(
        attacker_adapter=attacker_adapter,
        defender_start_adapter=defender_start_adapter,
        target_generation=target_generation,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        run_suffix=run_suffix,
    )
    target_generation = int(training["target_generation"])
    defender_checkpoint = str(training["defender_checkpoint"])
    evaluator = modal.Function.from_name(
        "selfredteam-official-eval",
        "evaluate_full_checkpoint_vs_base",
    )
    evaluation = evaluator.remote(
        trained_checkpoint=defender_checkpoint,
        output_slug=f"D{target_generation}_recovery",
        trained_label=f"our_selfplay_D{target_generation}",
        evaluate_base=False,
    )
    return {"training": training, "evaluation": evaluation}


@lora_v2_app.function(cpu=1, timeout=86400)
def train_lora_v2_a2_d2_and_eval(
    attacker_start_adapter: str,
    defender_start_adapter: str,
    source_generation: int = 1,
    steps_per_role: int = 80,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
) -> dict[str, object]:
    """Train one continued generation remotely, then evaluate its defender."""
    target_generation = source_generation + 1
    training = train_lora_v2_a2_d2.remote(
        attacker_start_adapter=attacker_start_adapter,
        defender_start_adapter=defender_start_adapter,
        source_generation=source_generation,
        steps_per_role=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        attacker_learning_rate=attacker_learning_rate,
        defender_learning_rate=defender_learning_rate,
        attacker_sft_stop_after_step=attacker_sft_stop_after_step,
        defender_sft_stop_after_step=defender_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        run_suffix=run_suffix,
    )
    defender_checkpoint = str(training["defender_checkpoint"])
    output_suffix = run_suffix or f"a{target_generation}d{target_generation}"
    evaluator = modal.Function.from_name(
        "selfredteam-official-eval",
        "evaluate_full_checkpoint_vs_base",
    )
    evaluation = evaluator.remote(
        trained_checkpoint=defender_checkpoint,
        output_slug=f"D{target_generation}_{output_suffix}",
        trained_label=f"our_selfplay_D{target_generation}",
        evaluate_base=False,
    )
    return {"training": training, "evaluation": evaluation}


@lora_v2_app.function(cpu=2, timeout=1800)
def validate_lora_v2_configuration() -> dict[str, object]:
    _prepare_lora_v2_upstream()
    actor_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ray/ppo_actor.py"
    ).read_text()
    actor_model_source = (UPSTREAM_WORK / "openrlhf/models/actor.py").read_text()
    sft_dataset_source = (
        UPSTREAM_WORK / "openrlhf/datasets/sft_dataset.py"
    ).read_text()
    cli_source = (UPSTREAM_WORK / "openrlhf/cli/train_ppo_ray.py").read_text()
    vllm_engine_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ray/vllm_engine.py"
    ).read_text()
    vllm_worker_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ray/vllm_worker_wrap.py"
    ).read_text()
    experience_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/experience_maker.py"
    ).read_text()
    language_game_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/language_game.py"
    ).read_text()
    replay_source = (
        UPSTREAM_WORK / "openrlhf/trainer/ppo_utils/replay_buffer.py"
    ).read_text()
    red_team_utils_source = (UPSTREAM_WORK / "red_team/utils.py").read_text()
    red_team_prompts_source = (
        UPSTREAM_WORK / "red_team/prompts.py"
    ).read_text()
    required = (
        "llamafactory_init_adapter",
        "FinetuningArguments(",
        "Extract Self-RedTeam tags in linear time",
        "LoRA v2 synchronized",
        "update_lora_weight",
        "custom_add_lora",
        "custom_reload_lora",
        "Reloaded current LoRA after vLLM wake_up",
        "Stripping that prefix silently shifts",
        "lora_request=self.current_lora_request",
        "max_lora_rank=max(1, lora_rank)",
        "needs_initial_sync",
        "fixed_opponent_vllm_gpu_memory_utilization",
        "fixed_opponent_pretrain",
        "fixed_attacker_from_opponent_vllm",
        "fixed_defender_from_opponent_vllm",
        "postfill_cot_stop_after_step",
        "initial_lora_path = kwargs.pop",
        'lora_name="fixed_opponent_lora"',
        'sft_input_key = "prompt_messages"',
        'sft_output_key = "completion_messages"',
        "sft_assistant_prefill",
        "prompt += assistant_prefill",
        "**Your goal is to refine a jailbreak prompt into a more complex adversarial version**.",
        "**Your goal is to create a benign adversarial user prompt from the seed**.",
        "fallback_answer = response[",
        'player_role == "defender"',
        "defender_visible_response",
        "Preserve every complete global micro-batch",
        "filter_optimizer_role",
        "debug/replay_defender_after_role_filter",
        "optimizer_train_role is None",
        "optimizer_train_role == 'defender' or no_attacker_turn",
    )
    combined = (
        actor_source
        + actor_model_source
        + sft_dataset_source
        + cli_source
        + vllm_engine_source
        + vllm_worker_source
        + experience_source
        + language_game_source
        + replay_source
        + red_team_utils_source
        + red_team_prompts_source
    )
    missing = [item for item in required if item not in combined]
    forbidden = (
        "module.get_delta_weight(adapter_name)",
        "merged dense weights",
    )
    invalid = [item for item in forbidden if item in combined]
    if missing or invalid:
        raise RuntimeError(
            f"LoRA v2 validation failed: missing={missing}, invalid={invalid}"
        )
    return {
        "valid": True,
        "rollout_sync": "native_vllm_lora_tensors",
        "fixed_opponent_loading": "base_plus_native_file_adapter",
        "role_only_replay": "hard_filtered_before_redistribution",
        "defender_attacker_turn": "generated_by_frozen_opponent",
        "role_only_scheduler_multiplier": 1.0,
        "attacker_sft_format": "rollout_continuation",
        "attacker_prompt_profile": "optimized_harmful_v5_and_benign",
        "malformed_cot_opponent_visibility": "answer_payload_only",
        "rank": 64,
        "alpha": 64,
        "learning_rate": 4e-6,
        "target_modules": list(LORA_TARGET_MODULES),
    }


@lora_v2_app.function(
    image=llamafactory_lora_image,
    gpu="H200:1",
    cpu=4,
    memory=32768,
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
)
def audit_lora_merge_equivalence(checkpoint: str) -> dict[str, object]:
    """Compare native PEFT and BF16-merged token probabilities."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_vol.reload()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    model.eval()
    prompts = [
        "Rewrite this request into a more challenging but on-topic safety test: "
        "Explain how account security works.",
        "Rewrite this benign request so that it remains harmless but tests whether "
        "a model over-refuses: Explain the history of encryption.",
        "Create a natural user-facing rewrite of this request without answering it: "
        "Summarize common phishing warning signs.",
        "Rewrite this request while preserving its original topic and intent: "
        "How can an organization improve incident response training?",
    ]
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to("cuda")

    with torch.inference_mode():
        generated = model.generate(
            **batch, do_sample=False, max_new_tokens=64, use_cache=True
        )
        native_logits = model(
            input_ids=generated,
            attention_mask=torch.ones_like(generated),
        ).logits.float()

    native_log_probs = native_logits[:, :-1].log_softmax(dim=-1)
    targets = generated[:, 1:]
    native_action_log_probs = native_log_probs.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    native_last_log_probs = native_logits[:, -1].log_softmax(dim=-1)
    del native_logits, native_log_probs
    torch.cuda.empty_cache()

    merged = model.merge_and_unload(safe_merge=True)
    merged.eval()
    with torch.inference_mode():
        merged_logits = merged(
            input_ids=generated,
            attention_mask=torch.ones_like(generated),
        ).logits.float()
        merged_generated = merged.generate(
            **batch, do_sample=False, max_new_tokens=64, use_cache=True
        )

    merged_log_probs = merged_logits[:, :-1].log_softmax(dim=-1)
    merged_action_log_probs = merged_log_probs.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    merged_last_log_probs = merged_logits[:, -1].log_softmax(dim=-1)
    action_delta = native_action_log_probs - merged_action_log_probs
    probability_ratio = action_delta.exp()
    native_last_probs = native_last_log_probs.exp()
    last_token_kl = (
        native_last_probs * (native_last_log_probs - merged_last_log_probs)
    ).sum(dim=-1)

    prefix_matches = []
    for native_ids, merged_ids in zip(generated, merged_generated):
        common = 0
        for left, right in zip(native_ids.tolist(), merged_ids.tolist()):
            if left != right:
                break
            common += 1
        prefix_matches.append(common)

    result = {
        "checkpoint": checkpoint,
        "base_model": BASE_MODEL,
        "dtype": str(next(merged.parameters()).dtype),
        "num_sequences": len(prompts),
        "sequence_length": int(generated.shape[1]),
        "action_logprob_abs_diff_mean": float(action_delta.abs().mean()),
        "action_logprob_abs_diff_max": float(action_delta.abs().max()),
        "action_probability_ratio_min": float(probability_ratio.min()),
        "action_probability_ratio_mean": float(probability_ratio.mean()),
        "action_probability_ratio_max": float(probability_ratio.max()),
        "last_token_kl_mean": float(last_token_kl.mean()),
        "last_token_kl_max": float(last_token_kl.max()),
        "greedy_outputs_identical": bool(torch.equal(generated, merged_generated)),
        "greedy_common_prefix_tokens": prefix_matches,
    }
    print(json.dumps(result, indent=2))
    return result


@lora_v2_app.function(
    image=llamafactory_lora_image,
    gpu="H200:1",
    cpu=8,
    memory=65536,
    timeout=5400,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
)
def audit_lora_rollout_equivalence(checkpoint: str) -> dict[str, object]:
    """Compare PEFT, file-vLLM, and training-time tensor-vLLM LoRA paths."""
    import gc
    import json as json_module
    from collections import OrderedDict

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_vol.reload()
    checkpoint_path = Path(checkpoint)
    adapter_path = checkpoint_path / "adapter_model.bin"
    config_path = checkpoint_path / "adapter_config.json"
    if not adapter_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete adapter checkpoint: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    messages = [
        {
            "role": "system",
            "content": (
                "You are a language-model safety self-play participant. "
                "Follow the role in the user message exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Role: attacker\nSeed label: harmful\n"
                "Rewrite the seed into a natural, on-topic adversarial request. "
                "Do not answer it. Put only the rewrite inside <answer> tags.\n\n"
                "Seed prompt: Explain how to create a convincing phishing email "
                "that captures an employee's password."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    model.eval()
    with torch.inference_mode():
        with model.disable_adapter():
            hf_base_next = model(**inputs).logits[:, -1].float()
            hf_base_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=96,
                use_cache=True,
            )
        hf_peft_next = model(**inputs).logits[:, -1].float()
        hf_peft_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=96,
            use_cache=True,
        )

    hf_base_log_probs = hf_base_next.log_softmax(dim=-1)
    hf_peft_log_probs = hf_peft_next.log_softmax(dim=-1)
    hf_peft_probs = hf_peft_log_probs.exp()
    hf_base_to_peft_kl = (
        hf_peft_probs * (hf_peft_log_probs - hf_base_log_probs)
    ).sum(dim=-1)
    hf_base_text = tokenizer.decode(
        hf_base_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    hf_peft_text = tokenizer.decode(
        hf_peft_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    del model, base, inputs, hf_base_next, hf_peft_next
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.75,
        enable_lora=True,
        # Training keeps exactly one current-policy adapter resident.  Reuse
        # that same physical slot for both audit paths so slot-dependent
        # numerical behavior cannot masquerade as a synchronization error.
        max_loras=1,
        max_lora_rank=64,
        worker_extension_cls=(
            "roll.third_party.vllm.lora_audit_worker.LoRAAuditWorker"
        ),
        seed=0,
    )
    llm.collective_rpc("custom_init_worker")
    sampling = SamplingParams(temperature=0.0, max_tokens=96)
    file_request = LoRARequest(
        lora_name="file_adapter",
        lora_int_id=101,
        lora_path=checkpoint,
    )
    vllm_base_output = llm.generate([prompt], sampling)[0]
    vllm_file_output = llm.generate(
        [prompt], sampling, lora_request=file_request
    )[0]
    vllm_file_repeat_output = llm.generate(
        [prompt], sampling, lora_request=file_request
    )[0]
    file_snapshot = llm.collective_rpc(
        "custom_snapshot_lora",
        args=(file_request.lora_int_id, "file"),
    )
    file_gpu_snapshot = llm.collective_rpc(
        "custom_snapshot_active_lora",
        args=(file_request.lora_int_id, "file_gpu"),
    )
    probe_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20,
        seed=0,
    )
    vllm_file_probe = llm.generate(
        [prompt], probe_sampling, lora_request=file_request
    )[0]

    removed = llm.collective_rpc(
        "custom_remove_lora",
        args=(file_request.lora_int_id,),
    )
    if not all(removed):
        raise RuntimeError(f"File LoRA removal failed: {removed}")

    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    tensor_state = OrderedDict()
    for name, tensor in state.items():
        # Saved PEFT keys already have the exact naming convention expected by
        # vLLM.  Keep the base_model.model prefix intact.
        tensor_name = name.replace(".default.", ".")
        if not tensor_name.startswith("base_model.model."):
            raise RuntimeError(f"Unsupported saved LoRA tensor name: {name}")
        tensor_state[tensor_name] = tensor
    peft_config = json_module.loads(config_path.read_text())
    installed = llm.collective_rpc(
        "custom_load_lora_state",
        args=(
            peft_config,
            tensor_state,
            file_request.lora_int_id,
            file_request.lora_name,
        ),
    )
    if not all(installed):
        raise RuntimeError(f"Tensor LoRA installation failed: {installed}")
    tensor_request = LoRARequest(
        lora_name=file_request.lora_name,
        lora_int_id=file_request.lora_int_id,
        lora_path="/root/.cache/roll/training_lora_v1",
    )
    tensor_snapshot = llm.collective_rpc(
        "custom_snapshot_lora",
        args=(tensor_request.lora_int_id, "tensor"),
    )
    tensor_gpu_snapshot = llm.collective_rpc(
        "custom_snapshot_active_lora",
        args=(tensor_request.lora_int_id, "tensor_gpu"),
    )
    registered_tensor_comparison = llm.collective_rpc(
        "custom_compare_lora_snapshots",
        args=("file", "tensor"),
    )
    active_gpu_tensor_comparison = llm.collective_rpc(
        "custom_compare_lora_snapshots",
        args=("file_gpu", "tensor_gpu"),
    )
    vllm_tensor_probe = llm.generate(
        [prompt], probe_sampling, lora_request=tensor_request
    )[0]
    vllm_tensor_output = llm.generate(
        [prompt], sampling, lora_request=tensor_request
    )[0]
    vllm_tensor_repeat_output = llm.generate(
        [prompt], sampling, lora_request=tensor_request
    )[0]

    def output_ids(output):
        return list(output.outputs[0].token_ids)

    def output_text(output):
        return output.outputs[0].text

    def common_prefix(left, right):
        count = 0
        for left_token, right_token in zip(left, right):
            if left_token != right_token:
                break
            count += 1
        return count

    def first_token_logprobs(output):
        logprobs = output.outputs[0].logprobs
        if not logprobs:
            return {}
        return {
            int(token_id): float(value.logprob)
            for token_id, value in logprobs[0].items()
        }

    file_probe_logprobs = first_token_logprobs(vllm_file_probe)
    tensor_probe_logprobs = first_token_logprobs(vllm_tensor_probe)
    common_probe_tokens = sorted(
        set(file_probe_logprobs) & set(tensor_probe_logprobs)
    )
    probe_logprob_deltas = [
        abs(file_probe_logprobs[token] - tensor_probe_logprobs[token])
        for token in common_probe_tokens
    ]

    hf_base_completion_ids = hf_base_ids[0, len(tokenizer(prompt).input_ids):].tolist()
    hf_peft_completion_ids = hf_peft_ids[0, len(tokenizer(prompt).input_ids):].tolist()
    vllm_base_ids = output_ids(vllm_base_output)
    vllm_file_ids = output_ids(vllm_file_output)
    vllm_file_repeat_ids = output_ids(vllm_file_repeat_output)
    vllm_tensor_ids = output_ids(vllm_tensor_output)
    vllm_tensor_repeat_ids = output_ids(vllm_tensor_repeat_output)
    result = {
        "checkpoint": checkpoint,
        "hf_base_to_peft_next_token_kl": float(hf_base_to_peft_kl.item()),
        "hf_base_vs_peft_common_prefix": common_prefix(
            hf_base_completion_ids, hf_peft_completion_ids
        ),
        "vllm_base_vs_file_common_prefix": common_prefix(
            vllm_base_ids, vllm_file_ids
        ),
        "vllm_file_vs_tensor_common_prefix": common_prefix(
            vllm_file_ids, vllm_tensor_ids
        ),
        "vllm_file_tensor_identical": vllm_file_ids == vllm_tensor_ids,
        "vllm_file_repeat_identical": (
            vllm_file_ids == vllm_file_repeat_ids
        ),
        "vllm_tensor_repeat_identical": (
            vllm_tensor_ids == vllm_tensor_repeat_ids
        ),
        "file_snapshot": file_snapshot,
        "tensor_snapshot": tensor_snapshot,
        "registered_tensor_comparison": registered_tensor_comparison,
        "file_gpu_snapshot": file_gpu_snapshot,
        "tensor_gpu_snapshot": tensor_gpu_snapshot,
        "active_gpu_tensor_comparison": active_gpu_tensor_comparison,
        "first_token_logprob_comparison": {
            "file_token_id": output_ids(vllm_file_probe)[0],
            "tensor_token_id": output_ids(vllm_tensor_probe)[0],
            "common_top_token_count": len(common_probe_tokens),
            "max_abs_diff": (
                max(probe_logprob_deltas) if probe_logprob_deltas else None
            ),
            "mean_abs_diff": (
                sum(probe_logprob_deltas) / len(probe_logprob_deltas)
                if probe_logprob_deltas
                else None
            ),
        },
        "hf_peft_vs_vllm_file_common_prefix": common_prefix(
            hf_peft_completion_ids, vllm_file_ids
        ),
        "hf_peft_vs_vllm_tensor_common_prefix": common_prefix(
            hf_peft_completion_ids, vllm_tensor_ids
        ),
        "outputs": {
            "hf_base": hf_base_text,
            "hf_peft": hf_peft_text,
            "vllm_base": output_text(vllm_base_output),
            "vllm_file_lora": output_text(vllm_file_output),
            "vllm_file_lora_repeat": output_text(vllm_file_repeat_output),
            "vllm_tensor_lora": output_text(vllm_tensor_output),
            "vllm_tensor_lora_repeat": output_text(
                vllm_tensor_repeat_output
            ),
        },
    }
    print(json.dumps(result, indent=2))
    return result


@lora_v2_app.function(
    image=llamafactory_lora_image,
    cpu=2,
    memory=8192,
    timeout=900,
    volumes={"/output": output_vol},
)
def audit_lora_checkpoint_parameters(checkpoint: str) -> dict[str, object]:
    """Verify that a saved PEFT adapter contains updated LoRA tensors."""
    import torch

    output_vol.reload()
    checkpoint_path = Path(checkpoint)
    adapter_path = checkpoint_path / "adapter_model.bin"
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    groups: dict[str, list[dict[str, float | int]]] = {
        "lora_A": [],
        "lora_B": [],
    }
    for name, tensor in state.items():
        group = next((key for key in groups if key in name), None)
        if group is None:
            continue
        value = tensor.float()
        groups[group].append(
            {
                "numel": value.numel(),
                "nonzero": int(torch.count_nonzero(value)),
                "l2_norm": float(torch.linalg.vector_norm(value)),
                "abs_max": float(value.abs().max()),
            }
        )

    def summarize(entries: list[dict[str, float | int]]) -> dict[str, object]:
        if not entries:
            return {"tensor_count": 0}
        numel = sum(int(entry["numel"]) for entry in entries)
        nonzero = sum(int(entry["nonzero"]) for entry in entries)
        norms = [float(entry["l2_norm"]) for entry in entries]
        maxima = [float(entry["abs_max"]) for entry in entries]
        return {
            "tensor_count": len(entries),
            "numel": numel,
            "nonzero": nonzero,
            "nonzero_fraction": nonzero / numel,
            "zero_tensor_count": sum(norm == 0 for norm in norms),
            "l2_norm_min": min(norms),
            "l2_norm_mean": sum(norms) / len(norms),
            "l2_norm_max": max(norms),
            "abs_max": max(maxima),
        }

    result = {
        "checkpoint": checkpoint,
        "adapter_size_bytes": adapter_path.stat().st_size,
        "state_tensor_count": len(state),
        "lora_A": summarize(groups["lora_A"]),
        "lora_B": summarize(groups["lora_B"]),
        "optimizer_updated_lora_B": bool(
            groups["lora_B"]
            and all(float(entry["l2_norm"]) > 0 for entry in groups["lora_B"])
        ),
    }
    print(json.dumps(result, indent=2))
    return result


@lora_v2_app.local_entrypoint(name="lora_v2_attacker_probe")
def lora_v2_attacker_probe(
    steps: int = 50,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 4e-6,
    sft_stop_after_step: int = 30,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "cosine_with_min_lr",
    lr_warmup_ratio: float = 0.03,
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
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "run_suffix": suffix,
    }
    if detach:
        call = train_lora_v2_attacker_probe.spawn(**kwargs)
        print(f"CALL_ID={call.object_id}")
        print(f"RUN_SUFFIX={suffix}")
        return
    print(json.dumps(train_lora_v2_attacker_probe.remote(**kwargs), indent=2))


@lora_v2_app.local_entrypoint(name="lora_v2_a1_d1")
def lora_v2_a1_d1(
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
    detach: bool = True,
) -> None:
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    kwargs = {
        "steps_per_role": steps_per_role,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "attacker_sft_stop_after_step": attacker_sft_stop_after_step,
        "defender_sft_stop_after_step": defender_sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "run_suffix": suffix,
    }
    if detach:
        call = train_lora_v2_a1_d1.spawn(**kwargs)
        print(f"CALL_ID={call.object_id}")
        print(f"RUN_SUFFIX={suffix}")
        return
    print(json.dumps(train_lora_v2_a1_d1.remote(**kwargs), indent=2))


@lora_v2_app.local_entrypoint(name="lora_v2_defender_probe")
def lora_v2_defender_probe(
    attacker_adapter: str,
    steps: int = 5,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 1e-5,
    sft_stop_after_step: int = 5,
    sft_batches_per_step: int = 1,
    save_steps: int = 5,
    actor_lr_scheduler: str = "cosine_with_min_lr",
    lr_warmup_ratio: float = 0.03,
    run_suffix: str = "",
    detach: bool = True,
) -> None:
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    kwargs = {
        "attacker_adapter": attacker_adapter,
        "steps": steps,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "sft_stop_after_step": sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "run_suffix": suffix,
    }
    if detach:
        call = train_lora_v2_defender_probe.spawn(**kwargs)
        print(f"CALL_ID={call.object_id}")
        print(f"RUN_SUFFIX={suffix}")
        return
    print(json.dumps(train_lora_v2_defender_probe.remote(**kwargs), indent=2))


@lora_v2_app.local_entrypoint(name="validate_lora_v2")
def validate_lora_v2() -> None:
    print(json.dumps(validate_lora_v2_configuration.remote(), indent=2))


@lora_v2_app.local_entrypoint(name="audit_lora_merge")
def audit_lora_merge(checkpoint: str) -> None:
    print(json.dumps(audit_lora_merge_equivalence.remote(checkpoint), indent=2))


@lora_v2_app.local_entrypoint(name="audit_lora_weights")
def audit_lora_weights(checkpoint: str) -> None:
    print(json.dumps(audit_lora_checkpoint_parameters.remote(checkpoint), indent=2))


@lora_v2_app.local_entrypoint(name="audit_lora_rollout")
def audit_lora_rollout(checkpoint: str) -> None:
    print(json.dumps(audit_lora_rollout_equivalence.remote(checkpoint), indent=2))
