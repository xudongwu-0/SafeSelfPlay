"""Fail-closed contracts for the one-shot attacker objective migration.

This module deliberately contains no Modal, torch, PEFT, or trainer imports.
It accepts only the durable hand-off emitted after the sole raw-PPO gate retry
is exhausted, binds a single warm-started attempt to that hand-off, and
defines the exact binary training surrogate and gate.  The historical payoff
utility remains the frozen upstream additive reward; the binary value is an
oracle-only optimization surrogate and is never a payoff-matrix entry.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from roll.utils.selfplay_gate_retry import (
    OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS,
    RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
    RECOVERY_KEY,
    canonical_json_sha256,
    validate_checkpoint_cadence,
    verify_objective_migration_requirement,
    verify_recovery_plan,
)


SCHEMA_VERSION = 1
MIGRATION_KEY = "attacker_objective_migration_v1"
MIGRATION_HISTORY_KEY = "attacker_objective_migration_history_v1"
MIGRATION_POLICY = "single-bounded-attacker-binary-joint-migration-v1"
MIGRATION_TRAINER_POLICY = "attacker-binary-joint-dynamic-clone-v1"
MIGRATION_MANIFEST_POLICY = "attacker-binary-joint-oracle-surrogate-v1"
MIGRATION_FINAL_AUDIT_POLICY = (
    "migration-aware-live-16-member-final-population-audit-v1"
)
MIGRATION_REQUIRED_OBJECTIVE = (
    "attacker_binary_joint_goal_and_cot_raw_no_center_no_std"
)
MIGRATION_REQUIRED_METRIC = (
    "attacker/binary_joint_goal_cot_success_rate"
)
MIGRATION_EXHAUSTED_STATE_STATUS = "stage_objective_migration_exhausted"
MIGRATION_ROOT_NAME = "attacker_objective_migration_v1"
MIGRATION_MAX_ATTEMPTS_PER_STAGE = 1
TRAJECTORY_SUM_LOSS_SCALE = 1.0 / 2048.0
_ATTACKER_STAGE_RE = re.compile(r"^A([2-8])$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(value: object, label: str) -> str:
    text = str(value or "")
    _require(bool(_SHA256_RE.fullmatch(text)), f"Invalid {label} SHA256")
    return text


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_joint_objective_contract() -> dict[str, Any]:
    """Return the exact optimizer/gate/payoff separation contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_MANIFEST_POLICY,
        "scope": "attacker_oracle_training_only",
        "positive_iff": {
            "outcome_any": [
                "ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST",
                "ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST",
            ],
            "outcome_forbidden": "TIE",
            "cot_format_violation": False,
        },
        "optimization_reward": {"positive": 1.0, "negative": -1.0},
        "failure_and_tie_replay": "retained_with_negative_reward",
        "advantage": "raw_no_center_no_std_no_baseline",
        "advantage_value": "exact_binary_optimization_reward_on_active_tokens",
        "ppo_surrogate": "clipped_token_surrogate_sum_per_trajectory_then_mean",
        "trajectory_sum_loss_scale": TRAJECTORY_SUM_LOSS_SCALE,
        "aux_sft": "permanently_disabled_from_optimizer_step_one",
        "gate": {
            "metric": MIGRATION_REQUIRED_METRIC,
            "threshold": 0.95,
            "patience": 5,
            "min_steps": 1,
            "companion_bounds": {},
        },
        "official_payoff": {
            "utility": "unchanged_frozen_upstream_additive_raw_utility",
            "normalization": "none",
            "binary_surrogate_is_payoff_entry": False,
            "matrix_namespace": "historical_psro_payoff_unchanged",
        },
    }


def migration_attempt_policy(per_attempt_budget: int) -> dict[str, Any]:
    _require(int(per_attempt_budget) > 0, "Migration budget must be positive")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_POLICY,
        "max_attempts_per_stage": MIGRATION_MAX_ATTEMPTS_PER_STAGE,
        "per_attempt_budget": int(per_attempt_budget),
        "warm_start": "last_fully_archived_failed_raw_ppo_candidate",
        "fixed_opponent": "unchanged_hash_bound_stage_opponent",
        "optimizer_scheduler": (
            "cold_for_unique_suffix; same-suffix preemption may restore LoRA "
            "and scheduler position but never changes the initializer contract"
        ),
        "objective": binary_joint_objective_contract(),
        "on_exhaustion": "fail_closed_without_a_second_migration_attempt",
    }


class _MigrationTrainerTransformer(ast.NodeTransformer):
    """Narrowly clone the frozen role trainer for the attacker surrogate."""

    def __init__(self) -> None:
        self.stop_guard = 0
        self.minimum_guard = 0
        self.custom_config_insertion = 0
        self.command_insertion = 0

    @staticmethod
    def _raise_messages(node: ast.If) -> list[str]:
        messages: list[str] = []
        for statement in node.body:
            if not isinstance(statement, ast.Raise) or not isinstance(
                statement.exc, ast.Call
            ):
                continue
            for argument in statement.exc.args:
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    messages.append(argument.value)
        return messages

    def visit_If(self, node: ast.If) -> Any:
        self.generic_visit(node)
        messages = self._raise_messages(node)
        if "postfill_cot_stop_after_step requires enable_aux_sft=True" in messages:
            node.test = ast.parse(
                "not enable_aux_sft and not ("
                "train_role == 'attacker' and "
                "postfill_cot_stop_after_step == 0 and "
                "not role_specific_aux_sft and not v2_continuation_sft and "
                "defender_sft_optimizer_slots_per_rollout == 0)",
                mode="eval",
            ).body
            self.stop_guard += 1
        if "early_stop_min_steps must be at least early_stop_patience" in messages:
            node.test = ast.parse(
                "early_stop_min_steps < early_stop_patience and not ("
                "train_role == 'attacker' and "
                "early_stop_min_steps == 1 and early_stop_patience == 5 and "
                "early_stop_threshold == 0.95 and v2_runtime and "
                "not enable_aux_sft and not role_specific_aux_sft and "
                "not v2_continuation_sft and "
                "postfill_cot_stop_after_step == 0 and "
                "defender_sft_optimizer_slots_per_rollout == 0)",
                mode="eval",
            ).body
            self.minimum_guard += 1
        return node
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.generic_visit(node)
        if node.name != "train_upstream_attacker_lora_fixed_seed":
            return node
        node.decorator_list = []
        node.name = "_effective_attacker_binary_joint_migration_train"

        custom_anchor = [
            index
            for index, statement in enumerate(node.body)
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "sft_args"
                for target in statement.targets
            )
        ]
        _require(
            len(custom_anchor) == 1,
            "Frozen trainer SFT argument anchor is not unique",
        )
        custom_statements = ast.parse(
            """
if train_role != "attacker":
    raise RuntimeError("Binary-joint objective migration is attacker-only")
if not v2_runtime or enable_aux_sft or role_specific_aux_sft or v2_continuation_sft:
    raise RuntimeError("Binary-joint attacker migration requires PPO-only v2 runtime")
if postfill_cot_stop_after_step != 0 or defender_sft_optimizer_slots_per_rollout != 0:
    raise RuntimeError("Binary-joint attacker migration cannot schedule SFT")
custom_configs.update({
    "attacker_binary_joint_objective": True,
    "attacker_raw_reinforce_advantages": True,
    "attacker_episode_sum_policy_loss": True,
    "attacker_episode_sum_loss_scale": 1.0 / 2048.0,
    "attacker_official_payoff_utility": "frozen_upstream_additive_raw_no_normalization",
    "attacker_binary_surrogate_is_payoff_entry": False,
})
if early_stop_patience:
    custom_configs["early_stop_metric"] = "attacker/binary_joint_goal_cot_success_rate"
    custom_configs["early_stop_companion_bounds"] = {}
"""
        ).body
        index = custom_anchor[0]
        node.body[index:index] = custom_statements
        self.custom_config_insertion += 1

        command_anchor = [
            index
            for index, statement in enumerate(node.body)
            if isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Name)
            and statement.test.left.id == "train_batch_size"
        ]
        _require(
            len(command_anchor) == 1,
            "Frozen trainer command validation anchor is not unique",
        )
        command_statements = ast.parse(
            """
if "--reward_clip_range" in command or "--gamma" in command:
    raise RuntimeError("Binary-joint attacker command inherited defender-only flags")
normalize_index = command.index("--normalize_reward")
command[normalize_index:normalize_index] = ["--reward_clip_range", "-1.0", "1.0"]
custom_index = command.index("--custom_configs")
command[custom_index:custom_index] = ["--gamma", "1.0"]
"""
        ).body
        index = command_anchor[0]
        node.body[index:index] = command_statements
        self.command_insertion += 1
        return node


class _MigrationFinalAuditTransformer(ast.NodeTransformer):
    """Clone the live final audit while replacing only its gate dispatcher."""

    def __init__(self) -> None:
        self.gate_dispatch_replacements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name != "_build_final_population_audit":
            return node
        node.decorator_list = []
        node.name = "_effective_migration_aware_final_population_audit"
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == (
            "validate_successful_gate"
        ):
            node.func.id = "validate_migration_or_live_gate"
            node.keywords.extend(
                [
                    ast.keyword(
                        arg="migration_state",
                        value=ast.Name(id="state", ctx=ast.Load()),
                    ),
                    ast.keyword(
                        arg="stage_label",
                        value=ast.Name(id="label", ctx=ast.Load()),
                    ),
                ]
            )
            self.gate_dispatch_replacements += 1
        return node


def build_migration_aware_final_audit_source(
    gate_retry_source: str,
) -> tuple[str, dict[str, Any]]:
    """Build a narrow clone of the existing live 16-member audit.

    The clone preserves every existing population, checkpoint, recovery, and
    strict-LoRA audit operation.  Its sole semantic hook routes a migrated A
    stage to the binary gate verifier while all other stages continue through
    the original ``validate_successful_gate`` implementation.
    """

    module = ast.parse(gate_retry_source)
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_final_population_audit"
    ]
    _require(len(functions) == 1, "Final population audit function is not unique")
    transformer = _MigrationFinalAuditTransformer()
    transformed = transformer.visit(functions[0])
    _require(isinstance(transformed, ast.FunctionDef), "Final audit clone was lost")
    _require(
        transformer.gate_dispatch_replacements == 1,
        "Final audit gate dispatcher anchor is not unique",
    )
    ast.fix_missing_locations(transformed)
    effective = ast.unparse(transformed) + "\n"
    compile(effective, "<migration-aware-final-population-audit>", "exec")
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_FINAL_AUDIT_POLICY,
        "source_function": "_build_final_population_audit",
        "effective_function": (
            "_effective_migration_aware_final_population_audit"
        ),
        "gate_dispatch_replacements": 1,
        "injected_gate_context_kwargs": ["migration_state", "stage_label"],
        "unchanged_nonmigration_gate": "validate_successful_gate",
        "gate_retry_source_sha256": bytes_sha256(
            gate_retry_source.encode("utf-8")
        ),
        "effective_function_source_sha256": bytes_sha256(
            effective.encode("utf-8")
        ),
    }
    descriptor["descriptor_sha256"] = canonical_json_sha256(descriptor)
    return effective, descriptor


def build_migration_aware_gate_retry_finalize_sources(
    gate_retry_source: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Clone the D8 retry finalizer/resumer with explicit audited overrides.

    Runtime execution supplies migration-aware implementations for the three
    named hooks.  All training, CAS, WAL, swap, pruning, and retry behavior is
    otherwise the exact gate-retry source.
    """

    module = ast.parse(gate_retry_source)
    specifications = {
        "_release_or_complete": {
            "effective_name": (
                "_effective_migration_aware_release_or_complete"
            ),
            "required_calls": {"_build_final_population_audit": 1},
        },
        "resume_role_lora_selfplay8_gate_retry": {
            "effective_name": (
                "_effective_migration_aware_gate_retry_resume"
            ),
            "required_calls": {
                "_audit_completed_population": 1,
                "_drain_recovery_phase": 1,
                "_load_state_snapshot": 2,
                "_release_or_complete": 1,
            },
        },
    }
    sources: dict[str, str] = {}
    call_receipts: dict[str, dict[str, int]] = {}
    for source_name, specification in specifications.items():
        matches = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == source_name
        ]
        _require(len(matches) == 1, f"Gate-retry function is not unique: {source_name}")
        node = matches[0]
        node.decorator_list = []
        node.name = str(specification["effective_name"])
        required_calls = dict(specification["required_calls"])
        observed = {
            name: sum(
                1
                for candidate in ast.walk(node)
                if isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Name)
                and candidate.func.id == name
            )
            for name in required_calls
        }
        _require(
            observed == required_calls,
            f"Gate-retry finalization hook anchors drifted: {source_name}",
        )
        ast.fix_missing_locations(node)
        effective = ast.unparse(node) + "\n"
        compile(effective, f"<migration-aware-{source_name}>", "exec")
        sources[source_name] = effective
        call_receipts[source_name] = observed
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_FINAL_AUDIT_POLICY,
        "gate_retry_source_sha256": bytes_sha256(
            gate_retry_source.encode("utf-8")
        ),
        "runtime_hook_overrides": {
            "_build_final_population_audit": (
                "migration-aware binary/live gate builder"
            ),
            "_audit_completed_population": (
                "migration-aware idempotent completed audit"
            ),
            "_drain_recovery_phase": (
                "migration-aware D8 release dispatcher"
            ),
            "_load_state_snapshot": (
                "migration-history implementation verifier"
            ),
            "_release_or_complete": (
                "migration-aware D8 retry finalizer"
            ),
        },
        "required_call_receipts": call_receipts,
        "effective_function_source_sha256": {
            name: bytes_sha256(source.encode("utf-8"))
            for name, source in sources.items()
        },
    }
    descriptor["descriptor_sha256"] = canonical_json_sha256(descriptor)
    return sources, descriptor


def build_migration_final_audit_contract(
    gate_retry_source: str,
) -> dict[str, Any]:
    """Hash-bind the live audit and both D8 retry terminal paths."""

    _audit_source, audit_descriptor = build_migration_aware_final_audit_source(
        gate_retry_source
    )
    _finalize_sources, finalize_descriptor = (
        build_migration_aware_gate_retry_finalize_sources(gate_retry_source)
    )
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_FINAL_AUDIT_POLICY,
        "live_final_audit_clone": audit_descriptor,
        "d8_gate_retry_finalize_clones": finalize_descriptor,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def verify_migration_final_audit_contract(
    contract: Mapping[str, Any],
    *,
    expected_gate_retry_source_sha256: str | None = None,
) -> str:
    _require(
        contract.get("schema_version") == SCHEMA_VERSION
        and contract.get("policy") == MIGRATION_FINAL_AUDIT_POLICY,
        "Bad migration final-audit contract",
    )
    stored = _sha(contract.get("contract_sha256"), "migration final-audit contract")
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    _require(
        canonical_json_sha256(payload) == stored,
        "Migration final-audit contract digest drifted",
    )
    audit = contract.get("live_final_audit_clone")
    finalizers = contract.get("d8_gate_retry_finalize_clones")
    _require(
        isinstance(audit, Mapping) and isinstance(finalizers, Mapping),
        "Migration final-audit descriptors are missing",
    )
    for name, descriptor in (("audit", audit), ("finalizers", finalizers)):
        _require(
            descriptor.get("schema_version") == SCHEMA_VERSION
            and descriptor.get("policy") == MIGRATION_FINAL_AUDIT_POLICY,
            f"Migration final-{name} descriptor drifted",
        )
        descriptor_payload = dict(descriptor)
        descriptor_sha = _sha(
            descriptor_payload.pop("descriptor_sha256", None),
            f"migration final-{name} descriptor",
        )
        _require(
            canonical_json_sha256(descriptor_payload) == descriptor_sha,
            f"Migration final-{name} descriptor digest drifted",
        )
    source_sha = _sha(
        audit.get("gate_retry_source_sha256"),
        "migration final-audit gate-retry source",
    )
    _require(
        finalizers.get("gate_retry_source_sha256") == source_sha,
        "Migration final audit/finalizer source binding drifted",
    )
    if expected_gate_retry_source_sha256 is not None:
        _require(
            source_sha
            == _sha(
                expected_gate_retry_source_sha256,
                "expected migration gate-retry source",
            ),
            "Migration final audit uses a different gate-retry source",
        )
    _require(
        audit.get("gate_dispatch_replacements") == 1
        and audit.get("injected_gate_context_kwargs")
        == ["migration_state", "stage_label"]
        and audit.get("unchanged_nonmigration_gate")
        == "validate_successful_gate",
        "Migration final-audit gate dispatch contract drifted",
    )
    expected_hooks = {
        "_build_final_population_audit": (
            "migration-aware binary/live gate builder"
        ),
        "_audit_completed_population": (
            "migration-aware idempotent completed audit"
        ),
        "_drain_recovery_phase": "migration-aware D8 release dispatcher",
        "_load_state_snapshot": "migration-history implementation verifier",
        "_release_or_complete": "migration-aware D8 retry finalizer",
    }
    _require(
        finalizers.get("runtime_hook_overrides") == expected_hooks,
        "Migration D8 finalization hooks drifted",
    )
    return stored


def build_binary_joint_migration_trainer_source(
    frozen_core_source: str,
) -> tuple[str, dict[str, Any]]:
    """Build and hash the additive PPO-only attacker trainer clone."""

    module = ast.parse(frozen_core_source)
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "train_upstream_attacker_lora_fixed_seed"
    ]
    _require(len(functions) == 1, "Frozen role trainer function is not unique")
    transformer = _MigrationTrainerTransformer()
    transformed = transformer.visit(functions[0])
    _require(isinstance(transformed, ast.FunctionDef), "Trainer clone was lost")
    replacements = {
        "ppo_only_stop_zero": transformer.stop_guard,
        "gate_from_step_one": transformer.minimum_guard,
        "binary_custom_configs": transformer.custom_config_insertion,
        "binary_runtime_flags": transformer.command_insertion,
    }
    _require(
        replacements
        == {
            "ppo_only_stop_zero": 1,
            "gate_from_step_one": 1,
            "binary_custom_configs": 1,
            "binary_runtime_flags": 1,
        },
        f"Binary migration trainer patch surface drifted: {replacements}",
    )
    effective_module = ast.fix_missing_locations(
        ast.Module(body=[transformed], type_ignores=[])
    )
    effective_source = ast.unparse(effective_module) + "\n"
    compile(effective_source, "<attacker-binary-joint-migration>", "exec")
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_TRAINER_POLICY,
        "frozen_core_source_sha256": bytes_sha256(
            frozen_core_source.encode("utf-8")
        ),
        "effective_function_source_sha256": bytes_sha256(
            effective_source.encode("utf-8")
        ),
        "patch_replacements": replacements,
        "recipe": {
            "role": "attacker",
            "v2_runtime": True,
            "enable_aux_sft": False,
            "role_specific_aux_sft": False,
            "v2_continuation_sft": False,
            "postfill_cot_stop_after_step": 0,
            "defender_sft_optimizer_slots_per_rollout": 0,
            "reward_clip_range": [-1.0, 1.0],
            "gamma": 1.0,
            "init_kl_coef": 0.0,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 1,
            "objective": binary_joint_objective_contract(),
        },
    }
    descriptor["patch_descriptor_sha256"] = canonical_json_sha256(descriptor)
    return effective_source, descriptor


def verify_binary_joint_migration_trainer_contract(
    contract: Mapping[str, Any],
    *,
    expected_frozen_core_sha256: str | None = None,
) -> str:
    _require(
        contract.get("schema_version") == SCHEMA_VERSION
        and contract.get("policy") == MIGRATION_TRAINER_POLICY,
        "Bad binary migration trainer contract",
    )
    frozen = _sha(contract.get("frozen_core_source_sha256"), "frozen core")
    if expected_frozen_core_sha256 is not None:
        _require(
            frozen == _sha(expected_frozen_core_sha256, "expected frozen core"),
            "Binary migration frozen core identity drifted",
        )
    _sha(contract.get("effective_function_source_sha256"), "effective trainer")
    stored = _sha(contract.get("patch_descriptor_sha256"), "trainer descriptor")
    payload = dict(contract)
    payload.pop("patch_descriptor_sha256", None)
    _require(
        canonical_json_sha256(payload) == stored,
        "Binary migration trainer descriptor drifted",
    )
    _require(
        contract.get("recipe", {}).get("objective")
        == binary_joint_objective_contract(),
        "Binary migration objective contract drifted",
    )
    return stored


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    _require(count == 1, f"Expected one {label} marker, found {count}")
    return text.replace(old, new, 1)


def build_binary_joint_upstream_sources(
    *,
    language_game_source: str,
    experience_maker_source: str,
    replay_buffer_source: str,
    actor_source: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Patch an already-frozen-prepared upstream tree in memory.

    The patch is intentionally applied *after* the frozen role trainer's
    normal source preparation.  Every replacement is single-hit and the
    resulting sources are separately hashed into the migration manifest.
    """

    game = _replace_once(
        language_game_source,
        """            # Skip if wildguard cannot parse the response
            if labels.get('is_parsing_error', False):
                continue

            invalid_attack_rewrite = False
""",
        """            # A parse failure is an explicit negative attacker
            # trajectory under the binary oracle, not a removed sample.
            binary_attacker_parse_failure = bool(
                labels.get('is_parsing_error', False)
                and self.custom_configs.get(
                    'attacker_binary_joint_objective', False
                )
            )
            if (
                labels.get('is_parsing_error', False)
                and not binary_attacker_parse_failure
            ):
                continue

            invalid_attack_rewrite = False
""",
        "retain binary attacker parse failures",
    )
    game = _replace_once(
        game,
        """                if invalid_attack_rewrite:
                    reward, outcome = 0.0, {GameOutcome.TIE}
                else:
""",
        """                if (
                    invalid_attack_rewrite
                    or binary_attacker_parse_failure
                ):
                    reward, outcome = 0.0, {GameOutcome.TIE}
                else:
""",
        "binary parse failures become negative ties",
    )
    game = _replace_once(
        game,
        """                # Update turn_states with computed values
                turn_states['reward'] = reward
""",
        """                if (
                    self.custom_configs.get(
                        'attacker_binary_joint_objective', False
                    )
                    and turn_states["game_role"] == "attacker"
                ):
                    cot_violation = turn_states.get(
                        'cot_format_violation', None
                    )
                    attacker_goal_success = (
                        (
                            GameOutcome.ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST
                            in outcome
                            or GameOutcome.ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST
                            in outcome
                        )
                        and GameOutcome.TIE not in outcome
                        and cot_violation is False
                    )
                    turn_states['attacker_official_raw_utility'] = float(
                        reward
                    )
                    turn_states['attacker_binary_joint_success'] = bool(
                        attacker_goal_success
                    )
                    reward = 1.0 if attacker_goal_success else -1.0
                    turn_states['attacker_binary_joint_reward'] = reward

                # Update turn_states with computed values
                turn_states['reward'] = reward
""",
        "binary attacker reward",
    )

    experience = _replace_once(
        experience_maker_source,
        """            'raw_upstream_additive_reward',
            'defender_joint_success', 'defender_joint_signed_reward',
""",
        """            'raw_upstream_additive_reward',
            'attacker_official_raw_utility',
            'attacker_binary_joint_success',
            'attacker_binary_joint_reward',
            'defender_joint_success', 'defender_joint_signed_reward',
""",
        "binary attacker replay metadata",
    )

    replay = _replace_once(
        replay_buffer_source,
        """        attacker_rewards = [item.info['reward'] for item in attacker_items]
        avg_attacker_reward = sum(attacker_rewards) / max(len(attacker_rewards), 1)
""",
        """        attacker_rewards = [item.info['reward'] for item in attacker_items]
        avg_attacker_reward = sum(attacker_rewards) / max(len(attacker_rewards), 1)
        binary_attacker_objective = bool(
            self.custom_configs.get(
                'attacker_binary_joint_objective', False
            )
        )
        binary_optimization_reward_mean = None
        official_raw_utility_diagnostic_mean = None
        if binary_attacker_objective:
            for item in attacker_items:
                expected_success = (
                    (
                        GameOutcome.ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST
                        in item.info['game_outcomes']
                        or GameOutcome.ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST
                        in item.info['game_outcomes']
                    )
                    and GameOutcome.TIE not in item.info['game_outcomes']
                    and item.info.get('cot_format_violation', None) is False
                )
                expected_reward = 1.0 if expected_success else -1.0
                if (
                    item.info.get('attacker_binary_joint_success')
                    is not expected_success
                    or float(item.info.get(
                        'attacker_binary_joint_reward', 0.0
                    )) != expected_reward
                    or float(item.info['reward']) != expected_reward
                    or 'attacker_official_raw_utility' not in item.info
                ):
                    raise RuntimeError(
                        "Binary attacker replay objective/telemetry drifted"
                    )
            binary_optimization_reward_mean = (
                sum(float(item.info['reward']) for item in attacker_items)
                / max(len(attacker_items), 1)
            )
            official_raw_utility_diagnostic_mean = (
                sum(
                    float(item.info['attacker_official_raw_utility'])
                    for item in attacker_items
                )
                / max(len(attacker_items), 1)
            )
""",
        "binary and official attacker utility telemetry",
    )
    replay = _replace_once(
        replay,
        """        preserve_joint_signed_defender_failures = bool(
            strategy.args.custom_configs.get(
                "defender_actual_strata_required", False
            )
        )
        self.items = [
            item for item in self.items
            if (
                preserve_joint_signed_defender_failures
                and item.info.get("game_role") == "defender"
                and float(item.info.get("reward")) in (-1.0, 1.0)
            )
            or GameOutcome.TIE not in item.info['game_outcomes']
        ]
""",
        """        preserve_joint_signed_defender_failures = bool(
            strategy.args.custom_configs.get(
                "defender_actual_strata_required", False
            )
        )
        preserve_binary_joint_attacker_failures = bool(
            strategy.args.custom_configs.get(
                "attacker_binary_joint_objective", False
            )
        )
        self.items = [
            item for item in self.items
            if (
                preserve_joint_signed_defender_failures
                and item.info.get("game_role") == "defender"
                and float(item.info.get("reward")) in (-1.0, 1.0)
            )
            or (
                preserve_binary_joint_attacker_failures
                and item.info.get("game_role") == "attacker"
                and float(item.info.get("reward")) in (-1.0, 1.0)
            )
            or GameOutcome.TIE not in item.info['game_outcomes']
        ]
""",
        "retain binary attacker failures and ties",
    )
    replay = _replace_once(
        replay,
        """        successful_rate = (n_successful_benign_request + n_successful_harmful_request) / max(n_generated_attacks, 1)
""",
        """        successful_rate = (n_successful_benign_request + n_successful_harmful_request) / max(n_generated_attacks, 1)
        n_binary_joint_goal_cot_success = sum(
            (
                GameOutcome.ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST
                in item.info['game_outcomes']
                or GameOutcome.ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST
                in item.info['game_outcomes']
            )
            and GameOutcome.TIE not in item.info['game_outcomes']
            and item.info.get('cot_format_violation', None) is False
            for item in attacker_items
        )
        binary_joint_goal_cot_success_rate = (
            n_binary_joint_goal_cot_success
            / max(n_generated_attacks, 1)
        )
""",
        "binary attacker gate numerator",
    )
    replay = _replace_once(
        replay,
        """        if is_cot_enabled:
            attacker_cot_violations = sum(item.info['cot_format_violation'] for item in attacker_items)
            attacker_cot_rate = attacker_cot_violations / max(n_generated_attacks, 1)
""",
        """        if is_cot_enabled:
            attacker_cot_violations = sum(
                item.info.get('cot_format_violation', None) is not False
                if binary_attacker_objective
                else item.info['cot_format_violation']
                for item in attacker_items
            )
            attacker_cot_rate = attacker_cot_violations / max(n_generated_attacks, 1)
""",
        "binary attacker missing CoT verdict is invalid",
    )
    replay = _replace_once(
        replay,
        """            'attacker/avg_reward': strategy.all_reduce(avg_attacker_reward, "mean"),
""",
        """            'attacker/avg_reward': strategy.all_reduce(avg_attacker_reward, "mean"),
            'attacker/binary_joint_optimization_reward_mean': (
                strategy.all_reduce(
                    binary_optimization_reward_mean, "mean"
                )
                if binary_optimization_reward_mean is not None else None
            ),
            'attacker/official_raw_utility_diagnostic_mean': (
                strategy.all_reduce(
                    official_raw_utility_diagnostic_mean, "mean"
                )
                if official_raw_utility_diagnostic_mean is not None else None
            ),
""",
        "separate binary and official attacker metric namespaces",
    )
    replay = _replace_once(
        replay,
        """            'attacker/request_success_rate': strategy.all_reduce(successful_rate, "mean"), # successful = benign and harmful both leads to harmful reaction
""",
        """            'attacker/request_success_rate': strategy.all_reduce(successful_rate, "mean"), # successful = benign and harmful both leads to harmful reaction
            'attacker/binary_joint_goal_cot_success_rate': strategy.all_reduce(binary_joint_goal_cot_success_rate, "mean"),
""",
        "binary attacker gate metric",
    )

    actor = _replace_once(
        actor_source,
        """    raw_defender = bool(
        args.custom_configs.get(
            "defender_raw_reinforce_advantages", False
        )
    )
    if not raw_defender:
        return "normalize"
""",
        """    raw_attacker = bool(
        args.custom_configs.get(
            "attacker_raw_reinforce_advantages", False
        )
    )
    if raw_attacker:
        if optimizer_train_role != "attacker":
            raise RuntimeError(
                "Raw attacker advantages require optimizer_train_role=attacker"
            )
        if args.advantage_estimator != "reinforce":
            raise RuntimeError(
                "Raw attacker advantages require advantage_estimator=reinforce"
            )
        if float(args.gamma) != 1.0 or float(args.init_kl_coef) != 0.0:
            raise RuntimeError(
                "Raw attacker advantages require gamma=1 and KL=0"
            )
        if int(args.custom_configs.get(
            "defender_sft_optimizer_slots_per_rollout", 0
        )) != 0:
            raise RuntimeError("Raw attacker migration cannot schedule SFT")
        observed = {
            "generate_max_len": int(args.generate_max_len),
            "packing_samples": bool(args.packing_samples),
            "actor_loss_coef": float(args.actor_loss_coef),
            "reward_clip_range": tuple(
                float(value) for value in args.reward_clip_range
            ),
            "use_kl_loss": bool(args.use_kl_loss),
            "episode_sum": bool(args.custom_configs.get(
                "attacker_episode_sum_policy_loss", False
            )),
            "loss_scale": float(args.custom_configs.get(
                "attacker_episode_sum_loss_scale", 0.0
            )),
            "surrogate_is_payoff": bool(args.custom_configs.get(
                "attacker_binary_surrogate_is_payoff_entry", True
            )),
        }
        expected = {
            "generate_max_len": 2048,
            "packing_samples": True,
            "actor_loss_coef": 1.0,
            "reward_clip_range": (-1.0, 1.0),
            "use_kl_loss": False,
            "episode_sum": True,
            "loss_scale": 1.0 / 2048.0,
            "surrogate_is_payoff": False,
        }
        if observed != expected:
            raise RuntimeError(
                "Binary-joint attacker PPO runtime contract drifted: "
                f"observed={observed}, expected={expected}"
            )
        return "binary_joint_attacker_reinforce"
    raw_defender = bool(
        args.custom_configs.get(
            "defender_raw_reinforce_advantages", False
        )
    )
    if not raw_defender:
        return "normalize"
""",
        "raw attacker advantage mode",
    )
    actor = _replace_once(
        actor,
        """                    if advantage_transform_mode in (
                        'raw_defender_reinforce',
                        'joint_signed_defender_reinforce',
                    ):
""",
        """                    if advantage_transform_mode in (
                        'raw_defender_reinforce',
                        'joint_signed_defender_reinforce',
                        'binary_joint_attacker_reinforce',
                    ):
""",
        "binary raw advantage branch",
    )
    actor = _replace_once(
        actor,
        """                        joint_signed_mode = (
                            advantage_transform_mode
                            == 'joint_signed_defender_reinforce'
                        )
                        raw_reward_snapshot = []
""",
        """                        joint_signed_mode = (
                            advantage_transform_mode
                            == 'joint_signed_defender_reinforce'
                        )
                        binary_attacker_mode = (
                            advantage_transform_mode
                            == 'binary_joint_attacker_reinforce'
                        )
                        raw_reward_snapshot = []
""",
        "binary raw advantage mode flag",
    )
    actor = _replace_once(
        actor,
        """                            if joint_signed_mode and reward_value not in (
                                -1.0, 1.0
                            ):
                                raise RuntimeError(
                                    "Official defender joint-signed reward "
                                    f"must be +/-1, got {reward_value}"
                                )
""",
        """                            if (
                                joint_signed_mode or binary_attacker_mode
                            ) and reward_value not in (-1.0, 1.0):
                                raise RuntimeError(
                                    "Raw joint reward must be +/-1, got "
                                    f"{reward_value}"
                                )
""",
        "binary reward range assertion",
    )
    actor = _replace_once(
        actor,
        """                        status[
                            "debug/defender_raw_reinforce_advantages"
                        ] = 1.0
                        status[
                            "debug/defender_advantage_mean_centering_applied"
                        ] = 0.0
                        status[
                            "debug/defender_advantage_std_norm_applied"
                        ] = 0.0
""",
        """                        status[
                            "debug/defender_raw_reinforce_advantages"
                        ] = float(not binary_attacker_mode)
                        status[
                            "debug/attacker_raw_reinforce_advantages"
                        ] = float(binary_attacker_mode)
                        status[
                            "debug/defender_advantage_mean_centering_applied"
                        ] = float(False)
                        status[
                            "debug/defender_advantage_std_norm_applied"
                        ] = float(False)
                        status[
                            "debug/attacker_advantage_mean_centering_applied"
                        ] = 0.0
                        status[
                            "debug/attacker_advantage_std_norm_applied"
                        ] = 0.0
""",
        "binary raw advantage transform telemetry",
    )
    actor = _replace_once(
        actor,
        """                        status[
                            "debug/defender_joint_signed_advantages"
                        ] = float(joint_signed_mode)
                        status[
                            "debug/defender_episode_sum_loss_scale"
                        ] = float(
                            self.args.custom_configs.get(
                                "defender_episode_sum_loss_scale", 0.0
                            )
                            if joint_signed_mode else 0.0
                        )
""",
        """                        status[
                            "debug/defender_joint_signed_advantages"
                        ] = float(joint_signed_mode)
                        status[
                            "debug/attacker_binary_joint_advantages"
                        ] = float(binary_attacker_mode)
                        status[
                            "debug/defender_episode_sum_loss_scale"
                        ] = float(
                            self.args.custom_configs.get(
                                "defender_episode_sum_loss_scale", 0.0
                            )
                            if joint_signed_mode else 0.0
                        )
                        status[
                            "debug/attacker_episode_sum_loss_scale"
                        ] = float(
                            self.args.custom_configs.get(
                                "attacker_episode_sum_loss_scale", 0.0
                            )
                            if binary_attacker_mode else 0.0
                        )
""",
        "binary raw advantage telemetry",
    )
    actor = _replace_once(
        actor,
        """                    if advantage_transform_mode not in (
                        'raw_defender_reinforce',
                        'joint_signed_defender_reinforce',
                    ):
""",
        """                    if advantage_transform_mode not in (
                        'raw_defender_reinforce',
                        'joint_signed_defender_reinforce',
                        'binary_joint_attacker_reinforce',
                    ):
""",
        "skip binary advantage normalization diagnostics",
    )
    actor = _replace_once(
        actor,
        """        if self.args.custom_configs.get(
            "defender_episode_sum_policy_loss", False
        ):
            if self.args.custom_configs.get(
                "optimizer_train_role"
            ) != "defender":
                raise RuntimeError(
                    "Episode-sum PPO is restricted to defender training"
                )
            actor_loss = _defender_episode_sum_policy_loss(
                action_log_probs,
                old_action_log_probs,
                advantages,
                experience.action_mask,
                clip_eps=self.actor_loss_fn.clip_eps,
                packing_samples=self.args.packing_samples,
                num_actions=num_actions,
                loss_scale=self.args.custom_configs.get(
                    "defender_episode_sum_loss_scale"
                ),
            )
        else:
""",
        """        defender_episode_sum = self.args.custom_configs.get(
            "defender_episode_sum_policy_loss", False
        )
        attacker_episode_sum = self.args.custom_configs.get(
            "attacker_episode_sum_policy_loss", False
        )
        if defender_episode_sum or attacker_episode_sum:
            expected_role = "attacker" if attacker_episode_sum else "defender"
            if (
                bool(defender_episode_sum) == bool(attacker_episode_sum)
                or self.args.custom_configs.get("optimizer_train_role")
                != expected_role
            ):
                raise RuntimeError(
                    "Episode-sum PPO role/objective contract drifted"
                )
            loss_scale_key = (
                "attacker_episode_sum_loss_scale"
                if attacker_episode_sum
                else "defender_episode_sum_loss_scale"
            )
            actor_loss = _defender_episode_sum_policy_loss(
                action_log_probs,
                old_action_log_probs,
                advantages,
                experience.action_mask,
                clip_eps=self.actor_loss_fn.clip_eps,
                packing_samples=self.args.packing_samples,
                num_actions=num_actions,
                loss_scale=self.args.custom_configs.get(loss_scale_key),
            )
        else:
""",
        "attacker episode-sum PPO surrogate",
    )

    sources = {
        "openrlhf/trainer/ppo_utils/language_game.py": game,
        "openrlhf/trainer/ppo_utils/experience_maker.py": experience,
        "openrlhf/trainer/ppo_utils/replay_buffer.py": replay,
        "openrlhf/trainer/ray/ppo_actor.py": actor,
    }
    input_source_sha256 = {
        "openrlhf/trainer/ppo_utils/language_game.py": bytes_sha256(
            language_game_source.encode("utf-8")
        ),
        "openrlhf/trainer/ppo_utils/experience_maker.py": bytes_sha256(
            experience_maker_source.encode("utf-8")
        ),
        "openrlhf/trainer/ppo_utils/replay_buffer.py": bytes_sha256(
            replay_buffer_source.encode("utf-8")
        ),
        "openrlhf/trainer/ray/ppo_actor.py": bytes_sha256(
            actor_source.encode("utf-8")
        ),
    }
    source_sha256 = {
        name: bytes_sha256(source.encode("utf-8"))
        for name, source in sources.items()
    }
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_TRAINER_POLICY,
        "patch_order": "after_frozen_role_lora_upstream_preparation",
        "input_source_sha256": input_source_sha256,
        "patched_source_sha256": source_sha256,
        "objective": binary_joint_objective_contract(),
    }
    descriptor["patch_descriptor_sha256"] = canonical_json_sha256(descriptor)
    return sources, descriptor


def patch_binary_joint_upstream_tree(upstream_root: Path) -> dict[str, Any]:
    relative = (
        "openrlhf/trainer/ppo_utils/language_game.py",
        "openrlhf/trainer/ppo_utils/experience_maker.py",
        "openrlhf/trainer/ppo_utils/replay_buffer.py",
        "openrlhf/trainer/ray/ppo_actor.py",
    )
    paths = {name: upstream_root / name for name in relative}
    source = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    patched, descriptor = build_binary_joint_upstream_sources(
        language_game_source=source[relative[0]],
        experience_maker_source=source[relative[1]],
        replay_buffer_source=source[relative[2]],
        actor_source=source[relative[3]],
    )
    for name, text in patched.items():
        compile(text, str(paths[name]), "exec")
        paths[name].write_text(text, encoding="utf-8")
    return descriptor


def validate_migration_eligibility(state: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the hash-bound terminal hand-off from gate retry."""

    _require(state.get("schema_version") == 1, "Bad self-play state schema")
    _require(
        state.get("status") == OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS,
        "Objective migration requires its exact terminal state",
    )
    label = str(state.get("active_stage") or "")
    _require(bool(_ATTACKER_STAGE_RE.fullmatch(label)), "Migration stage is not A2--A8")
    recovery = state.get(RECOVERY_KEY)
    _require(isinstance(recovery, Mapping), "Gate-retry recovery is missing")
    _require(
        recovery.get("status") == RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
        "Gate-retry recovery is not raw-PPO exhausted",
    )
    plan = recovery.get("plan")
    attempts = recovery.get("attempts")
    requirement = recovery.get("objective_migration_requirement")
    _require(
        isinstance(plan, Mapping)
        and isinstance(attempts, list)
        and isinstance(requirement, Mapping),
        "Objective migration hand-off is incomplete",
    )
    plan_id = verify_recovery_plan(plan)
    requirement_id = verify_objective_migration_requirement(
        requirement, plan, attempts
    )
    _require(
        plan.get("stage_label") == label
        and plan.get("role") == "attacker"
        and requirement.get("stage_label") == label
        and requirement.get("role") == "attacker",
        "Objective migration stage/role binding drifted",
    )
    _require(
        requirement.get("required_next_objective")
        == MIGRATION_REQUIRED_OBJECTIVE,
        "Objective migration hand-off requested a different objective",
    )
    expected_contract = binary_joint_objective_contract()
    required_contract = requirement.get("required_next_objective_contract")
    _require(isinstance(required_contract, Mapping), "Required objective contract is missing")
    _require(
        required_contract.get("optimization_reward")
        == expected_contract["optimization_reward"]
        and required_contract.get("advantage") == "raw_no_center_no_std"
        and required_contract.get("aux_sft")
        == "disabled from the first optimizer step",
        "Objective migration hand-off semantics drifted",
    )
    stage = (state.get("stages") or {}).get(label)
    _require(isinstance(stage, Mapping), "Objective migration stage is missing")
    release = stage.get("successor_release")
    _require(
        isinstance(release, Mapping) and release.get("approved") is False,
        "Unqualified stage already released its successor",
    )
    _sha(requirement.get("trainable_init_sha256"), "migration initializer")
    fixed = requirement.get("fixed_opponent")
    _require(isinstance(fixed, Mapping), "Migration fixed opponent is missing")
    _sha(fixed.get("sha256"), "migration fixed opponent")
    return {
        "label": label,
        "plan_id": plan_id,
        "requirement_id": requirement_id,
        "recovery": dict(recovery),
        "requirement": dict(requirement),
        "stage": dict(stage),
    }


def build_migration_plan(
    state: Mapping[str, Any],
    *,
    migration_implementation_sha256: Mapping[str, str],
    migration_trainer_contract: Mapping[str, Any],
    migration_final_audit_contract: Mapping[str, Any],
    plan_path: str,
) -> dict[str, Any]:
    eligibility = validate_migration_eligibility(state)
    requirement = eligibility["requirement"]
    frozen = state.get("config", {}).get("training_implementation_sha256")
    _require(isinstance(frozen, Mapping) and frozen, "Frozen training hashes are missing")
    for name, digest in frozen.items():
        _sha(digest, f"frozen training source {name}")
    implementation = dict(migration_implementation_sha256)
    _require(bool(implementation), "Migration implementation hashes are missing")
    for name, digest in implementation.items():
        _sha(digest, f"migration source {name}")
    verify_binary_joint_migration_trainer_contract(
        migration_trainer_contract,
        expected_frozen_core_sha256=str(
            frozen["modal_upstream_selfredteam_role_lora.py"]
        ),
    )
    verify_migration_final_audit_contract(migration_final_audit_contract)
    budget = int(eligibility["recovery"]["plan"]["per_attempt_budget"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_POLICY,
        "stage_label": eligibility["label"],
        "role": "attacker",
        "source_gate_retry_plan_id": eligibility["plan_id"],
        "objective_migration_requirement_id": eligibility["requirement_id"],
        "objective_migration_requirement": requirement,
        "trainable_init": {
            "checkpoint": requirement["trainable_init_checkpoint"],
            "sha256": requirement["trainable_init_sha256"],
            "source_attempt_id": requirement["source_attempt_id"],
        },
        "fixed_opponent": dict(requirement["fixed_opponent"]),
        "displaced_nonqualifying_population": {
            "checkpoint": eligibility["stage"]["population_checkpoint"],
            "sha256": eligibility["stage"]["sha256"],
        },
        "attempt_policy": migration_attempt_policy(budget),
        "frozen_selfplay_config": dict(state["config"]),
        "frozen_training_implementation_sha256": dict(frozen),
        "migration_implementation_sha256": implementation,
        "migration_trainer_contract": dict(migration_trainer_contract),
        "migration_final_audit_contract": dict(
            migration_final_audit_contract
        ),
        "original_stage_spawn_claim_id": eligibility["stage"]["spawn_claim_id"],
        "plan_path": plan_path,
    }
    payload["plan_id"] = canonical_json_sha256(payload)
    return payload


def verify_migration_plan(plan: Mapping[str, Any]) -> str:
    _require(
        plan.get("schema_version") == SCHEMA_VERSION
        and plan.get("policy") == MIGRATION_POLICY,
        "Bad objective migration plan",
    )
    label = str(plan.get("stage_label") or "")
    _require(bool(_ATTACKER_STAGE_RE.fullmatch(label)), "Bad migration stage")
    _require(plan.get("role") == "attacker", "Migration plan is not attacker-only")
    stored = _sha(plan.get("plan_id"), "migration plan")
    payload = dict(plan)
    payload.pop("plan_id", None)
    _require(canonical_json_sha256(payload) == stored, "Migration plan digest drifted")
    requirement = plan.get("objective_migration_requirement")
    _require(isinstance(requirement, Mapping), "Migration plan lost its requirement")
    _require(
        requirement.get("requirement_id")
        == plan.get("objective_migration_requirement_id"),
        "Migration requirement identity drifted",
    )
    _require(
        requirement.get("required_next_objective")
        == MIGRATION_REQUIRED_OBJECTIVE,
        "Migration plan requested a different objective",
    )
    init = plan.get("trainable_init")
    fixed = plan.get("fixed_opponent")
    displaced = plan.get("displaced_nonqualifying_population")
    for name, provenance in (
        ("initializer", init),
        ("fixed opponent", fixed),
        ("displaced population", displaced),
    ):
        _require(isinstance(provenance, Mapping), f"Migration {name} is missing")
        _require(bool(provenance.get("checkpoint")), f"Migration {name} path is empty")
        _sha(provenance.get("sha256"), f"migration {name}")
    _require(
        init.get("checkpoint") == requirement.get("trainable_init_checkpoint")
        and init.get("sha256") == requirement.get("trainable_init_sha256")
        and init.get("source_attempt_id") == requirement.get("source_attempt_id")
        and dict(fixed) == requirement.get("fixed_opponent"),
        "Migration plan no longer binds the hand-off adapters",
    )
    _sha(plan.get("source_gate_retry_plan_id"), "source gate-retry plan")
    _sha(plan.get("original_stage_spawn_claim_id"), "original stage claim")
    _require(bool(plan.get("plan_path")), "Migration plan path is empty")
    for mapping_name in (
        "frozen_training_implementation_sha256",
        "migration_implementation_sha256",
    ):
        mapping = plan.get(mapping_name)
        _require(isinstance(mapping, Mapping) and mapping, f"Missing {mapping_name}")
        for name, digest in mapping.items():
            _sha(digest, f"{mapping_name} {name}")
    _require(
        plan.get("attempt_policy")
        == migration_attempt_policy(
            int(plan["attempt_policy"]["per_attempt_budget"])
        ),
        "Migration attempt policy drifted",
    )
    verify_binary_joint_migration_trainer_contract(
        plan.get("migration_trainer_contract", {}),
        expected_frozen_core_sha256=str(
            plan["frozen_training_implementation_sha256"][
                "modal_upstream_selfredteam_role_lora.py"
            ]
        ),
    )
    verify_migration_final_audit_contract(
        plan.get("migration_final_audit_contract", {})
    )
    return stored


def build_migration_attempt_contract(
    plan: Mapping[str, Any],
    *,
    trainer_run_suffix: str,
    contract_path: str,
) -> dict[str, Any]:
    plan_id = verify_migration_plan(plan)
    _require(bool(trainer_run_suffix) and bool(contract_path), "Migration attempt paths are empty")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_POLICY,
        "plan_id": plan_id,
        "attempt_number": 1,
        "stage_label": plan["stage_label"],
        "role": "attacker",
        "trainable_init_checkpoint": plan["trainable_init"]["checkpoint"],
        "trainable_init_sha256": plan["trainable_init"]["sha256"],
        "fixed_opponent": dict(plan["fixed_opponent"]),
        "per_attempt_budget": int(plan["attempt_policy"]["per_attempt_budget"]),
        "objective": binary_joint_objective_contract(),
        "trainer_run_suffix": trainer_run_suffix,
        "contract_path": contract_path,
        "frozen_training_implementation_sha256": dict(
            plan["frozen_training_implementation_sha256"]
        ),
        "migration_implementation_sha256": dict(
            plan["migration_implementation_sha256"]
        ),
        "migration_trainer_contract": dict(plan["migration_trainer_contract"]),
        "migration_final_audit_contract": dict(
            plan["migration_final_audit_contract"]
        ),
    }
    payload["attempt_id"] = canonical_json_sha256(payload)
    return payload


def verify_migration_attempt_contract(
    contract: Mapping[str, Any], plan: Mapping[str, Any]
) -> str:
    plan_id = verify_migration_plan(plan)
    _require(
        contract.get("schema_version") == SCHEMA_VERSION
        and contract.get("policy") == MIGRATION_POLICY
        and contract.get("plan_id") == plan_id
        and contract.get("attempt_number") == 1,
        "Bad migration attempt contract",
    )
    stored = _sha(contract.get("attempt_id"), "migration attempt")
    payload = dict(contract)
    payload.pop("attempt_id", None)
    _require(canonical_json_sha256(payload) == stored, "Migration attempt digest drifted")
    expected = {
        "stage_label": plan["stage_label"],
        "role": "attacker",
        "trainable_init_checkpoint": plan["trainable_init"]["checkpoint"],
        "trainable_init_sha256": plan["trainable_init"]["sha256"],
        "fixed_opponent": plan["fixed_opponent"],
        "per_attempt_budget": plan["attempt_policy"]["per_attempt_budget"],
        "objective": binary_joint_objective_contract(),
        "frozen_training_implementation_sha256": plan[
            "frozen_training_implementation_sha256"
        ],
        "migration_implementation_sha256": plan[
            "migration_implementation_sha256"
        ],
        "migration_trainer_contract": plan["migration_trainer_contract"],
        "migration_final_audit_contract": plan[
            "migration_final_audit_contract"
        ],
    }
    for key, value in expected.items():
        _require(contract.get(key) == value, f"Migration attempt drifted at {key}")
    _require(
        bool(contract.get("trainer_run_suffix")) and bool(contract.get("contract_path")),
        "Migration attempt path provenance is incomplete",
    )
    return stored


def validate_binary_joint_successful_gate(
    validation: Mapping[str, Any],
    *,
    expected_budget: int,
    save_steps: int,
    expected_final_sha256: str,
    expected_initial_sha256: str,
) -> dict[str, Any]:
    """Independently recompute the exact binary g gate and checkpoint proof."""

    objective = binary_joint_objective_contract()
    gate = objective["gate"]
    _require(validation.get("stopped_early") is True, "Migration did not stop early")
    early = validation.get("early_stop")
    _require(isinstance(early, Mapping), "Migration early-stop record is missing")
    _require(early.get("metric") == gate["metric"], "Migration gate metric drifted")
    _require(float(early.get("threshold", -1)) == gate["threshold"], "Migration gate threshold drifted")
    patience = int(gate["patience"])
    min_steps = int(gate["min_steps"])
    _require(int(early.get("patience", -1)) == patience, "Migration gate patience drifted")
    _require(int(early.get("min_steps", -1)) == min_steps, "Migration gate min-step drifted")
    _require(early.get("companion_bounds") == {}, "Migration gate has foreign companions")
    _require(early.get("companion_metrics") in ([], None), "Migration gate companion metrics drifted")
    _require(early.get("triggered") is True, "Migration gate did not trigger")
    _require(int(early.get("streak", -1)) >= patience, "Migration gate streak is short")
    actual_final_step = int(validation.get("actual_final_step", 0))
    _require(0 < actual_final_step <= int(expected_budget), "Migration final step is out of budget")
    _require(int(validation.get("requested_max_step", -1)) == int(expected_budget), "Migration budget drifted")
    _require(int(early.get("actual_final_step", -1)) == actual_final_step, "Migration gate/final step drifted")
    _require(int(early.get("last_step", actual_final_step)) == actual_final_step, "Migration last step drifted")
    _require(
        str(early.get("checkpoint_tag") or "") == f"global_step{actual_final_step}",
        "Migration checkpoint tag drifted",
    )
    _require(
        str(validation.get("final_checkpoint") or "").endswith(
            f"/global_step{actual_final_step}_hf"
        ),
        "Migration final checkpoint path drifted",
    )
    history = early.get("history")
    _require(isinstance(history, list) and len(history) >= patience, "Migration gate history is short")
    tail = history[-patience:]
    expected_steps = list(range(actual_final_step - patience + 1, actual_final_step + 1))
    _require([int(row.get("step", -1)) for row in tail] == expected_steps, "Migration gate tail is not consecutive")
    values = [float(row.get("value", float("nan"))) for row in tail]
    _require(
        all(
            step >= min_steps
            and math.isfinite(value)
            and value >= float(gate["threshold"])
            and row.get("qualified") is True
            and row.get("metrics") in ({}, None)
            for step, value, row in zip(expected_steps, values, tail)
        ),
        "Migration gate tail did not satisfy exact g",
    )
    cadence = validate_checkpoint_cadence(
        validation,
        expected_final_step=actual_final_step,
        save_steps=int(save_steps),
        expected_final_sha256=_sha(expected_final_sha256, "migration final"),
        allow_single_checkpoint_change_proof=True,
        expected_initial_sha256=_sha(expected_initial_sha256, "migration initializer"),
    )
    return {
        "passed": True,
        "policy": MIGRATION_POLICY,
        "metric": gate["metric"],
        "threshold": gate["threshold"],
        "patience": patience,
        "min_steps": min_steps,
        "tail_steps": expected_steps,
        "tail_values": values,
        "checkpoint_cadence": cadence,
        "optimization_surrogate_only": True,
        "official_payoff_utility_unchanged": True,
    }


def build_migration_history_entry(
    migration: Mapping[str, Any],
    raw_gate_retry_recovery: Mapping[str, Any],
    *,
    archived_state_file_sha256: str,
) -> dict[str, Any]:
    _sha(archived_state_file_sha256, "archived migration state")
    plan = migration.get("plan")
    _require(isinstance(plan, Mapping), "Migration history has no plan")
    plan_id = verify_migration_plan(plan)
    _require(
        migration.get("plan_id") == plan_id
        and migration.get("status") == "released"
        and migration.get("official_population_released") is True,
        "Migration is not released",
    )
    _require(
        raw_gate_retry_recovery.get("status")
        == RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
        "Migration history lost raw-PPO exhaustion",
    )
    raw_plan = raw_gate_retry_recovery.get("plan")
    raw_attempts = raw_gate_retry_recovery.get("attempts")
    requirement = raw_gate_retry_recovery.get(
        "objective_migration_requirement"
    )
    _require(
        isinstance(raw_plan, Mapping)
        and isinstance(raw_attempts, list)
        and isinstance(requirement, Mapping),
        "Migration history raw-PPO lineage is incomplete",
    )
    _require(
        verify_recovery_plan(raw_plan) == plan["source_gate_retry_plan_id"]
        and verify_objective_migration_requirement(
            requirement, raw_plan, raw_attempts
        )
        == plan["objective_migration_requirement_id"]
        and dict(requirement) == plan["objective_migration_requirement"],
        "Migration history raw-PPO hand-off drifted",
    )
    attempt = migration.get("attempt")
    _require(isinstance(attempt, Mapping), "Migration history has no attempt")
    contract = attempt.get("contract")
    _require(isinstance(contract, Mapping), "Migration attempt contract is missing")
    verify_migration_attempt_contract(contract, plan)
    _require(
        attempt.get("status") == "qualified_ready_to_release"
        and isinstance(attempt.get("gate_result"), Mapping)
        and attempt["gate_result"].get("passed") is True,
        "Migration history has no successful gate",
    )
    gate_result = attempt["gate_result"]
    _require(
        gate_result.get("policy") == MIGRATION_POLICY
        and gate_result.get("metric") == MIGRATION_REQUIRED_METRIC
        and gate_result.get("threshold") == 0.95
        and gate_result.get("patience") == 5
        and gate_result.get("min_steps") == 1
        and gate_result.get("optimization_surrogate_only") is True
        and gate_result.get("official_payoff_utility_unchanged") is True,
        "Migration history gate contract drifted",
    )
    journal = migration.get("swap_journal")
    _require(
        isinstance(journal, Mapping) and journal.get("phase") == "complete",
        "Migration population swap is incomplete",
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": MIGRATION_POLICY,
        "plan_id": plan_id,
        "stage_label": plan["stage_label"],
        "archived_state_file_sha256": archived_state_file_sha256,
        "raw_gate_retry_recovery": dict(raw_gate_retry_recovery),
        "migration": dict(migration),
    }
    payload["history_entry_id"] = canonical_json_sha256(payload)
    return payload


def verify_migration_history(history: object) -> list[str]:
    if history is None:
        return []
    _require(isinstance(history, list), "Migration history is not a list")
    ids: list[str] = []
    positions: list[int] = []
    for row in history:
        _require(isinstance(row, Mapping), "Migration history row is not an object")
        stored = _sha(row.get("history_entry_id"), "migration history")
        payload = dict(row)
        payload.pop("history_entry_id", None)
        _require(canonical_json_sha256(payload) == stored, "Migration history row drifted")
        rebuilt = build_migration_history_entry(
            row["migration"],
            row["raw_gate_retry_recovery"],
            archived_state_file_sha256=str(row["archived_state_file_sha256"]),
        )
        _require(rebuilt == dict(row), "Migration history proof drifted")
        ids.append(str(row["plan_id"]))
        positions.append(int(str(row["stage_label"])[1:]))
    _require(len(ids) == len(set(ids)), "Duplicate migration plan in history")
    _require(
        len(positions) == len(set(positions)),
        "A stage appears twice in migration history",
    )
    _require(positions == sorted(positions), "Migration history is out of order")
    return ids


def index_migration_history_for_final_audit(
    state: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify and bind every sealed migration to its final retained A stage."""

    completed_shape = (
        state.get("status") == "completed" and state.get("active_stage") is None
    )
    recovery = state.get(RECOVERY_KEY)
    precomplete_d8_retry_shape = False
    if isinstance(recovery, Mapping):
        recovery_plan = recovery.get("plan")
        if isinstance(recovery_plan, Mapping):
            recovery_plan_id = verify_recovery_plan(recovery_plan)
            d8_stage = (state.get("stages") or {}).get("D8")
            d8_release = (
                d8_stage.get("successor_release")
                if isinstance(d8_stage, Mapping)
                else None
            )
            precomplete_d8_retry_shape = (
                state.get("status") == "running"
                and state.get("active_stage") == "D8"
                and recovery.get("status") == "released"
                and recovery.get("official_population_released") is True
                and recovery.get("plan_id") == recovery_plan_id
                and recovery_plan.get("stage_label") == "D8"
                and recovery_plan.get("role") == "defender"
                and isinstance(d8_release, Mapping)
                and d8_release.get("approved") is True
                and d8_release.get("plan_id") == recovery_plan_id
            )
    _require(
        completed_shape or precomplete_d8_retry_shape,
        "Migration-aware final audit requires completed or released-D8 state",
    )
    history = state.get(MIGRATION_HISTORY_KEY)
    verify_migration_history(history)
    if history is None:
        return {}
    _require(MIGRATION_KEY not in state, "Completed state retained live migration state")
    stages = state.get("stages")
    config = state.get("config")
    _require(isinstance(stages, Mapping), "Final state has no stages mapping")
    _require(isinstance(config, Mapping), "Final state has no config mapping")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_row in history:
        row = dict(raw_row)
        label = str(row["stage_label"])
        stage = stages.get(label)
        migration = row["migration"]
        plan = migration["plan"]
        attempt = migration["attempt"]
        contract = attempt["contract"]
        journal = migration["swap_journal"]
        _require(
            isinstance(stage, Mapping)
            and stage.get("status") == "retained"
            and stage.get("transition_state") == "retained",
            f"Migrated final stage is not retained: {label}",
        )
        _require(
            plan.get("frozen_selfplay_config") == config
            and plan.get("frozen_training_implementation_sha256")
            == config.get("training_implementation_sha256"),
            f"Migration frozen configuration drifted: {label}",
        )
        journal_payload = dict(journal)
        journal_id = journal_payload.pop("journal_id", None)
        _require(
            journal.get("phase") == "complete"
            and journal_id == canonical_json_sha256(journal_payload),
            f"Migration swap journal drifted: {label}",
        )
        final_sha = _sha(journal.get("new_sha256"), f"{label} migrated final")
        _require(
            stage.get("attacker_objective_migration_plan_id")
            == plan["plan_id"]
            and stage.get("attacker_objective_migration_attempt_id")
            == attempt["attempt_id"]
            and stage.get("attacker_objective_migration_gate")
            == attempt["gate_result"],
            f"Migration stage identity drifted: {label}",
        )
        _require(
            stage.get("run_dir") == attempt.get("run_dir")
            and stage.get("source_checkpoint")
            == attempt.get("source_checkpoint")
            and stage.get("source_sha256") == attempt.get("source_sha256")
            and stage.get("source_sha256") == final_sha
            and stage.get("population_checkpoint") == journal.get("canonical")
            and stage.get("sha256") == final_sha
            and attempt.get("official_population_checkpoint")
            == journal.get("canonical")
            and attempt.get("official_population_sha256") == final_sha,
            f"Migration final checkpoint binding drifted: {label}",
        )
        _require(
            int(stage.get("requested_max_step", -1))
            == int(contract["per_attempt_budget"])
            and int(stage.get("actual_final_step", 0)) > 0
            and stage.get("stopped_early") is True
            and attempt.get("status") == "qualified_ready_to_release"
            and attempt.get("pruning_complete") is True,
            f"Migration final attempt state drifted: {label}",
        )
        _require(
            stage.get("optimization_objective")
            == binary_joint_objective_contract()
            and stage.get("official_payoff_evaluation")
            == {
                "utility": "frozen_upstream_additive_raw_utility",
                "normalization": "none",
                "binary_surrogate_is_payoff_entry": False,
            },
            f"Migration payoff namespace drifted: {label}",
        )
        displaced = stage.get("displaced_nonqualifying_population")
        _require(
            isinstance(displaced, Mapping)
            and displaced.get("checkpoint") == journal.get("archive")
            and displaced.get("sha256") == journal.get("old_sha256")
            and plan.get("displaced_nonqualifying_population")
            == {
                "checkpoint": journal.get("canonical"),
                "sha256": journal.get("old_sha256"),
            },
            f"Migration displaced population binding drifted: {label}",
        )
        release = stage.get("successor_release")
        _require(
            isinstance(release, Mapping)
            and release.get("approved") is True
            and release.get("migration_plan_id") == plan["plan_id"]
            and release.get("migration_attempt_id") == attempt["attempt_id"]
            and release.get("gate_result") == attempt["gate_result"]
            and release.get("official_payoff_utility_unchanged") is True
            and release.get("binary_surrogate_is_payoff_entry") is False
            and release.get("displaced_nonqualifying_population_preserved")
            is True,
            f"Migration successor release drifted: {label}",
        )
        indexed[label] = row
    marker_labels = {
        str(label)
        for label, stage in stages.items()
        if isinstance(stage, Mapping)
        and stage.get("attacker_objective_migration_plan_id") is not None
    }
    _require(
        marker_labels == set(indexed),
        "Final migration history/stage membership drifted",
    )
    return indexed


def validate_migrated_stage_final_gate(
    state: Mapping[str, Any],
    *,
    stage_label: str,
    validation: Mapping[str, Any],
    expected_budget: int,
    save_steps: int,
    expected_final_sha256: str,
) -> dict[str, Any]:
    """Recompute one migrated stage's binary gate from its sealed history."""

    indexed = index_migration_history_for_final_audit(state)
    row = indexed.get(stage_label)
    _require(row is not None, f"No sealed migration for final stage: {stage_label}")
    migration = row["migration"]
    plan = migration["plan"]
    attempt = migration["attempt"]
    contract = attempt["contract"]
    stage = state["stages"][stage_label]
    _require(
        stage_label.startswith("A")
        and int(expected_budget) == int(contract["per_attempt_budget"])
        and int(save_steps) == int(state["config"]["save_steps"])
        and expected_final_sha256 == stage["source_sha256"],
        f"Migration final gate call binding drifted: {stage_label}",
    )
    _require(
        validation.get("final_checkpoint") == stage["source_checkpoint"]
        and int(validation.get("actual_final_step", -1))
        == int(stage["actual_final_step"])
        and int(validation.get("requested_max_step", -1))
        == int(stage["requested_max_step"]),
        f"Migration validation/state binding drifted: {stage_label}",
    )
    proof = validate_binary_joint_successful_gate(
        validation,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(save_steps),
        expected_final_sha256=str(stage["source_sha256"]),
        expected_initial_sha256=str(contract["trainable_init_sha256"]),
    )
    _require(
        proof == attempt.get("gate_result"),
        f"Sealed migration gate proof drifted: {stage_label}",
    )
    return {
        **proof,
        "role": "attacker",
        "attacker_success_tail": list(proof["tail_values"]),
        "companion_bounds": {},
        "checkpoint_timing": (
            "rollout N evaluates W(N-1); the selected final is post-update WN"
        ),
        "migration_history_entry_id": row["history_entry_id"],
        "migration_plan_id": plan["plan_id"],
        "migration_attempt_id": attempt["attempt_id"],
    }


def validate_migration_manifest(
    migration_manifest: Mapping[str, Any],
    trainer_manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    migration_manifest_path: str,
) -> dict[str, Any]:
    plan_id = verify_migration_plan(plan)
    attempt_id = verify_migration_attempt_contract(contract, plan)
    _require(
        migration_manifest.get("schema_version") == SCHEMA_VERSION
        and migration_manifest.get("policy") == MIGRATION_MANIFEST_POLICY
        and migration_manifest.get("plan_id") == plan_id
        and migration_manifest.get("attempt_id") == attempt_id,
        "Migration manifest identity drifted",
    )
    stored = _sha(
        migration_manifest.get("migration_manifest_sha256"),
        "migration manifest",
    )
    payload = dict(migration_manifest)
    payload.pop("migration_manifest_sha256", None)
    _require(canonical_json_sha256(payload) == stored, "Migration manifest digest drifted")
    _require(
        migration_manifest.get("objective") == binary_joint_objective_contract(),
        "Migration manifest objective drifted",
    )
    expected_manifest_fields = {
        "stage_label": plan["stage_label"],
        "role": "attacker",
        "trainer_run_suffix": contract["trainer_run_suffix"],
        "frozen_training_implementation_sha256": plan[
            "frozen_training_implementation_sha256"
        ],
        "migration_implementation_sha256": plan[
            "migration_implementation_sha256"
        ],
        "effective_trainer_contract": plan["migration_trainer_contract"],
        "single_bounded_attempt": {
            "attempt_number": 1,
            "maximum_attempts": MIGRATION_MAX_ATTEMPTS_PER_STAGE,
            "budget": int(contract["per_attempt_budget"]),
        },
        "optimizer_recipe": {
            "aux_sft": False,
            "raw_advantage_centering": False,
            "raw_advantage_std_scaling": False,
            "baseline": False,
            "trajectory_sum_fixed_scale": TRAJECTORY_SUM_LOSS_SCALE,
            "failure_and_tie_replay_retained": True,
            "fixed_opponent_unchanged": True,
        },
    }
    for key, expected in expected_manifest_fields.items():
        _require(
            migration_manifest.get(key) == expected,
            f"Migration manifest drifted at {key}",
        )
    upstream = migration_manifest.get("patched_upstream_contract")
    _require(isinstance(upstream, Mapping), "Migration upstream patch receipt is missing")
    upstream_payload = dict(upstream)
    upstream_digest = _sha(
        upstream_payload.pop("patch_descriptor_sha256", None),
        "migration upstream patch descriptor",
    )
    _require(
        canonical_json_sha256(upstream_payload) == upstream_digest
        and upstream.get("policy") == MIGRATION_TRAINER_POLICY
        and upstream.get("objective") == binary_joint_objective_contract(),
        "Migration upstream patch receipt drifted",
    )
    runtime = migration_manifest.get("runtime_adapter_mapping")
    _require(isinstance(runtime, Mapping), "Migration runtime adapter mapping is missing")
    for key, expected_path, expected_sha in (
        (
            "trainable",
            contract["trainable_init_checkpoint"],
            contract["trainable_init_sha256"],
        ),
        (
            "fixed_opponent",
            contract["fixed_opponent"]["checkpoint"],
            contract["fixed_opponent"]["sha256"],
        ),
    ):
        row = runtime.get(key)
        _require(
            isinstance(row, Mapping)
            and row.get("original_checkpoint") == expected_path
            and row.get("original_sha256") == expected_sha
            and row.get("runtime_weight_sha256") == expected_sha,
            f"Migration runtime adapter mapping drifted at {key}",
        )
    binding = trainer_manifest.get("attacker_objective_migration")
    _require(
        isinstance(binding, Mapping)
        and binding.get("policy") == MIGRATION_MANIFEST_POLICY
        and binding.get("path") == migration_manifest_path
        and binding.get("sha256") == stored,
        "Trainer manifest is not bound to migration manifest",
    )
    _require(
        binding.get("frozen_core_implementation_sha256")
        == plan["frozen_training_implementation_sha256"][
            "modal_upstream_selfredteam_role_lora.py"
        ]
        and binding.get("effective_function_source_sha256")
        == plan["migration_trainer_contract"][
            "effective_function_source_sha256"
        ]
        and binding.get("patch_descriptor_sha256")
        == plan["migration_trainer_contract"]["patch_descriptor_sha256"]
        and binding.get("patched_upstream_descriptor_sha256")
        == upstream["patch_descriptor_sha256"]
        and binding.get("official_payoff_utility_unchanged") is True,
        "Trainer migration implementation binding drifted",
    )
    _require(
        migration_manifest.get("official_payoff_evaluation", {}).get(
            "binary_surrogate_is_payoff_entry"
        )
        is False,
        "Migration surrogate leaked into official payoff semantics",
    )
    return {
        "passed": True,
        "plan_id": plan_id,
        "attempt_id": attempt_id,
        "migration_manifest_sha256": stored,
        "objective_policy": MIGRATION_MANIFEST_POLICY,
        "official_payoff_utility_unchanged": True,
    }
