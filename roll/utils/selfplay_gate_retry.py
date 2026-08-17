"""Pure fail-closed contracts for retrying a missed self-play stage gate.

The frozen eight-round coordinator intentionally stops when a role exhausts
its configured budget without a five-rollout success streak.  This module
defines an additive recovery protocol: the sole failed raw-PPO retry is an
immutable candidate, the original stage claim remains part of the lineage, and
an atomically exchanged canonical population directory is released only if
that bounded retry proves the original gate.  A second miss fails closed with
a hash-bound objective-migration requirement.

No Modal, torch, PEFT, or trainer imports belong here.  The remote entrypoint
is :mod:`modal_role_lora_selfplay8_gate_retry`.
"""

from __future__ import annotations

import ast
import ctypes
import errno
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
RECOVERY_KEY = "gate_retry_recovery_v1"
RECOVERY_HISTORY_KEY = "gate_retry_recovery_history_v1"
RECOVERY_POLICY = "same-label-gate-retry-v1"
PPO_ONLY_RECOVERY_TRAINER_POLICY = "same-label-ppo-only-dynamic-clone-v1"
RAW_PPO_RETRY_POLICY = "single-bounded-raw-ppo-retry-v1"
RAW_PPO_MAX_ATTEMPTS_PER_STAGE = 1
RAW_PPO_EXHAUSTED_RECOVERY_STATUS = "raw_ppo_exhausted"
OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS = (
    "stage_objective_migration_required"
)
OBJECTIVE_MIGRATION_REQUIREMENT_POLICY = (
    "binary-joint-objective-migration-required-v1"
)
_STAGE_LABEL_RE = re.compile(r"^[AD]([2-8])$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AT_FDCWD = -100
_RENAME_EXCHANGE = 2


def canonical_json_sha256(value: object) -> str:
    """Hash one JSON-compatible value without formatting ambiguity."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def bounded_raw_ppo_retry_policy() -> dict[str, Any]:
    """Return the exact immutable policy for one same-stage PPO-only retry."""

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": RAW_PPO_RETRY_POLICY,
        "max_attempts_per_stage": RAW_PPO_MAX_ATTEMPTS_PER_STAGE,
        "per_attempt_budget": "the frozen role-specific stage budget",
        "objective": "unchanged frozen role reward and advantage semantics",
        "official_payoff_evaluation": (
            "unchanged official environment reward with no normalization"
        ),
        "sft": "disabled from the first retry optimizer step",
        "early_stop_threshold": 0.95,
        "early_stop_patience": 5,
        "early_stop_min_steps": 1,
        "earliest_possible_stop_step": 5,
        "single_checkpoint_change_proof": (
            "final checkpoint SHA256 must differ from the hash-bound initializer"
        ),
        "on_exhaustion": "fail closed for binary joint objective migration",
    }


def build_ppo_only_recovery_trainer_source(
    frozen_core_source: str,
) -> tuple[str, dict[str, Any]]:
    """Build the narrowly patched, additive PPO-only recovery trainer.

    The frozen module is parsed but never edited.  Exactly three validation
    guards are narrowed in the cloned function: ``stop_after_step=0`` becomes
    the explicit no-SFT sentinel, joint-signed raw defender advantages are
    allowed under the exact PPO-only recipe, and that same hash-bound recipe
    alone may start its five-step gate at optimizer step one.
    """

    module = ast.parse(frozen_core_source)
    matches = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "train_upstream_attacker_lora_fixed_seed"
    ]
    _require(len(matches) == 1, "Frozen role trainer function is not unique")
    function = matches[0]
    _require(isinstance(function, ast.FunctionDef), "Role trainer must be synchronous")
    function.decorator_list = []
    function.name = "_effective_gate_retry_ppo_only_train"

    stop_message = "postfill_cot_stop_after_step requires enable_aux_sft=True"
    raw_message = (
        "defender_raw_reinforce_advantages is restricted to defender v2 "
        "continuation training"
    )
    minimum_message = (
        "early_stop_min_steps must be at least early_stop_patience"
    )
    replacements = {
        "ppo_only_stop_zero": 0,
        "ppo_only_joint_signed_raw": 0,
        "ppo_only_gate_from_step_one": 0,
    }

    def direct_raise_messages(node: ast.If) -> list[str]:
        values: list[str] = []
        for statement in node.body:
            if not isinstance(statement, ast.Raise) or not isinstance(
                statement.exc, ast.Call
            ):
                continue
            for argument in statement.exc.args:
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    values.append(argument.value)
        return values

    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        messages = direct_raise_messages(node)
        if stop_message in messages:
            node.test = ast.parse(
                "not enable_aux_sft and not ("
                "postfill_cot_stop_after_step == 0 and "
                "not role_specific_aux_sft and not v2_continuation_sft and "
                "defender_sft_optimizer_slots_per_rollout == 0)",
                mode="eval",
            ).body
            replacements["ppo_only_stop_zero"] += 1
        if raw_message in messages:
            node.test = ast.parse(
                "defender_raw_reinforce_advantages and not ("
                "(v2_continuation_sft and train_role == 'defender') or ("
                "train_role == 'defender' and v2_runtime and "
                "not enable_aux_sft and not role_specific_aux_sft and "
                "not v2_continuation_sft and "
                "postfill_cot_stop_after_step == 0 and "
                "defender_sft_optimizer_slots_per_rollout == 0))",
                mode="eval",
            ).body
            replacements["ppo_only_joint_signed_raw"] += 1
        if minimum_message in messages:
            node.test = ast.parse(
                "early_stop_min_steps < early_stop_patience and not ("
                "early_stop_min_steps == 1 and early_stop_patience == 5 and "
                "early_stop_threshold == 0.95 and v2_runtime and "
                "not enable_aux_sft and not role_specific_aux_sft and "
                "not v2_continuation_sft and "
                "postfill_cot_stop_after_step == 0 and "
                "defender_sft_optimizer_slots_per_rollout == 0)",
                mode="eval",
            ).body
            replacements["ppo_only_gate_from_step_one"] += 1
    _require(
        replacements == {
            "ppo_only_stop_zero": 1,
            "ppo_only_joint_signed_raw": 1,
            "ppo_only_gate_from_step_one": 1,
        },
        f"PPO-only recovery patch surface drifted: {replacements}",
    )
    effective_module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    effective_source = ast.unparse(effective_module) + "\n"
    compile(effective_source, "<ppo-only-recovery-trainer>", "exec")
    effective_sha = bytes_sha256(effective_source.encode("utf-8"))
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": PPO_ONLY_RECOVERY_TRAINER_POLICY,
        "frozen_core_source_sha256": bytes_sha256(
            frozen_core_source.encode("utf-8")
        ),
        "effective_function_source_sha256": effective_sha,
        "patch_replacements": replacements,
        "ppo_only_recipe": {
            "enable_aux_sft": False,
            "role_specific_aux_sft": False,
            "v2_continuation_sft": False,
            "postfill_cot_stop_after_step": 0,
            "defender_sft_optimizer_slots_per_rollout": 0,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 1,
        },
    }
    descriptor["patch_descriptor_sha256"] = canonical_json_sha256(descriptor)
    return effective_source, descriptor


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_sha(value: object, label: str) -> str:
    text = str(value or "")
    _require(bool(_SHA256_RE.fullmatch(text)), f"Invalid {label} SHA256")
    return text


def verify_ppo_only_recovery_trainer_contract(
    contract: Mapping[str, Any],
    *,
    expected_frozen_core_sha256: str | None = None,
) -> str:
    """Verify the effective trainer identity separately from frozen code."""

    _require(
        contract.get("schema_version") == SCHEMA_VERSION,
        "Bad PPO-only recovery trainer schema",
    )
    _require(
        contract.get("policy") == PPO_ONLY_RECOVERY_TRAINER_POLICY,
        "Bad PPO-only recovery trainer policy",
    )
    frozen_sha = _require_sha(
        contract.get("frozen_core_source_sha256"),
        "PPO-only frozen core source",
    )
    if expected_frozen_core_sha256 is not None:
        _require(
            frozen_sha == _require_sha(
                expected_frozen_core_sha256,
                "expected PPO-only frozen core source",
            ),
            "PPO-only trainer frozen core identity drifted",
        )
    _require_sha(
        contract.get("effective_function_source_sha256"),
        "PPO-only effective function source",
    )
    stored = _require_sha(
        contract.get("patch_descriptor_sha256"),
        "PPO-only patch descriptor",
    )
    payload = dict(contract)
    payload.pop("patch_descriptor_sha256", None)
    _require(
        canonical_json_sha256(payload) == stored,
        "PPO-only patch descriptor digest drifted",
    )
    _require(
        contract.get("patch_replacements")
        == {
            "ppo_only_stop_zero": 1,
            "ppo_only_joint_signed_raw": 1,
            "ppo_only_gate_from_step_one": 1,
        },
        "PPO-only patch replacement surface drifted",
    )
    _require(
        contract.get("ppo_only_recipe")
        == {
            "enable_aux_sft": False,
            "role_specific_aux_sft": False,
            "v2_continuation_sft": False,
            "postfill_cot_stop_after_step": 0,
            "defender_sft_optimizer_slots_per_rollout": 0,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 1,
        },
        "PPO-only trainer recipe drifted",
    )
    return stored


def validate_ppo_only_recovery_manifest(
    recovery_manifest: Mapping[str, Any],
    trainer_manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    recovery_manifest_path: str,
) -> dict[str, Any]:
    """Validate the immutable effective-trainer receipt and PEFT routing."""

    plan_id = verify_recovery_plan(plan)
    attempt_id = verify_attempt_contract(contract, plan)
    _require(
        recovery_manifest.get("schema_version") == SCHEMA_VERSION,
        "Bad PPO-only recovery manifest schema",
    )
    _require(
        recovery_manifest.get("policy") == PPO_ONLY_RECOVERY_TRAINER_POLICY,
        "Bad PPO-only recovery manifest policy",
    )
    expected_scalars = {
        "plan_id": plan_id,
        "attempt_id": attempt_id,
        "stage_label": contract["stage_label"],
        "role": contract["role"],
        "trainer_run_suffix": contract["trainer_run_suffix"],
    }
    for key, expected in expected_scalars.items():
        _require(
            recovery_manifest.get(key) == expected,
            f"PPO-only recovery manifest drifted at {key}",
        )
    _require(
        recovery_manifest.get("frozen_training_implementation_sha256")
        == contract["frozen_training_implementation_sha256"],
        "PPO-only frozen implementation identity drifted",
    )
    _require(
        recovery_manifest.get("recovery_implementation_sha256")
        == contract["recovery_implementation_sha256"],
        "PPO-only additive implementation identity drifted",
    )
    _require(
        recovery_manifest.get("bounded_raw_ppo_retry_policy")
        == contract["bounded_raw_ppo_retry_policy"]
        == plan["bounded_raw_ppo_retry_policy"],
        "PPO-only bounded retry policy drifted",
    )
    effective = recovery_manifest.get("effective_trainer_contract")
    _require(
        effective == contract["recovery_trainer_contract"],
        "PPO-only effective trainer contract drifted",
    )
    core_sha = contract["frozen_training_implementation_sha256"].get(
        "modal_upstream_selfredteam_role_lora.py"
    )
    verify_ppo_only_recovery_trainer_contract(
        effective,
        expected_frozen_core_sha256=str(core_sha or ""),
    )

    role = str(contract["role"])
    runtime_trainable = f"/tmp/{role}_lora_init_compatible"
    runtime_fixed = "/tmp/fixed_opponent_lora_compatible"
    expected_mapping = {
        "trainable": {
            "original_checkpoint": contract["trainable_init_checkpoint"],
            "original_sha256": contract["trainable_init_sha256"],
            "runtime_compatible_checkpoint": runtime_trainable,
            "runtime_weight_sha256": contract["trainable_init_sha256"],
        },
        "fixed_opponent": {
            "original_checkpoint": contract["fixed_opponent"]["checkpoint"],
            "original_sha256": contract["fixed_opponent"]["sha256"],
            "runtime_compatible_checkpoint": runtime_fixed,
            "runtime_weight_sha256": contract["fixed_opponent"]["sha256"],
        },
    }
    _require(
        recovery_manifest.get("runtime_adapter_mapping") == expected_mapping,
        "PPO-only original/runtime adapter mapping drifted",
    )
    expected_recipe = {
        **dict(effective["ppo_only_recipe"]),
        "monitor_reference_kl": False,
        "optimizer_state": "cold_on_unique_attempt_suffix",
        "cold_container_replay_runtime_adapter_policy": (
            "rebuild_missing_or_incomplete_with_frozen_"
            "prepare_peft_compatible_adapter_before_digest"
        ),
        "fixed_opponent_unchanged": True,
        "reward_and_advantage_semantics_unchanged": True,
    }
    _require(
        recovery_manifest.get("ppo_only_recipe") == expected_recipe,
        "PPO-only recovery recipe receipt drifted",
    )
    identity = recovery_manifest.get("implementation_identity")
    _require(
        identity
        == {
            "frozen_core": {
                "path": "modal_upstream_selfredteam_role_lora.py",
                "sha256": core_sha,
            },
            "additive_recovery_sources": contract[
                "recovery_implementation_sha256"
            ],
            "effective_dynamic_function": {
                "policy": PPO_ONLY_RECOVERY_TRAINER_POLICY,
                "source_sha256": effective["effective_function_source_sha256"],
                "patch_descriptor_sha256": effective[
                    "patch_descriptor_sha256"
                ],
            },
        },
        "PPO-only implementation layers are not explicitly bound",
    )
    stored_manifest_sha = _require_sha(
        recovery_manifest.get("recovery_manifest_sha256"),
        "PPO-only recovery manifest",
    )
    manifest_payload = dict(recovery_manifest)
    manifest_payload.pop("recovery_manifest_sha256", None)
    _require(
        canonical_json_sha256(manifest_payload) == stored_manifest_sha,
        "PPO-only recovery manifest digest drifted",
    )
    expected_binding = {
        "policy": PPO_ONLY_RECOVERY_TRAINER_POLICY,
        "path": recovery_manifest_path,
        "sha256": stored_manifest_sha,
        "frozen_core_implementation_sha256": core_sha,
        "effective_function_source_sha256": effective[
            "effective_function_source_sha256"
        ],
        "patch_descriptor_sha256": effective["patch_descriptor_sha256"],
    }
    _require(
        trainer_manifest.get("gate_retry_effective_implementation")
        == expected_binding,
        "Frozen trainer manifest/effective implementation binding drifted",
    )
    return {
        "passed": True,
        "plan_id": plan_id,
        "attempt_id": attempt_id,
        "recovery_manifest_sha256": stored_manifest_sha,
        "runtime_adapter_mapping": expected_mapping,
        "implementation_identity": identity,
    }


def _stage_by_label(
    schedule: Sequence[Mapping[str, Any] | Any],
    label: str,
) -> tuple[int, dict[str, Any]]:
    for position, raw in enumerate(schedule):
        value = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        if value.get("label") == label:
            return position, value
    raise RuntimeError(f"Stage is absent from the frozen schedule: {label}")


def validate_recovery_eligibility(
    state: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    """Validate the only state from which same-label recovery may start."""

    _require(
        state.get("schema_version") == 1,
        "Gate retry requires self-play state schema 1",
    )
    _require(
        state.get("status") == "stage_target_not_reached",
        "Gate retry requires stage_target_not_reached",
    )
    label = str(state.get("active_stage") or "")
    _require(bool(_STAGE_LABEL_RE.fullmatch(label)), "Invalid active retry stage")
    position, spec = _stage_by_label(schedule, label)
    expected_schedule = [
        raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        for raw in schedule
    ]
    _require(
        state.get("schedule") == expected_schedule,
        "Durable self-play schedule differs from the retry schedule",
    )
    stages = state.get("stages")
    _require(isinstance(stages, Mapping), "Self-play state has no stages map")
    stage = stages.get(label)
    _require(isinstance(stage, Mapping), f"Missing failed stage {label}")
    for key, value in spec.items():
        _require(stage.get(key) == value, f"Failed stage spec drifted at {key}")
    _require(
        stage.get("transition_state") == "retained"
        and stage.get("status") == "retained",
        f"Failed stage {label} is not durably retained",
    )
    _require(stage.get("stopped_early") is False, f"{label} already passed its gate")
    release = stage.get("successor_release")
    _require(
        isinstance(release, Mapping) and release.get("approved") is False,
        f"{label} does not have an explicit rejected successor release",
    )
    claim_id = _require_sha(stage.get("spawn_claim_id"), f"{label} spawn claim")
    population_sha = _require_sha(stage.get("sha256"), f"{label} population")
    population_path = str(stage.get("population_checkpoint") or "")
    _require(bool(population_path), f"{label} has no population checkpoint")

    if position + 1 < len(schedule):
        next_raw = schedule[position + 1]
        next_spec = (
            next_raw.to_dict() if hasattr(next_raw, "to_dict") else dict(next_raw)
        )
        _require(
            next_spec["label"] not in stages,
            f"Successor {next_spec['label']} already exists",
        )

    original_parent_label = str(spec["trainable_parent"])
    fixed_label = str(spec["fixed_opponent"])
    _require(original_parent_label != "base", "D1 is not retryable by this protocol")
    original_parent = stages.get(original_parent_label)
    fixed = stages.get(fixed_label)
    for dependency_label, dependency in (
        (original_parent_label, original_parent),
        (fixed_label, fixed),
    ):
        _require(
            isinstance(dependency, Mapping)
            and dependency.get("status") == "retained"
            and dependency.get("transition_state") == "retained",
            f"Dependency is not retained: {dependency_label}",
        )
        _require_sha(dependency.get("sha256"), dependency_label)
        _require(
            bool(dependency.get("population_checkpoint")),
            f"Dependency has no population path: {dependency_label}",
        )

    config = state.get("config")
    _require(isinstance(config, Mapping), "Self-play state has no frozen config")
    role = str(spec["role"])
    expected_budget = int(
        config["attacker_max_steps" if role == "attacker" else "defender_max_steps"]
    )
    _require(expected_budget > 0, "Retry budget must be positive")
    _require(
        int(stage.get("actual_final_step", -1)) == expected_budget
        and int(stage.get("requested_max_step", -1)) == expected_budget,
        f"{label} is not a completed budget-exhausted attempt",
    )
    return {
        "label": label,
        "position": position,
        "stage_spec": spec,
        "stage": dict(stage),
        "spawn_claim_id": claim_id,
        "population_checkpoint": population_path,
        "population_sha256": population_sha,
        "original_parent": {
            "label": original_parent_label,
            "checkpoint": str(original_parent["population_checkpoint"]),
            "sha256": str(original_parent["sha256"]),
        },
        "fixed_opponent": {
            "label": fixed_label,
            "checkpoint": str(fixed["population_checkpoint"]),
            "sha256": str(fixed["sha256"]),
        },
        "budget": expected_budget,
        "role": role,
    }


def build_recovery_plan(
    *,
    state: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    initial_state_file_sha256: str,
    frozen_training_sha256: Mapping[str, str],
    recovery_implementation_sha256: Mapping[str, str],
    recovery_trainer_contract: Mapping[str, Any],
    plan_path: str,
    original_failure_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the immutable write-ahead identity for one failed stage."""

    _require_sha(initial_state_file_sha256, "initial state file")
    for name, digest in frozen_training_sha256.items():
        _require_sha(digest, f"frozen source {name}")
    for name, digest in recovery_implementation_sha256.items():
        _require_sha(digest, f"recovery source {name}")
    trainer_contract = dict(recovery_trainer_contract)
    verify_ppo_only_recovery_trainer_contract(
        trainer_contract,
        expected_frozen_core_sha256=str(
            frozen_training_sha256.get(
                "modal_upstream_selfredteam_role_lora.py",
                "",
            )
        ),
    )
    retry_policy = bounded_raw_ppo_retry_policy()
    _require(
        float(state["config"].get("early_stop_threshold", -1.0))
        == float(retry_policy["early_stop_threshold"])
        and int(state["config"].get("early_stop_patience", -1))
        == int(retry_policy["early_stop_patience"]),
        "Frozen stage gate differs from the bounded raw-PPO retry gate",
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": RECOVERY_POLICY,
        "run_suffix": str(state.get("run_suffix") or ""),
        "stage_label": eligibility["label"],
        "stage_spec": dict(eligibility["stage_spec"]),
        "role": eligibility["role"],
        "per_attempt_budget": int(eligibility["budget"]),
        "original_stage_spawn_claim_id": eligibility["spawn_claim_id"],
        "original_nonqualifying_population": {
            "checkpoint": eligibility["population_checkpoint"],
            "sha256": eligibility["population_sha256"],
        },
        "original_trainable_parent": dict(eligibility["original_parent"]),
        "fixed_opponent": dict(eligibility["fixed_opponent"]),
        "initial_state_file_sha256": initial_state_file_sha256,
        "frozen_training_implementation_sha256": dict(frozen_training_sha256),
        "recovery_implementation_sha256": dict(
            recovery_implementation_sha256
        ),
        "recovery_trainer_contract": trainer_contract,
        "bounded_raw_ppo_retry_policy": retry_policy,
        "frozen_selfplay_config": dict(state["config"]),
        "original_failure_evidence": dict(original_failure_evidence),
        "plan_path": plan_path,
        "official_population_rule": (
            "canonical population/<label> is never consumed by a successor "
            "until a five-step gate passes; every displaced/failed adapter is "
            "preserved under population_attempts"
        ),
    }
    payload["plan_id"] = canonical_json_sha256(payload)
    return payload


def verify_recovery_plan(plan: Mapping[str, Any]) -> str:
    _require(plan.get("schema_version") == SCHEMA_VERSION, "Bad recovery plan schema")
    _require(plan.get("policy") == RECOVERY_POLICY, "Bad recovery policy")
    stored = _require_sha(plan.get("plan_id"), "recovery plan")
    payload = dict(plan)
    payload.pop("plan_id", None)
    _require(canonical_json_sha256(payload) == stored, "Recovery plan digest drifted")
    _require(
        bool(_STAGE_LABEL_RE.fullmatch(str(plan.get("stage_label") or ""))),
        "Recovery plan stage label drifted",
    )
    role = str(plan.get("role") or "")
    _require(
        role
        == ("attacker" if str(plan["stage_label"]).startswith("A") else "defender"),
        "Recovery plan stage role drifted",
    )
    _require(int(plan.get("per_attempt_budget", 0)) > 0, "Bad retry budget")
    _require(
        plan.get("bounded_raw_ppo_retry_policy")
        == bounded_raw_ppo_retry_policy(),
        "Bounded raw-PPO retry policy drifted",
    )
    _require_sha(plan.get("initial_state_file_sha256"), "initial state file")
    _require_sha(
        plan.get("original_stage_spawn_claim_id"),
        "original stage spawn claim",
    )
    stage_spec = plan.get("stage_spec")
    _require(
        isinstance(stage_spec, Mapping)
        and stage_spec.get("label") == plan["stage_label"]
        and stage_spec.get("role") == role,
        "Recovery plan stage spec drifted",
    )
    original = plan.get("original_nonqualifying_population")
    parent = plan.get("original_trainable_parent")
    fixed_opponent = plan.get("fixed_opponent")
    for label, provenance in (
        ("original population", original),
        ("original trainable parent", parent),
        ("fixed opponent", fixed_opponent),
    ):
        _require(isinstance(provenance, Mapping), f"Missing {label} provenance")
        _require(bool(provenance.get("checkpoint")), f"Missing {label} path")
        _require_sha(provenance.get("sha256"), label)
    _require(
        parent.get("label") == stage_spec.get("trainable_parent")
        and fixed_opponent.get("label") == stage_spec.get("fixed_opponent"),
        "Recovery plan parent/opponent labels drifted",
    )
    frozen = plan.get("frozen_training_implementation_sha256")
    _require(isinstance(frozen, Mapping), "Recovery plan has no frozen hashes")
    for name, digest in frozen.items():
        _require_sha(digest, f"frozen source {name}")
    additive = plan.get("recovery_implementation_sha256")
    _require(
        isinstance(additive, Mapping) and bool(additive),
        "Recovery plan has no additive hashes",
    )
    for name, digest in additive.items():
        _require_sha(digest, f"additive source {name}")
    _require(
        isinstance(plan.get("frozen_selfplay_config"), Mapping),
        "Recovery plan has no frozen self-play config",
    )
    retry_policy = plan["bounded_raw_ppo_retry_policy"]
    frozen_config = plan["frozen_selfplay_config"]
    _require(
        float(frozen_config.get("early_stop_threshold", -1.0))
        == float(retry_policy["early_stop_threshold"])
        and int(frozen_config.get("early_stop_patience", -1))
        == int(retry_policy["early_stop_patience"]),
        "Recovery plan frozen gate differs from its bounded retry gate",
    )
    trainer_contract = plan.get("recovery_trainer_contract")
    _require(
        isinstance(trainer_contract, Mapping),
        "Recovery plan has no effective trainer contract",
    )
    verify_ppo_only_recovery_trainer_contract(
        trainer_contract,
        expected_frozen_core_sha256=str(
            frozen.get("modal_upstream_selfredteam_role_lora.py") or ""
        ),
    )
    return stored


def build_attempt_contract(
    plan: Mapping[str, Any],
    *,
    attempt_number: int,
    trainable_init_checkpoint: str,
    trainable_init_sha256: str,
    trainer_run_suffix: str,
    contract_path: str,
) -> dict[str, Any]:
    """Bind one retry to its exact initializer, opponent, and frozen code."""

    plan_id = verify_recovery_plan(plan)
    _require(attempt_number >= 1, "Attempt number must be positive")
    _require(
        attempt_number
        <= int(
            plan["bounded_raw_ppo_retry_policy"][
                "max_attempts_per_stage"
            ]
        ),
        "Attempt number exceeds the bounded raw-PPO retry policy",
    )
    init_sha = _require_sha(trainable_init_sha256, "attempt initializer")
    _require(bool(trainable_init_checkpoint), "Attempt initializer path is empty")
    _require(bool(trainer_run_suffix), "Attempt trainer suffix is empty")
    fixed = dict(plan["fixed_opponent"])
    _require_sha(fixed.get("sha256"), "attempt fixed opponent")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": RECOVERY_POLICY,
        "plan_id": plan_id,
        "stage_label": plan["stage_label"],
        "role": plan["role"],
        "attempt_number": int(attempt_number),
        "trainable_init_checkpoint": trainable_init_checkpoint,
        "trainable_init_sha256": init_sha,
        "fixed_opponent": fixed,
        "original_stage_spawn_claim_id": plan[
            "original_stage_spawn_claim_id"
        ],
        "per_attempt_budget": int(plan["per_attempt_budget"]),
        "bounded_raw_ppo_retry_policy": dict(
            plan["bounded_raw_ppo_retry_policy"]
        ),
        "trainer_run_suffix": trainer_run_suffix,
        "optimizer_policy": (
            "cold optimizer/scheduler for the first entry of this unique "
            "attempt suffix; hash-bound same-suffix preemption replay may "
            "restore LoRA weights and scheduler position but not moments"
        ),
        "frozen_training_implementation_sha256": dict(
            plan["frozen_training_implementation_sha256"]
        ),
        "recovery_implementation_sha256": dict(
            plan["recovery_implementation_sha256"]
        ),
        "recovery_trainer_contract": dict(plan["recovery_trainer_contract"]),
        "contract_path": contract_path,
    }
    payload["attempt_id"] = canonical_json_sha256(payload)
    return payload


def verify_attempt_contract(
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    plan_id = verify_recovery_plan(plan)
    _require(contract.get("schema_version") == SCHEMA_VERSION, "Bad attempt schema")
    _require(contract.get("policy") == RECOVERY_POLICY, "Bad attempt policy")
    _require(contract.get("plan_id") == plan_id, "Attempt plan identity drifted")
    stored = _require_sha(contract.get("attempt_id"), "attempt")
    payload = dict(contract)
    payload.pop("attempt_id", None)
    _require(canonical_json_sha256(payload) == stored, "Attempt digest drifted")
    expected_plan_fields = {
        "stage_label": plan["stage_label"],
        "role": plan["role"],
        "original_stage_spawn_claim_id": plan[
            "original_stage_spawn_claim_id"
        ],
        "per_attempt_budget": plan["per_attempt_budget"],
        "bounded_raw_ppo_retry_policy": plan[
            "bounded_raw_ppo_retry_policy"
        ],
        "fixed_opponent": plan["fixed_opponent"],
        "frozen_training_implementation_sha256": plan[
            "frozen_training_implementation_sha256"
        ],
        "recovery_implementation_sha256": plan[
            "recovery_implementation_sha256"
        ],
        "recovery_trainer_contract": plan["recovery_trainer_contract"],
    }
    for key, expected in expected_plan_fields.items():
        _require(
            contract.get(key) == expected,
            f"Attempt/plan provenance drifted at {key}",
        )
    attempt_number = int(contract.get("attempt_number", 0))
    _require(attempt_number >= 1, "Attempt number must be positive")
    _require(
        attempt_number
        <= int(
            plan["bounded_raw_ppo_retry_policy"][
                "max_attempts_per_stage"
            ]
        ),
        "Attempt number exceeds the bounded raw-PPO retry policy",
    )
    _require(
        bool(contract.get("trainable_init_checkpoint"))
        and bool(contract.get("trainer_run_suffix"))
        and bool(contract.get("contract_path")),
        "Attempt path/suffix provenance is incomplete",
    )
    _require_sha(contract.get("trainable_init_sha256"), "attempt initializer")
    return stored


def validate_raw_ppo_attempt_capacity(
    plan: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> int:
    """Fail closed before a second raw-PPO attempt can be claimed."""

    verify_recovery_plan(plan)
    _require(
        not isinstance(attempts, (str, bytes)),
        "Raw-PPO attempts must be a sequence of objects",
    )
    limit = int(
        plan["bounded_raw_ppo_retry_policy"]["max_attempts_per_stage"]
    )
    _require(
        len(attempts) < limit,
        "Bounded raw-PPO retry is exhausted; objective migration is required",
    )
    return limit - len(attempts)


def build_objective_migration_requirement(
    plan: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal the sole failed retry as the initializer for a new objective."""

    plan_id = verify_recovery_plan(plan)
    _require(
        not isinstance(attempts, (str, bytes)),
        "Raw-PPO attempts must be a sequence of objects",
    )
    limit = int(
        plan["bounded_raw_ppo_retry_policy"]["max_attempts_per_stage"]
    )
    _require(
        len(attempts) == limit,
        "Objective migration requires the exact raw-PPO retry budget",
    )
    for index, attempt in enumerate(attempts, start=1):
        _require(isinstance(attempt, Mapping), "Raw-PPO attempt is not an object")
        contract = attempt.get("contract")
        _require(isinstance(contract, Mapping), "Raw-PPO attempt has no contract")
        attempt_id = verify_attempt_contract(contract, plan)
        _require(
            attempt.get("attempt_id") == attempt_id
            and int(attempt.get("attempt_number", -1)) == index
            and int(contract.get("attempt_number", -1)) == index,
            "Raw-PPO attempt ordering or identity drifted",
        )
    last = attempts[-1]
    _require(
        last.get("status") == "gate_not_reached"
        and last.get("pruning_complete") is True,
        "Objective migration requires a fully archived failed raw-PPO retry",
    )
    gate = last.get("gate_result")
    _require(
        isinstance(gate, Mapping)
        and gate.get("passed") is True
        and gate.get("classification")
        == "gate_not_reached_after_complete_budget",
        "Objective migration has no complete raw-PPO failure proof",
    )
    checkpoint = str(last.get("candidate_checkpoint") or "")
    _require(bool(checkpoint), "Objective migration initializer path is empty")
    checkpoint_sha256 = _require_sha(
        last.get("candidate_sha256"),
        "objective migration initializer",
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": OBJECTIVE_MIGRATION_REQUIREMENT_POLICY,
        "plan_id": plan_id,
        "stage_label": plan["stage_label"],
        "role": plan["role"],
        "completed_raw_ppo_attempts": len(attempts),
        "raw_ppo_attempt_limit": limit,
        "source_attempt_id": last["attempt_id"],
        "trainable_init_checkpoint": checkpoint,
        "trainable_init_sha256": checkpoint_sha256,
        "fixed_opponent": dict(plan["fixed_opponent"]),
        "required_next_objective": (
            "attacker_binary_joint_goal_and_cot_raw_no_center_no_std"
            if plan["role"] == "attacker"
            else "separately_versioned_gate_aligned_objective"
        ),
        "required_next_objective_contract": (
            {
                "positive_iff": (
                    "attacker goal success AND CoT valid AND not tie"
                ),
                "optimization_reward": {"positive": 1.0, "negative": -1.0},
                "advantage": "raw_no_center_no_std",
                "policy_loss": "episode_balanced",
                "aux_sft": "disabled from the first optimizer step",
            }
            if plan["role"] == "attacker"
            else {
                "status": "must be designed and separately versioned",
                "aux_sft": "disabled from the first optimizer step",
            }
        ),
        "official_payoff_evaluation": (
            "unchanged official environment reward with no normalization; "
            "the optimization surrogate must not replace or mix with the "
            "historical PSRO payoff matrix"
        ),
        "sft": "permanently disabled",
        "successor_release": "forbidden until a new objective passes the gate",
    }
    payload["requirement_id"] = canonical_json_sha256(payload)
    return payload


def verify_objective_migration_requirement(
    requirement: Mapping[str, Any],
    plan: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> str:
    """Verify the durable handoff without accepting caller-provided defaults."""

    expected = build_objective_migration_requirement(plan, attempts)
    _require(dict(requirement) == expected, "Objective migration handoff drifted")
    return str(expected["requirement_id"])


def build_recovery_history_entry(
    recovery: Mapping[str, Any],
    *,
    archived_state_file_sha256: str,
) -> dict[str, Any]:
    """Seal one released stage recovery before another stage takes the slot."""

    state_sha = _require_sha(archived_state_file_sha256, "archived state file")
    plan = recovery.get("plan")
    _require(isinstance(plan, Mapping), "Released recovery has no plan")
    plan_id = verify_recovery_plan(plan)
    _require(recovery.get("plan_id") == plan_id, "Released recovery plan drifted")
    _require(recovery.get("status") == "released", "Recovery is not released")
    _require(
        recovery.get("official_population_released") is True,
        "Recovery official population was not released",
    )
    attempts = recovery.get("attempts")
    _require(isinstance(attempts, list) and attempts, "Released recovery has no attempts")
    for index, attempt in enumerate(attempts):
        _require(isinstance(attempt, Mapping), "Recovery attempt is not an object")
        contract = attempt.get("contract")
        _require(isinstance(contract, Mapping), "Recovery attempt has no contract")
        attempt_id = verify_attempt_contract(contract, plan)
        _require(
            int(contract.get("attempt_number", -1)) == index + 1,
            "Recovery attempt ordering drifted",
        )
        _require(attempt.get("attempt_id") == attempt_id, "Recovery attempt drifted")
        expected_status = (
            "qualified_ready_to_release"
            if index == len(attempts) - 1
            else "gate_not_reached"
        )
        _require(
            attempt.get("status") == expected_status,
            "Recovery attempt terminal ordering drifted",
        )
    final_attempt = attempts[-1]
    gate = final_attempt.get("gate_result")
    _require(
        isinstance(gate, Mapping) and gate.get("passed") is True,
        "Released recovery has no passing final gate",
    )
    journal = recovery.get("swap_journal")
    _require(
        isinstance(journal, Mapping) and journal.get("phase") == "complete",
        "Released recovery population swap is incomplete",
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": RECOVERY_POLICY,
        "plan_id": plan_id,
        "stage_label": plan["stage_label"],
        "archived_state_file_sha256": state_sha,
        "recovery": dict(recovery),
    }
    payload["history_entry_id"] = canonical_json_sha256(payload)
    return payload


def verify_recovery_history(
    history: object,
) -> list[str]:
    """Verify an ordered, duplicate-free list of sealed stage recoveries."""

    if history is None:
        return []
    _require(isinstance(history, list), "Gate-retry history is not a list")
    plan_ids: list[str] = []
    stage_labels: list[str] = []
    for raw in history:
        _require(isinstance(raw, Mapping), "Gate-retry history row is not an object")
        stored = _require_sha(raw.get("history_entry_id"), "history entry")
        payload = dict(raw)
        payload.pop("history_entry_id", None)
        _require(canonical_json_sha256(payload) == stored, "History entry drifted")
        recovery = raw.get("recovery")
        _require(isinstance(recovery, Mapping), "History entry has no recovery")
        rebuilt = build_recovery_history_entry(
            recovery,
            archived_state_file_sha256=str(raw.get("archived_state_file_sha256") or ""),
        )
        _require(rebuilt == dict(raw), "History recovery proof drifted")
        plan_ids.append(str(raw["plan_id"]))
        stage_labels.append(str(raw["stage_label"]))
    _require(len(plan_ids) == len(set(plan_ids)), "Duplicate recovery plan in history")
    _require(
        len(stage_labels) == len(set(stage_labels)),
        "A stage appears twice in recovery history",
    )
    _require(
        all(_STAGE_LABEL_RE.fullmatch(label) for label in stage_labels),
        "Recovery history contains an invalid stage label",
    )
    stage_positions = [
        2 * (int(label[1:]) - 2) + (1 if label.startswith("D") else 0)
        for label in stage_labels
    ]
    _require(
        stage_positions == sorted(stage_positions),
        "Gate-retry history is not in schedule order",
    )
    return plan_ids


def _tail_rows(early_stop: Mapping[str, Any], patience: int) -> list[Mapping[str, Any]]:
    history = early_stop.get("history")
    _require(isinstance(history, list) and len(history) >= patience, "Short gate history")
    tail = history[-patience:]
    _require(
        all(isinstance(row, Mapping) for row in tail),
        "Gate tail contains a non-object row",
    )
    return tail


def validate_checkpoint_cadence(
    validation: Mapping[str, Any],
    *,
    expected_final_step: int,
    save_steps: int,
    expected_final_sha256: str | None = None,
    allow_single_checkpoint_change_proof: bool = False,
    expected_initial_sha256: str | None = None,
) -> dict[str, Any]:
    """Hard-check the real self-play validation shape (required is false)."""

    _require(expected_final_step > 0, "Checkpoint final step must be positive")
    _require(save_steps > 0, "Checkpoint save cadence must be positive")
    expected_steps = list(range(save_steps, expected_final_step + 1, save_steps))
    if not expected_steps or expected_steps[-1] != expected_final_step:
        expected_steps.append(expected_final_step)
    _require(
        int(validation.get("expected_step", -1)) == expected_final_step,
        "Checkpoint validation expected_step drifted",
    )
    _require(
        int(validation.get("final_step", -1)) == expected_final_step,
        "Checkpoint validation final_step drifted",
    )
    _require(
        validation.get("expected_checkpoint_steps") == expected_steps,
        "Checkpoint cadence steps drifted",
    )
    _require(
        int(validation.get("expected_checkpoint_count", -1))
        == len(expected_steps),
        "Checkpoint cadence count drifted",
    )
    observed = validation.get("observed_checkpoint_steps")
    _require(isinstance(observed, list), "Observed checkpoint cadence is missing")
    _require(
        observed == expected_steps,
        "Observed checkpoint cadence is not exact",
    )
    _require(
        int(validation.get("observed_expected_checkpoint_count", -1))
        == len(expected_steps),
        "Observed expected-checkpoint count drifted",
    )
    _require(
        validation.get("missing_checkpoint_steps") == [],
        "Checkpoint cadence has missing steps",
    )
    _require(
        validation.get("complete_cadence_required") is False,
        "Real self-play cadence-required flag drifted",
    )
    _require(
        validation.get("complete_cadence_verified") is True,
        "Checkpoint cadence was not verified",
    )
    expected_digests = validation.get("expected_checkpoint_sha256")
    all_digests = validation.get("checkpoint_sha256")
    expected_keys = [str(step) for step in expected_steps]
    _require(
        isinstance(expected_digests, Mapping)
        and list(expected_digests) == expected_keys,
        "Expected checkpoint digest cadence drifted",
    )
    _require(
        isinstance(all_digests, Mapping)
        and list(all_digests) == expected_keys,
        "Checkpoint digest map cadence drifted",
    )
    for key in expected_keys:
        digest = _require_sha(expected_digests.get(key), f"checkpoint step {key}")
        _require(all_digests.get(key) == digest, f"Checkpoint digest mismatch at {key}")
    if expected_final_sha256 is not None:
        final_sha = _require_sha(expected_final_sha256, "expected final checkpoint")
        _require(
            expected_digests[str(expected_final_step)] == final_sha,
            "Final checkpoint validation/population digest drifted",
        )
    changed_across_checkpoints = (
        validation.get("changed_across_checkpoints") is True
    )
    single_checkpoint_change_proof = False
    if not changed_across_checkpoints:
        _require(
            allow_single_checkpoint_change_proof
            and len(expected_steps) == 1
            and expected_final_sha256 is not None
            and expected_initial_sha256 is not None,
            "Checkpoint sequence did not prove changing LoRA weights",
        )
        initial_sha = _require_sha(
            expected_initial_sha256,
            "single-checkpoint initializer",
        )
        final_sha = _require_sha(
            expected_final_sha256,
            "single-checkpoint final checkpoint",
        )
        _require(
            final_sha != initial_sha,
            "Single-checkpoint retry did not change from its initializer",
        )
        single_checkpoint_change_proof = True
    return {
        "passed": True,
        "save_steps": save_steps,
        "expected_final_step": expected_final_step,
        "expected_checkpoint_steps": expected_steps,
        "complete_cadence_required": False,
        "complete_cadence_verified": True,
        "changed_across_checkpoints": changed_across_checkpoints,
        "single_checkpoint_change_proof": single_checkpoint_change_proof,
    }


def validate_successful_gate(
    validation: Mapping[str, Any],
    *,
    role: str,
    threshold: float = 0.95,
    patience: int = 5,
    attacker_min_steps: int = 30,
    defender_min_steps: int = 32,
    rollout_batch_size: int = 128,
    expected_budget: int | None = None,
    save_steps: int = 10,
    expected_final_sha256: str | None = None,
    allow_single_checkpoint_change_proof: bool = False,
    expected_initial_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute the exact frozen A or joint-H/B D early-stop proof."""

    _require(role in {"attacker", "defender"}, "Invalid gate role")
    _require(validation.get("stopped_early") is True, "Attempt did not stop early")
    early_stop = validation.get("early_stop")
    _require(isinstance(early_stop, Mapping), "Missing early-stop record")
    expected_metric = (
        "attacker/request_success_rate"
        if role == "attacker"
        else "defender/wildguard_actual_harmful_joint_success"
    )
    expected_min_steps = attacker_min_steps if role == "attacker" else defender_min_steps
    _require(early_stop.get("metric") == expected_metric, "Gate metric drifted")
    _require(float(early_stop.get("threshold", -1.0)) == threshold, "Gate threshold drifted")
    _require(int(early_stop.get("patience", -1)) == patience, "Gate patience drifted")
    _require(int(early_stop.get("min_steps", -1)) == expected_min_steps, "Gate minimum step drifted")
    _require(early_stop.get("triggered") is True, "Gate did not trigger")
    _require(int(early_stop.get("streak", -1)) >= patience, "Gate streak is too short")
    actual_final_step = int(validation.get("actual_final_step", 0))
    _require(actual_final_step > 0, "Gate final step is not positive")
    if expected_budget is not None:
        _require(expected_budget > 0, "Gate budget is not positive")
        _require(
            int(validation.get("requested_max_step", -1)) == expected_budget,
            "Gate requested budget drifted",
        )
        _require(
            actual_final_step <= expected_budget,
            "Gate final step exceeds its attempt budget",
        )
    _require(
        int(early_stop.get("actual_final_step", -1)) == actual_final_step,
        "Gate/final step mismatch",
    )
    _require(
        int(early_stop.get("last_step", actual_final_step)) == actual_final_step,
        "Gate last-step marker drifted",
    )
    _require(
        str(early_stop.get("checkpoint_tag") or f"global_step{actual_final_step}")
        == f"global_step{actual_final_step}",
        "Gate checkpoint tag drifted",
    )
    final_checkpoint = str(validation.get("final_checkpoint") or "")
    _require(
        final_checkpoint.endswith(f"/global_step{actual_final_step}_hf"),
        "Gate final checkpoint drifted",
    )
    cadence = validate_checkpoint_cadence(
        validation,
        expected_final_step=actual_final_step,
        save_steps=save_steps,
        expected_final_sha256=expected_final_sha256,
        allow_single_checkpoint_change_proof=(
            allow_single_checkpoint_change_proof
        ),
        expected_initial_sha256=expected_initial_sha256,
    )
    tail = _tail_rows(early_stop, patience)
    expected_steps = list(range(actual_final_step - patience + 1, actual_final_step + 1))
    _require(
        [int(row.get("step", -1)) for row in tail] == expected_steps,
        "Gate tail is not consecutive",
    )
    _require(
        all(step >= expected_min_steps for step in expected_steps),
        "Gate tail precedes the configured minimum step",
    )
    values = [float(row.get("value", float("nan"))) for row in tail]
    _require(
        all(math.isfinite(value) and value >= threshold for value in values),
        "Gate tail fell below threshold",
    )
    _require(all(row.get("qualified") is True for row in tail), "Unqualified gate row")

    expected_companions: dict[str, dict[str, float | str]]
    if role == "attacker":
        expected_companions = {
            "attacker/cot_format_violation": {
                "direction": "max",
                "bound": 0.05,
            }
        }
    else:
        minimum_count = float(math.ceil(0.95 * rollout_batch_size / 2))
        expected_companions = {
            "defender/wildguard_actual_harmful_count": {
                "direction": "min",
                "bound": minimum_count,
            },
            "defender/wildguard_actual_benign_joint_success": {
                "direction": "min",
                "bound": threshold,
            },
            "defender/wildguard_actual_benign_count": {
                "direction": "min",
                "bound": minimum_count,
            },
        }
    _require(
        early_stop.get("companion_bounds") == expected_companions,
        "Gate companion contract drifted",
    )
    for row in tail:
        metrics = row.get("metrics")
        _require(isinstance(metrics, Mapping), "Gate row has no companion metrics")
        for metric, requirement in expected_companions.items():
            value = float(metrics.get(metric, float("nan")))
            bound = float(requirement["bound"])
            direction = requirement["direction"]
            _require(math.isfinite(value), f"Non-finite gate companion: {metric}")
            _require(
                (direction == "min" and value >= bound)
                or (direction == "max" and value <= bound),
                f"Gate companion failed: {metric}",
            )
    return {
        "passed": True,
        "role": role,
        "metric": expected_metric,
        "threshold": threshold,
        "patience": patience,
        "tail_steps": expected_steps,
        "tail_values": values,
        "attacker_success_tail": (
            values if role == "attacker" else [1.0 - value for value in values]
        ),
        "companion_bounds": expected_companions,
        "checkpoint_cadence": cadence,
        "checkpoint_timing": (
            "rollout N evaluates W(N-1); the selected final is post-update WN"
        ),
    }


def validate_exhausted_attempt(
    validation: Mapping[str, Any],
    *,
    expected_budget: int,
    save_steps: int,
    expected_final_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove that a retry finished cleanly but did not meet the gate."""

    _require(validation.get("stopped_early") is False, "Attempt unexpectedly passed")
    _require(
        int(validation.get("requested_max_step", -1)) == expected_budget,
        "Attempt requested budget drifted",
    )
    _require(
        int(validation.get("actual_final_step", -1)) == expected_budget,
        "Attempt did not exhaust its budget",
    )
    final_checkpoint = str(validation.get("final_checkpoint") or "")
    _require(
        final_checkpoint.endswith(f"/global_step{expected_budget}_hf"),
        "Exhausted attempt final checkpoint drifted",
    )
    cadence = validate_checkpoint_cadence(
        validation,
        expected_final_step=expected_budget,
        save_steps=save_steps,
        expected_final_sha256=expected_final_sha256,
    )
    return {
        "passed": True,
        "classification": "gate_not_reached_after_complete_budget",
        "actual_final_step": expected_budget,
        "final_checkpoint": final_checkpoint,
        "checkpoint_cadence": cadence,
    }


def normalize_completed_retry_validation(
    validation: Mapping[str, Any],
    *,
    expected_budget: int,
    save_steps: int,
    early_stop_artifact_exists: bool,
) -> dict[str, Any]:
    """Normalize only the frozen trainer's exact completed-replay schema."""

    _require(expected_budget > 0, "Completed replay budget is not positive")
    _require(
        "actual_final_step" not in validation,
        "Full checkpoint validation does not need replay normalization",
    )
    forbidden_partial = {"requested_max_step", "stopped_early", "early_stop"}
    _require(
        not forbidden_partial.intersection(validation),
        "Partial retry checkpoint validation is ambiguous",
    )
    final_step = int(validation.get("final_step", -1))
    expected_step = int(validation.get("expected_step", -1))
    final_checkpoint = str(validation.get("final_checkpoint") or "")
    _require(
        final_step == expected_budget and expected_step == expected_budget,
        "Completed-run replay did not reach the exact attempt budget",
    )
    _require(
        final_checkpoint.endswith(f"/global_step{expected_budget}_hf"),
        "Completed-run replay final checkpoint drifted",
    )
    validate_checkpoint_cadence(
        validation,
        expected_final_step=expected_budget,
        save_steps=save_steps,
    )
    _require(
        not early_stop_artifact_exists,
        "Completed-run replay conflicts with an early-stop artifact",
    )
    return {
        **dict(validation),
        "requested_max_step": expected_budget,
        "actual_final_step": expected_budget,
        "stopped_early": False,
        "early_stop": None,
        "gate_retry_normalization": (
            "frozen completed-run replay schema normalized in memory"
        ),
    }


def _checkpoint_digest(
    path: Path,
    digest: Callable[[Path], str],
) -> str | None:
    if not path.exists():
        return None
    _require(path.is_dir(), f"Checkpoint exchange path is not a directory: {path}")
    return digest(path)


def exchange_directories_atomic(first: Path, second: Path) -> None:
    """Atomically exchange two existing directories with ``renameat2``."""

    _require(first.is_dir() and second.is_dir(), "Atomic exchange needs two directories")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc does not expose renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(first),
        _AT_FDCWD,
        os.fsencode(second),
        _RENAME_EXCHANGE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{first} <-> {second}")


def reconcile_atomic_population_swap(
    *,
    canonical: Path,
    staging: Path,
    archive: Path,
    old_sha256: str,
    new_sha256: str,
    checkpoint_digest: Callable[[Path], str],
    exchange: Callable[[Path, Path], None] = exchange_directories_atomic,
) -> dict[str, Any]:
    """Finish or replay an atomic canonical/staging exchange.

    Valid durable layouts are:

    * before exchange: canonical=old, staging=new, archive absent;
    * after exchange: canonical=new, staging=old, archive absent;
    * finished: canonical=new, staging absent, archive=old.

    Every other combination fails closed.  The caller commits the backing
    volume after each returned filesystem side effect.
    """

    old_sha = _require_sha(old_sha256, "old population")
    new_sha = _require_sha(new_sha256, "new population")
    _require(old_sha != new_sha, "Gate retry did not change the population weights")
    observed = {
        "canonical": _checkpoint_digest(canonical, checkpoint_digest),
        "staging": _checkpoint_digest(staging, checkpoint_digest),
        "archive": _checkpoint_digest(archive, checkpoint_digest),
    }
    if observed == {
        "canonical": old_sha,
        "staging": new_sha,
        "archive": None,
    }:
        exchange(canonical, staging)
        return {"action": "atomic_exchange", "complete": False}
    if observed == {
        "canonical": new_sha,
        "staging": old_sha,
        "archive": None,
    }:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise RuntimeError(f"Population archive collision: {archive}")
        staging.rename(archive)
        return {"action": "archive_displaced_original", "complete": True}
    if observed == {
        "canonical": new_sha,
        "staging": None,
        "archive": old_sha,
    }:
        return {"action": "already_complete", "complete": True}
    raise RuntimeError(f"Unrecognized population swap layout: {observed}")


def validate_final_population_state(
    state: Mapping[str, Any],
    *,
    expected_labels: Sequence[str],
    population_root: Path,
    checkpoint_digest: Callable[[Path], str],
    gate_proofs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate exact canonical membership and state-bound live digests."""

    expected = list(expected_labels)
    _require(len(expected) == 16 and len(set(expected)) == 16, "Expected 16 labels")
    observed_names = sorted(path.name for path in population_root.iterdir())
    _require(observed_names == sorted(expected), "Canonical population membership drifted")
    stages = state.get("stages")
    _require(isinstance(stages, Mapping), "State has no stages map")
    _require(set(stages) == set(expected), "Final state stage membership drifted")
    expected_gate_labels = [label for label in expected if label != "A1"]
    _require(
        set(gate_proofs) == set(expected_gate_labels),
        "Final gate-proof membership drifted",
    )
    rows: list[dict[str, Any]] = []
    for label in expected:
        stage = stages.get(label)
        _require(isinstance(stage, Mapping), f"Missing final stage {label}")
        _require(
            stage.get("status") == "retained"
            and stage.get("transition_state") == "retained",
            f"Final stage is not retained: {label}",
        )
        checkpoint = population_root / label
        _require(checkpoint.is_dir(), f"Missing canonical checkpoint: {label}")
        _require(
            str(stage.get("population_checkpoint") or "") == str(checkpoint),
            f"Canonical path drifted: {label}",
        )
        expected_sha = _require_sha(stage.get("sha256"), label)
        actual_sha = checkpoint_digest(checkpoint)
        _require(actual_sha == expected_sha, f"Canonical weight SHA drifted: {label}")
        if label != "A1":
            _require(stage.get("stopped_early") is True, f"Final gate not passed: {label}")
            proof = gate_proofs.get(label)
            _require(
                isinstance(proof, Mapping)
                and proof.get("passed") is True
                and proof.get("stage_label") == label,
                f"Final gate proof did not pass: {label}",
            )
            release = stage.get("successor_release")
            _require(
                isinstance(release, Mapping) and release.get("approved") is True,
                f"Final successor release is not approved: {label}",
            )
        rows.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "sha256": actual_sha,
            }
        )
    return rows
