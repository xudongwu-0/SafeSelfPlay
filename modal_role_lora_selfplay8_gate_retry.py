#!/usr/bin/env python3
"""Recover a budget-exhausted A2--D8 gate without changing frozen sources.

This entrypoint is deliberately additive.  It never edits the original stage
claim or the eight hash-bound training files.  Each retry starts from the most
recent immutable candidate, keeps the same frozen opponent, and uses a new
deterministic trainer suffix.  A write-ahead journal and Linux ``renameat2``
exchange make the successful adapter the canonical same-label population only
after its complete five-rollout gate has been independently recomputed.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import modal


if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

import modal_upstream_selfredteam_role_lora as frozen_role_lora  # noqa: E402
from modal_role_lora_selfplay8 import (  # noqa: E402
    SELFPLAY_ROOT,
    _assert_trainer_manifest_implementation,
    _assert_training_implementation_frozen,
    _dispatch_stage_claim,
    _read_json_object,
    _strict_audit,
    _write_json_atomic,
    app,
    output_vol,
)
from modal_upstream_selfredteam_role_lora import (  # noqa: E402
    DEFENDER_V2_WARMUP_OPTIMIZER_STEPS,
    LLAMA_ABLITERATED_MODEL,
    _stable_wildguard_rm_url,
    hf_cache,
)
from role_lora_selfplay8 import (  # noqa: E402
    atomic_copy_population_checkpoint,
    build_selfplay8_schedule,
    checkpoint_weight_digest,
    population_labels,
    prune_stage_hf_checkpoints,
)
from roll.utils import selfplay_gate_retry as gate_retry_contract  # noqa: E402
from roll.utils.selfplay_gate_retry import (  # noqa: E402
    RECOVERY_KEY,
    RECOVERY_HISTORY_KEY,
    PPO_ONLY_RECOVERY_TRAINER_POLICY,
    build_attempt_contract,
    build_ppo_only_recovery_trainer_source,
    build_recovery_history_entry,
    build_recovery_plan,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    normalize_completed_retry_validation,
    reconcile_atomic_population_swap,
    validate_exhausted_attempt,
    validate_final_population_state,
    validate_ppo_only_recovery_manifest,
    validate_recovery_eligibility,
    validate_successful_gate,
    verify_attempt_contract,
    verify_recovery_history,
    verify_recovery_plan,
)


_SAFE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RECOVERY_ROOT_NAME = "gate_retry_v1"
_POPULATION_ATTEMPTS_NAME = "population_attempts"


def _state_path(root: Path) -> Path:
    return root / "state.json"


def _load_state_snapshot(root: Path) -> tuple[dict[str, Any], str]:
    path = _state_path(root)
    try:
        raw = path.read_bytes()
        state = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid self-play state snapshot: {path}") from error
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported self-play state snapshot: {path}")
    return state, bytes_sha256(raw)


def _persist_state_cas(
    root: Path,
    state: dict[str, Any],
    *,
    expected_file_sha256: str,
) -> str:
    """Commit state only when the durable file is still the audited snapshot."""

    output_vol.reload()
    path = _state_path(root)
    try:
        current = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Cannot reload self-play state for CAS: {path}") from error
    observed = bytes_sha256(current)
    if observed != expected_file_sha256:
        raise RuntimeError(
            "Self-play state changed during gate-retry CAS: "
            f"expected={expected_file_sha256}, observed={observed}"
        )
    _write_json_atomic(path, state)
    output_vol.commit()
    return bytes_sha256(path.read_bytes())


def _write_exact_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json_object(path) != value:
            raise RuntimeError(f"Immutable gate-retry artifact drifted: {path}")
        return
    _write_json_atomic(path, value)


def _recovery_implementation_hashes() -> dict[str, str]:
    modal_path = Path(__file__).resolve()
    helper_path = Path(gate_retry_contract.__file__).resolve()
    hashes = {
        "modal_role_lora_selfplay8_gate_retry.py": file_sha256(modal_path),
        "roll/utils/selfplay_gate_retry.py": file_sha256(helper_path),
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()):
        raise RuntimeError("Gate-retry implementation hashing failed")
    return hashes


def _current_ppo_only_recovery_trainer_contract(
    frozen_training_sha256: dict[str, str],
) -> dict[str, Any]:
    core_path = Path(frozen_role_lora.__file__).resolve()
    source = core_path.read_text(encoding="utf-8")
    _effective_source, descriptor = build_ppo_only_recovery_trainer_source(source)
    expected_core = frozen_training_sha256.get(
        "modal_upstream_selfredteam_role_lora.py"
    )
    if descriptor["frozen_core_source_sha256"] != expected_core:
        raise RuntimeError("PPO-only recovery trainer uses a different frozen core")
    return descriptor


def _population_path(root: Path, label: str) -> Path:
    return root / "population" / label


def _recovery_root(root: Path, label: str) -> Path:
    return root / _RECOVERY_ROOT_NAME / label


def _strict_checkpoint(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise RuntimeError(f"Missing gate-retry checkpoint: {checkpoint}")
    audit = _strict_audit(checkpoint)
    contract = audit.get("llama_v2_contract")
    if (
        audit.get("weight_sha256") != expected_sha256
        or not isinstance(contract, dict)
        or contract.get("passed") is not True
        or int(audit.get("tensor_count", -1)) != 448
        or int(audit.get("rank", -1)) != 64
        or int(audit.get("alpha", -1)) != 64
    ):
        raise RuntimeError(f"Strict gate-retry audit failed: {checkpoint}")
    return audit


def _ensure_runtime_compatible_adapter_after_replay(
    *,
    source: Path,
    runtime: Path,
    destination_name: str,
    expected_sha256: str,
) -> str:
    """Rebuild a cold-container PEFT copy skipped by a trainer replay return."""

    if not frozen_role_lora._is_complete_hf_checkpoint(runtime):
        rebuilt = frozen_role_lora._prepare_peft_compatible_adapter(
            str(source),
            destination_name=destination_name,
        )
        if (
            Path(str(rebuilt)) != runtime
            or not frozen_role_lora._is_complete_hf_checkpoint(runtime)
        ):
            raise RuntimeError(
                "Frozen PEFT compatibility helper rebuilt an unexpected path"
            )
    runtime_sha256 = checkpoint_weight_digest(runtime)
    if runtime_sha256 != expected_sha256:
        raise RuntimeError(
            "PEFT runtime-compatible adapter weight digest drifted"
        )
    return runtime_sha256


@app.function(
    gpu=os.environ.get("UPSTREAM_ROLE_LORA_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=32768,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_role_lora_gate_retry_ppo_only(
    plan: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Run an additive, hash-bound clone with SFT disabled for same-stage retry."""

    plan_id = verify_recovery_plan(plan)
    attempt_id = verify_attempt_contract(contract, plan)
    if contract.get("plan_id") != plan_id:
        raise RuntimeError("PPO-only retry contract/plan identity drifted")
    frozen = dict(plan["frozen_training_implementation_sha256"])
    _assert_training_implementation_frozen(
        {"config": {"training_implementation_sha256": frozen}}
    )
    current_recovery = _recovery_implementation_hashes()
    if plan.get("recovery_implementation_sha256") != current_recovery:
        raise RuntimeError("PPO-only retry recovery implementation drifted")
    effective_source, effective_contract = build_ppo_only_recovery_trainer_source(
        Path(frozen_role_lora.__file__).read_text(encoding="utf-8")
    )
    if plan.get("recovery_trainer_contract") != effective_contract:
        raise RuntimeError("PPO-only effective trainer contract drifted")

    init_checkpoint = Path(str(contract["trainable_init_checkpoint"]))
    fixed_checkpoint = Path(str(contract["fixed_opponent"]["checkpoint"]))
    if (
        checkpoint_weight_digest(init_checkpoint)
        != contract["trainable_init_sha256"]
        or checkpoint_weight_digest(fixed_checkpoint)
        != contract["fixed_opponent"]["sha256"]
    ):
        raise RuntimeError("PPO-only retry input adapter digest drifted")

    namespace = dict(vars(frozen_role_lora))
    exec(
        compile(
            effective_source,
            str(Path(frozen_role_lora.__file__).resolve()),
            "exec",
        ),
        namespace,
    )
    effective_train = namespace.get("_effective_gate_retry_ppo_only_train")
    if not callable(effective_train):
        raise RuntimeError("PPO-only effective trainer was not constructed")

    config = dict(plan["frozen_selfplay_config"])
    is_attacker = contract["role"] == "attacker"
    run_dir_text = effective_train(
        remote_rm_url=_stable_wildguard_rm_url(),
        steps=int(contract["per_attempt_budget"]),
        normal_prompt_mix=True,
        normal_prompt_pool_size=0,
        normal_prompt_pool_profile="balanced",
        rollout_batch_size=128,
        micro_rollout_batch_size=8,
        micro_train_batch_size=8,
        train_batch_size=32,
        save_steps=int(config["save_steps"]),
        actor_learning_rate=float(
            config[
                "attacker_learning_rate"
                if is_attacker
                else "defender_learning_rate"
            ]
        ),
        init_kl_coef=0.0,
        actor_lr_scheduler="constant_with_warmup",
        lr_warmup_ratio=0.05,
        actor_lr_warmup_steps_override=(
            None if is_attacker else DEFENDER_V2_WARMUP_OPTIMIZER_STEPS
        ),
        enable_aux_sft=False,
        run_suffix=str(contract["trainer_run_suffix"]),
        train_role=str(contract["role"]),
        fixed_attacker_adapter=(str(fixed_checkpoint) if not is_attacker else ""),
        fixed_defender_adapter=(str(fixed_checkpoint) if is_attacker else ""),
        defender_prompt_profile="upstream",
        upstream_invalid_handling=True,
        base_model=LLAMA_ABLITERATED_MODEL,
        attacker_init_adapter=str(init_checkpoint),
        attacker_prompt_profile="optimized",
        strict_upstream_alignment=False,
        lora_rank=64,
        lora_alpha=64,
        monitor_reference_kl=False,
        postfill_cot_stop_after_step=0,
        role_specific_aux_sft=False,
        v2_runtime=True,
        v2_continuation_sft=False,
        defender_sft_optimizer_slots_per_rollout=0,
        defender_raw_reinforce_advantages=(not is_attacker),
        defender_reinforce_advantage_mode=(
            "raw_no_center" if is_attacker else "joint_signed"
        ),
        defender_reward_utility=(
            "upstream_additive" if is_attacker else "joint_signed"
        ),
        defender_prompt_pool_path=(
            ""
            if is_attacker
            else str(config["d1_data_contract"]["training_prompt_pool_path"])
        ),
        defender_prompt_pool_sha256=(
            ""
            if is_attacker
            else str(config["d1_data_contract"]["training_prompt_pool_sha256"])
        ),
        expected_implementation_sha256=frozen,
        early_stop_threshold=float(config["early_stop_threshold"]),
        early_stop_patience=int(config["early_stop_patience"]),
        early_stop_min_steps=int(
            config[
                "early_stop_min_steps"
                if is_attacker
                else "defender_early_stop_min_steps"
            ]
        ),
    )
    run_dir = Path(str(run_dir_text))
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json_object(manifest_path)
    trainable_runtime = Path(
        f"/tmp/{contract['role']}_lora_init_compatible"
    )
    fixed_runtime = Path("/tmp/fixed_opponent_lora_compatible")
    trainable_runtime_sha256 = _ensure_runtime_compatible_adapter_after_replay(
        source=init_checkpoint,
        runtime=trainable_runtime,
        destination_name=f"{contract['role']}_lora_init_compatible",
        expected_sha256=str(contract["trainable_init_sha256"]),
    )
    fixed_runtime_sha256 = _ensure_runtime_compatible_adapter_after_replay(
        source=fixed_checkpoint,
        runtime=fixed_runtime,
        destination_name="fixed_opponent_lora_compatible",
        expected_sha256=str(contract["fixed_opponent"]["sha256"]),
    )
    runtime_mapping = {
        "trainable": {
            "original_checkpoint": str(init_checkpoint),
            "original_sha256": contract["trainable_init_sha256"],
            "runtime_compatible_checkpoint": str(trainable_runtime),
            "runtime_weight_sha256": trainable_runtime_sha256,
        },
        "fixed_opponent": {
            "original_checkpoint": str(fixed_checkpoint),
            "original_sha256": contract["fixed_opponent"]["sha256"],
            "runtime_compatible_checkpoint": str(fixed_runtime),
            "runtime_weight_sha256": fixed_runtime_sha256,
        },
    }
    for mapping in runtime_mapping.values():
        if mapping["original_sha256"] != mapping["runtime_weight_sha256"]:
            raise RuntimeError("PEFT runtime-compatible adapter weight digest drifted")
    recovery_manifest: dict[str, Any] = {
        "schema_version": 1,
        "policy": PPO_ONLY_RECOVERY_TRAINER_POLICY,
        "plan_id": plan_id,
        "attempt_id": attempt_id,
        "stage_label": contract["stage_label"],
        "role": contract["role"],
        "trainer_run_suffix": contract["trainer_run_suffix"],
        "frozen_training_implementation_sha256": frozen,
        "recovery_implementation_sha256": current_recovery,
        "effective_trainer_contract": effective_contract,
        "implementation_identity": {
            "frozen_core": {
                "path": "modal_upstream_selfredteam_role_lora.py",
                "sha256": frozen[
                    "modal_upstream_selfredteam_role_lora.py"
                ],
            },
            "additive_recovery_sources": current_recovery,
            "effective_dynamic_function": {
                "policy": PPO_ONLY_RECOVERY_TRAINER_POLICY,
                "source_sha256": effective_contract[
                    "effective_function_source_sha256"
                ],
                "patch_descriptor_sha256": effective_contract[
                    "patch_descriptor_sha256"
                ],
            },
        },
        "runtime_adapter_mapping": runtime_mapping,
        "ppo_only_recipe": {
            **effective_contract["ppo_only_recipe"],
            "monitor_reference_kl": False,
            "optimizer_state": "cold_on_unique_attempt_suffix",
            "cold_container_replay_runtime_adapter_policy": (
                "rebuild_missing_or_incomplete_with_frozen_"
                "prepare_peft_compatible_adapter_before_digest"
            ),
            "fixed_opponent_unchanged": True,
            "reward_and_advantage_semantics_unchanged": True,
        },
    }
    recovery_manifest["recovery_manifest_sha256"] = canonical_json_sha256(
        recovery_manifest
    )
    recovery_manifest_path = run_dir / "gate_retry_recovery_manifest.json"
    _write_exact_json(recovery_manifest_path, recovery_manifest)
    manifest_binding = {
        "policy": PPO_ONLY_RECOVERY_TRAINER_POLICY,
        "path": str(recovery_manifest_path),
        "sha256": recovery_manifest["recovery_manifest_sha256"],
        "frozen_core_implementation_sha256": frozen[
            "modal_upstream_selfredteam_role_lora.py"
        ],
        "effective_function_source_sha256": effective_contract[
            "effective_function_source_sha256"
        ],
        "patch_descriptor_sha256": effective_contract[
            "patch_descriptor_sha256"
        ],
    }
    manifest["gate_retry_effective_implementation"] = manifest_binding
    _write_json_atomic(manifest_path, manifest)
    output_vol.commit()
    return {
        "run_dir": str(run_dir),
        "recovery_manifest_path": str(recovery_manifest_path),
        "recovery_manifest_sha256": recovery_manifest[
            "recovery_manifest_sha256"
        ],
        "effective_trainer_contract": effective_contract,
    }


def _verify_existing_recovery(
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    history = state.get(RECOVERY_HISTORY_KEY)
    verify_recovery_history(history)
    recovery = state.get(RECOVERY_KEY)
    if not isinstance(recovery, dict):
        raise RuntimeError("Gate-retry state is missing")
    if recovery.get("schema_version") != 1 or recovery.get("status") not in {
        "active",
        "swap_prepared",
        "promoted_pending_prune",
        "qualified_ready_to_release",
        "released",
        "completed",
    }:
        raise RuntimeError("Gate-retry durable status is invalid")
    plan = recovery.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("Gate-retry plan is missing from state")
    plan_id = verify_recovery_plan(plan)
    if recovery.get("plan_id") != plan_id:
        raise RuntimeError("Gate-retry state/plan identity drifted")
    plan_path = Path(str(plan.get("plan_path") or ""))
    if _read_json_object(plan_path) != plan:
        raise RuntimeError("Gate-retry plan artifact differs from state")
    current_recovery_hashes = _recovery_implementation_hashes()
    if plan.get("recovery_implementation_sha256") != current_recovery_hashes:
        raise RuntimeError("Gate-retry implementation changed during recovery")
    frozen = _assert_training_implementation_frozen(state)
    if plan.get("frozen_training_implementation_sha256") != frozen:
        raise RuntimeError("Frozen trainer identity changed during gate retry")
    if plan.get("frozen_selfplay_config") != state.get("config"):
        raise RuntimeError("Frozen self-play config changed during gate retry")
    if plan.get("recovery_trainer_contract") != (
        _current_ppo_only_recovery_trainer_contract(frozen)
    ):
        raise RuntimeError("Effective PPO-only recovery trainer changed")
    original_failure = plan.get("original_failure_evidence")
    if not isinstance(original_failure, dict):
        raise RuntimeError("Recovery plan has no original failure evidence")
    original_validation_path = Path(
        str(original_failure.get("checkpoint_validation_path") or "")
    )
    if file_sha256(original_validation_path) != original_failure.get(
        "checkpoint_validation_sha256"
    ):
        raise RuntimeError("Original failed checkpoint validation artifact drifted")
    rebuilt_failure = validate_exhausted_attempt(
        _read_json_object(original_validation_path),
        expected_budget=int(plan["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(
            plan["original_nonqualifying_population"]["sha256"]
        ),
    )
    rebuilt_failure.update(
        {
            "checkpoint_validation_path": str(original_validation_path),
            "checkpoint_validation_sha256": file_sha256(
                original_validation_path
            ),
        }
    )
    if rebuilt_failure != original_failure:
        raise RuntimeError("Original failed gate proof drifted")
    label = str(plan.get("stage_label") or "")
    schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
    if state.get("schedule") != [stage.to_dict() for stage in schedule]:
        raise RuntimeError("Durable schedule changed during gate retry")
    if history:
        previous_label = str(history[-1]["stage_label"])
        schedule_labels = [stage.label for stage in schedule]
        if schedule_labels.index(label) <= schedule_labels.index(previous_label):
            raise RuntimeError("Current recovery is not after its archived history")
    stage = (state.get("stages") or {}).get(label)
    if not isinstance(stage, dict):
        raise RuntimeError(f"Gate-retry stage disappeared: {label}")
    if any(
        stage.get(key) != value
        for key, value in dict(plan["stage_spec"]).items()
    ):
        raise RuntimeError("Gate-retry stage mapping drifted")
    if stage.get("spawn_claim_id") != plan.get("original_stage_spawn_claim_id"):
        raise RuntimeError("Original stage claim changed during gate retry")
    if (
        stage.get("status") != "retained"
        or stage.get("transition_state") != "retained"
    ):
        raise RuntimeError("Gate-retry stage is no longer retained")
    if recovery.get("status") in {"active", "swap_prepared"}:
        original = plan["original_nonqualifying_population"]
        if (
            stage.get("population_checkpoint") != original["checkpoint"]
            or stage.get("sha256") != original["sha256"]
        ):
            raise RuntimeError("Nonqualifying population changed before swap")
    else:
        journal = recovery.get("swap_journal")
        if (
            not isinstance(journal, dict)
            or journal.get("phase") != "complete"
            or stage.get("population_checkpoint") != journal.get("canonical")
            or stage.get("sha256") != journal.get("new_sha256")
        ):
            raise RuntimeError("Qualified population/journal provenance drifted")
    for provenance_key in ("original_trainable_parent", "fixed_opponent"):
        provenance = plan.get(provenance_key)
        dependency = (
            (state.get("stages") or {}).get(provenance.get("label"))
            if isinstance(provenance, dict)
            else None
        )
        if (
            not isinstance(dependency, dict)
            or dependency.get("status") != "retained"
            or dependency.get("transition_state") != "retained"
            or dependency.get("population_checkpoint")
            != provenance.get("checkpoint")
            or dependency.get("sha256") != provenance.get("sha256")
        ):
            raise RuntimeError(f"Recovery dependency drifted: {provenance_key}")
    released = recovery.get("status") in {"released", "completed"}
    if bool(recovery.get("official_population_released")) != released:
        raise RuntimeError("Gate-retry release flag/status drifted")
    return recovery


def _audit_released_recovery_artifacts(
    state: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """Live-audit the official adapter plus every displaced/failed candidate."""

    if (
        recovery.get("status") not in {"released", "completed"}
        or recovery.get("official_population_released") is not True
    ):
        raise RuntimeError("Recovery artifacts are not released for preservation audit")
    plan = recovery["plan"]
    label = str(plan["stage_label"])
    stage = (state.get("stages") or {}).get(label)
    attempts = recovery.get("attempts")
    journal = recovery.get("swap_journal")
    if (
        not isinstance(stage, dict)
        or not isinstance(attempts, list)
        or not attempts
        or not isinstance(journal, dict)
        or journal.get("phase") != "complete"
    ):
        raise RuntimeError("Released recovery preservation provenance is incomplete")
    if _read_json_object(Path(plan["plan_path"])) != plan:
        raise RuntimeError("Released recovery plan artifact drifted")
    official = _strict_checkpoint(
        Path(str(journal["canonical"])),
        expected_sha256=str(journal["new_sha256"]),
    )
    displaced = _strict_checkpoint(
        Path(str(journal["archive"])),
        expected_sha256=str(journal["old_sha256"]),
    )
    if (
        stage.get("population_checkpoint") != journal["canonical"]
        or stage.get("sha256") != journal["new_sha256"]
    ):
        raise RuntimeError("Released recovery official population drifted")
    failed_candidates = []
    for index, attempt in enumerate(attempts):
        contract = _verify_attempt(recovery, attempt)
        manifest_path = Path(str(attempt.get("manifest_path") or ""))
        validation_path = Path(
            str(attempt.get("checkpoint_validation_path") or "")
        )
        recovery_manifest_path = Path(
            str(attempt.get("recovery_manifest_path") or "")
        )
        expected_file_hashes = (
            (manifest_path, attempt.get("manifest_sha256"), "trainer manifest"),
            (
                validation_path,
                attempt.get("checkpoint_validation_sha256"),
                "checkpoint validation",
            ),
            (
                recovery_manifest_path,
                attempt.get("recovery_manifest_file_sha256"),
                "recovery manifest",
            ),
        )
        for path, expected_file_sha, artifact_name in expected_file_hashes:
            if file_sha256(path) != expected_file_sha:
                raise RuntimeError(
                    f"Released retry {artifact_name} artifact drifted"
                )
        trainer_manifest = _read_json_object(manifest_path)
        recovery_manifest = _read_json_object(recovery_manifest_path)
        receipt = validate_ppo_only_recovery_manifest(
            recovery_manifest,
            trainer_manifest,
            plan=plan,
            contract=contract,
            recovery_manifest_path=str(recovery_manifest_path),
        )
        if (
            receipt != attempt.get("ppo_only_recovery_receipt")
            or receipt["recovery_manifest_sha256"]
            != attempt.get("recovery_manifest_sha256")
        ):
            raise RuntimeError("Released retry effective-trainer receipt drifted")
        validation = _read_retry_checkpoint_validation(
            Path(str(attempt["run_dir"])),
            expected_budget=int(contract["per_attempt_budget"]),
            save_steps=int(state["config"]["save_steps"]),
        )
        if index == len(attempts) - 1:
            successful_gate = validate_successful_gate(
                validation,
                role=str(contract["role"]),
                threshold=float(state["config"]["early_stop_threshold"]),
                patience=int(state["config"]["early_stop_patience"]),
                attacker_min_steps=int(state["config"]["early_stop_min_steps"]),
                defender_min_steps=int(
                    state["config"]["defender_early_stop_min_steps"]
                ),
                rollout_batch_size=128,
                expected_budget=int(contract["per_attempt_budget"]),
                save_steps=int(state["config"]["save_steps"]),
                expected_final_sha256=str(journal["new_sha256"]),
            )
            if successful_gate != attempt.get("gate_result"):
                raise RuntimeError("Released retry successful gate proof drifted")
            continue
        checkpoint = Path(str(attempt.get("candidate_checkpoint") or ""))
        digest = str(attempt.get("candidate_sha256") or "")
        failed_gate = validate_exhausted_attempt(
            validation,
            expected_budget=int(contract["per_attempt_budget"]),
            save_steps=int(state["config"]["save_steps"]),
            expected_final_sha256=digest,
        )
        if failed_gate != attempt.get("gate_result"):
            raise RuntimeError("Released retry failed gate proof drifted")
        audit = _strict_checkpoint(checkpoint, expected_sha256=digest)
        failed_candidates.append(
            {
                "attempt_id": contract["attempt_id"],
                "checkpoint": str(checkpoint),
                "sha256": digest,
                "llama_v2_contract_passed": (
                    audit["llama_v2_contract"]["passed"] is True
                ),
            }
        )
    return {
        "passed": True,
        "stage_label": label,
        "plan_id": recovery["plan_id"],
        "official": {
            "checkpoint": journal["canonical"],
            "sha256": official["weight_sha256"],
        },
        "displaced_original_nonqualifying": {
            "checkpoint": journal["archive"],
            "sha256": displaced["weight_sha256"],
        },
        "failed_candidates": failed_candidates,
    }


def _rollover_released_recovery(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Archive an earlier released stage before recovering a later failure."""

    recovery = _verify_existing_recovery(root, state)
    old_label = str(recovery["plan"]["stage_label"])
    new_label = str(state.get("active_stage") or "")
    if state.get("status") != "stage_target_not_reached":
        return state, state_sha256
    if new_label == old_label:
        if recovery.get("status") == "released":
            raise RuntimeError("Released retry stage became blocked again")
        return state, state_sha256
    if recovery.get("status") != "released":
        raise RuntimeError(
            "A later stage failed before the prior gate retry was released"
        )
    history = list(state.get(RECOVERY_HISTORY_KEY) or [])
    verify_recovery_history(history)
    archived_recovery = copy.deepcopy(recovery)
    archived_recovery["release_preservation_audit"] = (
        _audit_released_recovery_artifacts(state, recovery)
    )
    entry = build_recovery_history_entry(
        archived_recovery,
        archived_state_file_sha256=state_sha256,
    )
    if any(row.get("plan_id") == entry["plan_id"] for row in history):
        raise RuntimeError("Released recovery is already present in history")
    updated = copy.deepcopy(state)
    updated[RECOVERY_HISTORY_KEY] = [*history, entry]
    del updated[RECOVERY_KEY]
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha


def _initialize_recovery(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
    eligibility = validate_recovery_eligibility(state, schedule)
    label = eligibility["label"]
    canonical = _population_path(root, label)
    if str(canonical) != eligibility["population_checkpoint"]:
        raise RuntimeError("Failed stage does not use its canonical population path")
    original_validation_path = Path(eligibility["stage"]["run_dir"]) / (
        "checkpoint_validation.json"
    )
    original_validation = _read_json_object(original_validation_path)
    original_failure = validate_exhausted_attempt(
        original_validation,
        expected_budget=int(eligibility["budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(eligibility["population_sha256"]),
    )
    original_failure.update(
        {
            "checkpoint_validation_path": str(original_validation_path),
            "checkpoint_validation_sha256": file_sha256(original_validation_path),
        }
    )
    frozen = _assert_training_implementation_frozen(state)
    recovery_hashes = _recovery_implementation_hashes()
    recovery_trainer_contract = _current_ppo_only_recovery_trainer_contract(
        frozen
    )
    live = {
        "original_nonqualifying_population": _strict_checkpoint(
            canonical,
            expected_sha256=eligibility["population_sha256"],
        ),
        "original_trainable_parent": _strict_checkpoint(
            Path(eligibility["original_parent"]["checkpoint"]),
            expected_sha256=eligibility["original_parent"]["sha256"],
        ),
        "fixed_opponent": _strict_checkpoint(
            Path(eligibility["fixed_opponent"]["checkpoint"]),
            expected_sha256=eligibility["fixed_opponent"]["sha256"],
        ),
    }
    output_vol.reload()
    current_state, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current_state != state:
        raise RuntimeError("Self-play state changed during recovery eligibility audits")
    plan_path = _recovery_root(root, label) / "plan.json"
    plan = build_recovery_plan(
        state=state,
        eligibility=eligibility,
        initial_state_file_sha256=state_sha256,
        frozen_training_sha256=frozen,
        recovery_implementation_sha256=recovery_hashes,
        recovery_trainer_contract=recovery_trainer_contract,
        plan_path=str(plan_path),
        original_failure_evidence=original_failure,
    )
    plan["initial_live_strict_audits"] = live
    plan_without_id = dict(plan)
    plan_without_id.pop("plan_id", None)
    plan["plan_id"] = canonical_json_sha256(plan_without_id)
    verify_recovery_plan(plan)
    _write_exact_json(plan_path, plan)
    output_vol.commit()
    updated = copy.deepcopy(state)
    updated[RECOVERY_KEY] = {
        "schema_version": 1,
        "status": "active",
        "plan_id": plan["plan_id"],
        "plan": plan,
        "attempts": [],
        "active_attempt_id": None,
        "swap_journal": None,
        "official_population_released": False,
    }
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha


def _attempt_seed(
    plan: dict[str, Any],
    *,
    attempt_number: int,
    init_sha256: str,
) -> str:
    return canonical_json_sha256(
        {
            "schema": "gate-retry-attempt-seed-v1",
            "plan_id": plan["plan_id"],
            "attempt_number": attempt_number,
            "trainable_init_sha256": init_sha256,
            "fixed_opponent_sha256": plan["fixed_opponent"]["sha256"],
        }
    )


def _create_attempt(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    recovery = _verify_existing_recovery(root, state)
    plan = recovery["plan"]
    attempts = recovery["attempts"]
    if attempts:
        last = attempts[-1]
        if last.get("status") != "gate_not_reached":
            raise RuntimeError("Cannot create a retry after a nonterminal attempt")
        init_checkpoint = str(last.get("candidate_checkpoint") or "")
        init_sha256 = str(last.get("candidate_sha256") or "")
    else:
        initial = plan["original_nonqualifying_population"]
        init_checkpoint = str(initial["checkpoint"])
        init_sha256 = str(initial["sha256"])
    _strict_checkpoint(Path(init_checkpoint), expected_sha256=init_sha256)
    fixed = plan["fixed_opponent"]
    _strict_checkpoint(
        Path(fixed["checkpoint"]),
        expected_sha256=str(fixed["sha256"]),
    )
    attempt_number = len(attempts) + 1
    seed = _attempt_seed(
        plan,
        attempt_number=attempt_number,
        init_sha256=init_sha256,
    )
    trainer_suffix = (
        f"{state['run_suffix']}_{plan['stage_label']}_gate_retry_"
        f"{attempt_number:03d}_{seed[:12]}"
    )
    attempt_root = _recovery_root(root, plan["stage_label"]) / "attempts" / (
        f"attempt_{attempt_number:03d}_{seed[:12]}"
    )
    contract_path = attempt_root / "contract.json"
    contract = build_attempt_contract(
        plan,
        attempt_number=attempt_number,
        trainable_init_checkpoint=init_checkpoint,
        trainable_init_sha256=init_sha256,
        trainer_run_suffix=trainer_suffix,
        contract_path=str(contract_path),
    )
    _write_exact_json(contract_path, contract)
    output_vol.commit()
    attempt = {
        "attempt_id": contract["attempt_id"],
        "attempt_number": attempt_number,
        "status": "training",
        "dispatch_state": "durably_claimed_before_trainer_call",
        "contract": contract,
        "contract_path": str(contract_path),
        "candidate_root": str(
            root
            / _POPULATION_ATTEMPTS_NAME
            / plan["stage_label"]
            / f"attempt_{attempt_number:03d}_{contract['attempt_id'][:12]}"
        ),
    }
    updated = copy.deepcopy(state)
    updated_recovery = updated[RECOVERY_KEY]
    updated_recovery["attempts"].append(attempt)
    updated_recovery["active_attempt_id"] = contract["attempt_id"]
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha, updated_recovery["attempts"][-1]


def _verify_attempt(
    recovery: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    contract = attempt.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Active attempt has no contract")
    attempt_id = verify_attempt_contract(contract, recovery["plan"])
    if attempt.get("attempt_id") != attempt_id:
        raise RuntimeError("Attempt state/contract identity drifted")
    matches = [
        index
        for index, row in enumerate(recovery.get("attempts") or [], start=1)
        if isinstance(row, dict) and row.get("attempt_id") == attempt_id
    ]
    if matches != [int(contract.get("attempt_number", -1))]:
        raise RuntimeError("Attempt number/list provenance drifted")
    contract_path = Path(str(attempt.get("contract_path") or ""))
    if _read_json_object(contract_path) != contract:
        raise RuntimeError("Attempt contract artifact differs from state")
    return contract


def _run_frozen_attempt(
    state: dict[str, Any],
    recovery: dict[str, Any],
    attempt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    plan = recovery["plan"]
    contract = _verify_attempt(recovery, attempt)
    init_checkpoint = str(contract["trainable_init_checkpoint"])
    fixed_checkpoint = str(contract["fixed_opponent"]["checkpoint"])
    _strict_checkpoint(
        Path(init_checkpoint),
        expected_sha256=str(contract["trainable_init_sha256"]),
    )
    _strict_checkpoint(
        Path(fixed_checkpoint),
        expected_sha256=str(contract["fixed_opponent"]["sha256"]),
    )
    result = train_role_lora_gate_retry_ppo_only.remote(
        plan=plan,
        contract=contract,
    )
    if not isinstance(result, dict):
        raise RuntimeError("PPO-only recovery trainer returned a non-object")
    return Path(str(result.get("run_dir") or "")), result


def _validate_attempt_output(
    state: dict[str, Any],
    recovery: dict[str, Any],
    attempt: dict[str, Any],
    run_dir: Path,
    trainer_result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    contract = _verify_attempt(recovery, attempt)
    frozen = dict(contract["frozen_training_implementation_sha256"])
    _assert_trainer_manifest_implementation(run_dir, frozen)
    manifest = _read_json_object(run_dir / "manifest.json")
    expected_early_stop = {
        "metric": (
            "attacker/request_success_rate"
            if contract["role"] == "attacker"
            else "defender/wildguard_actual_harmful_joint_success"
        ),
        "threshold": float(state["config"]["early_stop_threshold"]),
        "patience": int(state["config"]["early_stop_patience"]),
        "min_steps": int(
            state["config"][
                "early_stop_min_steps"
                if contract["role"] == "attacker"
                else "defender_early_stop_min_steps"
            ]
        ),
    }
    expected_companions = (
        {
            "attacker/cot_format_violation": {
                "direction": "max",
                "bound": 0.05,
            }
        }
        if contract["role"] == "attacker"
        else {
            "defender/wildguard_actual_harmful_count": {
                "direction": "min",
                "bound": 61.0,
            },
            "defender/wildguard_actual_benign_joint_success": {
                "direction": "min",
                "bound": float(state["config"]["early_stop_threshold"]),
            },
            "defender/wildguard_actual_benign_count": {
                "direction": "min",
                "bound": 61.0,
            },
        }
    )
    manifest_gate = manifest.get("early_stop")
    if (
        manifest.get("train_role") != contract["role"]
        or int(manifest.get("steps", -1)) != int(contract["per_attempt_budget"])
        or int(manifest.get("lora_rank", -1)) != 64
        or int(manifest.get("lora_alpha", -1)) != 64
        or not str(manifest.get("requested_run_name") or "").endswith(
            str(contract["trainer_run_suffix"])
        )
        or not isinstance(manifest_gate, dict)
        or any(manifest_gate.get(key) != value for key, value in expected_early_stop.items())
        or manifest_gate.get("companion_bounds") != expected_companions
    ):
        raise RuntimeError("Gate-retry trainer manifest contract drifted")
    is_attacker = contract["role"] == "attacker"
    fixed_registry = manifest.get("defender_actual_strata_training")
    fixed_pool = (
        fixed_registry.get("prompt_pool_exposure_registry")
        if isinstance(fixed_registry, dict)
        else None
    )
    expected_manifest_values = {
        "initial_base_model": LLAMA_ABLITERATED_MODEL,
        "initial_attacker_adapter": (
            contract["trainable_init_checkpoint"]
            if is_attacker
            else "/tmp/fixed_opponent_lora_compatible"
        ),
        "initial_defender_adapter": (
            "/tmp/fixed_opponent_lora_compatible"
            if is_attacker
            else contract["trainable_init_checkpoint"]
        ),
        "optimizer_train_role": contract["role"],
        "rollout_batch_size": 128,
        "micro_rollout_batch_size": 8,
        "micro_train_batch_size": 8,
        "train_batch_size": 32,
        "strict_upstream_alignment": False,
        "upstream_invalid_handling": True,
        "aux_sft_enabled": False,
        "online_sft_coef": 0.0,
        "postfill_cot_stop_after_step": 0,
        "role_specific_aux_sft": False,
        "v2_runtime": True,
        "v2_continuation_sft": False,
        "actor_learning_rate": float(
            state["config"][
                "attacker_learning_rate" if is_attacker else "defender_learning_rate"
            ]
        ),
        "actor_lr_scheduler": "constant_with_warmup",
        "actor_lr_warmup_steps_override": (
            None if is_attacker else DEFENDER_V2_WARMUP_OPTIMIZER_STEPS
        ),
        "init_kl_coef": 0.0,
        "reference_kl_monitoring": False,
        "defender_raw_reinforce_advantages": not is_attacker,
        "defender_reinforce_advantage_mode": (
            None if is_attacker else "joint_signed"
        ),
        "defender_reward_utility": (
            "upstream_additive" if is_attacker else "joint_signed"
        ),
    }
    drifted = {
        key: (manifest.get(key), expected)
        for key, expected in expected_manifest_values.items()
        if manifest.get(key) != expected
    }
    if drifted:
        raise RuntimeError(f"Gate-retry trainer recipe drifted: {drifted}")
    if (
        manifest.get("runtime_compatible_trainable_adapter")
        != f"/tmp/{contract['role']}_lora_init_compatible"
        or manifest.get("runtime_compatible_fixed_opponent_adapter")
        != "/tmp/fixed_opponent_lora_compatible"
    ):
        raise RuntimeError("Gate-retry PEFT initializer routing drifted")
    resume_step = int(manifest.get("lightweight_resume_step", -1))
    if not 0 <= resume_step <= int(contract["per_attempt_budget"]):
        raise RuntimeError("Gate-retry trainer resume step drifted")
    if is_attacker:
        if fixed_registry is not None or manifest.get("defender_sft_fixed_dose") is not None:
            raise RuntimeError("Attacker retry unexpectedly used defender data/SFT routing")
    else:
        expected_pool_sha = str(
            state["config"]["d1_data_contract"]["training_prompt_pool_sha256"]
        )
        if (
            not isinstance(fixed_pool, dict)
            or fixed_pool.get("path")
            != state["config"]["d1_data_contract"]["training_prompt_pool_path"]
            or fixed_pool.get("artifact_sha256") != expected_pool_sha
            or int(fixed_pool.get("rows", -1))
            != 128 * int(contract["per_attempt_budget"])
            or fixed_pool.get("interleave")
            != "four_rank_balanced_HHBBBBHH_cycle"
        ):
            raise RuntimeError("Defender retry prompt-pool routing drifted")
        if manifest.get("defender_sft_fixed_dose") is not None:
            raise RuntimeError("Defender retry unexpectedly retained an SFT dose")
    recovery_manifest_path = run_dir / "gate_retry_recovery_manifest.json"
    recovery_manifest = _read_json_object(recovery_manifest_path)
    expected_result = {
        "run_dir": str(run_dir),
        "recovery_manifest_path": str(recovery_manifest_path),
        "recovery_manifest_sha256": recovery_manifest.get(
            "recovery_manifest_sha256"
        ),
        "effective_trainer_contract": contract[
            "recovery_trainer_contract"
        ],
    }
    if trainer_result != expected_result:
        raise RuntimeError("PPO-only trainer return receipt drifted")
    receipt_proof = validate_ppo_only_recovery_manifest(
        recovery_manifest,
        manifest,
        plan=recovery["plan"],
        contract=contract,
        recovery_manifest_path=str(recovery_manifest_path),
    )
    validation = _read_retry_checkpoint_validation(
        run_dir,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
    )
    source_checkpoint = Path(str(validation["final_checkpoint"]))
    source_audit = _strict_audit(source_checkpoint)
    source_sha = str(source_audit.get("weight_sha256") or "")
    source_contract = source_audit.get("llama_v2_contract")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", source_sha)
        or not isinstance(source_contract, dict)
        or source_contract.get("passed") is not True
        or int(source_audit.get("tensor_count", -1)) != 448
        or int(source_audit.get("rank", -1)) != 64
        or int(source_audit.get("alpha", -1)) != 64
    ):
        raise RuntimeError("Retry source strict rank64/alpha64 audit failed")
    if source_sha == str(contract["trainable_init_sha256"]):
        raise RuntimeError("Gate-retry adapter did not change from its initializer")
    artifact = {
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / "manifest.json"),
        "manifest_sha256": file_sha256(run_dir / "manifest.json"),
        "checkpoint_validation_path": str(run_dir / "checkpoint_validation.json"),
        "checkpoint_validation_sha256": file_sha256(
            run_dir / "checkpoint_validation.json"
        ),
        "recovery_manifest_path": str(recovery_manifest_path),
        "recovery_manifest_file_sha256": file_sha256(recovery_manifest_path),
        "recovery_manifest_sha256": recovery_manifest[
            "recovery_manifest_sha256"
        ],
        "ppo_only_recovery_receipt": receipt_proof,
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": source_sha,
        "source_strict_audit": source_audit,
    }
    return validation, manifest, source_checkpoint, artifact


def _read_retry_checkpoint_validation(
    run_dir: Path,
    *,
    expected_budget: int,
    save_steps: int,
) -> dict[str, Any]:
    """Read a trainer result, including its frozen completed-run replay form.

    The frozen trainer's normal completion writes the full validation schema.
    If the exact deterministic suffix is entered again after already reaching
    its budget, its read-only completed-run branch rewrites that artifact with
    the lower-level checkpoint cadence schema.  Accept only that one exact,
    independently provable shape and normalize it in memory; never rewrite the
    trainer artifact or infer an early-stop success.
    """

    raw = _read_json_object(run_dir / "checkpoint_validation.json")
    if "actual_final_step" in raw:
        actual_step = int(raw.get("actual_final_step", 0))
        final_checkpoint = Path(str(raw.get("final_checkpoint") or ""))
        if (
            actual_step <= 0
            or final_checkpoint.name != f"global_step{actual_step}_hf"
        ):
            raise RuntimeError("Full retry checkpoint validation is malformed")
        return raw
    return normalize_completed_retry_validation(
        raw,
        expected_budget=expected_budget,
        save_steps=save_steps,
        early_stop_artifact_exists=(
            run_dir / "ckpt" / "early_stop.json"
        ).exists(),
    )


def _archive_failed_attempt(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
    attempt: dict[str, Any],
    *,
    validation: dict[str, Any],
    source_checkpoint: Path,
    artifact: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    recovery = _verify_existing_recovery(root, state)
    contract = _verify_attempt(recovery, attempt)
    exhausted = validate_exhausted_attempt(
        validation,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(artifact["source_sha256"]),
    )
    candidate_root = Path(str(attempt["candidate_root"]))
    promoted = atomic_copy_population_checkpoint(
        source_checkpoint,
        candidate_root,
        str(recovery["plan"]["stage_label"]),
    )
    output_vol.commit()
    candidate_audit = _strict_checkpoint(
        Path(promoted["path"]),
        expected_sha256=str(promoted["sha256"]),
    )
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while archiving a failed gate retry")
    updated = copy.deepcopy(state)
    current_attempt = updated[RECOVERY_KEY]["attempts"][-1]
    if current_attempt.get("attempt_id") != attempt.get("attempt_id"):
        raise RuntimeError("Active failed attempt changed")
    current_attempt.update(
        {
            "status": "gate_not_reached",
            "gate_result": exhausted,
            "candidate_checkpoint": promoted["path"],
            "candidate_sha256": promoted["sha256"],
            "candidate_strict_audit": candidate_audit,
            **artifact,
        }
    )
    updated[RECOVERY_KEY]["active_attempt_id"] = None
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    removed = prune_stage_hf_checkpoints(
        Path(artifact["run_dir"]) / "ckpt",
        audited_population_checkpoint=Path(promoted["path"]),
        audited_sha256=str(promoted["sha256"]),
    )
    output_vol.commit()
    output_vol.reload()
    latest, latest_sha = _load_state_snapshot(root)
    if latest_sha != new_sha:
        raise RuntimeError("State changed while pruning a failed gate retry")
    latest_attempt = latest[RECOVERY_KEY]["attempts"][-1]
    latest_attempt["pruned_source_hf_checkpoints"] = removed
    latest_attempt["pruning_complete"] = True
    final_sha = _persist_state_cas(
        root,
        latest,
        expected_file_sha256=latest_sha,
    )
    return latest, final_sha


def _finish_pending_failed_prune(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    recovery = _verify_existing_recovery(root, state)
    attempts = recovery["attempts"]
    if not attempts:
        return state, state_sha256
    last = attempts[-1]
    if last.get("status") != "gate_not_reached" or last.get("pruning_complete") is True:
        return state, state_sha256
    candidate = Path(str(last.get("candidate_checkpoint") or ""))
    digest = str(last.get("candidate_sha256") or "")
    _strict_checkpoint(candidate, expected_sha256=digest)
    removed = prune_stage_hf_checkpoints(
        Path(str(last["run_dir"])) / "ckpt",
        audited_population_checkpoint=candidate,
        audited_sha256=digest,
    )
    output_vol.commit()
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while reconciling failed retry pruning")
    updated = copy.deepcopy(state)
    updated_last = updated[RECOVERY_KEY]["attempts"][-1]
    updated_last["pruned_source_hf_checkpoints"] = removed
    updated_last["pruning_complete"] = True
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha


def _prepare_successful_swap(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
    attempt: dict[str, Any],
    *,
    validation: dict[str, Any],
    source_checkpoint: Path,
    artifact: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    recovery = _verify_existing_recovery(root, state)
    plan = recovery["plan"]
    contract = _verify_attempt(recovery, attempt)
    gate_proof = validate_successful_gate(
        validation,
        role=str(contract["role"]),
        threshold=float(state["config"]["early_stop_threshold"]),
        patience=int(state["config"]["early_stop_patience"]),
        attacker_min_steps=int(state["config"]["early_stop_min_steps"]),
        defender_min_steps=int(state["config"]["defender_early_stop_min_steps"]),
        rollout_batch_size=128,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(artifact["source_sha256"]),
    )
    label = str(plan["stage_label"])
    staging_root = _recovery_root(root, label) / "swap_staging" / str(
        attempt["attempt_id"]
    )
    promoted = atomic_copy_population_checkpoint(
        source_checkpoint,
        staging_root,
        label,
    )
    output_vol.commit()
    staging = Path(str(promoted["path"]))
    staging_audit = _strict_checkpoint(
        staging,
        expected_sha256=str(promoted["sha256"]),
    )
    original = plan["original_nonqualifying_population"]
    canonical = _population_path(root, label)
    archive = (
        root
        / _POPULATION_ATTEMPTS_NAME
        / label
        / f"original_nonqualifying_{str(original['sha256'])[:12]}"
        / label
    )
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while preparing the atomic population swap")
    updated = copy.deepcopy(state)
    current_attempt = updated[RECOVERY_KEY]["attempts"][-1]
    current_attempt.update(
        {
            "status": "qualified_staging",
            "gate_result": gate_proof,
            "staging_checkpoint": str(staging),
            "staging_sha256": promoted["sha256"],
            "staging_strict_audit": staging_audit,
            **artifact,
        }
    )
    journal = {
        "schema_version": 1,
        "attempt_id": attempt["attempt_id"],
        "phase": "prepared",
        "canonical": str(canonical),
        "staging": str(staging),
        "archive": str(archive),
        "old_sha256": original["sha256"],
        "new_sha256": promoted["sha256"],
        "atomic_primitive": "renameat2(RENAME_EXCHANGE)",
    }
    journal["journal_id"] = canonical_json_sha256(journal)
    updated[RECOVERY_KEY]["swap_journal"] = journal
    updated[RECOVERY_KEY]["status"] = "swap_prepared"
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha


def _reconcile_successful_swap(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    recovery = _verify_existing_recovery(root, state)
    journal = recovery.get("swap_journal")
    if not isinstance(journal, dict):
        return state, state_sha256
    payload = dict(journal)
    stored_journal_id = payload.pop("journal_id", None)
    if stored_journal_id != canonical_json_sha256(payload):
        raise RuntimeError("Population swap journal digest drifted")
    for _ in range(3):
        action = reconcile_atomic_population_swap(
            canonical=Path(journal["canonical"]),
            staging=Path(journal["staging"]),
            archive=Path(journal["archive"]),
            old_sha256=str(journal["old_sha256"]),
            new_sha256=str(journal["new_sha256"]),
            checkpoint_digest=checkpoint_weight_digest,
        )
        output_vol.commit()
        if action["complete"]:
            break
    else:
        raise RuntimeError("Population swap did not converge")
    canonical_audit = _strict_checkpoint(
        Path(journal["canonical"]),
        expected_sha256=str(journal["new_sha256"]),
    )
    archive_audit = _strict_checkpoint(
        Path(journal["archive"]),
        expected_sha256=str(journal["old_sha256"]),
    )
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while reconciling population swap")
    updated = copy.deepcopy(state)
    updated_recovery = updated[RECOVERY_KEY]
    attempt = updated_recovery["attempts"][-1]
    if attempt.get("attempt_id") != journal.get("attempt_id"):
        raise RuntimeError("Swap journal points at a different attempt")
    stage_label = str(updated_recovery["plan"]["stage_label"])
    stage = updated["stages"][stage_label]
    stage.update(
        {
            "work_status": "retained_after_gate_retry",
            "run_dir": attempt["run_dir"],
            "source_checkpoint": attempt["source_checkpoint"],
            "source_sha256": attempt["source_sha256"],
            "population_checkpoint": journal["canonical"],
            "actual_final_step": int(
                _read_json_object(Path(attempt["checkpoint_validation_path"]))[
                    "actual_final_step"
                ]
            ),
            "requested_max_step": int(
                updated_recovery["plan"]["per_attempt_budget"]
            ),
            "stopped_early": True,
            "sha256": journal["new_sha256"],
            "strict_audit": canonical_audit,
            "gate_retry_plan_id": updated_recovery["plan_id"],
            "gate_retry_attempt_id": attempt["attempt_id"],
            "displaced_nonqualifying_population": {
                "checkpoint": journal["archive"],
                "sha256": journal["old_sha256"],
                "strict_audit": archive_audit,
            },
        }
    )
    attempt["status"] = "promoted_pending_prune"
    attempt["official_population_checkpoint"] = journal["canonical"]
    attempt["official_population_sha256"] = journal["new_sha256"]
    attempt["official_population_strict_audit"] = canonical_audit
    updated_recovery["status"] = "promoted_pending_prune"
    updated_recovery["official_population_released"] = False
    updated_recovery["swap_journal"]["phase"] = "complete"
    journal_without_id = dict(updated_recovery["swap_journal"])
    journal_without_id.pop("journal_id", None)
    updated_recovery["swap_journal"]["journal_id"] = canonical_json_sha256(
        journal_without_id
    )
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha


def _prune_promoted_attempt(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    recovery = _verify_existing_recovery(root, state)
    if recovery.get("status") != "promoted_pending_prune":
        return state, state_sha256
    attempt = recovery["attempts"][-1]
    canonical = Path(str(attempt["official_population_checkpoint"]))
    digest = str(attempt["official_population_sha256"])
    _strict_checkpoint(canonical, expected_sha256=digest)
    removed = prune_stage_hf_checkpoints(
        Path(str(attempt["run_dir"])) / "ckpt",
        audited_population_checkpoint=canonical,
        audited_sha256=digest,
    )
    output_vol.commit()
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while pruning the promoted retry")
    updated = copy.deepcopy(state)
    updated_attempt = updated[RECOVERY_KEY]["attempts"][-1]
    updated_attempt["pruned_source_hf_checkpoints"] = removed
    updated_attempt["pruning_complete"] = True
    updated_attempt["status"] = "qualified_ready_to_release"
    updated[RECOVERY_KEY]["status"] = "qualified_ready_to_release"
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=state_sha256,
    )
    return updated, new_sha


def _build_final_population_audit(
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    labels = population_labels(int(state["config"]["rounds"]))
    gate_proofs: dict[str, dict[str, Any]] = {}
    for label in labels:
        if label == "A1":
            continue
        stage = (state.get("stages") or {}).get(label)
        if not isinstance(stage, dict):
            raise RuntimeError(f"Final gate proof has no stage: {label}")
        run_dir = Path(str(stage.get("run_dir") or ""))
        validation_path = run_dir / "checkpoint_validation.json"
        validation = _read_json_object(validation_path)
        role = "attacker" if label.startswith("A") else "defender"
        budget = int(
            state["config"][
                "attacker_max_steps"
                if role == "attacker"
                else "defender_max_steps"
            ]
        )
        if (
            validation.get("final_checkpoint") != stage.get("source_checkpoint")
            or int(validation.get("actual_final_step", -1))
            != int(stage.get("actual_final_step", -2))
            or int(validation.get("requested_max_step", -1))
            != int(stage.get("requested_max_step", -2))
            or int(stage.get("requested_max_step", -1)) != budget
            or validation.get("stopped_early") is not True
            or stage.get("stopped_early") is not True
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(stage.get("source_sha256") or ""),
            )
            or stage.get("source_sha256") != stage.get("sha256")
        ):
            raise RuntimeError(f"Final validation/state binding drifted: {label}")
        proof = validate_successful_gate(
            validation,
            role=role,
            threshold=float(state["config"]["early_stop_threshold"]),
            patience=int(state["config"]["early_stop_patience"]),
            attacker_min_steps=int(state["config"]["early_stop_min_steps"]),
            defender_min_steps=int(
                state["config"]["defender_early_stop_min_steps"]
            ),
            rollout_batch_size=128,
            expected_budget=budget,
            save_steps=int(state["config"]["save_steps"]),
            expected_final_sha256=str(stage["source_sha256"]),
        )
        gate_proofs[label] = {
            **proof,
            "stage_label": label,
            "run_dir": str(run_dir),
            "checkpoint_validation_path": str(validation_path),
            "checkpoint_validation_sha256": file_sha256(validation_path),
            "source_checkpoint": str(stage["source_checkpoint"]),
            "source_sha256": str(stage["source_sha256"]),
            "actual_final_step": int(stage["actual_final_step"]),
            "requested_max_step": budget,
        }
    history = state.get(RECOVERY_HISTORY_KEY)
    recovery_plan_ids = verify_recovery_history(history)
    recovery_archive_audits = [
        _audit_released_recovery_artifacts(state, row["recovery"])
        for row in (history or [])
    ]
    current_recovery = state.get(RECOVERY_KEY)
    if isinstance(current_recovery, dict):
        current_plan_id = str(current_recovery.get("plan_id") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", current_plan_id):
            raise RuntimeError("Current recovery has no valid plan identity")
        if current_plan_id in recovery_plan_ids:
            raise RuntimeError("Current recovery is duplicated in history")
        recovery_plan_ids.append(current_plan_id)
        recovery_archive_audits.append(
            _audit_released_recovery_artifacts(state, current_recovery)
        )
    digest_rows = validate_final_population_state(
        state,
        expected_labels=labels,
        population_root=root / "population",
        checkpoint_digest=checkpoint_weight_digest,
        gate_proofs=gate_proofs,
    )
    strict_rows = []
    for row in digest_rows:
        audit = _strict_checkpoint(
            Path(row["checkpoint"]),
            expected_sha256=row["sha256"],
        )
        strict_rows.append(
            {
                **row,
                "llama_v2_contract_passed": True,
                "tensor_count": audit["tensor_count"],
                "rank": audit["rank"],
                "alpha": audit["alpha"],
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "passed": True,
        "run_suffix": state["run_suffix"],
        "required_checkpoint_count": 16,
        "observed_checkpoint_count": len(strict_rows),
        "population_order": labels,
        "members": strict_rows,
        "gate_proofs": gate_proofs,
        "frozen_training_implementation_sha256": dict(
            state["config"]["training_implementation_sha256"]
        ),
        "recovery_implementation_sha256": _recovery_implementation_hashes(),
        "gate_retry_plan_ids": recovery_plan_ids,
        "gate_retry_archive_audits": recovery_archive_audits,
    }
    artifact["audit_sha256"] = canonical_json_sha256(artifact)
    return artifact


def _verify_final_population_audit_reference(
    state: dict[str, Any],
) -> dict[str, Any]:
    reference = state.get("final_population_audit")
    if not isinstance(reference, dict) or reference.get("passed") is not True:
        raise RuntimeError("Completed state has no passing final population audit")
    if int(reference.get("checkpoint_count", -1)) != 16:
        raise RuntimeError("Final population audit reference is not 16-member")
    artifact = _read_json_object(Path(str(reference.get("path") or "")))
    payload = dict(artifact)
    stored = payload.pop("audit_sha256", None)
    labels = population_labels(int(state["config"]["rounds"]))
    members = artifact.get("members")
    if (
        stored != reference.get("sha256")
        or canonical_json_sha256(payload) != stored
        or artifact.get("passed") is not True
        or artifact.get("run_suffix") != state.get("run_suffix")
        or int(artifact.get("required_checkpoint_count", -1)) != 16
        or int(artifact.get("observed_checkpoint_count", -1)) != 16
        or artifact.get("population_order") != labels
        or not isinstance(members, list)
        or not all(isinstance(row, dict) for row in members)
        or [row.get("label") for row in members] != labels
        or artifact.get("frozen_training_implementation_sha256")
        != state["config"]["training_implementation_sha256"]
    ):
        raise RuntimeError("Final population audit artifact drifted")
    return artifact


def _release_or_complete(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> dict[str, Any]:
    recovery = _verify_existing_recovery(root, state)
    if recovery.get("status") not in {
        "qualified_ready_to_release",
        "released",
    }:
        raise RuntimeError("Gate retry is not ready to release")
    plan = recovery["plan"]
    schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
    stage_position = next(
        index
        for index, stage in enumerate(schedule)
        if stage.label == plan["stage_label"]
    )
    attempt = recovery["attempts"][-1]
    if recovery.get("status") == "qualified_ready_to_release":
        updated = copy.deepcopy(state)
        updated_stage = updated["stages"][plan["stage_label"]]
        updated_stage["successor_release"] = {
            "approved": True,
            "basis": "same-label gate-retry passed the frozen five-step gate",
            "plan_id": recovery["plan_id"],
            "attempt_id": attempt["attempt_id"],
            "gate_result": attempt["gate_result"],
            "displaced_nonqualifying_population_preserved": True,
        }
        updated[RECOVERY_KEY]["status"] = "released"
        updated[RECOVERY_KEY]["official_population_released"] = True
        updated["status"] = "running"
        updated["active_stage"] = plan["stage_label"]
        state_sha256 = _persist_state_cas(
            root,
            updated,
            expected_file_sha256=state_sha256,
        )
        state = updated

    if stage_position + 1 < len(schedule):
        dispatch = _dispatch_stage_claim(
            root,
            state,
            run_suffix=str(state["run_suffix"]),
            stage=schedule[stage_position + 1],
        )
        return {
            "root": str(root),
            "state": dispatch["state"],
            "recovered_stage": plan["stage_label"],
            "attempt_id": attempt["attempt_id"],
            "spawned": dispatch["spawned"],
            "call_id": dispatch["call_id"],
            "spawn_claim_id": dispatch["spawn_claim_id"],
        }

    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed before final population audit")
    audit = _build_final_population_audit(root, state)
    output_vol.reload()
    latest, latest_sha = _load_state_snapshot(root)
    if latest_sha != state_sha256 or latest != state:
        raise RuntimeError("State changed during final population audits")
    audit_path = root / _RECOVERY_ROOT_NAME / "final_population_audit_v1.json"
    _write_exact_json(audit_path, audit)
    output_vol.commit()
    completed = copy.deepcopy(state)
    completed["status"] = "completed"
    completed["active_stage"] = None
    completed["completed_population"] = population_labels(
        int(completed["config"]["rounds"])
    )
    completed["final_population_audit"] = {
        "passed": True,
        "path": str(audit_path),
        "sha256": audit["audit_sha256"],
        "checkpoint_count": 16,
    }
    completed[RECOVERY_KEY]["status"] = "completed"
    final_sha = _persist_state_cas(
        root,
        completed,
        expected_file_sha256=latest_sha,
    )
    return {
        "root": str(root),
        "state": completed,
        "recovered_stage": plan["stage_label"],
        "attempt_id": attempt["attempt_id"],
        "spawned": False,
        "call_id": None,
        "final_population_audit": audit,
        "final_state_file_sha256": final_sha,
    }


def _continue_existing_phase(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str] | dict[str, Any]:
    recovery = _verify_existing_recovery(root, state)
    status = recovery.get("status")
    if recovery.get("swap_journal") is not None and status == "swap_prepared":
        return _reconcile_successful_swap(root, state, state_sha256)
    if status == "promoted_pending_prune":
        return _prune_promoted_attempt(root, state, state_sha256)
    if status in {"qualified_ready_to_release", "released"}:
        return _release_or_complete(root, state, state_sha256)
    return _finish_pending_failed_prune(root, state, state_sha256)


def _drain_recovery_phase(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str] | dict[str, Any]:
    """Drive journal phases locally without recursively entering Modal."""

    for _ in range(4):
        before_state = state
        before_sha = state_sha256
        result = _continue_existing_phase(root, state, state_sha256)
        if isinstance(result, dict):
            return result
        state, state_sha256 = result
        if state_sha256 == before_sha and state == before_state:
            return state, state_sha256
        status = _verify_existing_recovery(root, state).get("status")
        if status not in {
            "swap_prepared",
            "promoted_pending_prune",
            "qualified_ready_to_release",
            "released",
        }:
            return state, state_sha256
    raise RuntimeError("Gate-retry journal phase dispatcher did not converge")


def _audit_completed_population(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> dict[str, Any]:
    """Idempotently attach the live audit after a normally completed D8."""

    if state.get("status") != "completed" or state.get("active_stage") is not None:
        raise RuntimeError("Final population audit requires a completed chain")
    _assert_training_implementation_frozen(state)
    existing = state.get("final_population_audit")
    if isinstance(existing, dict) and existing.get("passed") is True:
        recorded = _verify_final_population_audit_reference(state)
        live = _build_final_population_audit(root, state)
        output_vol.reload()
        current, current_sha = _load_state_snapshot(root)
        if current_sha != state_sha256 or current != state:
            raise RuntimeError("State changed during repeated final population audit")
        reread = _verify_final_population_audit_reference(state)
        if reread != recorded or live != recorded:
            raise RuntimeError(
                "Repeated live 16-member audit differs from immutable artifact"
            )
        return {
            "root": str(root),
            "state": state,
            "already_audited": True,
            "spawned": False,
            "audit": live,
            "state_file_sha256": state_sha256,
        }
    audit = _build_final_population_audit(root, state)
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed during final population audit")
    audit_path = root / _RECOVERY_ROOT_NAME / "final_population_audit_v1.json"
    _write_exact_json(audit_path, audit)
    output_vol.commit()
    updated = copy.deepcopy(state)
    updated["final_population_audit"] = {
        "passed": True,
        "path": str(audit_path),
        "sha256": audit["audit_sha256"],
        "checkpoint_count": 16,
    }
    final_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return {
        "root": str(root),
        "state": updated,
        "already_audited": False,
        "spawned": False,
        "audit": audit,
        "final_state_file_sha256": final_sha,
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def resume_role_lora_selfplay8_gate_retry(
    run_suffix: str,
    attempt_limit: int = 0,
) -> dict[str, Any]:
    """Retry one blocked stage and recursively continue until its gate passes."""

    if not _SAFE_SUFFIX_RE.fullmatch(run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    if attempt_limit < 0:
        raise ValueError("attempt_limit must be non-negative; zero means unlimited")
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state, state_sha = _load_state_snapshot(root)
    if state.get("run_suffix") != run_suffix:
        raise RuntimeError("Gate-retry run suffix differs from durable state")
    if state.get("status") == "completed":
        if RECOVERY_KEY in state:
            recovery = _verify_existing_recovery(root, state)
            if recovery.get("status") not in {"released", "completed"}:
                raise RuntimeError("Completed chain has an unfinished gate retry")
        return _audit_completed_population(root, state, state_sha)
    if RECOVERY_KEY not in state:
        state, state_sha = _initialize_recovery(root, state, state_sha)
    else:
        _verify_existing_recovery(root, state)
        state, state_sha = _rollover_released_recovery(
            root,
            state,
            state_sha,
        )
        if RECOVERY_KEY not in state:
            state, state_sha = _initialize_recovery(root, state, state_sha)

    phase_result = _drain_recovery_phase(root, state, state_sha)
    if isinstance(phase_result, dict):
        return phase_result
    state, state_sha = phase_result
    recovery = _verify_existing_recovery(root, state)

    attempts = recovery["attempts"]
    if attempts and attempts[-1].get("status") == "training":
        attempt = attempts[-1]
    else:
        if attempt_limit and len(attempts) >= attempt_limit:
            return {
                "root": str(root),
                "state": state,
                "spawned": False,
                "reason": "attempt_limit_reached",
            }
        state, state_sha, attempt = _create_attempt(root, state, state_sha)
        recovery = _verify_existing_recovery(root, state)

    run_dir, trainer_result = _run_frozen_attempt(state, recovery, attempt)
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha or current != state:
        raise RuntimeError("State changed while the frozen retry trainer ran")
    current_recovery = _verify_existing_recovery(root, current)
    current_attempt = current_recovery["attempts"][-1]
    validation, _manifest, source_checkpoint, artifact = _validate_attempt_output(
        current,
        current_recovery,
        current_attempt,
        run_dir,
        trainer_result,
    )
    if validation.get("stopped_early") is True:
        state, state_sha = _prepare_successful_swap(
            root,
            current,
            current_sha,
            current_attempt,
            validation=validation,
            source_checkpoint=source_checkpoint,
            artifact=artifact,
        )
        state, state_sha = _reconcile_successful_swap(root, state, state_sha)
        state, state_sha = _prune_promoted_attempt(root, state, state_sha)
        return _release_or_complete(root, state, state_sha)

    state, state_sha = _archive_failed_attempt(
        root,
        current,
        current_sha,
        current_attempt,
        validation=validation,
        source_checkpoint=source_checkpoint,
        artifact=artifact,
    )
    completed_attempts = len(state[RECOVERY_KEY]["attempts"])
    if attempt_limit and completed_attempts >= attempt_limit:
        return {
            "root": str(root),
            "state": state,
            "spawned": False,
            "reason": "attempt_limit_reached_after_gate_failure",
        }
    state, state_sha, next_attempt = _create_attempt(root, state, state_sha)
    try:
        call = resume_role_lora_selfplay8_gate_retry.spawn(
            run_suffix=run_suffix,
            attempt_limit=attempt_limit,
        )
        call_id = call.object_id
    except BaseException as error:
        return {
            "root": str(root),
            "state": state,
            "spawned": False,
            "reason": "next_attempt_dispatch_unknown",
            "next_attempt_id": next_attempt["attempt_id"],
            "dispatch_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    return {
        "root": str(root),
        "state": state,
        "spawned": True,
        "call_id": call_id,
        "next_attempt_id": next_attempt["attempt_id"],
        "next_attempt_number": completed_attempts + 1,
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def audit_and_finalize_role_lora_selfplay8_population(
    run_suffix: str,
) -> dict[str, Any]:
    """Add the missing live 16-member audit to a normally completed chain."""

    if not _SAFE_SUFFIX_RE.fullmatch(run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state, state_sha = _load_state_snapshot(root)
    return _audit_completed_population(root, state, state_sha)


@app.local_entrypoint(name="resume_role_lora_selfplay8_gate_retry")
def resume_role_lora_selfplay8_gate_retry_local(
    run_suffix: str,
    attempt_limit: int = 0,
    wait_for_completion: bool = False,
) -> None:
    invoke = (
        resume_role_lora_selfplay8_gate_retry.remote
        if wait_for_completion
        else resume_role_lora_selfplay8_gate_retry.spawn
    )
    result = invoke(run_suffix=run_suffix, attempt_limit=attempt_limit)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"GATE_RETRY_CALL_ID={result.object_id}", flush=True)


@app.local_entrypoint(name="audit_and_finalize_role_lora_selfplay8_population")
def audit_and_finalize_role_lora_selfplay8_population_local(
    run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    invoke = (
        audit_and_finalize_role_lora_selfplay8_population.remote
        if wait_for_completion
        else audit_and_finalize_role_lora_selfplay8_population.spawn
    )
    result = invoke(run_suffix=run_suffix)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"FINAL_POPULATION_AUDIT_CALL_ID={result.object_id}", flush=True)
