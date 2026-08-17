#!/usr/bin/env python3
"""One-shot, additive attacker objective migration for an exhausted gate.

The entrypoint accepts only ``stage_objective_migration_required`` produced by
the hash-bound gate-retry protocol.  It warm-starts the last failed candidate,
keeps the frozen defender unchanged, disables SFT, and runs at most one bounded
attempt using an explicitly versioned binary goal+CoT surrogate.  The frozen
upstream additive utility remains the sole official payoff-matrix utility.

No frozen eight-round source is edited.  Successful promotion uses the same
CAS, write-ahead journal, ``renameat2(RENAME_EXCHANGE)``, preservation archive,
and strict LoRA audit primitives as gate retry.
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

import modal_role_lora_selfplay8_gate_retry as raw_retry  # noqa: E402
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
    LLAMA_ABLITERATED_MODEL,
    _stable_wildguard_rm_url,
    hf_cache,
)
from role_lora_selfplay8 import (  # noqa: E402
    atomic_copy_population_checkpoint,
    build_selfplay8_schedule,
    checkpoint_weight_digest,
    prune_stage_hf_checkpoints,
)
from roll.utils import selfplay_attacker_objective_migration as migration_contract  # noqa: E402
from roll.utils.selfplay_attacker_objective_migration import (  # noqa: E402
    MIGRATION_EXHAUSTED_STATE_STATUS,
    MIGRATION_FINAL_AUDIT_POLICY,
    MIGRATION_HISTORY_KEY,
    MIGRATION_KEY,
    MIGRATION_MANIFEST_POLICY,
    MIGRATION_MAX_ATTEMPTS_PER_STAGE,
    MIGRATION_POLICY,
    MIGRATION_REQUIRED_METRIC,
    MIGRATION_ROOT_NAME,
    MIGRATION_TRAINER_POLICY,
    binary_joint_objective_contract,
    build_binary_joint_migration_trainer_source,
    build_migration_attempt_contract,
    build_migration_aware_final_audit_source,
    build_migration_aware_gate_retry_finalize_sources,
    build_migration_final_audit_contract,
    build_migration_history_entry,
    build_migration_plan,
    canonical_json_sha256,
    file_sha256,
    index_migration_history_for_final_audit,
    patch_binary_joint_upstream_tree,
    validate_binary_joint_successful_gate,
    validate_migration_eligibility,
    validate_migration_manifest,
    validate_migrated_stage_final_gate,
    verify_binary_joint_migration_trainer_contract,
    verify_migration_attempt_contract,
    verify_migration_final_audit_contract,
    verify_migration_history,
    verify_migration_plan,
    verify_objective_migration_requirement,
    verify_recovery_plan,
)
from roll.utils.selfplay_gate_retry import (  # noqa: E402
    RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
    RECOVERY_KEY,
    bytes_sha256,
    reconcile_atomic_population_swap,
    validate_checkpoint_cadence,
)


_SAFE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_POPULATION_ATTEMPTS_NAME = "population_attempts"


def _state_path(root: Path) -> Path:
    return root / "state.json"


def _load_state_snapshot(root: Path) -> tuple[dict[str, Any], str]:
    return raw_retry._load_state_snapshot(root)


def _persist_state_cas(
    root: Path,
    state: dict[str, Any],
    *,
    expected_file_sha256: str,
) -> str:
    return raw_retry._persist_state_cas(
        root,
        state,
        expected_file_sha256=expected_file_sha256,
    )


def _write_exact_json(path: Path, value: dict[str, Any]) -> None:
    raw_retry._write_exact_json(path, value)


def _strict_checkpoint(checkpoint: Path, *, expected_sha256: str) -> dict[str, Any]:
    return raw_retry._strict_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
    )


def _ensure_runtime_compatible_adapter_after_replay(
    *,
    source: Path,
    runtime: Path,
    destination_name: str,
    expected_sha256: str,
) -> str:
    return raw_retry._ensure_runtime_compatible_adapter_after_replay(
        source=source,
        runtime=runtime,
        destination_name=destination_name,
        expected_sha256=expected_sha256,
    )


def _migration_root(root: Path, label: str) -> Path:
    return root / MIGRATION_ROOT_NAME / label


def _population_path(root: Path, label: str) -> Path:
    return root / "population" / label


def _migration_implementation_hashes() -> dict[str, str]:
    modal_path = Path(__file__).resolve()
    helper_path = Path(migration_contract.__file__).resolve()
    hashes = {
        "modal_role_lora_selfplay8_attacker_objective_migration.py": (
            file_sha256(modal_path)
        ),
        "roll/utils/selfplay_attacker_objective_migration.py": (
            file_sha256(helper_path)
        ),
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()):
        raise RuntimeError("Objective migration implementation hashing failed")
    return hashes


def _current_migration_trainer_contract(
    frozen_training_sha256: dict[str, str],
) -> dict[str, Any]:
    source = Path(frozen_role_lora.__file__).read_text(encoding="utf-8")
    _effective, descriptor = build_binary_joint_migration_trainer_source(source)
    expected_core = frozen_training_sha256.get(
        "modal_upstream_selfredteam_role_lora.py"
    )
    verify_binary_joint_migration_trainer_contract(
        descriptor,
        expected_frozen_core_sha256=expected_core,
    )
    return descriptor


def _current_migration_final_audit_contract() -> dict[str, Any]:
    source_path = Path(raw_retry.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    expected_source_sha = file_sha256(source_path)
    contract = build_migration_final_audit_contract(source)
    verify_migration_final_audit_contract(
        contract,
        expected_gate_retry_source_sha256=expected_source_sha,
    )
    return contract


def _verify_requirement_artifact(recovery: dict[str, Any]) -> dict[str, Any]:
    plan = recovery.get("plan")
    attempts = recovery.get("attempts")
    requirement = recovery.get("objective_migration_requirement")
    if not isinstance(plan, dict) or not isinstance(attempts, list) or not isinstance(
        requirement, dict
    ):
        raise RuntimeError("Objective migration hand-off is incomplete")
    requirement_id = verify_objective_migration_requirement(
        requirement, plan, attempts
    )
    artifact = recovery.get("objective_migration_requirement_artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("Objective migration hand-off artifact is missing")
    path = Path(str(artifact.get("path") or ""))
    if (
        _read_json_object(path) != requirement
        or file_sha256(path) != artifact.get("file_sha256")
        or artifact.get("requirement_id") != requirement_id
    ):
        raise RuntimeError("Objective migration hand-off artifact drifted")
    return requirement


def _verify_existing_migration(
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    verify_migration_history(state.get(MIGRATION_HISTORY_KEY))
    migration = state.get(MIGRATION_KEY)
    if not isinstance(migration, dict):
        raise RuntimeError("Attacker objective migration state is missing")
    allowed = {
        "active",
        "swap_prepared",
        "promoted_pending_prune",
        "qualified_ready_to_release",
        "released",
        "exhausted_pending_prune",
        "exhausted",
    }
    if migration.get("schema_version") != 1 or migration.get("status") not in allowed:
        raise RuntimeError("Attacker objective migration status is invalid")
    plan = migration.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("Attacker objective migration plan is missing")
    plan_id = verify_migration_plan(plan)
    if migration.get("plan_id") != plan_id:
        raise RuntimeError("Migration state/plan identity drifted")
    plan_path = Path(str(plan.get("plan_path") or ""))
    if _read_json_object(plan_path) != plan:
        raise RuntimeError("Migration plan artifact differs from state")
    frozen = _assert_training_implementation_frozen(state)
    if (
        plan.get("frozen_training_implementation_sha256") != frozen
        or plan.get("frozen_selfplay_config") != state.get("config")
        or plan.get("migration_implementation_sha256")
        != _migration_implementation_hashes()
        or plan.get("migration_trainer_contract")
        != _current_migration_trainer_contract(frozen)
        or plan.get("migration_final_audit_contract")
        != _current_migration_final_audit_contract()
    ):
        raise RuntimeError("Migration implementation or frozen config drifted")

    raw_recovery = state.get(RECOVERY_KEY)
    if not isinstance(raw_recovery, dict) or raw_recovery.get("status") != (
        RAW_PPO_EXHAUSTED_RECOVERY_STATUS
    ):
        raise RuntimeError("Migration lost its exhausted raw-PPO recovery")
    raw_plan = raw_recovery.get("plan")
    raw_attempts = raw_recovery.get("attempts")
    if not isinstance(raw_plan, dict) or not isinstance(raw_attempts, list):
        raise RuntimeError("Migration raw-PPO lineage is incomplete")
    if (
        verify_recovery_plan(raw_plan) != plan["source_gate_retry_plan_id"]
        or _read_json_object(Path(str(raw_plan.get("plan_path") or "")))
        != raw_plan
        or raw_plan.get("recovery_implementation_sha256")
        != raw_retry._recovery_implementation_hashes()
        or _verify_requirement_artifact(raw_recovery)
        != plan["objective_migration_requirement"]
    ):
        raise RuntimeError("Migration raw-PPO hand-off drifted")

    label = str(plan["stage_label"])
    if migration.get("status") == "released":
        schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
        position = next(
            index for index, row in enumerate(schedule) if row.label == label
        )
        allowed_active = {
            row.label for row in schedule[position:]
        } | {None}
    else:
        allowed_active = {label}
    if state.get("active_stage") not in allowed_active:
        raise RuntimeError("Migration active stage drifted")
    stage = (state.get("stages") or {}).get(label)
    if not isinstance(stage, dict):
        raise RuntimeError("Migration stage disappeared")
    if (
        stage.get("spawn_claim_id") != plan["original_stage_spawn_claim_id"]
        or stage.get("status") != "retained"
        or stage.get("transition_state") != "retained"
    ):
        raise RuntimeError("Migration stage lineage drifted")

    journal = migration.get("swap_journal")
    promoted = migration.get("status") in {
        "promoted_pending_prune",
        "qualified_ready_to_release",
        "released",
    }
    if promoted:
        if (
            not isinstance(journal, dict)
            or journal.get("phase") != "complete"
            or stage.get("population_checkpoint") != journal.get("canonical")
            or stage.get("sha256") != journal.get("new_sha256")
        ):
            raise RuntimeError("Migrated canonical population drifted")
    else:
        displaced = plan["displaced_nonqualifying_population"]
        if (
            stage.get("population_checkpoint") != displaced["checkpoint"]
            or stage.get("sha256") != displaced["sha256"]
        ):
            raise RuntimeError("Nonqualifying population changed before migration swap")
    released = migration.get("status") == "released"
    if bool(migration.get("official_population_released")) != released:
        raise RuntimeError("Migration release flag/status drifted")
    successor_release = stage.get("successor_release")
    if released and (
        not isinstance(successor_release, dict)
        or successor_release.get("approved") is not True
        or successor_release.get("migration_plan_id") != plan_id
    ):
        raise RuntimeError("Released migration authorization drifted")

    init = plan["trainable_init"]
    fixed = plan["fixed_opponent"]
    _strict_checkpoint(
        Path(str(init["checkpoint"])),
        expected_sha256=str(init["sha256"]),
    )
    _strict_checkpoint(
        Path(str(fixed["checkpoint"])),
        expected_sha256=str(fixed["sha256"]),
    )
    attempt = migration.get("attempt")
    if attempt is not None:
        if not isinstance(attempt, dict) or not isinstance(attempt.get("contract"), dict):
            raise RuntimeError("Migration attempt is malformed")
        attempt_id = verify_migration_attempt_contract(attempt["contract"], plan)
        if attempt.get("attempt_id") != attempt_id:
            raise RuntimeError("Migration attempt identity drifted")
        contract_path = Path(str(attempt.get("contract_path") or ""))
        if (
            _read_json_object(contract_path) != attempt["contract"]
            or file_sha256(contract_path)
            != attempt.get("contract_file_sha256")
        ):
            raise RuntimeError("Migration attempt contract artifact drifted")
    status = str(migration.get("status"))
    if status == "exhausted":
        if (
            state.get("status") != MIGRATION_EXHAUSTED_STATE_STATUS
            or migration.get("next_attempt_forbidden") is not True
            or int(migration.get("attempts_consumed", -1)) != 1
            or int(migration.get("attempt_limit", -1)) != 1
        ):
            raise RuntimeError("Exhausted migration terminal state drifted")
    elif status == "released":
        if state.get("status") not in {
            "objective_migration_released_pending_dispatch",
            "running",
            "spawn_pending_recovery",
            "child_started_recovery",
            "stage_target_not_reached",
            "stage_objective_migration_required",
            "awaiting_d1_paired_gate",
            "completed",
            "failed",
        }:
            raise RuntimeError("Released migration dispatch state drifted")
    elif state.get("status") != "stage_objective_migration_running":
        raise RuntimeError("Active migration state status drifted")
    return migration


def _verify_current_migration_history_implementations(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    history = state.get(MIGRATION_HISTORY_KEY)
    plan_ids = verify_migration_history(history)
    if not plan_ids or not isinstance(history, list):
        raise RuntimeError("Migration-aware finalization has no sealed migration")
    current_implementation = _migration_implementation_hashes()
    current_final_audit = _current_migration_final_audit_contract()
    frozen = _assert_training_implementation_frozen(state)
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("Migration history state has no stages mapping")
    rows: list[dict[str, Any]] = []
    for raw_row in history:
        row = dict(raw_row)
        migration = row["migration"]
        plan = migration["plan"]
        attempt = migration["attempt"]
        journal = migration["swap_journal"]
        label = str(row["stage_label"])
        stage = stages.get(label)
        if (
            plan.get("migration_implementation_sha256")
            != current_implementation
            or plan.get("migration_final_audit_contract")
            != current_final_audit
            or plan.get("frozen_training_implementation_sha256") != frozen
            or plan.get("frozen_selfplay_config") != state.get("config")
        ):
            raise RuntimeError(
                f"Sealed migration implementation drifted: {row['stage_label']}"
            )
        if (
            not isinstance(stage, dict)
            or stage.get("attacker_objective_migration_plan_id")
            != plan["plan_id"]
            or stage.get("attacker_objective_migration_attempt_id")
            != attempt["attempt_id"]
            or stage.get("population_checkpoint") != journal["canonical"]
            or stage.get("sha256") != journal["new_sha256"]
            or stage.get("attacker_objective_migration_gate")
            != attempt["gate_result"]
        ):
            raise RuntimeError(f"Sealed migration stage binding drifted: {label}")
        rows.append(row)
    marker_labels = {
        label
        for label, stage in stages.items()
        if isinstance(stage, dict)
        and stage.get("attacker_objective_migration_plan_id") is not None
    }
    if marker_labels != {str(row["stage_label"]) for row in rows}:
        raise RuntimeError("Migration history/stage membership drifted")
    return rows


def _audit_released_migration_artifacts(
    state: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Live-audit the binary promotion and its exhausted raw-PPO lineage."""

    indexed = index_migration_history_for_final_audit(state)
    label = str(row["stage_label"])
    if indexed.get(label) != row:
        raise RuntimeError(f"Migration final-audit index drifted: {label}")
    migration = row["migration"]
    plan = migration["plan"]
    attempt = migration["attempt"]
    contract = attempt["contract"]
    journal = migration["swap_journal"]
    if _read_json_object(Path(str(plan["plan_path"]))) != plan:
        raise RuntimeError(f"Migration plan artifact drifted: {label}")
    contract_path = Path(str(attempt["contract_path"]))
    if (
        _read_json_object(contract_path) != contract
        or file_sha256(contract_path) != attempt["contract_file_sha256"]
    ):
        raise RuntimeError(f"Migration attempt contract artifact drifted: {label}")
    artifact_paths = (
        (
            Path(str(attempt["manifest_path"])),
            attempt["manifest_sha256"],
            "trainer manifest",
        ),
        (
            Path(str(attempt["checkpoint_validation_path"])),
            attempt["checkpoint_validation_sha256"],
            "checkpoint validation",
        ),
        (
            Path(str(attempt["migration_manifest_path"])),
            attempt["migration_manifest_file_sha256"],
            "migration manifest",
        ),
    )
    for path, expected_sha, name in artifact_paths:
        if file_sha256(path) != expected_sha:
            raise RuntimeError(f"Migration {name} artifact drifted: {label}")
    trainer_manifest = _read_json_object(Path(str(attempt["manifest_path"])))
    migration_manifest = _read_json_object(
        Path(str(attempt["migration_manifest_path"]))
    )
    receipt = validate_migration_manifest(
        migration_manifest,
        trainer_manifest,
        plan=plan,
        contract=contract,
        migration_manifest_path=str(attempt["migration_manifest_path"]),
    )
    if (
        receipt != attempt["migration_manifest_receipt"]
        or receipt["migration_manifest_sha256"]
        != attempt["migration_manifest_sha256"]
    ):
        raise RuntimeError(f"Migration effective-trainer receipt drifted: {label}")
    validation = _read_json_object(
        Path(str(attempt["checkpoint_validation_path"]))
    )
    successful_gate = validate_migrated_stage_final_gate(
        state,
        stage_label=label,
        validation=validation,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(journal["new_sha256"]),
    )
    official = _strict_checkpoint(
        Path(str(journal["canonical"])),
        expected_sha256=str(journal["new_sha256"]),
    )
    displaced = _strict_checkpoint(
        Path(str(journal["archive"])),
        expected_sha256=str(journal["old_sha256"]),
    )

    raw_recovery = row["raw_gate_retry_recovery"]
    raw_plan = raw_recovery["plan"]
    if (
        raw_plan.get("recovery_implementation_sha256")
        != raw_retry._recovery_implementation_hashes()
        or raw_plan.get("frozen_training_implementation_sha256")
        != state["config"]["training_implementation_sha256"]
        or raw_plan.get("frozen_selfplay_config") != state["config"]
        or _read_json_object(Path(str(raw_plan["plan_path"]))) != raw_plan
    ):
        raise RuntimeError(f"Migration raw-PPO plan artifact drifted: {label}")
    original_failure = raw_plan.get("original_failure_evidence")
    if not isinstance(original_failure, dict):
        raise RuntimeError(f"Migration raw-PPO original failure is missing: {label}")
    original_validation_path = Path(
        str(original_failure.get("checkpoint_validation_path") or "")
    )
    if file_sha256(original_validation_path) != original_failure.get(
        "checkpoint_validation_sha256"
    ):
        raise RuntimeError(
            f"Migration original failed validation drifted: {label}"
        )
    rebuilt_original_failure = raw_retry.validate_exhausted_attempt(
        _read_json_object(original_validation_path),
        expected_budget=int(raw_plan["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(
            raw_plan["original_nonqualifying_population"]["sha256"]
        ),
    )
    rebuilt_original_failure.update(
        {
            "checkpoint_validation_path": str(original_validation_path),
            "checkpoint_validation_sha256": file_sha256(
                original_validation_path
            ),
        }
    )
    if rebuilt_original_failure != original_failure:
        raise RuntimeError(
            f"Migration original failed gate proof drifted: {label}"
        )
    _verify_requirement_artifact(raw_recovery)
    failed_candidates = []
    for raw_attempt in raw_recovery["attempts"]:
        raw_contract = raw_retry._verify_attempt(raw_recovery, raw_attempt)
        raw_artifact_paths = (
            (
                Path(str(raw_attempt["manifest_path"])),
                raw_attempt["manifest_sha256"],
                "trainer manifest",
            ),
            (
                Path(str(raw_attempt["checkpoint_validation_path"])),
                raw_attempt["checkpoint_validation_sha256"],
                "checkpoint validation",
            ),
            (
                Path(str(raw_attempt["recovery_manifest_path"])),
                raw_attempt["recovery_manifest_file_sha256"],
                "recovery manifest",
            ),
        )
        for path, expected_sha, name in raw_artifact_paths:
            if file_sha256(path) != expected_sha:
                raise RuntimeError(
                    f"Migration raw-PPO {name} drifted: {label}"
                )
        raw_trainer_manifest = _read_json_object(
            Path(str(raw_attempt["manifest_path"]))
        )
        raw_manifest = _read_json_object(
            Path(str(raw_attempt["recovery_manifest_path"]))
        )
        raw_receipt = raw_retry.validate_ppo_only_recovery_manifest(
            raw_manifest,
            raw_trainer_manifest,
            plan=raw_plan,
            contract=raw_contract,
            recovery_manifest_path=str(raw_attempt["recovery_manifest_path"]),
        )
        if (
            raw_receipt != raw_attempt["ppo_only_recovery_receipt"]
            or raw_receipt["recovery_manifest_sha256"]
            != raw_attempt["recovery_manifest_sha256"]
        ):
            raise RuntimeError(
                f"Migration raw-PPO trainer receipt drifted: {label}"
            )
        raw_validation = raw_retry._read_retry_checkpoint_validation(
            Path(str(raw_attempt["run_dir"])),
            expected_budget=int(raw_contract["per_attempt_budget"]),
            save_steps=int(state["config"]["save_steps"]),
        )
        failed_gate = raw_retry.validate_exhausted_attempt(
            raw_validation,
            expected_budget=int(raw_contract["per_attempt_budget"]),
            save_steps=int(state["config"]["save_steps"]),
            expected_final_sha256=str(raw_attempt["candidate_sha256"]),
        )
        if (
            raw_attempt.get("status") != "gate_not_reached"
            or raw_attempt.get("pruning_complete") is not True
            or failed_gate != raw_attempt.get("gate_result")
        ):
            raise RuntimeError(f"Migration raw-PPO failure proof drifted: {label}")
        candidate = _strict_checkpoint(
            Path(str(raw_attempt["candidate_checkpoint"])),
            expected_sha256=str(raw_attempt["candidate_sha256"]),
        )
        failed_candidates.append(
            {
                "attempt_id": raw_attempt["attempt_id"],
                "checkpoint": raw_attempt["candidate_checkpoint"],
                "sha256": candidate["weight_sha256"],
            }
        )
    return {
        "passed": True,
        "stage_label": label,
        "history_entry_id": row["history_entry_id"],
        "migration_plan_id": plan["plan_id"],
        "migration_attempt_id": attempt["attempt_id"],
        "binary_gate": successful_gate,
        "official": {
            "checkpoint": journal["canonical"],
            "sha256": official["weight_sha256"],
        },
        "displaced_nonqualifying": {
            "checkpoint": journal["archive"],
            "sha256": displaced["weight_sha256"],
        },
        "raw_gate_retry_plan_id": raw_plan["plan_id"],
        "raw_failed_candidates": failed_candidates,
        "official_payoff_utility_unchanged": True,
        "binary_surrogate_is_payoff_entry": False,
    }


def _build_migration_aware_final_population_audit(
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    rows = _verify_current_migration_history_implementations(state)
    indexed = index_migration_history_for_final_audit(state)
    migration_audits = [
        _audit_released_migration_artifacts(state, row) for row in rows
    ]
    source = Path(raw_retry.__file__).read_text(encoding="utf-8")
    effective_source, descriptor = build_migration_aware_final_audit_source(
        source
    )
    contract = _current_migration_final_audit_contract()
    if descriptor != contract["live_final_audit_clone"]:
        raise RuntimeError("Effective migration final-audit clone drifted")
    observed_migrations: set[str] = set()

    def validate_migration_or_live_gate(
        validation: dict[str, Any],
        *,
        migration_state: dict[str, Any],
        stage_label: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if migration_state is not state:
            raise RuntimeError("Migration final gate received a different state")
        if stage_label not in indexed:
            return raw_retry.validate_successful_gate(validation, **kwargs)
        if (
            kwargs.get("role") != "attacker"
            or float(kwargs.get("threshold", -1.0)) != 0.95
            or int(kwargs.get("patience", -1)) != 5
        ):
            raise RuntimeError(
                f"Migrated final gate caller drifted: {stage_label}"
            )
        observed_migrations.add(stage_label)
        return validate_migrated_stage_final_gate(
            state,
            stage_label=stage_label,
            validation=validation,
            expected_budget=int(kwargs["expected_budget"]),
            save_steps=int(kwargs["save_steps"]),
            expected_final_sha256=str(kwargs["expected_final_sha256"]),
        )

    namespace = dict(vars(raw_retry))
    namespace["validate_migration_or_live_gate"] = (
        validate_migration_or_live_gate
    )
    exec(
        compile(
            effective_source,
            "<migration-aware-final-population-audit>",
            "exec",
        ),
        namespace,
    )
    builder = namespace["_effective_migration_aware_final_population_audit"]
    artifact = builder(root, state)
    if observed_migrations != set(indexed):
        raise RuntimeError("Not every sealed migration reached final gate audit")
    artifact.pop("audit_sha256", None)
    artifact["attacker_objective_migration_final_audit"] = {
        "schema_version": 1,
        "policy": MIGRATION_FINAL_AUDIT_POLICY,
        "migration_history_entry_ids": [
            row["history_entry_id"] for row in rows
        ],
        "migration_plan_ids": [row["plan_id"] for row in rows],
        "migration_stage_labels": [row["stage_label"] for row in rows],
        "migration_artifact_audits": migration_audits,
        "migration_implementation_sha256": (
            _migration_implementation_hashes()
        ),
        "migration_final_audit_contract": contract,
        "nonmigration_gate_and_population_audit": (
            "unchanged live gate-retry strict implementation"
        ),
        "official_payoff_utility_unchanged": True,
        "binary_surrogate_is_payoff_entry": False,
    }
    artifact["audit_sha256"] = canonical_json_sha256(artifact)
    return artifact


def _verify_migration_final_population_audit_reference(
    state: dict[str, Any],
) -> dict[str, Any]:
    artifact = raw_retry._verify_final_population_audit_reference(state)
    extension = artifact.get("attacker_objective_migration_final_audit")
    rows = _verify_current_migration_history_implementations(state)
    if (
        not isinstance(extension, dict)
        or extension.get("schema_version") != 1
        or extension.get("policy") != MIGRATION_FINAL_AUDIT_POLICY
        or extension.get("migration_history_entry_ids")
        != [row["history_entry_id"] for row in rows]
        or extension.get("migration_plan_ids")
        != [row["plan_id"] for row in rows]
        or extension.get("migration_stage_labels")
        != [row["stage_label"] for row in rows]
        or extension.get("migration_implementation_sha256")
        != _migration_implementation_hashes()
        or extension.get("migration_final_audit_contract")
        != _current_migration_final_audit_contract()
        or extension.get("official_payoff_utility_unchanged") is not True
        or extension.get("binary_surrogate_is_payoff_entry") is not False
    ):
        raise RuntimeError("Migration final population audit reference drifted")
    return artifact


def _audit_completed_population_with_migrations(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> dict[str, Any]:
    if state.get("status") != "completed" or state.get("active_stage") is not None:
        raise RuntimeError("Final migration audit requires a completed chain")
    _verify_current_migration_history_implementations(state)
    existing = state.get("final_population_audit")
    if isinstance(existing, dict) and existing.get("passed") is True:
        recorded = _verify_migration_final_population_audit_reference(state)
        live = _build_migration_aware_final_population_audit(root, state)
        output_vol.reload()
        current, current_sha = _load_state_snapshot(root)
        if current_sha != state_sha256 or current != state:
            raise RuntimeError("State changed during repeated migration final audit")
        reread = _verify_migration_final_population_audit_reference(state)
        if reread != recorded or live != recorded:
            raise RuntimeError(
                "Repeated migration-aware live audit differs from artifact"
            )
        return {
            "root": str(root),
            "state": state,
            "already_audited": True,
            "spawned": False,
            "audit": live,
            "state_file_sha256": state_sha256,
        }
    audit = _build_migration_aware_final_population_audit(root, state)
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed during migration final population audit")
    audit_path = root / MIGRATION_ROOT_NAME / "final_population_audit_v1.json"
    _write_exact_json(audit_path, audit)
    output_vol.commit()
    updated = copy.deepcopy(state)
    updated["final_population_audit"] = {
        "passed": True,
        "policy": MIGRATION_FINAL_AUDIT_POLICY,
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


def _migration_aware_release_or_complete(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> dict[str, Any]:
    source = Path(raw_retry.__file__).read_text(encoding="utf-8")
    sources, descriptor = build_migration_aware_gate_retry_finalize_sources(
        source
    )
    contract = _current_migration_final_audit_contract()
    if descriptor != contract["d8_gate_retry_finalize_clones"]:
        raise RuntimeError("Effective migration D8 finalizer clone drifted")
    namespace = dict(vars(raw_retry))
    namespace["_build_final_population_audit"] = (
        _build_migration_aware_final_population_audit
    )
    exec(
        compile(
            sources["_release_or_complete"],
            "<migration-aware-release-or-complete>",
            "exec",
        ),
        namespace,
    )
    finalize = namespace["_effective_migration_aware_release_or_complete"]
    return finalize(root, state, state_sha256)


def _drain_recovery_phase_with_migration_audit(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str] | dict[str, Any]:
    for _ in range(4):
        before_state = state
        before_sha = state_sha256
        recovery = raw_retry._verify_existing_recovery(root, state)
        if recovery.get("status") in {
            "qualified_ready_to_release",
            "released",
        }:
            return _migration_aware_release_or_complete(
                root, state, state_sha256
            )
        result = raw_retry._continue_existing_phase(root, state, state_sha256)
        if isinstance(result, dict):
            return result
        state, state_sha256 = result
        if state_sha256 == before_sha and state == before_state:
            return state, state_sha256
        status = raw_retry._verify_existing_recovery(root, state).get("status")
        if status not in {
            "active",
            "swap_prepared",
            "promoted_pending_prune",
            "qualified_ready_to_release",
            "released",
        }:
            return state, state_sha256
    raise RuntimeError("Migration-aware gate-retry phase did not converge")


def _resume_gate_retry_with_migration_audit(run_suffix: str) -> dict[str, Any]:
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state, _state_sha = _load_state_snapshot(root)
    _verify_current_migration_history_implementations(state)
    source = Path(raw_retry.__file__).read_text(encoding="utf-8")
    sources, descriptor = build_migration_aware_gate_retry_finalize_sources(
        source
    )
    contract = _current_migration_final_audit_contract()
    if descriptor != contract["d8_gate_retry_finalize_clones"]:
        raise RuntimeError("Effective migration gate-retry resumer clone drifted")

    def load_state_with_migration_verification(
        candidate_root: Path,
    ) -> tuple[dict[str, Any], str]:
        candidate_state, candidate_sha = raw_retry._load_state_snapshot(
            candidate_root
        )
        _verify_current_migration_history_implementations(candidate_state)
        return candidate_state, candidate_sha

    namespace = dict(vars(raw_retry))
    namespace.update(
        {
            "_audit_completed_population": (
                _audit_completed_population_with_migrations
            ),
            "_drain_recovery_phase": (
                _drain_recovery_phase_with_migration_audit
            ),
            "_load_state_snapshot": load_state_with_migration_verification,
            "_release_or_complete": _migration_aware_release_or_complete,
        }
    )
    exec(
        compile(
            sources["resume_role_lora_selfplay8_gate_retry"],
            "<migration-aware-gate-retry-resume>",
            "exec",
        ),
        namespace,
    )
    resume = namespace["_effective_migration_aware_gate_retry_resume"]
    return resume(run_suffix)


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
def train_attacker_binary_joint_objective_migration(
    plan: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Run the single hash-bound binary-joint attacker oracle attempt."""

    plan_id = verify_migration_plan(plan)
    attempt_id = verify_migration_attempt_contract(contract, plan)
    frozen = dict(plan["frozen_training_implementation_sha256"])
    _assert_training_implementation_frozen(
        {"config": {"training_implementation_sha256": frozen}}
    )
    current_implementation = _migration_implementation_hashes()
    if plan["migration_implementation_sha256"] != current_implementation:
        raise RuntimeError("Migration implementation drifted")
    core_source = Path(frozen_role_lora.__file__).read_text(encoding="utf-8")
    effective_source, effective_contract = (
        build_binary_joint_migration_trainer_source(core_source)
    )
    if plan["migration_trainer_contract"] != effective_contract:
        raise RuntimeError("Migration effective trainer drifted")

    init_checkpoint = Path(str(contract["trainable_init_checkpoint"]))
    fixed_checkpoint = Path(str(contract["fixed_opponent"]["checkpoint"]))
    if (
        checkpoint_weight_digest(init_checkpoint)
        != contract["trainable_init_sha256"]
        or checkpoint_weight_digest(fixed_checkpoint)
        != contract["fixed_opponent"]["sha256"]
    ):
        raise RuntimeError("Migration input adapter digest drifted")

    upstream_patch: dict[str, Any] | None = None
    original_prepare = frozen_role_lora._prepare_role_lora_upstream

    def migration_prepare(*args: Any, **kwargs: Any) -> None:
        nonlocal upstream_patch
        original_prepare(*args, **kwargs)
        upstream_patch = patch_binary_joint_upstream_tree(
            Path(frozen_role_lora.UPSTREAM_WORK)
        )

    namespace = dict(vars(frozen_role_lora))
    namespace["_prepare_role_lora_upstream"] = migration_prepare
    exec(
        compile(
            effective_source,
            str(Path(frozen_role_lora.__file__).resolve()),
            "exec",
        ),
        namespace,
    )
    effective_train = namespace.get(
        "_effective_attacker_binary_joint_migration_train"
    )
    if not callable(effective_train):
        raise RuntimeError("Migration effective trainer was not constructed")

    config = dict(plan["frozen_selfplay_config"])
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
        actor_learning_rate=float(config["attacker_learning_rate"]),
        init_kl_coef=0.0,
        actor_lr_scheduler="constant_with_warmup",
        lr_warmup_ratio=0.05,
        actor_lr_warmup_steps_override=None,
        enable_aux_sft=False,
        run_suffix=str(contract["trainer_run_suffix"]),
        train_role="attacker",
        fixed_attacker_adapter="",
        fixed_defender_adapter=str(fixed_checkpoint),
        defender_prompt_profile="upstream",
        balance_defender_refusal_replay=False,
        balance_attacker_goal_replay=False,
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
        defender_raw_reinforce_advantages=False,
        defender_reinforce_advantage_mode="raw_no_center",
        defender_reward_utility="upstream_additive",
        defender_prompt_pool_path="",
        defender_prompt_pool_sha256="",
        expected_implementation_sha256=frozen,
        early_stop_threshold=0.95,
        early_stop_patience=5,
        early_stop_min_steps=1,
    )
    run_dir = Path(str(run_dir_text))
    if upstream_patch is None:
        # A cold container may replay a completed/early-stopped suffix before
        # the frozen trainer reaches source preparation.  Reconstruct the exact
        # patch receipt without changing the already persisted run artifacts.
        migration_prepare(
            "optimized",
            strict_upstream_alignment=False,
            dynamic_role_sft=False,
            v2_runtime=True,
            v2_continuation_sft=False,
        )
    if upstream_patch is None:
        raise RuntimeError("Migration upstream patch receipt is missing")

    trainable_runtime = Path("/tmp/attacker_lora_init_compatible")
    fixed_runtime = Path("/tmp/fixed_opponent_lora_compatible")
    runtime_mapping = {
        "trainable": {
            "original_checkpoint": str(init_checkpoint),
            "original_sha256": contract["trainable_init_sha256"],
            "runtime_compatible_checkpoint": str(trainable_runtime),
            "runtime_weight_sha256": _ensure_runtime_compatible_adapter_after_replay(
                source=init_checkpoint,
                runtime=trainable_runtime,
                destination_name="attacker_lora_init_compatible",
                expected_sha256=str(contract["trainable_init_sha256"]),
            ),
        },
        "fixed_opponent": {
            "original_checkpoint": str(fixed_checkpoint),
            "original_sha256": contract["fixed_opponent"]["sha256"],
            "runtime_compatible_checkpoint": str(fixed_runtime),
            "runtime_weight_sha256": _ensure_runtime_compatible_adapter_after_replay(
                source=fixed_checkpoint,
                runtime=fixed_runtime,
                destination_name="fixed_opponent_lora_compatible",
                expected_sha256=str(contract["fixed_opponent"]["sha256"]),
            ),
        },
    }
    if any(
        row["original_sha256"] != row["runtime_weight_sha256"]
        for row in runtime_mapping.values()
    ):
        raise RuntimeError("Migration runtime adapter digest drifted")

    manifest_path = run_dir / "manifest.json"
    trainer_manifest = _read_json_object(manifest_path)
    migration_manifest: dict[str, Any] = {
        "schema_version": 1,
        "policy": MIGRATION_MANIFEST_POLICY,
        "plan_id": plan_id,
        "attempt_id": attempt_id,
        "stage_label": contract["stage_label"],
        "role": "attacker",
        "trainer_run_suffix": contract["trainer_run_suffix"],
        "frozen_training_implementation_sha256": frozen,
        "migration_implementation_sha256": current_implementation,
        "effective_trainer_contract": effective_contract,
        "patched_upstream_contract": upstream_patch,
        "runtime_adapter_mapping": runtime_mapping,
        "objective": binary_joint_objective_contract(),
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
            "trajectory_sum_fixed_scale": 1.0 / 2048.0,
            "failure_and_tie_replay_retained": True,
            "fixed_opponent_unchanged": True,
        },
        "official_payoff_evaluation": {
            "utility": "frozen_upstream_additive_raw_utility",
            "normalization": "none",
            "matrix_namespace": "historical_psro_payoff_unchanged",
            "binary_surrogate_is_payoff_entry": False,
            "required_path": (
                "separate frozen payoff evaluator; never consume migration "
                "training reward logs as matrix cells"
            ),
            "training_metric_namespaces": {
                "attacker/binary_joint_optimization_reward_mean": (
                    "oracle_surrogate_only"
                ),
                "attacker/official_raw_utility_diagnostic_mean": (
                    "diagnostic_only_not_a_matrix_estimate"
                ),
                "attacker/avg_reward": (
                    "legacy_alias_of_binary_surrogate_for_this_run_only"
                ),
            },
        },
    }
    migration_manifest["migration_manifest_sha256"] = canonical_json_sha256(
        migration_manifest
    )
    migration_manifest_path = run_dir / "attacker_objective_migration_manifest.json"
    _write_exact_json(migration_manifest_path, migration_manifest)
    trainer_manifest["attacker_objective_migration"] = {
        "policy": MIGRATION_MANIFEST_POLICY,
        "path": str(migration_manifest_path),
        "sha256": migration_manifest["migration_manifest_sha256"],
        "frozen_core_implementation_sha256": frozen[
            "modal_upstream_selfredteam_role_lora.py"
        ],
        "effective_function_source_sha256": effective_contract[
            "effective_function_source_sha256"
        ],
        "patch_descriptor_sha256": effective_contract[
            "patch_descriptor_sha256"
        ],
        "patched_upstream_descriptor_sha256": upstream_patch[
            "patch_descriptor_sha256"
        ],
        "official_payoff_utility_unchanged": True,
    }
    _write_json_atomic(manifest_path, trainer_manifest)
    output_vol.commit()
    return {
        "run_dir": str(run_dir),
        "migration_manifest_path": str(migration_manifest_path),
        "migration_manifest_sha256": migration_manifest[
            "migration_manifest_sha256"
        ],
        "effective_trainer_contract": effective_contract,
        "patched_upstream_contract": upstream_patch,
    }


def _initialize_migration(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    if MIGRATION_KEY in state:
        raise RuntimeError("Objective migration is already initialized")
    verify_migration_history(state.get(MIGRATION_HISTORY_KEY))
    eligibility = validate_migration_eligibility(state)
    # Reuse the complete gate-retry live verifier at the only point where its
    # terminal-state invariant still applies.
    raw_retry._verify_existing_recovery(root, state)
    frozen = _assert_training_implementation_frozen(state)
    implementation = _migration_implementation_hashes()
    trainer_contract = _current_migration_trainer_contract(frozen)
    final_audit_contract = _current_migration_final_audit_contract()
    label = str(eligibility["label"])
    plan_path = _migration_root(root, label) / "plan.json"
    plan = build_migration_plan(
        state,
        migration_implementation_sha256=implementation,
        migration_trainer_contract=trainer_contract,
        migration_final_audit_contract=final_audit_contract,
        plan_path=str(plan_path),
    )
    _write_exact_json(plan_path, plan)
    output_vol.commit()
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while initializing objective migration")
    updated = copy.deepcopy(state)
    updated[MIGRATION_KEY] = {
        "schema_version": 1,
        "policy": MIGRATION_POLICY,
        "status": "active",
        "plan_id": plan["plan_id"],
        "plan": plan,
        "attempt": None,
        "active_attempt_id": None,
        "swap_journal": None,
        "official_population_released": False,
    }
    updated["status"] = "stage_objective_migration_running"
    updated["active_stage"] = label
    updated["stages"][label]["work_status"] = (
        "attacker_binary_joint_objective_migration_running"
    )
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha


def _create_attempt(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    migration = _verify_existing_migration(root, state)
    if migration.get("status") != "active" or migration.get("attempt") is not None:
        raise RuntimeError("Migration attempt cannot be created in this phase")
    plan = migration["plan"]
    label = str(plan["stage_label"])
    suffix = (
        f"{state['run_suffix']}_{label.lower()}_binary_joint_"
        f"{str(plan['plan_id'])[:16]}"
    )
    attempt_root = _migration_root(root, label) / "attempt_01"
    contract_path = attempt_root / "contract.json"
    contract = build_migration_attempt_contract(
        plan,
        trainer_run_suffix=suffix,
        contract_path=str(contract_path),
    )
    _write_exact_json(contract_path, contract)
    output_vol.commit()
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while claiming migration attempt")
    attempt = {
        "schema_version": 1,
        "attempt_number": 1,
        "attempt_id": contract["attempt_id"],
        "status": "training",
        "contract": contract,
        "contract_path": str(contract_path),
        "contract_file_sha256": file_sha256(contract_path),
    }
    updated = copy.deepcopy(state)
    updated[MIGRATION_KEY]["attempt"] = attempt
    updated[MIGRATION_KEY]["active_attempt_id"] = contract["attempt_id"]
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha, attempt


def _run_attempt(
    migration: dict[str, Any],
    attempt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    contract = attempt["contract"]
    verify_migration_attempt_contract(contract, migration["plan"])
    result = train_attacker_binary_joint_objective_migration.remote(
        plan=migration["plan"],
        contract=contract,
    )
    if not isinstance(result, dict) or not result.get("run_dir"):
        raise RuntimeError("Migration trainer returned an invalid result")
    return Path(str(result["run_dir"])), result


def _read_checkpoint_validation(
    run_dir: Path,
    *,
    expected_budget: int,
    save_steps: int,
) -> dict[str, Any]:
    validation = _read_json_object(run_dir / "checkpoint_validation.json")
    if (
        int(validation.get("requested_max_step", -1)) != expected_budget
        or int(validation.get("actual_final_step", 0)) <= 0
        or int(validation.get("actual_final_step", 0)) > expected_budget
    ):
        raise RuntimeError("Migration checkpoint validation budget drifted")
    if validation.get("stopped_early") is not True:
        if int(validation.get("actual_final_step", -1)) != expected_budget:
            raise RuntimeError("Migration attempt neither stopped nor exhausted")
        validate_checkpoint_cadence(
            validation,
            expected_final_step=expected_budget,
            save_steps=save_steps,
        )
    return validation


def _validate_attempt_output(
    state: dict[str, Any],
    migration: dict[str, Any],
    attempt: dict[str, Any],
    run_dir: Path,
    trainer_result: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = migration["plan"]
    contract = attempt["contract"]
    _assert_trainer_manifest_implementation(
        run_dir,
        dict(plan["frozen_training_implementation_sha256"]),
    )
    manifest_path = run_dir / "manifest.json"
    trainer_manifest = _read_json_object(manifest_path)
    migration_manifest_path = Path(str(trainer_result["migration_manifest_path"]))
    migration_manifest = _read_json_object(migration_manifest_path)
    receipt = validate_migration_manifest(
        migration_manifest,
        trainer_manifest,
        plan=plan,
        contract=contract,
        migration_manifest_path=str(migration_manifest_path),
    )
    validation = _read_checkpoint_validation(
        run_dir,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
    )
    source_checkpoint = Path(str(validation.get("final_checkpoint") or ""))
    source_audit = _strict_audit(source_checkpoint)
    source_sha = str(source_audit.get("weight_sha256") or "")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", source_sha)
        or source_sha == contract["trainable_init_sha256"]
        or checkpoint_weight_digest(source_checkpoint) != source_sha
    ):
        raise RuntimeError("Migration checkpoint did not prove a weight change")
    artifact = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_validation_path": str(
            run_dir / "checkpoint_validation.json"
        ),
        "checkpoint_validation_sha256": file_sha256(
            run_dir / "checkpoint_validation.json"
        ),
        "migration_manifest_path": str(migration_manifest_path),
        "migration_manifest_file_sha256": file_sha256(migration_manifest_path),
        "migration_manifest_sha256": migration_manifest[
            "migration_manifest_sha256"
        ],
        "migration_manifest_receipt": receipt,
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": source_sha,
        "source_strict_audit": source_audit,
    }
    return validation, source_checkpoint, artifact


def _prepare_successful_swap(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
    *,
    validation: dict[str, Any],
    source_checkpoint: Path,
    artifact: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    migration = _verify_existing_migration(root, state)
    plan = migration["plan"]
    attempt = migration["attempt"]
    contract = attempt["contract"]
    gate = validate_binary_joint_successful_gate(
        validation,
        expected_budget=int(contract["per_attempt_budget"]),
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(artifact["source_sha256"]),
        expected_initial_sha256=str(contract["trainable_init_sha256"]),
    )
    label = str(plan["stage_label"])
    staging_root = _migration_root(root, label) / "swap_staging" / str(
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
    original = plan["displaced_nonqualifying_population"]
    canonical = _population_path(root, label)
    archive = (
        root
        / _POPULATION_ATTEMPTS_NAME
        / label
        / f"pre_migration_nonqualifying_{str(original['sha256'])[:12]}"
        / label
    )
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while preparing migration swap")
    updated = copy.deepcopy(state)
    updated_attempt = updated[MIGRATION_KEY]["attempt"]
    updated_attempt.update(
        {
            "status": "qualified_staging",
            "gate_result": gate,
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
    updated[MIGRATION_KEY]["swap_journal"] = journal
    updated[MIGRATION_KEY]["status"] = "swap_prepared"
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha


def _reconcile_swap(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    migration = _verify_existing_migration(root, state)
    if migration.get("status") != "swap_prepared":
        return state, state_sha256
    journal = migration.get("swap_journal")
    if not isinstance(journal, dict):
        raise RuntimeError("Migration swap journal is missing")
    payload = dict(journal)
    stored_id = payload.pop("journal_id", None)
    if stored_id != canonical_json_sha256(payload):
        raise RuntimeError("Migration swap journal digest drifted")
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
        raise RuntimeError("Migration population swap did not converge")
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
        raise RuntimeError("State changed while reconciling migration swap")
    updated = copy.deepcopy(state)
    current_migration = updated[MIGRATION_KEY]
    attempt = current_migration["attempt"]
    label = str(current_migration["plan"]["stage_label"])
    validation = _read_json_object(Path(attempt["checkpoint_validation_path"]))
    updated["stages"][label].update(
        {
            "work_status": "retained_after_attacker_objective_migration",
            "run_dir": attempt["run_dir"],
            "source_checkpoint": attempt["source_checkpoint"],
            "source_sha256": attempt["source_sha256"],
            "population_checkpoint": journal["canonical"],
            "actual_final_step": int(validation["actual_final_step"]),
            "requested_max_step": int(
                current_migration["plan"]["attempt_policy"][
                    "per_attempt_budget"
                ]
            ),
            "stopped_early": True,
            "sha256": journal["new_sha256"],
            "strict_audit": canonical_audit,
            "attacker_objective_migration_plan_id": current_migration[
                "plan_id"
            ],
            "attacker_objective_migration_attempt_id": attempt["attempt_id"],
            "attacker_objective_migration_gate": attempt["gate_result"],
            "optimization_objective": binary_joint_objective_contract(),
            "official_payoff_evaluation": {
                "utility": "frozen_upstream_additive_raw_utility",
                "normalization": "none",
                "binary_surrogate_is_payoff_entry": False,
            },
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
    current_migration["status"] = "promoted_pending_prune"
    current_migration["swap_journal"]["phase"] = "complete"
    journal_payload = dict(current_migration["swap_journal"])
    journal_payload.pop("journal_id", None)
    current_migration["swap_journal"]["journal_id"] = canonical_json_sha256(
        journal_payload
    )
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha


def _prune_promoted(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    migration = _verify_existing_migration(root, state)
    if migration.get("status") != "promoted_pending_prune":
        return state, state_sha256
    attempt = migration["attempt"]
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
        raise RuntimeError("State changed while pruning migrated checkpoint")
    updated = copy.deepcopy(state)
    updated_attempt = updated[MIGRATION_KEY]["attempt"]
    updated_attempt["pruned_source_hf_checkpoints"] = removed
    updated_attempt["pruning_complete"] = True
    updated_attempt["status"] = "qualified_ready_to_release"
    updated[MIGRATION_KEY]["status"] = "qualified_ready_to_release"
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha


def _release_and_dispatch(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> dict[str, Any]:
    migration = _verify_existing_migration(root, state)
    if migration.get("status") not in {
        "qualified_ready_to_release",
        "released",
    }:
        raise RuntimeError("Migration is not ready to release")
    plan = migration["plan"]
    label = str(plan["stage_label"])
    attempt = migration["attempt"]
    if migration.get("status") == "qualified_ready_to_release":
        updated = copy.deepcopy(state)
        released = updated[MIGRATION_KEY]
        released["status"] = "released"
        released["official_population_released"] = True
        updated_stage = updated["stages"][label]
        updated_stage["successor_release"] = {
            "approved": True,
            "basis": (
                "single bounded attacker binary goal+CoT oracle migration "
                "passed five consecutive exact-g point estimates"
            ),
            "migration_plan_id": plan["plan_id"],
            "migration_attempt_id": attempt["attempt_id"],
            "gate_result": attempt["gate_result"],
            "official_payoff_utility_unchanged": True,
            "binary_surrogate_is_payoff_entry": False,
            "displaced_nonqualifying_population_preserved": True,
        }
        updated["status"] = "objective_migration_released_pending_dispatch"
        updated["active_stage"] = label
        state_sha256 = _persist_state_cas(
            root,
            updated,
            expected_file_sha256=state_sha256,
        )
        state = updated

    schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
    position = next(index for index, stage in enumerate(schedule) if stage.label == label)
    if position + 1 >= len(schedule):
        raise RuntimeError("Attacker migration unexpectedly targeted final D stage")
    successor = schedule[position + 1]
    successor_state = (state.get("stages") or {}).get(successor.label)
    retry_pending = bool(
        isinstance(successor_state, dict)
        and successor_state.get("transition_state") == "spawn_pending"
    )
    dispatch = _dispatch_stage_claim(
        root,
        state,
        run_suffix=str(state["run_suffix"]),
        stage=successor,
        retry_existing_pending=retry_pending,
    )

    # The successor claim is durable before the migration/raw-retry keys are
    # archived and removed.  A crash before this cleanup re-enters the
    # ``released`` phase and safely reconciles the deterministic claim.
    output_vol.reload()
    latest, latest_sha = _load_state_snapshot(root)
    latest_migration = _verify_existing_migration(root, latest)
    if latest_migration.get("status") != "released":
        raise RuntimeError("Released migration changed during successor dispatch")
    raw_recovery = copy.deepcopy(latest[RECOVERY_KEY])
    history = list(latest.get(MIGRATION_HISTORY_KEY) or [])
    entry = build_migration_history_entry(
        latest_migration,
        raw_recovery,
        archived_state_file_sha256=latest_sha,
    )
    if any(row.get("plan_id") == entry["plan_id"] for row in history):
        raise RuntimeError("Migration is already archived")
    cleaned = copy.deepcopy(latest)
    cleaned[MIGRATION_HISTORY_KEY] = [*history, entry]
    del cleaned[MIGRATION_KEY]
    del cleaned[RECOVERY_KEY]
    final_sha = _persist_state_cas(
        root,
        cleaned,
        expected_file_sha256=latest_sha,
    )
    return {
        "root": str(root),
        "state": cleaned,
        "migrated_stage": label,
        "attempt_id": attempt["attempt_id"],
        "spawned": dispatch["spawned"],
        "call_id": dispatch["call_id"],
        "spawn_claim_id": dispatch["spawn_claim_id"],
        "final_state_file_sha256": final_sha,
    }


def _prepare_exhausted_archive(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
    *,
    validation: dict[str, Any],
    source_checkpoint: Path,
    artifact: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    migration = _verify_existing_migration(root, state)
    plan = migration["plan"]
    attempt = migration["attempt"]
    contract = attempt["contract"]
    budget = int(contract["per_attempt_budget"])
    if (
        validation.get("stopped_early") is not False
        or int(validation.get("actual_final_step", -1)) != budget
        or int(validation.get("requested_max_step", -1)) != budget
    ):
        raise RuntimeError("Migration exhaustion proof is incomplete")
    cadence = validate_checkpoint_cadence(
        validation,
        expected_final_step=budget,
        save_steps=int(state["config"]["save_steps"]),
        expected_final_sha256=str(artifact["source_sha256"]),
    )
    label = str(plan["stage_label"])
    archive_root = (
        root
        / _POPULATION_ATTEMPTS_NAME
        / label
        / f"failed_binary_migration_{attempt['attempt_id'][:12]}"
    )
    archived = atomic_copy_population_checkpoint(
        source_checkpoint,
        archive_root,
        label,
    )
    output_vol.commit()
    archive_audit = _strict_checkpoint(
        Path(str(archived["path"])),
        expected_sha256=str(archived["sha256"]),
    )
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while archiving exhausted migration")
    updated = copy.deepcopy(state)
    updated_attempt = updated[MIGRATION_KEY]["attempt"]
    updated_attempt.update(
        {
            "status": "exhausted_pending_prune",
            "gate_result": {
                "passed": False,
                "classification": "binary_objective_gate_not_reached_after_complete_budget",
                "checkpoint_cadence": cadence,
                "metric": MIGRATION_REQUIRED_METRIC,
            },
            "candidate_checkpoint": archived["path"],
            "candidate_sha256": archived["sha256"],
            "candidate_strict_audit": archive_audit,
            "pruning_complete": False,
            **artifact,
        }
    )
    updated[MIGRATION_KEY]["status"] = "exhausted_pending_prune"
    updated[MIGRATION_KEY]["active_attempt_id"] = None
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha


def _prune_exhausted(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str]:
    migration = _verify_existing_migration(root, state)
    if migration.get("status") != "exhausted_pending_prune":
        return state, state_sha256
    attempt = migration["attempt"]
    candidate = Path(str(attempt["candidate_checkpoint"]))
    digest = str(attempt["candidate_sha256"])
    _strict_checkpoint(candidate, expected_sha256=digest)
    removed = prune_stage_hf_checkpoints(
        Path(str(attempt["run_dir"])) / "ckpt",
        audited_population_checkpoint=candidate,
        audited_sha256=digest,
    )
    output_vol.commit()
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha256 or current != state:
        raise RuntimeError("State changed while pruning exhausted migration")
    updated = copy.deepcopy(state)
    updated_attempt = updated[MIGRATION_KEY]["attempt"]
    updated_attempt["pruned_source_hf_checkpoints"] = removed
    updated_attempt["pruning_complete"] = True
    updated_attempt["status"] = "exhausted"
    updated[MIGRATION_KEY]["status"] = "exhausted"
    updated[MIGRATION_KEY]["next_attempt_forbidden"] = True
    updated[MIGRATION_KEY]["attempts_consumed"] = 1
    updated[MIGRATION_KEY]["attempt_limit"] = 1
    label = str(updated[MIGRATION_KEY]["plan"]["stage_label"])
    updated["status"] = MIGRATION_EXHAUSTED_STATE_STATUS
    updated["active_stage"] = label
    updated["stages"][label]["work_status"] = (
        "single_binary_objective_migration_exhausted"
    )
    new_sha = _persist_state_cas(
        root,
        updated,
        expected_file_sha256=current_sha,
    )
    return updated, new_sha


def _drain_phase(
    root: Path,
    state: dict[str, Any],
    state_sha256: str,
) -> tuple[dict[str, Any], str] | dict[str, Any]:
    migration = _verify_existing_migration(root, state)
    status = migration.get("status")
    if status == "swap_prepared":
        return _reconcile_swap(root, state, state_sha256)
    if status == "promoted_pending_prune":
        return _prune_promoted(root, state, state_sha256)
    if status == "qualified_ready_to_release":
        return _release_and_dispatch(root, state, state_sha256)
    if status == "released":
        return _release_and_dispatch(root, state, state_sha256)
    if status == "exhausted_pending_prune":
        return _prune_exhausted(root, state, state_sha256)
    if status == "exhausted":
        return {
            "root": str(root),
            "state": state,
            "state_file_sha256": state_sha256,
            "spawned": False,
            "reason": "single_binary_objective_migration_exhausted",
        }
    return state, state_sha256


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def resume_attacker_binary_joint_objective_migration(
    run_suffix: str,
) -> dict[str, Any]:
    """Run or reconcile exactly one bounded attacker objective migration."""

    if not _SAFE_SUFFIX_RE.fullmatch(run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state, state_sha = _load_state_snapshot(root)
    if state.get("run_suffix") != run_suffix:
        raise RuntimeError("Migration run suffix differs from durable state")
    if (
        state.get("status") == "completed"
        and state.get(MIGRATION_HISTORY_KEY)
    ):
        return _audit_completed_population_with_migrations(
            root, state, state_sha
        )
    if MIGRATION_KEY not in state:
        state, state_sha = _initialize_migration(root, state, state_sha)

    while True:
        phase = _drain_phase(root, state, state_sha)
        if isinstance(phase, dict):
            return phase
        next_state, next_sha = phase
        if next_sha == state_sha:
            state, state_sha = next_state, next_sha
            break
        state, state_sha = next_state, next_sha

    migration = _verify_existing_migration(root, state)
    attempt = migration.get("attempt")
    if attempt is None:
        state, state_sha, attempt = _create_attempt(
            root, state, state_sha
        )
        migration = _verify_existing_migration(root, state)
    elif attempt.get("status") != "training":
        raise RuntimeError("Migration attempt is in an undrainable phase")

    run_dir, trainer_result = _run_attempt(migration, attempt)
    output_vol.reload()
    current, current_sha = _load_state_snapshot(root)
    if current_sha != state_sha or current != state:
        raise RuntimeError("State changed while migration trainer ran")
    current_migration = _verify_existing_migration(root, current)
    validation, source_checkpoint, artifact = _validate_attempt_output(
        current,
        current_migration,
        current_migration["attempt"],
        run_dir,
        trainer_result,
    )
    if validation.get("stopped_early") is True:
        state, state_sha = _prepare_successful_swap(
            root,
            current,
            current_sha,
            validation=validation,
            source_checkpoint=source_checkpoint,
            artifact=artifact,
        )
        state, state_sha = _reconcile_swap(root, state, state_sha)
        state, state_sha = _prune_promoted(root, state, state_sha)
        return _release_and_dispatch(root, state, state_sha)

    state, state_sha = _prepare_exhausted_archive(
        root,
        current,
        current_sha,
        validation=validation,
        source_checkpoint=source_checkpoint,
        artifact=artifact,
    )
    state, state_sha = _prune_exhausted(root, state, state_sha)
    return {
        "root": str(root),
        "state": state,
        "state_file_sha256": state_sha,
        "spawned": False,
        "reason": "single_binary_objective_migration_exhausted",
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def resume_role_lora_selfplay8_gate_retry_after_objective_migration(
    run_suffix: str,
) -> dict[str, Any]:
    """Resume a later raw retry with migration-aware D8 finalization."""

    if not _SAFE_SUFFIX_RE.fullmatch(run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    return _resume_gate_retry_with_migration_audit(run_suffix)


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def audit_and_finalize_role_lora_selfplay8_population_after_objective_migration(
    run_suffix: str,
) -> dict[str, Any]:
    """Attach/recheck the migration-aware live audit after normal D8."""

    if not _SAFE_SUFFIX_RE.fullmatch(run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state, state_sha = _load_state_snapshot(root)
    return _audit_completed_population_with_migrations(
        root, state, state_sha
    )


@app.local_entrypoint(name="resume_attacker_binary_joint_objective_migration")
def resume_attacker_binary_joint_objective_migration_local(
    run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    invoke = (
        resume_attacker_binary_joint_objective_migration.remote
        if wait_for_completion
        else resume_attacker_binary_joint_objective_migration.spawn
    )
    result = invoke(run_suffix=run_suffix)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"OBJECTIVE_MIGRATION_CALL_ID={result.object_id}", flush=True)


@app.local_entrypoint(
    name="resume_role_lora_selfplay8_gate_retry_after_objective_migration"
)
def resume_role_lora_selfplay8_gate_retry_after_objective_migration_local(
    run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    invoke = (
        resume_role_lora_selfplay8_gate_retry_after_objective_migration.remote
        if wait_for_completion
        else resume_role_lora_selfplay8_gate_retry_after_objective_migration.spawn
    )
    result = invoke(run_suffix=run_suffix)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"MIGRATION_AWARE_GATE_RETRY_CALL_ID={result.object_id}", flush=True)


@app.local_entrypoint(
    name="audit_and_finalize_role_lora_selfplay8_population_after_objective_migration"
)
def audit_and_finalize_role_lora_selfplay8_population_after_objective_migration_local(
    run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    invoke = (
        audit_and_finalize_role_lora_selfplay8_population_after_objective_migration.remote
        if wait_for_completion
        else audit_and_finalize_role_lora_selfplay8_population_after_objective_migration.spawn
    )
    result = invoke(run_suffix=run_suffix)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"MIGRATION_FINAL_AUDIT_CALL_ID={result.object_id}", flush=True)
