"""Tests for the additive one-shot attacker objective migration."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from role_lora_selfplay8 import build_selfplay8_schedule, population_labels
from roll.utils.selfplay_attacker_objective_migration import (
    MIGRATION_MANIFEST_POLICY,
    MIGRATION_FINAL_AUDIT_POLICY,
    MIGRATION_POLICY,
    MIGRATION_REQUIRED_METRIC,
    TRAJECTORY_SUM_LOSS_SCALE,
    binary_joint_objective_contract,
    build_binary_joint_migration_trainer_source,
    build_binary_joint_upstream_sources,
    build_migration_attempt_contract,
    build_migration_aware_final_audit_source,
    build_migration_aware_gate_retry_finalize_sources,
    build_migration_final_audit_contract,
    build_migration_history_entry,
    build_migration_plan,
    canonical_json_sha256,
    file_sha256,
    index_migration_history_for_final_audit,
    validate_binary_joint_successful_gate,
    validate_migration_eligibility,
    validate_migration_manifest,
    validate_migrated_stage_final_gate,
    verify_migration_attempt_contract,
    verify_migration_final_audit_contract,
    verify_migration_history,
    verify_migration_plan,
)
from roll.utils.selfplay_gate_retry import (
    OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS,
    RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
    RECOVERY_KEY,
    RECOVERY_HISTORY_KEY,
    bounded_raw_ppo_retry_policy,
    build_attempt_contract,
    build_objective_migration_requirement,
    build_ppo_only_recovery_trainer_source,
    build_recovery_plan,
    validate_final_population_state,
    validate_recovery_eligibility,
    validate_successful_gate,
    verify_recovery_history,
)


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modal_upstream_selfredteam_role_lora.py"
GATE_RETRY = ROOT / "modal_role_lora_selfplay8_gate_retry.py"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _terminal_state(base: str = "/output/run") -> dict:
    schedule = build_selfplay8_schedule(8)
    core_sha = hashlib.sha256(CORE.read_bytes()).hexdigest()
    frozen = {"modal_upstream_selfredteam_role_lora.py": core_sha}
    state = {
        "schema_version": 1,
        "run_suffix": "migration_test",
        "status": "stage_target_not_reached",
        "active_stage": "A2",
        "schedule": [stage.to_dict() for stage in schedule],
        "config": {
            "rounds": 8,
            "attacker_max_steps": 100,
            "defender_max_steps": 200,
            "save_steps": 10,
            "attacker_learning_rate": 1e-5,
            "defender_learning_rate": 1e-5,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 30,
            "defender_early_stop_min_steps": 32,
            "training_implementation_sha256": frozen,
        },
        "stages": {
            "A1": {
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": f"{base}/population/A1",
                "sha256": _sha("A1"),
            },
            "D1": {
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": f"{base}/population/D1",
                "sha256": _sha("D1"),
            },
            "A2": {
                **schedule[1].to_dict(),
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": f"{base}/population/A2",
                "sha256": _sha("A2-original-failed"),
                "spawn_claim_id": _sha("A2-claim"),
                "run_dir": f"{base}/A2",
                "requested_max_step": 100,
                "actual_final_step": 100,
                "stopped_early": False,
                "successor_release": {
                    "approved": False,
                    "basis": "target not reached",
                },
            },
        },
    }
    eligible = validate_recovery_eligibility(state, schedule)
    frozen_source = CORE.read_text(encoding="utf-8")
    _raw_source, raw_trainer = build_ppo_only_recovery_trainer_source(
        frozen_source
    )
    raw_plan = build_recovery_plan(
        state=state,
        eligibility=eligible,
        initial_state_file_sha256=_sha("initial-state"),
        frozen_training_sha256=frozen,
        recovery_implementation_sha256={"raw_retry.py": _sha("raw-retry")},
        recovery_trainer_contract=raw_trainer,
        plan_path=f"{base}/gate_retry_v1/A2/plan.json",
        original_failure_evidence={"passed": True, "actual_final_step": 100},
    )
    raw_contract = build_attempt_contract(
        raw_plan,
        attempt_number=1,
        trainable_init_checkpoint=f"{base}/population/A2",
        trainable_init_sha256=_sha("A2-original-failed"),
        trainer_run_suffix="migration_test_a2_raw_retry",
        contract_path=f"{base}/gate_retry_v1/A2/attempt_01/contract.json",
    )
    raw_attempt = {
        "schema_version": 1,
        "attempt_id": raw_contract["attempt_id"],
        "attempt_number": 1,
        "status": "gate_not_reached",
        "contract": raw_contract,
        "candidate_checkpoint": f"{base}/population_attempts/A2/raw_failed/A2",
        "candidate_sha256": _sha("A2-raw-retry-failed"),
        "pruning_complete": True,
        "gate_result": {
            "passed": True,
            "classification": "gate_not_reached_after_complete_budget",
        },
    }
    requirement = build_objective_migration_requirement(
        raw_plan, [raw_attempt]
    )
    state["status"] = OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS
    state["stages"]["A2"]["work_status"] = (
        "raw_ppo_retry_exhausted_objective_migration_required"
    )
    state[RECOVERY_KEY] = {
        "schema_version": 1,
        "status": RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
        "plan_id": raw_plan["plan_id"],
        "plan": raw_plan,
        "attempts": [raw_attempt],
        "active_attempt_id": None,
        "official_population_released": False,
        "objective_migration_requirement": requirement,
        "next_required_action": requirement["required_next_objective"],
    }
    return state


def _terminal_attacker_state(label: str, base: str) -> dict:
    if label == "A2":
        return _terminal_state(base)
    schedule = build_selfplay8_schedule(8)
    specs = {stage.label: stage.to_dict() for stage in schedule}
    position = next(
        index for index, stage in enumerate(schedule) if stage.label == label
    )
    core_sha = hashlib.sha256(CORE.read_bytes()).hexdigest()
    frozen = {"modal_upstream_selfredteam_role_lora.py": core_sha}
    stages = {
        "A1": {
            "status": "retained",
            "transition_state": "retained",
            "population_checkpoint": f"{base}/population/A1",
            "sha256": _sha("A1"),
        }
    }
    for stage in schedule[: position + 1]:
        is_target = stage.label == label
        stages[stage.label] = {
            **specs[stage.label],
            "status": "retained",
            "transition_state": "retained",
            "population_checkpoint": f"{base}/population/{stage.label}",
            "sha256": _sha(
                f"{stage.label}-original-failed"
                if is_target
                else f"retained-{stage.label}"
            ),
            "spawn_claim_id": _sha(f"{stage.label}-claim"),
            "run_dir": f"{base}/runs/{stage.label}",
            "requested_max_step": 100 if stage.role == "attacker" else 200,
            "actual_final_step": 100 if stage.role == "attacker" else 200,
            "stopped_early": not is_target,
            "successor_release": {
                "approved": not is_target,
                "basis": "target not reached" if is_target else "passed",
            },
        }
    state = {
        "schema_version": 1,
        "run_suffix": "migration_test",
        "status": "stage_target_not_reached",
        "active_stage": label,
        "schedule": [stage.to_dict() for stage in schedule],
        "config": {
            "rounds": 8,
            "attacker_max_steps": 100,
            "defender_max_steps": 200,
            "save_steps": 10,
            "attacker_learning_rate": 1e-5,
            "defender_learning_rate": 1e-5,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 30,
            "defender_early_stop_min_steps": 32,
            "training_implementation_sha256": frozen,
        },
        "stages": stages,
    }
    eligibility = validate_recovery_eligibility(state, schedule)
    frozen_source = CORE.read_text(encoding="utf-8")
    _raw_source, raw_trainer = build_ppo_only_recovery_trainer_source(
        frozen_source
    )
    raw_plan = build_recovery_plan(
        state=state,
        eligibility=eligibility,
        initial_state_file_sha256=_sha(f"{label}-initial-state"),
        frozen_training_sha256=frozen,
        recovery_implementation_sha256={
            "raw_retry.py": _sha(f"{label}-raw-retry")
        },
        recovery_trainer_contract=raw_trainer,
        plan_path=f"{base}/gate_retry_v1/{label}/plan.json",
        original_failure_evidence={"passed": True, "actual_final_step": 100},
    )
    raw_contract = build_attempt_contract(
        raw_plan,
        attempt_number=1,
        trainable_init_checkpoint=f"{base}/population/{label}",
        trainable_init_sha256=stages[label]["sha256"],
        trainer_run_suffix=f"migration_test_{label.lower()}_raw_retry",
        contract_path=f"{base}/gate_retry_v1/{label}/attempt_01/contract.json",
    )
    raw_attempt = {
        "schema_version": 1,
        "attempt_id": raw_contract["attempt_id"],
        "attempt_number": 1,
        "status": "gate_not_reached",
        "contract": raw_contract,
        "candidate_checkpoint": (
            f"{base}/population_attempts/{label}/raw_failed/{label}"
        ),
        "candidate_sha256": _sha(f"{label}-raw-retry-failed"),
        "pruning_complete": True,
        "gate_result": {
            "passed": True,
            "classification": "gate_not_reached_after_complete_budget",
        },
    }
    requirement = build_objective_migration_requirement(
        raw_plan, [raw_attempt]
    )
    state["status"] = OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS
    state["stages"][label]["work_status"] = (
        "raw_ppo_retry_exhausted_objective_migration_required"
    )
    state[RECOVERY_KEY] = {
        "schema_version": 1,
        "status": RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
        "plan_id": raw_plan["plan_id"],
        "plan": raw_plan,
        "attempts": [raw_attempt],
        "active_attempt_id": None,
        "official_population_released": False,
        "objective_migration_requirement": requirement,
        "next_required_action": requirement["required_next_objective"],
    }
    return state


def _migration_plan(state: dict) -> dict:
    core_source = CORE.read_text(encoding="utf-8")
    _source, trainer = build_binary_joint_migration_trainer_source(core_source)
    return build_migration_plan(
        state,
        migration_implementation_sha256={
            "migration.py": _sha("migration")
        },
        migration_trainer_contract=trainer,
        migration_final_audit_contract=build_migration_final_audit_contract(
            GATE_RETRY.read_text(encoding="utf-8")
        ),
        plan_path="/output/run/attacker_objective_migration_v1/A2/plan.json",
    )


def _with_cadence(
    validation: dict,
    *,
    final_step: int,
    final_sha: str,
    save_steps: int = 10,
) -> dict:
    steps = list(range(save_steps, final_step + 1, save_steps))
    if not steps or steps[-1] != final_step:
        steps.append(final_step)
    digests = {
        str(step): final_sha if step == final_step else _sha(f"step-{step}")
        for step in steps
    }
    validation.update(
        {
            "expected_step": final_step,
            "final_step": final_step,
            "checkpoint_sha256": dict(digests),
            "expected_checkpoint_steps": steps,
            "expected_checkpoint_count": len(steps),
            "observed_checkpoint_steps": steps,
            "observed_expected_checkpoint_count": len(steps),
            "missing_checkpoint_steps": [],
            "expected_checkpoint_sha256": dict(digests),
            "complete_cadence_required": False,
            "complete_cadence_verified": True,
            "changed_across_checkpoints": len(steps) > 1,
        }
    )
    return validation


def _successful_validation(final_step: int = 5) -> tuple[dict, str, str]:
    final_sha = _sha("migration-final")
    initial_sha = _sha("migration-initial")
    history = [
        {
            "step": step,
            "value": 0.96,
            "qualified": True,
            "metrics": {},
        }
        for step in range(final_step - 4, final_step + 1)
    ]
    validation = {
        "requested_max_step": 100,
        "actual_final_step": final_step,
        "final_checkpoint": f"/output/run/ckpt/global_step{final_step}_hf",
        "stopped_early": True,
        "early_stop": {
            "metric": MIGRATION_REQUIRED_METRIC,
            "threshold": 0.95,
            "patience": 5,
            "min_steps": 1,
            "triggered": True,
            "streak": 5,
            "last_step": final_step,
            "actual_final_step": final_step,
            "checkpoint_tag": f"global_step{final_step}",
            "companion_metrics": [],
            "companion_bounds": {},
            "history": history,
        },
    }
    return (
        _with_cadence(
            validation,
            final_step=final_step,
            final_sha=final_sha,
        ),
        final_sha,
        initial_sha,
    )


def _standard_successful_validation(
    *,
    role: str,
    budget: int,
    final_step: int,
    final_checkpoint: str,
    final_sha: str,
) -> dict:
    if role == "attacker":
        metric = "attacker/request_success_rate"
        min_steps = 30
        companion_bounds = {
            "attacker/cot_format_violation": {
                "direction": "max",
                "bound": 0.05,
            }
        }
        metrics = {"attacker/cot_format_violation": 0.0}
    else:
        metric = "defender/wildguard_actual_harmful_joint_success"
        min_steps = 32
        minimum_count = float(math.ceil(0.95 * 128 / 2))
        companion_bounds = {
            "defender/wildguard_actual_harmful_count": {
                "direction": "min",
                "bound": minimum_count,
            },
            "defender/wildguard_actual_benign_joint_success": {
                "direction": "min",
                "bound": 0.95,
            },
            "defender/wildguard_actual_benign_count": {
                "direction": "min",
                "bound": minimum_count,
            },
        }
        metrics = {
            "defender/wildguard_actual_harmful_count": minimum_count,
            "defender/wildguard_actual_benign_joint_success": 0.96,
            "defender/wildguard_actual_benign_count": minimum_count,
        }
    history = [
        {
            "step": step,
            "value": 0.96,
            "qualified": True,
            "metrics": dict(metrics),
        }
        for step in range(final_step - 4, final_step + 1)
    ]
    return _with_cadence(
        {
            "requested_max_step": budget,
            "actual_final_step": final_step,
            "final_checkpoint": final_checkpoint,
            "stopped_early": True,
            "early_stop": {
                "metric": metric,
                "threshold": 0.95,
                "patience": 5,
                "min_steps": min_steps,
                "triggered": True,
                "streak": 5,
                "last_step": final_step,
                "actual_final_step": final_step,
                "checkpoint_tag": f"global_step{final_step}",
                "companion_bounds": companion_bounds,
                "history": history,
            },
        },
        final_step=final_step,
        final_sha=final_sha,
    )


def _completed_state_with_a2_migration(base: Path) -> tuple[dict, dict[str, dict]]:
    base_text = str(base)
    terminal = _terminal_state(base_text)
    plan = _migration_plan(terminal)
    contract = build_migration_attempt_contract(
        plan,
        trainer_run_suffix="migration_final16_a2_binary_joint",
        contract_path=f"{base_text}/migration/A2/attempt_01/contract.json",
    )
    migration_validation, migration_sha, _initial_sha = _successful_validation()
    migration_run = base / "runs" / "A2"
    migration_source = migration_run / "ckpt" / "global_step5_hf"
    migration_validation["final_checkpoint"] = str(migration_source)
    migration_gate = validate_binary_joint_successful_gate(
        migration_validation,
        expected_budget=100,
        save_steps=10,
        expected_final_sha256=migration_sha,
        expected_initial_sha256=contract["trainable_init_sha256"],
    )
    canonical = base / "population" / "A2"
    archive = base / "population_attempts" / "A2" / "pre_migration" / "A2"
    journal = {
        "schema_version": 1,
        "attempt_id": contract["attempt_id"],
        "phase": "complete",
        "canonical": str(canonical),
        "staging": f"{base_text}/migration/A2/staging/A2",
        "archive": str(archive),
        "old_sha256": plan["displaced_nonqualifying_population"]["sha256"],
        "new_sha256": migration_sha,
        "atomic_primitive": "renameat2(RENAME_EXCHANGE)",
    }
    journal["journal_id"] = canonical_json_sha256(journal)
    attempt = {
        "schema_version": 1,
        "attempt_id": contract["attempt_id"],
        "attempt_number": 1,
        "status": "qualified_ready_to_release",
        "contract": contract,
        "contract_path": contract["contract_path"],
        "contract_file_sha256": _sha("contract-file"),
        "run_dir": str(migration_run),
        "source_checkpoint": str(migration_source),
        "source_sha256": migration_sha,
        "gate_result": migration_gate,
        "official_population_checkpoint": str(canonical),
        "official_population_sha256": migration_sha,
        "pruning_complete": True,
    }
    migration = {
        "schema_version": 1,
        "policy": MIGRATION_POLICY,
        "status": "released",
        "plan_id": plan["plan_id"],
        "plan": plan,
        "attempt": attempt,
        "swap_journal": journal,
        "official_population_released": True,
    }
    entry = build_migration_history_entry(
        migration,
        terminal[RECOVERY_KEY],
        archived_state_file_sha256=_sha("migration-final16-pre-release"),
    )

    schedule = build_selfplay8_schedule(8)
    specs = {stage.label: stage.to_dict() for stage in schedule}
    validations: dict[str, dict] = {}
    stages: dict[str, dict] = {}
    labels = population_labels(8)
    for label in labels:
        population = base / "population" / label
        population.mkdir(parents=True, exist_ok=True)
        if label == "A1":
            digest = _sha("A1")
            stages[label] = {
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": str(population),
                "sha256": digest,
            }
            (population / "digest.txt").write_text(digest, encoding="utf-8")
            continue
        role = "attacker" if label.startswith("A") else "defender"
        budget = 100 if role == "attacker" else 200
        final_step = 5 if label == "A2" else 36
        digest = migration_sha if label == "A2" else _sha(f"final-{label}")
        run_dir = migration_run if label == "A2" else base / "runs" / label
        source = run_dir / "ckpt" / f"global_step{final_step}_hf"
        if label == "A2":
            validation = migration_validation
        else:
            validation = _standard_successful_validation(
                role=role,
                budget=budget,
                final_step=final_step,
                final_checkpoint=str(source),
                final_sha=digest,
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoint_validation.json").write_text(
            json.dumps(validation, sort_keys=True),
            encoding="utf-8",
        )
        validations[label] = validation
        stage = {
            **specs[label],
            "status": "retained",
            "transition_state": "retained",
            "population_checkpoint": str(population),
            "sha256": digest,
            "run_dir": str(run_dir),
            "source_checkpoint": str(source),
            "source_sha256": digest,
            "requested_max_step": budget,
            "actual_final_step": final_step,
            "stopped_early": True,
            "successor_release": {"approved": True},
            "spawn_claim_id": _sha(f"claim-{label}"),
        }
        if label == "A2":
            stage.update(
                {
                    "work_status": (
                        "retained_after_attacker_objective_migration"
                    ),
                    "attacker_objective_migration_plan_id": plan["plan_id"],
                    "attacker_objective_migration_attempt_id": contract[
                        "attempt_id"
                    ],
                    "attacker_objective_migration_gate": migration_gate,
                    "optimization_objective": binary_joint_objective_contract(),
                    "official_payoff_evaluation": {
                        "utility": "frozen_upstream_additive_raw_utility",
                        "normalization": "none",
                        "binary_surrogate_is_payoff_entry": False,
                    },
                    "displaced_nonqualifying_population": {
                        "checkpoint": str(archive),
                        "sha256": journal["old_sha256"],
                        "strict_audit": {"passed": True},
                    },
                    "successor_release": {
                        "approved": True,
                        "migration_plan_id": plan["plan_id"],
                        "migration_attempt_id": contract["attempt_id"],
                        "gate_result": migration_gate,
                        "official_payoff_utility_unchanged": True,
                        "binary_surrogate_is_payoff_entry": False,
                        "displaced_nonqualifying_population_preserved": True,
                    },
                }
            )
        stages[label] = stage
        (population / "digest.txt").write_text(digest, encoding="utf-8")
    final_state = copy.deepcopy(terminal)
    final_state["status"] = "completed"
    final_state["active_stage"] = None
    final_state["completed_population"] = labels
    final_state["stages"] = stages
    final_state["attacker_objective_migration_history_v1"] = [entry]
    del final_state[RECOVERY_KEY]
    return final_state, validations


def _precomplete_d8_retry_state(completed: dict, base: Path) -> dict:
    state = copy.deepcopy(completed)
    state.pop("completed_population", None)
    state.pop("final_population_audit", None)
    state["status"] = "stage_target_not_reached"
    state["active_stage"] = "D8"
    d8 = state["stages"]["D8"]
    d8["stopped_early"] = False
    d8["actual_final_step"] = 200
    d8["requested_max_step"] = 200
    d8["successor_release"] = {"approved": False}
    schedule = build_selfplay8_schedule(8)
    eligibility = validate_recovery_eligibility(state, schedule)
    frozen_source = CORE.read_text(encoding="utf-8")
    _raw_source, raw_trainer = build_ppo_only_recovery_trainer_source(
        frozen_source
    )
    plan = build_recovery_plan(
        state=state,
        eligibility=eligibility,
        initial_state_file_sha256=_sha("d8-precomplete-state"),
        frozen_training_sha256=state["config"][
            "training_implementation_sha256"
        ],
        recovery_implementation_sha256={
            "raw_retry.py": _sha("raw-d8-retry")
        },
        recovery_trainer_contract=raw_trainer,
        plan_path=f"{base}/gate_retry_v1/D8/plan.json",
        original_failure_evidence={"passed": True},
    )
    attempt_id = _sha("d8-qualified-retry")
    state[RECOVERY_KEY] = {
        "schema_version": 1,
        "status": "released",
        "plan_id": plan["plan_id"],
        "plan": plan,
        "attempts": [{"attempt_id": attempt_id}],
        "official_population_released": True,
    }
    state["status"] = "running"
    state["active_stage"] = "D8"
    d8["stopped_early"] = True
    d8["actual_final_step"] = 36
    d8["successor_release"] = {
        "approved": True,
        "plan_id": plan["plan_id"],
        "attempt_id": attempt_id,
    }
    return state


def _add_released_migration_to_final_state(
    final_state: dict,
    *,
    label: str,
    terminal: dict,
    base: Path,
) -> dict:
    plan = _migration_plan(terminal)
    contract = build_migration_attempt_contract(
        plan,
        trainer_run_suffix=f"migration_final16_{label.lower()}_binary_joint",
        contract_path=f"{base}/migration/{label}/attempt_01/contract.json",
    )
    validation, final_sha, _initial_sha = _successful_validation()
    run_dir = base / "runs" / label
    source = run_dir / "ckpt" / "global_step5_hf"
    validation["final_checkpoint"] = str(source)
    gate = validate_binary_joint_successful_gate(
        validation,
        expected_budget=100,
        save_steps=10,
        expected_final_sha256=final_sha,
        expected_initial_sha256=contract["trainable_init_sha256"],
    )
    canonical = base / "population" / label
    archive = base / "population_attempts" / label / "pre_migration" / label
    journal = {
        "schema_version": 1,
        "attempt_id": contract["attempt_id"],
        "phase": "complete",
        "canonical": str(canonical),
        "staging": f"{base}/migration/{label}/staging/{label}",
        "archive": str(archive),
        "old_sha256": plan["displaced_nonqualifying_population"]["sha256"],
        "new_sha256": final_sha,
        "atomic_primitive": "renameat2(RENAME_EXCHANGE)",
    }
    journal["journal_id"] = canonical_json_sha256(journal)
    attempt = {
        "schema_version": 1,
        "attempt_id": contract["attempt_id"],
        "attempt_number": 1,
        "status": "qualified_ready_to_release",
        "contract": contract,
        "contract_path": contract["contract_path"],
        "contract_file_sha256": _sha(f"{label}-contract-file"),
        "run_dir": str(run_dir),
        "source_checkpoint": str(source),
        "source_sha256": final_sha,
        "gate_result": gate,
        "official_population_checkpoint": str(canonical),
        "official_population_sha256": final_sha,
        "pruning_complete": True,
    }
    migration = {
        "schema_version": 1,
        "policy": MIGRATION_POLICY,
        "status": "released",
        "plan_id": plan["plan_id"],
        "plan": plan,
        "attempt": attempt,
        "swap_journal": journal,
        "official_population_released": True,
    }
    entry = build_migration_history_entry(
        migration,
        terminal[RECOVERY_KEY],
        archived_state_file_sha256=_sha(f"{label}-pre-release-state"),
    )
    updated = copy.deepcopy(final_state)
    history = list(updated.get("attacker_objective_migration_history_v1") or [])
    history.append(entry)
    updated["attacker_objective_migration_history_v1"] = history
    stage = updated["stages"][label]
    stage.update(
        {
            "run_dir": str(run_dir),
            "source_checkpoint": str(source),
            "source_sha256": final_sha,
            "population_checkpoint": str(canonical),
            "sha256": final_sha,
            "requested_max_step": 100,
            "actual_final_step": 5,
            "stopped_early": True,
            "attacker_objective_migration_plan_id": plan["plan_id"],
            "attacker_objective_migration_attempt_id": contract["attempt_id"],
            "attacker_objective_migration_gate": gate,
            "optimization_objective": binary_joint_objective_contract(),
            "official_payoff_evaluation": {
                "utility": "frozen_upstream_additive_raw_utility",
                "normalization": "none",
                "binary_surrogate_is_payoff_entry": False,
            },
            "displaced_nonqualifying_population": {
                "checkpoint": str(archive),
                "sha256": journal["old_sha256"],
                "strict_audit": {"passed": True},
            },
            "successor_release": {
                "approved": True,
                "migration_plan_id": plan["plan_id"],
                "migration_attempt_id": contract["attempt_id"],
                "gate_result": gate,
                "official_payoff_utility_unchanged": True,
                "binary_surrogate_is_payoff_entry": False,
                "displaced_nonqualifying_population_preserved": True,
            },
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint_validation.json").write_text(
        json.dumps(validation, sort_keys=True), encoding="utf-8"
    )
    (canonical / "digest.txt").write_text(final_sha, encoding="utf-8")
    return updated


class ObjectiveContractTests(unittest.TestCase):
    def test_binary_goal_cot_contract_is_exact_and_not_a_payoff_entry(self):
        contract = binary_joint_objective_contract()
        self.assertEqual(
            contract["positive_iff"]["outcome_any"],
            [
                "ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST",
                "ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST",
            ],
        )
        self.assertEqual(contract["positive_iff"]["outcome_forbidden"], "TIE")
        self.assertFalse(contract["positive_iff"]["cot_format_violation"])
        self.assertEqual(
            contract["optimization_reward"],
            {"positive": 1.0, "negative": -1.0},
        )
        self.assertEqual(
            contract["failure_and_tie_replay"],
            "retained_with_negative_reward",
        )
        self.assertEqual(contract["advantage"], "raw_no_center_no_std_no_baseline")
        self.assertEqual(
            contract["trajectory_sum_loss_scale"],
            TRAJECTORY_SUM_LOSS_SCALE,
        )
        self.assertEqual(contract["gate"]["min_steps"], 1)
        self.assertEqual(contract["gate"]["patience"], 5)
        self.assertFalse(contract["official_payoff"]["binary_surrogate_is_payoff_entry"])

    def test_dynamic_trainer_clone_is_narrow_ppo_only_attacker_recipe(self):
        frozen = CORE.read_text(encoding="utf-8")
        effective, descriptor = build_binary_joint_migration_trainer_source(
            frozen
        )
        self.assertIn(
            "def _effective_attacker_binary_joint_migration_train(",
            effective,
        )
        self.assertIn("attacker_binary_joint_objective", effective)
        self.assertIn("attacker_raw_reinforce_advantages", effective)
        self.assertIn("attacker_episode_sum_policy_loss", effective)
        self.assertIn("early_stop_min_steps == 1", effective)
        self.assertIn("early_stop_patience == 5", effective)
        self.assertIn("command.index('--normalize_reward')", effective)
        self.assertIn("['--gamma', '1.0']", effective)
        self.assertEqual(
            descriptor["recipe"]["objective"],
            binary_joint_objective_contract(),
        )
        self.assertEqual(
            hashlib.sha256(CORE.read_bytes()).hexdigest(),
            descriptor["frozen_core_source_sha256"],
        )

    def test_upstream_patch_retains_ties_uses_raw_advantage_and_fixed_sum(self):
        game = """            # Skip if wildguard cannot parse the response
            if labels.get('is_parsing_error', False):
                continue

            invalid_attack_rewrite = False
                if invalid_attack_rewrite:
                    reward, outcome = 0.0, {GameOutcome.TIE}
                else:
                # Update turn_states with computed values
                turn_states['reward'] = reward
"""
        experience = """            'raw_upstream_additive_reward',
            'defender_joint_success', 'defender_joint_signed_reward',
"""
        replay = """        preserve_joint_signed_defender_failures = bool(
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
        attacker_rewards = [item.info['reward'] for item in attacker_items]
        avg_attacker_reward = sum(attacker_rewards) / max(len(attacker_rewards), 1)
        successful_rate = (n_successful_benign_request + n_successful_harmful_request) / max(n_generated_attacks, 1)
        if is_cot_enabled:
            attacker_cot_violations = sum(item.info['cot_format_violation'] for item in attacker_items)
            attacker_cot_rate = attacker_cot_violations / max(n_generated_attacks, 1)
            'attacker/avg_reward': strategy.all_reduce(avg_attacker_reward, "mean"),
            'attacker/request_success_rate': strategy.all_reduce(successful_rate, "mean"), # successful = benign and harmful both leads to harmful reaction
"""
        actor = """    raw_defender = bool(
        args.custom_configs.get(
            "defender_raw_reinforce_advantages", False
        )
    )
    if not raw_defender:
        return "normalize"
                    if advantage_transform_mode in (
                        'raw_defender_reinforce',
                        'joint_signed_defender_reinforce',
                    ):
                        joint_signed_mode = (
                            advantage_transform_mode
                            == 'joint_signed_defender_reinforce'
                        )
                        raw_reward_snapshot = []
                            if joint_signed_mode and reward_value not in (
                                -1.0, 1.0
                            ):
                                raise RuntimeError(
                                    "Official defender joint-signed reward "
                                    f"must be +/-1, got {reward_value}"
                                )
                        status[
                            "debug/defender_raw_reinforce_advantages"
                        ] = 1.0
                        status[
                            "debug/defender_advantage_mean_centering_applied"
                        ] = 0.0
                        status[
                            "debug/defender_advantage_std_norm_applied"
                        ] = 0.0
                        status[
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
                    if advantage_transform_mode not in (
                        'raw_defender_reinforce',
                        'joint_signed_defender_reinforce',
                    ):
        if self.args.custom_configs.get(
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
"""
        patched, descriptor = build_binary_joint_upstream_sources(
            language_game_source=game,
            experience_maker_source=experience,
            replay_buffer_source=replay,
            actor_source=actor,
        )
        joined = "\n".join(patched.values())
        self.assertIn("ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST", joined)
        self.assertIn("ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST", joined)
        self.assertIn("GameOutcome.TIE not in outcome", joined)
        self.assertIn("binary_attacker_parse_failure", joined)
        self.assertIn("attacker_official_raw_utility", joined)
        self.assertIn("official_raw_utility_diagnostic_mean", joined)
        self.assertIn("binary_joint_optimization_reward_mean", joined)
        self.assertIn("preserve_binary_joint_attacker_failures", joined)
        self.assertIn("binary_joint_attacker_reinforce", joined)
        self.assertIn("attacker_episode_sum_loss_scale", joined)
        self.assertEqual(
            descriptor["objective"], binary_joint_objective_contract()
        )


class IdentityAndEligibilityTests(unittest.TestCase):
    def test_only_exact_hash_bound_attacker_terminal_is_eligible(self):
        state = _terminal_state()
        eligible = validate_migration_eligibility(state)
        self.assertEqual(eligible["label"], "A2")
        self.assertEqual(
            eligible["requirement"]["trainable_init_sha256"],
            _sha("A2-raw-retry-failed"),
        )
        self.assertEqual(
            eligible["requirement"]["fixed_opponent"]["sha256"],
            _sha("D1"),
        )

        wrong_status = copy.deepcopy(state)
        wrong_status["status"] = "stage_target_not_reached"
        with self.assertRaisesRegex(RuntimeError, "exact terminal state"):
            validate_migration_eligibility(wrong_status)

        wrong_requirement = copy.deepcopy(state)
        wrong_requirement[RECOVERY_KEY]["objective_migration_requirement"][
            "trainable_init_sha256"
        ] = _sha("forged")
        with self.assertRaisesRegex(RuntimeError, "handoff drifted"):
            validate_migration_eligibility(wrong_requirement)

        defender = copy.deepcopy(state)
        defender["active_stage"] = "D2"
        with self.assertRaisesRegex(RuntimeError, "not A2--A8"):
            validate_migration_eligibility(defender)

    def test_plan_and_single_attempt_bind_last_failure_and_fixed_defender(self):
        state = _terminal_state()
        plan = _migration_plan(state)
        self.assertEqual(verify_migration_plan(plan), plan["plan_id"])
        self.assertEqual(
            plan["trainable_init"]["sha256"], _sha("A2-raw-retry-failed")
        )
        self.assertEqual(plan["fixed_opponent"]["sha256"], _sha("D1"))
        self.assertEqual(
            plan["attempt_policy"]["max_attempts_per_stage"], 1
        )
        contract = build_migration_attempt_contract(
            plan,
            trainer_run_suffix="migration_test_a2_binary_joint",
            contract_path="/output/run/migration/A2/attempt_01/contract.json",
        )
        self.assertEqual(
            verify_migration_attempt_contract(contract, plan),
            contract["attempt_id"],
        )
        self.assertEqual(
            contract["trainable_init_sha256"], _sha("A2-raw-retry-failed")
        )
        self.assertEqual(contract["fixed_opponent"]["sha256"], _sha("D1"))

        forged = copy.deepcopy(contract)
        forged["fixed_opponent"]["sha256"] = _sha("different-D")
        forged["attempt_id"] = canonical_json_sha256(
            {key: value for key, value in forged.items() if key != "attempt_id"}
        )
        with self.assertRaisesRegex(RuntimeError, "drifted at fixed_opponent"):
            verify_migration_attempt_contract(forged, plan)


class GateAndManifestTests(unittest.TestCase):
    def test_exact_g_gate_can_stop_at_step_five(self):
        validation, final_sha, initial_sha = _successful_validation()
        proof = validate_binary_joint_successful_gate(
            validation,
            expected_budget=100,
            save_steps=10,
            expected_final_sha256=final_sha,
            expected_initial_sha256=initial_sha,
        )
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["tail_steps"], [1, 2, 3, 4, 5])
        self.assertTrue(
            proof["checkpoint_cadence"]["single_checkpoint_change_proof"]
        )

        old_metric = copy.deepcopy(validation)
        old_metric["early_stop"]["metric"] = "attacker/request_success_rate"
        with self.assertRaisesRegex(RuntimeError, "metric drifted"):
            validate_binary_joint_successful_gate(
                old_metric,
                expected_budget=100,
                save_steps=10,
                expected_final_sha256=final_sha,
                expected_initial_sha256=initial_sha,
            )

        foreign_companion = copy.deepcopy(validation)
        foreign_companion["early_stop"]["companion_bounds"] = {
            "attacker/cot_format_violation": {
                "direction": "max",
                "bound": 0.05,
            }
        }
        with self.assertRaisesRegex(RuntimeError, "foreign companions"):
            validate_binary_joint_successful_gate(
                foreign_companion,
                expected_budget=100,
                save_steps=10,
                expected_final_sha256=final_sha,
                expected_initial_sha256=initial_sha,
            )

    def test_manifest_keeps_surrogate_out_of_official_payoff_namespace(self):
        state = _terminal_state()
        plan = _migration_plan(state)
        contract = build_migration_attempt_contract(
            plan,
            trainer_run_suffix="migration_test_a2_binary_joint",
            contract_path="/output/run/migration/A2/attempt_01/contract.json",
        )
        migration_manifest = {
            "schema_version": 1,
            "policy": MIGRATION_MANIFEST_POLICY,
            "plan_id": plan["plan_id"],
            "attempt_id": contract["attempt_id"],
            "stage_label": "A2",
            "role": "attacker",
            "trainer_run_suffix": contract["trainer_run_suffix"],
            "frozen_training_implementation_sha256": plan[
                "frozen_training_implementation_sha256"
            ],
            "migration_implementation_sha256": plan[
                "migration_implementation_sha256"
            ],
            "effective_trainer_contract": plan[
                "migration_trainer_contract"
            ],
            "single_bounded_attempt": {
                "attempt_number": 1,
                "maximum_attempts": 1,
                "budget": 100,
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
            "objective": binary_joint_objective_contract(),
            "official_payoff_evaluation": {
                "utility": "frozen_upstream_additive_raw_utility",
                "normalization": "none",
                "matrix_namespace": "historical_psro_payoff_unchanged",
                "binary_surrogate_is_payoff_entry": False,
            },
            "runtime_adapter_mapping": {
                "trainable": {
                    "original_checkpoint": contract[
                        "trainable_init_checkpoint"
                    ],
                    "original_sha256": contract[
                        "trainable_init_sha256"
                    ],
                    "runtime_weight_sha256": contract[
                        "trainable_init_sha256"
                    ],
                },
                "fixed_opponent": {
                    "original_checkpoint": contract["fixed_opponent"][
                        "checkpoint"
                    ],
                    "original_sha256": contract["fixed_opponent"][
                        "sha256"
                    ],
                    "runtime_weight_sha256": contract["fixed_opponent"][
                        "sha256"
                    ],
                },
            },
        }
        upstream = {
            "schema_version": 1,
            "policy": plan["migration_trainer_contract"]["policy"],
            "objective": binary_joint_objective_contract(),
            "input_source_sha256": {"input.py": _sha("input")},
            "patched_source_sha256": {"patched.py": _sha("patched")},
        }
        upstream["patch_descriptor_sha256"] = canonical_json_sha256(
            upstream
        )
        migration_manifest["patched_upstream_contract"] = upstream
        migration_manifest["migration_manifest_sha256"] = canonical_json_sha256(
            migration_manifest
        )
        path = "/output/run/attacker_objective_migration_manifest.json"
        trainer_manifest = {
            "attacker_objective_migration": {
                "policy": MIGRATION_MANIFEST_POLICY,
                "path": path,
                "sha256": migration_manifest["migration_manifest_sha256"],
                "frozen_core_implementation_sha256": plan[
                    "frozen_training_implementation_sha256"
                ]["modal_upstream_selfredteam_role_lora.py"],
                "effective_function_source_sha256": plan[
                    "migration_trainer_contract"
                ]["effective_function_source_sha256"],
                "patch_descriptor_sha256": plan[
                    "migration_trainer_contract"
                ]["patch_descriptor_sha256"],
                "patched_upstream_descriptor_sha256": upstream[
                    "patch_descriptor_sha256"
                ],
                "official_payoff_utility_unchanged": True,
            }
        }
        receipt = validate_migration_manifest(
            migration_manifest,
            trainer_manifest,
            plan=plan,
            contract=contract,
            migration_manifest_path=path,
        )
        self.assertTrue(receipt["official_payoff_utility_unchanged"])
        mixed = copy.deepcopy(migration_manifest)
        mixed["official_payoff_evaluation"][
            "binary_surrogate_is_payoff_entry"
        ] = True
        mixed["migration_manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in mixed.items() if key != "migration_manifest_sha256"}
        )
        trainer_manifest["attacker_objective_migration"]["sha256"] = mixed[
            "migration_manifest_sha256"
        ]
        with self.assertRaisesRegex(RuntimeError, "leaked"):
            validate_migration_manifest(
                mixed,
                trainer_manifest,
                plan=plan,
                contract=contract,
                migration_manifest_path=path,
            )

    def test_released_history_seals_raw_failure_and_migration(self):
        state = _terminal_state()
        plan = _migration_plan(state)
        contract = build_migration_attempt_contract(
            plan,
            trainer_run_suffix="migration_test_a2_binary_joint",
            contract_path="/output/run/migration/A2/attempt_01/contract.json",
        )
        migration = {
            "schema_version": 1,
            "policy": MIGRATION_POLICY,
            "status": "released",
            "plan_id": plan["plan_id"],
            "plan": plan,
            "attempt": {
                "attempt_id": contract["attempt_id"],
                "contract": contract,
                "status": "qualified_ready_to_release",
                "gate_result": {
                    "passed": True,
                    "policy": MIGRATION_POLICY,
                    "metric": MIGRATION_REQUIRED_METRIC,
                    "threshold": 0.95,
                    "patience": 5,
                    "min_steps": 1,
                    "optimization_surrogate_only": True,
                    "official_payoff_utility_unchanged": True,
                },
            },
            "swap_journal": {"phase": "complete"},
            "official_population_released": True,
        }
        entry = build_migration_history_entry(
            migration,
            state[RECOVERY_KEY],
            archived_state_file_sha256=_sha("pre-release-state"),
        )
        self.assertEqual(verify_migration_history([entry]), [plan["plan_id"]])
        tampered = copy.deepcopy(entry)
        tampered["raw_gate_retry_recovery"]["status"] = "active"
        with self.assertRaisesRegex(RuntimeError, "history row drifted"):
            verify_migration_history([tampered])


class FinalPopulationMigrationAuditTests(unittest.TestCase):
    def test_final_audit_and_d8_finalize_clones_are_hash_bound(self):
        source = GATE_RETRY.read_text(encoding="utf-8")
        effective, audit_descriptor = build_migration_aware_final_audit_source(
            source
        )
        finalize_sources, finalize_descriptor = (
            build_migration_aware_gate_retry_finalize_sources(source)
        )
        contract = build_migration_final_audit_contract(source)
        self.assertEqual(
            verify_migration_final_audit_contract(
                contract,
                expected_gate_retry_source_sha256=hashlib.sha256(
                    GATE_RETRY.read_bytes()
                ).hexdigest(),
            ),
            contract["contract_sha256"],
        )
        self.assertIn("migration_state=state", effective)
        self.assertIn("stage_label=label", effective)
        self.assertEqual(
            audit_descriptor,
            contract["live_final_audit_clone"],
        )
        self.assertEqual(
            finalize_descriptor,
            contract["d8_gate_retry_finalize_clones"],
        )
        self.assertEqual(
            set(finalize_sources),
            {
                "_release_or_complete",
                "resume_role_lora_selfplay8_gate_retry",
            },
        )

    def test_completed_chain_audits_all_16_and_routes_only_migrated_a2_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, _validations = _completed_state_with_a2_migration(root)
            indexed = index_migration_history_for_final_audit(state)
            self.assertEqual(list(indexed), ["A2"])
            source = GATE_RETRY.read_text(encoding="utf-8")
            effective, _descriptor = build_migration_aware_final_audit_source(
                source
            )
            migrated_calls: list[str] = []
            live_calls: list[str] = []

            def route_gate(
                validation: dict,
                *,
                migration_state: dict,
                stage_label: str,
                **kwargs: Any,
            ) -> dict:
                self.assertIs(migration_state, state)
                if stage_label in indexed:
                    migrated_calls.append(stage_label)
                    return validate_migrated_stage_final_gate(
                        state,
                        stage_label=stage_label,
                        validation=validation,
                        expected_budget=int(kwargs["expected_budget"]),
                        save_steps=int(kwargs["save_steps"]),
                        expected_final_sha256=str(
                            kwargs["expected_final_sha256"]
                        ),
                    )
                live_calls.append(stage_label)
                return validate_successful_gate(validation, **kwargs)

            def checkpoint_digest(checkpoint: Path) -> str:
                return (checkpoint / "digest.txt").read_text(encoding="utf-8")

            def strict_checkpoint(
                checkpoint: Path,
                *,
                expected_sha256: str,
            ) -> dict:
                self.assertEqual(checkpoint_digest(checkpoint), expected_sha256)
                return {
                    "tensor_count": 8,
                    "rank": 64,
                    "alpha": 64,
                }

            namespace = {
                "Any": Any,
                "Path": Path,
                "RECOVERY_HISTORY_KEY": RECOVERY_HISTORY_KEY,
                "RECOVERY_KEY": RECOVERY_KEY,
                "_audit_released_recovery_artifacts": (
                    lambda _state, _recovery: self.fail(
                        "No ordinary recovery history was expected"
                    )
                ),
                "_read_json_object": lambda path: json.loads(
                    path.read_text(encoding="utf-8")
                ),
                "_recovery_implementation_hashes": lambda: {
                    "gate_retry": _sha("gate-retry")
                },
                "_released_retry_plan_for_stage": (
                    lambda _state, _label: None
                ),
                "_strict_checkpoint": strict_checkpoint,
                "canonical_json_sha256": canonical_json_sha256,
                "checkpoint_weight_digest": checkpoint_digest,
                "file_sha256": file_sha256,
                "population_labels": population_labels,
                "re": re,
                "validate_final_population_state": (
                    validate_final_population_state
                ),
                "validate_migration_or_live_gate": route_gate,
                "verify_recovery_history": verify_recovery_history,
            }
            exec(
                compile(
                    effective,
                    "<test-migration-aware-final-audit>",
                    "exec",
                ),
                namespace,
            )
            audit = namespace[
                "_effective_migration_aware_final_population_audit"
            ](root, state)
            self.assertEqual(audit["observed_checkpoint_count"], 16)
            self.assertEqual(len(audit["members"]), 16)
            self.assertEqual(set(audit["gate_proofs"]), set(population_labels(8)) - {"A1"})
            self.assertEqual(migrated_calls, ["A2"])
            self.assertEqual(len(live_calls), 14)
            self.assertEqual(
                audit["gate_proofs"]["A2"]["metric"],
                MIGRATION_REQUIRED_METRIC,
            )
            self.assertEqual(audit["gate_proofs"]["A2"]["min_steps"], 1)
            self.assertEqual(
                audit["gate_proofs"]["A2"]["companion_bounds"], {}
            )
            self.assertEqual(
                audit["gate_proofs"]["A3"]["metric"],
                "attacker/request_success_rate",
            )
            tampered = copy.deepcopy(state)
            tampered["stages"]["A2"][
                "attacker_objective_migration_plan_id"
            ] = _sha("forged-plan")
            with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                index_migration_history_for_final_audit(tampered)

    def test_released_d8_retry_clone_audits_before_marking_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed, _validations = _completed_state_with_a2_migration(root)
            state = _precomplete_d8_retry_state(completed, root)
            self.assertEqual(
                list(index_migration_history_for_final_audit(state)),
                ["A2"],
            )
            source = GATE_RETRY.read_text(encoding="utf-8")
            sources, _descriptor = (
                build_migration_aware_gate_retry_finalize_sources(source)
            )
            audit_sha = _sha("precomplete-d8-final-audit")
            audited: list[bool] = []
            persisted: list[dict] = []

            class Volume:
                @staticmethod
                def reload() -> None:
                    return None

                @staticmethod
                def commit() -> None:
                    return None

            def build_audit(_root: Path, candidate: dict) -> dict:
                self.assertEqual(
                    list(index_migration_history_for_final_audit(candidate)),
                    ["A2"],
                )
                audited.append(True)
                return {"passed": True, "audit_sha256": audit_sha}

            def persist(
                _root: Path,
                candidate: dict,
                *,
                expected_file_sha256: str,
            ) -> str:
                self.assertEqual(expected_file_sha256, _sha("state"))
                persisted.append(copy.deepcopy(candidate))
                return _sha("completed-state")

            namespace = {
                "Any": Any,
                "Path": Path,
                "RECOVERY_KEY": RECOVERY_KEY,
                "_RECOVERY_ROOT_NAME": "gate_retry_v1",
                "_build_final_population_audit": build_audit,
                "_load_state_snapshot": lambda _root: (
                    state,
                    _sha("state"),
                ),
                "_persist_state_cas": persist,
                "_verify_existing_recovery": (
                    lambda _root, candidate: candidate[RECOVERY_KEY]
                ),
                "_write_exact_json": lambda _path, _value: None,
                "build_selfplay8_schedule": build_selfplay8_schedule,
                "copy": copy,
                "output_vol": Volume(),
                "population_labels": population_labels,
            }
            exec(
                compile(
                    sources["_release_or_complete"],
                    "<test-migration-aware-d8-release>",
                    "exec",
                ),
                namespace,
            )
            result = namespace[
                "_effective_migration_aware_release_or_complete"
            ](root, state, _sha("state"))
            self.assertEqual(audited, [True])
            self.assertEqual(result["state"]["status"], "completed")
            self.assertIsNone(result["state"]["active_stage"])
            self.assertEqual(len(result["state"]["completed_population"]), 16)
            self.assertEqual(
                result["state"]["final_population_audit"]["sha256"],
                audit_sha,
            )
            self.assertEqual(persisted[0][RECOVERY_KEY]["status"], "completed")

    def test_multiple_migrated_attackers_are_resolved_per_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed, _validations = _completed_state_with_a2_migration(root)
            a3_terminal = _terminal_attacker_state("A3", str(root))
            completed = _add_released_migration_to_final_state(
                completed,
                label="A3",
                terminal=a3_terminal,
                base=root,
            )
            indexed = index_migration_history_for_final_audit(completed)
            self.assertEqual(list(indexed), ["A2", "A3"])
            for label in indexed:
                stage = completed["stages"][label]
                validation = json.loads(
                    (Path(stage["run_dir"]) / "checkpoint_validation.json")
                    .read_text(encoding="utf-8")
                )
                proof = validate_migrated_stage_final_gate(
                    completed,
                    stage_label=label,
                    validation=validation,
                    expected_budget=100,
                    save_steps=10,
                    expected_final_sha256=stage["source_sha256"],
                )
                self.assertEqual(proof["migration_plan_id"], indexed[label]["plan_id"])
                self.assertEqual(proof["metric"], MIGRATION_REQUIRED_METRIC)
            reordered = copy.deepcopy(completed)
            reordered["attacker_objective_migration_history_v1"].reverse()
            with self.assertRaisesRegex(RuntimeError, "out of order"):
                index_migration_history_for_final_audit(reordered)


class EntrypointStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / (
            "modal_role_lora_selfplay8_attacker_objective_migration.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_entrypoint_is_additive_and_does_not_launch_or_stop_on_import(self):
        top_level_calls = [
            node
            for statement in self.tree.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ]
        forbidden = {"subprocess.run", "subprocess.Popen", "os.system"}

        def call_name(call: ast.Call) -> str:
            parts = []
            value = call.func
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            return ".".join(reversed(parts))

        # Function bodies are descendants of top-level definitions; only
        # inspect direct expression statements executed during import.
        direct_calls = [
            statement.value
            for statement in self.tree.body
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
        ]
        self.assertFalse(forbidden.intersection(map(call_name, direct_calls)))
        self.assertNotIn("modal app run", self.source)
        self.assertNotIn("modal app stop", self.source)
        self.assertIn("train_attacker_binary_joint_objective_migration.remote", self.source)

    def test_remote_trainer_keeps_fixed_defender_and_sft_off(self):
        functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        trainer = functions["train_attacker_binary_joint_objective_migration"]
        calls = [node for node in ast.walk(trainer) if isinstance(node, ast.Call)]
        effective_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == "effective_train"
        ]
        self.assertEqual(len(effective_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in effective_calls[0].keywords}
        self.assertIsInstance(keywords["enable_aux_sft"], ast.Constant)
        self.assertFalse(keywords["enable_aux_sft"].value)
        self.assertEqual(keywords["train_role"].value, "attacker")
        self.assertEqual(keywords["early_stop_min_steps"].value, 1)
        self.assertIsInstance(keywords["fixed_defender_adapter"], ast.Call)
        self.assertNotEqual(keywords["fixed_defender_adapter"].args[0], ast.Constant(value=""))


if __name__ == "__main__":
    unittest.main()
