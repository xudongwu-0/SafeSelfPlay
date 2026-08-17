"""Fail-closed tests for the additive same-label gate-retry protocol."""

from __future__ import annotations

import ast
import copy
import errno
import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from role_lora_selfplay8 import build_selfplay8_schedule, population_labels
from roll.utils.selfplay_gate_retry import (
    OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS,
    RAW_PPO_EXHAUSTED_RECOVERY_STATUS,
    RAW_PPO_MAX_ATTEMPTS_PER_STAGE,
    bounded_raw_ppo_retry_policy,
    build_attempt_contract,
    build_objective_migration_requirement,
    build_ppo_only_recovery_trainer_source,
    build_recovery_history_entry,
    build_recovery_plan,
    canonical_json_sha256,
    normalize_completed_retry_validation,
    reconcile_atomic_population_swap,
    validate_exhausted_attempt,
    validate_final_population_state,
    validate_checkpoint_cadence,
    validate_ppo_only_recovery_manifest,
    validate_raw_ppo_attempt_capacity,
    validate_recovery_eligibility,
    validate_successful_gate,
    verify_attempt_contract,
    verify_recovery_history,
    verify_recovery_plan,
    verify_ppo_only_recovery_trainer_contract,
    verify_objective_migration_requirement,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "modal_role_lora_selfplay8.py": (
        "5eedfda2e111af4b398a117801bcca29064e7a166600a63abb88c8417491c4c2"
    ),
    "modal_upstream_selfredteam_role_lora.py": (
        "d8950d4487dff1df8901ee4ff10542e13249ad8b6aae3dc9e9959f5bb314e340"
    ),
    "role_lora_selfplay8.py": (
        "2098f73699b17497ca9ce113337d47ee0fe699cb71415c1c936da8a776a96ffa"
    ),
    "roll/utils/upstream_v2_payoff.py": (
        "a57552c6d5b42e8fcbdf7ae3cb1beafd53032c36fdd15bc79aa60b440f389b93"
    ),
    "modal_upstream_selfredteam_fixed_seed.py": (
        "72207bbb1c43b644ccd4c6194ca908fdc2c2879eabf76de5acc83b5e51a5b01c"
    ),
    "roll/utils/lora_sync_contract.py": (
        "a730240409baf01639cef68908aac4c90e808d5a611fc13e1bbdee5b147bba6e"
    ),
    "roll/third_party/vllm/worker.py": (
        "f8439c4d4bd7d32d6a76f0cd405e52809665d81ea1b6c7b0df09af74ec620272"
    ),
    "roll/third_party/deepspeed/model_update.py": (
        "90fc3a24b1a123b7aa7b4fbbfab259c48be2424c23a3f6f93d5804c3127f4b21"
    ),
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _retry_state() -> tuple[dict, list]:
    schedule = build_selfplay8_schedule(8)
    a2 = schedule[1].to_dict()
    state = {
        "schema_version": 1,
        "run_suffix": "test_run",
        "status": "stage_target_not_reached",
        "active_stage": "A2",
        "schedule": [stage.to_dict() for stage in schedule],
        "config": {
            "rounds": 8,
            "attacker_max_steps": 100,
            "defender_max_steps": 200,
            "save_steps": 10,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 30,
            "defender_early_stop_min_steps": 32,
        },
        "stages": {
            "A1": {
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": "/output/run/population/A1",
                "sha256": _sha("A1"),
            },
            "D1": {
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": "/output/run/population/D1",
                "sha256": _sha("D1"),
            },
            "A2": {
                **a2,
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": "/output/run/population/A2",
                "sha256": _sha("A2-failed"),
                "spawn_claim_id": _sha("A2-claim"),
                "run_dir": "/output/run/A2",
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
    return state, schedule


def _plan() -> dict:
    state, schedule = _retry_state()
    eligible = validate_recovery_eligibility(state, schedule)
    frozen_source = (
        ROOT / "modal_upstream_selfredteam_role_lora.py"
    ).read_text(encoding="utf-8")
    _effective_source, trainer_contract = (
        build_ppo_only_recovery_trainer_source(frozen_source)
    )
    return build_recovery_plan(
        state=state,
        eligibility=eligible,
        initial_state_file_sha256=_sha("initial-state"),
        frozen_training_sha256={
            "modal_upstream_selfredteam_role_lora.py": FROZEN_SHA256[
                "modal_upstream_selfredteam_role_lora.py"
            ]
        },
        recovery_implementation_sha256={"recovery.py": _sha("recovery")},
        recovery_trainer_contract=trainer_contract,
        plan_path="/output/run/gate_retry_v1/A2/plan.json",
        original_failure_evidence={"passed": True, "actual_final_step": 100},
    )


def _d2_plan() -> dict:
    schedule = build_selfplay8_schedule(8)
    d2 = schedule[2].to_dict()
    state = {
        "schema_version": 1,
        "run_suffix": "test_run",
        "status": "stage_target_not_reached",
        "active_stage": "D2",
        "schedule": [stage.to_dict() for stage in schedule],
        "config": {
            "rounds": 8,
            "attacker_max_steps": 100,
            "defender_max_steps": 200,
            "save_steps": 10,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 30,
            "defender_early_stop_min_steps": 32,
        },
        "stages": {
            label: {
                "status": "retained",
                "transition_state": "retained",
                "population_checkpoint": f"/output/run/population/{label}",
                "sha256": _sha(label),
            }
            for label in ("A1", "D1", "A2")
        },
    }
    state["stages"]["D2"] = {
        **d2,
        "status": "retained",
        "transition_state": "retained",
        "population_checkpoint": "/output/run/population/D2",
        "sha256": _sha("D2-failed"),
        "spawn_claim_id": _sha("D2-claim"),
        "run_dir": "/output/run/D2",
        "requested_max_step": 200,
        "actual_final_step": 200,
        "stopped_early": False,
        "successor_release": {"approved": False, "basis": "target not reached"},
    }
    eligible = validate_recovery_eligibility(state, schedule)
    frozen_source = (
        ROOT / "modal_upstream_selfredteam_role_lora.py"
    ).read_text(encoding="utf-8")
    _effective_source, trainer_contract = (
        build_ppo_only_recovery_trainer_source(frozen_source)
    )
    return build_recovery_plan(
        state=state,
        eligibility=eligible,
        initial_state_file_sha256=_sha("initial-D2-state"),
        frozen_training_sha256={
            "modal_upstream_selfredteam_role_lora.py": FROZEN_SHA256[
                "modal_upstream_selfredteam_role_lora.py"
            ]
        },
        recovery_implementation_sha256={"recovery.py": _sha("recovery")},
        recovery_trainer_contract=trainer_contract,
        plan_path="/output/run/gate_retry_v1/D2/plan.json",
        original_failure_evidence={"passed": True, "actual_final_step": 200},
    )


def _released_recovery() -> dict:
    plan = _plan()
    contract = build_attempt_contract(
        plan,
        attempt_number=1,
        trainable_init_checkpoint="/output/run/population/A2",
        trainable_init_sha256=_sha("A2-failed"),
        trainer_run_suffix="test_A2_gate_retry_001",
        contract_path="/output/attempt_1/contract.json",
    )
    return {
        "schema_version": 1,
        "status": "released",
        "plan_id": plan["plan_id"],
        "plan": plan,
        "attempts": [
            {
                "attempt_id": contract["attempt_id"],
                "contract": contract,
                "status": "qualified_ready_to_release",
                "gate_result": {"passed": True},
            }
        ],
        "swap_journal": {"phase": "complete"},
        "official_population_released": True,
    }


def _early_stop_validation(role: str) -> dict:
    is_attacker = role == "attacker"
    final_step = 34 if is_attacker else 36
    companions = (
        {
            "attacker/cot_format_violation": {
                "direction": "max",
                "bound": 0.05,
            }
        }
        if is_attacker
        else {
            "defender/wildguard_actual_harmful_count": {
                "direction": "min",
                "bound": 61.0,
            },
            "defender/wildguard_actual_benign_joint_success": {
                "direction": "min",
                "bound": 0.95,
            },
            "defender/wildguard_actual_benign_count": {
                "direction": "min",
                "bound": 61.0,
            },
        }
    )
    rows = []
    for step in range(final_step - 4, final_step + 1):
        metrics = (
            {"attacker/cot_format_violation": 0.01}
            if is_attacker
            else {
                "defender/wildguard_actual_harmful_count": 64,
                "defender/wildguard_actual_benign_joint_success": 0.97,
                "defender/wildguard_actual_benign_count": 63,
            }
        )
        rows.append(
            {"step": step, "value": 0.96, "qualified": True, "metrics": metrics}
        )
    early_stop = {
        "metric": (
            "attacker/request_success_rate"
            if is_attacker
            else "defender/wildguard_actual_harmful_joint_success"
        ),
        "threshold": 0.95,
        "patience": 5,
        "min_steps": 30 if is_attacker else 32,
        "triggered": True,
        "streak": 5,
        "last_step": final_step,
        "actual_final_step": final_step,
        "checkpoint_tag": f"global_step{final_step}",
        "companion_bounds": companions,
        "history": rows,
    }
    validation = {
        "requested_max_step": 100 if is_attacker else 200,
        "actual_final_step": final_step,
        "final_checkpoint": f"/output/run/ckpt/global_step{final_step}_hf",
        "stopped_early": True,
        "early_stop": early_stop,
    }
    return _with_cadence(validation, final_step=final_step)


def _with_cadence(validation: dict, *, final_step: int, save_steps: int = 10) -> dict:
    steps = list(range(save_steps, final_step + 1, save_steps))
    if not steps or steps[-1] != final_step:
        steps.append(final_step)
    digests = {str(step): _sha(f"checkpoint-{step}") for step in steps}
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
            "changed_across_checkpoints": True,
        }
    )
    return validation


class RecoveryIdentityTests(unittest.TestCase):
    def test_eligibility_binds_schedule_claim_parent_and_opponent(self):
        state, schedule = _retry_state()
        result = validate_recovery_eligibility(state, schedule)
        self.assertEqual(result["label"], "A2")
        self.assertEqual(result["role"], "attacker")
        self.assertEqual(result["original_parent"]["label"], "A1")
        self.assertEqual(result["fixed_opponent"]["label"], "D1")
        self.assertEqual(result["spawn_claim_id"], _sha("A2-claim"))

    def test_eligibility_rejects_successor_or_schedule_drift(self):
        state, schedule = _retry_state()
        state["stages"]["D2"] = {"status": "spawn_pending"}
        with self.assertRaisesRegex(RuntimeError, "Successor D2 already exists"):
            validate_recovery_eligibility(state, schedule)
        state, schedule = _retry_state()
        state["schedule"][1]["fixed_opponent"] = "D8"
        with self.assertRaisesRegex(RuntimeError, "schedule differs"):
            validate_recovery_eligibility(state, schedule)

    def test_plan_and_attempt_contracts_are_self_hashing_and_immutable(self):
        plan = _plan()
        self.assertEqual(verify_recovery_plan(plan), plan["plan_id"])
        contract = build_attempt_contract(
            plan,
            attempt_number=1,
            trainable_init_checkpoint="/output/run/population/A2",
            trainable_init_sha256=_sha("A2-failed"),
            trainer_run_suffix="test_A2_gate_retry_001",
            contract_path="/output/attempt_1/contract.json",
        )
        self.assertEqual(verify_attempt_contract(contract, plan), contract["attempt_id"])
        self.assertIn("cold optimizer", contract["optimizer_policy"])
        changed = copy.deepcopy(contract)
        changed["fixed_opponent"]["sha256"] = _sha("different")
        with self.assertRaisesRegex(RuntimeError, "digest drifted"):
            verify_attempt_contract(changed, plan)
        changed["attempt_id"] = canonical_json_sha256(
            {key: value for key, value in changed.items() if key != "attempt_id"}
        )
        with self.assertRaisesRegex(RuntimeError, "provenance drifted"):
            verify_attempt_contract(changed, plan)
        with self.assertRaisesRegex(RuntimeError, "exceeds the bounded"):
            build_attempt_contract(
                plan,
                attempt_number=2,
                trainable_init_checkpoint="/output/attempt_1/A2",
                trainable_init_sha256=_sha("attempt-1"),
                trainer_run_suffix="test_A2_gate_retry_002",
                contract_path="/output/attempt_2/contract.json",
            )
        forged_second = copy.deepcopy(contract)
        forged_second["attempt_number"] = 2
        forged_second["attempt_id"] = canonical_json_sha256(
            {
                key: value
                for key, value in forged_second.items()
                if key != "attempt_id"
            }
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds the bounded"):
            verify_attempt_contract(forged_second, plan)

    def test_single_raw_retry_capacity_and_migration_handoff_are_hash_bound(self):
        plan = _plan()
        self.assertEqual(
            plan["bounded_raw_ppo_retry_policy"],
            bounded_raw_ppo_retry_policy(),
        )
        self.assertEqual(
            plan["bounded_raw_ppo_retry_policy"]["early_stop_min_steps"],
            1,
        )
        self.assertEqual(
            plan["bounded_raw_ppo_retry_policy"][
                "earliest_possible_stop_step"
            ],
            5,
        )
        self.assertEqual(RAW_PPO_MAX_ATTEMPTS_PER_STAGE, 1)
        self.assertEqual(validate_raw_ppo_attempt_capacity(plan, []), 1)
        contract = build_attempt_contract(
            plan,
            attempt_number=1,
            trainable_init_checkpoint="/output/run/population/A2",
            trainable_init_sha256=_sha("A2-failed"),
            trainer_run_suffix="test_A2_gate_retry_001",
            contract_path="/output/attempt_1/contract.json",
        )
        attempt = {
            "attempt_id": contract["attempt_id"],
            "attempt_number": 1,
            "status": "gate_not_reached",
            "contract": contract,
            "candidate_checkpoint": "/output/attempt_1/population/A2",
            "candidate_sha256": _sha("A2-retry-failed"),
            "pruning_complete": True,
            "gate_result": {
                "passed": True,
                "classification": "gate_not_reached_after_complete_budget",
            },
        }
        with self.assertRaisesRegex(RuntimeError, "objective migration"):
            validate_raw_ppo_attempt_capacity(plan, [attempt])
        requirement = build_objective_migration_requirement(plan, [attempt])
        self.assertEqual(
            requirement["trainable_init_sha256"],
            _sha("A2-retry-failed"),
        )
        self.assertEqual(
            requirement["required_next_objective_contract"][
                "optimization_reward"
            ],
            {"positive": 1.0, "negative": -1.0},
        )
        self.assertIn(
            "must not replace or mix",
            requirement["official_payoff_evaluation"],
        )
        self.assertEqual(
            verify_objective_migration_requirement(
                requirement, plan, [attempt]
            ),
            requirement["requirement_id"],
        )
        changed = copy.deepcopy(requirement)
        changed["trainable_init_checkpoint"] = "/output/wrong/A2"
        with self.assertRaisesRegex(RuntimeError, "handoff drifted"):
            verify_objective_migration_requirement(changed, plan, [attempt])

    def test_released_recovery_history_preserves_prior_stage_provenance(self):
        recovery = _released_recovery()
        plan = recovery["plan"]
        entry = build_recovery_history_entry(
            recovery,
            archived_state_file_sha256=_sha("released-state"),
        )
        self.assertEqual(verify_recovery_history([entry]), [plan["plan_id"]])
        changed = copy.deepcopy(entry)
        changed["recovery"]["official_population_released"] = False
        with self.assertRaisesRegex(RuntimeError, "History entry drifted"):
            verify_recovery_history([changed])


class GateValidationTests(unittest.TestCase):
    def test_retry_gate_can_stop_at_step_five_with_initializer_change_proof(self):
        validation = _early_stop_validation("attacker")
        validation["actual_final_step"] = 5
        validation["final_checkpoint"] = (
            "/output/run/ckpt/global_step5_hf"
        )
        early = validation["early_stop"]
        early["min_steps"] = 1
        early["actual_final_step"] = 5
        early["last_step"] = 5
        early["checkpoint_tag"] = "global_step5"
        early["history"] = [
            {**row, "step": step}
            for row, step in zip(early["history"], range(1, 6))
        ]
        _with_cadence(validation, final_step=5)
        validation["changed_across_checkpoints"] = False
        final_sha = _sha("checkpoint-5")
        proof = validate_successful_gate(
            validation,
            role="attacker",
            attacker_min_steps=1,
            expected_budget=100,
            expected_final_sha256=final_sha,
            allow_single_checkpoint_change_proof=True,
            expected_initial_sha256=_sha("initializer"),
        )
        self.assertEqual(proof["tail_steps"], [1, 2, 3, 4, 5])
        self.assertTrue(
            proof["checkpoint_cadence"][
                "single_checkpoint_change_proof"
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "did not change"):
            validate_successful_gate(
                validation,
                role="attacker",
                attacker_min_steps=1,
                expected_budget=100,
                expected_final_sha256=final_sha,
                allow_single_checkpoint_change_proof=True,
                expected_initial_sha256=final_sha,
            )

    def test_captured_live_d1_tail_recomputes_and_rejects_b_companion_drift(self):
        fixture_path = (
            ROOT
            / "tests/fixtures/selfplay_gate_retry_d1_live_tail.json"
        )
        validation = json.loads(fixture_path.read_text(encoding="utf-8"))
        proof = validate_successful_gate(
            validation,
            role="defender",
            expected_budget=200,
            save_steps=10,
        )
        self.assertEqual(proof["tail_steps"], [53, 54, 55, 56, 57])
        self.assertEqual(
            proof["attacker_success_tail"],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        )
        for mutation in ("benign_rate", "benign_count", "qualified"):
            changed = copy.deepcopy(validation)
            row = changed["early_stop"]["history"][-1]
            if mutation == "benign_rate":
                row["metrics"][
                    "defender/wildguard_actual_benign_joint_success"
                ] = 0.94
            elif mutation == "benign_count":
                row["metrics"]["defender/wildguard_actual_benign_count"] = 60
            else:
                row["qualified"] = False
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                validate_successful_gate(
                    changed,
                    role="defender",
                    expected_budget=200,
                    save_steps=10,
                )

    def test_attacker_gate_requires_five_successes_and_cot_companion(self):
        validation = _early_stop_validation("attacker")
        result = validate_successful_gate(
            validation,
            role="attacker",
            expected_budget=100,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["tail_steps"], [30, 31, 32, 33, 34])
        validation["early_stop"]["history"][-1]["metrics"][
            "attacker/cot_format_violation"
        ] = 0.06
        with self.assertRaisesRegex(RuntimeError, "companion failed"):
            validate_successful_gate(
                validation,
                role="attacker",
                expected_budget=100,
            )

    def test_defender_gate_recomputes_h_and_direct_b_conjunction(self):
        validation = _early_stop_validation("defender")
        result = validate_successful_gate(
            validation,
            role="defender",
            expected_budget=200,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["tail_steps"], [32, 33, 34, 35, 36])
        validation["early_stop"]["history"][2]["metrics"][
            "defender/wildguard_actual_benign_joint_success"
        ] = 0.94
        with self.assertRaisesRegex(RuntimeError, "companion failed"):
            validate_successful_gate(
                validation,
                role="defender",
                expected_budget=200,
            )

    def test_gate_rejects_tail_before_minimum_step(self):
        validation = _early_stop_validation("defender")
        validation["actual_final_step"] = 35
        early = validation["early_stop"]
        early["actual_final_step"] = 35
        early["last_step"] = 35
        early["checkpoint_tag"] = "global_step35"
        early["history"] = [
            {**row, "step": step}
            for row, step in zip(early["history"], range(31, 36))
        ]
        validation["final_checkpoint"] = "/output/run/ckpt/global_step35_hf"
        _with_cadence(validation, final_step=35)
        with self.assertRaisesRegex(RuntimeError, "minimum step"):
            validate_successful_gate(
                validation,
                role="defender",
                expected_budget=200,
            )

    def test_budget_exhaustion_and_frozen_completed_replay(self):
        exhausted = _with_cadence({
            "requested_max_step": 100,
            "actual_final_step": 100,
            "final_checkpoint": "/output/run/ckpt/global_step100_hf",
            "stopped_early": False,
        }, final_step=100)
        self.assertTrue(
            validate_exhausted_attempt(
                exhausted,
                expected_budget=100,
                save_steps=10,
            )["passed"]
        )
        replay = _with_cadence({
            "final_checkpoint": "/output/run/ckpt/global_step100_hf",
        }, final_step=100)
        normalized = normalize_completed_retry_validation(
            replay,
            expected_budget=100,
            save_steps=10,
            early_stop_artifact_exists=False,
        )
        self.assertIs(normalized["stopped_early"], False)
        self.assertEqual(normalized["actual_final_step"], 100)
        with self.assertRaisesRegex(RuntimeError, "early-stop artifact"):
            normalize_completed_retry_validation(
                replay,
                expected_budget=100,
                save_steps=10,
                early_stop_artifact_exists=True,
            )

    def test_real_cadence_flag_is_false_but_every_cadence_field_is_hard(self):
        validation = _early_stop_validation("attacker")
        proof = validate_checkpoint_cadence(
            validation,
            expected_final_step=34,
            save_steps=10,
            expected_final_sha256=_sha("checkpoint-34"),
        )
        self.assertFalse(proof["complete_cadence_required"])
        for field, value, error in (
            ("complete_cadence_required", True, "cadence-required"),
            ("complete_cadence_verified", False, "not verified"),
            ("missing_checkpoint_steps", [20], "missing steps"),
            ("observed_checkpoint_steps", [10, 30, 34], "not exact"),
            ("expected_checkpoint_count", 3, "count drifted"),
            (
                "expected_checkpoint_sha256",
                {
                    **validation["expected_checkpoint_sha256"],
                    "34": _sha("wrong-final"),
                },
                "digest mismatch",
            ),
        ):
            changed = copy.deepcopy(validation)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError,
                error,
            ):
                validate_checkpoint_cadence(
                    changed,
                    expected_final_step=34,
                    save_steps=10,
                    expected_final_sha256=_sha("checkpoint-34"),
                )
        with self.assertRaisesRegex(RuntimeError, "population digest drifted"):
            validate_checkpoint_cadence(
                validation,
                expected_final_step=34,
                save_steps=10,
                expected_final_sha256=_sha("different-population"),
            )


class AtomicSwapTests(unittest.TestCase):
    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256((path / "weights").read_bytes()).hexdigest()

    def test_swap_reconciles_pre_exchange_post_exchange_and_completed_layouts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            staging = root / "staging"
            archive = root / "archive"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "weights").write_bytes(b"old")
            (staging / "weights").write_bytes(b"new")
            old_sha = _sha("old")
            new_sha = _sha("new")
            try:
                first = reconcile_atomic_population_swap(
                    canonical=canonical,
                    staging=staging,
                    archive=archive,
                    old_sha256=old_sha,
                    new_sha256=new_sha,
                    checkpoint_digest=self._digest,
                )
            except OSError as error:
                if error.errno in {errno.ENOSYS, errno.EINVAL, errno.EXDEV}:
                    self.skipTest(f"renameat2 exchange unavailable: {error}")
                raise
            self.assertEqual(first, {"action": "atomic_exchange", "complete": False})
            self.assertEqual(self._digest(canonical), new_sha)
            self.assertEqual(self._digest(staging), old_sha)
            second = reconcile_atomic_population_swap(
                canonical=canonical,
                staging=staging,
                archive=archive,
                old_sha256=old_sha,
                new_sha256=new_sha,
                checkpoint_digest=self._digest,
            )
            self.assertTrue(second["complete"])
            self.assertEqual(self._digest(archive), old_sha)
            third = reconcile_atomic_population_swap(
                canonical=canonical,
                staging=staging,
                archive=archive,
                old_sha256=old_sha,
                new_sha256=new_sha,
                checkpoint_digest=self._digest,
            )
            self.assertEqual(third["action"], "already_complete")

    def test_swap_fails_closed_on_unknown_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            canonical.mkdir()
            (canonical / "weights").write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
                reconcile_atomic_population_swap(
                    canonical=canonical,
                    staging=root / "missing",
                    archive=root / "archive",
                    old_sha256=_sha("old"),
                    new_sha256=_sha("new"),
                    checkpoint_digest=self._digest,
                )


class FinalPopulationTests(unittest.TestCase):
    def test_exact_16_live_digests_retention_and_releases(self):
        with TemporaryDirectory() as directory:
            population = Path(directory)
            labels = population_labels(8)
            stages = {}
            gate_proofs = {}
            for label in labels:
                checkpoint = population / label
                checkpoint.mkdir()
                (checkpoint / "weights").write_text(label)
                stage = {
                    "status": "retained",
                    "transition_state": "retained",
                    "population_checkpoint": str(checkpoint),
                    "sha256": _sha(label),
                }
                if label != "A1":
                    stage["successor_release"] = {"approved": True}
                    stage["stopped_early"] = True
                    gate_proofs[label] = {
                        "passed": True,
                        "stage_label": label,
                    }
                stages[label] = stage
            rows = validate_final_population_state(
                {"stages": stages},
                expected_labels=labels,
                population_root=population,
                checkpoint_digest=lambda path: _sha((path / "weights").read_text()),
                gate_proofs=gate_proofs,
            )
            self.assertEqual(len(rows), 16)
            (population / "unexpected").mkdir()
            with self.assertRaisesRegex(RuntimeError, "membership drifted"):
                validate_final_population_state(
                    {"stages": stages},
                    expected_labels=labels,
                    population_root=population,
                    checkpoint_digest=lambda path: _sha(
                        (path / "weights").read_text()
                    ),
                    gate_proofs=gate_proofs,
                )

    def test_only_a1_is_exempt_from_a_recomputed_gate(self):
        with TemporaryDirectory() as directory:
            population = Path(directory)
            labels = population_labels(8)
            stages = {}
            proofs = {}
            for label in labels:
                checkpoint = population / label
                checkpoint.mkdir()
                (checkpoint / "weights").write_text(label)
                stages[label] = {
                    "status": "retained",
                    "transition_state": "retained",
                    "population_checkpoint": str(checkpoint),
                    "sha256": _sha(label),
                    "stopped_early": label != "A1",
                    "successor_release": {"approved": label != "A1"},
                }
                if label != "A1":
                    proofs[label] = {"passed": True, "stage_label": label}
            del proofs["D1"]
            with self.assertRaisesRegex(RuntimeError, "proof membership"):
                validate_final_population_state(
                    {"stages": stages},
                    expected_labels=labels,
                    population_root=population,
                    checkpoint_digest=lambda path: _sha(
                        (path / "weights").read_text()
                    ),
                    gate_proofs=proofs,
                )
            proofs["D1"] = {"passed": True, "stage_label": "D1"}
            stages["D1"]["stopped_early"] = False
            with self.assertRaisesRegex(RuntimeError, "gate not passed"):
                validate_final_population_state(
                    {"stages": stages},
                    expected_labels=labels,
                    population_root=population,
                    checkpoint_digest=lambda path: _sha(
                        (path / "weights").read_text()
                    ),
                    gate_proofs=proofs,
                )


class PpoOnlyRecoveryTrainerTests(unittest.TestCase):
    def test_cold_container_completed_and_early_stop_replay_rebuilds_a_and_d_runtime_copies(self):
        source_path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_ensure_runtime_compatible_adapter_after_replay"
        )
        for role in ("attacker", "defender"):
            for replay_kind in ("completed", "early_stop"):
                with self.subTest(role=role, replay_kind=replay_kind), TemporaryDirectory() as directory:
                    root = Path(directory)
                    trainable_source = root / "trainable_source"
                    fixed_source = root / "fixed_source"
                    trainable_source.mkdir()
                    fixed_source.mkdir()
                    (trainable_source / "weights").write_bytes(
                        f"{role}-{replay_kind}-trainable".encode()
                    )
                    (fixed_source / "weights").write_bytes(
                        f"{role}-{replay_kind}-fixed".encode()
                    )
                    trainable_runtime = root / f"{role}_lora_init_compatible"
                    fixed_runtime = root / "fixed_opponent_lora_compatible"
                    if role == "attacker" and replay_kind == "completed":
                        trainable_runtime.mkdir()
                    destinations = {
                        f"{role}_lora_init_compatible": trainable_runtime,
                        "fixed_opponent_lora_compatible": fixed_runtime,
                    }
                    prepare_calls = []

                    class FrozenRole:
                        @staticmethod
                        def _is_complete_hf_checkpoint(path):
                            return path.is_dir() and (path / "weights").is_file()

                        @staticmethod
                        def _prepare_peft_compatible_adapter(source, destination_name):
                            prepare_calls.append((source, destination_name))
                            destination = destinations[destination_name]
                            if destination.exists():
                                for child in destination.iterdir():
                                    child.unlink()
                                destination.rmdir()
                            destination.mkdir()
                            (destination / "weights").write_bytes(
                                (Path(source) / "weights").read_bytes()
                            )
                            return str(destination)

                    def digest(path):
                        return hashlib.sha256((path / "weights").read_bytes()).hexdigest()

                    namespace = {
                        "Path": Path,
                        "frozen_role_lora": FrozenRole,
                        "checkpoint_weight_digest": digest,
                    }
                    exec(
                        compile(
                            ast.fix_missing_locations(
                                ast.Module(body=[helper], type_ignores=[])
                            ),
                            str(source_path),
                            "exec",
                        ),
                        namespace,
                    )
                    ensure = namespace[
                        "_ensure_runtime_compatible_adapter_after_replay"
                    ]
                    trainable_sha = digest(trainable_source)
                    fixed_sha = digest(fixed_source)
                    self.assertEqual(
                        ensure(
                            source=trainable_source,
                            runtime=trainable_runtime,
                            destination_name=f"{role}_lora_init_compatible",
                            expected_sha256=trainable_sha,
                        ),
                        trainable_sha,
                    )
                    self.assertEqual(
                        ensure(
                            source=fixed_source,
                            runtime=fixed_runtime,
                            destination_name="fixed_opponent_lora_compatible",
                            expected_sha256=fixed_sha,
                        ),
                        fixed_sha,
                    )
                    self.assertEqual(len(prepare_calls), 2)
                    ensure(
                        source=trainable_source,
                        runtime=trainable_runtime,
                        destination_name=f"{role}_lora_init_compatible",
                        expected_sha256=trainable_sha,
                    )
                    self.assertEqual(len(prepare_calls), 2)

    def test_dynamic_clone_is_hash_bound_and_frozen_source_is_unchanged(self):
        frozen_path = ROOT / "modal_upstream_selfredteam_role_lora.py"
        before = frozen_path.read_bytes()
        effective_source, descriptor = build_ppo_only_recovery_trainer_source(
            before.decode("utf-8")
        )
        self.assertEqual(
            descriptor["frozen_core_source_sha256"],
            FROZEN_SHA256["modal_upstream_selfredteam_role_lora.py"],
        )
        self.assertEqual(
            verify_ppo_only_recovery_trainer_contract(
                descriptor,
                expected_frozen_core_sha256=FROZEN_SHA256[
                    "modal_upstream_selfredteam_role_lora.py"
                ],
            ),
            descriptor["patch_descriptor_sha256"],
        )
        compile(effective_source, "<effective-ppo-only>", "exec")
        self.assertIn("postfill_cot_stop_after_step == 0", effective_source)
        self.assertIn("defender_sft_optimizer_slots_per_rollout == 0", effective_source)
        self.assertIn("early_stop_min_steps == 1", effective_source)
        self.assertEqual(
            descriptor["patch_replacements"][
                "ppo_only_gate_from_step_one"
            ],
            1,
        )
        self.assertEqual(frozen_path.read_bytes(), before)
        changed = copy.deepcopy(descriptor)
        changed["ppo_only_recipe"]["enable_aux_sft"] = True
        with self.assertRaisesRegex(RuntimeError, "descriptor digest drifted"):
            verify_ppo_only_recovery_trainer_contract(changed)

    def test_exact_a_and_d_ppo_only_recipes_pass_the_frozen_guard_prelude(self):
        effective_source, _descriptor = build_ppo_only_recovery_trainer_source(
            (
                ROOT / "modal_upstream_selfredteam_role_lora.py"
            ).read_text(encoding="utf-8")
        )

        class ReachedRuntimeSetup(Exception):
            pass

        class StopBeforeFilesystem:
            def reload(self):
                raise ReachedRuntimeSetup

        namespace = {
            "DEFAULT_FIXED_SEED": "seed",
            "BASE_MODEL": "base",
            "SFT_ADAPTER": "sft",
            "Path": Path,
            "hashlib": hashlib,
            "os": os,
            "__file__": str(
                (ROOT / "modal_upstream_selfredteam_role_lora.py").resolve()
            ),
            "_sha256_path": lambda _path: "implementation-sha",
            "_validate_defender_joint_runtime_configuration": (
                lambda *_args, **_kwargs: None
            ),
            "_hf_token": lambda: "token",
            "_warmup_wildguard_endpoint": lambda _url: None,
            "OUTPUT_ROOT": "/output",
            "output_vol": StopBeforeFilesystem(),
        }
        exec(compile(effective_source, "<ppo-only-prelude>", "exec"), namespace)
        train = namespace["_effective_gate_retry_ppo_only_train"]
        defender = {
            "remote_rm_url": "rm",
            "steps": 200,
            "normal_prompt_mix": True,
            "normal_prompt_pool_size": 0,
            "rollout_batch_size": 128,
            "micro_rollout_batch_size": 8,
            "micro_train_batch_size": 8,
            "train_batch_size": 32,
            "save_steps": 10,
            "actor_learning_rate": 1e-5,
            "init_kl_coef": 0.0,
            "actor_lr_scheduler": "constant_with_warmup",
            "lr_warmup_ratio": 0.05,
            "actor_lr_warmup_steps_override": 20,
            "enable_aux_sft": False,
            "run_suffix": "guard-test",
            "train_role": "defender",
            "fixed_attacker_adapter": "/fixed/A2",
            "defender_prompt_profile": "upstream",
            "upstream_invalid_handling": True,
            "base_model": "base",
            "attacker_init_adapter": "/init/D1",
            "attacker_prompt_profile": "optimized",
            "strict_upstream_alignment": False,
            "lora_rank": 64,
            "lora_alpha": 64,
            "monitor_reference_kl": False,
            "postfill_cot_stop_after_step": 0,
            "role_specific_aux_sft": False,
            "v2_runtime": True,
            "v2_continuation_sft": False,
            "defender_sft_optimizer_slots_per_rollout": 0,
            "defender_raw_reinforce_advantages": True,
            "defender_reinforce_advantage_mode": "joint_signed",
            "defender_reward_utility": "joint_signed",
            "defender_prompt_pool_path": "/pool",
            "defender_prompt_pool_sha256": "a" * 64,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 1,
        }
        attacker = {
            **defender,
            "steps": 100,
            "train_role": "attacker",
            "fixed_attacker_adapter": "",
            "fixed_defender_adapter": "/fixed/D1",
            "actor_lr_warmup_steps_override": None,
            "attacker_init_adapter": "/init/A1",
            "defender_raw_reinforce_advantages": False,
            "defender_reinforce_advantage_mode": "raw_no_center",
            "defender_reward_utility": "upstream_additive",
            "defender_prompt_pool_path": "",
            "defender_prompt_pool_sha256": "",
            "early_stop_min_steps": 1,
        }
        for role, recipe in (("defender", defender), ("attacker", attacker)):
            with self.subTest(role=role), self.assertRaises(ReachedRuntimeSetup):
                train(**recipe)
        for role, recipe in (("defender", defender), ("attacker", attacker)):
            invalid_minimum = {**recipe, "early_stop_min_steps": 0}
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError,
                "at least early_stop_patience",
            ):
                train(**invalid_minimum)
        invalid = {
            **defender,
            "role_specific_aux_sft": True,
            "early_stop_min_steps": 5,
        }
        with self.assertRaisesRegex(ValueError, "requires enable_aux_sft"):
            train(**invalid)

    def test_recovery_manifest_binds_original_and_runtime_adapter_paths(self):
        cases = (
            (
                "attacker",
                "A2",
                _plan(),
                "/output/run/population/A2",
                _sha("A2-failed"),
            ),
            (
                "defender",
                "D2",
                _d2_plan(),
                "/output/run/population/D2",
                _sha("D2-failed"),
            ),
        )
        for role, label, plan, init_checkpoint, init_sha in cases:
            with self.subTest(role=role):
                suffix = f"test_{label}_gate_retry_001"
                contract = build_attempt_contract(
                    plan,
                    attempt_number=1,
                    trainable_init_checkpoint=init_checkpoint,
                    trainable_init_sha256=init_sha,
                    trainer_run_suffix=suffix,
                    contract_path=f"/output/{label}/attempt_1/contract.json",
                )
                effective = contract["recovery_trainer_contract"]
                path = f"/output/run/{label}/gate_retry_recovery_manifest.json"
                receipt = {
                    "schema_version": 1,
                    "policy": "same-label-ppo-only-dynamic-clone-v1",
                    "plan_id": plan["plan_id"],
                    "attempt_id": contract["attempt_id"],
                    "stage_label": label,
                    "role": role,
                    "trainer_run_suffix": suffix,
                    "frozen_training_implementation_sha256": contract[
                        "frozen_training_implementation_sha256"
                    ],
                    "recovery_implementation_sha256": contract[
                        "recovery_implementation_sha256"
                    ],
                    "effective_trainer_contract": effective,
                    "bounded_raw_ppo_retry_policy": contract[
                        "bounded_raw_ppo_retry_policy"
                    ],
                    "implementation_identity": {
                        "frozen_core": {
                            "path": "modal_upstream_selfredteam_role_lora.py",
                            "sha256": FROZEN_SHA256[
                                "modal_upstream_selfredteam_role_lora.py"
                            ],
                        },
                        "additive_recovery_sources": contract[
                            "recovery_implementation_sha256"
                        ],
                        "effective_dynamic_function": {
                            "policy": "same-label-ppo-only-dynamic-clone-v1",
                            "source_sha256": effective[
                                "effective_function_source_sha256"
                            ],
                            "patch_descriptor_sha256": effective[
                                "patch_descriptor_sha256"
                            ],
                        },
                    },
                    "runtime_adapter_mapping": {
                        "trainable": {
                            "original_checkpoint": init_checkpoint,
                            "original_sha256": init_sha,
                            "runtime_compatible_checkpoint": (
                                f"/tmp/{role}_lora_init_compatible"
                            ),
                            "runtime_weight_sha256": init_sha,
                        },
                        "fixed_opponent": {
                            "original_checkpoint": contract["fixed_opponent"][
                                "checkpoint"
                            ],
                            "original_sha256": contract["fixed_opponent"][
                                "sha256"
                            ],
                            "runtime_compatible_checkpoint": (
                                "/tmp/fixed_opponent_lora_compatible"
                            ),
                            "runtime_weight_sha256": contract["fixed_opponent"][
                                "sha256"
                            ],
                        },
                    },
                    "ppo_only_recipe": {
                        **effective["ppo_only_recipe"],
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
                receipt["recovery_manifest_sha256"] = canonical_json_sha256(
                    receipt
                )
                trainer_manifest = {
                    "gate_retry_effective_implementation": {
                        "policy": "same-label-ppo-only-dynamic-clone-v1",
                        "path": path,
                        "sha256": receipt["recovery_manifest_sha256"],
                        "frozen_core_implementation_sha256": FROZEN_SHA256[
                            "modal_upstream_selfredteam_role_lora.py"
                        ],
                        "effective_function_source_sha256": effective[
                            "effective_function_source_sha256"
                        ],
                        "patch_descriptor_sha256": effective[
                            "patch_descriptor_sha256"
                        ],
                    }
                }
                proof = validate_ppo_only_recovery_manifest(
                    receipt,
                    trainer_manifest,
                    plan=plan,
                    contract=contract,
                    recovery_manifest_path=path,
                )
                self.assertTrue(proof["passed"])
                self.assertEqual(
                    proof["runtime_adapter_mapping"]["trainable"][
                        "runtime_compatible_checkpoint"
                    ],
                    f"/tmp/{role}_lora_init_compatible",
                )
                changed = copy.deepcopy(receipt)
                changed["runtime_adapter_mapping"]["trainable"][
                    "original_checkpoint"
                ] = f"/tmp/{role}_lora_init_compatible"
                with self.assertRaisesRegex(
                    RuntimeError,
                    "adapter mapping drifted",
                ):
                    validate_ppo_only_recovery_manifest(
                        changed,
                        trainer_manifest,
                        plan=plan,
                        contract=contract,
                        recovery_manifest_path=path,
                    )


class AdditiveEntrypointTests(unittest.TestCase):
    def test_frozen_sources_are_byte_identical(self):
        for relative, expected in FROZEN_SHA256.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_modal_entrypoint_is_additive_and_contains_safety_phases(self):
        source = (ROOT / "modal_role_lora_selfplay8_gate_retry.py").read_text()
        required = (
            "resume_role_lora_selfplay8_gate_retry",
            "audit_and_finalize_role_lora_selfplay8_population",
            "_persist_state_cas",
            "_drain_recovery_phase",
            "_rollover_released_recovery",
            "_reconcile_successful_swap",
            "population_attempts",
            "validate_final_population_state",
            "train_role_lora_gate_retry_ppo_only.remote",
            "_dispatch_stage_claim",
            '"joint_signed"',
            '"raw_no_center"',
        )
        for marker in required:
            self.assertIn(marker, source)
        self.assertNotIn("resume_role_lora_selfplay8_gate_retry.local", source)
        self.assertNotIn("datetime.now", source)

    def test_retry_trainer_call_is_exactly_ppo_only(self):
        path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_role_lora_gate_retry_ppo_only"
        )
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "effective_train"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        expected_constants = {
            "enable_aux_sft": False,
            "role_specific_aux_sft": False,
            "v2_continuation_sft": False,
            "postfill_cot_stop_after_step": 0,
            "defender_sft_optimizer_slots_per_rollout": 0,
            "monitor_reference_kl": False,
        }
        for key, expected in expected_constants.items():
            self.assertEqual(ast.literal_eval(keywords[key]), expected, key)
        rendered_call = ast.unparse(calls[0])
        for required in (
            "defender_raw_reinforce_advantages=not is_attacker",
            "'joint_signed'",
            "config['d1_data_contract']['training_prompt_pool_path']",
            "config['d1_data_contract']['training_prompt_pool_sha256']",
            "contract['bounded_raw_ppo_retry_policy']['early_stop_min_steps']",
        ):
            self.assertIn(required, rendered_call)
        rendered_function = ast.unparse(function)
        self.assertLess(
            rendered_function.index("run_dir_text = effective_train("),
            rendered_function.index(
                "trainable_runtime_sha256 = "
                "_ensure_runtime_compatible_adapter_after_replay("
            ),
        )
        self.assertIn(
            "frozen_role_lora._prepare_peft_compatible_adapter",
            path.read_text(encoding="utf-8"),
        )
        validator = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_attempt_output"
        )
        rendered_validator = ast.unparse(validator)
        self.assertIn("wildguard_actual_harmful_joint_success", rendered_validator)
        self.assertIn("wildguard_actual_benign_joint_success", rendered_validator)
        self.assertIn("four_rank_balanced_HHBBBBHH_cycle", rendered_validator)

    def test_later_failed_stage_rolls_released_recovery_into_history(self):
        source_path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(source_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_rollover_released_recovery"
        )
        namespace = {
            "Any": object,
            "Path": Path,
            "copy": copy,
            "RECOVERY_KEY": "gate_retry_recovery_v1",
            "RECOVERY_HISTORY_KEY": "gate_retry_recovery_history_v1",
            "_verify_existing_recovery": (
                lambda _root, state: state["gate_retry_recovery_v1"]
            ),
            "verify_recovery_history": verify_recovery_history,
            "_audit_released_recovery_artifacts": (
                lambda _state, _recovery: {"passed": True}
            ),
            "build_recovery_history_entry": build_recovery_history_entry,
            "_persist_state_cas": (
                lambda _root, state, expected_file_sha256: _sha(
                    canonical_json_sha256(state) + expected_file_sha256
                )
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        state = {
            "status": "stage_target_not_reached",
            "active_stage": "D2",
            "gate_retry_recovery_v1": _released_recovery(),
        }
        updated, _updated_sha = namespace["_rollover_released_recovery"](
            Path("/output/run"),
            state,
            _sha("state-before-D2-recovery"),
        )
        self.assertNotIn("gate_retry_recovery_v1", updated)
        history = updated["gate_retry_recovery_history_v1"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["stage_label"], "A2")
        self.assertTrue(
            history[0]["recovery"]["release_preservation_audit"]["passed"]
        )

    def test_raw_retry_has_no_unbounded_argument_or_recursive_dispatch(self):
        source = (ROOT / "modal_role_lora_selfplay8_gate_retry.py").read_text()
        tree = ast.parse(source)
        remote = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "resume_role_lora_selfplay8_gate_retry"
        )
        self.assertEqual([arg.arg for arg in remote.args.args], ["run_suffix"])
        self.assertNotIn("attempt_limit", ast.unparse(remote))
        self.assertNotIn(
            "resume_role_lora_selfplay8_gate_retry.spawn",
            ast.unparse(remote),
        )
        self.assertIn("_mark_raw_ppo_exhausted", ast.unparse(remote))

    def test_create_attempt_checks_durable_capacity_before_claim(self):
        source = (ROOT / "modal_role_lora_selfplay8_gate_retry.py").read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_create_attempt"
        )
        rendered = ast.unparse(function)
        self.assertLess(
            rendered.index("validate_raw_ppo_attempt_capacity"),
            rendered.index("build_attempt_contract"),
        )

    def test_success_and_failure_paths_use_matching_gate_validator_apis(self):
        source = (
            ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        ).read_text()
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        def call_keywords(function_name, callee_name):
            calls = [
                node
                for node in ast.walk(functions[function_name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == callee_name
            ]
            self.assertEqual(len(calls), 1)
            return {keyword.arg for keyword in calls[0].keywords}

        success_keywords = call_keywords(
            "_prepare_successful_swap",
            "validate_successful_gate",
        )
        self.assertIn(
            "allow_single_checkpoint_change_proof",
            success_keywords,
        )
        self.assertIn("expected_initial_sha256", success_keywords)
        failure_keywords = call_keywords(
            "_archive_failed_attempt",
            "validate_exhausted_attempt",
        )
        self.assertNotIn(
            "allow_single_checkpoint_change_proof",
            failure_keywords,
        )
        self.assertNotIn("expected_initial_sha256", failure_keywords)

    def test_failed_retry_seals_terminal_objective_migration_state(self):
        plan = _plan()
        contract = build_attempt_contract(
            plan,
            attempt_number=1,
            trainable_init_checkpoint="/output/run/population/A2",
            trainable_init_sha256=_sha("A2-failed"),
            trainer_run_suffix="test_A2_gate_retry_001",
            contract_path="/output/attempt_1/contract.json",
        )
        attempt = {
            "attempt_id": contract["attempt_id"],
            "attempt_number": 1,
            "status": "gate_not_reached",
            "contract": contract,
            "candidate_checkpoint": "/output/attempt_1/population/A2",
            "candidate_sha256": _sha("A2-retry-failed"),
            "pruning_complete": True,
            "gate_result": {
                "passed": True,
                "classification": "gate_not_reached_after_complete_budget",
            },
        }
        state = {
            "status": "stage_target_not_reached",
            "active_stage": "A2",
            "stages": {"A2": {"work_status": "retained"}},
            "gate_retry_recovery_v1": {
                "status": "active",
                "plan": plan,
                "attempts": [attempt],
                "active_attempt_id": None,
            },
        }
        source_path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(source_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_mark_raw_ppo_exhausted"
        )

        class Volume:
            def commit(self):
                return None

            def reload(self):
                return None

        captured = {}

        def persist(_root, value, *, expected_file_sha256):
            self.assertEqual(expected_file_sha256, _sha("state"))
            captured["state"] = copy.deepcopy(value)
            return _sha("terminal-state")

        namespace = {
            "Any": object,
            "Path": Path,
            "RuntimeError": RuntimeError,
            "copy": copy,
            "RECOVERY_KEY": "gate_retry_recovery_v1",
            "RAW_PPO_EXHAUSTED_RECOVERY_STATUS": (
                RAW_PPO_EXHAUSTED_RECOVERY_STATUS
            ),
            "OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS": (
                OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS
            ),
            "_verify_existing_recovery": (
                lambda _root, value: value["gate_retry_recovery_v1"]
            ),
            "build_objective_migration_requirement": (
                build_objective_migration_requirement
            ),
            "_strict_checkpoint": lambda *_args, **_kwargs: {},
            "_recovery_root": lambda _root, _label: Path("/recovery/A2"),
            "_write_exact_json": (
                lambda path, value: captured.update(
                    artifact_path=str(path), artifact=copy.deepcopy(value)
                )
            ),
            "output_vol": Volume(),
            "_load_state_snapshot": lambda _root: (state, _sha("state")),
            "file_sha256": lambda _path: _sha("artifact"),
            "_persist_state_cas": persist,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                ),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        updated, updated_sha = namespace["_mark_raw_ppo_exhausted"](
            Path("/output/run"), state, _sha("state")
        )
        self.assertEqual(updated_sha, _sha("terminal-state"))
        self.assertEqual(
            updated["status"], OBJECTIVE_MIGRATION_REQUIRED_STATE_STATUS
        )
        recovery = updated["gate_retry_recovery_v1"]
        self.assertEqual(
            recovery["status"], RAW_PPO_EXHAUSTED_RECOVERY_STATUS
        )
        self.assertIsNone(recovery["active_attempt_id"])
        self.assertEqual(
            recovery["objective_migration_requirement"][
                "trainable_init_sha256"
            ],
            _sha("A2-retry-failed"),
        )
        self.assertEqual(
            recovery["next_required_action"],
            "attacker_binary_joint_goal_and_cot_raw_no_center_no_std",
        )
        self.assertEqual(len(recovery["attempts"]), 1)
        self.assertEqual(
            updated["stages"]["A2"]["work_status"],
            "raw_ppo_retry_exhausted_objective_migration_required",
        )

    def test_preempted_failed_retry_is_sealed_instead_of_claiming_attempt_two(self):
        source_path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(source_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_continue_existing_phase"
        )
        state = {
            "gate_retry_recovery_v1": {
                "status": "active",
                "attempts": [
                    {
                        "status": "gate_not_reached",
                        "pruning_complete": True,
                    }
                ],
            }
        }
        calls = []
        namespace = {
            "Any": object,
            "Path": Path,
            "RECOVERY_KEY": "gate_retry_recovery_v1",
            "_verify_existing_recovery": (
                lambda _root, value: value["gate_retry_recovery_v1"]
            ),
            "_reconcile_successful_swap": lambda *_args: "swap",
            "_prune_promoted_attempt": lambda *_args: "prune",
            "_release_or_complete": lambda *_args: "release",
            "_mark_raw_ppo_exhausted": (
                lambda *_args: calls.append("exhausted") or "sealed"
            ),
            "_finish_pending_failed_prune": (
                lambda *_args: calls.append("prune-failed") or "pending"
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                ),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        result = namespace["_continue_existing_phase"](
            Path("/output/run"), state, _sha("state")
        )
        self.assertEqual(result, "sealed")
        self.assertEqual(calls, ["exhausted"])

    def test_terminal_reentry_live_audits_migration_initializer(self):
        source = (
            ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        ).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_existing_recovery"
        )
        strict_calls = [
            ast.unparse(node)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_strict_checkpoint"
        ]
        self.assertTrue(
            any(
                "requirement['trainable_init_checkpoint']" in call
                and "requirement['trainable_init_sha256']" in call
                for call in strict_calls
            )
        )

    def test_already_audited_path_rebuilds_live_and_requires_exact_artifact(self):
        source_path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(source_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_audit_completed_population"
        )

        class Volume:
            def reload(self):
                return None

        state = {
            "status": "completed",
            "active_stage": None,
            "final_population_audit": {"passed": True},
        }
        recorded = {"passed": True, "members": list(range(16))}
        live_value = copy.deepcopy(recorded)
        calls = {"live": 0}

        def build(_root, _state):
            calls["live"] += 1
            return copy.deepcopy(live_value)

        namespace = {
            "Any": object,
            "Path": Path,
            "copy": copy,
            "_assert_training_implementation_frozen": lambda _state: None,
            "_verify_final_population_audit_reference": lambda _state: recorded,
            "_build_final_population_audit": build,
            "output_vol": Volume(),
            "_load_state_snapshot": lambda _root: (state, _sha("state")),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                ),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        result = namespace["_audit_completed_population"](
            Path("/output/run"),
            state,
            _sha("state"),
        )
        self.assertTrue(result["already_audited"])
        self.assertEqual(calls["live"], 1)
        live_value["members"] = list(range(15))
        with self.assertRaisesRegex(RuntimeError, "differs"):
            namespace["_audit_completed_population"](
                Path("/output/run"),
                state,
                _sha("state"),
            )

    def test_final_audit_recomputes_every_gate_except_a1_including_d1(self):
        source_path = ROOT / "modal_role_lora_selfplay8_gate_retry.py"
        tree = ast.parse(source_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_final_population_audit"
        )
        labels = population_labels(8)
        stages = {}
        validations = {}
        for label in labels:
            digest = _sha(label)
            stage = {
                "sha256": digest,
                "source_sha256": digest,
            }
            if label != "A1":
                role = "attacker" if label.startswith("A") else "defender"
                budget = 100 if role == "attacker" else 200
                run_dir = f"/run/{label}"
                source_checkpoint = f"{run_dir}/ckpt/global_step50_hf"
                stage.update(
                    {
                        "run_dir": run_dir,
                        "source_checkpoint": source_checkpoint,
                        "actual_final_step": 50,
                        "requested_max_step": budget,
                        "stopped_early": True,
                    }
                )
                validations[f"{run_dir}/checkpoint_validation.json"] = {
                    "final_checkpoint": source_checkpoint,
                    "actual_final_step": 50,
                    "requested_max_step": budget,
                    "stopped_early": True,
                }
            stages[label] = stage
        state = {
            "run_suffix": "run",
            "config": {
                "rounds": 8,
                "attacker_max_steps": 100,
                "defender_max_steps": 200,
                "early_stop_threshold": 0.95,
                "early_stop_patience": 5,
                "early_stop_min_steps": 30,
                "defender_early_stop_min_steps": 32,
                "save_steps": 10,
                "training_implementation_sha256": {"frozen": _sha("frozen")},
            },
            "stages": stages,
        }
        retry_plan = _plan()
        stages["A2"]["gate_retry_plan_id"] = retry_plan["plan_id"]
        stages["A2"]["gate_retry_early_stop_min_steps"] = 1
        recomputed = []

        def recompute(_validation, *, role, expected_budget, **kwargs):
            recomputed.append(
                {
                    "role": role,
                    "expected_budget": expected_budget,
                    **kwargs,
                }
            )
            return {"passed": True, "role": role}

        captured = {}

        def final_state(_state, *, expected_labels, gate_proofs, **_kwargs):
            captured["labels"] = list(expected_labels)
            captured["proofs"] = copy.deepcopy(gate_proofs)
            return [
                {
                    "label": label,
                    "checkpoint": f"/population/{label}",
                    "sha256": _sha(label),
                }
                for label in expected_labels
            ]

        namespace = {
            "Any": object,
            "Path": Path,
            "re": __import__("re"),
            "RECOVERY_HISTORY_KEY": "gate_retry_recovery_history_v1",
            "RECOVERY_KEY": "gate_retry_recovery_v1",
            "population_labels": lambda _rounds: labels,
            "_read_json_object": lambda path: validations[str(path)],
            "validate_successful_gate": recompute,
            "file_sha256": lambda path: _sha(str(path)),
            "verify_recovery_history": lambda _history: [],
            "_released_retry_plan_for_stage": (
                lambda _state, label: retry_plan if label == "A2" else None
            ),
            "_audit_released_recovery_artifacts": lambda *_args: None,
            "validate_final_population_state": final_state,
            "checkpoint_weight_digest": lambda _path: "unused",
            "_strict_checkpoint": lambda _path, expected_sha256: {
                "tensor_count": 448,
                "rank": 64,
                "alpha": 64,
            },
            "_recovery_implementation_hashes": lambda: {"new": _sha("new")},
            "canonical_json_sha256": canonical_json_sha256,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                ),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        artifact = namespace["_build_final_population_audit"](
            Path("/population-root"),
            state,
        )
        self.assertEqual(len(recomputed), 15)
        self.assertEqual(set(captured["proofs"]), set(labels) - {"A1"})
        self.assertIn("D1", captured["proofs"])
        self.assertNotIn("A1", captured["proofs"])
        self.assertEqual(artifact["observed_checkpoint_count"], 16)
        a2 = recomputed[1]
        self.assertEqual(a2["role"], "attacker")
        self.assertEqual(a2["attacker_min_steps"], 1)
        self.assertTrue(a2["allow_single_checkpoint_change_proof"])
        self.assertEqual(
            a2["expected_initial_sha256"],
            retry_plan["original_nonqualifying_population"]["sha256"],
        )
        self.assertEqual(recomputed[0]["defender_min_steps"], 32)
        self.assertFalse(
            recomputed[0]["allow_single_checkpoint_change_proof"]
        )

    def test_canonical_hash_is_format_independent(self):
        first = {"b": [2, 1], "a": {"x": True}}
        second = {"a": {"x": True}, "b": [2, 1]}
        self.assertEqual(canonical_json_sha256(first), canonical_json_sha256(second))


if __name__ == "__main__":
    unittest.main()
