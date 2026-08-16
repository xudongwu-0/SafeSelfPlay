"""Tests for the explicit seed-diversity-only A2 continuation policy."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from roll.utils import selfplay_training_continuation as contract


ROOT = Path(__file__).resolve().parents[1]


def _strict_audit(weight_sha256: str) -> dict:
    return {
        "weight_sha256": weight_sha256,
        "llama_v2_contract": {"passed": True},
    }


def _state() -> dict:
    return {
        "schema_version": 1,
        "run_suffix": contract.AUTHORIZED_RUN_SUFFIX,
        "status": "awaiting_d1_paired_gate",
        "active_stage": None,
        "config": {
            "rounds": 8,
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "training_implementation_sha256": copy.deepcopy(
                contract.EXPECTED_TRAINING_IMPLEMENTATION_SHA256
            ),
            "d1_data_contract": {
                "partition_sha256": contract.EXPECTED_PARTITION_SHA256,
                "training_prompt_pool_sha256": (
                    contract.EXPECTED_TRAINING_POOL_FILE_SHA256
                ),
            },
        },
        "stages": {
            "A1": {
                "status": "retained",
                "transition_state": "retained",
                "sha256": contract.EXPECTED_A1_SHA256,
                "strict_audit": _strict_audit(contract.EXPECTED_A1_SHA256),
            },
            "D1": {
                "status": "retained",
                "transition_state": "retained",
                "sha256": contract.EXPECTED_D1_SHA256,
                "strict_audit": _strict_audit(contract.EXPECTED_D1_SHA256),
                "stopped_early": True,
                "actual_final_step": 57,
            },
        },
    }


def _early_stop() -> dict:
    benign = [0.967741935483871, 0.984375, 0.9682539682539683,
              0.9682539682539683, 0.9523809523809523]
    history = []
    for index, step in enumerate(range(53, 58)):
        history.append(
            {
                "step": step,
                "value": 1.0,
                "metrics": {
                    "defender/wildguard_actual_harmful_count": 64,
                    "defender/wildguard_actual_benign_joint_success": benign[index],
                    "defender/wildguard_actual_benign_count": [62, 64, 63, 63, 63][index],
                },
                "qualified": True,
            }
        )
    return {
        "metric": "defender/wildguard_actual_harmful_joint_success",
        "threshold": 0.95,
        "patience": 5,
        "min_steps": 32,
        "last_step": 57,
        "streak": 5,
        "triggered": True,
        "history": history,
        "checkpoint_tag": "global_step57",
        "actual_final_step": 57,
    }


def _partition() -> dict:
    empty_sets = {
        f"{split}.{stratum}": set()
        for split in ("train", "dev", "final")
        for stratum in ("actual_harmful", "actual_benign")
    }
    return {
        "partition_sha256": contract.EXPECTED_PARTITION_SHA256,
        "_test_named_sets": empty_sets,
        "metadata": {
            "counts": {
                "train": {"actual_harmful": 32_452, "actual_benign": 2_376},
                "dev": {"actual_harmful": 512, "actual_benign": 512},
                "final": {"actual_harmful": 2_048, "actual_benign": 2_048},
            },
            "exclusions": {
                "actual_harmful": {
                    "source_unique": 50_050,
                    "source_duplicate_rows": 0,
                },
                "actual_benign": {
                    "source_unique": 20_000,
                    "source_duplicate_rows": 0,
                },
            },
        },
    }


def _pool_manifest() -> dict:
    return {
        "rows": 25_600,
        "occurrences_per_stratum": {
            "actual_harmful": 12_800,
            "actual_benign": 12_800,
        },
        "unique_seed_prompts_used": {
            "actual_harmful": 12_800,
            "actual_benign": 2_376,
        },
        "repeated_occurrences": {
            "actual_harmful": 0,
            "actual_benign": 10_424,
        },
        "direct_benign_bypasses_attacker": True,
        "expected_data_parallel_ranks": 4,
        "shuffle_allowed": False,
    }


def _ppo_registry() -> dict:
    return {
        "registry_sha256": contract.EXPECTED_PPO_REGISTRY_SHA256,
        "group_counts": {
            "ppo.actual_harmful_concrete_request": 3_648,
            "ppo.direct_benign_request": 3_648,
        },
        "provenance": {
            "drop_counts": {
                "actual_harmful": {
                    "accepted": 3_648,
                    "parse": 0,
                    "label_mismatch": 0,
                },
                "actual_benign": {
                    "accepted": 3_550,
                    "parse": 0,
                    "label_mismatch": 98,
                },
            }
        },
    }


class UserContinuationContractTests(unittest.TestCase):
    def test_d1_stop_is_five_consecutive_zero_attack_success_steps(self):
        early_stop = _early_stop()
        result = contract.audit_d1_training_stop(
            state=_state(),
            checkpoint_validation={
                "stopped_early": True,
                "actual_final_step": 57,
                "early_stop": early_stop,
            },
            early_stop=early_stop,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["tail_steps"], [53, 54, 55, 56, 57])
        self.assertEqual(
            result["tail_rollout_weight_versions"], [52, 53, 54, 55, 56]
        )
        self.assertEqual(result["tail_attacker_success_rate"], [0.0] * 5)
        self.assertEqual(result["qualification"], "at_or_below")
        self.assertFalse(
            result["selected_saved_checkpoint_directly_evaluated_by_tail"]
        )
        self.assertEqual(result["selected_saved_checkpoint"], "global_step57")

    def test_d1_stop_rejects_one_nonqualifying_or_post_threshold_row(self):
        for mutation in ("qualified", "value"):
            with self.subTest(mutation=mutation):
                early_stop = _early_stop()
                if mutation == "qualified":
                    early_stop["history"][-1]["qualified"] = False
                else:
                    early_stop["history"][-1]["value"] = 0.94
                with self.assertRaises(RuntimeError):
                    contract.audit_d1_training_stop(
                        state=_state(),
                        checkpoint_validation={
                            "stopped_early": True,
                            "actual_final_step": 57,
                            "early_stop": early_stop,
                        },
                        early_stop=early_stop,
                    )

    @mock.patch.object(contract, "validate_d1_exposure_registry")
    @mock.patch.object(contract, "validate_d1_training_prompt_pool")
    @mock.patch.object(contract, "validate_d1_canonical_partitions")
    def test_seed_distribution_records_real_diversity_and_reuse(
        self,
        validate_partition,
        validate_pool,
        validate_registry,
    ):
        partition = _partition()
        validate_partition.return_value = partition["_test_named_sets"]
        validate_pool.return_value = {
            "jsonl_sha256": contract.EXPECTED_TRAINING_POOL_FILE_SHA256,
            "manifest_sha256": "manifest-logical-sha",
        }
        validate_registry.return_value = set()
        result = contract.audit_seed_prompt_distribution(
            state=_state(),
            partition=partition,
            training_pool_rows=[],
            training_pool_manifest=_pool_manifest(),
            ppo_registry=_ppo_registry(),
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["semantic_policy_generalization_claimed"])
        self.assertEqual(
            result["actual_d1_exposure"]["unique_seed_prompts"],
            {"actual_harmful": 3_648, "actual_benign": 2_376},
        )
        disclosure = result["future_stage_reuse_disclosure"]
        self.assertTrue(
            disclosure["D2_through_D8_reuse_same_25600_occurrence_pool_order"]
        )
        self.assertFalse(disclosure["cross_stage_freshness_claimed"])

    @mock.patch.object(contract, "validate_d1_exposure_registry")
    @mock.patch.object(contract, "validate_d1_training_prompt_pool")
    @mock.patch.object(contract, "validate_d1_canonical_partitions")
    def test_seed_distribution_rejects_hidden_benign_repeat_drift(
        self,
        validate_partition,
        validate_pool,
        validate_registry,
    ):
        partition = _partition()
        validate_partition.return_value = partition["_test_named_sets"]
        validate_pool.return_value = {
            "jsonl_sha256": contract.EXPECTED_TRAINING_POOL_FILE_SHA256,
            "manifest_sha256": "manifest-logical-sha",
        }
        validate_registry.return_value = set()
        manifest = _pool_manifest()
        manifest["repeated_occurrences"]["actual_benign"] -= 1
        with self.assertRaisesRegex(RuntimeError, "repetition"):
            contract.audit_seed_prompt_distribution(
                state=_state(),
                partition=partition,
                training_pool_rows=[],
                training_pool_manifest=manifest,
                ppo_registry=_ppo_registry(),
            )

    def test_authorization_is_nonpaired_hot_start_and_requires_16_members(self):
        seed_audit = {"passed": True}
        stop_audit = {"passed": True}
        approval = contract.build_authorization_payload(
            run_suffix=contract.AUTHORIZED_RUN_SUFFIX,
            seed_prompt_audit=seed_audit,
            d1_training_stop=stop_audit,
        )
        policy = approval["promotion_policy"]
        self.assertTrue(policy["paired_gate_canceled_as_promotion_requirement"])
        self.assertFalse(policy["heldout_policy_generalization_required"])
        self.assertFalse(policy["semantic_policy_generalization_claimed"])
        schedule = approval["hot_start_schedule"]
        self.assertEqual(schedule[0], {"train": "A2", "init": "A1", "opponent": "D1"})
        self.assertEqual(schedule[1], {"train": "D2", "init": "D1", "opponent": "A2"})
        self.assertEqual(schedule[-1], {"train": "D8", "init": "D7", "opponent": "A8"})
        population = approval["population_contract"]
        self.assertEqual(population["required_checkpoint_count"], 16)
        self.assertEqual(
            population["required_labels"],
            [label for i in range(1, 9) for label in (f"A{i}", f"D{i}")],
        )
        stop_contract = approval["early_stop_contract"]
        self.assertFalse(stop_contract["direct_metric_checkpoint_equivalence_claimed"])
        self.assertIn("W(N-1)", stop_contract["frozen_checkpoint_timing"])
        digest_payload = dict(approval)
        digest = digest_payload.pop("authorization_sha256")
        self.assertEqual(digest, contract.canonical_json_sha256(digest_payload))

    def test_frozen_training_sources_still_match_d1_state(self):
        for relative, expected in contract.EXPECTED_TRAINING_IMPLEMENTATION_SHA256.items():
            observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)

    def test_modal_override_is_explicit_CAS_without_paired_forgery(self):
        path = ROOT / "modal_role_lora_selfplay8_user_continuation.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("EXPECTED_INITIAL_STATE_SHA256", source)
        self.assertIn("deterministic_stage_spawn_claim", source)
        self.assertIn("EXPECTED_A2_SPAWN_CLAIM_ID", source)
        self.assertIn("_dispatch_stage_claim", source)
        self.assertNotIn('state["d1_paired_promotion"]', source)
        entrypoints = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertIn("approve_seed_prompt_policy_and_resume_a2", entrypoints)
        self.assertIn("resume_a2_after_user_seed_diversity_policy", entrypoints)

    def test_modal_override_adds_roll_before_importing_sibling_modules(self):
        path = ROOT / "modal_role_lora_selfplay8_user_continuation.py"
        source = path.read_text(encoding="utf-8")
        roll_path_index = source.index('sys.path.insert(0, "/roll")')
        coordinator_import_index = source.index("from modal_role_lora_selfplay8 import")
        helper_import_index = source.index(
            "from roll.utils.selfplay_training_continuation import"
        )
        self.assertLess(roll_path_index, coordinator_import_index)
        self.assertLess(roll_path_index, helper_import_index)


if __name__ == "__main__":
    unittest.main()
