"""Auditable operator policy for continuing the frozen eight-round run.

This module deliberately lives outside the training implementation hash list.
It does not change PPO, rewards, model routing, or checkpoint handling.  It
only verifies the already-persisted D1 training and seed-prompt evidence before
an explicit operator decision releases A2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from roll.utils.upstream_v2_payoff import (
    D1_ACTUAL_BENIGN,
    D1_ACTUAL_HARMFUL,
    validate_d1_canonical_partitions,
    validate_d1_exposure_registry,
    validate_d1_training_prompt_pool,
)


AUTHORIZATION_POLICY_VERSION = "seed_prompt_distribution_only_v1"
AUTHORIZED_RUN_SUFFIX = "selfplay8_v3_joint_20260817_013808"
EXPECTED_INITIAL_STATE_SHA256 = (
    "48551de4fe371b853abc6602e84773d817638cf309ebfd43a137b7d04d62fc84"
)
EXPECTED_A1_SHA256 = (
    "2fa9969621fe06e5cb99689835733bbecf254d1874ec2a45de9a444de07aa35c"
)
EXPECTED_D1_SHA256 = (
    "d388528b9b37d8e82eef18797d72c7f49cff1b17ad5f3e4bc7e123c60e496c15"
)
EXPECTED_PARTITION_FILE_SHA256 = (
    "41529acd390bb2d02ab29134419fbe2e6d90edc09128cbf8c6fbfb01191effc4"
)
EXPECTED_PARTITION_SHA256 = (
    "32bbf9b905ff9766320a5ecb7556aed7fb6ffbad0813ae5e749c73016fa4ca23"
)
EXPECTED_TRAINING_POOL_FILE_SHA256 = (
    "bb01562af4ea59e5ab28cd7a1c825833aae113a103f5c73498510d3718bccd56"
)
EXPECTED_TRAINING_POOL_MANIFEST_FILE_SHA256 = (
    "9e374d5469f89f3470b1ba6e1324fd4754d27b066a2273086e8b28a5e19bd5d4"
)
EXPECTED_PPO_REGISTRY_FILE_SHA256 = (
    "74a834765cf44d364f51cd8c7dd604850c1691ae44cfe0ba302a72e9f6895d60"
)
EXPECTED_PPO_REGISTRY_SHA256 = (
    "a491b4848d65b0b5238da323e4ae56923c2b6b5383ecf4da49a3e8d512d1234e"
)
EXPECTED_GAME_LOG_SHA256 = (
    "da2cabbc5c72fdd9b2bc916b800a55dea30629d3f6d8d4647a3ae3fa3642ec4f"
)
EXPECTED_EARLY_STOP_FILE_SHA256 = (
    "69f5fc39ce3c2fd6b866abff6b093fe9aabaebd682ca41fab132716706d79b85"
)
EXPECTED_CHECKPOINT_VALIDATION_FILE_SHA256 = (
    "1e4017012ccef6333b4871b1bf08d6b294e9a49fd879c8da0fbbaf16f0a34df6"
)
EXPECTED_CONFIG_SHA256 = (
    "41740cf10e941d6e722703803cceefaf6f3abbf2021f9fd1b7fe4c00309e9802"
)
EXPECTED_SCHEDULE_SHA256 = (
    "e4c37cce1e2f34353d95776a7181397f504ad906afa9979a1c4bd3ad33dc3bc2"
)

EXPECTED_TRAINING_IMPLEMENTATION_SHA256 = {
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


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Return the stable digest used to bind the authorization artifact."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_seed_prompt_distribution(
    *,
    state: Mapping[str, Any],
    partition: Mapping[str, Any],
    training_pool_rows: Sequence[Mapping[str, Any]],
    training_pool_manifest: Mapping[str, Any],
    ppo_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify prompt diversity/routing without claiming policy generalization."""

    config = state.get("config")
    _require(isinstance(config, Mapping), "Self-play config is missing")
    contract = config.get("d1_data_contract")
    _require(isinstance(contract, Mapping), "D1 data contract is missing")
    _require(int(config.get("rounds", 0)) == 8, "Run is not configured for 8 rounds")
    _require(
        contract.get("partition_sha256") == EXPECTED_PARTITION_SHA256,
        "State partition SHA drifted",
    )
    _require(
        contract.get("training_prompt_pool_sha256")
        == EXPECTED_TRAINING_POOL_FILE_SHA256,
        "State training-pool SHA drifted",
    )

    named_sets = validate_d1_canonical_partitions(partition)
    pool_verification = validate_d1_training_prompt_pool(
        training_pool_rows,
        training_pool_manifest,
        partition,
    )
    validate_d1_exposure_registry(ppo_registry)

    counts = partition["metadata"]["counts"]
    exclusions = partition["metadata"]["exclusions"]
    train_h = int(counts["train"][D1_ACTUAL_HARMFUL])
    train_b = int(counts["train"][D1_ACTUAL_BENIGN])
    source_h = int(exclusions[D1_ACTUAL_HARMFUL]["source_unique"])
    source_b = int(exclusions[D1_ACTUAL_BENIGN]["source_unique"])
    source_h_duplicates = int(
        exclusions[D1_ACTUAL_HARMFUL]["source_duplicate_rows"]
    )
    source_b_duplicates = int(
        exclusions[D1_ACTUAL_BENIGN]["source_duplicate_rows"]
    )
    unique_used = training_pool_manifest["unique_seed_prompts_used"]
    repeated = training_pool_manifest["repeated_occurrences"]
    occurrences = training_pool_manifest["occurrences_per_stratum"]
    ppo_groups = ppo_registry.get("group_counts")
    provenance = ppo_registry.get("provenance")
    _require(isinstance(ppo_groups, Mapping), "PPO registry group counts are missing")
    _require(isinstance(provenance, Mapping), "PPO registry provenance is missing")

    exact_named_sets = {
        "train.actual_harmful",
        "train.actual_benign",
        "dev.actual_harmful",
        "dev.actual_benign",
        "final.actual_harmful",
        "final.actual_benign",
    }
    _require(set(named_sets) == exact_named_sets, "Canonical split set drifted")
    _require(source_h == 50_050 and source_b == 20_000, "Source coverage drifted")
    _require(
        source_h_duplicates == 0 and source_b_duplicates == 0,
        "Seed source contains duplicate canonical prompts",
    )
    _require(train_h == 32_452 and train_b == 2_376, "Train pool coverage drifted")
    _require(
        int(training_pool_manifest.get("rows", 0)) == 25_600,
        "Training occurrence count drifted",
    )
    _require(
        int(occurrences[D1_ACTUAL_HARMFUL]) == 12_800
        and int(occurrences[D1_ACTUAL_BENIGN]) == 12_800,
        "Training pool is no longer exact 50/50 H/B",
    )
    _require(
        int(unique_used[D1_ACTUAL_HARMFUL]) == 12_800
        and int(unique_used[D1_ACTUAL_BENIGN]) == 2_376,
        "Training-pool unique seed coverage drifted",
    )
    _require(
        int(repeated[D1_ACTUAL_HARMFUL]) == 0
        and int(repeated[D1_ACTUAL_BENIGN]) == 10_424,
        "Explicit seed repetition accounting drifted",
    )
    _require(
        training_pool_manifest.get("direct_benign_bypasses_attacker") is True
        and training_pool_manifest.get("expected_data_parallel_ranks") == 4
        and training_pool_manifest.get("shuffle_allowed") is False,
        "H/B routing or four-rank deterministic pool contract drifted",
    )
    _require(
        ppo_registry.get("registry_sha256") == EXPECTED_PPO_REGISTRY_SHA256,
        "PPO registry logical SHA drifted",
    )
    _require(
        int(ppo_groups.get("ppo.actual_harmful_concrete_request", -1)) == 3_648
        and int(ppo_groups.get("ppo.direct_benign_request", -1)) == 3_648,
        "Actual D1 H/B exposure counts drifted",
    )
    _require(
        int(provenance["drop_counts"][D1_ACTUAL_HARMFUL]["accepted"]) == 3_648
        and int(provenance["drop_counts"][D1_ACTUAL_HARMFUL]["label_mismatch"])
        == 0
        and int(provenance["drop_counts"][D1_ACTUAL_BENIGN]["accepted"]) == 3_550
        and int(provenance["drop_counts"][D1_ACTUAL_BENIGN]["label_mismatch"])
        == 98,
        "Actual D1 WildGuard routing counts drifted",
    )

    return {
        "passed": True,
        "scope": "seed_prompt_distribution_and_routing_only",
        "heldout_policy_generalization_required": False,
        "semantic_policy_generalization_claimed": False,
        "canonical_sources": {
            "actual_harmful_unique": source_h,
            "actual_benign_unique": source_b,
            "source_duplicate_rows": 0,
        },
        "canonical_train_unique": {
            "actual_harmful": train_h,
            "actual_benign": train_b,
        },
        "actual_d1_exposure": {
            "rollouts": 57,
            "candidates": {"actual_harmful": 3_648, "actual_benign": 3_648},
            "unique_seed_prompts": {
                "actual_harmful": 3_648,
                "actual_benign": 2_376,
            },
            "actual_harmful_repeat_occurrences": 0,
            "actual_benign_repeat_epoch_1_occurrences": 1_272,
            "actual_benign_all_train_seeds_covered": True,
            "actual_harmful_concrete_requests_unique": 3_648,
            "actual_harmful_wildguard_harmful_accepted": 3_648,
            "direct_benign_unharmful_accepted": 3_550,
            "direct_benign_label_mismatch_dropped": 98,
        },
        "future_stage_reuse_disclosure": {
            "D2_through_D8_reuse_same_25600_occurrence_pool_order": True,
            "actual_harmful_unique_per_full_pool": 12_800,
            "actual_benign_unique_per_full_pool": 2_376,
            "cross_stage_freshness_claimed": False,
        },
        "partition_verification": {
            "six_split_stratum_sets_pairwise_disjoint": True,
            "partition_sha256": partition["partition_sha256"],
            "training_pool_sha256": pool_verification["jsonl_sha256"],
            "training_pool_manifest_sha256": pool_verification[
                "manifest_sha256"
            ],
            "ppo_registry_sha256": ppo_registry["registry_sha256"],
        },
    }


def audit_d1_training_stop(
    *,
    state: Mapping[str, Any],
    checkpoint_validation: Mapping[str, Any],
    early_stop: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the five-step training stop used by the operator decision."""

    _require(state.get("run_suffix") == AUTHORIZED_RUN_SUFFIX, "Run suffix drifted")
    _require(
        state.get("status") == "awaiting_d1_paired_gate",
        "Initial state is not awaiting the D1 decision",
    )
    config = state.get("config")
    stages = state.get("stages")
    _require(isinstance(config, Mapping), "Self-play config is missing")
    _require(isinstance(stages, Mapping), "Self-play stages are missing")
    _require(config.get("training_implementation_sha256") == EXPECTED_TRAINING_IMPLEMENTATION_SHA256,
             "Frozen training implementation drifted")
    _require(int(config.get("rounds", 0)) == 8, "Run is not eight rounds")
    _require(float(config.get("early_stop_threshold", 0.0)) == 0.95,
             "Early-stop threshold drifted")
    _require(int(config.get("early_stop_patience", 0)) == 5,
             "Early-stop patience drifted")
    _require("A2" not in stages, "A2 already exists in the initial decision state")
    for label, expected_sha in (("A1", EXPECTED_A1_SHA256), ("D1", EXPECTED_D1_SHA256)):
        stage = stages.get(label)
        _require(isinstance(stage, Mapping), f"{label} stage is missing")
        _require(stage.get("status") == "retained", f"{label} is not retained")
        _require(stage.get("transition_state") == "retained", f"{label} transition drifted")
        _require(stage.get("sha256") == expected_sha, f"{label} weight SHA drifted")
        strict = stage.get("strict_audit")
        _require(isinstance(strict, Mapping), f"{label} strict audit is missing")
        _require(strict.get("weight_sha256") == expected_sha, f"{label} strict audit SHA drifted")
        _require(
            isinstance(strict.get("llama_v2_contract"), Mapping)
            and strict["llama_v2_contract"].get("passed") is True,
            f"{label} strict LoRA contract did not pass",
        )

    d1 = stages["D1"]
    _require(d1.get("stopped_early") is True, "D1 did not stop early")
    _require(int(d1.get("actual_final_step", 0)) == 57, "D1 final step drifted")
    _require(checkpoint_validation.get("stopped_early") is True,
             "Checkpoint validation does not record early stop")
    _require(int(checkpoint_validation.get("actual_final_step", 0)) == 57,
             "Checkpoint validation final step drifted")
    _require(checkpoint_validation.get("early_stop") == dict(early_stop),
             "Checkpoint validation and early_stop.json differ")
    _require(
        early_stop.get("metric")
        == "defender/wildguard_actual_harmful_joint_success",
        "D1 stop metric drifted",
    )
    _require(float(early_stop.get("threshold", 0.0)) == 0.95,
             "D1 stop threshold drifted")
    _require(int(early_stop.get("patience", 0)) == 5,
             "D1 stop patience drifted")
    _require(early_stop.get("triggered") is True and int(early_stop.get("streak", 0)) == 5,
             "D1 five-step stop was not triggered")
    _require(early_stop.get("checkpoint_tag") == "global_step57",
             "D1 trigger checkpoint drifted")
    history = early_stop.get("history")
    _require(isinstance(history, list) and len(history) >= 5,
             "D1 stop history is incomplete")
    tail = history[-5:]
    _require([int(row["step"]) for row in tail] == [53, 54, 55, 56, 57],
             "D1 stop tail is not the expected consecutive prefix")
    _require(all(row.get("qualified") is True for row in tail),
             "D1 stop tail contains an unqualified row")
    harmful_joint = [float(row["value"]) for row in tail]
    attacker_success = [1.0 - value for value in harmful_joint]
    _require(all(value >= 0.95 for value in harmful_joint),
             "D1 harmful joint success fell below 95%")
    _require(all(value <= 0.05 for value in attacker_success),
             "D1 attacker success exceeded 5%")

    return {
        "passed": True,
        "metric": "defender/actual_harmful_attack_success_rate",
        "qualification": "at_or_below",
        "upper_bound": 0.05,
        "patience": 5,
        "tail_steps": [53, 54, 55, 56, 57],
        "tail_rollout_weight_versions": [52, 53, 54, 55, 56],
        "tail_attacker_success_rate": attacker_success,
        "selected_saved_checkpoint": "global_step57",
        "selected_saved_checkpoint_weight_version": 57,
        "selected_saved_checkpoint_sha256": EXPECTED_D1_SHA256,
        "selected_saved_checkpoint_directly_evaluated_by_tail": False,
        "frozen_runtime_timing_caveat": (
            "rollout step N evaluates W(N-1), then optimizer update N creates "
            "and saves WN; the five-step tail evaluates W52-W56 and selects "
            "the post-update W57 checkpoint without directly evaluating W57"
        ),
    }


def build_authorization_payload(
    *,
    run_suffix: str,
    seed_prompt_audit: Mapping[str, Any],
    d1_training_stop: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic, non-paired continuation decision."""

    _require(run_suffix == AUTHORIZED_RUN_SUFFIX, "Unauthorized run suffix")
    _require(seed_prompt_audit.get("passed") is True, "Seed prompt audit failed")
    _require(d1_training_stop.get("passed") is True, "D1 training stop audit failed")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "promotion_policy": {
            "version": AUTHORIZATION_POLICY_VERSION,
            "decision": "GO",
            "release": "A2_then_selfplay_rounds_2_through_8",
            "required": (
                "broad deterministic seed/request environment with verified "
                "frozen-A-H and direct-B routing"
            ),
            "heldout_policy_generalization_required": False,
            "semantic_policy_generalization_claimed": False,
            "paired_gate_canceled_as_promotion_requirement": True,
            "scope": "self_play_training_continuation_only",
        },
        "run_suffix": run_suffix,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "a1_weight_sha256": EXPECTED_A1_SHA256,
        "d1_weight_sha256": EXPECTED_D1_SHA256,
        "bound_artifacts": {
            "canonical_partition_file_sha256": EXPECTED_PARTITION_FILE_SHA256,
            "canonical_partition_logical_sha256": EXPECTED_PARTITION_SHA256,
            "training_prompt_pool_file_sha256": EXPECTED_TRAINING_POOL_FILE_SHA256,
            "training_prompt_pool_manifest_file_sha256": (
                EXPECTED_TRAINING_POOL_MANIFEST_FILE_SHA256
            ),
            "ppo_exposure_registry_file_sha256": EXPECTED_PPO_REGISTRY_FILE_SHA256,
            "ppo_exposure_registry_logical_sha256": EXPECTED_PPO_REGISTRY_SHA256,
            "training_game_log_file_sha256": EXPECTED_GAME_LOG_SHA256,
            "early_stop_file_sha256": EXPECTED_EARLY_STOP_FILE_SHA256,
            "checkpoint_validation_file_sha256": (
                EXPECTED_CHECKPOINT_VALIDATION_FILE_SHA256
            ),
        },
        "seed_prompt_audit": dict(seed_prompt_audit),
        "d1_training_stop": dict(d1_training_stop),
        "hot_start_schedule": [
            stage
            for index in range(2, 9)
            for stage in (
                {
                    "train": f"A{index}",
                    "init": f"A{index - 1}",
                    "opponent": f"D{index - 1}",
                },
                {
                    "train": f"D{index}",
                    "init": f"D{index - 1}",
                    "opponent": f"A{index}",
                },
            )
        ],
        "early_stop_contract": {
            "consecutive_steps": 5,
            "attacker_stage": "attacker success rate >= 0.95",
            "defender_stage": "attacker success rate <= 0.05",
            "frozen_runtime_implementation_note": (
                "D stages enforce the equivalent H joint success >= 0.95 and "
                "also retain the stricter direct-B joint-success companion"
            ),
            "frozen_checkpoint_timing": (
                "For every A2-D8 stage, rollout N evaluates W(N-1); update N "
                "then saves WN. The final checkpoint is the post-update "
                "checkpoint selected by the fifth qualifying rollout, not a "
                "checkpoint directly evaluated by that rollout streak."
            ),
            "direct_metric_checkpoint_equivalence_claimed": False,
        },
        "population_contract": {
            "required_labels": [
                label
                for index in range(1, 9)
                for label in (f"A{index}", f"D{index}")
            ],
            "required_checkpoint_count": 16,
            "one_immutable_final_checkpoint_per_label": True,
            "overwrite_forbidden": True,
        },
    }
    payload["authorization_sha256"] = canonical_json_sha256(payload)
    return payload
