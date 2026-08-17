"""Pure contract helpers for one unchanged-framework self-play baseline repeat."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
POLICY = "unchanged-original-attacker-framework-baseline-repeat-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    _require(bool(_SHA256_RE.fullmatch(text)), f"Invalid SHA256 for {field}")
    return text


def build_a3_baseline_repeat_contract(
    *,
    state: Mapping[str, Any],
    root: Path,
    frozen_training_sha256: Mapping[str, str],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Bind one additional A3-vs-D2 run to the unchanged frozen recipe."""

    _require(state.get("schema_version") == 1, "Unsupported self-play state")
    _require(
        state.get("status") == "stage_target_not_reached"
        and state.get("active_stage") == "A3",
        "Baseline repeat requires the exhausted A3 stage",
    )
    stages = state.get("stages")
    _require(isinstance(stages, Mapping), "Self-play state has no stages")
    a3 = stages.get("A3")
    d2 = stages.get("D2")
    _require(isinstance(a3, Mapping), "Missing A3 stage")
    _require(isinstance(d2, Mapping), "Missing D2 stage")
    _require(
        a3.get("status") == "retained"
        and a3.get("transition_state") == "retained"
        and a3.get("stopped_early") is False,
        "A3 is not a retained budget-exhausted baseline candidate",
    )
    _require(
        d2.get("status") == "retained"
        and d2.get("transition_state") == "retained",
        "D2 is not retained",
    )
    config = state.get("config")
    _require(isinstance(config, Mapping), "Missing self-play config")
    _require(int(config.get("attacker_max_steps", -1)) == 100, "A budget drifted")
    _require(int(a3.get("actual_final_step", -1)) == 100, "A3 did not run 100 steps")
    _require(int(config.get("save_steps", -1)) == 10, "Save cadence drifted")
    _require(
        float(config.get("attacker_learning_rate", -1.0)) == 1e-5,
        "Attacker learning rate drifted",
    )
    _require(
        float(config.get("early_stop_threshold", -1.0)) == 0.95
        and int(config.get("early_stop_patience", -1)) == 5
        and int(config.get("early_stop_min_steps", -1)) == 30,
        "Attacker gate drifted",
    )

    run_suffix = str(state.get("run_suffix") or "")
    _require(bool(run_suffix), "Missing self-play run suffix")
    init_path = str(a3.get("population_checkpoint") or "")
    fixed_path = str(d2.get("population_checkpoint") or "")
    _require(init_path == str(root / "population" / "A3"), "A3 path drifted")
    _require(fixed_path == str(root / "population" / "D2"), "D2 path drifted")

    frozen = {str(k): _sha(v, f"frozen source {k}") for k, v in frozen_training_sha256.items()}
    implementation = {
        str(k): _sha(v, f"baseline implementation {k}")
        for k, v in implementation_sha256.items()
    }
    trainer_suffix = f"{run_suffix}_A3_baseline_repeat_original_001"
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "run_suffix": run_suffix,
        "stage_label": "A3",
        "attempt_number": 1,
        "trainer_run_suffix": trainer_suffix,
        "trainable_init": {
            "label": "A3",
            "checkpoint": init_path,
            "sha256": _sha(a3.get("sha256"), "A3"),
        },
        "fixed_opponent": {
            "label": "D2",
            "checkpoint": fixed_path,
            "sha256": _sha(d2.get("sha256"), "D2"),
        },
        "recipe": {
            "train_role": "attacker",
            "steps": 100,
            "rollout_batch_size": 128,
            "micro_rollout_batch_size": 8,
            "micro_train_batch_size": 8,
            "train_batch_size": 32,
            "save_steps": 10,
            "actor_learning_rate": 1e-5,
            "init_kl_coef": 0.0,
            "actor_lr_scheduler": "constant_with_warmup",
            "lr_warmup_ratio": 0.05,
            "enable_aux_sft": True,
            "postfill_cot_stop_after_step": 30,
            "role_specific_aux_sft": True,
            "v2_runtime": True,
            "v2_continuation_sft": True,
            "defender_sft_optimizer_slots_per_rollout": 0,
            "defender_raw_reinforce_advantages": False,
            "defender_reinforce_advantage_mode": "raw_no_center",
            "defender_reward_utility": "upstream_additive",
            "early_stop_threshold": 0.95,
            "early_stop_patience": 5,
            "early_stop_min_steps": 30,
            "lora_rank": 64,
            "lora_alpha": 64,
        },
        "frozen_training_implementation_sha256": frozen,
        "baseline_implementation_sha256": implementation,
        "canonical_population_mutation_allowed": False,
        "successor_dispatch_allowed": False,
    }
    contract["contract_id"] = canonical_json_sha256(contract)
    return contract


def verify_a3_baseline_repeat_contract(contract: Mapping[str, Any]) -> str:
    value = dict(contract)
    claimed = _sha(value.pop("contract_id", None), "contract id")
    _require(canonical_json_sha256(value) == claimed, "Contract hash drifted")
    _require(value.get("schema_version") == SCHEMA_VERSION, "Schema drifted")
    _require(value.get("policy") == POLICY, "Policy drifted")
    _require(value.get("stage_label") == "A3", "Stage drifted")
    _require(value.get("attempt_number") == 1, "Attempt drifted")
    _require(value.get("canonical_population_mutation_allowed") is False, "Population mutation enabled")
    _require(value.get("successor_dispatch_allowed") is False, "Successor dispatch enabled")
    recipe = value.get("recipe")
    _require(isinstance(recipe, Mapping), "Missing recipe")
    expected = {
        "steps": 100,
        "enable_aux_sft": True,
        "postfill_cot_stop_after_step": 30,
        "defender_raw_reinforce_advantages": False,
        "defender_reward_utility": "upstream_additive",
        "early_stop_threshold": 0.95,
        "early_stop_patience": 5,
        "early_stop_min_steps": 30,
    }
    for key, expected_value in expected.items():
        _require(recipe.get(key) == expected_value, f"Recipe drifted at {key}")
    return claimed
