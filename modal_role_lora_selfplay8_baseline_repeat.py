#!/usr/bin/env python3
"""Run one additional A3-vs-D2 baseline with the frozen A1/A2 recipe."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import modal


if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_role_lora_selfplay8 import (  # noqa: E402
    SELFPLAY_ROOT,
    _assert_trainer_manifest_implementation,
    _assert_training_implementation_frozen,
    _read_json_object,
    _strict_audit,
    _write_json_atomic,
    app,
    output_vol,
)
from modal_upstream_selfredteam_role_lora import (  # noqa: E402
    LLAMA_ABLITERATED_MODEL,
    _stable_wildguard_rm_url,
    train_upstream_attacker_lora_fixed_seed,
)
from role_lora_selfplay8 import read_checkpoint_validation  # noqa: E402
from roll.utils.selfplay_baseline_repeat import (  # noqa: E402
    build_a3_baseline_repeat_contract,
    verify_a3_baseline_repeat_contract,
)
from roll.utils.selfplay_gate_retry import file_sha256  # noqa: E402


_SAFE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_BASELINE_ROOT = "baseline_repeats_v1"


def _implementation_hashes() -> dict[str, str]:
    return {
        "modal_role_lora_selfplay8_baseline_repeat.py": file_sha256(
            Path(__file__).resolve()
        ),
        "roll/utils/selfplay_baseline_repeat.py": file_sha256(
            Path(sys.modules[build_a3_baseline_repeat_contract.__module__].__file__).resolve()
        ),
    }


def _write_exact_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json_object(path) != value:
            raise RuntimeError(f"Immutable baseline artifact drifted: {path}")
        return
    _write_json_atomic(path, value)


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def run_a3_original_framework_baseline_repeat(run_suffix: str) -> dict[str, Any]:
    """Call the frozen trainer unchanged and store an audit-only result."""

    if not _SAFE_SUFFIX_RE.fullmatch(run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state = _read_json_object(root / "state.json")
    frozen = _assert_training_implementation_frozen(state)
    implementation = _implementation_hashes()
    contract = build_a3_baseline_repeat_contract(
        state=state,
        root=root,
        frozen_training_sha256=frozen,
        implementation_sha256=implementation,
    )
    contract_id = verify_a3_baseline_repeat_contract(contract)
    attempt_root = root / _BASELINE_ROOT / "A3" / "attempt_001"
    contract_path = attempt_root / "contract.json"
    _write_exact_json(contract_path, contract)
    output_vol.commit()

    init_checkpoint = Path(contract["trainable_init"]["checkpoint"])
    fixed_checkpoint = Path(contract["fixed_opponent"]["checkpoint"])
    init_audit = _strict_audit(init_checkpoint)
    fixed_audit = _strict_audit(fixed_checkpoint)
    if init_audit.get("weight_sha256") != contract["trainable_init"]["sha256"]:
        raise RuntimeError("A3 baseline initializer digest drifted")
    if fixed_audit.get("weight_sha256") != contract["fixed_opponent"]["sha256"]:
        raise RuntimeError("D2 baseline opponent digest drifted")

    recipe = contract["recipe"]
    role_run_dir = train_upstream_attacker_lora_fixed_seed.remote(
        remote_rm_url=_stable_wildguard_rm_url(),
        steps=recipe["steps"],
        normal_prompt_mix=True,
        normal_prompt_pool_size=0,
        normal_prompt_pool_profile="balanced",
        rollout_batch_size=recipe["rollout_batch_size"],
        micro_rollout_batch_size=recipe["micro_rollout_batch_size"],
        micro_train_batch_size=recipe["micro_train_batch_size"],
        train_batch_size=recipe["train_batch_size"],
        save_steps=recipe["save_steps"],
        actor_learning_rate=recipe["actor_learning_rate"],
        init_kl_coef=recipe["init_kl_coef"],
        actor_lr_scheduler=recipe["actor_lr_scheduler"],
        lr_warmup_ratio=recipe["lr_warmup_ratio"],
        actor_lr_warmup_steps_override=None,
        enable_aux_sft=recipe["enable_aux_sft"],
        run_suffix=contract["trainer_run_suffix"],
        train_role="attacker",
        fixed_attacker_adapter="",
        fixed_defender_adapter=str(fixed_checkpoint),
        defender_prompt_profile="upstream",
        upstream_invalid_handling=True,
        base_model=LLAMA_ABLITERATED_MODEL,
        attacker_init_adapter=str(init_checkpoint),
        attacker_prompt_profile="optimized",
        strict_upstream_alignment=False,
        lora_rank=recipe["lora_rank"],
        lora_alpha=recipe["lora_alpha"],
        monitor_reference_kl=True,
        postfill_cot_stop_after_step=recipe["postfill_cot_stop_after_step"],
        role_specific_aux_sft=recipe["role_specific_aux_sft"],
        v2_runtime=recipe["v2_runtime"],
        v2_continuation_sft=recipe["v2_continuation_sft"],
        defender_sft_optimizer_slots_per_rollout=recipe[
            "defender_sft_optimizer_slots_per_rollout"
        ],
        defender_raw_reinforce_advantages=recipe[
            "defender_raw_reinforce_advantages"
        ],
        defender_reinforce_advantage_mode=recipe[
            "defender_reinforce_advantage_mode"
        ],
        defender_reward_utility=recipe["defender_reward_utility"],
        defender_prompt_pool_path="",
        defender_prompt_pool_sha256="",
        expected_implementation_sha256=frozen,
        early_stop_threshold=recipe["early_stop_threshold"],
        early_stop_patience=recipe["early_stop_patience"],
        early_stop_min_steps=recipe["early_stop_min_steps"],
    )

    output_vol.reload()
    run_dir = Path(str(role_run_dir))
    _assert_trainer_manifest_implementation(run_dir, frozen)
    manifest = _read_json_object(run_dir / "manifest.json")
    validation = read_checkpoint_validation(run_dir)
    source_checkpoint = Path(str(validation["final_checkpoint"]))
    source_audit = _strict_audit(source_checkpoint)
    expected_manifest = {
        "train_role": "attacker",
        "steps": 100,
        "aux_sft_enabled": True,
        "postfill_cot_stop_after_step": 30,
        "role_specific_aux_sft": True,
        "v2_runtime": True,
        "v2_continuation_sft": True,
        "defender_raw_reinforce_advantages": False,
        "defender_reward_utility": "upstream_additive",
        "lora_rank": 64,
        "lora_alpha": 64,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Baseline trainer manifest drifted at {key}")
    if manifest.get("initial_attacker_adapter") != str(init_checkpoint):
        raise RuntimeError("Baseline trainer initializer path drifted")
    if (
        manifest.get("initial_defender_adapter")
        != "/tmp/fixed_opponent_lora_compatible"
        or manifest.get("runtime_compatible_fixed_opponent_adapter")
        != "/tmp/fixed_opponent_lora_compatible"
    ):
        raise RuntimeError("Baseline trainer fixed opponent runtime drifted")

    result = {
        "schema_version": 1,
        "policy": contract["policy"],
        "contract_id": contract_id,
        "contract_path": str(contract_path),
        "run_dir": str(run_dir),
        "checkpoint_validation_path": str(run_dir / "checkpoint_validation.json"),
        "actual_final_step": int(validation["actual_final_step"]),
        "stopped_early": bool(validation.get("stopped_early")),
        "final_checkpoint": str(source_checkpoint),
        "final_sha256": source_audit["weight_sha256"],
        "strict_audit": source_audit,
        "canonical_population_mutated": False,
        "successor_dispatched": False,
    }
    _write_exact_json(attempt_root / "result.json", result)
    output_vol.commit()
    return result


@app.local_entrypoint(name="run_a3_original_framework_baseline_repeat")
def run_a3_original_framework_baseline_repeat_local(
    run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    invoke = (
        run_a3_original_framework_baseline_repeat.remote
        if wait_for_completion
        else run_a3_original_framework_baseline_repeat.spawn
    )
    result = invoke(run_suffix=run_suffix)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"A3_BASELINE_REPEAT_CALL_ID={result.object_id}", flush=True)
