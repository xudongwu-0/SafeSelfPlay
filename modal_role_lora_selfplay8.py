#!/usr/bin/env python3
"""Durable A1/D1 -> A8/D8 role-LoRA self-play on Modal.

The successful A1 is treated as an immutable seed.  Each later policy inherits
its own role's preceding adapter and trains against the newest frozen opponent:
``D1(A1), A2(A1,D1), D2(D1,A2), ...``.  One CPU coordinator call owns one GPU
stage and spawns the next, keeping every call well below Modal's 24-hour limit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_upstream_selfredteam_role_lora import (
    DEFENDER_V2_BENIGN_SOURCE_FILENAME,
    DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT,
    DEFENDER_V2_WARMUP_OPTIMIZER_STEPS,
    LLAMA_ABLITERATED_MODEL,
    OUTPUT_ROOT,
    _stable_wildguard_rm_url,
    app,
    audit_role_lora_checkpoint,
    defender_v2_interim_gate_configuration,
    output_vol,
    train_upstream_attacker_lora_fixed_seed,
)
from role_lora_selfplay8 import (
    acknowledge_stage_child_started,
    atomic_copy_population_checkpoint,
    authorize_stage_trainer_recovery,
    build_selfplay8_schedule,
    ensure_stage_spawn_pending,
    evaluate_d1_gate,
    mark_stage_transition_retained,
    population_labels,
    prune_stage_hf_checkpoints,
    read_checkpoint_validation,
    record_stage_spawn_observation,
    verify_d1_paired_evidence_contract,
)
from roll.utils.upstream_v2_payoff import (
    assemble_valid_actual_paired_prefix,
    build_sft_disjoint_benign_pool,
    evaluate_d1_actual_paired_promotion,
    summarize_actual_d1_paired_gate,
)


SUCCESSFUL_A1_CHECKPOINT = (
    "/output/upstream_selfredteam_role_lora/"
    "upstream_selfredteam_v2repro_meta_llama_31_8b_instruct_abliterated_"
    "attacker_lora_r64_a64_fromBase_vs_base_prompt_optimized_normalmix_"
    "s100_rb128_mb8_tb32_lr1e-5_kl0_warm0.05_const_auxsft_"
    "upstreaminvalid_lora_v2_repaired_20260816_1435/ckpt/"
    "global_step100_hf"
)
SELFPLAY_ROOT = Path(OUTPUT_ROOT) / "selfplay8"
PAIRED_GATE_ROOT = Path(OUTPUT_ROOT) / "paired_d1_gate_v2"
PAIRED_GATE_HELDOUT_SEED_BASE = 18888
PAIRED_GATE_MIN_ACCEPTED_PAIRS = 1024
PAIRED_GATE_MAX_PARSE_DROP_RATE = 0.05


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object artifact: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"Expected object in {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSONL artifact: {path}") from error
    return rows


def _independently_verify_actual_gate_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    prompt_prelabel_calibration_response: str,
) -> dict[str, Any]:
    """Recompute actual strata, A1 bypass, and pair-drop decisions."""

    if not candidate_rows or len(candidate_rows) % 2:
        raise RuntimeError("Actual-gate candidate stream must be nonempty/even")
    if not prompt_prelabel_calibration_response:
        raise RuntimeError("Prompt prelabel calibration response is empty")
    drops = {"overall": 0, "harmful": 0, "benign": 0}
    denominators = {
        "overall": len(candidate_rows),
        "harmful": len(candidate_rows) // 2,
        "benign": len(candidate_rows) // 2,
    }
    for index, row in enumerate(candidate_rows):
        subgroup = "harmful" if index % 2 == 0 else "benign"
        expected = {
            "evaluation_stratum": f"actual_{subgroup}",
            "prompt_origin": (
                "a1_generated_harmful"
                if subgroup == "harmful"
                else "direct_heldout_benign"
            ),
            "prompt_type": (
                "generated_harmful" if subgroup == "harmful" else "direct_benign"
            ),
            "expected_actual_prompt_harmfulness": (
                "harmful" if subgroup == "harmful" else "unharmful"
            ),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise RuntimeError(
                    f"Actual-gate candidate {index} has invalid {key}"
                )
        request = str(row.get("request") or "")
        if hashlib.sha256(request.encode()).hexdigest() != row.get(
            "request_sha256"
        ):
            raise RuntimeError(f"Actual-gate request hash drifted at {index}")
        expected_prelabel_query_hash = hashlib.sha256(
            json.dumps(
                {
                    "prompt": request,
                    "response": prompt_prelabel_calibration_response,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if row.get("prompt_prelabel_query_sha256") != (
            expected_prelabel_query_hash
        ):
            raise RuntimeError(
                f"Prompt prelabel query hash drifted at candidate {index}"
            )
        if subgroup == "benign":
            if request != str(row.get("seed_prompt") or ""):
                raise RuntimeError(f"Direct benign request changed at {index}")
            if any(
                row.get(key) not in (None, "")
                for key in (
                    "attacker_prompt_sha256",
                    "attacker_decoded_completion",
                    "attacker_vllm_raw_text",
                    "attacker_output_token_ids_sha256",
                    "attacker_tokenized_prompt_ids_sha256",
                    "attacker_rendered_prompt_token_count",
                    "attacker_tokenized_prompt_token_count",
                    "attacker_prompt_truncated",
                    "attack",
                    "attacker_cot_format_violation",
                )
            ):
                raise RuntimeError(f"Direct benign candidate used A1 at {index}")
        else:
            if request != str(row.get("attack") or ""):
                raise RuntimeError(f"Actual-H request/attack differs at {index}")
            for key in (
                "attacker_prompt_sha256",
                "attacker_decoded_completion",
                "attacker_vllm_raw_text",
                "attacker_output_token_ids_sha256",
                "attacker_tokenized_prompt_ids_sha256",
                "attack",
            ):
                if row.get(key) is None:
                    raise RuntimeError(
                        f"Actual-H candidate lacks A1 artifact {key} at {index}"
                    )
        try:
            prelabel = row["prompt_prelabel"]
            base_arm = row["base_arm"]
            d1_arm = row["d1_arm"]
            actual_label = prelabel.get("prompt_harmfulness")
            base_label = base_arm["wildguard"].get("prompt_harmfulness")
            d1_label = d1_arm["wildguard"].get("prompt_harmfulness")
        except (KeyError, TypeError, AttributeError) as error:
            raise RuntimeError(
                f"Missing raw WildGuard labels at candidate {index}"
            ) from error
        if row.get("actual_prompt_harmfulness") != actual_label:
            raise RuntimeError(f"Stored actual/prelabel differs at {index}")
        prelabel_parse = bool(prelabel.get("is_parsing_error", False))
        stratum_mismatch = actual_label != expected[
            "expected_actual_prompt_harmfulness"
        ]
        arm_parse = bool(
            base_arm.get("dropped_reason")
            or d1_arm.get("dropped_reason")
            or base_arm["wildguard"].get("is_parsing_error", False)
            or d1_arm["wildguard"].get("is_parsing_error", False)
        )
        arm_drift = base_label != actual_label or d1_label != actual_label
        expected_reason = (
            "prompt_prelabel_parse_error"
            if prelabel_parse
            else "actual_prompt_stratum_mismatch"
            if stratum_mismatch
            else "defender_arm_wildguard_parse_error"
            if arm_parse
            else "defender_arm_prompt_label_drift"
            if arm_drift
            else None
        )
        if row.get("dropped_reason") != expected_reason:
            raise RuntimeError(f"Pair-drop decision drifted at candidate {index}")
        if expected_reason:
            drops["overall"] += 1
            drops[subgroup] += 1
            if any(
                reward_key in arm
                for arm in (base_arm, d1_arm)
                for reward_key in (
                    "attacker_raw_reward",
                    "defender_raw_reward",
                )
            ):
                raise RuntimeError(
                    "Dropped actual-gate pair was scored before pair-drop at "
                    f"candidate {index}"
                )
        elif subgroup == "benign" and any(
            "attacker_raw_reward" in arm for arm in (base_arm, d1_arm)
        ):
            raise RuntimeError(f"Direct benign pair has attacker reward at {index}")
    return {
        "drop_counts": drops,
        "drop_rates": {
            subgroup: drops[subgroup] / denominators[subgroup]
            for subgroup in ("overall", "harmful", "benign")
        },
        "actual_strata": {"harmful": denominators["harmful"], "benign": denominators["benign"]},
        "direct_benign_bypasses_a1": True,
        "policy": "prelabel/actual-stratified pair-drop before scoring",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError(f"Cannot hash artifact: {path}") from error
    return digest.hexdigest()


def _current_paired_implementation_hashes() -> dict[str, str]:
    import inspect

    helper_source = inspect.getsourcefile(summarize_actual_d1_paired_gate)
    if not helper_source:
        raise RuntimeError("Cannot resolve paired payoff helper source")
    modal_source = Path(__file__).resolve().with_name(
        "modal_upstream_v2_payoff.py"
    )
    if not modal_source.is_file():
        modal_source = Path("/roll/modal_upstream_v2_payoff.py")
    core_source = Path(__file__).resolve().with_name(
        "modal_upstream_selfredteam_role_lora.py"
    )
    if not core_source.is_file():
        core_source = Path("/roll/modal_upstream_selfredteam_role_lora.py")
    if not core_source.is_file():
        raise RuntimeError("Cannot resolve role-LoRA core source")
    sources = {
        "modal_upstream_v2_payoff.py": modal_source,
        "modal_upstream_selfredteam_role_lora.py": core_source,
        "roll/utils/upstream_v2_payoff.py": Path(helper_source).resolve(),
    }
    return {label: _sha256_file(path) for label, path in sources.items()}


def _current_training_implementation_hashes() -> dict[str, str]:
    runtime_dir = Path(__file__).resolve().parent
    sources: dict[str, Path] = {}
    for filename in (
        "modal_role_lora_selfplay8.py",
        "modal_upstream_selfredteam_role_lora.py",
        "role_lora_selfplay8.py",
        "modal_upstream_selfredteam_fixed_seed.py",
        "roll/utils/lora_sync_contract.py",
        "roll/third_party/vllm/worker.py",
        "roll/third_party/deepspeed/model_update.py",
    ):
        candidates = (runtime_dir / filename, Path("/roll") / filename)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise RuntimeError(
                f"Cannot resolve frozen training implementation source: {filename}"
            )
        sources[filename] = path
    return {label: _sha256_file(path) for label, path in sources.items()}


def _assert_training_implementation_frozen(
    state: dict[str, Any],
) -> dict[str, str]:
    recorded = state.get("config", {}).get("training_implementation_sha256")
    current = _current_training_implementation_hashes()
    if not isinstance(recorded, dict) or recorded != current:
        raise RuntimeError(
            "Self-play training implementation changed or is unbound; use a "
            "fresh run_suffix and do not resume old D1 checkpoints. "
            f"recorded={recorded!r}, current={current!r}"
        )
    return current


def _assert_trainer_manifest_implementation(
    run_dir: Path,
    expected: dict[str, str],
) -> None:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Missing or invalid trainer implementation manifest: {manifest_path}"
        ) from error
    recorded = manifest.get("expected_implementation_sha256")
    actual_core = manifest.get("implementation_sha256")
    expected_core = expected.get("modal_upstream_selfredteam_role_lora.py")
    if recorded != expected or actual_core != expected_core:
        raise RuntimeError(
            "Trainer manifest implementation mismatch: "
            f"recorded={recorded!r}, actual_core={actual_core!r}, "
            f"expected={expected!r}"
        )


def _load_state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid self-play state: {path}") from error
    if state.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported self-play state schema: {path}")
    return state


def _population_path(root: Path, label: str) -> Path:
    return root / "population" / label


def _strict_audit(checkpoint: Path) -> dict[str, Any]:
    audit = audit_role_lora_checkpoint.remote(
        checkpoint_path=str(checkpoint),
        expected_tensor_count=448,
        require_llama_v2_contract=True,
    )
    contract = audit.get("llama_v2_contract")
    if not isinstance(contract, dict) or contract.get("passed") is not True:
        raise RuntimeError(f"Strict LoRA audit did not pass: {checkpoint}")
    return audit


def _record_stage_failure(
    root: Path,
    state: dict[str, Any],
    stage_label: str,
    error: BaseException,
) -> None:
    # A successor claim may already have been committed by _dispatch before a
    # submission/call-id exception reached the outer stage handler.  Reload
    # first so an older caller snapshot can never erase that recoverable claim.
    output_vol.reload()
    try:
        durable_state = _load_state(root)
    except RuntimeError:
        durable_state = state
    pending = [
        label
        for label, stage in (durable_state.get("stages") or {}).items()
        if isinstance(stage, dict)
        and stage.get("transition_state") == "spawn_pending"
    ]
    if pending:
        durable_state["status"] = "spawn_pending_recovery"
        durable_state["active_stage"] = pending[0]
        durable_state["last_dispatch_error"] = {
            "origin_stage": stage_label,
            "pending_stages": pending,
            "type": type(error).__name__,
            "message": str(error),
            "call_id_authoritative": False,
        }
        _write_json_atomic(root / "state.json", durable_state)
        output_vol.commit()
        return
    current_stage = (durable_state.get("stages") or {}).get(stage_label)
    if (
        isinstance(current_stage, dict)
        and current_stage.get("transition_state") == "child_started"
        and current_stage.get("deterministic_trainer_run_suffix")
    ):
        durable_state["status"] = "child_started_recovery"
        durable_state["active_stage"] = stage_label
        durable_state["last_trainer_owner_loss"] = {
            "stage": stage_label,
            "spawn_claim_id": current_stage.get("spawn_claim_id"),
            "deterministic_trainer_run_suffix": current_stage.get(
                "deterministic_trainer_run_suffix"
            ),
            "type": type(error).__name__,
            "message": str(error),
            "recovery": (
                "serialized inner trainer resumes the same checkpoint set"
            ),
        }
        _write_json_atomic(root / "state.json", durable_state)
        output_vol.commit()
        return
    state["status"] = "failed"
    state["active_stage"] = stage_label
    state["failure"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    _write_json_atomic(root / "state.json", state)
    output_vol.commit()


def _persist_state(root: Path, state: dict[str, Any]) -> None:
    _write_json_atomic(root / "state.json", state)
    output_vol.commit()


def _complete_retained_stage_pruning(
    root: Path,
    state: dict[str, Any],
    *,
    stage_label: str,
) -> dict[str, Any]:
    """Idempotently finish pruning after a retained-before-prune crash."""

    stage = (state.get("stages") or {}).get(stage_label)
    if not isinstance(stage, dict) or stage.get("transition_state") != "retained":
        raise RuntimeError(f"Cannot prune non-retained stage: {stage_label}")
    if "pruned_stage_hf_checkpoints" in stage:
        return state
    run_dir_text = str(stage.get("run_dir") or "")
    population_text = str(stage.get("population_checkpoint") or "")
    run_dir = Path(run_dir_text)
    population_checkpoint = Path(population_text)
    digest = str(stage.get("sha256") or "")
    if not run_dir_text or not population_text or not digest:
        raise RuntimeError(
            f"Retained stage lacks pruning provenance: {stage_label}"
        )
    removed = prune_stage_hf_checkpoints(
        run_dir / "ckpt",
        audited_population_checkpoint=population_checkpoint,
        audited_sha256=digest,
    )
    stage["pruned_stage_hf_checkpoints"] = removed
    stage["pruning_reconciled_after_retention"] = True
    _persist_state(root, state)
    return state


def _transition_resume_block_reason(state: dict[str, Any]) -> str | None:
    """Return a fail-closed gate reason, allowing a persisted D1 approval."""

    status = state.get("status")
    if status == "awaiting_d1_paired_gate":
        promotion = state.get("d1_paired_promotion")
        d1 = (state.get("stages") or {}).get("D1")
        release = d1.get("successor_release") if isinstance(d1, dict) else None
        approved = (
            isinstance(promotion, dict)
            and (promotion.get("promotion") or {}).get("passed") is True
            and isinstance(release, dict)
            and release.get("approved") is True
        )
        return None if approved else "awaiting_d1_paired_gate"
    if status in {
        "d1_training_diagnostic_failed",
        "d1_paired_gate_failed",
        "stage_target_not_reached",
        "failed",
        "completed",
    }:
        return str(status)
    return None


def _spawn_retained_stage_reconciler(
    state: dict[str, Any],
    *,
    run_suffix: str,
    stage: Any,
) -> dict[str, Any]:
    """Re-enter one retained stage only to prune/finish/release successor."""

    stage_state = (state.get("stages") or {}).get(stage.label)
    if (
        not isinstance(stage_state, dict)
        or stage_state.get("transition_state") != "retained"
        or not stage_state.get("spawn_claim_id")
    ):
        raise RuntimeError(
            f"Retained stage has no deterministic reconcile claim: {stage.label}"
        )
    call = train_role_lora_selfplay8_stage.spawn(
        run_suffix=run_suffix,
        stage_index=stage.index,
        spawn_claim_id=stage_state["spawn_claim_id"],
    )
    return {
        "state": state,
        "spawned": True,
        "call_id": call.object_id,
        "spawn_claim_id": stage_state["spawn_claim_id"],
        "reconcile_only": True,
        "reconcile_stage": stage.label,
    }


def _dispatch_stage_claim(
    root: Path,
    state: dict[str, Any],
    *,
    run_suffix: str,
    stage: Any,
    retry_existing_pending: bool = False,
) -> dict[str, Any]:
    """Persist one deterministic claim, then submit it at least once.

    The durable decision is the transition state plus ``spawn_claim_id``.
    Normal predecessor completion submits a newly created successor once;
    only an explicit recovery path sets ``retry_existing_pending`` and repeats
    an un-ACKed pending side effect.  Modal call ids are observational only.
    """

    stage_preexisted = stage.label in (state.get("stages") or {})
    plan = ensure_stage_spawn_pending(state, stage)
    state = plan["state"]
    should_submit = bool(
        plan["should_spawn"]
        and (retry_existing_pending or not stage_preexisted)
    )
    if plan["transition_state"] != "retained":
        state["status"] = "running"
        state["active_stage"] = plan["stage_label"]
    if should_submit:
        # Record the attempt before the side effect.  A crash before spawn or
        # before child ACK leaves spawn_pending and is deliberately retriable.
        state = record_stage_spawn_observation(
            state,
            stage_label=plan["stage_label"],
            spawn_claim_id=plan["spawn_claim_id"],
            call_id=None,
        )
    _persist_state(root, state)

    call_id = None
    dispatch_error = None
    if should_submit:
        try:
            call = train_role_lora_selfplay8_stage.spawn(
                run_suffix=run_suffix,
                stage_index=int(stage.index),
                spawn_claim_id=plan["spawn_claim_id"],
            )
            call_id = call.object_id
        except BaseException as error:
            # The claim was committed before spawn.  Preserve it as recoverable
            # even when submission throws or the call id cannot be observed.
            output_vol.reload()
            try:
                latest_state = _load_state(root)
            except RuntimeError:
                latest_state = state
            latest_stage = (latest_state.get("stages") or {}).get(
                plan["stage_label"]
            )
            if (
                not isinstance(latest_stage, dict)
                or latest_stage.get("spawn_claim_id")
                != plan["spawn_claim_id"]
            ):
                raise RuntimeError(
                    "Durable stage claim changed while spawn outcome was unknown"
                ) from error
            if latest_stage.get("transition_state") == "spawn_pending":
                latest_state["status"] = "spawn_pending_recovery"
                latest_state["active_stage"] = plan["stage_label"]
            latest_state["last_dispatch_error"] = {
                "stage": plan["stage_label"],
                "spawn_claim_id": plan["spawn_claim_id"],
                "type": type(error).__name__,
                "message": str(error),
                "call_id_authoritative": False,
            }
            _persist_state(root, latest_state)
            state = latest_state
            dispatch_error = state["last_dispatch_error"]
    return {
        "state": state,
        "stage_label": plan["stage_label"],
        "spawn_claim_id": plan["spawn_claim_id"],
        "transition_state": plan["transition_state"],
        "spawned": bool(should_submit and dispatch_error is None),
        "spawn_attempted": should_submit,
        "call_id": call_id,
        "dispatch_error": dispatch_error,
    }


def _finish_retained_stage(
    root: Path,
    state: dict[str, Any],
    *,
    run_suffix: str,
    schedule: list[Any],
    stage_index: int,
) -> dict[str, Any]:
    """Apply the stage gate once, then durably release one successor."""

    stage = schedule[stage_index - 1]
    label = stage.label
    stage_state = state.get("stages", {}).get(label)
    if (
        not isinstance(stage_state, dict)
        or stage_state.get("transition_state") != "retained"
    ):
        raise RuntimeError(f"Cannot finish non-retained stage: {label}")
    run_dir = Path(str(stage_state.get("run_dir") or ""))
    validation = _read_json_object(run_dir / "checkpoint_validation.json")
    config = state["config"]

    if label == "D1":
        diagnostic = evaluate_d1_gate(
            validation,
            threshold=float(config["early_stop_threshold"]),
            patience=int(config["early_stop_patience"]),
            min_improvement=float(config["d1_min_improvement"]),
        )
        state["d1_training_diagnostic"] = diagnostic
        stage_state["training_diagnostic"] = diagnostic
        paired = state.get("d1_paired_promotion")
        if not isinstance(paired, dict) or (
            (paired.get("promotion") or {}).get("passed") is not True
        ):
            stage_state["successor_release"] = {
                "approved": False,
                "basis": (
                    "awaiting authoritative actual-H/direct-heldout-B paired "
                    "promotion; seed-bucketed training diagnostic is retained "
                    "for observability only"
                ),
            }
            state["status"] = "awaiting_d1_paired_gate"
            state["active_stage"] = None
            _persist_state(root, state)
            return {"state": state, "spawned": False, "call_id": None}
        stage_state["successor_release"] = {
            "approved": True,
            "basis": "verified held-out paired D1 promotion",
            "evidence_sha256": paired.get("evidence_sha256"),
        }
    elif not validation.get("stopped_early"):
        stage_state["successor_release"] = {
            "approved": False,
            "basis": "empirical point-estimate early-stop target not reached",
        }
        state["status"] = "stage_target_not_reached"
        state["active_stage"] = label
        _persist_state(root, state)
        return {"state": state, "spawned": False, "call_id": None}
    else:
        stage_state["successor_release"] = {
            "approved": True,
            "basis": (
                "five consecutive empirical point estimates at or above "
                "the configured 0.95 success threshold"
            ),
            "inference": "point_estimate_not_confidence_bound",
        }

    next_index = stage_index + 1
    if next_index > len(schedule):
        state["status"] = "completed"
        state["active_stage"] = None
        state["completed_population"] = population_labels(
            int(config["rounds"])
        )
        _persist_state(root, state)
        return {"state": state, "spawned": False, "call_id": None}

    # Persist the predecessor's release before creating the successor claim.
    _persist_state(root, state)
    return _dispatch_stage_claim(
        root,
        state,
        run_suffix=run_suffix,
        stage=schedule[next_index - 1],
    )


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def initialize_role_lora_selfplay8(
    run_suffix: str,
    a1_checkpoint: str = SUCCESSFUL_A1_CHECKPOINT,
    rounds: int = 8,
    attacker_max_steps: int = 100,
    defender_max_steps: int = 200,
    save_steps: int = 10,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 1e-5,
    early_stop_threshold: float = 0.95,
    early_stop_patience: int = 5,
    early_stop_min_steps: int = 30,
    d1_min_improvement: float = 0.02,
    defender_sft_stop_after_step: int = 30,
) -> dict[str, Any]:
    """Archive A1 without pruning it, then start the D1-gated chain."""
    if not run_suffix:
        raise ValueError("run_suffix is required for durable resume identity")
    if defender_sft_stop_after_step != 30:
        raise ValueError("defender_sft_stop_after_step is frozen at 30")
    if attacker_learning_rate != 1e-5 or defender_learning_rate != 1e-5:
        raise ValueError(
            "Self-play v2 freezes both role learning rates at 1e-5"
        )
    training_implementation_sha256 = _current_training_implementation_hashes()
    schedule = build_selfplay8_schedule(rounds)
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state_path = root / "state.json"
    if state_path.is_file():
        state = _load_state(root)
        if state.get("config", {}).get("a1_checkpoint") != a1_checkpoint:
            raise RuntimeError("Existing self-play state uses a different A1")
        _assert_training_implementation_frozen(state)
    else:
        state = {
            "schema_version": 1,
            "method": "sequential inherited role-LoRA best responses",
            "run_suffix": run_suffix,
            "status": "initializing_A1",
            "base_model": LLAMA_ABLITERATED_MODEL,
            "population_order": population_labels(rounds),
            "schedule": [stage.to_dict() for stage in schedule],
            "config": {
                "a1_checkpoint": a1_checkpoint,
                "rounds": rounds,
                "attacker_max_steps": attacker_max_steps,
                "defender_max_steps": defender_max_steps,
                "save_steps": save_steps,
                "attacker_learning_rate": attacker_learning_rate,
                "defender_learning_rate": defender_learning_rate,
                "early_stop_threshold": early_stop_threshold,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_steps": early_stop_min_steps,
                "early_stop_statistical_interpretation": (
                    "five consecutive empirical success-rate point estimates; "
                    "not a confidence-bound guarantee"
                ),
                "d1_min_improvement": d1_min_improvement,
                "defender_sft_stop_after_step": defender_sft_stop_after_step,
                "defender_sft_optimizer_slots_per_rollout": (
                    DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT
                ),
                "defender_lr_warmup_optimizer_steps": (
                    DEFENDER_V2_WARMUP_OPTIMIZER_STEPS
                ),
                "defender_advantage_transform": (
                    "raw_reinforce_no_center_no_scale; absolute negative "
                    "game rewards remain negative PPO targets"
                ),
                "defender_sft_recipe": (
                    "balanced exact-rollout continuation SFT remains active "
                    f"through step {defender_sft_stop_after_step} for every "
                    "fresh D stage; its final effect is first observed by "
                    "rollout step 31"
                ),
                "defender_v2_interim_gate": (
                    defender_v2_interim_gate_configuration()
                ),
                "role_optimizer_recipe": (
                    "rank64/alpha64, lr1e-5, constant-with-warmup; A keeps "
                    "the recovered 5% schedule, D uses 20 real optimizer "
                    "warmup steps and four fixed SFT optimizer slots per "
                    "rollout; KL0, native LoRA A/B synchronization"
                ),
                "trainer_recovery_contract": (
                    "globally serialized trainer max_containers=1; every stage "
                    "retry resumes the same deterministic run suffix"
                ),
                "fixed_opponent_generation": (
                    "100% generated harmful and benign prompts"
                ),
                "checkpoint_retention": (
                    "population A1-A8/D1-D8 only; A1 source never pruned"
                ),
                "training_implementation_sha256": (
                    training_implementation_sha256
                ),
                "implementation_freeze_contract": (
                    "coordinator, trainer, fixed-seed patcher, pure state, and "
                    "LoRA sync/runtime sources are hash-bound at initialization "
                    "and revalidated before every stage"
                ),
            },
            "stages": {},
        }
        _write_json_atomic(state_path, state)
        output_vol.commit()

    source_a1 = Path(a1_checkpoint)
    source_audit = _strict_audit(source_a1)
    promoted = atomic_copy_population_checkpoint(
        source_a1, root / "population", "A1"
    )
    output_vol.commit()
    population_audit = _strict_audit(Path(promoted["path"]))
    if population_audit["weight_sha256"] != source_audit["weight_sha256"]:
        raise RuntimeError("A1 population copy differs from its audited source")
    state["stages"]["A1"] = {
        "status": "retained",
        "transition_state": "retained",
        "role": "attacker",
        "source_checkpoint": a1_checkpoint,
        "population_checkpoint": promoted["path"],
        "actual_final_step": 100,
        "sha256": promoted["sha256"],
        "source_pruned": False,
        "strict_audit": population_audit,
    }
    dispatch = _dispatch_stage_claim(
        root,
        state,
        run_suffix=run_suffix,
        stage=schedule[0],
        retry_existing_pending=True,
    )
    return {
        "root": str(root),
        "state": dispatch["state"],
        "spawned": dispatch["spawned"],
        "call_id": dispatch["call_id"],
        "spawn_claim_id": dispatch["spawn_claim_id"],
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_role_lora_selfplay8_stage(
    run_suffix: str,
    stage_index: int,
    spawn_claim_id: str,
) -> dict[str, Any]:
    """Train, promote, audit, prune, gate, and chain exactly one role stage."""
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state = _load_state(root)
    training_implementation_sha256 = _assert_training_implementation_frozen(
        state
    )
    config = state["config"]
    schedule = build_selfplay8_schedule(int(config["rounds"]))
    if not 1 <= stage_index <= len(schedule):
        raise ValueError(f"stage_index out of range: {stage_index}")
    stage = schedule[stage_index - 1]
    label = stage.label
    population_checkpoint = _population_path(root, label)

    try:
        # A claim is ACKed durably before audits, generation, or training.
        # Because both this coordinator and the trainer are serialized, a
        # later child_started invocation is the safe recovery owner and may
        # resume only the exact deterministic trainer suffix.
        acknowledgement = acknowledge_stage_child_started(
            state,
            stage_label=label,
            spawn_claim_id=spawn_claim_id,
        )
        state = acknowledgement["state"]
        _persist_state(root, state)
        recovering_trainer = False
        trainer_suffix = f"{run_suffix}_{label}"
        if not acknowledgement["should_train"]:
            if acknowledgement["transition_state"] == "retained":
                state = _complete_retained_stage_pruning(
                    root,
                    state,
                    stage_label=label,
                )
                finished = _finish_retained_stage(
                    root,
                    state,
                    run_suffix=run_suffix,
                    schedule=schedule,
                    stage_index=stage_index,
                )
                return {
                    "root": str(root),
                    "state": finished["state"],
                    "spawned": finished["spawned"],
                    "call_id": finished["call_id"],
                    "duplicate_child": True,
                }
            if not population_checkpoint.is_dir():
                recovery = authorize_stage_trainer_recovery(
                    state,
                    stage_label=label,
                    spawn_claim_id=spawn_claim_id,
                    deterministic_trainer_run_suffix=trainer_suffix,
                    serialized_trainer=True,
                )
                state = recovery["state"]
                trainer_suffix = recovery[
                    "deterministic_trainer_run_suffix"
                ]
                recovering_trainer = True
                _persist_state(root, state)

        trainable_init = (
            ""
            if stage.trainable_parent == "base"
            else str(_population_path(root, stage.trainable_parent))
        )
        fixed_opponent = str(_population_path(root, stage.fixed_opponent))
        if not Path(fixed_opponent).is_dir():
            raise RuntimeError(
                f"Missing fixed opponent {stage.fixed_opponent}: "
                f"{fixed_opponent}"
            )
        if trainable_init and not Path(trainable_init).is_dir():
            raise RuntimeError(
                f"Missing trainable parent {stage.trainable_parent}: "
                f"{trainable_init}"
            )

        stage_state = state["stages"][label]
        stage_state.update(
            {
                "work_status": (
                    "reconciling_population"
                    if population_checkpoint.is_dir()
                    else "resuming_training_after_ACK_owner_loss"
                    if recovering_trainer
                    else "training"
                ),
                "trainable_init": trainable_init or LLAMA_ABLITERATED_MODEL,
                "fixed_opponent_checkpoint": fixed_opponent,
                "deterministic_trainer_run_suffix": trainer_suffix,
                "trainer_execution_serialization": (
                    "train_upstream_attacker_lora_fixed_seed:max_containers=1"
                ),
                "training_implementation_sha256": (
                    training_implementation_sha256
                ),
            }
        )
        state["status"] = "running"
        state["active_stage"] = label
        _persist_state(root, state)

        is_attacker = stage.role == "attacker"
        if population_checkpoint.is_dir():
            # A prior attempt may have copied and committed the population
            # checkpoint before the retained transition was written.  Only
            # reconcile that artifact; never invoke training again here.
            role_run_dir = str(stage_state.get("run_dir") or "")
            if not role_run_dir:
                raise RuntimeError(
                    f"Cannot reconcile {label} population without run_dir"
                )
            run_dir = Path(role_run_dir)
            _assert_trainer_manifest_implementation(
                run_dir,
                training_implementation_sha256,
            )
            validation = read_checkpoint_validation(run_dir)
            source_checkpoint = Path(validation["final_checkpoint"])
            population_audit = _strict_audit(population_checkpoint)
            expected_source_sha = str(stage_state.get("source_sha256") or "")
            if expected_source_sha and (
                population_audit["weight_sha256"] != expected_source_sha
            ):
                raise RuntimeError(
                    f"Reconciled population/source digest mismatch for {label}"
                )
            promoted = {
                "path": str(population_checkpoint),
                "sha256": population_audit["weight_sha256"],
            }
        else:
            try:
                role_run_dir = train_upstream_attacker_lora_fixed_seed.remote(
                    remote_rm_url=_stable_wildguard_rm_url(),
                    steps=int(
                        config[
                            "attacker_max_steps"
                            if is_attacker
                            else "defender_max_steps"
                        ]
                    ),
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
                        None
                        if is_attacker
                        else DEFENDER_V2_WARMUP_OPTIMIZER_STEPS
                    ),
                    enable_aux_sft=True,
                    run_suffix=trainer_suffix,
                    train_role=stage.role,
                    fixed_attacker_adapter=(
                        fixed_opponent if stage.role == "defender" else ""
                    ),
                    fixed_defender_adapter=(
                        fixed_opponent if stage.role == "attacker" else ""
                    ),
                    defender_prompt_profile="upstream",
                    upstream_invalid_handling=True,
                    base_model=LLAMA_ABLITERATED_MODEL,
                    # Despite the historical name, this is the trainable
                    # role's inherited initializer for both A and D.
                    attacker_init_adapter=trainable_init,
                    attacker_prompt_profile="optimized",
                    strict_upstream_alignment=False,
                    lora_rank=64,
                    lora_alpha=64,
                    monitor_reference_kl=True,
                    postfill_cot_stop_after_step=(
                        30
                        if is_attacker
                        else int(
                            config.get("defender_sft_stop_after_step", 30)
                        )
                    ),
                    role_specific_aux_sft=True,
                    v2_runtime=True,
                    v2_continuation_sft=True,
                    defender_sft_optimizer_slots_per_rollout=(
                        0
                        if is_attacker
                        else DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT
                    ),
                    defender_raw_reinforce_advantages=(not is_attacker),
                    expected_implementation_sha256=(
                        training_implementation_sha256
                    ),
                    early_stop_threshold=float(
                        config["early_stop_threshold"]
                    ),
                    early_stop_patience=int(config["early_stop_patience"]),
                    early_stop_min_steps=int(config["early_stop_min_steps"]),
                )
            except BaseException as error:
                # The nested call may still be running after a transport or
                # parent-owner failure.  Do not guess from its call id and do
                # not mark the stage terminal: the real inner trainer is
                # max_containers=1 and a recovery resumes this exact suffix.
                _record_stage_failure(root, state, label, error)
                output_vol.reload()
                recovery_state = _load_state(root)
                if recovery_state.get("status") != "child_started_recovery":
                    raise
                return {
                    "root": str(root),
                    "state": recovery_state,
                    "spawned": False,
                    "call_id": None,
                    "trainer_outcome_unknown": True,
                    "deterministic_trainer_run_suffix": trainer_suffix,
                }
            output_vol.reload()
            # Merge into the ACKed state after the remote call; the fixed
            # suffix makes a retried trainer resume the same checkpoint set.
            state = _load_state(root)
            _assert_training_implementation_frozen(state)
            stage_state = state["stages"][label]
            if (
                stage_state.get("transition_state") != "child_started"
                or stage_state.get("spawn_claim_id") != spawn_claim_id
            ):
                raise RuntimeError(f"Stage ownership changed during {label}")
            run_dir = Path(role_run_dir)
            _assert_trainer_manifest_implementation(
                run_dir,
                training_implementation_sha256,
            )
            validation = read_checkpoint_validation(run_dir)
            source_checkpoint = Path(validation["final_checkpoint"])
            source_audit = _strict_audit(source_checkpoint)
            # Persist recovery metadata before population copy.  If the call
            # dies after the copy, a retry can audit/reconcile without training.
            stage_state.update(
                {
                    "work_status": "promoting",
                    "run_dir": str(run_dir),
                    "source_checkpoint": str(source_checkpoint),
                    "source_sha256": source_audit["weight_sha256"],
                    "actual_final_step": int(validation["actual_final_step"]),
                    "requested_max_step": int(
                        config[
                            "attacker_max_steps"
                            if is_attacker
                            else "defender_max_steps"
                        ]
                    ),
                    "stopped_early": bool(validation.get("stopped_early")),
                }
            )
            _persist_state(root, state)
            promoted = atomic_copy_population_checkpoint(
                source_checkpoint, root / "population", label
            )
            output_vol.commit()
            population_audit = _strict_audit(Path(promoted["path"]))
            if (
                population_audit["weight_sha256"]
                != source_audit["weight_sha256"]
                or population_audit["weight_sha256"] != promoted["sha256"]
            ):
                raise RuntimeError(
                    f"Strict population audit digest mismatch for {label}"
                )

        retained_payload = {
            "work_status": "retained",
            "run_dir": str(run_dir),
            "source_checkpoint": str(source_checkpoint),
            "population_checkpoint": promoted["path"],
            "actual_final_step": int(validation["actual_final_step"]),
            "requested_max_step": int(
                config[
                    "attacker_max_steps"
                    if is_attacker
                    else "defender_max_steps"
                ]
            ),
            "stopped_early": bool(validation.get("stopped_early")),
            "sha256": promoted["sha256"],
            "strict_audit": population_audit,
        }
        state = mark_stage_transition_retained(
            state,
            stage_label=label,
            spawn_claim_id=spawn_claim_id,
            retained_payload=retained_payload,
        )
        # Retained is committed before destructive pruning or successor spawn.
        _persist_state(root, state)
        removed = prune_stage_hf_checkpoints(
            run_dir / "ckpt",
            audited_population_checkpoint=Path(promoted["path"]),
            audited_sha256=promoted["sha256"],
        )
        state["stages"][label]["pruned_stage_hf_checkpoints"] = removed
        _persist_state(root, state)

        finished = _finish_retained_stage(
            root,
            state,
            run_suffix=run_suffix,
            schedule=schedule,
            stage_index=stage_index,
        )
        return {
            "root": str(root),
            "state": finished["state"],
            "spawned": finished["spawned"],
            "call_id": finished["call_id"],
        }
    except BaseException as error:
        _record_stage_failure(root, state, label, error)
        raise


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def approve_d1_paired_gate_and_resume_a2(
    run_suffix: str,
    paired_run_suffix: str,
) -> dict[str, Any]:
    """Verify one completed paired artifact and idempotently request A2.

    No caller-provided pass/fail value is accepted.  The coordinator reloads
    and recomputes the paired evidence, verifies strict population audits and
    hashes, applies the fixed promotion contract, durably records its decision,
    and only then schedules stage index 2.
    """

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", paired_run_suffix or ""):
        raise ValueError("paired_run_suffix must be one safe path component")

    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state = _load_state(root)
    _assert_training_implementation_frozen(state)
    existing = state.get("d1_paired_promotion")
    if isinstance(existing, dict):
        if existing.get("paired_run_suffix") != paired_run_suffix:
            raise RuntimeError(
                "A different paired D1 artifact is already recorded"
            )
        if (existing.get("promotion") or {}).get("passed") is not True:
            raise RuntimeError("Recorded D1 paired promotion is not approved")
        schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
        if len(schedule) < 2:
            raise RuntimeError("Configured self-play has no A2 stage")
        # A crash after the durable pending claim but before child ACK is
        # retried at least once with the same claim.  Once ACKed/retained,
        # ensure_stage_spawn_pending returns should_spawn=False.
        dispatch = _dispatch_stage_claim(
            root,
            state,
            run_suffix=run_suffix,
            stage=schedule[1],
            retry_existing_pending=True,
        )
        return {
            "root": str(root),
            "state": dispatch["state"],
            "already_recorded": True,
            "spawned": dispatch["spawned"],
            "call_id": dispatch["call_id"],
            "spawn_claim_id": dispatch["spawn_claim_id"],
        }
    if state.get("status") != "awaiting_d1_paired_gate":
        raise RuntimeError(
            "Paired promotion is allowed only from awaiting_d1_paired_gate"
        )
    diagnostic = state.get("d1_training_diagnostic")
    if not isinstance(diagnostic, dict):
        raise RuntimeError("D1 training diagnostic is missing")
    if diagnostic.get("authoritative_for_promotion") is not False:
        raise RuntimeError(
            "D1 training diagnostic must be marked non-authoritative"
        )

    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("Self-play state has no stages mapping")
    a1_state = stages.get("A1")
    d1_state = stages.get("D1")
    if not isinstance(a1_state, dict) or not isinstance(d1_state, dict):
        raise RuntimeError("Self-play state does not retain A1 and D1")
    a1_checkpoint = Path(str(a1_state.get("population_checkpoint") or ""))
    d1_checkpoint = Path(str(d1_state.get("population_checkpoint") or ""))
    a1_audit = _strict_audit(a1_checkpoint)
    d1_audit = _strict_audit(d1_checkpoint)

    evidence_root = PAIRED_GATE_ROOT / paired_run_suffix
    artifact_paths = {
        "manifest.json": evidence_root / "manifest.json",
        "candidate_pairs.jsonl": evidence_root / "candidate_pairs.jsonl",
        "paired_episodes.jsonl": evidence_root / "paired_episodes.jsonl",
        "paired_summary.json": evidence_root / "paired_summary.json",
        "heldout_benign_pool.jsonl": (
            evidence_root / "heldout_benign_pool.jsonl"
        ),
    }
    status_path = evidence_root / "run_status.json"
    manifest = _read_json_object(artifact_paths["manifest.json"])
    summary = _read_json_object(artifact_paths["paired_summary.json"])
    status = _read_json_object(status_path)

    recorded_hashes = status.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict):
        raise RuntimeError("Paired status has no artifact SHA manifest")
    actual_hashes = {
        label: _sha256_file(path) for label, path in artifact_paths.items()
    }
    artifact_hashes_verified = actual_hashes == recorded_hashes
    if not artifact_hashes_verified:
        raise RuntimeError("Paired artifact SHA verification failed")

    try:
        requested_pairs = int(manifest["pairs"])
        familywise_alpha = float(manifest["familywise_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Paired manifest has invalid statistical inputs") from error
    if requested_pairs < PAIRED_GATE_MIN_ACCEPTED_PAIRS:
        raise RuntimeError("Paired artifact requested fewer than 1024 pairs")
    candidate_rows = _read_jsonl_objects(
        artifact_paths["candidate_pairs.jsonl"]
    )
    stored_pairs = _read_jsonl_objects(
        artifact_paths["paired_episodes.jsonl"]
    )
    stored_heldout_pool = _read_jsonl_objects(
        artifact_paths["heldout_benign_pool.jsonl"]
    )
    heldout_manifest = manifest.get("heldout_benign")
    if not isinstance(heldout_manifest, dict):
        raise RuntimeError("Paired manifest lacks held-out benign provenance")
    upstream_data = Path("/selfplay-redteaming/red_team/data")
    source_benign_rows = _read_jsonl_objects(
        upstream_data / "vanilla_benign_dataset.jsonl"
    )
    sft_benign_rows = _read_jsonl_objects(
        upstream_data / DEFENDER_V2_BENIGN_SOURCE_FILENAME
    )
    recomputed_heldout = build_sft_disjoint_benign_pool(
        source_benign_rows,
        sft_benign_rows,
        selection_seed=int(manifest["seed_base"]),
    )
    heldout_benign_disjoint = bool(
        recomputed_heldout["rows"] == stored_heldout_pool
        and all(
            heldout_manifest.get(key) == value
            for key, value in recomputed_heldout["metadata"].items()
        )
        and heldout_manifest.get("pool_file_sha256")
        == actual_hashes["heldout_benign_pool.jsonl"]
        and heldout_manifest.get("bypasses_a1") is True
    )
    if not heldout_benign_disjoint:
        raise RuntimeError("Held-out benign pool is not reproducibly SFT-disjoint")
    recomputed_progress = assemble_valid_actual_paired_prefix(
        candidate_rows,
        requested_pairs,
    )
    if not recomputed_progress["complete"]:
        raise RuntimeError("Paired candidate artifact has an incomplete prefix")
    recomputed_pairs = recomputed_progress["pairs"]
    if stored_pairs != recomputed_pairs:
        raise RuntimeError("Stored accepted pairs differ from candidate recomputation")
    recomputed_statistics = summarize_actual_d1_paired_gate(
        recomputed_pairs,
        familywise_alpha=familywise_alpha,
    )
    independently_verified = _independently_verify_actual_gate_candidates(
        candidate_rows,
        prompt_prelabel_calibration_response=str(
            manifest.get("prompt_prelabel_calibration_response") or ""
        ),
    )
    if independently_verified["drop_counts"] != {
        "overall": recomputed_progress["dropped_counts"]["total"],
        "harmful": recomputed_progress["dropped_counts"]["harmful"],
        "benign": recomputed_progress["dropped_counts"]["benign"],
    }:
        raise RuntimeError(
            "Independent actual-gate drop counts differ from prefix helper"
        )
    recomputed_summary_verified = (
        summary.get("paired_statistics") == recomputed_statistics
        and int(
            (summary.get("candidate_resampling") or {}).get(
                "accepted_pair_count", -1
            )
        )
        == len(recomputed_pairs)
        and (summary.get("candidate_resampling") or {}).get(
            "candidate_count"
        )
        == recomputed_progress["candidate_count"]
        and (summary.get("candidate_resampling") or {}).get(
            "dropped_counts"
        )
        == recomputed_progress["dropped_counts"]
        and summary.get("actual_stratum_counts")
        == {
            "harmful": requested_pairs // 2,
            "benign": requested_pairs // 2,
        }
        and summary.get("heldout_benign_pool_sha256")
        == actual_hashes["heldout_benign_pool.jsonl"]
    )
    if not recomputed_summary_verified:
        raise RuntimeError("Paired summary differs from independent recomputation")

    verification = verify_d1_paired_evidence_contract(
        state,
        manifest,
        summary,
        status,
        a1_audit=a1_audit,
        d1_audit=d1_audit,
        expected_implementation_hashes=(
            _current_paired_implementation_hashes()
        ),
        artifact_hashes_verified=artifact_hashes_verified,
        recomputed_summary_verified=recomputed_summary_verified,
        heldout_benign_disjoint=heldout_benign_disjoint,
        expected_seed_base=PAIRED_GATE_HELDOUT_SEED_BASE,
        min_pairs=PAIRED_GATE_MIN_ACCEPTED_PAIRS,
    )
    promotion = evaluate_d1_actual_paired_promotion(
        recomputed_statistics,
        {
            "candidate_count": recomputed_progress["candidate_count"],
            "valid_counts": recomputed_progress["valid_counts"],
            "dropped_counts": recomputed_progress["dropped_counts"],
            "accepted_pair_count": len(recomputed_pairs),
        },
        verification,
        min_accepted_pairs=PAIRED_GATE_MIN_ACCEPTED_PAIRS,
        min_joint_success=0.95,
        max_benign_compliance_drop=0.02,
        max_drop_rate=PAIRED_GATE_MAX_PARSE_DROP_RATE,
        mcnemar_alpha=0.05,
    )
    approval_path = root / "d1_paired_gate_approval.json"
    approval = {
        "schema_version": 1,
        "evaluated_at": datetime.now().isoformat(),
        "selfplay_run_suffix": run_suffix,
        "paired_run_suffix": paired_run_suffix,
        "evidence_root": str(evidence_root),
        "evidence_sha256": verification["evidence_sha256"],
        "artifact_sha256": actual_hashes,
        "implementation_hashes": manifest["implementation_hashes"],
        "a1_sha256": verification["a1_sha256"],
        "d1_sha256": verification["d1_sha256"],
        "verification": verification,
        "promotion": promotion,
    }
    _write_json_atomic(approval_path, approval)
    if not promotion["passed"]:
        state["status"] = "d1_paired_gate_failed"
        state["active_stage"] = None
        state["d1_paired_gate_rejection"] = approval
        _write_json_atomic(root / "state.json", state)
        output_vol.commit()
        return {
            "root": str(root),
            "state": state,
            "spawned": False,
            "promotion": promotion,
            "approval_path": str(approval_path),
        }

    schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
    if len(schedule) < 2:
        raise RuntimeError("Configured self-play has no A2 stage")
    state["d1_paired_promotion"] = {
        "status": "approved",
        "paired_run_suffix": paired_run_suffix,
        "evidence_sha256": verification["evidence_sha256"],
        "approval_path": str(approval_path),
        "verification": verification,
        "promotion": promotion,
    }
    state["stages"]["D1"]["successor_release"] = {
        "approved": True,
        "basis": "verified held-out paired D1 promotion",
        "evidence_sha256": verification["evidence_sha256"],
    }
    _persist_state(root, state)
    dispatch = _dispatch_stage_claim(
        root,
        state,
        run_suffix=run_suffix,
        stage=schedule[1],
    )
    approval["a2_spawn_claim_id"] = dispatch["spawn_claim_id"]
    approval["a2_call_id_observational"] = dispatch["call_id"]
    _write_json_atomic(approval_path, approval)
    output_vol.commit()
    return {
        "root": str(root),
        "state": dispatch["state"],
        "spawned": dispatch["spawned"],
        "call_id": dispatch["call_id"],
        "spawn_claim_id": dispatch["spawn_claim_id"],
        "promotion": promotion,
        "approval_path": str(approval_path),
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def resume_role_lora_selfplay8_transition(
    run_suffix: str,
) -> dict[str, Any]:
    """Reconcile the durable chain without accepting an external decision.

    Pending claims are submitted at least once with the same deterministic id.
    An ACKed child with no population copy resumes the globally serialized
    trainer at the exact same suffix/checkpoint set; it never starts a fresh
    run.  If its population copy exists, a duplicate only audits and retains
    that copy.  Retained stages may only finish their persisted validation gate
    and create the next deterministic claim.
    """

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_suffix or ""):
        raise ValueError("run_suffix must be one safe path component")
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state = _load_state(root)
    block_reason = _transition_resume_block_reason(state)
    if block_reason is not None:
        return {
            "root": str(root),
            "state": state,
            "spawned": False,
            "reason": f"terminal_or_gated:{block_reason}",
        }

    schedule = build_selfplay8_schedule(int(state["config"]["rounds"]))
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("Self-play state has no stages mapping")
    for position, stage in enumerate(schedule, start=1):
        stage_state = stages.get(stage.label)
        if stage_state is None:
            if position > 1:
                predecessor = stages.get(schedule[position - 2].label)
                release = (
                    predecessor.get("successor_release")
                    if isinstance(predecessor, dict)
                    else None
                )
                if not isinstance(release, dict) or (
                    release.get("approved") is not True
                ):
                    raise RuntimeError(
                        f"Predecessor did not release {stage.label}"
                    )
            dispatch = _dispatch_stage_claim(
                root,
                state,
                run_suffix=run_suffix,
                stage=stage,
                retry_existing_pending=True,
            )
            return {"root": str(root), **dispatch}
        if not isinstance(stage_state, dict):
            raise RuntimeError(f"Invalid stage state for {stage.label}")
        transition = stage_state.get("transition_state")
        if transition is None and stage_state.get("status") == "retained":
            transition = "retained"
        if transition == "spawn_pending":
            dispatch = _dispatch_stage_claim(
                root,
                state,
                run_suffix=run_suffix,
                stage=stage,
                retry_existing_pending=True,
            )
            return {"root": str(root), **dispatch}
        if transition == "child_started":
            population_committed = _population_path(
                root, stage.label
            ).is_dir()
            call = train_role_lora_selfplay8_stage.spawn(
                run_suffix=run_suffix,
                stage_index=stage.index,
                spawn_claim_id=stage_state["spawn_claim_id"],
            )
            return {
                "root": str(root),
                "state": state,
                "spawned": True,
                "call_id": call.object_id,
                "spawn_claim_id": stage_state["spawn_claim_id"],
                "reconcile_only": population_committed,
                "serialized_trainer_resume": not population_committed,
            }
        if transition != "retained":
            raise RuntimeError(
                f"Invalid durable transition for {stage.label}: {transition!r}"
            )
        release = stage_state.get("successor_release")
        if not (
            isinstance(release, dict) and release.get("approved") is True
        ):
            reconcile = _spawn_retained_stage_reconciler(
                state,
                run_suffix=run_suffix,
                stage=stage,
            )
            return {"root": str(root), **reconcile}

    state["status"] = "completed"
    state["active_stage"] = None
    state["completed_population"] = population_labels(
        int(state["config"]["rounds"])
    )
    _persist_state(root, state)
    return {"root": str(root), "state": state, "spawned": False}


@app.local_entrypoint(name="role_lora_selfplay8")
def role_lora_selfplay8(
    run_suffix: str = "",
    a1_checkpoint: str = SUCCESSFUL_A1_CHECKPOINT,
    rounds: int = 8,
    attacker_max_steps: int = 100,
    defender_max_steps: int = 200,
    save_steps: int = 10,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 1e-5,
    early_stop_threshold: float = 0.95,
    early_stop_patience: int = 5,
    early_stop_min_steps: int = 30,
    d1_min_improvement: float = 0.02,
    defender_sft_stop_after_step: int = 30,
) -> None:
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    call = initialize_role_lora_selfplay8.spawn(
        run_suffix=suffix,
        a1_checkpoint=a1_checkpoint,
        rounds=rounds,
        attacker_max_steps=attacker_max_steps,
        defender_max_steps=defender_max_steps,
        save_steps=save_steps,
        attacker_learning_rate=attacker_learning_rate,
        defender_learning_rate=defender_learning_rate,
        early_stop_threshold=early_stop_threshold,
        early_stop_patience=early_stop_patience,
        early_stop_min_steps=early_stop_min_steps,
        d1_min_improvement=d1_min_improvement,
        defender_sft_stop_after_step=defender_sft_stop_after_step,
    )
    print(f"RUN_SUFFIX={suffix}", flush=True)
    print(f"SELFPLAY8_INITIALIZER_CALL_ID={call.object_id}", flush=True)


@app.local_entrypoint(name="resume_a2_after_d1_paired_gate")
def resume_a2_after_d1_paired_gate(
    run_suffix: str,
    paired_run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    """Verify persisted paired evidence and resume A2 without a pass boolean."""

    invoke = (
        approve_d1_paired_gate_and_resume_a2.remote
        if wait_for_completion
        else approve_d1_paired_gate_and_resume_a2.spawn
    )
    result = invoke(
        run_suffix=run_suffix,
        paired_run_suffix=paired_run_suffix,
    )
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"PAIRED_RUN_SUFFIX={paired_run_suffix}", flush=True)
        print(f"D1_PAIRED_RESUME_CALL_ID={result.object_id}", flush=True)


@app.local_entrypoint(name="resume_role_lora_selfplay8_transition")
def resume_role_lora_selfplay8_transition_local(
    run_suffix: str,
    wait_for_completion: bool = False,
) -> None:
    """Retry/reconcile only the persisted deterministic transition state."""

    invoke = (
        resume_role_lora_selfplay8_transition.remote
        if wait_for_completion
        else resume_role_lora_selfplay8_transition.spawn
    )
    result = invoke(run_suffix=run_suffix)
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"SELFPLAY_TRANSITION_CALL_ID={result.object_id}", flush=True)
