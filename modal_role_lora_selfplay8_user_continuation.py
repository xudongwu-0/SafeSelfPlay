#!/usr/bin/env python3
"""Explicitly release A2 under the user's seed-diversity-only policy.

The frozen training implementation remains byte-for-byte identical to the D1
run.  This entrypoint verifies immutable state, prompt artifacts, the D1
five-step stop, and both retained adapters; it then commits one auditable
operator-policy artifact together with the deterministic A2 claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import modal

if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from modal_role_lora_selfplay8 import (
    SELFPLAY_ROOT,
    _assert_training_implementation_frozen,
    _dispatch_stage_claim,
    _load_state,
    _read_json_object,
    _read_jsonl_objects,
    _sha256_file,
    _strict_audit,
    _write_json_atomic,
    app,
    output_vol,
)
from role_lora_selfplay8 import (
    build_selfplay8_schedule,
    deterministic_stage_spawn_claim,
)
from roll.utils.selfplay_training_continuation import (
    AUTHORIZATION_POLICY_VERSION,
    AUTHORIZED_RUN_SUFFIX,
    EXPECTED_A1_SHA256,
    EXPECTED_CHECKPOINT_VALIDATION_FILE_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_D1_SHA256,
    EXPECTED_EARLY_STOP_FILE_SHA256,
    EXPECTED_GAME_LOG_SHA256,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_PARTITION_FILE_SHA256,
    EXPECTED_PPO_REGISTRY_FILE_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    EXPECTED_TRAINING_IMPLEMENTATION_SHA256,
    EXPECTED_TRAINING_POOL_FILE_SHA256,
    EXPECTED_TRAINING_POOL_MANIFEST_FILE_SHA256,
    audit_d1_training_stop,
    audit_seed_prompt_distribution,
    build_authorization_payload,
    canonical_json_sha256,
)


FROZEN_HELPER_SHA256 = (
    "28cfeef165ead97fb80a0cfe7a5e9bdaa1d3695d529e8d49307c4bc192e741ee"
)
EXPECTED_A2_SPAWN_CLAIM_ID = (
    "28ad0830232d6fba78faa6c3296ba59f1fcd6d3bec24ce4ff027f0d7e5e3b984"
)
APPROVAL_FILENAME = "user_authorized_seed_diversity_continuation_v1.json"


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _assert_source_freeze(expected_override_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_override_sha256 or ""):
        raise ValueError("expected_override_sha256 must be one lowercase SHA256")
    observed = _source_sha256()
    if observed != expected_override_sha256:
        raise RuntimeError(
            "User-continuation entrypoint source drifted: "
            f"expected={expected_override_sha256}, observed={observed}"
        )
    helper_path = Path(__file__).resolve().parent / (
        "roll/utils/selfplay_training_continuation.py"
    )
    if not helper_path.is_file():
        helper_path = Path("/roll/roll/utils/selfplay_training_continuation.py")
    if not helper_path.is_file() or _sha256_file(helper_path) != FROZEN_HELPER_SHA256:
        raise RuntimeError("User-continuation helper differs from its frozen SHA256")


def _assert_file_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA drifted: expected={expected}, observed={observed}"
        )


def _verify_population_audits(state: dict[str, Any]) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for label, expected_sha in (
        ("A1", EXPECTED_A1_SHA256),
        ("D1", EXPECTED_D1_SHA256),
    ):
        stage = state["stages"][label]
        audit = _strict_audit(Path(stage["population_checkpoint"]))
        if audit.get("weight_sha256") != expected_sha:
            raise RuntimeError(f"{label} live population SHA drifted")
        contract = audit.get("llama_v2_contract")
        if not isinstance(contract, dict) or contract.get("passed") is not True:
            raise RuntimeError(f"{label} live population LoRA audit failed")
        if audit != stage.get("strict_audit"):
            raise RuntimeError(f"{label} live and retained strict audits differ")
        audits[label] = audit
    return audits


def _verify_existing_authorization(
    root: Path,
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = state.get("d1_user_authorized_training_continuation")
    if not isinstance(marker, dict):
        raise RuntimeError("State changed without the expected authorization marker")
    approval_path = root / APPROVAL_FILENAME
    approval = _read_json_object(approval_path)
    if approval.get("authorization_sha256") != marker.get("authorization_sha256"):
        raise RuntimeError("Authorization artifact/state digest mismatch")
    digest_payload = dict(approval)
    stored_digest = digest_payload.pop("authorization_sha256", None)
    if stored_digest != canonical_json_sha256(digest_payload):
        raise RuntimeError("Authorization artifact canonical digest drifted")
    if approval.get("promotion_policy", {}).get("version") != (
        AUTHORIZATION_POLICY_VERSION
    ):
        raise RuntimeError("Authorization policy version drifted")
    release = state.get("stages", {}).get("D1", {}).get("successor_release")
    if (
        not isinstance(release, dict)
        or release.get("approved") is not True
        or release.get("authorization_sha256") != stored_digest
    ):
        raise RuntimeError("D1 successor release differs from authorization")
    return approval, marker


@app.function(
    cpu=2,
    memory=8192,
    timeout=43200,
    max_containers=1,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def approve_seed_prompt_policy_and_resume_a2(
    run_suffix: str,
    expected_override_sha256: str,
) -> dict[str, Any]:
    """CAS one seed-diversity-only decision and create the A2 claim."""

    if run_suffix != AUTHORIZED_RUN_SUFFIX:
        raise ValueError(f"Only the audited run may continue: {AUTHORIZED_RUN_SUFFIX}")
    _assert_source_freeze(expected_override_sha256)
    output_vol.reload()
    root = SELFPLAY_ROOT / run_suffix
    state_path = root / "state.json"
    raw_state = state_path.read_bytes()
    state_sha256 = hashlib.sha256(raw_state).hexdigest()
    state = _load_state(root)
    schedule = build_selfplay8_schedule(8)
    if canonical_json_sha256(state.get("config", {})) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Frozen self-play config SHA drifted")
    if canonical_json_sha256([stage.to_dict() for stage in schedule]) != (
        EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError("Eight-round hot-start schedule SHA drifted")

    existing_marker = state.get("d1_user_authorized_training_continuation")
    if existing_marker is not None:
        approval, marker = _verify_existing_authorization(root, state)
        a2_claim = deterministic_stage_spawn_claim(state, schedule[1])
        if a2_claim != EXPECTED_A2_SPAWN_CLAIM_ID:
            raise RuntimeError("Existing A2 deterministic claim identity drifted")
        dispatch = _dispatch_stage_claim(
            root,
            state,
            run_suffix=run_suffix,
            stage=schedule[1],
            retry_existing_pending=True,
        )
        return {
            "root": str(root),
            "idempotent": True,
            "authorization": approval,
            "marker": marker,
            **dispatch,
        }

    if state_sha256 != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(
            "Initial self-play state failed compare-and-swap: "
            f"expected={EXPECTED_INITIAL_STATE_SHA256}, observed={state_sha256}"
        )
    if state.get("status") != "awaiting_d1_paired_gate" or state.get(
        "active_stage"
    ) is not None:
        raise RuntimeError("Self-play state is not idle at the D1 decision")
    if "A2" in state.get("stages", {}):
        raise RuntimeError("A2 already exists before the authorization marker")
    implementation_hashes = _assert_training_implementation_frozen(state)
    if implementation_hashes != EXPECTED_TRAINING_IMPLEMENTATION_SHA256:
        raise RuntimeError("Runtime training implementation differs from D1")

    contract = state["config"]["d1_data_contract"]
    partition_path = Path(contract["paths"]["canonical_partition"])
    pool_path = Path(contract["paths"]["training_prompt_pool"])
    pool_manifest_path = Path(contract["paths"]["training_prompt_pool_manifest"])
    ppo_registry_path = Path(contract["paths"]["ppo_exposure_registry"])
    d1_run_dir = Path(state["stages"]["D1"]["run_dir"])
    game_log_path = d1_run_dir / "run_tables" / "game_log.csv"
    early_stop_path = d1_run_dir / "ckpt" / "early_stop.json"
    checkpoint_validation_path = d1_run_dir / "checkpoint_validation.json"
    for path, expected, label in (
        (partition_path, EXPECTED_PARTITION_FILE_SHA256, "canonical partition"),
        (pool_path, EXPECTED_TRAINING_POOL_FILE_SHA256, "training prompt pool"),
        (
            pool_manifest_path,
            EXPECTED_TRAINING_POOL_MANIFEST_FILE_SHA256,
            "training prompt-pool manifest",
        ),
        (ppo_registry_path, EXPECTED_PPO_REGISTRY_FILE_SHA256, "PPO registry"),
        (game_log_path, EXPECTED_GAME_LOG_SHA256, "D1 game log"),
        (early_stop_path, EXPECTED_EARLY_STOP_FILE_SHA256, "D1 early-stop record"),
        (
            checkpoint_validation_path,
            EXPECTED_CHECKPOINT_VALIDATION_FILE_SHA256,
            "D1 checkpoint validation",
        ),
    ):
        _assert_file_sha(path, expected, label)

    partition = _read_json_object(partition_path)
    pool_rows = _read_jsonl_objects(pool_path)
    pool_manifest = _read_json_object(pool_manifest_path)
    ppo_registry = _read_json_object(ppo_registry_path)
    early_stop = _read_json_object(early_stop_path)
    checkpoint_validation = _read_json_object(checkpoint_validation_path)
    seed_audit = audit_seed_prompt_distribution(
        state=state,
        partition=partition,
        training_pool_rows=pool_rows,
        training_pool_manifest=pool_manifest,
        ppo_registry=ppo_registry,
    )
    stop_audit = audit_d1_training_stop(
        state=state,
        checkpoint_validation=checkpoint_validation,
        early_stop=early_stop,
    )
    population_audits = _verify_population_audits(state)
    approval = build_authorization_payload(
        run_suffix=run_suffix,
        seed_prompt_audit=seed_audit,
        d1_training_stop=stop_audit,
    )
    approval["population_strict_audits"] = population_audits
    approval_without_digest = dict(approval)
    approval_without_digest.pop("authorization_sha256", None)
    approval["authorization_sha256"] = canonical_json_sha256(
        approval_without_digest
    )

    # Re-check the durable compare-and-swap after nested strict audits.
    output_vol.reload()
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != (
        EXPECTED_INITIAL_STATE_SHA256
    ):
        raise RuntimeError("Self-play state changed during authorization audits")
    state = _load_state(root)
    approval_path = root / APPROVAL_FILENAME
    if approval_path.exists():
        if _read_json_object(approval_path) != approval:
            raise RuntimeError("A different authorization artifact already exists")
    else:
        _write_json_atomic(approval_path, approval)

    marker = {
        "status": "approved",
        "policy_version": AUTHORIZATION_POLICY_VERSION,
        "authorization_path": str(approval_path),
        "authorization_sha256": approval["authorization_sha256"],
        "scope": "self_play_training_continuation_only",
        "paired_heldout_required": False,
        "paired_generalization_claimed": False,
    }
    state["d1_user_authorized_training_continuation"] = marker
    state["stages"]["D1"]["successor_release"] = {
        "approved": True,
        "basis": (
            "user-authorized seed-prompt-distribution-only training "
            "continuation; paired heldout gate canceled"
        ),
        "policy_version": AUTHORIZATION_POLICY_VERSION,
        "authorization_path": str(approval_path),
        "authorization_sha256": approval["authorization_sha256"],
        "heldout_policy_generalization_required": False,
    }
    a2_claim = deterministic_stage_spawn_claim(state, schedule[1])
    if a2_claim != EXPECTED_A2_SPAWN_CLAIM_ID:
        raise RuntimeError(
            "A2 deterministic claim differs from the pre-audited identity: "
            f"{a2_claim}"
        )
    # _dispatch_stage_claim commits the approval file, state marker, release,
    # and A2 spawn_pending claim together before attempting the spawn.
    dispatch = _dispatch_stage_claim(
        root,
        state,
        run_suffix=run_suffix,
        stage=schedule[1],
        retry_existing_pending=True,
    )
    return {
        "root": str(root),
        "idempotent": False,
        "authorization": approval,
        "marker": marker,
        **dispatch,
    }


@app.local_entrypoint(name="resume_a2_after_user_seed_diversity_policy")
def resume_a2_after_user_seed_diversity_policy(
    run_suffix: str,
    expected_override_sha256: str,
    wait_for_completion: bool = False,
) -> None:
    """Release A2 without accepting a caller-provided pass/fail boolean."""

    invoke = (
        approve_seed_prompt_policy_and_resume_a2.remote
        if wait_for_completion
        else approve_seed_prompt_policy_and_resume_a2.spawn
    )
    result = invoke(
        run_suffix=run_suffix,
        expected_override_sha256=expected_override_sha256,
    )
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"SELFPLAY_RUN_SUFFIX={run_suffix}", flush=True)
        print(f"USER_CONTINUATION_CALL_ID={result.object_id}", flush=True)
