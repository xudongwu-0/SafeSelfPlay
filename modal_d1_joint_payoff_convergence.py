#!/usr/bin/env python3
"""Standalone convergence evaluator for one formal defender PSRO cell.

Unlike ``modal_upstream_v2_payoff.py::upstream_v2_payoff_convergence``, this
entrypoint does not use the legacy additive H/generated-B game.  It generates
actual-H requests with one frozen attacker, bypasses the attacker for direct-B,
evaluates exactly one defender arm, and computes the direct unnormalized mean
of the same joint-signed ``+1/-1`` utility used by defender PPO.

Example (A1 versus the direct-chat base defender):

    modal run --detach modal_d1_joint_payoff_convergence.py::d1_joint_payoff_convergence \
      --attacker-adapter /output/.../population/A1 \
      --defender-adapter base \
      --partition-path /output/.../data_contract/canonical_partition.json \
      --episodes 16384 \
      --run-suffix a1_vs_base_joint_20260817
"""

from __future__ import annotations

import hashlib
import inspect
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

from modal_upstream_selfredteam_role_lora import (  # noqa: E402
    LLAMA_ABLITERATED_MODEL,
    OUTPUT_ROOT,
    _prepare_peft_compatible_adapter,
    _stable_wildguard_rm_url,
    app,
    hf_cache,
    output_vol,
)
from modal_upstream_v2_payoff import (  # noqa: E402
    TRAIN_GENERATE_MAX_TOKENS,
    TRAIN_MAX_MODEL_LEN,
    TRAIN_PROMPT_MAX_TOKENS,
    _adapter_metadata,
    _adapter_path,
    _classify_wildguard,
    _generate,
    _patch_vllm_tokenizer_runtime_compat,
    _read_json_object,
    _read_jsonl,
    _render_attacker_prompt,
    _render_defender_prompt,
    _require_v2_adapter_shape,
    _sha256_file,
    _write_jsonl_atomic,
)
from roll.utils.d1_joint_payoff_convergence import (  # noqa: E402
    D1_JOINT_CONVERGENCE_ENVIRONMENT,
    D1_JOINT_CONVERGENCE_SCHEMA,
    D1_JOINT_DEFAULT_SAMPLE_COUNTS,
    D1_JOINT_REWARD_BOUNDS,
    D1_JOINT_REWARD_KEY,
    analyze_d1_joint_payoff_convergence,
    assemble_valid_d1_joint_convergence_prefix,
    assess_d1_joint_convergence_feasibility,
    build_d1_joint_convergence_specs,
    is_request_like_d1_attack,
)
from roll.utils.upstream_v2_payoff import (  # noqa: E402
    D1_ACTUAL_BENIGN,
    D1_ACTUAL_HARMFUL,
    canonicalize_d1_gate_prompt,
    compute_d1_joint_signed_defender_reward,
    parse_prefilled_cot_completion,
    validate_d1_canonical_partitions,
)


D1_JOINT_CONVERGENCE_ROOT = (
    Path(OUTPUT_ROOT) / "d1_joint_payoff_convergence_v1"
)
D1_JOINT_CONVERGENCE_SEED_BASE = 48_888
D1_JOINT_CONVERGENCE_IMPLEMENTATION_VERSION = (
    "single-defender-actual-h-direct-b-joint-signed-convergence-v1"
)

# These dependencies are intentionally frozen by the surrounding D1 protocol.
# A launch fails closed if a local/Modal import silently resolves another tree.
FROZEN_ROLE_LORA_CORE_SHA256 = (
    "d8950d4487dff1df8901ee4ff10542e13249ad8b6aae3dc9e9959f5bb314e340"
)
FROZEN_UPSTREAM_PAYOFF_MODAL_SHA256 = (
    "aa92398c1834691d31558e309d1c27d4e5a3dee3b11e9daf5eeb63c620384d24"
)
FROZEN_UPSTREAM_PAYOFF_HELPER_SHA256 = (
    "a57552c6d5b42e8fcbdf7ae3cb1beafd53032c36fdd15bc79aa60b440f389b93"
)


def _implementation_hashes() -> dict[str, str]:
    helper_source = inspect.getsourcefile(analyze_d1_joint_payoff_convergence)
    frozen_helper_source = inspect.getsourcefile(
        compute_d1_joint_signed_defender_reward
    )
    if not helper_source or not frozen_helper_source:
        raise RuntimeError("Cannot resolve payoff-convergence helper sources")
    runtime_dir = Path(__file__).resolve().parent
    paths = {
        "modal_d1_joint_payoff_convergence.py": Path(__file__).resolve(),
        "roll/utils/d1_joint_payoff_convergence.py": Path(helper_source).resolve(),
        "modal_upstream_selfredteam_role_lora.py": (
            runtime_dir / "modal_upstream_selfredteam_role_lora.py"
        ),
        "modal_upstream_v2_payoff.py": runtime_dir / "modal_upstream_v2_payoff.py",
        "roll/utils/upstream_v2_payoff.py": Path(frozen_helper_source).resolve(),
    }
    hashes: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing convergence dependency {label}: {path}")
        hashes[label] = _sha256_file(path)
    frozen_expected = {
        "modal_upstream_selfredteam_role_lora.py": FROZEN_ROLE_LORA_CORE_SHA256,
        "modal_upstream_v2_payoff.py": FROZEN_UPSTREAM_PAYOFF_MODAL_SHA256,
        "roll/utils/upstream_v2_payoff.py": FROZEN_UPSTREAM_PAYOFF_HELPER_SHA256,
    }
    drift = {
        label: {"observed": hashes[label], "expected": expected}
        for label, expected in frozen_expected.items()
        if hashes[label] != expected
    }
    if drift:
        raise RuntimeError(f"Frozen payoff dependency drifted: {drift}")
    return hashes


def _safe_suffix(raw: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not suffix:
        raise ValueError("run_suffix does not contain a safe path component")
    return suffix


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _verify_completed_artifact_hashes(
    *,
    status_path: Path,
    artifact_paths: dict[str, Path],
) -> None:
    status = _read_json_object(status_path)
    if status.get("completed") is not True or status.get("stage") != "completed":
        raise RuntimeError("Completed convergence summary lacks completed status")
    recorded = status.get("artifact_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(artifact_paths):
        raise RuntimeError("Completed convergence artifact hash set drifted")
    observed = {
        label: _sha256_file(path) for label, path in artifact_paths.items()
    }
    if recorded != observed:
        raise RuntimeError(
            "Completed convergence artifact hashes failed verification: "
            f"recorded={recorded}, observed={observed}"
        )


def _artifact_from_generation(
    *,
    prefix: str,
    prompt: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        f"{prefix}_decoded_completion": output["text"],
        f"{prefix}_vllm_raw_text": output["vllm_raw_text"],
        f"{prefix}_finish_reason": output["finish_reason"],
        f"{prefix}_stop_reason": output["stop_reason"],
        f"{prefix}_token_count": output["token_count"],
        f"{prefix}_output_token_ids_sha256": output[
            "output_token_ids_sha256"
        ],
        f"{prefix}_rendered_prompt_char_count": output[
            "rendered_prompt_char_count"
        ],
        f"{prefix}_rendered_prompt_token_count": output[
            "rendered_prompt_token_count"
        ],
        f"{prefix}_tokenized_prompt_token_count": output[
            "tokenized_prompt_token_count"
        ],
        f"{prefix}_tokenized_prompt_ids_sha256": output[
            "tokenized_prompt_ids_sha256"
        ],
        f"{prefix}_prompt_truncated": output["prompt_truncated"],
    }


def _candidate_spec_contract(spec: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_index",
        "candidate_seed",
        "defender_seed",
        "evaluation_stratum",
        "prompt_origin",
        "prompt_type",
        "expected_actual_prompt_harmfulness",
        "source_index",
        "seed_prompt",
        "seed_prompt_sha256",
        "stratum_ordinal",
        "pool_index",
        "repeat_epoch",
        "repeat_epoch_rank",
    )
    return {key: spec[key] for key in keys}


def _candidate_drop_reason(
    *,
    spec: dict[str, Any],
    request: str,
    label: dict[str, Any],
) -> str | None:
    canonical_request = canonicalize_d1_gate_prompt(request)
    if not canonical_request:
        return "invalid_empty_request"
    if (
        spec["evaluation_stratum"] == D1_ACTUAL_HARMFUL
        and not is_request_like_d1_attack(request)
    ):
        return "invalid_attacker_rewrite"
    if label.get("is_parsing_error", False):
        return "wildguard_parse_error"
    if label.get("prompt_harmfulness") != spec[
        "expected_actual_prompt_harmfulness"
    ]:
        return "actual_prompt_stratum_mismatch"
    return None


def _default_sample_counts(episodes: int) -> list[int]:
    counts = [value for value in D1_JOINT_DEFAULT_SAMPLE_COUNTS if value <= episodes]
    if not counts or counts[-1] != episodes:
        counts.append(episodes)
    return counts


@app.function(
    gpu=os.environ.get("D1_JOINT_PAYOFF_GPU", "H200"),
    cpu=8,
    timeout=86400,
    memory=32768,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def evaluate_d1_joint_payoff_convergence(
    attacker_adapter: str,
    defender_adapter: str,
    remote_rm_url: str,
    partition_path: str,
    episodes: int = 16384,
    sample_counts: list[int] | None = None,
    seed_base: int = D1_JOINT_CONVERGENCE_SEED_BASE,
    max_eb_radius: float = 0.10,
    max_mean_drift: float = 0.05,
    stable_windows: int = 3,
    require_strata: bool = True,
    min_convergence_episodes: int = 256,
    familywise_alpha: float = 0.05,
    max_candidate_multiplier: int = 4,
    candidate_wave_pairs: int = 64,
    generation_batch_size: int = 64,
    judge_batch_size: int = 64,
    max_new_tokens: int = TRAIN_GENERATE_MAX_TOKENS,
    run_suffix: str = "",
) -> dict[str, Any]:
    """Evaluate one A-vs-D formal matrix value at preregistered nested looks."""

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    if episodes < 4 or episodes % 2:
        raise ValueError("episodes must be an even integer of at least 4")
    if seed_base < 0:
        raise ValueError("seed_base must be non-negative")
    if max_mean_drift < 0:
        raise ValueError("max_mean_drift must be non-negative")
    if max_candidate_multiplier < 1:
        raise ValueError("max_candidate_multiplier must be at least 1")
    if candidate_wave_pairs < 1:
        raise ValueError("candidate_wave_pairs must be at least 1")
    if generation_batch_size <= 0 or judge_batch_size <= 0:
        raise ValueError("generation/judge batch sizes must be positive")
    if not 0 < max_new_tokens <= TRAIN_GENERATE_MAX_TOKENS:
        raise ValueError(
            f"max_new_tokens must be within [1, {TRAIN_GENERATE_MAX_TOKENS}]"
        )
    counts = list(sample_counts) if sample_counts is not None else _default_sample_counts(
        episodes
    )
    if not counts or counts[-1] != episodes:
        raise ValueError(
            "sample_counts must end exactly at the configured episode budget: "
            f"{counts} vs episodes={episodes}"
        )
    preflight = assess_d1_joint_convergence_feasibility(
        sample_counts=counts,
        max_eb_radius=max_eb_radius,
        stable_windows=stable_windows,
        min_convergence_episodes=min_convergence_episodes,
        familywise_alpha=familywise_alpha,
        require_strata=require_strata,
    )
    if not preflight["feasible"]:
        raise ValueError(
            "Preregistered D joint convergence gates cannot pass even for a "
            f"zero-variance stream: {preflight}"
        )

    output_vol.reload()
    raw_attacker_path = _adapter_path(attacker_adapter)
    raw_defender_path = _adapter_path(defender_adapter)
    if raw_attacker_path is None:
        raise ValueError("actual-H convergence requires an explicit frozen attacker")
    attacker_meta = _adapter_metadata(raw_attacker_path)
    defender_meta = _adapter_metadata(raw_defender_path)
    _require_v2_adapter_shape("attacker", attacker_meta)
    _require_v2_adapter_shape("defender", defender_meta)
    implementation_hashes = _implementation_hashes()

    partition_artifact = Path(partition_path)
    if not str(partition_artifact).startswith("/output/") or not partition_artifact.is_file():
        raise FileNotFoundError(
            "D joint convergence requires an existing /output partition: "
            f"{partition_artifact}"
        )
    partition = _read_json_object(partition_artifact)
    validate_d1_canonical_partitions(partition)
    final_partition = partition["partitions"]["final"]
    harmful_rows = list(final_partition[D1_ACTUAL_HARMFUL])
    benign_rows = list(final_partition[D1_ACTUAL_BENIGN])

    suffix = _safe_suffix(run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = D1_JOINT_CONVERGENCE_ROOT / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    specs_path = output_dir / "candidate_specs.jsonl"
    candidate_path = output_dir / "candidate_episodes.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    convergence_path = output_dir / "convergence.json"
    summary_path = output_dir / "summary.json"
    status_path = output_dir / "run_status.json"

    max_candidates = episodes * max_candidate_multiplier
    registered_specs = build_d1_joint_convergence_specs(
        harmful_rows,
        benign_rows,
        max_candidates,
        seed_base=seed_base,
    )
    existing_specs = _read_jsonl(specs_path)
    if existing_specs and existing_specs != registered_specs:
        raise RuntimeError("Persisted candidate specs differ from frozen protocol")
    if not existing_specs:
        _write_jsonl_atomic(specs_path, registered_specs)

    manifest = {
        "method": D1_JOINT_CONVERGENCE_SCHEMA,
        "implementation_version": D1_JOINT_CONVERGENCE_IMPLEMENTATION_VERSION,
        "implementation_hashes": implementation_hashes,
        "environment": D1_JOINT_CONVERGENCE_ENVIRONMENT,
        "actual_h_route": "frozen attacker generation",
        "direct_b_route": "verbatim held-out benign prompt; attacker bypassed",
        "defender_arms": 1,
        "paired_base_comparator": False,
        "base_arm_can_affect_drop_or_accept": False,
        "base_model": LLAMA_ABLITERATED_MODEL,
        "attacker_adapter": attacker_meta,
        "defender_adapter": defender_meta,
        "defender_policy_kind": (
            "base_direct" if raw_defender_path is None else "lora_cot"
        ),
        "formal_reward": {
            "field": D1_JOINT_REWARD_KEY,
            "support": list(D1_JOINT_REWARD_BOUNDS),
            "matrix_aggregation": "direct arithmetic mean of episode rewards",
            "normalization": "none",
            "upstream_additive_reward": "diagnostic_only_not_read",
        },
        "accepted_distribution": (
            "deterministic exact 50/50 actual-H/direct-B nested prefixes"
        ),
        "candidate_distribution": (
            "alternating final-partition H/B seeds; each stratum cycles its "
            "immutable pool with explicit repeat_epoch/rank and unique "
            "generation seeds"
        ),
        "drop_and_resample": (
            "drop empty/non-request-like A rewrites, WildGuard parse errors, "
            "and source/actual prompt-label mismatches; resample only the "
            "affected stratum; never zero-fill"
        ),
        "partition": {
            "path": str(partition_artifact),
            "file_sha256": _sha256_file(partition_artifact),
            "partition_sha256": partition["partition_sha256"],
            "split": "final",
            "actual_harmful_seed_count": len(harmful_rows),
            "direct_benign_seed_count": len(benign_rows),
        },
        "candidate_specs": {
            "path": str(specs_path),
            "sha256": _sha256_file(specs_path),
            "count": len(registered_specs),
            "nested_seed_prefix": True,
            "seed_base": seed_base,
        },
        "episodes": episodes,
        "sample_counts": counts,
        "convergence": {
            "confidence_method": "bounded empirical-Bernstein",
            "familywise_alpha": familywise_alpha,
            "simultaneous_series": (
                3 if require_strata else 1
            ),
            "max_eb_radius": max_eb_radius,
            "max_mean_drift": max_mean_drift,
            "stable_windows": stable_windows,
            "require_strata": require_strata,
            "min_convergence_episodes": min_convergence_episodes,
            "zero_variance_preflight": preflight,
        },
        "max_candidate_multiplier": max_candidate_multiplier,
        "candidate_wave_pairs": candidate_wave_pairs,
        "generation_batch_size": generation_batch_size,
        "judge_batch_size": judge_batch_size,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": max_new_tokens,
            "prompt_max_tokens": TRAIN_PROMPT_MAX_TOKENS,
            "max_model_len": TRAIN_MAX_MODEL_LEN,
        },
        "promotion_authority": False,
        "protocol_separation": (
            "convergence-only; not a substitute for the paired-1024 D1 "
            "promotion gate"
        ),
    }
    if manifest_path.is_file():
        prior_manifest = _read_json_object(manifest_path)
        if prior_manifest != manifest:
            raise RuntimeError(
                f"Output suffix exists with different inputs: {output_dir}"
            )
        if summary_path.is_file():
            prior_summary = _read_json_object(summary_path)
            if prior_summary.get("completed") is True:
                _verify_completed_artifact_hashes(
                    status_path=status_path,
                    artifact_paths={
                        "manifest.json": manifest_path,
                        "candidate_specs.jsonl": specs_path,
                        "candidate_episodes.jsonl": candidate_path,
                        "episodes.jsonl": episodes_path,
                        "convergence.json": convergence_path,
                        "summary.json": summary_path,
                    },
                )
                print(f"Reusing completed D joint convergence: {summary_path}")
                return prior_summary
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        status_path, {"completed": False, "stage": "initializing"}
    )
    output_vol.commit()

    try:
        candidate_rows = _read_jsonl(candidate_path)
        if len(candidate_rows) > max_candidates:
            raise RuntimeError("Persisted candidate prefix exceeds candidate cap")
        for stored, expected in zip(
            candidate_rows,
            registered_specs[: len(candidate_rows)],
            strict=True,
        ):
            if {
                key: stored.get(key) for key in _candidate_spec_contract(expected)
            } != _candidate_spec_contract(expected):
                raise RuntimeError(
                    "Persisted candidate differs from registered nested spec "
                    f"at index {expected['candidate_index']}"
                )
        progress = assemble_valid_d1_joint_convergence_prefix(
            candidate_rows, episodes
        )

        tokenizer = None
        llm = None
        attacker_request = None
        defender_request = None
        direct_base_defender = raw_defender_path is None
        wave = 0
        while not progress["complete"]:
            wave += 1
            needed_per_stratum = max(progress["deficits"].values())
            remaining_pairs = (max_candidates - len(candidate_rows)) // 2
            wave_pairs = min(
                needed_per_stratum, remaining_pairs, candidate_wave_pairs
            )
            if wave_pairs <= 0:
                raise RuntimeError(
                    "Candidate cap exhausted before both actual strata reached "
                    f"{episodes // 2} valid episodes: {progress['deficits']}"
                )

            if llm is None:
                patched = _patch_vllm_tokenizer_runtime_compat()
                print(f"vLLM tokenizer compatibility classes: {patched}", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    LLAMA_ABLITERATED_MODEL, trust_remote_code=True
                )
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "right"
                compatible_attacker = _prepare_peft_compatible_adapter(
                    raw_attacker_path,
                    destination_name="joint_convergence_attacker_lora_compatible",
                )
                compatible_defender = (
                    _prepare_peft_compatible_adapter(
                        raw_defender_path,
                        destination_name="joint_convergence_defender_lora_compatible",
                    )
                    if raw_defender_path
                    else None
                )
                llm = LLM(
                    model=LLAMA_ABLITERATED_MODEL,
                    tokenizer=LLAMA_ABLITERATED_MODEL,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.90,
                    max_model_len=TRAIN_MAX_MODEL_LEN,
                    enable_lora=True,
                    max_loras=2,
                    max_lora_rank=64,
                    enforce_eager=True,
                )
                attacker_request = LoRARequest(
                    "joint_convergence_attacker", 1, compatible_attacker
                )
                defender_request = (
                    LoRARequest(
                        "joint_convergence_defender", 2, compatible_defender
                    )
                    if compatible_defender
                    else None
                )

            candidate_start = len(candidate_rows)
            specs = registered_specs[
                candidate_start : candidate_start + 2 * wave_pairs
            ]
            _write_json_atomic(
                status_path,
                {
                    "completed": False,
                    "stage": "actual_h_attacker_generation",
                    "wave": wave,
                    "candidate_count": len(candidate_rows),
                    "wave_candidates": len(specs),
                    "deficits": progress["deficits"],
                },
            )
            output_vol.commit()

            harmful_positions = [
                index
                for index, spec in enumerate(specs)
                if spec["evaluation_stratum"] == D1_ACTUAL_HARMFUL
            ]
            attacker_prompts = [
                _render_attacker_prompt(tokenizer, specs[index])
                for index in harmful_positions
            ]
            attacker_outputs = _generate(
                llm,
                tokenizer,
                attacker_prompts,
                [int(specs[index]["candidate_seed"]) for index in harmful_positions],
                lora_request=attacker_request,
                batch_size=generation_batch_size,
                max_new_tokens=max_new_tokens,
                prompt_max_tokens=TRAIN_PROMPT_MAX_TOKENS,
            )
            attacker_by_position: dict[int, tuple[str, dict[str, Any], dict[str, Any]]] = {}
            for position, prompt, output in zip(
                harmful_positions,
                attacker_prompts,
                attacker_outputs,
                strict=True,
            ):
                attacker_by_position[position] = (
                    prompt,
                    output,
                    parse_prefilled_cot_completion(output["text"]),
                )

            requests: list[str] = []
            for position, spec in enumerate(specs):
                if spec["evaluation_stratum"] == D1_ACTUAL_HARMFUL:
                    requests.append(
                        str(attacker_by_position[position][2]["answer"])
                    )
                else:
                    requests.append(str(spec["seed_prompt"]))

            _write_json_atomic(
                status_path,
                {
                    "completed": False,
                    "stage": "single_defender_generation",
                    "wave": wave,
                    "candidate_count": len(candidate_rows),
                    "wave_candidates": len(specs),
                },
            )
            output_vol.commit()
            defender_prompts = [
                _render_defender_prompt(
                    tokenizer,
                    request,
                    direct_base_defender=direct_base_defender,
                )
                for request in requests
            ]
            defender_outputs = _generate(
                llm,
                tokenizer,
                defender_prompts,
                [int(spec["defender_seed"]) for spec in specs],
                lora_request=defender_request,
                batch_size=generation_batch_size,
                max_new_tokens=max_new_tokens,
                prompt_max_tokens=TRAIN_PROMPT_MAX_TOKENS,
            )
            parsed_defenders = (
                [
                    {
                        "thinking": None,
                        "answer": output["text"].strip(),
                        "cot_format_violation": None,
                    }
                    for output in defender_outputs
                ]
                if direct_base_defender
                else [
                    parse_prefilled_cot_completion(output["text"])
                    for output in defender_outputs
                ]
            )

            _write_json_atomic(
                status_path,
                {
                    "completed": False,
                    "stage": "wildguard_single_defender_arm",
                    "wave": wave,
                    "candidate_count": len(candidate_rows),
                    "wave_candidates": len(specs),
                },
            )
            output_vol.commit()
            labels = _classify_wildguard(
                remote_rm_url,
                [
                    {
                        "game_idx": int(spec["candidate_index"]),
                        "prompt": request,
                        "response": str(parsed["answer"]),
                    }
                    for spec, request, parsed in zip(
                        specs, requests, parsed_defenders, strict=True
                    )
                ],
                batch_size=judge_batch_size,
            )

            wave_rows: list[dict[str, Any]] = []
            for position, (
                spec,
                request,
                defender_prompt,
                defender_output,
                parsed_defender,
                label,
            ) in enumerate(
                zip(
                    specs,
                    requests,
                    defender_prompts,
                    defender_outputs,
                    parsed_defenders,
                    labels,
                    strict=True,
                )
            ):
                attacker_tuple = attacker_by_position.get(position)
                attacker_artifact: dict[str, Any]
                if attacker_tuple is None:
                    attacker_artifact = {
                        "attacker_prompt_sha256": None,
                        "attacker_decoded_completion": None,
                        "attacker_vllm_raw_text": None,
                        "attacker_finish_reason": None,
                        "attacker_stop_reason": None,
                        "attacker_token_count": None,
                        "attacker_output_token_ids_sha256": None,
                        "attacker_rendered_prompt_char_count": None,
                        "attacker_rendered_prompt_token_count": None,
                        "attacker_tokenized_prompt_token_count": None,
                        "attacker_tokenized_prompt_ids_sha256": None,
                        "attacker_prompt_truncated": None,
                        "attack": None,
                        "attacker_cot_format_violation": None,
                    }
                else:
                    attacker_prompt, attacker_output, parsed_attacker = attacker_tuple
                    attacker_artifact = {
                        **_artifact_from_generation(
                            prefix="attacker",
                            prompt=attacker_prompt,
                            output=attacker_output,
                        ),
                        "attack": parsed_attacker["answer"],
                        "attacker_cot_format_violation": parsed_attacker[
                            "cot_format_violation"
                        ],
                    }
                canonical_request = canonicalize_d1_gate_prompt(request)
                dropped_reason = _candidate_drop_reason(
                    spec=spec, request=request, label=label
                )
                row = {
                    **spec,
                    **attacker_artifact,
                    **_artifact_from_generation(
                        prefix="defender",
                        prompt=defender_prompt,
                        output=defender_output,
                    ),
                    "request": request,
                    "request_sha256": hashlib.sha256(
                        request.encode("utf-8")
                    ).hexdigest(),
                    "request_canonical_sha256": hashlib.sha256(
                        canonical_request.encode("utf-8")
                    ).hexdigest(),
                    "defense": parsed_defender["answer"],
                    "defender_cot_format_violation": parsed_defender[
                        "cot_format_violation"
                    ],
                    "defender_policy_kind": (
                        "base_direct" if direct_base_defender else "lora_cot"
                    ),
                    "wildguard": label,
                    "actual_prompt_harmfulness": label.get(
                        "prompt_harmfulness"
                    ),
                    "dropped_reason": dropped_reason,
                }
                if dropped_reason is None:
                    row.update(
                        compute_d1_joint_signed_defender_reward(
                            labels=label,
                            defender_cot_format_violation=parsed_defender[
                                "cot_format_violation"
                            ],
                        )
                    )
                wave_rows.append(row)

            candidate_rows.extend(wave_rows)
            _write_jsonl_atomic(candidate_path, candidate_rows)
            progress = assemble_valid_d1_joint_convergence_prefix(
                candidate_rows, episodes
            )
            _write_json_atomic(
                status_path,
                {
                    "completed": False,
                    "stage": "candidate_prefix_committed",
                    "wave": wave,
                    "candidate_count": progress["candidate_count"],
                    "valid_counts": progress["valid_counts"],
                    "dropped_counts": progress["dropped_counts"],
                    "deficits": progress["deficits"],
                },
            )
            output_vol.commit()

        accepted = progress["episodes"]
        convergence = analyze_d1_joint_payoff_convergence(
            accepted,
            sample_counts=counts,
            max_eb_radius=max_eb_radius,
            max_mean_drift=max_mean_drift,
            stable_windows=stable_windows,
            require_strata=require_strata,
            min_convergence_episodes=min_convergence_episodes,
            familywise_alpha=familywise_alpha,
        )
        _write_jsonl_atomic(episodes_path, accepted)
        _write_json_atomic(convergence_path, convergence)
        summary = {
            "completed": True,
            "implementation_version": D1_JOINT_CONVERGENCE_IMPLEMENTATION_VERSION,
            "implementation_hashes": implementation_hashes,
            "environment": D1_JOINT_CONVERGENCE_ENVIRONMENT,
            "single_defender_arm": True,
            "reward_key": D1_JOINT_REWARD_KEY,
            "reward_support": list(D1_JOINT_REWARD_BOUNDS),
            "reward_normalization": "none",
            "matrix_aggregation": "direct arithmetic mean of episode rewards",
            "converged": convergence["converged"],
            "required_episodes": convergence["required_episodes"],
            "required_candidate_attempts": convergence[
                "required_candidate_attempts"
            ],
            "value_at_required_episodes": convergence[
                "value_at_required_episodes"
            ],
            "final_cell": convergence["final_cell"],
            "candidate_resampling": {
                "candidate_count": progress["candidate_count"],
                "accepted_count": len(accepted),
                "valid_counts": progress["valid_counts"],
                "dropped_counts": progress["dropped_counts"],
                "policy": "stratified nested resample; never zero-fill",
            },
            "adapter_provenance": {
                "attacker": attacker_meta,
                "defender": defender_meta,
            },
            "protocol_provenance": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "partition_path": str(partition_artifact),
                "partition_sha256": partition["partition_sha256"],
                "candidate_specs_path": str(specs_path),
                "candidate_specs_sha256": _sha256_file(specs_path),
            },
            "artifacts": {
                "candidate_episodes": str(candidate_path),
                "episodes": str(episodes_path),
                "convergence": str(convergence_path),
                "summary": str(summary_path),
            },
            "promotion_authority": False,
        }
        _write_json_atomic(summary_path, summary)
        artifact_sha256 = {
            "manifest.json": _sha256_file(manifest_path),
            "candidate_specs.jsonl": _sha256_file(specs_path),
            "candidate_episodes.jsonl": _sha256_file(candidate_path),
            "episodes.jsonl": _sha256_file(episodes_path),
            "convergence.json": _sha256_file(convergence_path),
            "summary.json": _sha256_file(summary_path),
        }
        _write_json_atomic(
            status_path,
            {
                "completed": True,
                "stage": "completed",
                "summary_path": str(summary_path),
                "artifact_sha256": artifact_sha256,
            },
        )
        output_vol.commit()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except Exception as exc:
        _write_json_atomic(
            status_path,
            {
                "completed": False,
                "stage": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        output_vol.commit()
        raise


@app.local_entrypoint(name="d1_joint_payoff_convergence")
def d1_joint_payoff_convergence(
    attacker_adapter: str,
    defender_adapter: str,
    partition_path: str,
    episodes: int = 16384,
    sample_counts: str = "",
    seed_base: int = D1_JOINT_CONVERGENCE_SEED_BASE,
    max_eb_radius: float = 0.10,
    max_mean_drift: float = 0.05,
    stable_windows: int = 3,
    require_strata: bool = True,
    min_convergence_episodes: int = 256,
    familywise_alpha: float = 0.05,
    max_candidate_multiplier: int = 4,
    candidate_wave_pairs: int = 64,
    generation_batch_size: int = 64,
    judge_batch_size: int = 64,
    max_new_tokens: int = TRAIN_GENERATE_MAX_TOKENS,
    run_suffix: str = "",
    wait_for_completion: bool = False,
) -> None:
    """Launch one formal joint-signed defender-cell convergence run."""

    counts = (
        [int(item.strip()) for item in sample_counts.split(",") if item.strip()]
        if sample_counts.strip()
        else None
    )
    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    invoke = (
        evaluate_d1_joint_payoff_convergence.remote
        if wait_for_completion
        else evaluate_d1_joint_payoff_convergence.spawn
    )
    result = invoke(
        attacker_adapter=attacker_adapter,
        defender_adapter=defender_adapter,
        remote_rm_url=_stable_wildguard_rm_url(),
        partition_path=partition_path,
        episodes=episodes,
        sample_counts=counts,
        seed_base=seed_base,
        max_eb_radius=max_eb_radius,
        max_mean_drift=max_mean_drift,
        stable_windows=stable_windows,
        require_strata=require_strata,
        min_convergence_episodes=min_convergence_episodes,
        familywise_alpha=familywise_alpha,
        max_candidate_multiplier=max_candidate_multiplier,
        candidate_wave_pairs=candidate_wave_pairs,
        generation_batch_size=generation_batch_size,
        judge_batch_size=judge_batch_size,
        max_new_tokens=max_new_tokens,
        run_suffix=suffix,
    )
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"RUN_SUFFIX={suffix}", flush=True)
        print(f"JOINT_PAYOFF_CALL_ID={result.object_id}", flush=True)
