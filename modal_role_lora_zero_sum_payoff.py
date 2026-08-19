#!/usr/bin/env python3
"""Project terminal general-sum A x D outcomes into zero-sum PSRO cells."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import modal

if os.path.isdir("/roll") and "/roll" not in sys.path:
    sys.path.insert(0, "/roll")

from roll.utils.role_lora_zero_sum_psro import (
    ZERO_SUM_REWARD_VERSION,
    analyze_zero_sum_convergence,
    assemble_valid_zero_sum_prefix,
    rescore_zero_sum_episodes,
    solve_zero_sum_meta_game,
    zero_sum_cell_cache_key,
)
from roll.utils.upstream_v2_payoff import mean_ci95


PSRO_OUTPUT_ROOT = Path("/output/role_lora_zero_sum_psro")
RAW_PAYOFF_ROOT = Path(
    "/output/upstream_selfredteam_role_lora/raw_payoff_v2"
)
DEFAULT_SAMPLE_COUNTS = (
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    3072,
    3584,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
)
LOCAL_REPOSITORY = Path(__file__).resolve().parent
LOCAL_SELFPLAY_SOURCE = LOCAL_REPOSITORY.parent / "selfplay-redteaming"
PROMPT_DATA_FILENAMES = (
    "vanilla_harmful_dataset.jsonl",
    "vanilla_benign_dataset.jsonl",
)
REMOTE_PROMPT_DATA_ROOT = Path("/psro_prompt_data")

output_vol = modal.Volume.from_name(
    "roll-abs-benchmark-output", create_if_missing=True
)
payoff_app = modal.App("role-lora-zero-sum-payoff")
payoff_image = (
    modal.Image.debian_slim()
    .pip_install("numpy")
    .add_local_dir(
        str(LOCAL_REPOSITORY / "roll"),
        "/roll/roll",
        copy=False,
        ignore=["__pycache__", "**/*.pyc", "tests/", "docs/"],
    )
)
for prompt_filename in PROMPT_DATA_FILENAMES:
    payoff_image = payoff_image.add_local_file(
        str(LOCAL_SELFPLAY_SOURCE / "red_team" / "data" / prompt_filename),
        str(REMOTE_PROMPT_DATA_ROOT / prompt_filename),
        copy=False,
    )


def _safe_component(value: str, *, label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    if not safe or safe != value:
        raise ValueError(f"{label} must be one safe path component")
    return safe


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _mounted_prompt_dataset_contract() -> tuple[
    str, dict[str, list[dict[str, Any]]]
]:
    file_hashes = {
        filename: _sha256_file(REMOTE_PROMPT_DATA_ROOT / filename)
        for filename in PROMPT_DATA_FILENAMES
    }
    combined = hashlib.sha256(
        json.dumps(
            file_hashes, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    rows_by_type = {
        prompt_type: _read_jsonl(REMOTE_PROMPT_DATA_ROOT / filename)
        for prompt_type, filename in (
            ("generated_harmful", "vanilla_harmful_dataset.jsonl"),
            ("generated_benign", "vanilla_benign_dataset.jsonl"),
        )
    }
    return combined, rows_by_type


@payoff_app.function(
    image=payoff_image,
    cpu=2,
    timeout=1800,
    memory=4096,
    volumes={"/output": output_vol},
)
def rescore_raw_cell_zero_sum(
    *,
    raw_payoff_suffix: str,
    psro_run_suffix: str,
    attacker_label: str = "A1",
    defender_label: str = "D1",
    zero_sum_episodes: int = 0,
) -> dict[str, Any]:
    """Rescore a completed raw A x D artifact and solve its current matrix."""

    output_vol.reload()
    raw_suffix = _safe_component(raw_payoff_suffix, label="raw_payoff_suffix")
    run_suffix = _safe_component(psro_run_suffix, label="psro_run_suffix")
    attacker_name = _safe_component(attacker_label, label="attacker_label")
    defender_name = _safe_component(defender_label, label="defender_label")
    source = RAW_PAYOFF_ROOT / raw_suffix
    source_manifest_path = source / "manifest.json"
    source_summary_path = source / "summary.json"
    source_candidates_path = source / "candidate_episodes.jsonl"
    source_summary = _read_json(source_summary_path)
    if source_summary.get("completed") is not True:
        raise RuntimeError(f"raw payoff cell is not completed: {source}")
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("prompt_distribution") != (
        "deterministic exact 50/50 harmful/benign interleave"
    ):
        raise RuntimeError("raw cell does not use the frozen 50/50 protocol")

    source_rows = _read_jsonl(source_candidates_path)
    mounted_prompt_sha256, prompt_rows_by_type = (
        _mounted_prompt_dataset_contract()
    )
    prompt_dataset_sha256 = source_manifest.get("prompt_dataset_sha256")
    prompt_dataset_provenance = "source_manifest_and_mounted_files_match"
    if prompt_dataset_sha256 is None:
        for episode_index, row in enumerate(source_rows):
            prompt_type = str(row.get("prompt_type", ""))
            source_index = int(row.get("source_index", -1))
            pool = prompt_rows_by_type.get(prompt_type)
            if pool is None or not 0 <= source_index < len(pool):
                raise RuntimeError(
                    f"legacy prompt reference is invalid at episode {episode_index}"
                )
            if str(pool[source_index].get("vanilla", "")).strip() != str(
                row.get("seed_prompt", "")
            ).strip():
                raise RuntimeError(
                    f"legacy prompt text mismatch at episode {episode_index}"
                )
        prompt_dataset_sha256 = mounted_prompt_sha256
        prompt_dataset_provenance = (
            "legacy source lacked a dataset hash; every accepted source "
            "index/text was verified against the pinned mounted files"
        )
    elif prompt_dataset_sha256 != mounted_prompt_sha256:
        raise RuntimeError(
            "raw-cell prompt dataset hash differs from the pinned PSRO files"
        )
    if not isinstance(prompt_dataset_sha256, str) or len(prompt_dataset_sha256) != 64:
        raise RuntimeError("invalid prompt dataset SHA-256")

    source_episode_budget = int(source_manifest.get("episodes", 0))
    requested_episodes = zero_sum_episodes or source_episode_budget
    if requested_episodes < 256 or requested_episodes % 2:
        raise ValueError("zero_sum_episodes must be even and at least 256")
    progress = assemble_valid_zero_sum_prefix(
        source_rows,
        episodes=requested_episodes,
    )
    if not progress["complete"]:
        raise RuntimeError(
            "raw candidate artifact cannot fill the requested balanced "
            f"zero-sum prefix: {progress}"
        )
    rows = rescore_zero_sum_episodes(progress["episodes"])
    counts = [value for value in DEFAULT_SAMPLE_COUNTS if value <= len(rows)]
    if not counts or counts[-1] != len(rows):
        counts.append(len(rows))
    convergence = analyze_zero_sum_convergence(rows, sample_counts=counts)
    attacker_values = [
        float(row["attacker_zero_sum_reward"]) for row in rows
    ]
    cell_value = sum(attacker_values) / len(attacker_values)
    if not math.isfinite(cell_value):
        raise RuntimeError("non-finite zero-sum cell value")

    attacker_meta = source_manifest.get("attacker_adapter") or {}
    defender_meta = source_manifest.get("defender_adapter") or {}
    generation_keys = (
        "temperature",
        "top_p",
        "top_k",
        "min_new_tokens",
        "max_new_tokens",
        "prompt_max_tokens",
        "max_model_len",
        "generation_seed_scheme",
        "attacker_prompt_profile",
        "resolved_defender_prompt_protocol",
    )
    cell_contract = {
        "attacker_adapter_sha256": attacker_meta.get("sha256") or "base_model",
        "defender_adapter_sha256": defender_meta.get("sha256") or "base_model",
        "prompt_dataset_sha256": prompt_dataset_sha256,
        "seed_base": int(source_manifest["seed_base"]),
        "episodes": len(rows),
        "generation": {
            key: source_manifest.get(key) for key in generation_keys
        },
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_candidates_sha256": _sha256_file(source_candidates_path),
    }
    cell_key = zero_sum_cell_cache_key(cell_contract)
    destination = (
        PSRO_OUTPUT_ROOT
        / run_suffix
        / "cells"
        / f"{attacker_name}__{defender_name}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = {
        "schema_version": "role-lora-zero-sum-cell-v1",
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "payoff_orientation": "attacker/maximizer",
        "strict_zero_sum": True,
        "attacker_label": attacker_name,
        "defender_label": defender_name,
        "cell_cache_key": cell_key,
        "cell_contract": cell_contract,
        "source": str(source),
        "prompt_dataset_provenance": prompt_dataset_provenance,
    }
    if manifest_path.is_file() and _read_json(manifest_path) != manifest:
        raise RuntimeError(
            f"cell output already has a different contract: {destination}"
        )
    _write_json_atomic(manifest_path, manifest)
    _write_jsonl_atomic(destination / "episodes.jsonl", rows)
    _write_json_atomic(destination / "convergence.json", convergence)

    meta_solution = solve_zero_sum_meta_game([[cell_value]])
    summary = {
        "completed": True,
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "strict_zero_sum": True,
        "attacker_label": attacker_name,
        "defender_label": defender_name,
        "episodes": len(rows),
        "prompt_counts": {
            "harmful": sum(
                row["prompt_type"] == "generated_harmful" for row in rows
            ),
            "benign": sum(
                row["prompt_type"] == "generated_benign" for row in rows
            ),
        },
        "candidate_resampling": {
            "candidate_count": progress["candidate_count"],
            "valid_counts": progress["valid_counts"],
            "dropped_counts": progress["dropped_counts"],
            "accepted_count": progress["accepted_count"],
            "policy": (
                "drop only WildGuard parse errors, score label drift normally, "
                "then take the first exact 50/50 nested prefix; never zero-fill"
            ),
        },
        "attacker_payoff": mean_ci95(attacker_values),
        "defender_payoff": mean_ci95([-value for value in attacker_values]),
        "payoff_matrix": [[cell_value]],
        "meta_solution": meta_solution,
        "convergence": convergence,
        "manifest_path": str(manifest_path),
        "episodes_path": str(destination / "episodes.jsonl"),
    }
    _write_json_atomic(destination / "summary.json", summary)
    output_vol.commit()
    return summary


@payoff_app.local_entrypoint(name="rescore_cell")
def rescore_cell(
    raw_payoff_suffix: str,
    psro_run_suffix: str,
    attacker_label: str = "A1",
    defender_label: str = "D1",
    zero_sum_episodes: int = 0,
    wait_for_completion: bool = True,
) -> None:
    invoke = (
        rescore_raw_cell_zero_sum.remote
        if wait_for_completion
        else rescore_raw_cell_zero_sum.spawn
    )
    result = invoke(
        raw_payoff_suffix=raw_payoff_suffix,
        psro_run_suffix=psro_run_suffix,
        attacker_label=attacker_label,
        defender_label=defender_label,
        zero_sum_episodes=zero_sum_episodes,
    )
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"RESCORE_CALL_ID={result.object_id}", flush=True)
