#!/usr/bin/env python3
"""Cold-start role-LoRA PSRO with a projected terminal zero-sum payoff.

This module owns a new run root and does not edit the sibling Self-RedTeam
checkout.  Iteration one intentionally uses one fixed opponent per oracle; the
multi-opponent router is needed only after the first non-degenerate mixture.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

if os.path.isdir("/roll") and "/roll" not in os.sys.path:
    os.sys.path.insert(0, "/roll")

from modal_upstream_selfredteam_role_lora_v2 import (
    BASE_MODEL,
    _adapter_checkpoint,
    _normalize_fixed_opponent_pool,
    _run_role,
    _stable_wildguard_rm_url,
    hf_cache,
    llamafactory_lora_image,
    output_vol,
)
from modal_upstream_v2_payoff import (
    evaluate_upstream_v2_raw_payoff_cell,
)
from modal_abs_benchmark import image as payoff_image
from roll.utils.role_lora_cold_psro import (
    next_action as next_cold_psro_action,
    opponent_pool as build_psro_opponent_pool,
    payoff_matrix as build_psro_payoff_matrix,
)
from roll.utils.role_lora_zero_sum_psro import (
    ZERO_SUM_REWARD_VERSION,
    analyze_zero_sum_convergence,
    assemble_valid_zero_sum_prefix,
    rescore_zero_sum_episodes,
    solve_zero_sum_meta_game,
    zero_sum_cell_cache_key,
)
from roll.utils.role_lora_training_reward import (
    TRAINING_LABEL_DRIFT_POLICY,
)
from roll.utils.role_lora_naive_selfplay import (
    build_latest_opponent_schedule,
)
from roll.utils.upstream_v2_payoff import mean_ci95


_evaluate_upstream_v2_raw_payoff_cell = (
    evaluate_upstream_v2_raw_payoff_cell.get_raw_f()
)


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

psro_app = modal.App("role-lora-zero-sum-psro")
psro_lora_image = llamafactory_lora_image.add_local_file(
    str(LOCAL_REPOSITORY / "modal_upstream_selfredteam_role_lora_v2.py"),
    "/root/modal_upstream_selfredteam_role_lora_v2.py",
    copy=False,
)
psro_payoff_image = payoff_image


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


def _psro_implementation_hashes() -> dict[str, str]:
    candidates = {
        "coordinator": Path(__file__).resolve(),
        "trainer": LOCAL_REPOSITORY
        / "modal_upstream_selfredteam_role_lora_v2.py",
        "payoff": LOCAL_REPOSITORY / "modal_upstream_v2_payoff.py",
        "schedule": Path("/roll/roll/utils/role_lora_cold_psro.py"),
        "reward": Path("/roll/roll/utils/role_lora_zero_sum_psro.py"),
    }
    local_roll = LOCAL_REPOSITORY / "roll" / "utils"
    for key, filename in (
        ("schedule", "role_lora_cold_psro.py"),
        ("reward", "role_lora_zero_sum_psro.py"),
    ):
        if not candidates[key].is_file():
            candidates[key] = local_roll / filename
    missing = {key: str(path) for key, path in candidates.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"missing PSRO implementation sources: {missing}")
    return {key: _sha256_file(path) for key, path in candidates.items()}


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
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


def _adapter_identity(path: Path) -> dict[str, Any]:
    if not _adapter_checkpoint(path):
        raise FileNotFoundError(f"Missing adapter checkpoint: {path}")
    config_path = path / "adapter_config.json"
    config = _read_json(config_path)
    weights = next(
        candidate
        for candidate in (
            path / "adapter_model.safetensors",
            path / "adapter_model.bin",
        )
        if candidate.is_file()
    )
    return {
        "path": str(path),
        "adapter_sha256": _sha256_file(weights),
        "config_sha256": _sha256_file(config_path),
        "weights_file": weights.name,
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
    }


@psro_app.function(
    image=psro_lora_image,
    gpu=os.environ.get("ROLE_LORA_PSRO_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_cold_start_iteration_one(
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
) -> dict[str, Any]:
    """Train cold A1 and D1 with role-specific label-drift handling."""

    if steps_per_role < 1:
        raise ValueError("steps_per_role must be positive")
    if lora_rank != 64 or lora_alpha != 64:
        raise ValueError("the canonical cold-start path is fixed at rank/alpha 64")
    if attacker_learning_rate <= 0 or defender_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 <= attacker_sft_stop_after_step <= steps_per_role:
        raise ValueError("attacker SFT cutoff must be within the role budget")
    if not 0 <= defender_sft_stop_after_step <= steps_per_role:
        raise ValueError("defender SFT cutoff must be within the role budget")
    if sft_batches_per_step < 1:
        raise ValueError("sft_batches_per_step must be positive")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr",
        "constant",
        "constant_with_warmup",
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")

    output_vol.reload()
    suffix = _safe_component(
        run_suffix or datetime.now().strftime("cold_A1D1_%Y%m%d_%H%M%S"),
        label="run_suffix",
    )
    root = PSRO_OUTPUT_ROOT / suffix
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    contract = {
        "schema_version": "role-lora-zero-sum-psro-cold-v4",
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "init_mode": "cold",
        "definition": (
            "each active role starts from the same frozen base model with a "
            "new rank-64 adapter and a new optimizer"
        ),
        "training_reward": "original general_sum including existing shaping",
        "training_label_drift_policy": TRAINING_LABEL_DRIFT_POLICY,
        "training_label_drift_effect": {
            "attacker": "final shaped reward capped at zero",
            "defender": "entire generated-drift game omitted from replay",
        },
        "matrix_reward": ZERO_SUM_REWARD_VERSION,
        "matrix_label_drift_policy": (
            "drop generated_benign to harmful drift; score other valid rollouts"
        ),
        "base_model": BASE_MODEL,
        "prompt_mix": {
            "generated_harmful": 0.5,
            "generated_benign": 0.5,
        },
        "steps_per_role": steps_per_role,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "attacker_sft_stop_after_step": attacker_sft_stop_after_step,
        "defender_sft_stop_after_step": defender_sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "seed": 8888,
    }
    prior = _read_json(state_path) if state_path.is_file() else None
    if prior is not None and prior.get("contract") != contract:
        raise RuntimeError(f"run suffix already has a different contract: {root}")
    if prior is not None and prior.get("completed") is True:
        return prior

    state: dict[str, Any] = prior or {
        "completed": False,
        "stage": "initialized",
        "contract": contract,
        "population": {},
    }
    _write_json_atomic(state_path, state)
    output_vol.commit()

    a1_expected = root / "training" / "A1" / "ckpt" / (
        f"global_step{steps_per_role}_hf"
    )
    if _adapter_checkpoint(a1_expected):
        a1_checkpoint = a1_expected
    else:
        state["stage"] = "training_A1"
        _write_json_atomic(state_path, state)
        output_vol.commit()
        a1_checkpoint = _run_role(
            role="attacker",
            role_start_adapter=None,
            fixed_opponent=BASE_MODEL,
            fixed_opponent_adapter=None,
            remote_rm_url=_stable_wildguard_rm_url(),
            run_dir=root / "training" / "A1",
            steps=steps_per_role,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=attacker_learning_rate,
            sft_stop_after_step=attacker_sft_stop_after_step,
            sft_batches_per_step=sft_batches_per_step,
            save_steps=save_steps,
            actor_lr_scheduler=actor_lr_scheduler,
            lr_warmup_ratio=lr_warmup_ratio,
            exact_prompt_label_balance=True,
            label_drift_training_policy=TRAINING_LABEL_DRIFT_POLICY,
            wandb_identity=f"role_lora_zero_sum_psro__{suffix}__A1",
        )
    state["population"]["A1"] = _adapter_identity(a1_checkpoint)
    state["stage"] = "A1_completed"
    _write_json_atomic(state_path, state)
    output_vol.commit()

    d1_expected = root / "training" / "D1" / "ckpt" / (
        f"global_step{steps_per_role}_hf"
    )
    if _adapter_checkpoint(d1_expected):
        d1_checkpoint = d1_expected
    else:
        state["stage"] = "training_D1"
        _write_json_atomic(state_path, state)
        output_vol.commit()
        d1_checkpoint = _run_role(
            role="defender",
            role_start_adapter=None,
            fixed_opponent=BASE_MODEL,
            fixed_opponent_adapter=str(a1_checkpoint),
            remote_rm_url=_stable_wildguard_rm_url(),
            run_dir=root / "training" / "D1",
            steps=steps_per_role,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=defender_learning_rate,
            sft_stop_after_step=defender_sft_stop_after_step,
            sft_batches_per_step=sft_batches_per_step,
            save_steps=save_steps,
            actor_lr_scheduler=actor_lr_scheduler,
            lr_warmup_ratio=lr_warmup_ratio,
            exact_prompt_label_balance=True,
            label_drift_training_policy=TRAINING_LABEL_DRIFT_POLICY,
            wandb_identity=f"role_lora_zero_sum_psro__{suffix}__D1",
        )
    state["population"]["D1"] = _adapter_identity(d1_checkpoint)
    state.update(
        {
            "completed": True,
            "stage": "oracle_training_completed",
            "next_stage": "evaluate_A1_x_D1_then_solve_1x1_meta_game",
        }
    )
    _write_json_atomic(state_path, state)
    output_vol.commit()
    return state


def _verify_population_identity(
    label: str,
    record: dict[str, Any],
    *,
    lora_rank: int,
    lora_alpha: int,
) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    current = _adapter_identity(path)
    for field in ("path", "adapter_sha256", "config_sha256"):
        if current[field] != record.get(field):
            raise RuntimeError(
                f"{label} population identity changed at {field}: "
                f"state={record.get(field)!r}, current={current[field]!r}"
            )
    if current["rank"] != lora_rank or current["alpha"] != lora_alpha:
        raise RuntimeError(
            f"{label} LoRA contract mismatch: "
            f"r={current['rank']}, alpha={current['alpha']}"
        )
    return current


@psro_app.function(
    image=psro_lora_image,
    gpu=os.environ.get("ROLE_LORA_PSRO_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_naive_latest_opponent_role(
    *,
    run_root: str,
    target: str,
    role: str,
    role_start_adapter: str,
    role_start_sha256: str,
    opponent_adapter: str,
    opponent_sha256: str,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    sft_stop_after_step: int,
    sft_batches_per_step: int,
    save_steps: int,
    actor_lr_scheduler: str,
    lr_warmup_ratio: float,
    wandb_identity: str,
) -> dict[str, Any]:
    """Train exactly one warm-start role against its latest frozen opponent."""

    if role not in {"attacker", "defender"}:
        raise ValueError(f"invalid role: {role!r}")
    expected_prefix = "A" if role == "attacker" else "D"
    if not re.fullmatch(rf"{expected_prefix}[2-9][0-9]*", target):
        raise ValueError(f"invalid target label for {role}: {target!r}")
    root = Path(run_root)
    if root == PSRO_OUTPUT_ROOT or PSRO_OUTPUT_ROOT not in root.parents:
        raise ValueError("run_root must be below the PSRO output root")

    output_vol.reload()
    start_identity = _adapter_identity(Path(role_start_adapter))
    opponent_identity = _adapter_identity(Path(opponent_adapter))
    if start_identity["adapter_sha256"] != role_start_sha256:
        raise RuntimeError("active-role start adapter changed before training")
    if opponent_identity["adapter_sha256"] != opponent_sha256:
        raise RuntimeError("latest-opponent adapter changed before training")
    for label, identity in (
        ("role start", start_identity),
        ("opponent", opponent_identity),
    ):
        if identity["rank"] != lora_rank or identity["alpha"] != lora_alpha:
            raise RuntimeError(
                f"{label} LoRA contract mismatch: "
                f"r={identity['rank']}, alpha={identity['alpha']}"
            )

    run_dir = root / "training" / target
    checkpoint = run_dir / "ckpt" / f"global_step{steps}_hf"
    if _adapter_checkpoint(checkpoint):
        manifest = _read_json(run_dir / "manifest.json")
        expected_manifest = {
            "role": role,
            "steps": steps,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "role_start": role_start_adapter,
            "fixed_opponent_adapter": opponent_adapter,
            "label_drift_training_policy": TRAINING_LABEL_DRIFT_POLICY,
            "prompt_label_balance": (
                "deterministic exact 50/50 harmful/benign finite prefix"
            ),
        }
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                raise RuntimeError(
                    f"completed {target} manifest mismatch at {field}: "
                    f"expected={expected!r}, actual={manifest.get(field)!r}"
                )
        return _adapter_identity(checkpoint)

    checkpoint = _run_role(
        role=role,
        role_start_adapter=role_start_adapter,
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=opponent_adapter,
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=run_dir,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        exact_prompt_label_balance=True,
        label_drift_training_policy=TRAINING_LABEL_DRIFT_POLICY,
        wandb_identity=wandb_identity,
    )
    return _adapter_identity(checkpoint)


@psro_app.function(
    image=psro_lora_image,
    cpu=1,
    timeout=86400,
    volumes={"/output": output_vol},
)
def advance_naive_latest_opponent(
    *,
    source_run_suffix: str,
    continuation_suffix: str,
    last_generation: int = 5,
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
) -> dict[str, Any]:
    """Advance one phase, persist it, then detach the next controller."""

    if last_generation < 2:
        raise ValueError("last_generation must be at least 2")
    if steps_per_role < 1:
        raise ValueError("steps_per_role must be positive")
    if lora_rank != 64 or lora_alpha != 64:
        raise ValueError("the canonical continuation is fixed at rank/alpha 64")
    if not 0 <= attacker_sft_stop_after_step <= steps_per_role:
        raise ValueError("attacker SFT cutoff must be within the role budget")
    if not 0 <= defender_sft_stop_after_step <= steps_per_role:
        raise ValueError("defender SFT cutoff must be within the role budget")
    if save_steps < 1 or sft_batches_per_step < 1:
        raise ValueError("save_steps and sft_batches_per_step must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr",
        "constant",
        "constant_with_warmup",
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")

    output_vol.reload()
    source_suffix = _safe_component(
        source_run_suffix,
        label="source_run_suffix",
    )
    continuation = _safe_component(
        continuation_suffix,
        label="continuation_suffix",
    )
    source_root = PSRO_OUTPUT_ROOT / source_suffix
    source_state_path = source_root / "state.json"
    source_state = _read_json(source_state_path)
    if not source_state.get("completed"):
        raise RuntimeError("cold A1/D1 source run is not complete")
    source_population = source_state.get("population")
    if not isinstance(source_population, dict):
        raise RuntimeError("cold source state has no population")
    verified_source = {
        label: _verify_population_identity(
            label,
            dict(source_population.get(label, {})),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        for label in ("A1", "D1")
    }

    schedule = build_latest_opponent_schedule(
        first_generation=2,
        last_generation=last_generation,
    )
    root = source_root / "naive_latest_opponent" / continuation
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    contract = {
        "schema_version": "role-lora-naive-latest-opponent-v1",
        "algorithm": "naive_self_play_latest_opponent",
        "initialization": (
            "warm-start previous same-role adapter with a new optimizer"
        ),
        "source_run_suffix": source_suffix,
        "source_state_sha256": _sha256_file(source_state_path),
        "schedule": schedule,
        "training_reward": "original general_sum including existing shaping",
        "training_label_drift_policy": TRAINING_LABEL_DRIFT_POLICY,
        "prompt_mix": {
            "generated_harmful": 0.5,
            "generated_benign": 0.5,
        },
        "exact_prompt_label_balance": True,
        "psro_matrix_or_meta_solver_during_training": False,
        "steps_per_role": steps_per_role,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "attacker_sft_stop_after_step": attacker_sft_stop_after_step,
        "defender_sft_stop_after_step": defender_sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "seed": 8888,
    }
    prior = _read_json(state_path) if state_path.is_file() else None
    if prior is not None and prior.get("contract") != contract:
        raise RuntimeError(
            f"continuation suffix already has a different contract: {root}"
        )
    state: dict[str, Any] = prior or {
        "completed": False,
        "stage": "initialized",
        "contract": contract,
        "population": verified_source,
        "completed_targets": [],
    }
    population = state.get("population")
    if not isinstance(population, dict):
        raise RuntimeError("continuation state has no population")

    phase = next(
        (item for item in schedule if item["target"] not in population),
        None,
    )
    if phase is None:
        state.update(
            completed=True,
            stage=f"A2_through_D{last_generation}_completed",
        )
        _write_json_atomic(state_path, state)
        output_vol.commit()
        return state

    start_label = phase["initialized_from"]
    opponent_label = phase["opponent"]
    if start_label not in population or opponent_label not in population:
        raise RuntimeError(
            f"{phase['target']} prerequisites are incomplete: "
            f"start={start_label}, opponent={opponent_label}"
        )
    start_identity = _verify_population_identity(
        start_label,
        dict(population[start_label]),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    opponent_identity = _verify_population_identity(
        opponent_label,
        dict(population[opponent_label]),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    target = phase["target"]
    state["stage"] = f"training_{target}"
    _write_json_atomic(state_path, state)
    output_vol.commit()

    role = phase["role"]
    result = train_naive_latest_opponent_role.remote(
        run_root=str(root),
        target=target,
        role=role,
        role_start_adapter=start_identity["path"],
        role_start_sha256=start_identity["adapter_sha256"],
        opponent_adapter=opponent_identity["path"],
        opponent_sha256=opponent_identity["adapter_sha256"],
        steps=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=(
            attacker_learning_rate
            if role == "attacker"
            else defender_learning_rate
        ),
        sft_stop_after_step=(
            attacker_sft_stop_after_step
            if role == "attacker"
            else defender_sft_stop_after_step
        ),
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        wandb_identity=(
            f"role_lora_naive__{continuation}__{target}"
        ),
    )
    population[target] = result
    completed_targets = list(state.get("completed_targets", []))
    if target not in completed_targets:
        completed_targets.append(target)
    state["completed_targets"] = completed_targets
    state["stage"] = f"{target}_completed"
    _write_json_atomic(state_path, state)
    output_vol.commit()

    remaining = any(item["target"] not in population for item in schedule)
    if remaining:
        next_call = advance_naive_latest_opponent.spawn(
            source_run_suffix=source_suffix,
            continuation_suffix=continuation,
            last_generation=last_generation,
            steps_per_role=steps_per_role,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            attacker_learning_rate=attacker_learning_rate,
            defender_learning_rate=defender_learning_rate,
            attacker_sft_stop_after_step=attacker_sft_stop_after_step,
            defender_sft_stop_after_step=defender_sft_stop_after_step,
            sft_batches_per_step=sft_batches_per_step,
            save_steps=save_steps,
            actor_lr_scheduler=actor_lr_scheduler,
            lr_warmup_ratio=lr_warmup_ratio,
        )
        return {
            "completed_phase": target,
            "next_controller_call_id": next_call.object_id,
            "state_path": str(state_path),
        }

    state.update(
        completed=True,
        stage=f"A2_through_D{last_generation}_completed",
    )
    _write_json_atomic(state_path, state)
    output_vol.commit()
    return state


rescore_image = (
    modal.Image.debian_slim()
    .pip_install("numpy")
    .add_local_file(
        str(LOCAL_REPOSITORY / "modal_upstream_selfredteam_role_lora_v2.py"),
        "/root/modal_upstream_selfredteam_role_lora_v2.py",
        copy=False,
    )
    .add_local_file(
        str(LOCAL_REPOSITORY / "modal_upstream_v2_payoff.py"),
        "/root/modal_upstream_v2_payoff.py",
        copy=False,
    )
    .add_local_dir(
        str(Path(__file__).resolve().parent / "roll"),
        "/roll/roll",
        copy=False,
        ignore=["__pycache__", "**/*.pyc", "tests/", "docs/"],
    )
)
for prompt_filename in PROMPT_DATA_FILENAMES:
    rescore_image = rescore_image.add_local_file(
        str(LOCAL_SELFPLAY_SOURCE / "red_team" / "data" / prompt_filename),
        str(REMOTE_PROMPT_DATA_ROOT / prompt_filename),
        copy=False,
    )


def _mounted_prompt_dataset_contract() -> tuple[str, dict[str, list[dict[str, Any]]]]:
    file_hashes = {
        filename: _sha256_file(REMOTE_PROMPT_DATA_ROOT / filename)
        for filename in PROMPT_DATA_FILENAMES
    }
    combined = hashlib.sha256(
        json.dumps(
            file_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    rows_by_type: dict[str, list[dict[str, Any]]] = {}
    for prompt_type, filename in (
        ("generated_harmful", "vanilla_harmful_dataset.jsonl"),
        ("generated_benign", "vanilla_benign_dataset.jsonl"),
    ):
        rows_by_type[prompt_type] = _read_jsonl(
            REMOTE_PROMPT_DATA_ROOT / filename
        )
    return combined, rows_by_type


@psro_app.function(
    image=rescore_image,
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
        # Audited legacy cells stored the exact source index and seed text in
        # every accepted episode. Verify all of them against the pinned prompt
        # files before assigning the newly introduced dataset content hash.
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

    requested_episodes = zero_sum_episodes
    if requested_episodes and (
        requested_episodes < 256 or requested_episodes % 2
    ):
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
    if progress["requested_episodes"] < 256:
        raise RuntimeError(
            "raw candidate artifact has fewer than 256 retained games in "
            f"its largest balanced zero-sum prefix: {progress}"
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
        raise RuntimeError(f"cell output already has a different contract: {destination}")
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
                "drop WildGuard parse errors and generated-benign attacks "
                "classified as harmful, then take the first exact 50/50 "
                "nested prefix; retained games use the unchanged zero-sum "
                "terminal projection and are never zero-filled"
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


def _base_population_identity(label: str) -> dict[str, Any]:
    if label not in {"A0", "D0"}:
        raise ValueError(label)
    return {
        "path": None,
        "adapter_sha256": f"base:{BASE_MODEL}",
        "config_sha256": None,
        "weights_file": None,
        "rank": 0,
        "alpha": 0,
        "base_model": BASE_MODEL,
    }


def _persist_psro_inventory(root: Path, state: dict[str, Any]) -> None:
    population = state.get("population") or {}
    _write_json_atomic(
        root / "checkpoint_inventory.json",
        {
            "schema_version": "role-lora-cold-psro-checkpoints-v1",
            "run_root": str(root),
            "base_model": BASE_MODEL,
            "checkpoints": population,
            "warning": (
                "Do not prune these adapter directories; state.json and all "
                "matrix-cell manifests bind their weight SHA-256 identities."
            ),
        },
    )


def _persist_matrix_snapshot(
    root: Path,
    state: dict[str, Any],
    *,
    snapshot_name: str,
    attackers: list[str],
    defenders: list[str],
    matrix: list[list[float]],
    solution: dict[str, Any],
) -> dict[str, Any]:
    directory = root / "matrices"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot_name}.json"
    snapshot = {
        "schema_version": "role-lora-zero-sum-matrix-snapshot-v1",
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "payoff_orientation": "attacker/maximizer",
        "attackers": attackers,
        "defenders": defenders,
        "shape": [len(attackers), len(defenders)],
        "payoff": matrix,
        "meta_solution": solution,
        "cells": {
            f"{attacker}__{defender}": {
                "value": state["cells"][f"{attacker}__{defender}"]["value"],
                "episodes": state["cells"][f"{attacker}__{defender}"][
                    "episodes"
                ],
                "manifest_path": state["cells"][f"{attacker}__{defender}"][
                    "manifest_path"
                ],
            }
            for attacker in attackers
            for defender in defenders
        },
    }
    if path.is_file() and _read_json(path) != snapshot:
        raise RuntimeError(f"matrix snapshot changed: {path}")
    _write_json_atomic(path, snapshot)
    record = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "shape": snapshot["shape"],
    }
    state.setdefault("matrix_snapshots", {})[snapshot_name] = record
    return record


@psro_app.function(
    image=psro_payoff_image,
    gpu=os.environ.get("ROLE_LORA_PSRO_PAYOFF_GPU", "H200"),
    cpu=8,
    timeout=86400,
    memory=32768,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def evaluate_zero_sum_psro_cell(
    *,
    psro_run_suffix: str,
    attacker_label: str,
    defender_label: str,
    attacker_adapter: str,
    defender_adapter: str,
    attacker_sha256: str,
    defender_sha256: str,
    episodes: int = 4000,
    seed_base: int = 8888,
    max_candidate_multiplier: int = 4,
    candidate_wave_pairs: int = 64,
    generation_batch_size: int = 64,
    judge_batch_size: int = 64,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Generate and persist one exact retained zero-sum PSRO matrix cell."""

    if episodes < 256 or episodes % 2:
        raise ValueError("matrix episodes must be even and at least 256")
    run_suffix = _safe_component(psro_run_suffix, label="psro_run_suffix")
    attacker_name = _safe_component(attacker_label, label="attacker_label")
    defender_name = _safe_component(defender_label, label="defender_label")
    if not re.fullmatch(r"A[0-9]+", attacker_name):
        raise ValueError(f"invalid attacker label: {attacker_name}")
    if not re.fullmatch(r"D[0-9]+", defender_name):
        raise ValueError(f"invalid defender label: {defender_name}")

    raw_suffix = (
        f"{run_suffix}__{attacker_name}__{defender_name}__n{episodes}__zs_v4"
    )
    raw_summary = _evaluate_upstream_v2_raw_payoff_cell(
        attacker_adapter=attacker_adapter,
        defender_adapter=defender_adapter,
        remote_rm_url=_stable_wildguard_rm_url(),
        episodes=episodes,
        sample_counts=None,
        seed_base=seed_base,
        max_ci95_half_width=0.10,
        max_mean_drift=0.05,
        stable_windows=3,
        require_strata=True,
        min_convergence_episodes=256,
        familywise_alpha=0.05,
        max_candidate_multiplier=max_candidate_multiplier,
        candidate_wave_pairs=candidate_wave_pairs,
        generation_batch_size=generation_batch_size,
        judge_batch_size=judge_batch_size,
        max_new_tokens=max_new_tokens,
        retention_policy="zero_sum_psro_v4",
        run_suffix=raw_suffix,
        reuse_source_suffix="",
    )
    if raw_summary.get("completed") is not True:
        raise RuntimeError(f"raw generation did not complete: {raw_summary}")

    output_vol.reload()
    source = RAW_PAYOFF_ROOT / raw_suffix
    source_manifest_path = source / "manifest.json"
    source_candidates_path = source / "candidate_episodes.jsonl"
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("retention_policy") != "zero_sum_psro_v4":
        raise RuntimeError("raw cell did not use the zero-sum PSRO retention policy")
    source_attacker = source_manifest.get("attacker_adapter") or {}
    source_defender = source_manifest.get("defender_adapter") or {}
    observed_attacker_sha = source_attacker.get("sha256") or f"base:{BASE_MODEL}"
    observed_defender_sha = source_defender.get("sha256") or f"base:{BASE_MODEL}"
    if observed_attacker_sha != attacker_sha256:
        raise RuntimeError("attacker identity changed during matrix evaluation")
    if observed_defender_sha != defender_sha256:
        raise RuntimeError("defender identity changed during matrix evaluation")

    candidates = _read_jsonl(source_candidates_path)
    progress = assemble_valid_zero_sum_prefix(candidates, episodes=episodes)
    if not progress["complete"] or progress["accepted_count"] != episodes:
        raise RuntimeError(f"matrix cell retained-prefix contract failed: {progress}")
    rows = rescore_zero_sum_episodes(progress["episodes"])
    attacker_values = [
        float(row["attacker_zero_sum_reward"]) for row in rows
    ]
    value = sum(attacker_values) / episodes
    counts = [count for count in DEFAULT_SAMPLE_COUNTS if count <= episodes]
    if counts[-1] != episodes:
        counts.append(episodes)
    convergence = analyze_zero_sum_convergence(rows, sample_counts=counts)

    destination = (
        PSRO_OUTPUT_ROOT
        / run_suffix
        / "cells"
        / f"{attacker_name}__{defender_name}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    cell_contract = {
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "attacker_label": attacker_name,
        "defender_label": defender_name,
        "attacker_sha256": attacker_sha256,
        "defender_sha256": defender_sha256,
        "prompt_dataset_sha256": source_manifest["prompt_dataset_sha256"],
        "seed_base": seed_base,
        "retained_episodes": episodes,
        "retained_prompt_counts": {
            "harmful": episodes // 2,
            "benign": episodes // 2,
        },
        "generation_hyperparameters": {
            "max_candidate_multiplier": max_candidate_multiplier,
            "candidate_wave_pairs": candidate_wave_pairs,
            "generation_batch_size": generation_batch_size,
            "judge_batch_size": judge_batch_size,
            "max_new_tokens": max_new_tokens,
        },
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_candidates_sha256": _sha256_file(source_candidates_path),
    }
    manifest = {
        "schema_version": "role-lora-zero-sum-cell-v2",
        "cell_cache_key": zero_sum_cell_cache_key(cell_contract),
        "cell_contract": cell_contract,
        "source": str(source),
        "strict_zero_sum": True,
        "payoff_orientation": "attacker/maximizer",
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and _read_json(manifest_path) != manifest:
        raise RuntimeError(f"matrix cell contract changed: {destination}")
    _write_json_atomic(manifest_path, manifest)
    _write_jsonl_atomic(destination / "episodes.jsonl", rows)
    _write_json_atomic(destination / "convergence.json", convergence)
    summary = {
        "completed": True,
        "attacker_label": attacker_name,
        "defender_label": defender_name,
        "value": value,
        "attacker_payoff": mean_ci95(attacker_values),
        "defender_payoff": mean_ci95([-item for item in attacker_values]),
        "episodes": episodes,
        "prompt_counts": {
            "harmful": episodes // 2,
            "benign": episodes // 2,
        },
        "candidate_count": progress["candidate_count"],
        "dropped_counts": progress["dropped_counts"],
        "convergence": convergence,
        "manifest_path": str(manifest_path),
        "episodes_path": str(destination / "episodes.jsonl"),
        "raw_source": str(source),
    }
    _write_json_atomic(destination / "summary.json", summary)
    output_vol.commit()
    return summary


@psro_app.function(
    image=psro_lora_image,
    gpu=os.environ.get("ROLE_LORA_PSRO_GPU", "H200:4"),
    cpu=8,
    timeout=43200,
    memory=65536,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("roll-secrets")],
)
def train_cold_psro_oracle(
    *,
    run_root: str,
    target: str,
    role: str,
    opponent_pool: list[dict[str, object]],
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    sft_stop_after_step: int,
    sft_batches_per_step: int,
    save_steps: int,
    actor_lr_scheduler: str,
    lr_warmup_ratio: float,
    training_seed: int,
    wandb_identity: str,
) -> dict[str, Any]:
    """Train one cold-start best response against a frozen Nash mixture."""

    expected_prefix = "A" if role == "attacker" else "D"
    if not re.fullmatch(rf"{expected_prefix}[1-9][0-9]*", target):
        raise ValueError(f"invalid cold PSRO target: {target!r}")
    root = Path(run_root)
    if root == PSRO_OUTPUT_ROOT or PSRO_OUTPUT_ROOT not in root.parents:
        raise ValueError("run_root must be below the PSRO output root")
    output_vol.reload()
    normalized_pool = _normalize_fixed_opponent_pool(opponent_pool)
    run_dir = root / "training" / target
    checkpoint = run_dir / "ckpt" / f"global_step{steps}_hf"
    if _adapter_checkpoint(checkpoint):
        manifest = _read_json(run_dir / "manifest.json")
        if (
            manifest.get("role_start") != BASE_MODEL
            or manifest.get("fixed_opponent_pool") != normalized_pool
            or manifest.get("steps") != steps
        ):
            raise RuntimeError(f"existing {target} checkpoint contract changed")
        return _adapter_identity(checkpoint)
    checkpoint = _run_role(
        role=role,
        role_start_adapter=None,
        fixed_opponent=BASE_MODEL,
        fixed_opponent_adapter=None,
        fixed_opponent_pool=normalized_pool,
        remote_rm_url=_stable_wildguard_rm_url(),
        run_dir=run_dir,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        sft_stop_after_step=sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        seed=training_seed,
        exact_prompt_label_balance=True,
        label_drift_training_policy=TRAINING_LABEL_DRIFT_POLICY,
        wandb_identity=wandb_identity,
    )
    identity = _adapter_identity(checkpoint)
    output_vol.commit()
    return identity


def _advance_kwargs(contract: dict[str, Any], run_suffix: str) -> dict[str, Any]:
    return {
        "run_suffix": run_suffix,
        "generations": contract["generations"],
        "matrix_episodes": contract["matrix_episodes_per_cell"],
        "steps_per_role": contract["steps_per_role"],
        "lora_rank": contract["lora_rank"],
        "lora_alpha": contract["lora_alpha"],
        "attacker_learning_rate": contract["attacker_learning_rate"],
        "defender_learning_rate": contract["defender_learning_rate"],
        "attacker_sft_stop_after_step": contract[
            "attacker_sft_stop_after_step"
        ],
        "defender_sft_stop_after_step": contract[
            "defender_sft_stop_after_step"
        ],
        "sft_batches_per_step": contract["sft_batches_per_step"],
        "save_steps": contract["save_steps"],
        "actor_lr_scheduler": contract["actor_lr_scheduler"],
        "lr_warmup_ratio": contract["lr_warmup_ratio"],
        "training_seed": contract["training_seed"],
        "seed_base": contract["seed_base"],
        "max_candidate_multiplier": contract["matrix_generation"][
            "max_candidate_multiplier"
        ],
        "candidate_wave_pairs": contract["matrix_generation"][
            "candidate_wave_pairs"
        ],
        "generation_batch_size": contract["matrix_generation"][
            "generation_batch_size"
        ],
        "judge_batch_size": contract["matrix_generation"]["judge_batch_size"],
        "max_new_tokens": contract["matrix_generation"]["max_new_tokens"],
    }


@psro_app.function(
    image=rescore_image,
    cpu=2,
    timeout=86400,
    memory=8192,
    volumes={"/output": output_vol},
)
def advance_cold_psro(
    *,
    run_suffix: str,
    generations: int = 5,
    matrix_episodes: int = 4000,
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    training_seed: int = 8888,
    seed_base: int = 8888,
    max_candidate_multiplier: int = 4,
    candidate_wave_pairs: int = 64,
    generation_batch_size: int = 64,
    judge_batch_size: int = 64,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Advance one durable cell/oracle/evaluation action, then detach the next."""

    if generations < 1:
        raise ValueError("generations must be positive")
    if matrix_episodes < 256 or matrix_episodes % 2:
        raise ValueError("matrix_episodes must be even and at least 256")
    if steps_per_role < 1 or lora_rank != 64 or lora_alpha != 64:
        raise ValueError("the current role-LoRA/payoff runtime requires rank/alpha 64")
    if attacker_learning_rate <= 0 or defender_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 <= attacker_sft_stop_after_step <= steps_per_role:
        raise ValueError("attacker SFT cutoff must be within the role budget")
    if not 0 <= defender_sft_stop_after_step <= steps_per_role:
        raise ValueError("defender SFT cutoff must be within the role budget")
    if sft_batches_per_step < 1 or save_steps < 1:
        raise ValueError("SFT batches and checkpoint interval must be positive")
    if actor_lr_scheduler not in {
        "cosine_with_min_lr",
        "constant",
        "constant_with_warmup",
    }:
        raise ValueError("unsupported actor_lr_scheduler")
    if not 0 <= lr_warmup_ratio <= 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1]")
    if min(
        max_candidate_multiplier,
        candidate_wave_pairs,
        generation_batch_size,
        judge_batch_size,
        max_new_tokens,
    ) < 1:
        raise ValueError("matrix generation hyperparameters must be positive")
    suffix = _safe_component(run_suffix, label="run_suffix")
    root = PSRO_OUTPUT_ROOT / suffix
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    contract = {
        "schema_version": "role-lora-cold-zero-sum-psro-v1",
        "implementation_sha256": _psro_implementation_hashes(),
        "algorithm": "sequential_double_oracle_zero_sum_psro",
        "init_mode": "cold",
        "cold_definition": (
            "every learned A_i and D_i starts from the same base with a new "
            f"rank-{lora_rank} adapter and fresh optimizer"
        ),
        "base_model": BASE_MODEL,
        "base_in_population": {"attacker": "A0", "defender": "D0"},
        "generations": generations,
        "final_matrix_shape": [generations + 1, generations + 1],
        "matrix_episodes_per_cell": matrix_episodes,
        "matrix_prompt_mix": {"generated_harmful": 0.5, "generated_benign": 0.5},
        "matrix_reward": ZERO_SUM_REWARD_VERSION,
        "matrix_drift_policy": "drop benign-to-harmful and replace",
        "training_reward": "existing general_sum pipeline unchanged",
        "training_label_drift_policy": TRAINING_LABEL_DRIFT_POLICY,
        "opponent_rule": (
            "sample one strategy per episode from the frozen zero-sum Nash "
            "mixture; base is eligible and no exploration floor is added"
        ),
        "steps_per_role": steps_per_role,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "attacker_learning_rate": attacker_learning_rate,
        "defender_learning_rate": defender_learning_rate,
        "attacker_sft_stop_after_step": attacker_sft_stop_after_step,
        "defender_sft_stop_after_step": defender_sft_stop_after_step,
        "sft_batches_per_step": sft_batches_per_step,
        "save_steps": save_steps,
        "actor_lr_scheduler": actor_lr_scheduler,
        "lr_warmup_ratio": lr_warmup_ratio,
        "training_seed": training_seed,
        "seed_base": seed_base,
        "matrix_generation": {
            "max_candidate_multiplier": max_candidate_multiplier,
            "candidate_wave_pairs": candidate_wave_pairs,
            "generation_batch_size": generation_batch_size,
            "judge_batch_size": judge_batch_size,
            "max_new_tokens": max_new_tokens,
        },
        "post_training_evaluation": (
            "current selfredteam-official-eval defender workflow for every "
            f"D1-D{generations}"
        ),
    }
    output_vol.reload()
    prior = _read_json(state_path) if state_path.is_file() else None
    if prior is not None and prior.get("contract") != contract:
        raise RuntimeError(f"run suffix already has a different contract: {root}")
    state: dict[str, Any] = prior or {
        "completed": False,
        "stage": "initialized",
        "contract": contract,
        "population": {
            "A0": _base_population_identity("A0"),
            "D0": _base_population_identity("D0"),
        },
        "cells": {},
        "oracle_specs": {},
        "matrix_snapshots": {},
        "evaluations": {},
    }
    if state.get("completed") is True:
        return state
    population = state["population"]
    for label, record in population.items():
        if label in {"A0", "D0"}:
            if record != _base_population_identity(label):
                raise RuntimeError(f"base population identity changed: {label}")
        else:
            _verify_population_identity(
                label,
                record,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
    _persist_psro_inventory(root, state)
    _write_json_atomic(state_path, state)
    output_vol.commit()

    action = next_cold_psro_action(
        population,
        state["cells"],
        state["evaluations"],
        generations=generations,
    )
    kind = action["kind"]
    if kind == "cell":
        attacker = action["attacker"]
        defender = action["defender"]
        state["stage"] = f"evaluating_{attacker}_vs_{defender}"
        _write_json_atomic(state_path, state)
        output_vol.commit()
        attacker_record = population[attacker]
        defender_record = population[defender]
        result = evaluate_zero_sum_psro_cell.remote(
            psro_run_suffix=suffix,
            attacker_label=attacker,
            defender_label=defender,
            attacker_adapter=attacker_record.get("path") or "base",
            defender_adapter=defender_record.get("path") or "base",
            attacker_sha256=attacker_record["adapter_sha256"],
            defender_sha256=defender_record["adapter_sha256"],
            episodes=matrix_episodes,
            seed_base=seed_base,
            max_candidate_multiplier=max_candidate_multiplier,
            candidate_wave_pairs=candidate_wave_pairs,
            generation_batch_size=generation_batch_size,
            judge_batch_size=judge_batch_size,
            max_new_tokens=max_new_tokens,
        )
        state["cells"][f"{attacker}__{defender}"] = result
        state["stage"] = f"completed_{attacker}_vs_{defender}"
    elif kind == "oracle":
        role = action["role"]
        target = action["target"]
        attackers, defenders, matrix = build_psro_payoff_matrix(
            population, state["cells"]
        )
        solution = solve_zero_sum_meta_game(matrix)
        _persist_matrix_snapshot(
            root,
            state,
            snapshot_name=f"before_{target}",
            attackers=attackers,
            defenders=defenders,
            matrix=matrix,
            solution=solution,
        )
        if role == "attacker":
            pool = build_psro_opponent_pool(
                defenders, solution["defender_strategy"], population
            )
            learning_rate = attacker_learning_rate
            sft_cutoff = attacker_sft_stop_after_step
        else:
            pool = build_psro_opponent_pool(
                attackers, solution["attacker_strategy"], population
            )
            learning_rate = defender_learning_rate
            sft_cutoff = defender_sft_stop_after_step
        oracle_spec = {
            "role": role,
            "target": target,
            "matrix_attackers": attackers,
            "matrix_defenders": defenders,
            "payoff_matrix": matrix,
            "meta_solution": solution,
            "opponent_pool": pool,
        }
        existing_spec = state["oracle_specs"].get(target)
        if existing_spec is not None and existing_spec != oracle_spec:
            raise RuntimeError(f"frozen oracle specification changed: {target}")
        state["oracle_specs"][target] = oracle_spec
        state["stage"] = f"training_{target}"
        _write_json_atomic(state_path, state)
        output_vol.commit()
        result = train_cold_psro_oracle.remote(
            run_root=str(root),
            target=target,
            role=role,
            opponent_pool=pool,
            steps=steps_per_role,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            sft_stop_after_step=sft_cutoff,
            sft_batches_per_step=sft_batches_per_step,
            save_steps=save_steps,
            actor_lr_scheduler=actor_lr_scheduler,
            lr_warmup_ratio=lr_warmup_ratio,
            training_seed=training_seed,
            wandb_identity=f"role_lora_cold_psro__{suffix}__{target}",
        )
        population[target] = result
        state["stage"] = f"completed_{target}"
    elif kind == "evaluation":
        defender = action["defender"]
        state["stage"] = f"official_evaluation_{defender}"
        _write_json_atomic(state_path, state)
        output_vol.commit()
        evaluator = modal.Function.from_name(
            "selfredteam-official-eval",
            "evaluate_full_checkpoint_vs_base",
        )
        result = evaluator.remote(
            trained_checkpoint=population[defender]["path"],
            output_slug=f"cold_psro_{suffix}_{defender}",
            trained_label=f"cold_psro_{defender}",
            evaluate_base=False,
        )
        state["evaluations"][defender] = result
        state["stage"] = f"completed_official_evaluation_{defender}"
    elif kind == "complete":
        attackers, defenders, matrix = build_psro_payoff_matrix(
            population, state["cells"]
        )
        final_solution = solve_zero_sum_meta_game(matrix)
        _persist_matrix_snapshot(
            root,
            state,
            snapshot_name="final",
            attackers=attackers,
            defenders=defenders,
            matrix=matrix,
            solution=final_solution,
        )
        state.update(
            {
                "completed": True,
                "stage": "A1_A5_D1_D5_and_official_evaluations_completed",
                "final_matrix": {
                    "attackers": attackers,
                    "defenders": defenders,
                    "payoff": matrix,
                    "meta_solution": final_solution,
                },
            }
        )
        _persist_psro_inventory(root, state)
        _write_json_atomic(state_path, state)
        output_vol.commit()
        return state
    else:
        raise RuntimeError(f"unknown PSRO action: {action}")

    _persist_psro_inventory(root, state)
    _write_json_atomic(state_path, state)
    output_vol.commit()
    next_call = advance_cold_psro.spawn(**_advance_kwargs(contract, suffix))
    return {
        "completed_action": action,
        "next_controller_call_id": next_call.object_id,
        "state_path": str(state_path),
    }


@psro_app.local_entrypoint(name="cold_start_train")
def cold_start_train(
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    run_suffix: str = "",
    wait_for_completion: bool = False,
) -> None:
    suffix = run_suffix or datetime.now().strftime("cold_A1D1_%Y%m%d_%H%M%S")
    invoke = (
        train_cold_start_iteration_one.remote
        if wait_for_completion
        else train_cold_start_iteration_one.spawn
    )
    result = invoke(
        steps_per_role=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        attacker_learning_rate=attacker_learning_rate,
        defender_learning_rate=defender_learning_rate,
        attacker_sft_stop_after_step=attacker_sft_stop_after_step,
        defender_sft_stop_after_step=defender_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        run_suffix=suffix,
    )
    if wait_for_completion:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"PSRO_RUN_SUFFIX={suffix}", flush=True)
        print(f"TRAIN_CALL_ID={result.object_id}", flush=True)


@psro_app.local_entrypoint(name="cold_psro_train_and_eval")
def cold_psro_train_and_eval(
    run_suffix: str = "",
    generations: int = 5,
    matrix_episodes: int = 4000,
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
    training_seed: int = 8888,
    seed_base: int = 8888,
    max_candidate_multiplier: int = 4,
    candidate_wave_pairs: int = 64,
    generation_batch_size: int = 64,
    judge_batch_size: int = 64,
    max_new_tokens: int = 2048,
) -> None:
    """Launch a parameterized base-inclusive cold PSRO workflow."""

    suffix = run_suffix or datetime.now().strftime(
        f"cold_psro{generations}_base_n{matrix_episodes}_s{steps_per_role}_"
        "%Y%m%d_%H%M%S"
    )
    call = advance_cold_psro.spawn(
        run_suffix=suffix,
        generations=generations,
        matrix_episodes=matrix_episodes,
        steps_per_role=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        attacker_learning_rate=attacker_learning_rate,
        defender_learning_rate=defender_learning_rate,
        attacker_sft_stop_after_step=attacker_sft_stop_after_step,
        defender_sft_stop_after_step=defender_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        training_seed=training_seed,
        seed_base=seed_base,
        max_candidate_multiplier=max_candidate_multiplier,
        candidate_wave_pairs=candidate_wave_pairs,
        generation_batch_size=generation_batch_size,
        judge_batch_size=judge_batch_size,
        max_new_tokens=max_new_tokens,
    )
    print(f"PSRO_RUN_SUFFIX={suffix}", flush=True)
    print(f"PSRO_STATE=/output/role_lora_zero_sum_psro/{suffix}/state.json", flush=True)
    print(f"CONTROLLER_CALL_ID={call.object_id}", flush=True)


@psro_app.local_entrypoint(name="naive_selfplay_train")
def naive_selfplay_train(
    source_run_suffix: str,
    continuation_suffix: str = "",
    last_generation: int = 5,
    steps_per_role: int = 100,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    attacker_learning_rate: float = 1e-5,
    defender_learning_rate: float = 4e-5,
    attacker_sft_stop_after_step: int = 30,
    defender_sft_stop_after_step: int = 10,
    sft_batches_per_step: int = 1,
    save_steps: int = 10,
    actor_lr_scheduler: str = "constant_with_warmup",
    lr_warmup_ratio: float = 0.05,
) -> None:
    continuation = continuation_suffix or datetime.now().strftime(
        "naive_A2_to_D5_s100_%Y%m%d_%H%M%S"
    )
    call = advance_naive_latest_opponent.spawn(
        source_run_suffix=source_run_suffix,
        continuation_suffix=continuation,
        last_generation=last_generation,
        steps_per_role=steps_per_role,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        attacker_learning_rate=attacker_learning_rate,
        defender_learning_rate=defender_learning_rate,
        attacker_sft_stop_after_step=attacker_sft_stop_after_step,
        defender_sft_stop_after_step=defender_sft_stop_after_step,
        sft_batches_per_step=sft_batches_per_step,
        save_steps=save_steps,
        actor_lr_scheduler=actor_lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
    )
    print(f"SOURCE_RUN_SUFFIX={source_run_suffix}", flush=True)
    print(f"CONTINUATION_SUFFIX={continuation}", flush=True)
    print(f"CONTROLLER_CALL_ID={call.object_id}", flush=True)


@psro_app.local_entrypoint(name="rescore_cell")
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
