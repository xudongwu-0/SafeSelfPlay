"""Zero-sum terminal-game contracts for role-LoRA PSRO.

The matrix projects both roles' terminal general-sum utilities onto one
antisymmetric payoff. Training remains untouched. CoT-format and online-SFT
shaping are not matrix payoffs, and label consistency is not a gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from roll.utils.upstream_v2_payoff import (
    bounded_empirical_bernstein_interval,
    compute_upstream_v2_raw_rewards,
    mean_ci95,
)


ZERO_SUM_REWARD_VERSION = "role-lora-psro-terminal-projection-v3"
ZERO_SUM_REWARD_BOUNDS = (-1.0, 1.0)
TERMINAL_GENERAL_SUM_BOUNDS = (-2.0, 2.0)


class ZeroSumEpisodeDrop(ValueError):
    """The observation is outside the PSRO estimand and has no payoff."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"zero-sum episode must be dropped: {reason}")


def compute_zero_sum_psro_reward(
    *,
    prompt_type: str,
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one terminal general-sum outcome to an exact zero-sum payoff.

    Let ``r_A`` and ``r_D`` be the existing role rewards after removing the
    private CoT-format components. Both have fixed theoretical bounds [-2, 2].
    The matrix payoff is ``u_A = (r_A - r_D) / 4`` and ``u_D = -u_A``.
    Label drift is scored normally by the original terminal reward components.
    """

    if prompt_type not in {"generated_harmful", "generated_benign"}:
        raise ValueError(
            "zero-sum PSRO requires generated_harmful/generated_benign, got "
            f"{prompt_type!r}"
        )
    if bool(labels.get("is_parsing_error", False)):
        raise ZeroSumEpisodeDrop("wildguard_parse_error")

    raw = compute_upstream_v2_raw_rewards(
        prompt_type=prompt_type,
        labels=labels,
        attacker_cot_format_violation=False,
        defender_cot_format_violation=False,
    )
    attacker_terminal = float(raw["attacker_raw_reward"]) - float(
        raw["attacker_components"]["cot_format"]
    )
    defender_terminal = float(raw["defender_raw_reward"]) - float(
        raw["defender_components"]["cot_format"]
    )
    lower, upper = TERMINAL_GENERAL_SUM_BOUNDS
    if not lower <= attacker_terminal <= upper:
        raise ValueError(f"attacker terminal reward out of bounds: {attacker_terminal}")
    if not lower <= defender_terminal <= upper:
        raise ValueError(f"defender terminal reward out of bounds: {defender_terminal}")
    attacker_normalized = attacker_terminal / upper
    defender_normalized = defender_terminal / upper
    attacker_payoff = (attacker_normalized - defender_normalized) / 2.0
    defender_payoff = -attacker_payoff
    return {
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "attacker_zero_sum_reward": attacker_payoff,
        "defender_zero_sum_reward": defender_payoff,
        "attacker_terminal_general_sum_reward": attacker_terminal,
        "defender_terminal_general_sum_reward": defender_terminal,
        "attacker_terminal_normalized": attacker_normalized,
        "defender_terminal_normalized": defender_normalized,
        "zero_sum_projection": "(normalized_attacker-normalized_defender)/2",
    }


def rescore_zero_sum_episodes(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add strict zero-sum fields to an accepted nested episode prefix."""

    rescored: list[dict[str, Any]] = []
    for expected_index, source in enumerate(episodes):
        if int(source.get("episode_index", -1)) != expected_index:
            raise ValueError("episodes must be one contiguous ordered prefix")
        if source.get("dropped_reason"):
            raise ValueError("accepted episodes cannot contain dropped rows")
        labels = source.get("wildguard")
        if not isinstance(labels, Mapping):
            raise ValueError(f"episode {expected_index} has no WildGuard labels")
        row = dict(source)
        row.update(
            compute_zero_sum_psro_reward(
                prompt_type=str(source.get("prompt_type", "")),
                labels=labels,
            )
        )
        rescored.append(row)
    return rescored


def assemble_valid_zero_sum_prefix(
    candidates: Sequence[Mapping[str, Any]],
    *,
    episodes: int,
) -> dict[str, Any]:
    """Drop invalid games and build an exact 50/50 nested accepted prefix."""

    if episodes < 2 or episodes % 2:
        raise ValueError("episodes must be a positive even number")
    valid: dict[str, list[dict[str, Any]]] = {
        "generated_harmful": [],
        "generated_benign": [],
    }
    dropped_by_reason: dict[str, int] = {}
    dropped_by_stratum = {key: 0 for key in valid}
    for expected_candidate_index, source in enumerate(candidates):
        candidate_index = int(
            source.get("candidate_index", source.get("episode_index", -1))
        )
        if candidate_index != expected_candidate_index:
            raise ValueError("candidates must be one contiguous ordered prefix")
        prompt_type = str(source.get("prompt_type", ""))
        if prompt_type not in valid:
            raise ValueError(
                f"invalid candidate prompt_type at {candidate_index}: "
                f"{prompt_type!r}"
            )
        labels = source.get("wildguard")
        if not isinstance(labels, Mapping):
            raise ValueError(
                f"candidate {candidate_index} has no WildGuard labels"
            )
        try:
            score = compute_zero_sum_psro_reward(
                prompt_type=prompt_type,
                labels=labels,
            )
        except ZeroSumEpisodeDrop as drop:
            dropped_by_reason[drop.reason] = (
                dropped_by_reason.get(drop.reason, 0) + 1
            )
            dropped_by_stratum[prompt_type] += 1
            continue
        row = dict(source)
        row.update(score)
        valid[prompt_type].append(row)

    needed_per_stratum = episodes // 2
    accepted_per_stratum = min(
        needed_per_stratum,
        len(valid["generated_harmful"]),
        len(valid["generated_benign"]),
    )
    accepted: list[dict[str, Any]] = []
    for ordinal in range(accepted_per_stratum):
        for prompt_type in ("generated_harmful", "generated_benign"):
            row = dict(valid[prompt_type][ordinal])
            prior_episode_index = row.get("episode_index")
            if prior_episode_index is not None:
                row["source_episode_index"] = int(prior_episode_index)
            row["episode_index"] = len(accepted)
            accepted.append(row)
    deficits = {
        prompt_type: max(0, needed_per_stratum - len(rows))
        for prompt_type, rows in valid.items()
    }
    return {
        "complete": not any(deficits.values()),
        "episodes": accepted,
        "candidate_count": len(candidates),
        "requested_episodes": episodes,
        "accepted_count": len(accepted),
        "valid_counts": {key: len(value) for key, value in valid.items()},
        "deficits": deficits,
        "dropped_counts": {
            "total": sum(dropped_by_reason.values()),
            "by_reason": dropped_by_reason,
            "harmful": dropped_by_stratum["generated_harmful"],
            "benign": dropped_by_stratum["generated_benign"],
        },
    }


def analyze_zero_sum_convergence(
    episodes: Sequence[Mapping[str, Any]],
    *,
    sample_counts: Sequence[int],
    max_confidence_radius: float = 0.10,
    max_mean_drift: float = 0.05,
    stable_windows: int = 3,
    min_convergence_episodes: int = 256,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Analyze nested 50/50 prefixes using bounded simultaneous intervals."""

    if max_confidence_radius <= 0:
        raise ValueError("max_confidence_radius must be positive")
    if max_mean_drift < 0:
        raise ValueError("max_mean_drift must be non-negative")
    if stable_windows <= 0:
        raise ValueError("stable_windows must be positive")
    if min_convergence_episodes < 256 or min_convergence_episodes % 2:
        raise ValueError(
            "min_convergence_episodes must be even and at least 256"
        )
    if not 0 < familywise_alpha < 1:
        raise ValueError("familywise_alpha must be between zero and one")

    ordered = list(episodes)
    for index, row in enumerate(ordered):
        if int(row.get("episode_index", -1)) != index:
            raise ValueError("episodes must be one contiguous ordered prefix")
        expected_type = (
            "generated_harmful" if index % 2 == 0 else "generated_benign"
        )
        if row.get("prompt_type") != expected_type:
            raise ValueError(
                "episodes must be exactly harmful/benign interleaved at "
                f"index {index}"
            )
        value = float(row["attacker_zero_sum_reward"])
        if not ZERO_SUM_REWARD_BOUNDS[0] <= value <= ZERO_SUM_REWARD_BOUNDS[1]:
            raise ValueError(
                f"attacker zero-sum reward is outside [-1, 1] at index {index}"
            )
        if float(row["defender_zero_sum_reward"]) != -value:
            raise ValueError(f"non-zero-sum episode at index {index}")

    counts = [int(value) for value in sample_counts]
    if (
        not counts
        or counts != sorted(set(counts))
        or any(value < 2 or value % 2 for value in counts)
        or counts[-1] > len(ordered)
    ):
        raise ValueError("sample_counts must be increasing even nested prefixes")

    # One payoff and its H/B strata are the only simultaneous series.  The
    # defender is an algebraic negative, not a fourth statistical estimate.
    simultaneous_series = 3
    per_interval_alpha = familywise_alpha / (
        len(counts) * simultaneous_series
    )
    gates: list[dict[str, Any]] = []
    previous_mean: float | None = None
    stable_run = 0
    required_episodes: int | None = None

    for count in counts:
        prefix = ordered[:count]
        values_by_series = {
            "overall": [
                float(row["attacker_zero_sum_reward"]) for row in prefix
            ],
            "harmful": [
                float(row["attacker_zero_sum_reward"])
                for row in prefix
                if row["prompt_type"] == "generated_harmful"
            ],
            "benign": [
                float(row["attacker_zero_sum_reward"])
                for row in prefix
                if row["prompt_type"] == "generated_benign"
            ],
        }
        descriptive = {
            name: mean_ci95(values)
            for name, values in values_by_series.items()
        }
        simultaneous = {
            name: bounded_empirical_bernstein_interval(
                values,
                alpha=per_interval_alpha,
                lower_bound=ZERO_SUM_REWARD_BOUNDS[0],
                upper_bound=ZERO_SUM_REWARD_BOUNDS[1],
            )
            for name, values in values_by_series.items()
        }
        current_mean = float(descriptive["overall"]["mean"])
        drift = (
            None
            if previous_mean is None
            else abs(current_mean - previous_mean)
        )
        confidence_stable = all(
            report["confidence_radius"] is not None
            and float(report["confidence_radius"]) <= max_confidence_radius
            for report in simultaneous.values()
        )
        drift_stable = drift is not None and drift <= max_mean_drift
        stable = bool(
            count >= min_convergence_episodes
            and confidence_stable
            and drift_stable
        )
        stable_run = stable_run + 1 if stable else 0
        if required_episodes is None and stable_run >= stable_windows:
            required_episodes = count
        gates.append(
            {
                "episodes": count,
                **descriptive,
                "bounded_simultaneous": simultaneous,
                "mean_drift_from_previous_gate": drift,
                "confidence_stable": confidence_stable,
                "drift_stable": drift_stable,
                "stable": stable,
                "consecutive_stable_gates": stable_run,
            }
        )
        previous_mean = current_mean

    return {
        "reward_version": ZERO_SUM_REWARD_VERSION,
        "payoff_orientation": "attacker/maximizer",
        "reward_bounds": list(ZERO_SUM_REWARD_BOUNDS),
        "prompt_distribution": "exactly 50/50 harmful/benign interleaved",
        "criterion": {
            "confidence_method": (
                "bounded empirical-Bernstein intervals with Bonferroni "
                "allocation over pre-registered gates and overall/H/B series"
            ),
            "familywise_alpha": familywise_alpha,
            "pre_registered_gate_count": len(counts),
            "simultaneous_series": simultaneous_series,
            "per_interval_alpha": per_interval_alpha,
            "max_confidence_radius": max_confidence_radius,
            "max_mean_drift": max_mean_drift,
            "stable_windows": stable_windows,
            "min_convergence_episodes": min_convergence_episodes,
        },
        "converged": required_episodes is not None,
        "required_episodes": required_episodes,
        "gates": gates,
    }


def zero_sum_cell_cache_key(contract: Mapping[str, Any]) -> str:
    """Return a content-addressed key for an immutable payoff-cell contract."""

    required = {
        "attacker_adapter_sha256",
        "defender_adapter_sha256",
        "prompt_dataset_sha256",
        "seed_base",
        "episodes",
        "generation",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(f"cell contract is missing: {missing}")
    payload = {"reward_version": ZERO_SUM_REWARD_VERSION, **dict(contract)}
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def solve_zero_sum_meta_game(
    payoff_matrix: Sequence[Sequence[float]],
    *,
    max_exploitability: float = 0.02,
) -> dict[str, Any]:
    """Solve a rectangular zero-sum game and fail closed on a poor solution."""

    if max_exploitability < 0:
        raise ValueError("max_exploitability must be non-negative")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("NumPy is required for the zero-sum solver") from exc
    from roll.pipeline.agentic.meta_solver import compute_nash

    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("payoff_matrix must be non-empty and rectangular")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("payoff_matrix contains a non-finite value")

    attacker, defender = compute_nash(matrix)
    if (
        attacker.shape != (matrix.shape[0],)
        or defender.shape != (matrix.shape[1],)
        or np.any(attacker < 0)
        or np.any(defender < 0)
        or not math.isclose(float(attacker.sum()), 1.0, abs_tol=1e-8)
        or not math.isclose(float(defender.sum()), 1.0, abs_tol=1e-8)
    ):
        raise RuntimeError("meta-solver returned an invalid probability vector")

    value = float(attacker @ matrix @ defender)
    attacker_best_response_value = float(np.max(matrix @ defender))
    defender_best_response_value = float(np.min(attacker @ matrix))
    attacker_regret = attacker_best_response_value - value
    defender_regret = value - defender_best_response_value
    exploitability = attacker_best_response_value - defender_best_response_value
    if exploitability > max_exploitability:
        raise RuntimeError(
            "zero-sum meta-solver exploitability exceeds tolerance: "
            f"{exploitability:.8f} > {max_exploitability:.8f}"
        )
    return {
        "attacker_strategy": [float(value) for value in attacker],
        "defender_strategy": [float(value) for value in defender],
        "game_value": value,
        "attacker_best_response_value": attacker_best_response_value,
        "defender_best_response_value": defender_best_response_value,
        "attacker_regret": attacker_regret,
        "defender_regret": defender_regret,
        "exploitability": exploitability,
        "max_exploitability": max_exploitability,
        "solver": "hedge-50000-fail-closed",
    }
