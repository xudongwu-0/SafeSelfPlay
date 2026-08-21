"""Pure scheduling helpers for the five-generation cold role-LoRA PSRO run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def population_labels(role: str, learned: int) -> list[str]:
    if role not in {"attacker", "defender"}:
        raise ValueError(f"invalid role: {role!r}")
    if learned < 0:
        raise ValueError("learned population size cannot be negative")
    prefix = "A" if role == "attacker" else "D"
    return [f"{prefix}{index}" for index in range(learned + 1)]


def learned_count(population: Mapping[str, Any], role: str) -> int:
    prefix = "A" if role == "attacker" else "D"
    indices = sorted(
        int(label[1:])
        for label in population
        if label.startswith(prefix) and label[1:].isdigit()
    )
    if not indices or indices[0] != 0 or indices != list(range(indices[-1] + 1)):
        raise ValueError(f"{role} population is not contiguous from {prefix}0")
    return indices[-1]


def missing_matrix_cells(
    population: Mapping[str, Any],
    cells: Mapping[str, Any],
) -> list[tuple[str, str]]:
    attackers = population_labels(
        "attacker", learned_count(population, "attacker")
    )
    defenders = population_labels(
        "defender", learned_count(population, "defender")
    )
    return [
        (attacker, defender)
        for attacker in attackers
        for defender in defenders
        if f"{attacker}__{defender}" not in cells
    ]


def payoff_matrix(
    population: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[list[float]]]:
    attackers = population_labels(
        "attacker", learned_count(population, "attacker")
    )
    defenders = population_labels(
        "defender", learned_count(population, "defender")
    )
    missing = missing_matrix_cells(population, cells)
    if missing:
        raise ValueError(f"payoff matrix is incomplete: {missing}")
    matrix = [
        [float(cells[f"{attacker}__{defender}"]["value"]) for defender in defenders]
        for attacker in attackers
    ]
    return attackers, defenders, matrix


def opponent_pool(
    labels: Sequence[str],
    probabilities: Sequence[float],
    population: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("opponent labels/probabilities must be non-empty and aligned")
    entries: list[dict[str, object]] = []
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in population:
            raise ValueError(f"missing opponent population record: {label}")
        probability = float(probability)
        if probability <= 0:
            continue
        record = population[label]
        entries.append(
            {
                "id": label,
                "adapter": record.get("path"),
                "sha256": record["adapter_sha256"],
                "probability": probability,
            }
        )
    if not entries:
        raise ValueError("meta-strategy has no positive-mass opponent")
    total = sum(float(entry["probability"]) for entry in entries)
    if abs(total - 1.0) > 1e-8:
        # The solver may return tiny zeroed coordinates. Renormalize only the
        # positive support and record exactly what the oracle samples.
        for entry in entries:
            entry["probability"] = float(entry["probability"]) / total
    return entries


def next_action(
    population: Mapping[str, Any],
    cells: Mapping[str, Any],
    evaluations: Mapping[str, Any],
    *,
    generations: int = 5,
) -> dict[str, Any]:
    """Return the only legal next action in sequential double-oracle PSRO."""

    if generations < 1:
        raise ValueError("generations must be positive")
    missing = missing_matrix_cells(population, cells)
    if missing:
        attacker, defender = missing[0]
        return {"kind": "cell", "attacker": attacker, "defender": defender}

    attacker_count = learned_count(population, "attacker")
    defender_count = learned_count(population, "defender")
    if attacker_count == defender_count:
        if attacker_count < generations:
            return {
                "kind": "oracle",
                "role": "attacker",
                "target": f"A{attacker_count + 1}",
            }
        for index in range(1, generations + 1):
            label = f"D{index}"
            if label not in evaluations:
                return {"kind": "evaluation", "defender": label}
        return {"kind": "complete"}
    if attacker_count == defender_count + 1:
        return {
            "kind": "oracle",
            "role": "defender",
            "target": f"D{attacker_count}",
        }
    raise ValueError(
        "illegal sequential double-oracle population sizes: "
        f"attackers={attacker_count}, defenders={defender_count}"
    )
