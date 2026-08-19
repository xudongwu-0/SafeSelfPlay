"""Pure contracts for warm-start, latest-opponent role self-play."""

from __future__ import annotations

from typing import TypedDict


class LatestOpponentPhase(TypedDict):
    target: str
    role: str
    initialized_from: str
    opponent: str
    generation: int


def build_latest_opponent_schedule(
    *,
    first_generation: int = 2,
    last_generation: int = 5,
) -> list[LatestOpponentPhase]:
    """Build A_g vs D_(g-1), then D_g vs A_g, in strict order."""

    if first_generation < 2:
        raise ValueError("first_generation must be at least 2")
    if last_generation < first_generation:
        raise ValueError("last_generation must not precede first_generation")

    schedule: list[LatestOpponentPhase] = []
    for generation in range(first_generation, last_generation + 1):
        schedule.extend(
            [
                {
                    "target": f"A{generation}",
                    "role": "attacker",
                    "initialized_from": f"A{generation - 1}",
                    "opponent": f"D{generation - 1}",
                    "generation": generation,
                },
                {
                    "target": f"D{generation}",
                    "role": "defender",
                    "initialized_from": f"D{generation - 1}",
                    "opponent": f"A{generation}",
                    "generation": generation,
                },
            ]
        )
    return schedule
