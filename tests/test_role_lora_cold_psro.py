from __future__ import annotations

from pathlib import Path

import pytest

from roll.utils.role_lora_cold_psro import (
    missing_matrix_cells,
    next_action,
    opponent_pool,
    payoff_matrix,
)


BASE = {
    "A0": {"path": None, "adapter_sha256": "base:model"},
    "D0": {"path": None, "adapter_sha256": "base:model"},
}


def test_sequential_double_oracle_schedule_includes_base() -> None:
    population = dict(BASE)
    cells: dict[str, dict[str, float]] = {}
    evaluations: dict[str, object] = {}

    assert next_action(population, cells, evaluations, generations=1) == {
        "kind": "cell",
        "attacker": "A0",
        "defender": "D0",
    }
    cells["A0__D0"] = {"value": 0.0}
    assert next_action(population, cells, evaluations, generations=1) == {
        "kind": "oracle",
        "role": "attacker",
        "target": "A1",
    }

    population["A1"] = {"path": "/A1", "adapter_sha256": "a1"}
    assert missing_matrix_cells(population, cells) == [("A1", "D0")]
    cells["A1__D0"] = {"value": 0.25}
    assert next_action(population, cells, evaluations, generations=1) == {
        "kind": "oracle",
        "role": "defender",
        "target": "D1",
    }

    population["D1"] = {"path": "/D1", "adapter_sha256": "d1"}
    assert missing_matrix_cells(population, cells) == [
        ("A0", "D1"),
        ("A1", "D1"),
    ]
    cells["A0__D1"] = {"value": -0.1}
    cells["A1__D1"] = {"value": 0.1}
    assert next_action(population, cells, evaluations, generations=1) == {
        "kind": "evaluation",
        "defender": "D1",
    }
    evaluations["D1"] = {"completed": True}
    assert next_action(population, cells, evaluations, generations=1) == {
        "kind": "complete"
    }
    assert next_action(population, cells, evaluations, generations=2) == {
        "kind": "oracle",
        "role": "attacker",
        "target": "A2",
    }


def test_matrix_and_positive_support_pool_are_ordered() -> None:
    population = {
        **BASE,
        "A1": {"path": "/A1", "adapter_sha256": "a1"},
        "D1": {"path": "/D1", "adapter_sha256": "d1"},
    }
    cells = {
        "A0__D0": {"value": 0.0},
        "A0__D1": {"value": -0.5},
        "A1__D0": {"value": 0.5},
        "A1__D1": {"value": 0.25},
    }
    attackers, defenders, matrix = payoff_matrix(population, cells)
    assert attackers == ["A0", "A1"]
    assert defenders == ["D0", "D1"]
    assert matrix == [[0.0, -0.5], [0.5, 0.25]]
    assert opponent_pool(defenders, [0.75, 0.25], population) == [
        {
            "id": "D0",
            "adapter": None,
            "sha256": "base:model",
            "probability": 0.75,
        },
        {
            "id": "D1",
            "adapter": "/D1",
            "sha256": "d1",
            "probability": 0.25,
        },
    ]


def test_incomplete_or_noncontiguous_population_fails_closed() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        missing_matrix_cells(
            {**BASE, "A2": {"path": "/A2", "adapter_sha256": "a2"}},
            {},
        )


def test_modal_sources_bind_pool_and_exact_retained_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "modal_upstream_selfredteam_role_lora_v2.py").read_text()
    payoff = (root / "modal_upstream_v2_payoff.py").read_text()
    coordinator = (root / "modal_role_lora_zero_sum_psro.py").read_text()
    assert "fixed_opponent_pool_json" in trainer
    assert "deterministic per-episode PSRO opponent selection" in trainer
    assert 'retention_policy="zero_sum_psro_v4"' in coordinator
    assert '"matrix_snapshots": {}' in coordinator
    assert "append-only generation extension" in coordinator
    assert 'final_snapshot_name = f"final_g{generations}"' in coordinator
    assert '"zero_sum_psro_v4"' in payoff
