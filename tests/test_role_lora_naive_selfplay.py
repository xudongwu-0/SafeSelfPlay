"""Contracts for the durable latest-opponent A2-through-D5 chain."""

from __future__ import annotations

import unittest
from pathlib import Path

from roll.utils.role_lora_naive_selfplay import (
    build_latest_opponent_schedule,
)


class LatestOpponentScheduleTest(unittest.TestCase):
    def test_a2_through_d5_has_exact_latest_opponent_lineage(self):
        schedule = build_latest_opponent_schedule(
            first_generation=2,
            last_generation=5,
        )
        self.assertEqual(
            [phase["target"] for phase in schedule],
            ["A2", "D2", "A3", "D3", "A4", "D4", "A5", "D5"],
        )
        self.assertEqual(
            [
                (
                    phase["target"],
                    phase["initialized_from"],
                    phase["opponent"],
                )
                for phase in schedule
            ],
            [
                ("A2", "A1", "D1"),
                ("D2", "D1", "A2"),
                ("A3", "A2", "D2"),
                ("D3", "D2", "A3"),
                ("A4", "A3", "D3"),
                ("D4", "D3", "A4"),
                ("A5", "A4", "D4"),
                ("D5", "D4", "A5"),
            ],
        )

    def test_invalid_generation_range_fails_closed(self):
        with self.assertRaises(ValueError):
            build_latest_opponent_schedule(
                first_generation=1,
                last_generation=5,
            )
        with self.assertRaises(ValueError):
            build_latest_opponent_schedule(
                first_generation=3,
                last_generation=2,
            )

    def test_modal_chain_pins_current_training_contract(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository / "modal_role_lora_zero_sum_psro.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def train_naive_latest_opponent_role(", source)
        self.assertIn("def advance_naive_latest_opponent(", source)
        self.assertIn("steps_per_role: int = 100", source)
        self.assertIn("exact_prompt_label_balance=True", source)
        self.assertIn(
            "label_drift_training_policy=TRAINING_LABEL_DRIFT_POLICY",
            source,
        )
        self.assertIn(
            '"psro_matrix_or_meta_solver_during_training": False',
            source,
        )
        self.assertIn("advance_naive_latest_opponent.spawn(", source)

    def test_canonical_launcher_targets_the_durable_chain(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository / "run_selfredteam_lora_next_round_h200x4.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('STEPS_PER_ROLE="${STEPS_PER_ROLE:-100}"', source)
        self.assertIn("modal_role_lora_zero_sum_psro.py::naive_selfplay_train", source)
        self.assertIn('--last-generation "$LAST_GENERATION"', source)
        self.assertNotIn("ATTACKER_START_ADAPTER", source)


if __name__ == "__main__":
    unittest.main()
