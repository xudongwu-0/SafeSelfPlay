from __future__ import annotations

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path

from roll.utils.selfplay_baseline_repeat import (
    build_a3_baseline_repeat_contract,
    verify_a3_baseline_repeat_contract,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class BaselineRepeatContractTests(unittest.TestCase):
    def _state(self, root: Path) -> dict:
        return {
            "schema_version": 1,
            "run_suffix": "run",
            "status": "stage_target_not_reached",
            "active_stage": "A3",
            "config": {
                "attacker_max_steps": 100,
                "save_steps": 10,
                "attacker_learning_rate": 1e-5,
                "early_stop_threshold": 0.95,
                "early_stop_patience": 5,
                "early_stop_min_steps": 30,
            },
            "stages": {
                "A3": {
                    "status": "retained",
                    "transition_state": "retained",
                    "stopped_early": False,
                    "actual_final_step": 100,
                    "population_checkpoint": str(root / "population" / "A3"),
                    "sha256": _sha("A3"),
                },
                "D2": {
                    "status": "retained",
                    "transition_state": "retained",
                    "population_checkpoint": str(root / "population" / "D2"),
                    "sha256": _sha("D2"),
                },
            },
        }

    def test_contract_locks_original_recipe_and_no_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            contract = build_a3_baseline_repeat_contract(
                state=self._state(root),
                root=root,
                frozen_training_sha256={"core.py": _sha("core")},
                implementation_sha256={"wrapper.py": _sha("wrapper")},
            )
        self.assertEqual(verify_a3_baseline_repeat_contract(contract), contract["contract_id"])
        self.assertTrue(contract["recipe"]["enable_aux_sft"])
        self.assertEqual(contract["recipe"]["postfill_cot_stop_after_step"], 30)
        self.assertFalse(contract["recipe"]["defender_raw_reinforce_advantages"])
        self.assertEqual(contract["recipe"]["defender_reward_utility"], "upstream_additive")
        self.assertFalse(contract["canonical_population_mutation_allowed"])
        self.assertFalse(contract["successor_dispatch_allowed"])

    def test_rejects_non_exhausted_a3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            state = self._state(root)
            state["stages"]["A3"]["actual_final_step"] = 99
            with self.assertRaisesRegex(RuntimeError, "100 steps"):
                build_a3_baseline_repeat_contract(
                    state=state,
                    root=root,
                    frozen_training_sha256={"core.py": _sha("core")},
                    implementation_sha256={"wrapper.py": _sha("wrapper")},
                )

    def test_contract_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            contract = build_a3_baseline_repeat_contract(
                state=self._state(root),
                root=root,
                frozen_training_sha256={"core.py": _sha("core")},
                implementation_sha256={"wrapper.py": _sha("wrapper")},
            )
        contract["recipe"]["steps"] = 101
        with self.assertRaisesRegex(RuntimeError, "hash drifted"):
            verify_a3_baseline_repeat_contract(contract)

    def test_wrapper_passes_the_same_frozen_trainer_keyword_surface(self) -> None:
        root = Path(__file__).resolve().parents[1]

        def keywords(path: Path, function_name: str) -> list[str]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            call = next(
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and "train_upstream_attacker_lora_fixed_seed.remote"
                in ast.unparse(node.func)
            )
            return [str(item.arg) for item in call.keywords]

        original = keywords(
            root / "modal_role_lora_selfplay8.py",
            "train_role_lora_selfplay8_stage",
        )
        repeated = keywords(
            root / "modal_role_lora_selfplay8_baseline_repeat.py",
            "run_a3_original_framework_baseline_repeat",
        )
        self.assertEqual(repeated, original)


if __name__ == "__main__":
    unittest.main()
