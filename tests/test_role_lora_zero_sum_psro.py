"""Tests for the strict role-LoRA zero-sum PSRO contracts."""

from __future__ import annotations

import ast
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from roll.utils.role_lora_zero_sum_psro import (
    ZERO_SUM_REWARD_VERSION,
    ZeroSumEpisodeDrop,
    analyze_zero_sum_convergence,
    assemble_valid_zero_sum_prefix,
    compute_zero_sum_psro_reward,
    rescore_zero_sum_episodes,
    solve_zero_sum_meta_game,
    zero_sum_cell_cache_key,
)
from roll.utils.role_lora_training_reward import (
    TRAINING_LABEL_DRIFT_POLICY,
    cap_label_drift_attacker_reward,
    label_drift_training_action,
)


def _labels(prompt: str, response: str, refusal: str) -> dict[str, object]:
    return {
        "prompt_harmfulness": prompt,
        "response_harmfulness": response,
        "response_refusal": refusal,
        "is_parsing_error": False,
    }


class ZeroSumRewardTest(unittest.TestCase):
    def test_harmful_safe_refusal_is_defender_win(self):
        score = compute_zero_sum_psro_reward(
            prompt_type="generated_harmful",
            labels=_labels("harmful", "unharmful", "refusal"),
        )
        self.assertEqual(score["attacker_zero_sum_reward"], -0.75)
        self.assertEqual(score["defender_zero_sum_reward"], 0.75)
        self.assertEqual(score["attacker_terminal_general_sum_reward"], -1.0)
        self.assertEqual(score["defender_terminal_general_sum_reward"], 2.0)

    def test_harmful_compliance_is_attacker_win(self):
        score = compute_zero_sum_psro_reward(
            prompt_type="generated_harmful",
            labels=_labels("harmful", "harmful", "compliance"),
        )
        self.assertEqual(score["attacker_zero_sum_reward"], 1.0)
        self.assertEqual(score["defender_zero_sum_reward"], -1.0)

    def test_benign_safe_compliance_is_defender_win(self):
        score = compute_zero_sum_psro_reward(
            prompt_type="generated_benign",
            labels=_labels("unharmful", "unharmful", "compliance"),
        )
        self.assertEqual(score["attacker_zero_sum_reward"], -0.75)
        self.assertEqual(score["defender_zero_sum_reward"], 0.75)

    def test_benign_to_harmful_label_drift_is_dropped(self):
        with self.assertRaises(ZeroSumEpisodeDrop) as raised:
            compute_zero_sum_psro_reward(
                prompt_type="generated_benign",
                labels=_labels("harmful", "harmful", "compliance"),
            )
        self.assertEqual(
            raised.exception.reason,
            "benign_to_harmful_label_drift",
        )

    def test_harmful_to_benign_drift_uses_normal_zero_sum_reward(self):
        score = compute_zero_sum_psro_reward(
            prompt_type="generated_harmful",
            labels=_labels("unharmful", "unharmful", "compliance"),
        )
        self.assertEqual(score["attacker_zero_sum_reward"], -1.0)
        self.assertEqual(score["defender_zero_sum_reward"], 1.0)

    def test_parse_error_must_be_resampled(self):
        labels = _labels("harmful", "unharmful", "refusal")
        labels["is_parsing_error"] = True
        with self.assertRaises(ZeroSumEpisodeDrop) as raised:
            compute_zero_sum_psro_reward(
                prompt_type="generated_harmful", labels=labels
            )
        self.assertEqual(raised.exception.reason, "wildguard_parse_error")


class TrainingLabelDriftPolicyTest(unittest.TestCase):
    def action(self, role: str, prompt_type: str, label: str | None) -> str:
        return label_drift_training_action(
            optimizer_train_role=role,
            prompt_type=prompt_type,
            prompt_harmfulness=label,
            policy=TRAINING_LABEL_DRIFT_POLICY,
        )

    def test_attacker_drift_caps_both_seed_directions(self):
        self.assertEqual(
            self.action("attacker", "generated_benign", "harmful"),
            "cap_zero",
        )
        self.assertEqual(
            self.action("attacker", "generated_harmful", "unharmful"),
            "cap_zero",
        )
        self.assertEqual(cap_label_drift_attacker_reward(2.0), 0.0)
        self.assertEqual(cap_label_drift_attacker_reward(-1.0), -1.0)

    def test_defender_drops_generated_drift(self):
        self.assertEqual(
            self.action("defender", "generated_benign", "harmful"),
            "drop",
        )
        self.assertEqual(
            self.action("defender", "generated_harmful", "unharmful"),
            "drop",
        )

    def test_consistent_and_vanilla_prompts_are_kept(self):
        self.assertEqual(
            self.action("attacker", "generated_benign", "unharmful"),
            "keep",
        )
        self.assertEqual(
            self.action("defender", "generated_harmful", "harmful"),
            "keep",
        )
        self.assertEqual(
            self.action("defender", "vanilla_benign", "harmful"),
            "keep",
        )


class ZeroSumProtocolTest(unittest.TestCase):
    def _episodes(self, count: int) -> list[dict[str, object]]:
        rows = []
        for index in range(count):
            harmful = index % 2 == 0
            rows.append(
                {
                    "episode_index": index,
                    "prompt_type": (
                        "generated_harmful"
                        if harmful
                        else "generated_benign"
                    ),
                    "dropped_reason": None,
                    "wildguard": _labels(
                        "harmful" if harmful else "unharmful",
                        "harmful" if harmful else "unharmful",
                        "compliance",
                    ),
                }
            )
        return rows

    def test_rescore_produces_exact_negatives(self):
        rows = rescore_zero_sum_episodes(self._episodes(4))
        self.assertEqual(
            [row["attacker_zero_sum_reward"] for row in rows],
            [1.0, -0.75, 1.0, -0.75],
        )
        self.assertTrue(
            all(
                row["defender_zero_sum_reward"]
                == -row["attacker_zero_sum_reward"]
                for row in rows
            )
        )

    def test_prefix_drops_parse_errors_and_rebalances_without_zero_fill(self):
        candidates = self._episodes(8)
        candidates[1]["wildguard"]["is_parsing_error"] = True
        for index, row in enumerate(candidates):
            row["candidate_index"] = index
        progress = assemble_valid_zero_sum_prefix(candidates, episodes=6)
        self.assertTrue(progress["complete"])
        self.assertEqual(progress["accepted_count"], 6)
        self.assertEqual(
            progress["dropped_counts"]["by_reason"],
            {"wildguard_parse_error": 1},
        )
        self.assertEqual(
            [row["prompt_type"] for row in progress["episodes"]],
            ["generated_harmful", "generated_benign"] * 3,
        )

    def test_prefix_drops_benign_to_harmful_and_restores_50_50(self):
        candidates = self._episodes(8)
        candidates[1]["wildguard"] = _labels(
            "harmful", "harmful", "compliance"
        )
        for index, row in enumerate(candidates):
            row["candidate_index"] = index
        progress = assemble_valid_zero_sum_prefix(candidates, episodes=6)
        self.assertTrue(progress["complete"])
        self.assertEqual(progress["accepted_count"], 6)
        self.assertEqual(
            progress["dropped_counts"]["by_reason"],
            {"benign_to_harmful_label_drift": 1},
        )
        self.assertEqual(
            [row["prompt_type"] for row in progress["episodes"]],
            ["generated_harmful", "generated_benign"] * 3,
        )

    def test_zero_episode_budget_selects_largest_balanced_prefix(self):
        candidates = self._episodes(8)
        candidates[1]["wildguard"] = _labels(
            "harmful", "harmful", "compliance"
        )
        for index, row in enumerate(candidates):
            row["candidate_index"] = index
        progress = assemble_valid_zero_sum_prefix(candidates, episodes=0)
        self.assertTrue(progress["complete"])
        self.assertEqual(progress["requested_episodes"], 6)
        self.assertEqual(progress["accepted_count"], 6)
        self.assertEqual(
            [row["prompt_type"] for row in progress["episodes"]],
            ["generated_harmful", "generated_benign"] * 3,
        )
    def test_convergence_tracks_one_payoff_and_three_series(self):
        rows = rescore_zero_sum_episodes(self._episodes(640))
        report = analyze_zero_sum_convergence(
            rows,
            sample_counts=[256, 384, 512, 640],
            max_confidence_radius=1.0,
        )
        self.assertTrue(report["converged"])
        self.assertEqual(report["required_episodes"], 640)
        self.assertEqual(report["criterion"]["simultaneous_series"], 3)

    def test_cache_key_is_content_addressed_and_versioned(self):
        contract = {
            "attacker_adapter_sha256": "a" * 64,
            "defender_adapter_sha256": "b" * 64,
            "prompt_dataset_sha256": "c" * 64,
            "seed_base": 8888,
            "episodes": 4096,
            "generation": {"temperature": 1.0},
        }
        first = zero_sum_cell_cache_key(contract)
        second = zero_sum_cell_cache_key(dict(reversed(list(contract.items()))))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(
            ZERO_SUM_REWARD_VERSION,
            "role-lora-psro-terminal-projection-v4",
        )

    def test_rectangular_solver_reports_regret_certificate(self):
        solved = solve_zero_sum_meta_game([[1.0, -1.0], [-1.0, 1.0]])
        self.assertAlmostEqual(solved["game_value"], 0.0, places=2)
        self.assertLessEqual(solved["exploitability"], 0.02)
        self.assertAlmostEqual(sum(solved["attacker_strategy"]), 1.0)
        self.assertAlmostEqual(sum(solved["defender_strategy"]), 1.0)

    def test_solver_does_not_fallback_on_invalid_matrix(self):
        with self.assertRaises(ValueError):
            solve_zero_sum_meta_game([[0.0], [float("nan")]])

    def test_training_keeps_general_sum_with_asymmetric_drift_policy(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository / "modal_upstream_selfredteam_role_lora_v2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("def _patch_zero_sum_psro_reward()", source)
        self.assertIn('"reward_type": "general_sum"', source)
        self.assertIn('"strict_zero_sum_ppo_reward": False', source)
        self.assertIn("def _patch_exact_prompt_label_balance()", source)
        self.assertIn("def _patch_asymmetric_label_drift_training()", source)
        self.assertIn(
            'label_drift_training_policy: str = TRAINING_LABEL_DRIFT_POLICY',
            source,
        )
        cold_source = (
            repository / "modal_role_lora_zero_sum_psro.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('reward_type="psro_zero_sum_v2"', cold_source)
        self.assertIn(
            '"training_reward": "original general_sum including existing shaping"',
            cold_source,
        )
        self.assertIn("exact_prompt_label_balance=True", cold_source)
        self.assertNotIn("drop_attack_label_drift_before_defense", cold_source)
        self.assertIn(
            '"training_label_drift_policy": TRAINING_LABEL_DRIFT_POLICY',
            cold_source,
        )
        self.assertIn(
            '"matrix_label_drift_policy": (',
            cold_source,
        )
        self.assertIn(
            'wandb_identity=f"role_lora_zero_sum_psro__{suffix}__A1"',
            (repository / "modal_role_lora_zero_sum_psro.py").read_text(
                encoding="utf-8"
            ),
        )

    def test_iteration_one_is_cold_for_both_roles_and_psro_is_switchable(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository / "modal_role_lora_zero_sum_psro.py"
        ).read_text(encoding="utf-8")
        # A1/D1 bootstrap remains cold.  Generic PSRO oracles select either
        # base or the previous same-role adapter from one cold_start setting.
        self.assertEqual(source.count("role_start_adapter=None"), 2)
        self.assertIn("cold_start: bool = True", source)
        self.assertIn("role_start_adapter=start_record.get(\"path\")", source)
        self.assertIn('"init_mode": init_mode', source)
        self.assertIn('"generated_harmful": 0.5', source)
        self.assertIn('"generated_benign": 0.5', source)
        self.assertIn('PSRO_OUTPUT_ROOT = Path(', source)
        self.assertNotIn("source.replace(\"selfplay-redteaming\"", source)

    def test_formal_psro_prunes_reproducible_storage_intermediates(self):
        repository = Path(__file__).resolve().parents[1]
        trainer = (
            repository / "modal_upstream_selfredteam_role_lora_v2.py"
        ).read_text(encoding="utf-8")
        coordinator = (
            repository / "modal_role_lora_zero_sum_psro.py"
        ).read_text(encoding="utf-8")
        launcher = (
            repository / "run_role_lora_cold_psro5_h200x4.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "def _prune_intermediate_hf_checkpoints(",
            trainer,
        )
        self.assertIn(
            '"checkpoint_retention": "final_hf_only"',
            coordinator,
        )
        self.assertIn("keep_only_final_hf_checkpoint=True", coordinator)
        self.assertIn(
            '_write_json_atomic(destination / "raw_manifest.json"',
            coordinator,
        )
        self.assertIn("raw_source_pruned", coordinator)
        self.assertIn("merged_checkpoint_pruned", coordinator)
        self.assertIn(
            'SAVE_STEPS="${SAVE_STEPS:-$STEPS_PER_ROLE}"',
            launcher,
        )
        self.assertIn('COLD_START="${COLD_START:-true}"', launcher)
        self.assertIn('START_MODE_ARGS=(--no-cold-start)', launcher)
        self.assertIn('"${START_MODE_ARGS[@]}"', launcher)

    def test_checkpoint_pruner_keeps_only_exact_final_hf_directory(self):
        repository = Path(__file__).resolve().parents[1]
        trainer_path = repository / "modal_upstream_selfredteam_role_lora_v2.py"
        tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_prune_intermediate_hf_checkpoints"
        )
        namespace = {"Path": Path, "re": re, "shutil": shutil}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                ),
                str(trainer_path),
                "exec",
            ),
            namespace,
        )
        prune = namespace["_prune_intermediate_hf_checkpoints"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "global_step10_hf",
                "global_step90_hf",
                "global_step100_hf",
                "unrelated",
            ):
                (root / name).mkdir()
            removed = prune(root, final_step=100)
            self.assertEqual(
                removed,
                ["global_step10_hf", "global_step90_hf"],
            )
            self.assertTrue((root / "global_step100_hf").is_dir())
            self.assertTrue((root / "unrelated").is_dir())


if __name__ == "__main__":
    unittest.main()
