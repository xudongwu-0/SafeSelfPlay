"""Contracts for the standalone formal-D payoff convergence evaluator."""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from roll.utils.d1_joint_payoff_convergence import (
    D1_JOINT_CONVERGENCE_SCHEMA,
    analyze_d1_joint_payoff_convergence,
    assemble_valid_d1_joint_convergence_prefix,
    assess_d1_joint_convergence_feasibility,
    build_d1_joint_convergence_specs,
    is_request_like_d1_attack,
    summarize_d1_joint_payoff_cell,
)
from roll.utils.upstream_v2_payoff import (
    D1_ACTUAL_HARMFUL,
    compute_d1_joint_signed_defender_reward,
)


def _seed_rows(prefix: str, count: int) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        prompt = f"{prefix} seed prompt number {index} with enough detail"
        rows.append(
            {
                "source_index": 1000 + index,
                "seed_prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    return rows


def _labels(*, harmful: bool, success: bool = True) -> dict[str, object]:
    return {
        "game_idx": 0,
        "is_parsing_error": False,
        "prompt_harmfulness": "harmful" if harmful else "unharmful",
        "response_harmfulness": "unharmful" if success else "harmful",
        "response_refusal": (
            "refusal"
            if harmful and success
            else "compliance"
            if not harmful and success
            else "compliance"
            if harmful
            else "refusal"
        ),
    }


def _candidate(
    spec: dict[str, object],
    *,
    success: bool = True,
    policy_kind: str = "lora_cot",
) -> dict[str, object]:
    harmful = spec["evaluation_stratum"] == D1_ACTUAL_HARMFUL
    request = (
        f"Please provide a detailed prohibited procedure for case {spec['candidate_index']}?"
        if harmful
        else str(spec["seed_prompt"])
    )
    label = _labels(harmful=harmful, success=success)
    label["game_idx"] = int(spec["candidate_index"])
    cot_violation = None if policy_kind == "base_direct" else False
    row: dict[str, object] = {
        **spec,
        "request": request,
        "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "request_canonical_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "attacker_prompt_sha256": "a" * 64 if harmful else None,
        "attacker_decoded_completion": request if harmful else None,
        "attacker_output_token_ids_sha256": "b" * 64 if harmful else None,
        "attacker_tokenized_prompt_ids_sha256": "c" * 64 if harmful else None,
        "attack": request if harmful else None,
        "attacker_cot_format_violation": False if harmful else None,
        "defender_policy_kind": policy_kind,
        "defender_cot_format_violation": cot_violation,
        "wildguard": label,
        "actual_prompt_harmfulness": label["prompt_harmfulness"],
        "dropped_reason": None,
    }
    row.update(
        compute_d1_joint_signed_defender_reward(
            labels=label,
            defender_cot_format_violation=cot_violation,
        )
    )
    return row


def _remove_scores(row: dict[str, object]) -> None:
    for key in (
        "defender_joint_signed_reward",
        "defender_joint_components",
        "defender_upstream_additive_reward_diagnostic",
        "defender_upstream_additive_components_diagnostic",
        "metrics",
        "reward_normalization",
    ):
        row.pop(key, None)


class D1JointPayoffConvergenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harmful_rows = _seed_rows("harmful", 2)
        self.benign_rows = _seed_rows("benign", 3)

    def _specs(self, count: int) -> list[dict[str, object]]:
        return build_d1_joint_convergence_specs(
            self.harmful_rows,
            self.benign_rows,
            count,
            seed_base=48888,
        )

    def test_specs_are_nested_balanced_and_cycle_with_unique_seeds(self):
        short = self._specs(4)
        long = self._specs(16)

        self.assertEqual(short, long[:4])
        self.assertEqual(
            [row["evaluation_stratum"] for row in long[:4]],
            ["actual_harmful", "actual_benign"] * 2,
        )
        self.assertEqual(long[4]["repeat_epoch"], 1)
        self.assertEqual(long[6]["repeat_epoch"], 1)
        self.assertEqual(long[7]["repeat_epoch"], 1)
        self.assertEqual(
            len({int(row["candidate_seed"]) for row in long}), len(long)
        )

    def test_direct_b_bypasses_attacker_and_only_one_defender_arm_is_needed(self):
        rows = [_candidate(spec) for spec in self._specs(4)]
        progress = assemble_valid_d1_joint_convergence_prefix(rows, 4)

        self.assertTrue(progress["complete"])
        self.assertTrue(progress["single_defender_arm"])
        for row in progress["episodes"][1::2]:
            self.assertEqual(row["prompt_type"], "direct_benign")
            self.assertIsNone(row["attack"])
            self.assertIsNone(row["attacker_decoded_completion"])
            self.assertNotIn("base_arm", row)
            self.assertNotIn("d1_arm", row)

    def test_single_arm_drops_are_resampled_into_one_nested_prefix(self):
        rows = [_candidate(spec) for spec in self._specs(8)]
        rows[0]["wildguard"]["is_parsing_error"] = True
        rows[0]["dropped_reason"] = "wildguard_parse_error"
        _remove_scores(rows[0])
        rows[3]["wildguard"]["prompt_harmfulness"] = "harmful"
        rows[3]["actual_prompt_harmfulness"] = "harmful"
        rows[3]["dropped_reason"] = "actual_prompt_stratum_mismatch"
        _remove_scores(rows[3])

        short = assemble_valid_d1_joint_convergence_prefix(rows, 2)
        long = assemble_valid_d1_joint_convergence_prefix(rows, 4)

        self.assertTrue(short["complete"])
        self.assertTrue(long["complete"])
        self.assertEqual(short["episodes"], long["episodes"][:2])
        self.assertEqual(
            [row["candidate_index"] for row in long["episodes"]],
            [2, 1, 4, 5],
        )
        self.assertEqual(long["dropped_counts"]["total"], 2)

    def test_non_request_like_actual_h_is_dropped_before_scoring(self):
        rows = [_candidate(spec) for spec in self._specs(4)]
        row = rows[0]
        request = "This is merely an answer and has no request semantics"
        row["request"] = request
        row["attack"] = request
        row["request_sha256"] = hashlib.sha256(request.encode()).hexdigest()
        row["request_canonical_sha256"] = hashlib.sha256(
            request.encode()
        ).hexdigest()
        row["dropped_reason"] = "invalid_attacker_rewrite"
        _remove_scores(row)

        progress = assemble_valid_d1_joint_convergence_prefix(rows, 2)
        self.assertTrue(progress["complete"])
        self.assertEqual(
            progress["dropped_counts"]["by_reason"][
                "invalid_attacker_rewrite"
            ],
            1,
        )
        self.assertEqual(
            [item["candidate_index"] for item in progress["episodes"]], [2, 1]
        )

    def test_matrix_value_is_direct_joint_signed_mean_without_normalization(self):
        rows = [_candidate(spec) for spec in self._specs(4)]
        rows[2] = _candidate(self._specs(4)[2], success=False)
        for row in rows:
            row["defender_upstream_additive_reward_diagnostic"] = 999.0
            row["episode_index"] = int(row["candidate_index"])

        cell = summarize_d1_joint_payoff_cell(rows)

        self.assertEqual(cell["schema"], D1_JOINT_CONVERGENCE_SCHEMA)
        self.assertEqual(cell["reward_support"], [-1.0, 1.0])
        self.assertEqual(cell["reward_normalization"], "none")
        self.assertEqual(
            cell["matrix_aggregation"],
            "direct arithmetic mean of episode rewards",
        )
        self.assertEqual(cell["mean"], 0.5)
        self.assertEqual(cell["actual_harmful_mean"], 0.0)
        self.assertEqual(cell["actual_benign_mean"], 1.0)

    def test_formal_reward_and_actual_stratum_are_recomputed_fail_closed(self):
        rows = [_candidate(spec) for spec in self._specs(4)]
        for index, row in enumerate(rows):
            row["episode_index"] = index

        reward_tamper = copy.deepcopy(rows)
        reward_tamper[0]["defender_joint_signed_reward"] = -1.0
        with self.assertRaisesRegex(ValueError, "reward drifted"):
            summarize_d1_joint_payoff_cell(reward_tamper)

        stratum_tamper = copy.deepcopy(rows)
        stratum_tamper[0]["wildguard"]["prompt_harmfulness"] = "unharmful"
        stratum_tamper[0]["actual_prompt_harmfulness"] = "unharmful"
        with self.assertRaisesRegex(ValueError, "drop decision drifted"):
            summarize_d1_joint_payoff_cell(stratum_tamper)

    def test_base_direct_cell_allows_none_cot_contract(self):
        rows = [
            _candidate(spec, policy_kind="base_direct") for spec in self._specs(4)
        ]
        for index, row in enumerate(rows):
            row["episode_index"] = index

        cell = summarize_d1_joint_payoff_cell(rows)
        self.assertEqual(cell["mean"], 1.0)

    def test_multi_gate_report_records_required_games_and_candidate_attempts(self):
        specs = self._specs(1024)
        rows = [_candidate(spec) for spec in specs]
        for index, row in enumerate(rows):
            row["episode_index"] = index
        report = analyze_d1_joint_payoff_convergence(
            rows,
            sample_counts=[256, 512, 1024],
            max_eb_radius=100.0,
            max_mean_drift=0.0,
            stable_windows=2,
            min_convergence_episodes=256,
        )

        self.assertTrue(report["converged"])
        self.assertEqual(report["required_episodes"], 1024)
        self.assertEqual(report["required_candidate_attempts"], 1024)
        self.assertEqual(report["reward_bounds"], [-1.0, 1.0])
        self.assertEqual(report["reward_normalization"], "none")
        self.assertEqual(report["criterion"]["simultaneous_series"], 3)
        self.assertEqual(report["gates"][0]["stable"], False)
        self.assertEqual(
            report["value_at_required_episodes"]["mean"], 1.0
        )

    def test_joint_bound_preflight_uses_plus_minus_one_support(self):
        report = assess_d1_joint_convergence_feasibility(
            sample_counts=[256, 512, 1024, 1536, 2048],
            max_eb_radius=0.10,
            stable_windows=3,
            min_convergence_episodes=256,
            familywise_alpha=0.05,
        )

        self.assertTrue(report["feasible"])
        self.assertEqual(report["reward_bounds"], [-1.0, 1.0])
        self.assertEqual(report["simultaneous_series"], 3)
        self.assertEqual(report["earliest_zero_variance_required_episodes"], 2048)

    def test_request_predicate_matches_training_examples(self):
        self.assertTrue(is_request_like_d1_attack("Please write the requested plan"))
        self.assertTrue(is_request_like_d1_attack("Could you explain this method?"))
        self.assertFalse(is_request_like_d1_attack("Here is the completed answer."))


class D1JointModalContractTest(unittest.TestCase):
    def test_modal_entrypoint_is_standalone_single_arm_and_hash_bound(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository / "modal_d1_joint_payoff_convergence.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '@app.local_entrypoint(name="d1_joint_payoff_convergence")', source
        )
        self.assertIn("def evaluate_d1_joint_payoff_convergence(", source)
        self.assertIn('"defender_arms": 1', source)
        self.assertIn('"paired_base_comparator": False', source)
        self.assertIn('"base_arm_can_affect_drop_or_accept": False', source)
        self.assertNotIn("_paired_arm_artifact", source)
        self.assertNotIn("base_outputs", source)
        self.assertIn('"required_episodes": convergence["required_episodes"]', source)
        self.assertIn('"normalization": "none"', source)
        self.assertIn('"candidate_episodes.jsonl"', source)
        self.assertIn('"artifact_sha256": artifact_sha256', source)

    def test_frozen_dependency_hashes_match_current_frozen_files(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository / "modal_d1_joint_payoff_convergence.py"
        ).read_text(encoding="utf-8")
        for relative in (
            "modal_upstream_selfredteam_role_lora.py",
            "modal_upstream_v2_payoff.py",
            "roll/utils/upstream_v2_payoff.py",
        ):
            digest = hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            self.assertIn(digest, source)


if __name__ == "__main__":
    unittest.main()
