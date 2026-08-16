"""Regression tests for the recovered role-LoRA v2 training recipe."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_MODULE = REPO_ROOT / "modal_upstream_selfredteam_role_lora.py"
FIXED_MODULE = REPO_ROOT / "modal_upstream_selfredteam_fixed_seed.py"
UPSTREAM_ACTOR = (
    REPO_ROOT.parent
    / "selfplay-redteaming/openrlhf/trainer/ray/ppo_actor.py"
)


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment: {name}")


def _load_functions(names: set[str], namespace: dict):
    tree = ast.parse(ROLE_MODULE.read_text(), filename=str(ROLE_MODULE))
    functions = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError(f"Missing functions: {names}")
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[])
    )
    exec(compile(module, str(ROLE_MODULE), "exec"), namespace)
    return namespace


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize or not add_generation_prompt:
            raise AssertionError("Unexpected fake-tokenizer invocation")
        rendered = "<bos>"
        for message in messages:
            rendered += f"<{message['role']}>{message['content']}<eot>"
        return rendered + "<assistant>"

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool):
        if add_special_tokens:
            raise AssertionError("Unexpected special-token request")
        return list(text.encode("utf-8"))


class RoleLoRAV2RecipeTest(unittest.TestCase):
    @staticmethod
    def _checkpoint_validation_namespace():
        namespace = {
            "Path": Path,
            "hashlib": hashlib,
            "_HF_CHECKPOINT_RE": re.compile(r"^global_step([0-9]+)_hf$"),
        }
        return _load_functions(
            {
                "_is_complete_hf_checkpoint",
                "_latest_complete_hf_checkpoint",
                "_checkpoint_weight_digest",
                "_validate_role_checkpoints",
            },
            namespace,
        )

    @staticmethod
    def _write_checkpoint(root: Path, step: int) -> str:
        checkpoint = root / f"global_step{step}_hf"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}")
        payload = f"role-lora-v2-step-{step}".encode()
        (checkpoint / "adapter_model.safetensors").write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def test_continuation_sft_starts_at_the_rollout_prefill_boundary(self):
        namespace = {
            "Path": Path,
            "json": json,
            "hashlib": hashlib,
            "ATTACKER_ROLE_SFT_ROWS": 2,
            "ATTACKER_SYSTEM_PROMPT": _literal_assignment(
                ROLE_MODULE, "ATTACKER_SYSTEM_PROMPT"
            ),
            "ATTACKER_COT_RESPONSE_FORMAT_SECTION": _literal_assignment(
                ROLE_MODULE, "ATTACKER_COT_RESPONSE_FORMAT_SECTION"
            ),
            "ASSISTANT_THINKING_PREFIX": _literal_assignment(
                ROLE_MODULE, "ASSISTANT_THINKING_PREFIX"
            ),
            "OPTIMIZED_HARMFUL_PROMPT": _literal_assignment(
                FIXED_MODULE, "OPTIMIZED_HARMFUL_PROMPT"
            ),
            "OPTIMIZED_BENIGN_PROMPT": _literal_assignment(
                FIXED_MODULE, "OPTIMIZED_BENIGN_PROMPT"
            ),
        }
        _load_functions({"_write_attacker_v2_continuation_sft"}, namespace)
        rows = [
            {
                "id": "benign",
                "target_rewrite": "Could you explain this safely?",
                "metadata": {"label": "benign", "seed_prompt": "seed b"},
            },
            {
                "id": "harmful",
                "target_rewrite": "Provide the requested harmful payload.",
                "metadata": {"label": "harmful", "seed_prompt": "seed h"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            destination = Path(directory) / "continuation.jsonl"
            output, metadata = namespace[
                "_write_attacker_v2_continuation_sft"
            ](source, _FakeTokenizer(), destination)
            rendered = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(metadata["label_counts"], {"harmful": 1, "benign": 1})
        for row in rendered:
            prompt = row["prompt_messages"]
            completion = row["completion_messages"]
            self.assertTrue(
                prompt.endswith(namespace["ASSISTANT_THINKING_PREFIX"])
            )
            self.assertTrue(completion.startswith(" I should"))
            self.assertFalse(completion.startswith("<answer>"))
            self.assertEqual(completion.count("</think>"), 1)
            self.assertEqual(completion.count("<answer>"), 1)
            self.assertTrue(completion.endswith("</answer>"))
        self.assertIn("test over-refusal", rendered[0]["completion_messages"])
        self.assertIn("harmful capability", rendered[1]["completion_messages"])

    def test_v2_online_sft_patch_uses_raw_continuation_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            actor.parent.mkdir(parents=True)
            actor.write_text(UPSTREAM_ACTOR.read_text())
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {
                    "_replace_once",
                    "_patch_upstream_role_specific_online_sft",
                },
                namespace,
            )
            namespace["_patch_upstream_role_specific_online_sft"](
                continuation_format=True
            )
            source = actor.read_text()

        self.assertIn("sft_strategy.args.apply_chat_template = False", source)
        self.assertIn("sft_strategy.args.sft_input_key = 'prompt_messages'", source)
        self.assertIn(
            "sft_strategy.args.sft_output_key = 'completion_messages'", source
        )
        self.assertIn("multiturn=attacker_role_sft and False", source)
        compile(source, str(UPSTREAM_ACTOR), "exec")

    def test_scheduler_patch_contains_constant_with_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            actor.parent.mkdir(parents=True)
            actor.write_text(UPSTREAM_ACTOR.read_text())
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {"_replace_once", "_patch_upstream_role_lr_scheduler"},
                namespace,
            )
            namespace["_patch_upstream_role_lr_scheduler"]()
            source = actor.read_text()

        self.assertIn('actor_lr_scheduler == "constant_with_warmup"', source)
        self.assertIn('"constant_with_warmup",', source)
        self.assertIn("max_steps * args.lr_warmup_ratio", source)
        compile(source, str(UPSTREAM_ACTOR), "exec")

    def test_v2_entrypoint_pins_the_successful_recipe(self):
        tree = ast.parse(ROLE_MODULE.read_text(), filename=str(ROLE_MODULE))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "role_lora_v2_reproduction"
        )
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "invoke"
        )
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        expected_literals = {
            "steps": 100,
            "rollout_batch_size": 128,
            "micro_rollout_batch_size": 8,
            "micro_train_batch_size": 8,
            "train_batch_size": 32,
            "save_steps": 10,
            "actor_learning_rate": 1e-5,
            "init_kl_coef": 0.0,
            "actor_lr_scheduler": "constant_with_warmup",
            "lr_warmup_ratio": 0.05,
            "enable_aux_sft": True,
            "upstream_invalid_handling": True,
            "attacker_init_adapter": "",
            "attacker_prompt_profile": "optimized",
            "lora_rank": 64,
            "lora_alpha": 64,
            "monitor_reference_kl": True,
            "postfill_cot_stop_after_step": 30,
            "role_specific_aux_sft": True,
            "v2_reproduction": True,
        }
        observed = {
            name: ast.literal_eval(keywords[name]) for name in expected_literals
        }
        self.assertEqual(observed, expected_literals)

        train_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_upstream_attacker_lora_fixed_seed"
        )
        train_source = ast.unparse(train_function)
        self.assertIn("'optimizer_train_role': train_role", train_source)
        self.assertIn(
            "custom_configs['postfill_cot_stop_after_step']",
            train_source,
        )

    def test_v2_checkpoint_validation_requires_and_hashes_full_cadence(self):
        namespace = self._checkpoint_validation_namespace()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            expected_hashes = {
                str(step): self._write_checkpoint(checkpoint_root, step)
                for step in range(10, 101, 10)
            }
            result = namespace["_validate_role_checkpoints"](
                checkpoint_root,
                100,
                10,
                require_complete_cadence=True,
            )

        self.assertEqual(result["expected_checkpoint_steps"], list(range(10, 101, 10)))
        self.assertEqual(result["expected_checkpoint_count"], 10)
        self.assertEqual(result["observed_expected_checkpoint_count"], 10)
        self.assertEqual(result["missing_checkpoint_steps"], [])
        self.assertEqual(result["expected_checkpoint_sha256"], expected_hashes)
        self.assertTrue(result["complete_cadence_verified"])

    def test_v2_checkpoint_validation_rejects_a_missing_intermediate_step(self):
        namespace = self._checkpoint_validation_namespace()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            for step in range(10, 101, 10):
                if step != 50:
                    self._write_checkpoint(checkpoint_root, step)
            with self.assertRaisesRegex(RuntimeError, r"missing=\[50\]"):
                namespace["_validate_role_checkpoints"](
                    checkpoint_root,
                    100,
                    10,
                    require_complete_cadence=True,
                )

    def test_sparse_checkpoint_cadence_remains_allowed_for_generic_runs(self):
        namespace = self._checkpoint_validation_namespace()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            self._write_checkpoint(checkpoint_root, 10)
            self._write_checkpoint(checkpoint_root, 100)
            result = namespace["_validate_role_checkpoints"](
                checkpoint_root,
                100,
                10,
            )

        self.assertEqual(result["missing_checkpoint_steps"], list(range(20, 100, 10)))
        self.assertFalse(result["complete_cadence_required"])
        self.assertFalse(result["complete_cadence_verified"])

    def test_llama_v2_audit_contract_requires_nonzero_b_224_of_224(self):
        namespace = _load_functions(
            {"_validate_lora_checkpoint_audit_contract"},
            {},
        )
        valid_stats = {
            "A": {"tensor_count": 224, "all_finite": True},
            "B": {
                "tensor_count": 224,
                "tensor_count_with_nonzero": 224,
                "all_finite": True,
            },
        }
        contract = namespace["_validate_lora_checkpoint_audit_contract"](
            448,
            valid_stats,
            require_llama_v2_contract=True,
        )
        self.assertEqual(contract["b_tensor_nonzero_coverage"], "224/224")
        self.assertTrue(contract["all_tensors_finite"])
        self.assertTrue(contract["passed"])

        invalid_stats = copy.deepcopy(valid_stats)
        invalid_stats["B"]["tensor_count_with_nonzero"] = 223
        with self.assertRaisesRegex(RuntimeError, "nonzero_B=223/224"):
            namespace["_validate_lora_checkpoint_audit_contract"](
                448,
                invalid_stats,
                require_llama_v2_contract=True,
            )

        nonfinite_stats = copy.deepcopy(valid_stats)
        nonfinite_stats["B"]["all_finite"] = False
        with self.assertRaisesRegex(RuntimeError, "all_finite=False"):
            namespace["_validate_lora_checkpoint_audit_contract"](
                448,
                nonfinite_stats,
                require_llama_v2_contract=True,
            )


if __name__ == "__main__":
    unittest.main()
