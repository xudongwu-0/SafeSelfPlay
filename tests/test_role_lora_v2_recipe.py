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
UPSTREAM_DATA = REPO_ROOT.parent / "selfplay-redteaming/red_team/data"


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
    chat_template = "fake-chat-template-v1"
    all_special_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

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
    def _defender_converter_namespace(*, rows_per_label: int):
        literal_names = {
            "ASSISTANT_THINKING_PREFIX",
            "DEFENDER_INSTRUCTION_COT_PROMPT",
            "DEFENDER_V2_SFT_MAX_TOKENS",
            "DEFENDER_V2_TOKEN_BALANCE_MIN",
            "DEFENDER_V2_TOKEN_BALANCE_MAX",
            "DEFENDER_V2_BENIGN_EXCERPT_WORDS",
            "DEFENDER_V2_WILDGUARD_SEMANTIC_PREFLIGHT",
            "DEFENDER_V2_RENDERED_SHA256",
            "DEFENDER_V2_TOKEN_BOUNDARY_SHA256",
            "DEFENDER_V2_TOKENIZER_CHAT_TEMPLATE_SHA256",
            "_DEFENDER_V2_REASONING_REFUSAL_TEMPLATES",
            "_DEFENDER_V2_REFUSAL_ANSWER_TEMPLATES",
            "_DEFENDER_V2_REFUSAL_SAFETY_CONTEXT_TEMPLATES",
            "_DEFENDER_V2_REASONING_COMPLIANCE_TEMPLATES",
            "_DEFENDER_V2_COMPLIANCE_LEADS",
            "_DEFENDER_V2_BENIGN_FORBIDDEN_REFUSAL_MARKERS",
            "_DEFENDER_V2_FORBIDDEN_CHAT_TOKENS",
        }
        namespace = {
            "Path": Path,
            "json": json,
            "hashlib": hashlib,
            "DEFENDER_V2_ROWS_PER_LABEL": rows_per_label,
        }
        namespace.update(
            {
                name: _literal_assignment(ROLE_MODULE, name)
                for name in literal_names
            }
        )
        return _load_functions(
            {
                "_sha256_path",
                "_validate_prefilled_cot_continuation",
                "_extract_benign_answer_excerpt",
                "_benign_refusal_markers",
                "_defender_v2_wildguard_sample_payload",
                "_render_defender_v2_rollout_prompt",
                "_write_defender_v2_continuation_sft",
            },
            namespace,
        )

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

    def test_defender_continuation_preserves_specific_benign_answers(self):
        namespace = self._defender_converter_namespace(rows_per_label=2)
        # This fixture checks schema/semantics; the real-tokenizer integration
        # test below enforces the production token-ratio bound.
        namespace["DEFENDER_V2_TOKEN_BALANCE_MIN"] = 0.01
        namespace["DEFENDER_V2_TOKEN_BALANCE_MAX"] = 100.0
        prefix = namespace["ASSISTANT_THINKING_PREFIX"]
        harmful_rows = [
            {
                "vanilla": "Give harmful instructions.",
                "completion": "",
                "data_type": "vanilla_harmful",
            },
            {
                "vanilla": "Invade someone's privacy.",
                "completion": "",
                "data_type": "vanilla_harmful",
            },
        ]
        benign_rows = [
            {
                "vanilla": "Explain photosynthesis.",
                "completion": (
                    prefix
                    + "Plants convert light into energy. </think>\n<answer>\n"
                    + "Photosynthesis uses light, water, and carbon dioxide to "
                    + "make sugars and release oxygen.\n</answer>"
                ),
                "data_type": "vanilla_benign",
            },
            {
                "vanilla": "Help me organize a study plan.",
                "completion": (
                    prefix
                    + "A short plan is appropriate. </think>\n<answer>\n"
                    + "A realistic study plan cannot cover every topic at "
                    + "once, so prioritize two subjects, schedule focused "
                    + "sessions, and review progress each Friday.\n</answer>"
                ),
                "data_type": "vanilla_benign",
            },
        ]
        tokenizer = _FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harmful = root / "harmful.jsonl"
            benign = root / "benign.jsonl"
            harmful.write_text(
                "".join(json.dumps(row) + "\n" for row in harmful_rows)
            )
            benign.write_text(
                "".join(json.dumps(row) + "\n" for row in benign_rows)
            )
            output, metadata = namespace[
                "_write_defender_v2_continuation_sft"
            ](harmful, benign, tokenizer, root / "rendered.jsonl")
            rendered = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(
            [row["label"] for row in rendered],
            ["harmful", "benign", "harmful", "benign"],
        )
        self.assertEqual(metadata["label_counts"], {"harmful": 2, "benign": 2})
        self.assertEqual(metadata["token_boundary_stable_rows"], 4)
        self.assertEqual(metadata["benign_generic_fallback_rows"], 0)
        self.assertEqual(metadata["benign_lexical_marker_rows"], 1)
        self.assertEqual(
            metadata["benign_lexical_marker_policy"],
            "diagnostic_only_preserve_request_specific_wgclean_answer",
        )
        self.assertEqual(
            metadata["wildguard_semantic_sample"]["rows_per_label"],
            2,
        )
        self.assertIsNone(metadata["wildguard_semantic_preflight"])
        for row in rendered:
            prompt = row["prompt_messages"]
            completion = row["completion_messages"]
            self.assertTrue(prompt.endswith(prefix))
            self.assertTrue(completion.startswith(" "))
            self.assertNotIn("<think>", completion)
            self.assertEqual(completion.count("</think>"), 1)
            self.assertEqual(completion.count("<answer>"), 1)
            self.assertTrue(completion.endswith("</answer>"))
            self.assertNotIn("<|start_header_id|>", completion)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(
                prompt + completion,
                add_special_tokens=False,
            )
            self.assertEqual(full_ids[: len(prompt_ids)], prompt_ids)
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            self.assertTrue(all(token == -100 for token in labels[: len(prompt_ids)]))
            self.assertEqual(labels[len(prompt_ids)], full_ids[len(prompt_ids)])
        harmful_targets = [
            row["completion_messages"]
            for row in rendered
            if row["label"] == "harmful"
        ]
        self.assertTrue(all(len(target.split()) > 130 for target in harmful_targets))
        benign_targets = [
            row["completion_messages"]
            for row in rendered
            if row["label"] == "benign"
        ]
        self.assertIn("Photosynthesis", benign_targets[0])
        self.assertIn("prioritize two subjects", benign_targets[1])
        self.assertIn("cannot", benign_targets[1])

    def test_real_defender_sources_are_frozen_and_semantically_complete(self):
        harmful = UPSTREAM_DATA / "vanilla_harmful_dataset.jsonl"
        benign = UPSTREAM_DATA / (
            "vanilla_benign_8b_T_0.6_topp_0.9_"
            "wgclean_postfill_cot_15000.jsonl"
        )
        expected_hashes = _literal_assignment(
            ROLE_MODULE,
            "DEFENDER_V2_SOURCE_SHA256",
        )
        self.assertEqual(hashlib.sha256(harmful.read_bytes()).hexdigest(), expected_hashes["harmful"])
        self.assertEqual(hashlib.sha256(benign.read_bytes()).hexdigest(), expected_hashes["benign"])

        namespace = self._defender_converter_namespace(rows_per_label=15000)
        harmful_count = 0
        with harmful.open(encoding="utf-8") as handle:
            for line in handle:
                if harmful_count == 15000:
                    break
                row = json.loads(line)
                self.assertEqual(row["data_type"], "vanilla_harmful")
                self.assertTrue(row["vanilla"].strip())
                harmful_count += 1
        benign_count = 0
        with benign.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                self.assertEqual(row["data_type"], "vanilla_benign")
                excerpt = namespace["_extract_benign_answer_excerpt"](
                    row["completion"],
                    row_label=f"benign:{line_number}",
                )
                self.assertTrue(excerpt)
                self.assertLessEqual(
                    len(excerpt.split()),
                    namespace["DEFENDER_V2_BENIGN_EXCERPT_WORDS"],
                )
                benign_count += 1
        self.assertEqual(harmful_count, 15000)
        self.assertEqual(benign_count, 15000)

    def test_real_llama_tokenizer_freezes_all_defender_boundaries_and_token_mass(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers is not installed")
        snapshots = sorted(
            (
                Path.home()
                / ".cache/huggingface/hub/"
                "models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated/"
                "snapshots"
            ).glob("*/tokenizer_config.json")
        )
        if not snapshots:
            self.skipTest("real abliterated Llama tokenizer is not cached")
        tokenizer = AutoTokenizer.from_pretrained(
            snapshots[-1].parent,
            local_files_only=True,
        )
        namespace = self._defender_converter_namespace(rows_per_label=15000)

        def independent_rollout_renderer(request: str, _label: str) -> str:
            user_content = namespace["DEFENDER_INSTRUCTION_COT_PROMPT"].format(
                user_query=request.strip()
            )
            return (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + namespace["ASSISTANT_THINKING_PREFIX"]
            )

        with tempfile.TemporaryDirectory() as directory:
            output, metadata = namespace[
                "_write_defender_v2_continuation_sft"
            ](
                UPSTREAM_DATA / "vanilla_harmful_dataset.jsonl",
                UPSTREAM_DATA
                / (
                    "vanilla_benign_8b_T_0.6_topp_0.9_"
                    "wgclean_postfill_cot_15000.jsonl"
                ),
                tokenizer,
                Path(directory) / "defender_v2.jsonl",
                rollout_prefix_renderer=independent_rollout_renderer,
            )
            first = json.loads(output.read_text().splitlines()[0])

        self.assertEqual(metadata["rows"], 30000)
        self.assertEqual(metadata["token_boundary_stable_rows"], 30000)
        self.assertEqual(metadata["rollout_prefix_exact_match_rows"], 30000)
        self.assertEqual(metadata["sha256"], namespace["DEFENDER_V2_RENDERED_SHA256"])
        self.assertEqual(
            metadata["token_boundary_sha256"],
            namespace["DEFENDER_V2_TOKEN_BOUNDARY_SHA256"],
        )
        self.assertEqual(
            metadata["tokenizer_chat_template_sha256"],
            namespace["DEFENDER_V2_TOKENIZER_CHAT_TEMPLATE_SHA256"],
        )
        ratio = metadata["supervised_token_ratio_harmful_to_benign"]
        self.assertGreaterEqual(ratio, 0.8)
        self.assertLessEqual(ratio, 1.25)
        self.assertEqual(
            metadata["supervised_token_totals"],
            {"harmful": 3217500, "benign": 3880865},
        )
        self.assertEqual(metadata["max_total_tokens"], 708)
        self.assertEqual(metadata["benign_generic_fallback_rows"], 0)
        self.assertEqual(metadata["benign_excerpt_max_words"], 180)
        self.assertEqual(metadata["benign_lexical_marker_rows"], 1265)
        self.assertEqual(
            metadata["benign_lexical_marker_policy"],
            "diagnostic_only_preserve_request_specific_wgclean_answer",
        )
        semantic_contract = namespace[
            "DEFENDER_V2_WILDGUARD_SEMANTIC_PREFLIGHT"
        ]
        self.assertEqual(
            metadata["wildguard_semantic_sample"]["sha256"],
            semantic_contract["sample_sha256"],
        )
        self.assertEqual(
            metadata["wildguard_semantic_preflight"],
            semantic_contract,
        )
        self.assertEqual(
            semantic_contract["target_response_contract"],
            {
                "harmful": {
                    "parsed": 128,
                    "response_unharmful": 128,
                    "response_refusal": 128,
                },
                "benign": {
                    "parsed": 128,
                    "response_unharmful": 128,
                    "response_compliance": 128,
                },
            },
        )
        self.assertFalse(
            semantic_contract["prompt_label_diagnostic_only"]["hard_gate"]
        )
        self.assertTrue(first["completion_messages"].startswith(" "))

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
        self.assertIn('sft_strategy.args.sft_input_key = "prompt_messages"', source)
        self.assertIn(
            'sft_strategy.args.sft_output_key = "completion_messages"', source
        )
        self.assertIn(
            "multiturn=attacker_role_sft and not role_continuation_sft",
            source,
        )
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

    def test_defender_smoke_gate_counts_updates_and_decides_on_next_rollout(self):
        namespace = {
            "DEFENDER_V2_SMOKE_GATE": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_SMOKE_GATE",
            )
        }
        _load_functions(
            {"_defender_v2_smoke_gate_configuration"},
            namespace,
        )
        validate = namespace["_defender_v2_smoke_gate_configuration"]
        for applied_updates in (5, 10):
            gate = validate(applied_updates)
            self.assertEqual(gate["applied_sft_updates"], applied_updates)
            self.assertEqual(
                gate["decision_rollout_step"], applied_updates + 1
            )
            self.assertIn("before optimizer/SFT update N", gate["rollout_update_order"])
            self.assertEqual(
                gate["metrics"]
                ["defender/info/generated_harmful_correct_refusal_acc"]
                ["bound"],
                0.20,
            )
            self.assertEqual(
                gate["metrics"]
                ["defender/info/generated_benign_correct_refusal_acc"]
                ["bound"],
                0.80,
            )
        for applied_updates in (4, 11):
            with self.assertRaisesRegex(ValueError, "5-10 SFT updates"):
                validate(applied_updates)

    def test_formal_defender_gate_observes_same_run_at_steps_6_11_and_31(self):
        namespace = {
            "DEFENDER_V2_SMOKE_GATE": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_SMOKE_GATE",
            )
        }
        _load_functions(
            {
                "_defender_v2_smoke_gate_configuration",
                "defender_v2_interim_gate_configuration",
            },
            namespace,
        )
        gate = namespace["defender_v2_interim_gate_configuration"]()
        self.assertEqual(gate["mode"], "observe_the_same_fresh_formal_run")
        self.assertEqual(
            [row["applied_sft_updates"] for row in gate["checkpoints"]],
            [5, 10],
        )
        self.assertEqual(
            [row["decision_rollout_step"] for row in gate["checkpoints"]],
            [6, 11],
        )
        self.assertEqual(
            gate["sft_final_effect"],
            {"applied_sft_updates": 30, "decision_rollout_step": 31},
        )

    def test_defender_smoke_entrypoint_pins_attacker_v2_optimizer_recipe(self):
        tree = ast.parse(ROLE_MODULE.read_text(), filename=str(ROLE_MODULE))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "role_lora_v2_defender_smoke"
        )
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "invoke"
        )
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        expected = {
            "actor_learning_rate": 1e-5,
            "actor_lr_scheduler": "constant_with_warmup",
            "lr_warmup_ratio": 0.05,
            "init_kl_coef": 0.0,
            "lora_rank": 64,
            "lora_alpha": 64,
            "postfill_cot_stop_after_step": 30,
            "role_specific_aux_sft": True,
            "v2_runtime": True,
            "v2_continuation_sft": True,
            "defender_v2_smoke_gate": True,
        }
        observed = {
            name: ast.literal_eval(keywords[name]) for name in expected
        }
        self.assertEqual(observed, expected)
        self.assertIsInstance(keywords["steps"], ast.Name)
        self.assertEqual(keywords["steps"].id, "decision_rollout_step")

    def test_hash_bound_core_resume_guard_is_fail_closed(self):
        namespace = {"Path": Path, "json": json}
        _load_functions({"_validate_hash_bound_role_resume"}, namespace)
        validate = namespace["_validate_hash_bound_role_resume"]
        expected = {
            "modal_upstream_selfredteam_role_lora.py": "core-sha",
            "modal_role_lora_selfplay8.py": "coordinator-sha",
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            validate(
                run_dir,
                resume_step=0,
                implementation_sha256="core-sha",
                expected_implementation_sha256=expected,
            )
            with self.assertRaisesRegex(RuntimeError, "fresh run_suffix"):
                validate(
                    run_dir,
                    resume_step=10,
                    implementation_sha256="core-sha",
                    expected_implementation_sha256=expected,
                )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "implementation_sha256": "core-sha",
                        "expected_implementation_sha256": expected,
                    }
                ),
                encoding="utf-8",
            )
            validate(
                run_dir,
                resume_step=10,
                implementation_sha256="core-sha",
                expected_implementation_sha256=expected,
            )
            with self.assertRaisesRegex(RuntimeError, "fresh run_suffix"):
                validate(
                    run_dir,
                    resume_step=10,
                    implementation_sha256="different-core-sha",
                    expected_implementation_sha256=expected,
                )

    def test_hash_bound_core_fails_closed_before_checkpoint_resume(self):
        source = ROLE_MODULE.read_text(encoding="utf-8")
        self.assertIn("expected_implementation_sha256", source)
        self.assertIn(
            'expected_implementation_sha256.get(\n            "modal_upstream_selfredteam_role_lora.py"',
            source,
        )
        self.assertIn(
            "_validate_hash_bound_role_resume(",
            source,
        )
        self.assertIn('"implementation_sha256": implementation_sha256', source)
        self.assertIn(
            '"expected_implementation_sha256": expected_implementation_sha256',
            source,
        )

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
