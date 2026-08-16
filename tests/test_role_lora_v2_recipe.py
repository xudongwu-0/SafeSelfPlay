"""Regression tests for the recovered role-LoRA v2 training recipe."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT.parent / "selfplay-redteaming"
ROLE_MODULE = REPO_ROOT / "modal_upstream_selfredteam_role_lora.py"
FIXED_MODULE = REPO_ROOT / "modal_upstream_selfredteam_fixed_seed.py"
UPSTREAM_ACTOR = UPSTREAM_ROOT / "openrlhf/trainer/ray/ppo_actor.py"
UPSTREAM_DATA = UPSTREAM_ROOT / "red_team/data"


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


def _load_path_functions(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
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
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _load_functions(names: set[str], namespace: dict):
    return _load_path_functions(ROLE_MODULE, names, namespace)


def _load_source_functions(source: str, names: set[str], namespace: dict):
    tree = ast.parse(source)
    functions = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError(f"Missing generated functions: {names}")
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[])
    )
    exec(compile(module, "<generated-upstream>", "exec"), namespace)
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

    def test_defender_fixed_sft_dose_fills_four_real_optimizer_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            relative_sources = (
                "openrlhf/trainer/ray/ppo_actor.py",
                "openrlhf/trainer/ppo_utils/replay_buffer.py",
                "openrlhf/trainer/ppo_utils/language_game.py",
                "openrlhf/datasets/prompts_dataset.py",
                "openrlhf/datasets/sft_dataset.py",
            )
            for relative_source in relative_sources:
                destination = upstream / relative_source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(UPSTREAM_ROOT / relative_source, destination)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            replay = upstream / "openrlhf/trainer/ppo_utils/replay_buffer.py"
            dataset = upstream / "openrlhf/datasets/sft_dataset.py"
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {
                    "_replace_once",
                    "_patch_upstream_fixed_attacker_lora",
                    "_patch_upstream_role_advantage_normalization",
                    "_patch_upstream_role_specific_online_sft",
                    "_patch_upstream_defender_fixed_sft_dose",
                },
                namespace,
            )
            _load_path_functions(
                FIXED_MODULE,
                {"_patch_upstream_attacker_only_sampling"},
                namespace,
            )
            namespace["_patch_upstream_attacker_only_sampling"]()
            namespace["_patch_upstream_fixed_attacker_lora"]()
            namespace["_patch_upstream_role_advantage_normalization"]()
            namespace["_patch_upstream_role_specific_online_sft"](
                continuation_format=True
            )
            namespace["_patch_upstream_defender_fixed_sft_dose"]()
            actor_source = actor.read_text()
            replay_source = replay.read_text()
            dataset_source = dataset.read_text()

        self.assertIn(
            "defender_sft_optimizer_slots_per_rollout", actor_source
        )
        self.assertIn(
            "for _ in range(filler_sft_slots)", actor_source
        )
        self.assertIn(
            "self._defender_sft_only_optimizer_step(global_steps)",
            actor_source,
        )
        self.assertIn(
            '"defender_sft/rollout_sft_optimizer_slots"', actor_source
        )
        self.assertIn(
            '"defender_sft/rollout_supervised_tokens"', actor_source
        )
        self.assertIn(
            'status_mean["total_sft_samples_trained"] = cumulative_samples',
            actor_source,
        )
        self.assertIn("defender_sft_runtime.json", actor_source)
        self.assertIn("no finite nonzero LoRA gradient", actor_source)
        self.assertIn("_fixed_defender_uses_sft_only_replay", replay_source)
        self.assertIn("self.items = []", replay_source)
        self.assertIn('self.sample_labels = processed_dataset["sample_label"]', dataset_source)
        self.assertIn('infos["sample_label"].append', dataset_source)
        # The historical A path is retained under the zero-slot branch.
        self.assertIn("else:\n            sft_samples_this_step = 0", actor_source)
        compile(actor_source, str(UPSTREAM_ACTOR), "exec")
        compile(replay_source, str(replay), "exec")
        compile(dataset_source, str(dataset), "exec")

    def test_fixed_defender_zero_rl_chain_reaches_four_sft_only_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            relative_sources = (
                "openrlhf/trainer/ray/ppo_actor.py",
                "openrlhf/trainer/ppo_utils/replay_buffer.py",
                "openrlhf/trainer/ppo_utils/language_game.py",
                "openrlhf/datasets/prompts_dataset.py",
                "openrlhf/datasets/sft_dataset.py",
            )
            for relative_source in relative_sources:
                destination = upstream / relative_source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(UPSTREAM_ROOT / relative_source, destination)
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {
                    "_replace_once",
                    "_patch_upstream_fixed_attacker_lora",
                    "_patch_upstream_role_advantage_normalization",
                    "_patch_upstream_role_specific_online_sft",
                    "_patch_upstream_defender_fixed_sft_dose",
                },
                namespace,
            )
            _load_path_functions(
                FIXED_MODULE,
                {"_patch_upstream_attacker_only_sampling"},
                namespace,
            )
            namespace["_patch_upstream_attacker_only_sampling"]()
            namespace["_patch_upstream_fixed_attacker_lora"]()
            namespace["_patch_upstream_role_advantage_normalization"]()
            namespace["_patch_upstream_role_specific_online_sft"](
                continuation_format=True
            )
            namespace["_patch_upstream_defender_fixed_sft_dose"]()
            replay_source = (
                upstream / "openrlhf/trainer/ppo_utils/replay_buffer.py"
            ).read_text()
            dataset_source = (
                upstream / "openrlhf/datasets/prompts_dataset.py"
            ).read_text()
            actor_source = (
                upstream / "openrlhf/trainer/ray/ppo_actor.py"
            ).read_text()

        replay_tree = ast.parse(replay_source)
        replay_helper = next(
            copy.deepcopy(node)
            for node in replay_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_fixed_defender_uses_sft_only_replay"
        )
        replay_class = next(
            node
            for node in replay_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "NaiveReplayBuffer"
        )
        method_names = {
            "__len__",
            "__getitem__",
            "normalize",
            "optimizer_train_role",
            "assert_single_train_role",
            "truncate_buffer",
        }
        methods = [
            copy.deepcopy(node)
            for node in replay_class.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ]
        self.assertEqual({node.name for node in methods}, method_names)
        harness = ast.ClassDef(
            name="ReplayHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        replay_namespace: dict[str, object] = {}
        replay_module = ast.fix_missing_locations(
            ast.Module(body=[replay_helper, harness], type_ignores=[])
        )
        exec(compile(replay_module, "<replay-harness>", "exec"), replay_namespace)

        defender_configs = {
            "optimizer_train_role": "defender",
            "defender_sft_optimizer_slots_per_rollout": 4,
        }
        attacker_configs = {
            "optimizer_train_role": "attacker",
            "defender_sft_optimizer_slots_per_rollout": 0,
        }
        allow_sft_only = replay_namespace[
            "_fixed_defender_uses_sft_only_replay"
        ]
        self.assertTrue(allow_sft_only(defender_configs, [0, 8, 8, 8]))
        self.assertTrue(allow_sft_only(defender_configs, [0, 0, 0, 0]))
        self.assertFalse(allow_sft_only(defender_configs, [8, 8, 8, 8]))
        self.assertFalse(allow_sft_only(attacker_configs, [0, 8, 8, 8]))

        class FakeStrategy:
            def __init__(self, custom_configs, gathered):
                self.args = SimpleNamespace(
                    custom_configs=custom_configs,
                    micro_train_batch_size=8,
                )
                self._gathered = iter(gathered)

            @staticmethod
            def is_rank_0():
                return True

            def all_gather(self, _value):
                return next(self._gathered)

            @staticmethod
            def print(*_args, **_kwargs):
                return None

        replay = object.__new__(replay_namespace["ReplayHarness"])
        replay.items = [
            SimpleNamespace(info={"game_role": "defender"})
            for _ in range(8)
        ]
        strategy = FakeStrategy(
            defender_configs,
            ([0, 1, 1, 1], [0, 8, 8, 8]),
        )
        replay.truncate_buffer(strategy, mode="batch")
        self.assertEqual(replay.items, [])
        replay.assert_single_train_role(strategy)
        replay.normalize(strategy, "advantages", role="defender")

        actor_helpers: dict[str, object] = {}
        _load_source_functions(
            actor_source,
            {
                "_defender_replay_dataloader_shuffle",
                "_defender_fixed_sft_filler_slots",
            },
            actor_helpers,
        )
        shuffle = actor_helpers["_defender_replay_dataloader_shuffle"](
            4, len(replay), False
        )
        self.assertFalse(shuffle)

        class StrictDataLoader:
            def __init__(self, dataset, *, shuffle):
                if shuffle and len(dataset) == 0:
                    raise ValueError("RandomSampler rejected empty dataset")
                self.dataset = dataset

            def __iter__(self):
                return iter(self.dataset.items)

        dataloader = StrictDataLoader(replay, shuffle=shuffle)
        rl_optimizer_slots = sum(1 for _ in dataloader)
        filler_slots = actor_helpers[
            "_defender_fixed_sft_filler_slots"
        ](4, True, rl_optimizer_slots)
        self.assertEqual(rl_optimizer_slots, 0)
        self.assertEqual(filler_slots, 4)
        self.assertEqual(
            actor_helpers["_defender_fixed_sft_filler_slots"](4, True, 1),
            3,
        )
        self.assertEqual(
            actor_helpers["_defender_fixed_sft_filler_slots"](4, True, 4),
            0,
        )
        self.assertEqual(
            actor_helpers["_defender_fixed_sft_filler_slots"](0, False, 0),
            0,
        )

        attacker_replay = object.__new__(replay_namespace["ReplayHarness"])
        attacker_replay.items = []
        attacker_strategy = FakeStrategy(
            attacker_configs,
            ([0, 1, 1, 1], [0, 8, 8, 8]),
        )
        with self.assertRaisesRegex(AssertionError, "No samples"):
            attacker_replay.truncate_buffer(attacker_strategy, mode="batch")

    def test_defender_metrics_use_actual_wildguard_prompt_strata(self):
        source = ROLE_MODULE.read_text(encoding="utf-8")
        self.assertIn(
            'item.info.get("actual_prompt_harmfulness") == "harmful"',
            source,
        )
        self.assertIn(
            'item.info.get("actual_prompt_harmfulness") == "unharmful"',
            source,
        )
        self.assertIn(
            "defender/wildguard_actual_harmful_joint_success", source
        )
        self.assertIn(
            "defender/wildguard_actual_benign_joint_success", source
        )
        self.assertIn(
            "only the disjoint paired-1024 gate authorizes A2", source
        )
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            replay = upstream / "openrlhf/trainer/ppo_utils/replay_buffer.py"
            replay.parent.mkdir(parents=True)
            shutil.copyfile(
                UPSTREAM_ROOT
                / "openrlhf/trainer/ppo_utils/replay_buffer.py",
                replay,
            )
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {"_replace_once", "_patch_upstream_defender_metric_keys"},
                namespace,
            )
            namespace["_patch_upstream_defender_metric_keys"]()
            generated = replay.read_text()
        self.assertIn(
            "actual_strata_accounting[\"joint_success_harmful\"]",
            generated,
        )
        self.assertIn(
            "defender/actual_harmful_attack_success_rate", generated
        )
        self.assertIn("defender/replay_actual_harmful_count", generated)
        replay_tree = ast.parse(generated)
        replay_class = next(
            node for node in replay_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "NaiveReplayBuffer"
        )
        metrics_method = next(
            node for node in replay_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_defender_metrics"
        )
        joint_metric = next(
            copy.deepcopy(node) for node in ast.walk(metrics_method)
            if isinstance(node, ast.FunctionDef)
            and node.name == "joint_success"
        )
        metric_outcomes = SimpleNamespace(
            DEFENDER_WIN_CORRECT_REFUSAL=object(),
            DEFENDER_WIN_SUCCESSFUL_DEFENSE=object(),
            TIE=object(),
        )
        metric_namespace = {"GameOutcome": metric_outcomes}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[joint_metric], type_ignores=[])
                ),
                "<joint-metric>",
                "exec",
            ),
            metric_namespace,
        )
        benign_missing_refusal = SimpleNamespace(
            info={
                "game_outcomes": {
                    metric_outcomes.DEFENDER_WIN_CORRECT_REFUSAL,
                    metric_outcomes.DEFENDER_WIN_SUCCESSFUL_DEFENSE,
                    metric_outcomes.TIE,
                },
                "cot_format_violation": False,
            }
        )
        self.assertFalse(
            metric_namespace["joint_success"](benign_missing_refusal)
        )
        compile(generated, str(replay), "exec")

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
        self.assertIn("actor_lr_warmup_steps_override", source)
        self.assertIn("num_warmup_steps=actor_lr_warmup_steps", source)
        compile(source, str(UPSTREAM_ACTOR), "exec")

    def test_v2_defender_keeps_raw_reinforce_advantage_signs(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            actor.parent.mkdir(parents=True)
            actor.write_text(UPSTREAM_ACTOR.read_text())
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {
                    "_replace_once",
                    "_patch_upstream_role_advantage_normalization",
                },
                namespace,
            )
            namespace["_patch_upstream_role_advantage_normalization"]()
            source = actor.read_text()

        helpers: dict[str, object] = {}
        _load_source_functions(
            source, {"_role_advantage_transform_mode"}, helpers
        )
        transform_mode = helpers["_role_advantage_transform_mode"]
        raw_args = SimpleNamespace(
            custom_configs={
                "defender_raw_reinforce_advantages": True,
                "defender_sft_optimizer_slots_per_rollout": 4,
            },
            advantage_estimator="reinforce",
            gamma=1.0,
            init_kl_coef=0.0,
            no_advantage_std_norm=False,
        )
        self.assertEqual(
            transform_mode(raw_args, "defender"),
            "raw_defender_reinforce",
        )
        # This legacy flag disables only division by std; it still selects
        # mean-centering and therefore cannot substitute for the raw-D mode.
        historical_args = SimpleNamespace(
            custom_configs={
                "defender_raw_reinforce_advantages": False,
                "defender_sft_optimizer_slots_per_rollout": 4,
            },
            advantage_estimator="reinforce",
            init_kl_coef=0.0,
            no_advantage_std_norm=True,
        )
        self.assertEqual(
            transform_mode(historical_args, "defender"), "normalize"
        )
        attacker_args = SimpleNamespace(
            custom_configs={
                "defender_raw_reinforce_advantages": False,
                "defender_sft_optimizer_slots_per_rollout": 0,
            },
            advantage_estimator="reinforce",
            init_kl_coef=0.0,
            no_advantage_std_norm=False,
        )
        self.assertEqual(
            transform_mode(attacker_args, "attacker"), "normalize"
        )
        for field, value, message in (
            ("advantage_estimator", "gae", "advantage_estimator"),
            ("gamma", 0.99, "gamma=1.0"),
            ("init_kl_coef", 0.01, "init_kl_coef"),
        ):
            invalid_args = SimpleNamespace(**vars(raw_args))
            setattr(invalid_args, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, message
            ):
                transform_mode(invalid_args, "defender")
        with self.assertRaisesRegex(RuntimeError, "optimizer_train_role"):
            transform_mode(raw_args, "attacker")
        no_fixed_dose = SimpleNamespace(**vars(raw_args))
        no_fixed_dose.custom_configs = dict(raw_args.custom_configs)
        no_fixed_dose.custom_configs[
            "defender_sft_optimizer_slots_per_rollout"
        ] = 0
        with self.assertRaisesRegex(RuntimeError, "fixed-dose"):
            transform_mode(no_fixed_dose, "defender")

        raw_branch = source[
            source.index("if advantage_transform_mode in (") : source.index(
                "elif optimizer_train_role == 'attacker' or no_defender_turn:"
            )
        ]
        self.assertNotIn("replay_buffer.normalize", raw_branch)
        self.assertIn("post_transform_metrics", raw_branch)
        self.assertIn("math.isclose(", raw_branch)
        self.assertIn(
            'debug/defender_advantage_mean_centering_applied', raw_branch
        )
        attacker_branch = source[
            source.index(
                "elif optimizer_train_role == 'attacker' or no_defender_turn:"
            ) : source.index(
                "elif optimizer_train_role == 'defender' or no_attacker_turn:"
            )
        ]
        self.assertIn("self.replay_buffer.normalize", attacker_branch)
        compile(source, str(UPSTREAM_ACTOR), "exec")

    def test_negative_defender_rewards_cannot_flip_or_collapse_in_raw_mode(self):
        # Exact reconstruction from fresh cap180 step 1:
        # H -1.5 -> +1.0926 and B -2.5 -> -0.0379 under the old transform.
        h_pre, h_post = -1.5, 1.0926
        b_pre, b_post = -2.5, -0.0379
        old_sigma = (h_pre - b_pre) / (h_post - b_post)
        old_mean = h_pre - h_post * old_sigma
        self.assertAlmostEqual(old_sigma, 0.884564352, places=8)
        self.assertAlmostEqual(old_mean, -2.466475011, places=8)
        self.assertGreater((h_pre - old_mean) / old_sigma, 0.0)
        self.assertLess(h_pre, 0.0)

        # A valid-format but harmful/wrong-refusal response has reward -1.
        # Against three reward=-3 trajectories the old replay whitening turns
        # it into +sqrt(3), whereas raw D keeps the absolute negative target.
        rewards = [-1.0, -3.0, -3.0, -3.0]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(
            sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        )
        self.assertAlmostEqual((rewards[0] - mean) / std, math.sqrt(3))
        self.assertEqual(rewards[0], -1.0)

        # If every observed response is the same -1 failure, centering erases
        # the entire RL signal. Raw REINFORCE preserves every -1 exactly.
        all_failures = [-1.0] * 32
        centered = [value - sum(all_failures) / 32 for value in all_failures]
        self.assertEqual(centered, [0.0] * 32)
        self.assertEqual(all_failures, [-1.0] * 32)

        # Around ratio=1 the unclipped PPO loss is -exp(delta)*A. Gradient
        # descent changes log-prob delta in the sign of A.
        def ppo_update_direction(advantage: float, epsilon: float = 1e-6):
            def loss(delta: float) -> float:
                return -math.exp(delta) * advantage

            gradient = (loss(epsilon) - loss(-epsilon)) / (2 * epsilon)
            return -gradient

        self.assertGreater(ppo_update_direction(math.sqrt(3)), 0.0)
        self.assertLess(ppo_update_direction(-1.0), 0.0)

    def test_joint_signed_defender_reward_and_advantage_contract(self):
        runtime_helpers: dict[str, object] = {}
        _load_functions(
            {"_validate_defender_joint_runtime_configuration"},
            runtime_helpers,
        )
        validate_runtime = runtime_helpers[
            "_validate_defender_joint_runtime_configuration"
        ]
        valid_runtime = {
            "v2_runtime": True,
            "fixed_attacker_adapter": "/ckpt/A1",
            "exact_fixed_attack_text": False,
        }
        validate_runtime(
            "raw_no_center",
            v2_runtime=False,
            fixed_attacker_adapter="",
            exact_fixed_attack_text=True,
        )
        validate_runtime("joint_signed", **valid_runtime)
        with self.assertRaisesRegex(ValueError, "v2_runtime=True"):
            validate_runtime(
                "joint_signed", **{**valid_runtime, "v2_runtime": False}
            )
        with self.assertRaisesRegex(ValueError, "non-empty frozen A1"):
            validate_runtime(
                "joint_signed",
                **{**valid_runtime, "fixed_attacker_adapter": "  "},
            )
        with self.assertRaisesRegex(ValueError, "exact_fixed_attack_text=False"):
            validate_runtime(
                "joint_signed",
                **{**valid_runtime, "exact_fixed_attack_text": True},
            )

        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            relative_sources = (
                "openrlhf/trainer/ray/ppo_actor.py",
                "openrlhf/trainer/ppo_utils/replay_buffer.py",
                "openrlhf/trainer/ppo_utils/language_game.py",
                "openrlhf/datasets/prompts_dataset.py",
            )
            for relative_source in relative_sources:
                destination = upstream / relative_source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(UPSTREAM_ROOT / relative_source, destination)
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {
                    "_replace_once",
                    "_patch_upstream_fixed_attacker_lora",
                    "_patch_upstream_role_advantage_normalization",
                },
                namespace,
            )
            _load_path_functions(
                FIXED_MODULE,
                {"_patch_upstream_attacker_only_sampling"},
                namespace,
            )
            namespace["_patch_upstream_attacker_only_sampling"]()
            namespace["_patch_upstream_fixed_attacker_lora"]()
            namespace["_patch_upstream_role_advantage_normalization"]()
            actor_source = (
                upstream / "openrlhf/trainer/ray/ppo_actor.py"
            ).read_text()
            game_source = (
                upstream / "openrlhf/trainer/ppo_utils/language_game.py"
            ).read_text()
            replay_source = (
                upstream / "openrlhf/trainer/ppo_utils/replay_buffer.py"
            ).read_text()
            dataset_source = (
                upstream / "openrlhf/datasets/prompts_dataset.py"
            ).read_text()

        outcome = SimpleNamespace(
            DEFENDER_WIN_CORRECT_REFUSAL=object(),
            DEFENDER_WIN_SUCCESSFUL_DEFENSE=object(),
            TIE=object(),
        )
        import os
        import unicodedata

        game_helpers: dict[str, object] = {
            "GameOutcome": outcome,
            "hashlib": hashlib,
            "json": json,
            "os": os,
            "re": re,
            "unicodedata": unicodedata,
        }
        _load_source_functions(
            game_source,
            {
                "_fixed_defender_actual_stratum",
                "_defender_joint_signed_reward",
                "_fixed_defender_exposure_record",
                "_append_fixed_defender_exposure_records",
            },
            game_helpers,
        )
        route = game_helpers["_fixed_defender_actual_stratum"]
        joint_reward = game_helpers["_defender_joint_signed_reward"]
        exposure_record = game_helpers["_fixed_defender_exposure_record"]
        append_exposure = game_helpers[
            "_append_fixed_defender_exposure_records"
        ]

        self.assertEqual(
            route(
                "generated_harmful",
                True,
                False,
                {"prompt_harmfulness": "harmful"},
            ),
            ("harmful", None),
        )
        self.assertEqual(
            route(
                "vanilla_benign",
                False,
                True,
                {"prompt_harmfulness": "unharmful"},
            ),
            ("benign", None),
        )
        self.assertEqual(
            route(
                "generated_harmful",
                True,
                False,
                {"prompt_harmfulness": "unharmful"},
            ),
            ("harmful", "label_mismatch"),
        )
        with self.assertRaisesRegex(RuntimeError, "rewritten by frozen A"):
            route(
                "vanilla_benign",
                True,
                False,
                {"prompt_harmfulness": "unharmful"},
            )

        correct = outcome.DEFENDER_WIN_CORRECT_REFUSAL
        harmless = outcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE
        # Blanket refusal on direct B is harmless but not the correct action.
        self.assertEqual(joint_reward({harmless}, False), (-1.0, False))
        self.assertEqual(
            joint_reward({correct, harmless}, False), (1.0, True)
        )
        self.assertEqual(
            joint_reward({correct, harmless}, True), (-1.0, False)
        )
        # Upstream labels an actual-B refusal=None as CORRECT_REFUSAL+TIE;
        # formal joint utility requires explicit compliance, never a tie.
        self.assertEqual(
            joint_reward({correct, harmless, outcome.TIE}, False),
            (-1.0, False),
        )
        with self.assertRaisesRegex(RuntimeError, "explicit boolean"):
            joint_reward({correct, harmless}, None)
        self.assertIn(
            "turn_states.get('cot_format_violation', None)",
            game_source,
        )
        self.assertNotIn(
            "turn_states.get('cot_format_violation', False),\n"
            "                        )\n"
            "                    )\n"
            "                    turn_states['defender_joint_success']",
            game_source,
        )

        pool_sha = "a" * 64
        record = exposure_record(
            "  A1\u3000generated   request  ",
            "generated_harmful",
            "harmful",
            {"prompt_harmfulness": "unharmful"},
            "label_mismatch",
            pool_sha,
        )
        expected_canonical = "A1 generated request"
        self.assertEqual(
            record["canonical_request_sha256"],
            hashlib.sha256(expected_canonical.encode()).hexdigest(),
        )
        self.assertEqual(record["drop_reason"], "label_mismatch")
        self.assertNotIn("request", record)
        with tempfile.TemporaryDirectory() as ledger_root:
            strategy = SimpleNamespace(
                args=SimpleNamespace(ckpt_path=ledger_root),
                get_rank=lambda: 2,
            )
            ledger_path = Path(append_exposure(strategy, [record]))
            self.assertEqual(ledger_path.name, "rank_02.jsonl")
            observed = json.loads(ledger_path.read_text().strip())
            self.assertEqual(observed, record)

        exposure_validator_namespace = {
            "Path": Path,
            "hashlib": hashlib,
            "json": json,
            "re": re,
        }
        _load_functions(
            {"_validate_defender_actual_request_exposure"},
            exposure_validator_namespace,
        )
        validate_exposure = exposure_validator_namespace[
            "_validate_defender_actual_request_exposure"
        ]
        with tempfile.TemporaryDirectory() as checkpoint_root:
            checkpoint = Path(checkpoint_root)
            ledger_dir = checkpoint / "actual_request_exposure"
            ledger_dir.mkdir()
            harmful_record = exposure_record(
                "unique harmful request",
                "generated_harmful",
                "harmful",
                {"prompt_harmfulness": "harmful"},
                None,
                pool_sha,
            )
            benign_record = exposure_record(
                "unique benign request",
                "vanilla_benign",
                "benign",
                {"prompt_harmfulness": "unharmful"},
                None,
                pool_sha,
            )
            (ledger_dir / "rank_00.jsonl").write_text(
                json.dumps(harmful_record) + "\n"
            )
            (ledger_dir / "rank_01.jsonl").write_text(
                json.dumps(benign_record) + "\n"
            )
            validation = validate_exposure(
                checkpoint,
                expected_prompt_pool_sha256=pool_sha,
                expected_rollouts=1,
                rollout_batch_size=2,
                expected_ranks=2,
            )
            self.assertEqual(validation["cross_stratum_overlap"], 0)
            self.assertEqual(
                validation["strata"]["harmful"]["occurrences"], 1
            )
            (ledger_dir / "rank_01.jsonl").unlink()
            with self.assertRaisesRegex(RuntimeError, "rank files drifted"):
                validate_exposure(
                    checkpoint,
                    expected_prompt_pool_sha256=pool_sha,
                    expected_rollouts=1,
                    rollout_batch_size=2,
                    expected_ranks=2,
                )
        self.assertIn(
            "attacker rewrite heuristics cannot override its utility",
            game_source,
        )
        self.assertIn(
            'joint_signed requires upstream_invalid_handling=True',
            ROLE_MODULE.read_text(),
        )

        actor_helpers: dict[str, object] = {"math": math}
        _load_source_functions(
            actor_source,
            {"_role_advantage_transform_mode"},
            actor_helpers,
        )
        joint_args = SimpleNamespace(
            custom_configs={
                "defender_raw_reinforce_advantages": True,
                "defender_sft_optimizer_slots_per_rollout": 4,
                "defender_reinforce_advantage_mode": "joint_signed",
                "defender_reward_utility": "joint_signed",
                "defender_actual_strata_required": True,
                "defender_episode_sum_policy_loss": True,
                "defender_episode_sum_loss_scale": 1.0 / 2048.0,
            },
            advantage_estimator="reinforce",
            gamma=1.0,
            init_kl_coef=0.0,
            generate_max_len=2048,
            packing_samples=True,
            actor_loss_coef=1.0,
            reward_clip_range=(-1.0, 1.0),
            use_kl_loss=False,
        )
        self.assertEqual(
            actor_helpers["_role_advantage_transform_mode"](
                joint_args, "defender"
            ),
            "joint_signed_defender_reinforce",
        )
        for field, value in (
            ("generate_max_len", 1024),
            ("packing_samples", False),
            ("actor_loss_coef", 0.5),
            ("reward_clip_range", (-10.0, 10.0)),
            ("use_kl_loss", True),
        ):
            drifted = SimpleNamespace(**vars(joint_args))
            setattr(drifted, field, value)
            with self.subTest(runtime_field=field), self.assertRaisesRegex(
                RuntimeError, "runtime contract drifted"
            ):
                actor_helpers["_role_advantage_transform_mode"](
                    drifted, "defender"
                )
        joint_branch = actor_source[
            actor_source.index("if advantage_transform_mode in (") :
            actor_source.index(
                "elif optimizer_train_role == 'attacker' or no_defender_turn:"
            )
        ]
        self.assertNotIn("replay_buffer.normalize", joint_branch)
        self.assertIn("reward_value not in (", joint_branch)
        self.assertIn("item_advantages", joint_branch)
        self.assertIn("defender_episode_sum_policy_loss", actor_source)
        role_source = ROLE_MODULE.read_text(encoding="utf-8")
        self.assertIn(
            '["--gamma", str(DEFENDER_V2_REINFORCE_GAMMA)]',
            role_source,
        )
        self.assertIn(
            '["--reward_clip_range", "-1.0", "1.0"]',
            role_source,
        )
        self.assertIn("not deterministic_defender_pool", actor_source)
        self.assertIn(
            "joint-signed defender failures survive legacy tie removal",
            ROLE_MODULE.read_text(),
        )
        self.assertIn("preserve_joint_signed_defender_failures", replay_source)

        dataset_tree = ast.parse(dataset_source)
        dataset_class = next(
            node for node in dataset_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RedTeamGamePromptDataset"
        )
        mark_method = next(
            copy.deepcopy(node) for node in dataset_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_mark_prompts_to_generate"
        )
        harness = ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ClassDef(
                        name="DatasetHarness",
                        bases=[],
                        keywords=[],
                        body=[mark_method],
                        decorator_list=[],
                    )
                ],
                type_ignores=[],
            )
        )
        dataset_namespace: dict[str, object] = {}
        exec(compile(harness, "<dataset-harness>", "exec"), dataset_namespace)

        class FakeStrategy:
            args = SimpleNamespace(seed=7)

            @staticmethod
            def print(*_args, **_kwargs):
                return None

        dataset = object.__new__(dataset_namespace["DatasetHarness"])
        dataset.labels = [
            "vanilla_harmful", "vanilla_harmful",
            "vanilla_benign", "vanilla_benign",
            "vanilla_benign", "vanilla_benign",
            "vanilla_harmful", "vanilla_harmful",
        ]
        dataset.prompts = list(range(8))
        dataset.custom_configs = {
            "fixed_opponent_generate_all_prompts": True,
            "fixed_opponent_generated_harmful_fraction": 1.0,
            "fixed_opponent_generated_benign_fraction": 0.0,
            "defender_actual_strata_required": True,
            "defender_deterministic_prompt_pool": True,
        }
        dataset._mark_prompts_to_generate(FakeStrategy())
        self.assertEqual(
            dataset.labels,
            [
                "generated_harmful", "generated_harmful",
                "vanilla_benign", "vanilla_benign",
                "vanilla_benign", "vanilla_benign",
                "generated_harmful", "generated_harmful",
            ],
        )
        compile(actor_source, str(UPSTREAM_ACTOR), "exec")
        compile(game_source, "<language-game>", "exec")
        compile(replay_source, "<replay-buffer>", "exec")

    def test_defender_episode_sum_ppo_uses_fixed_length_independent_scale(self):
        try:
            import torch
        except ImportError:  # Local unit-test image is intentionally CPU-light.
            torch = None

        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            actor.parent.mkdir(parents=True)
            actor.write_text(UPSTREAM_ACTOR.read_text())
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {
                    "_replace_once",
                    "_patch_upstream_role_advantage_normalization",
                },
                namespace,
            )
            namespace["_patch_upstream_role_advantage_normalization"]()
            actor_source = actor.read_text()

        self.assertIn("torch.split(", actor_source)
        self.assertIn("(token_loss * active_mask).sum(dim=-1).mean()", actor_source)
        self.assertIn("* loss_scale", actor_source)
        scale = 1.0 / 2048.0

        def scalar_reference(token_losses, action_counts):
            self.assertEqual(sum(action_counts), len(token_losses))
            offset = 0
            totals = []
            for count in action_counts:
                totals.append(sum(token_losses[offset : offset + count]))
                offset += count
            return sum(totals) / len(totals) * scale

        self.assertAlmostEqual(
            scalar_reference([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0], [2, 4]),
            scale,
            places=12,
        )
        self.assertAlmostEqual(scalar_reference([-1.0], [1]), -scale)
        self.assertAlmostEqual(
            scalar_reference([-1.0] * 4, [4]), -4 * scale
        )
        if torch is None:
            return

        helpers = {"torch": torch, "math": math}
        _load_source_functions(
            actor_source,
            {"_defender_episode_sum_policy_loss"},
            helpers,
        )
        loss_fn = helpers["_defender_episode_sum_policy_loss"]
        log_probs = torch.zeros((2, 4))
        old_log_probs = torch.zeros_like(log_probs)
        advantages = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]
        )
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        nonpacked = loss_fn(
            log_probs,
            old_log_probs,
            advantages,
            mask,
            clip_eps=0.2,
            packing_samples=False,
            num_actions=4,
            loss_scale=scale,
        )
        # Per-trajectory sums are -2 and +4; batch mean is +1.
        self.assertAlmostEqual(nonpacked.item(), scale, places=10)
        packed = loss_fn(
            torch.zeros((1, 6)),
            torch.zeros((1, 6)),
            torch.tensor([[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]]),
            None,
            clip_eps=0.2,
            packing_samples=True,
            num_actions=[2, 4],
            loss_scale=scale,
        )
        self.assertAlmostEqual(packed.item(), scale, places=10)

        short_positive = loss_fn(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
            clip_eps=0.2,
            packing_samples=False,
            num_actions=1,
            loss_scale=scale,
        )
        long_positive = loss_fn(
            torch.zeros((1, 4)),
            torch.zeros((1, 4)),
            torch.ones((1, 4)),
            torch.ones((1, 4), dtype=torch.bool),
            clip_eps=0.2,
            packing_samples=False,
            num_actions=4,
            loss_scale=scale,
        )
        self.assertAlmostEqual(short_positive.item(), -scale, places=10)
        self.assertAlmostEqual(long_positive.item(), -4 * scale, places=10)

    def test_fixed_defender_prompt_pool_is_hash_bound_and_interleaved(self):
        namespace = {
            "Path": Path,
            "hashlib": hashlib,
            "json": json,
            "re": re,
        }
        import unicodedata

        namespace["unicodedata"] = unicodedata
        _load_functions(
            {
                "_sha256_path",
                "_canonical_prompt_text",
                "_validate_defender_fixed_prompt_pool",
            },
            namespace,
        )
        validate = namespace["_validate_defender_fixed_prompt_pool"]

        rows = []
        ordinals = {"harmful": 0, "benign": 0}
        cycle = ("harmful", "harmful", "benign", "benign",
                 "benign", "benign", "harmful", "harmful")
        prompt_names = {
            "harmful": ("harmful one", "harmful two"),
            "benign": ("benign one", "benign two"),
        }
        for pool_index in range(16):
            stratum = cycle[pool_index % 8]
            harmful = stratum == "harmful"
            ordinal = ordinals[stratum]
            ordinals[stratum] += 1
            repeat_epoch, repeat_epoch_rank = divmod(ordinal, 2)
            prompt_index = (repeat_epoch + repeat_epoch_rank) % 2
            prompt = prompt_names[stratum][prompt_index]
            seed_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            source_index = prompt_index + (0 if harmful else 100)
            rows.append(
                {
                    "vanilla": prompt,
                    "adversarial": "",
                    "completion": "",
                    "data_type": (
                        "vanilla_harmful" if harmful else "vanilla_benign"
                    ),
                    "source_metadata": {
                        "pool_index": pool_index,
                        "rollout_step": pool_index // 8 + 1,
                        "rollout_offset": pool_index % 8,
                        "stratum_ordinal": ordinal,
                        "repeat_epoch": repeat_epoch,
                        "repeat_epoch_rank": repeat_epoch_rank,
                        "evaluation_stratum": f"actual_{stratum}",
                        "prompt_origin": (
                            "a1_generated_harmful"
                            if harmful else "direct_heldout_benign"
                        ),
                        "prompt_type": (
                            "generated_harmful"
                            if harmful else "direct_benign"
                        ),
                        "expected_actual_prompt_harmfulness": (
                            "harmful" if harmful else "unharmful"
                        ),
                        "request_route": (
                            "frozen_attacker_generate"
                            if harmful else "direct_bypass_attacker"
                        ),
                        "source_index": source_index,
                        "seed_prompt_sha256": seed_sha256,
                        "partition_split": "train",
                        "partition_selection_rank": source_index,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / "pool.jsonl"

            def write_pool(payload):
                pool.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in payload
                    )
                )
                return hashlib.sha256(pool.read_bytes()).hexdigest()

            digest = write_pool(rows)
            metadata = validate(
                pool,
                expected_sha256=digest,
                expected_rows=16,
                expected_rollout_batch_size=8,
            )
            self.assertEqual(
                metadata["interleave"],
                "four_rank_balanced_HHBBBBHH_cycle",
            )
            self.assertFalse(metadata["shuffle"])
            self.assertEqual(metadata["strata"]["harmful"]["rows"], 8)
            self.assertEqual(
                metadata["strata"]["harmful"]["unique_canonical_prompts"],
                2,
            )
            self.assertEqual(
                metadata["strata"]["harmful"]["repeat_occurrences"], 6,
            )
            self.assertEqual(len(metadata["source_metadata_keys"]), 15)

            bad_completion = copy.deepcopy(rows)
            bad_completion[1]["completion"] = "teacher answer leak"
            digest = write_pool(bad_completion)
            with self.assertRaisesRegex(RuntimeError, "must be empty"):
                validate(
                    pool,
                    expected_sha256=digest,
                    expected_rows=16,
                    expected_rollout_batch_size=8,
                )

            bad_metadata = copy.deepcopy(rows)
            bad_metadata[0]["source_metadata"]["unexpected"] = True
            digest = write_pool(bad_metadata)
            with self.assertRaisesRegex(RuntimeError, "schema drifted"):
                validate(
                    pool,
                    expected_sha256=digest,
                    expected_rows=16,
                    expected_rollout_batch_size=8,
                )

            bad_seed_hash = copy.deepcopy(rows)
            bad_seed_hash[0]["source_metadata"]["seed_prompt_sha256"] = (
                "0" * 64
            )
            digest = write_pool(bad_seed_hash)
            with self.assertRaisesRegex(RuntimeError, "seed hash drifted"):
                validate(
                    pool,
                    expected_sha256=digest,
                    expected_rows=16,
                    expected_rollout_batch_size=8,
                )

            wrong_order = copy.deepcopy(rows)
            wrong_order[0], wrong_order[2] = wrong_order[2], wrong_order[0]
            digest = write_pool(wrong_order)
            with self.assertRaisesRegex(RuntimeError, "HHBBBBHH cycle"):
                validate(
                    pool,
                    expected_sha256=digest,
                    expected_rows=16,
                    expected_rollout_batch_size=8,
                )

            digest = write_pool(rows)
            with self.assertRaisesRegex(RuntimeError, "artifact hash drifted"):
                validate(
                    pool,
                    expected_sha256="0" * 64,
                    expected_rows=16,
                    expected_rollout_batch_size=8,
                )

    def test_defender_resume_sidecar_is_exact_at_steps_10_30_and_40(self):
        namespace = {
            "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT": (
                _literal_assignment(
                    ROLE_MODULE,
                    "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT",
                )
            ),
            "DEFENDER_V2_GLOBAL_SAMPLES_PER_SFT_SLOT": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_GLOBAL_SAMPLES_PER_SFT_SLOT",
            ),
        }
        _load_functions(
            {"_validate_defender_sft_runtime_counters"}, namespace
        )
        validate = namespace["_validate_defender_sft_runtime_counters"]

        def runtime_state(step: int, actor_slots: int) -> dict[str, int]:
            sft_slots = min(step, 30) * 4
            samples = sft_slots * 16
            return {
                "schema_version": 1,
                "global_step": step,
                "cumulative_samples": samples,
                "cumulative_supervised_tokens": max(samples * 20, 1),
                "cumulative_harmful_samples": samples // 2,
                "cumulative_benign_samples": samples - samples // 2,
                "cumulative_sft_optimizer_slots": sft_slots,
                "cumulative_actor_optimizer_slots": actor_slots,
            }

        expected_actor_scheduler_updates = {10: 43, 30: 137, 40: 181}
        states = {
            step: runtime_state(step, actor_slots)
            for step, actor_slots in expected_actor_scheduler_updates.items()
        }
        for step, expected_updates in expected_actor_scheduler_updates.items():
            with self.subTest(step=step):
                validated = validate(
                    states[step], resume_step=step, stop_after_step=30
                )
                self.assertEqual(
                    validated["cumulative_actor_optimizer_slots"],
                    expected_updates,
                )
                self.assertEqual(
                    validated["cumulative_sft_optimizer_slots"],
                    min(step, 30) * 4,
                )

        corruptions = {
            "schema": {"schema_version": 2},
            "step": {"global_step": 9},
            "slots": {"cumulative_sft_optimizer_slots": 39},
            "samples": {"cumulative_samples": 639},
            "label_sum": {"cumulative_benign_samples": 319},
            "empty_benign_stratum": {
                "cumulative_harmful_samples": 640,
                "cumulative_benign_samples": 0,
            },
            "tokens": {"cumulative_supervised_tokens": 0},
            "actor_slots": {"cumulative_actor_optimizer_slots": 39},
            "boolean_integer": {"cumulative_samples": True},
        }
        baseline = states[10]
        for name, updates in corruptions.items():
            with self.subTest(corruption=name):
                damaged = dict(baseline)
                damaged.update(updates)
                with self.assertRaises(RuntimeError):
                    validate(damaged, resume_step=10, stop_after_step=30)
        with self.assertRaises(RuntimeError):
            validate([], resume_step=10, stop_after_step=30)

        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            actor.parent.mkdir(parents=True)
            actor.write_text(UPSTREAM_ACTOR.read_text())
            patch_namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                {"_replace_once", "_patch_upstream_lightweight_resume"},
                patch_namespace,
            )
            patch_namespace["_patch_upstream_lightweight_resume"]()
            source = actor.read_text()
        self.assertIn(
            '"lightweight_resume_actor_optimizer_slots", -1', source
        )
        self.assertIn("self.actor_scheduler.step(resume_updates)", source)
        self.assertNotIn("completed_fixed_dose_rollouts", source)
        compile(source, str(UPSTREAM_ACTOR), "exec")

    def test_defender_smoke_gate_is_dose_based_and_decides_next_rollout(self):
        namespace = {
            "DEFENDER_V2_SMOKE_GATE": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_SMOKE_GATE",
            ),
            "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT": (
                _literal_assignment(
                    ROLE_MODULE,
                    "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT",
                )
            ),
            "DEFENDER_V2_GLOBAL_SAMPLES_PER_SFT_SLOT": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_GLOBAL_SAMPLES_PER_SFT_SLOT",
            ),
        }
        _load_functions(
            {"_defender_v2_smoke_gate_configuration"},
            namespace,
        )
        validate = namespace["_defender_v2_smoke_gate_configuration"]
        expected = {
            1: ("mechanical", 2, 64, 4),
            6: ("dose_trend", 7, 384, 24),
            12: ("hard", 13, 768, 48),
        }
        for completed_rollouts, values in expected.items():
            name, decision, examples, minimum_slots = values
            gate = validate(completed_rollouts)
            self.assertEqual(gate["name"], name)
            self.assertEqual(gate["decision_rollout_step"], decision)
            self.assertEqual(
                gate["minimum_cumulative_sft_examples"], examples
            )
            self.assertEqual(
                gate["minimum_cumulative_sft_optimizer_slots"],
                minimum_slots,
            )
            self.assertIn("before optimizer/SFT dose N", gate["rollout_update_order"])
            self.assertFalse(
                gate["on_policy_seed_prompt_type_metrics"]["hard_gate"]
            )
        mechanical = validate(1)["requirements"]
        self.assertEqual(
            mechanical["defender_sft/rollout_samples"],
            {"direction": "eq", "bound": 64},
        )
        hard = validate(12)["requirements"]
        self.assertEqual(
            hard["defender_sft/cumulative_samples"]["bound"], 768
        )
        self.assertEqual(
            hard["defender_sft/cumulative_sft_optimizer_slots"]["bound"],
            48,
        )
        self.assertEqual(
            hard["defender_sft/cumulative_actor_optimizer_slots"]["bound"],
            48,
        )
        self.assertEqual(
            hard[
                "defender/wildguard_actual_harmful_correct_refusal_acc"
            ],
            {"direction": "min", "bound": 0.20},
        )
        self.assertEqual(
            hard["defender/cot_format_violation"],
            {"direction": "max", "bound": 0.10},
        )
        with self.assertRaisesRegex(ValueError, r"\[1, 6, 12\]"):
            validate(5)

    def test_formal_defender_gate_observes_rollouts_2_7_13_and_31(self):
        namespace = {
            "DEFENDER_V2_SMOKE_GATE": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_SMOKE_GATE",
            ),
            "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT": (
                _literal_assignment(
                    ROLE_MODULE,
                    "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT",
                )
            ),
            "DEFENDER_V2_GLOBAL_SAMPLES_PER_SFT_SLOT": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_GLOBAL_SAMPLES_PER_SFT_SLOT",
            ),
            "DEFENDER_V2_WARMUP_OPTIMIZER_STEPS": _literal_assignment(
                ROLE_MODULE,
                "DEFENDER_V2_WARMUP_OPTIMIZER_STEPS",
            ),
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
        self.assertEqual(gate["execution"], "monitoring_only")
        self.assertIn("post-update counters", gate["decision_timing"])
        self.assertEqual(
            [row["completed_sft_rollouts"] for row in gate["checkpoints"]],
            [1, 6, 12],
        )
        self.assertEqual(
            [row["decision_rollout_step"] for row in gate["checkpoints"]],
            [2, 7, 13],
        )
        self.assertEqual(
            gate["sft_final_effect"],
            {
                "completed_sft_rollouts": 30,
                "decision_rollout_step": 31,
                "cumulative_sft_examples": 1920,
                "cumulative_sft_optimizer_slots": 120,
            },
        )
        self.assertEqual(gate["warmup_optimizer_steps"], 20)
        self.assertIn("actual WildGuard", gate["semantic_label_contract"])
        self.assertTrue(
            gate["final_promotion_gate"][
                "interim_gate_does_not_cover_true_benign"
            ]
        )

        tree = ast.parse(ROLE_MODULE.read_text(), filename=str(ROLE_MODULE))
        preflight = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_defender_role_lora_v2_continuation"
        )
        preflight_source = ast.unparse(preflight)
        self.assertIn("defender_v2_interim_gate_configuration()", preflight_source)
        self.assertNotIn("_defender_v2_smoke_gate_configuration(10)", preflight_source)

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
            "defender_raw_reinforce_advantages": True,
            "defender_reinforce_advantage_mode": "joint_signed",
            "defender_reward_utility": "joint_signed",
        }
        observed = {
            name: ast.literal_eval(keywords[name]) for name in expected
        }
        self.assertEqual(observed, expected)
        self.assertIsInstance(
            keywords["actor_lr_warmup_steps_override"], ast.Name
        )
        self.assertEqual(
            keywords["actor_lr_warmup_steps_override"].id,
            "DEFENDER_V2_WARMUP_OPTIMIZER_STEPS",
        )
        fixed_slots = keywords[
            "defender_sft_optimizer_slots_per_rollout"
        ]
        self.assertIsInstance(fixed_slots, ast.Name)
        self.assertEqual(
            fixed_slots.id,
            "DEFENDER_V2_SFT_OPTIMIZER_SLOTS_PER_ROLLOUT",
        )
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

    def test_fixed_defender_checkpoint_is_incomplete_without_runtime_sidecar(self):
        namespace = self._checkpoint_validation_namespace()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            step10 = checkpoint_root / "global_step10_hf"
            self._write_checkpoint(checkpoint_root, 10)
            step20 = checkpoint_root / "global_step20_hf"
            self._write_checkpoint(checkpoint_root, 20)

            self.assertTrue(namespace["_is_complete_hf_checkpoint"](step20))
            self.assertFalse(
                namespace["_is_complete_hf_checkpoint"](
                    step20, require_defender_sft_runtime=True
                )
            )
            (step10 / "defender_sft_runtime.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            self.assertTrue(
                namespace["_is_complete_hf_checkpoint"](
                    step10, require_defender_sft_runtime=True
                )
            )
            default_step, _ = namespace["_latest_complete_hf_checkpoint"](
                checkpoint_root
            )
            defender_step, defender_path = namespace[
                "_latest_complete_hf_checkpoint"
            ](
                checkpoint_root,
                require_defender_sft_runtime=True,
            )

        self.assertEqual(default_step, 20)
        self.assertEqual(defender_step, 10)
        self.assertEqual(defender_path, step10)

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
