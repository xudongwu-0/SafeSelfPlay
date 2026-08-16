"""Regression contracts for the upstream role-LoRA/vLLM bridge.

The production bridge is injected into a pinned upstream checkout at runtime.
These tests apply that real source patch to a temporary copy, then execute only
the relevant AST nodes with small standard-library fakes.  They intentionally
do not import Modal, torch, Ray, PEFT, or vLLM.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
import queue
import shutil
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT.parent / "selfplay-redteaming"
PATCH_MODULE = REPO_ROOT / "modal_upstream_selfredteam_role_lora.py"
CONTRACT_MODULE = REPO_ROOT / "roll/utils/lora_sync_contract.py"

PATCHED_UPSTREAM_FILES = (
    "openrlhf/trainer/ray/vllm_worker_wrap.py",
    "openrlhf/trainer/ray/vllm_engine.py",
    "openrlhf/cli/train_ppo_ray.py",
    "openrlhf/trainer/ppo_utils/experience_maker.py",
    "openrlhf/trainer/ray/ppo_actor.py",
)


def _load_contract_module():
    spec = importlib.util.spec_from_file_location("lora_sync_contract_under_test", CONTRACT_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load LoRA sync contract: {CONTRACT_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_patch_functions(upstream_work: Path):
    """Load two pure patch functions without importing the Modal entrypoint."""
    tree = ast.parse(PATCH_MODULE.read_text(), filename=str(PATCH_MODULE))
    wanted = {"_replace_once", "_patch_upstream_vllm_lora_sync"}
    functions = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise AssertionError(f"Could not find patch functions {sorted(wanted)}")
    namespace = {"Path": Path, "UPSTREAM_WORK": upstream_work}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(PATCH_MODULE), "exec"), namespace)
    return namespace


def _strip_annotations(function):
    function = copy.deepcopy(function)
    function.decorator_list = []
    function.returns = None
    arguments = function.args
    for argument in list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs):
        argument.annotation = None
    if arguments.vararg is not None:
        arguments.vararg.annotation = None
    if arguments.kwarg is not None:
        arguments.kwarg.annotation = None
    return function


def _find_method(source: str, class_name: str, method_name: str):
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )


def _compile_method_as_function(
    source: str,
    class_name: str,
    method_name: str,
    output_name: str,
    namespace: dict,
):
    function = _strip_annotations(_find_method(source, class_name, method_name))
    function.name = output_name
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, f"<{class_name}.{method_name}>", "exec"), namespace)
    return namespace[output_name]


def _compile_class_with_methods(
    source: str,
    source_class_name: str,
    method_names: tuple[str, ...],
    output_name: str,
    namespace: dict,
    base_name: str | None = None,
):
    methods = [
        _strip_annotations(_find_method(source, source_class_name, method_name)) for method_name in method_names
    ]
    bases = [ast.Name(id=base_name, ctx=ast.Load())] if base_name else []
    class_node = ast.ClassDef(
        name=output_name,
        bases=bases,
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[]))
    exec(compile(module, f"<{source_class_name}>", "exec"), namespace)
    return namespace[output_name]


class _FakeDistributed:
    @staticmethod
    def get_rank():
        return 0

    @staticmethod
    def get_world_size():
        return 1

    @staticmethod
    def broadcast(*_args, **_kwargs):
        return None

    @staticmethod
    def barrier():
        return None

    @staticmethod
    def all_gather_object(outputs, value):
        outputs[0] = value


class _FakeCuda:
    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def synchronize():
        return None


FAKE_TORCH = SimpleNamespace(distributed=_FakeDistributed, cuda=_FakeCuda)
FAKE_DEEPSPEED = SimpleNamespace(zero=SimpleNamespace(GatheredParameters=lambda *_args, **_kwargs: nullcontext()))
FAKE_RAY = SimpleNamespace(get=lambda value: value)


class _RemoteMethod:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def remote(self, *args, **kwargs):
        self.events.append((self.label, args, kwargs))
        if self.label == "finalize":
            return {
                "sync_version": 1,
                "tensor_count": kwargs["expected_tensor_count"],
            }
        return True


class _FakeEngine:
    def __init__(self, events):
        self.update_lora_weight = _RemoteMethod("update", events)
        self.update_lora_weight_cuda_ipc = _RemoteMethod("update_ipc", events)
        self.finalize_lora = _RemoteMethod("finalize", events)
        self.reset_prefix_cache = _RemoteMethod("reset_prefix_cache", events)


class _FakeParameter:
    def __init__(self, shape, label):
        self.shape = tuple(shape)
        self.ds_shape = tuple(shape)
        self.dtype = "bfloat16"
        self.data = self
        self.label = label

    def clone(self):
        return self


@dataclass
class _FakePeftConfig:
    r: int = 2
    lora_alpha: int = 2
    target_modules: tuple[str, ...] = ("q_proj",)


class _FakePeftModel:
    def __init__(self, specs):
        self._parameters = [(name, _FakeParameter(shape, name)) for name, shape in specs]
        self.peft_config = {"default": _FakePeftConfig()}
        self.config = SimpleNamespace(num_hidden_layers=1)

    def named_parameters(self):
        return iter(self._parameters)


class _FakeLoRARequest:
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeTokensPrompt:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class UpstreamRoleLoRASyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract_module()
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.upstream_work = Path(cls._temporary_directory.name) / "upstream"
        for relative_path in PATCHED_UPSTREAM_FILES:
            source = UPSTREAM_ROOT / relative_path
            destination = cls.upstream_work / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        patch_namespace = _load_patch_functions(cls.upstream_work)
        patch_namespace["_patch_upstream_vllm_lora_sync"]()
        cls.sources = {
            relative_path: (cls.upstream_work / relative_path).read_text() for relative_path in PATCHED_UPSTREAM_FILES
        }

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    def test_canonical_peft_name_only_removes_default_adapter_segment(self):
        raw_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        expected = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        self.assertEqual(self.contract.normalize_peft_lora_name(raw_name), expected)
        self.assertEqual(self.contract.normalize_peft_lora_name(expected), expected)
        self.assertEqual(
            self.contract.normalize_peft_lora_name("base_model.model.default_block.q_proj.lora_B.default.weight"),
            "base_model.model.default_block.q_proj.lora_B.weight",
        )

    def test_canonical_name_matches_vllm_08_parser_module_path(self):
        """vLLM 0.8 drops the first two PEFT path components."""
        canonical = self.contract.normalize_peft_lora_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        )

        # Mirrors the vLLM 0.8.x name parser: discard ``base_model.model``
        # and the terminal ``lora_[AB].weight`` pair.
        parsed_module = ".".join(canonical.split(".")[2:-2])
        self.assertEqual(parsed_module, "model.layers.0.self_attn.q_proj")

    def test_tensor_contract_returns_complete_canonical_ab_pairs(self):
        raw_prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        specs = [
            (f"{raw_prefix}.lora_A.default.weight", (2, 3)),
            (f"{raw_prefix}.lora_B.default.weight", (4, 2)),
        ]
        self.assertEqual(
            self.contract.validate_lora_tensor_specs(specs, target_modules=("q_proj",)),
            [
                (f"{raw_prefix}.lora_A.weight", (2, 3)),
                (f"{raw_prefix}.lora_B.weight", (4, 2)),
            ],
        )

    def test_tensor_contract_supports_embedding_lora_pairs(self):
        prefix = "base_model.model.model.embed_tokens"
        self.assertEqual(
            self.contract.validate_lora_tensor_specs(
                [
                    (f"{prefix}.lora_embedding_A.default", (2, 10)),
                    (f"{prefix}.lora_embedding_B.default", (4, 2)),
                ],
                target_modules=("embed_tokens",),
            ),
            [
                (f"{prefix}.lora_embedding_A", (2, 10)),
                (f"{prefix}.lora_embedding_B", (4, 2)),
            ],
        )

    def test_runtime_module_contract_accepts_packed_sources_and_rejects_misses(
        self,
    ):
        prefix = "model.layers.0.self_attn"
        parsed = {f"{prefix}.q_proj", f"{prefix}.k_proj"}
        runtime = {f"{prefix}.qkv_proj", f"{prefix}.o_proj"}
        self.assertEqual(
            self.contract.validate_lora_runtime_module_mapping(
                parsed,
                runtime,
                packed_module_sources={
                    f"{prefix}.q_proj",
                    f"{prefix}.k_proj",
                    f"{prefix}.v_proj",
                },
            ),
            parsed,
        )
        with self.assertRaises(ValueError):
            self.contract.validate_lora_runtime_module_mapping(
                {"0.self_attn.q_proj"},
                runtime,
            )

    def test_tensor_contract_rejects_missing_duplicate_and_rank_mismatch(self):
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        invalid_cases = {
            "missing B": [
                (f"{prefix}.lora_A.default.weight", (2, 3)),
            ],
            "duplicate canonical A": [
                (f"{prefix}.lora_A.default.weight", (2, 3)),
                (f"{prefix}.lora_A.weight", (2, 3)),
                (f"{prefix}.lora_B.default.weight", (4, 2)),
            ],
            "A/B rank mismatch": [
                (f"{prefix}.lora_A.default.weight", (2, 3)),
                (f"{prefix}.lora_B.default.weight", (4, 3)),
            ],
        }
        for label, specs in invalid_cases.items():
            with self.subTest(label=label):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.contract.validate_lora_tensor_specs(specs, target_modules=("q_proj",))

    def _compile_broadcast(self):
        namespace = {
            "torch": FAKE_TORCH,
            "deepspeed": FAKE_DEEPSPEED,
            "ray": FAKE_RAY,
            "normalize_peft_lora_name": self.contract.normalize_peft_lora_name,
            "validate_lora_tensor_specs": self.contract.validate_lora_tensor_specs,
        }
        return _compile_method_as_function(
            self.sources["openrlhf/trainer/ray/ppo_actor.py"],
            "ActorPPOTrainer",
            "_broadcast_to_vllm",
            "broadcast_to_vllm",
            namespace,
        )

    @staticmethod
    def _broadcast_subject(specs, events):
        model = _FakePeftModel(specs)
        return SimpleNamespace(
            actor=SimpleNamespace(model=SimpleNamespace(module=model)),
            strategy=SimpleNamespace(
                args=SimpleNamespace(
                    lora_rank=2,
                    zero_stage=2,
                    enable_prefix_caching=False,
                ),
                print=lambda *_args, **_kwargs: None,
            ),
            use_cuda_ipc=False,
            vllm_engines=[_FakeEngine(events)],
            _model_update_group="test-model-update-group",
        )

    def _contract_module_patch(self):
        roll_module = types.ModuleType("roll")
        utils_module = types.ModuleType("roll.utils")
        roll_module.utils = utils_module
        utils_module.lora_sync_contract = self.contract
        return mock.patch.dict(
            sys.modules,
            {
                "roll": roll_module,
                "roll.utils": utils_module,
                "roll.utils.lora_sync_contract": self.contract,
            },
        )

    def test_broadcast_maps_every_ab_tensor_once_before_finalize(self):
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        specs = [
            (f"{prefix}.lora_A.default.weight", (2, 3)),
            (f"{prefix}.lora_B.default.weight", (4, 2)),
            ("base_model.model.model.embed_tokens.weight", (10, 3)),
        ]
        events = []
        with self._contract_module_patch():
            self._compile_broadcast()(self._broadcast_subject(specs, events))

        update_names = [args[0] for label, args, _kwargs in events if label == "update"]
        self.assertEqual(
            update_names,
            [
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
            ],
        )
        self.assertEqual(len(update_names), len(set(update_names)))
        self.assertEqual([label for label, _args, _kwargs in events].count("finalize"), 1)
        self.assertLess(
            max(i for i, event in enumerate(events) if event[0] == "update"),
            next(i for i, event in enumerate(events) if event[0] == "finalize"),
        )

    def test_broadcast_fails_before_partial_send_for_invalid_ab_specs(self):
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        invalid_cases = {
            "missing B": [
                (f"{prefix}.lora_A.default.weight", (2, 3)),
            ],
            "duplicate canonical A": [
                (f"{prefix}.lora_A.default.weight", (2, 3)),
                (f"{prefix}.lora_A.weight", (2, 3)),
                (f"{prefix}.lora_B.default.weight", (4, 2)),
            ],
            "rank mismatch": [
                (f"{prefix}.lora_A.default.weight", (2, 3)),
                (f"{prefix}.lora_B.default.weight", (4, 3)),
            ],
        }
        for label, specs in invalid_cases.items():
            with self.subTest(label=label):
                events = []
                with self._contract_module_patch():
                    with self.assertRaises((ValueError, RuntimeError)):
                        self._compile_broadcast()(self._broadcast_subject(specs, events))
                self.assertEqual(events, [])

    def _compile_llm_actor(self):
        engine_source = self.sources["openrlhf/trainer/ray/vllm_engine.py"]
        namespace = {
            "LoRARequest": _FakeLoRARequest,
            "TokensPrompt": _FakeTokensPrompt,
            "_TRAINING_LORA_INT_ID": 424242,
            "os": os,
        }
        return _compile_class_with_methods(
            engine_source,
            "LLMRayActor",
            ("finalize_lora", "_resolve_lora_request", "add_requests"),
            "PatchedLLMActor",
            namespace,
        )

    @staticmethod
    def _new_llm_actor(actor_class, llm):
        actor = actor_class()
        actor.llm = llm
        actor.num_actors = 1
        actor.actor_counter = 0
        actor.requests = {}
        actor.request_lora_selectors = {}
        actor.response_queues = defaultdict(queue.Queue)
        actor.current_lora_request = None
        actor.lora_sync_version = 0
        actor.fixed_opponent_lora_request = object()
        return actor

    def test_failed_finalize_never_publishes_current_request(self):
        class FailedLLM:
            def __init__(self, worker_results):
                self.worker_results = worker_results

            def collective_rpc(self, name, args=()):
                self.last_rpc = (name, args)
                return self.worker_results

        actor_class = self._compile_llm_actor()
        invalid_results = (
            [False],
            [True],
            [
                {
                    "ok": True,
                    "tensor_count": 2,
                    "parsed_module_count": 1,
                    "loaded_module_count": 0,
                }
            ],
        )
        for worker_results in invalid_results:
            with self.subTest(worker_results=worker_results):
                actor = self._new_llm_actor(
                    actor_class,
                    FailedLLM(worker_results),
                )
                with self.assertRaises(RuntimeError):
                    actor.finalize_lora({"r": 2}, expected_tensor_count=2)
                self.assertIsNone(actor.current_lora_request)

    def test_generate_fails_closed_before_finalize_and_routes_true_false(self):
        class RecordingLLM:
            def __init__(self):
                self.generate_requests = []

            def collective_rpc(self, _name, args=()):
                self.finalized_config = args[0]
                return [
                    {
                        "ok": True,
                        "tensor_count": args[1],
                        "parsed_module_count": 1,
                        "loaded_module_count": 1,
                    }
                ]

            def generate(self, *, prompts, sampling_params, lora_request=None):
                self.generate_requests.append(lora_request)
                label = "adapter" if lora_request is not None else "base"
                return [label for _prompt in prompts]

        actor_class = self._compile_llm_actor()
        pre_finalize_llm = RecordingLLM()
        pre_finalize = self._new_llm_actor(actor_class, pre_finalize_llm)
        with self.assertRaises(RuntimeError):
            pre_finalize.add_requests(
                0,
                sampling_params=object(),
                prompt_token_ids=[[1, 2]],
                use_lora=True,
            )
        self.assertEqual(pre_finalize_llm.generate_requests, [])

        llm = RecordingLLM()
        actor = self._new_llm_actor(actor_class, llm)
        actor.finalize_lora({"r": 2}, expected_tensor_count=2)
        current_request = actor.current_lora_request
        self.assertIsNotNone(current_request)

        actor.add_requests(
            0,
            sampling_params=object(),
            prompt_token_ids=[[1, 2]],
            use_lora=True,
        )
        actor.add_requests(
            0,
            sampling_params=object(),
            prompt_token_ids=[[1, 2]],
            use_lora=False,
        )
        self.assertIs(llm.generate_requests[0], current_request)
        self.assertIsNone(llm.generate_requests[1])

    def test_second_finalize_with_fixed_id_evicts_then_reloads_new_tensors(self):
        worker_source = (REPO_ROOT / "roll/third_party/vllm/worker.py").read_text()

        class FakeWorkerBase:
            def reload_model(self):
                self.events.append(("reload_model",))

        worker_namespace = {
            "FakeWorkerBase": FakeWorkerBase,
            "logger": SimpleNamespace(info=lambda *_args, **_kwargs: None),
            "validate_lora_runtime_module_mapping": (self.contract.validate_lora_runtime_module_mapping),
        }
        worker_class = _compile_class_with_methods(
            worker_source,
            "WorkerV1",
            ("custom_add_lora",),
            "TestWorker",
            worker_namespace,
            base_name="FakeWorkerBase",
        )

        class RequestManager:
            def __init__(self):
                self.version = 0

            def build_request(self, _config):
                self.version += 1
                version = self.version
                prefix = "base_model.model.model.layers.0.self_attn.q_proj"
                return SimpleNamespace(
                    lora_int_id=424242,
                    lora_tensors={
                        f"{prefix}.lora_A.weight": f"A{version}",
                        f"{prefix}.lora_B.weight": f"B{version}",
                    },
                )

        module_name = "model.layers.0.self_attn.q_proj"

        class AdapterManager:
            def __init__(self):
                self.model = SimpleNamespace(hf_to_vllm_mapper=None)
                self.modules = {module_name: object()}
                self.packed_modules = {}
                self.adapters = {}

            def get_adapter(self, lora_id):
                return self.adapters.get(lora_id)

        class ModelRunner:
            def __init__(self, worker):
                self.worker = worker
                self.cache = {}
                self.adapter_manager = AdapterManager()
                self.lora_manager = SimpleNamespace(_adapter_manager=self.adapter_manager)

            def remove_lora(self, lora_id):
                was_loaded = lora_id in self.cache
                self.worker.events.append(("remove", lora_id, was_loaded))
                self.cache.pop(lora_id, None)
                return was_loaded

            def add_lora(self, request):
                tensors = dict(request.lora_tensors)
                self.worker.events.append(("add", request.lora_int_id, tensors))
                self.cache[request.lora_int_id] = tensors
                self.adapter_manager.adapters[request.lora_int_id] = SimpleNamespace(loras={module_name: object()})
                return True

            def list_loras(self):
                return set(self.cache)

        worker = worker_class()
        worker.events = []
        worker.tensor_lora_manager = RequestManager()
        worker.model_runner = ModelRunner(worker)

        class WorkerBackedLLM:
            def collective_rpc(self, name, args=()):
                if name != "custom_add_lora":
                    raise AssertionError(name)
                return [worker.custom_add_lora(*args)]

        def parse_fine_tuned_lora_name(name, _weights_mapper=None):
            parts = name.split(".")
            return ".".join(parts[2:-2]), parts[-2] == "lora_A", False

        vllm_module = types.ModuleType("vllm")
        vllm_lora_module = types.ModuleType("vllm.lora")
        vllm_utils_module = types.ModuleType("vllm.lora.utils")
        vllm_utils_module.parse_fine_tuned_lora_name = parse_fine_tuned_lora_name
        vllm_module.lora = vllm_lora_module
        vllm_lora_module.utils = vllm_utils_module

        with mock.patch.dict(
            sys.modules,
            {
                "vllm": vllm_module,
                "vllm.lora": vllm_lora_module,
                "vllm.lora.utils": vllm_utils_module,
            },
        ):
            actor = self._new_llm_actor(self._compile_llm_actor(), WorkerBackedLLM())
            actor.finalize_lora({"version": 1}, expected_tensor_count=2)
            first_request_id = actor.current_lora_request.lora_int_id
            actor.finalize_lora({"version": 2}, expected_tensor_count=2)
            second_request_id = actor.current_lora_request.lora_int_id

        self.assertEqual(first_request_id, second_request_id)
        self.assertEqual(first_request_id, 424242)
        self.assertEqual(
            worker.events,
            [
                ("reload_model",),
                ("remove", 424242, False),
                (
                    "add",
                    424242,
                    {
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": "A1",
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": "B1",
                    },
                ),
                ("reload_model",),
                ("remove", 424242, True),
                (
                    "add",
                    424242,
                    {
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": "A2",
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": "B2",
                    },
                ),
            ],
        )
        self.assertEqual(
            worker.model_runner.cache[424242],
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": "A2",
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": "B2",
            },
        )

    def test_initial_lora_sync_happens_before_first_rollout(self):
        actor_source = self.sources["openrlhf/trainer/ray/ppo_actor.py"]
        events = []

        class FakeTrainer:
            def _broadcast_to_vllm(self):
                events.append("initial_sync")

            def fit(self, *_args):
                events.append("rollout_fit")

        namespace = {
            "ActorPPOTrainer": lambda *_args, **_kwargs: FakeTrainer(),
            "batch_vllm_engine_call": lambda *_args, **_kwargs: None,
            "os": os,
            "torch": FAKE_TORCH,
        }
        fit = _compile_method_as_function(
            actor_source,
            "ActorModelRayActor",
            "fit",
            "actor_model_fit",
            namespace,
        )

        class Args:
            load_checkpoint = False
            ckpt_path = "/definitely/not/a/checkpoint"
            lora_rank = 2
            vllm_enable_sleep = False

            def __getattr__(self, _name):
                return None

        subject = SimpleNamespace(
            strategy=SimpleNamespace(args=Args()),
            actor=object(),
            ema_model=None,
            actor_scheduler=None,
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1),
            prompts_dataloader=None,
            pretrain_dataloader=None,
            holdout_dataloader=None,
            sft_dataloader=None,
            consumed_samples=0,
            num_update_steps_per_episodes=1,
        )
        fit(
            subject,
            critic_model=None,
            initial_model=None,
            reward_model=[],
            vllm_engines=[object()],
        )
        self.assertEqual(events, ["initial_sync", "rollout_fit"])

    def test_each_optimizer_step_is_followed_by_exactly_one_sync(self):
        actor_source = self.sources["openrlhf/trainer/ray/ppo_actor.py"]
        namespace = {
            "batch_vllm_engine_call": lambda *_args, **_kwargs: None,
            "ray": FAKE_RAY,
            "torch": FAKE_TORCH,
        }
        ppo_train = _compile_method_as_function(
            actor_source,
            "ActorPPOTrainer",
            "ppo_train",
            "ppo_train",
            namespace,
        )
        events = []

        class Args:
            deepspeed_enable_sleep = False
            colocate_all_models = False
            vllm_enable_sleep = False

        subject = SimpleNamespace(
            experience_maker=SimpleNamespace(flush=lambda: None),
            critic_train_remote=False,
            strategy=SimpleNamespace(args=Args()),
            freezing_actor_steps=0,
            ppo_train_actor=lambda step: events.append(("optimizer", step)) or {},
            vllm_engines=[object()],
            _broadcast_to_vllm=lambda: events.append(("sync", None)),
            args=SimpleNamespace(eval_start_steps=999, eval_steps=10),
        )
        ppo_train(subject, 1)
        ppo_train(subject, 2)
        self.assertEqual(
            events,
            [
                ("optimizer", 1),
                ("sync", None),
                ("optimizer", 2),
                ("sync", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
