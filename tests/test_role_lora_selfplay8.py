"""Contracts for inherited A1/D1 -> A8/D8 role-LoRA self-play."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "role_lora_selfplay8.py"
ROLE_MODULE = REPO_ROOT / "modal_upstream_selfredteam_role_lora.py"
COORDINATOR_MODULE = REPO_ROOT / "modal_role_lora_selfplay8.py"
FIXED_MODULE = REPO_ROOT / "modal_upstream_selfredteam_fixed_seed.py"
UPSTREAM_ROOT = REPO_ROOT.parent / "selfplay-redteaming"


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "role_lora_selfplay8_under_test", CONTRACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_functions(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError(f"Missing functions in {path}: {names}")
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class RoleLoRASelfPlay8Test(unittest.TestCase):
    def test_modal_entrypoint_adds_roll_before_importing_sibling_modules(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")
        path_setup = source.index('sys.path.insert(0, "/roll")')
        training_import = source.index(
            "from modal_upstream_selfredteam_role_lora import"
        )
        contract_import = source.index("from role_lora_selfplay8 import")
        self.assertLess(path_setup, training_import)
        self.assertLess(path_setup, contract_import)

    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract()

    @staticmethod
    def _write_checkpoint(root: Path, label: str, payload: bytes) -> Path:
        checkpoint = root / label
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"r": 64, "lora_alpha": 64})
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(payload)
        return checkpoint

    def test_schedule_inherits_each_role_and_uses_latest_opponent(self):
        schedule = self.contract.build_selfplay8_schedule()
        self.assertEqual(len(schedule), 15)
        self.assertEqual(
            schedule[0].to_dict(),
            {
                "index": 1,
                "round_index": 1,
                "label": "D1",
                "role": "defender",
                "trainable_parent": "base",
                "fixed_opponent": "A1",
            },
        )
        self.assertEqual(
            (schedule[1].label, schedule[1].trainable_parent, schedule[1].fixed_opponent),
            ("A2", "A1", "D1"),
        )
        self.assertEqual(
            (schedule[-1].label, schedule[-1].trainable_parent, schedule[-1].fixed_opponent),
            ("D8", "D7", "A8"),
        )
        self.assertEqual(
            self.contract.population_labels(),
            [
                label
                for index in range(1, 9)
                for label in (f"A{index}", f"D{index}")
            ],
        )

    def test_population_copy_is_atomic_and_pruning_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt_dir = root / "one_stage" / "ckpt"
            final = self._write_checkpoint(
                ckpt_dir, "global_step67_hf", b"final-67"
            )
            intermediate = self._write_checkpoint(
                ckpt_dir, "global_step10_hf", b"intermediate-10"
            )
            marker = ckpt_dir / "early_stop.json"
            marker.write_text("{}")
            promoted = self.contract.atomic_copy_population_checkpoint(
                final, root / "population", "A2"
            )
            destination = Path(promoted["path"])

            self.assertTrue(final.is_dir())
            self.assertTrue(intermediate.is_dir())
            self.assertTrue(destination.is_dir())
            self.assertFalse((root / "population" / ".A2.incomplete").exists())
            self.assertEqual(
                promoted["sha256"],
                hashlib.sha256(b"final-67").hexdigest(),
            )

            removed = self.contract.prune_stage_hf_checkpoints(
                ckpt_dir,
                audited_population_checkpoint=destination,
                audited_sha256=promoted["sha256"],
            )
            self.assertEqual(len(removed), 2)
            self.assertFalse(final.exists())
            self.assertFalse(intermediate.exists())
            self.assertTrue(marker.is_file())
            self.assertTrue(destination.is_dir())

    def test_pruning_refuses_wrong_population_digest_or_broad_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._write_checkpoint(root, "A2", b"adapter")
            with self.assertRaises(RuntimeError):
                self.contract.prune_stage_hf_checkpoints(
                    root,
                    audited_population_checkpoint=checkpoint,
                    audited_sha256="wrong",
                )
            with self.assertRaises(ValueError):
                self.contract.prune_stage_hf_checkpoints(
                    root,
                    audited_population_checkpoint=checkpoint,
                    audited_sha256=self.contract.checkpoint_weight_digest(
                        checkpoint
                    ),
                )

    @staticmethod
    def _d1_history(start: float, tail: float) -> list[dict]:
        rows = []
        for step in range(1, 31):
            value = start if step <= 5 else tail if step >= 26 else 0.80
            rows.append(
                {
                    "step": step,
                    "value": value,
                    "metrics": {
                        "defender/overall_harmless_rate": 0.97,
                        "defender/cot_format_violation": 0.01,
                        "defender/info/generated_harmful_correct_refusal_acc": 0.94,
                        "defender/info/generated_benign_correct_refusal_acc": 0.96,
                    },
                }
            )
        return rows

    def test_d1_gate_requires_generated_only_streak_and_improvement(self):
        validation = {
            "early_stop": {
                "triggered": True,
                "history": self._d1_history(0.70, 0.96),
            }
        }
        gate = self.contract.evaluate_d1_gate(validation)
        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(gate["improvement"], 0.26)
        self.assertEqual(
            gate["metric"],
            "defender/generated_prompts_correct_refusal_acc",
        )

        no_improvement = {
            "early_stop": {
                "triggered": True,
                "history": self._d1_history(0.96, 0.96),
            }
        }
        failed = self.contract.evaluate_d1_gate(no_improvement)
        self.assertFalse(failed["passed"])
        self.assertTrue(
            any("improvement" in reason for reason in failed["failures"])
        )

    def test_d1_gate_rejects_blanket_refusal_or_bad_subgroup(self):
        history = self._d1_history(0.60, 0.96)
        for row in history[-5:]:
            row["metrics"][
                "defender/info/generated_benign_correct_refusal_acc"
            ] = 0.50
        gate = self.contract.evaluate_d1_gate(
            {"early_stop": {"history": history}}
        )
        self.assertFalse(gate["passed"])
        self.assertTrue(
            any("generated_benign" in reason for reason in gate["failures"])
        )

    @staticmethod
    def _paired_contract_fixture():
        implementation_hashes = {
            "modal_upstream_v2_payoff.py": "modal-hash",
            "roll/utils/upstream_v2_payoff.py": "helper-hash",
        }
        state = {
            "schema_version": 1,
            "run_suffix": "selfplay-run",
            "status": "awaiting_d1_paired_gate",
            "stages": {
                "A1": {
                    "status": "retained",
                    "sha256": "a1-sha",
                    "population_checkpoint": "/output/population/A1",
                },
                "D1": {
                    "status": "retained",
                    "sha256": "d1-sha",
                    "population_checkpoint": "/output/population/D1",
                },
            },
        }
        manifest = {
            "implementation_version": "paired-d1-promotion-v1",
            "implementation_hashes": implementation_hashes,
            "attacker_adapter": {
                "path": "/output/population/A1",
                "sha256": "a1-sha",
                "rank": 64,
                "alpha": 64,
            },
            "base_arm": {
                "adapter": "base_model",
                "prompt_protocol": "direct_chat_no_cot",
            },
            "d1_arm": {
                "adapter": {
                    "path": "/output/population/D1",
                    "sha256": "d1-sha",
                    "rank": 64,
                    "alpha": 64,
                },
                "prompt_protocol": "upstream_defender_cot",
            },
            "held_out_seed_stream": {"passed": True, "seed_base": 18888},
            "seed_base": 18888,
            "pairs": 1024,
            "prompt_distribution": (
                "deterministic exact 50/50 harmful/benign interleave"
            ),
            "nested_seed_prefix": True,
            "pairing": {
                "defender_seed": "identical within pair for base and D1",
                "prompt_harmfulness_agreement": (
                    "base and D1 WildGuard prompt_harmfulness must be exactly "
                    "equal (including None==None); mismatch drops the whole "
                    "pair before any reward/delta/McNemar computation"
                ),
            },
            "reward_normalization": {
                "attacker": "none",
                "defender": "none",
                "paired_delta": "none (D1 minus base)",
            },
            "zero_sum_assumption": False,
        }
        summary = {
            "completed": True,
            "implementation_version": "paired-d1-promotion-v1",
            "implementation_hashes": implementation_hashes,
        }
        status = {
            "completed": True,
            "stage": "completed",
            "artifact_sha256": {"paired_summary.json": "artifact-hash"},
        }
        audits = {
            "A1": {
                "weight_sha256": "a1-sha",
                "llama_v2_contract": {"passed": True},
            },
            "D1": {
                "weight_sha256": "d1-sha",
                "llama_v2_contract": {"passed": True},
            },
        }
        return state, manifest, summary, status, audits, implementation_hashes

    def test_paired_contract_rejects_adapter_or_implementation_hash_mismatch(self):
        state, manifest, summary, status, audits, hashes = (
            self._paired_contract_fixture()
        )
        verified = self.contract.verify_d1_paired_evidence_contract(
            state,
            manifest,
            summary,
            status,
            a1_audit=audits["A1"],
            d1_audit=audits["D1"],
            expected_implementation_hashes=hashes,
            artifact_hashes_verified=True,
            recomputed_summary_verified=True,
        )
        self.assertTrue(verified["adapter_hashes"])
        self.assertTrue(verified["artifact_integrity"])

        bad_adapter = copy.deepcopy(manifest)
        bad_adapter["d1_arm"]["adapter"]["sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "D1 adapter SHA mismatch"):
            self.contract.verify_d1_paired_evidence_contract(
                state,
                bad_adapter,
                summary,
                status,
                a1_audit=audits["A1"],
                d1_audit=audits["D1"],
                expected_implementation_hashes=hashes,
                artifact_hashes_verified=True,
                recomputed_summary_verified=True,
            )

        bad_implementation = copy.deepcopy(manifest)
        bad_implementation["implementation_hashes"][
            "modal_upstream_v2_payoff.py"
        ] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "implementation hash"):
            self.contract.verify_d1_paired_evidence_contract(
                state,
                bad_implementation,
                summary,
                status,
                a1_audit=audits["A1"],
                d1_audit=audits["D1"],
                expected_implementation_hashes=hashes,
                artifact_hashes_verified=True,
                recomputed_summary_verified=True,
            )

    @staticmethod
    def _durable_chain_state():
        return {
            "schema_version": 1,
            "run_suffix": "selfplay-run",
            "base_model": "meta-llama/base",
            "config": {"rounds": 8, "recipe": "v2"},
            "status": "running",
            "stages": {
                "A1": {"status": "retained", "sha256": "a1-sha"},
                "D1": {"status": "retained", "sha256": "d1-sha"},
            },
        }

    def test_durable_claim_retries_crash_before_spawn(self):
        a2 = self.contract.build_selfplay8_schedule()[1]
        first = self.contract.ensure_stage_spawn_pending(
            self._durable_chain_state(), a2
        )
        retry = self.contract.ensure_stage_spawn_pending(first["state"], a2)

        self.assertTrue(first["should_spawn"])
        self.assertTrue(retry["should_spawn"])
        self.assertEqual(first["spawn_claim_id"], retry["spawn_claim_id"])
        self.assertEqual(retry["transition_state"], "spawn_pending")

    def test_spawn_before_call_id_crash_keeps_call_id_observational(self):
        a2 = self.contract.build_selfplay8_schedule()[1]
        first = self.contract.ensure_stage_spawn_pending(
            self._durable_chain_state(), a2
        )
        attempted = self.contract.record_stage_spawn_observation(
            first["state"],
            stage_label="A2",
            spawn_claim_id=first["spawn_claim_id"],
            call_id=None,
        )
        retry = self.contract.ensure_stage_spawn_pending(attempted, a2)
        observed = self.contract.record_stage_spawn_observation(
            retry["state"],
            stage_label="A2",
            spawn_claim_id=retry["spawn_claim_id"],
            call_id="fc-observational-only",
        )

        self.assertTrue(retry["should_spawn"])
        self.assertEqual(first["spawn_claim_id"], retry["spawn_claim_id"])
        self.assertEqual(observed["stages"]["A2"]["spawn_attempts"], 2)
        self.assertEqual(
            observed["stages"]["A2"]["transition_state"], "spawn_pending"
        )

    def test_stage_claim_fails_closed_on_config_or_dependency_hash_change(self):
        a2 = self.contract.build_selfplay8_schedule()[1]
        first = self.contract.ensure_stage_spawn_pending(
            self._durable_chain_state(), a2
        )
        changed_config = copy.deepcopy(first["state"])
        changed_config["config"]["recipe"] = "different"
        with self.assertRaisesRegex(RuntimeError, "spawn claim changed"):
            self.contract.ensure_stage_spawn_pending(changed_config, a2)

        changed_parent = copy.deepcopy(first["state"])
        changed_parent["stages"]["A1"]["sha256"] = "different-a1"
        with self.assertRaisesRegex(RuntimeError, "spawn claim changed"):
            self.contract.ensure_stage_spawn_pending(changed_parent, a2)

    def test_duplicate_child_ACK_requires_explicit_recovery_authorization(self):
        a2 = self.contract.build_selfplay8_schedule()[1]
        pending = self.contract.ensure_stage_spawn_pending(
            self._durable_chain_state(), a2
        )
        first_ack = self.contract.acknowledge_stage_child_started(
            pending["state"],
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
        )
        duplicate_ack = self.contract.acknowledge_stage_child_started(
            first_ack["state"],
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
        )
        retained = self.contract.mark_stage_transition_retained(
            first_ack["state"],
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
            retained_payload={"sha256": "a2-sha"},
        )
        after_retained = self.contract.acknowledge_stage_child_started(
            retained,
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
        )

        self.assertTrue(first_ack["should_train"])
        self.assertFalse(duplicate_ack["should_train"])
        self.assertFalse(after_retained["should_train"])
        self.assertEqual(
            duplicate_ack["state"]["stages"]["A2"]["child_ack_count"], 1
        )

    def test_ACK_owner_crash_authorizes_serialized_same_suffix_resume(self):
        a2 = self.contract.build_selfplay8_schedule()[1]
        pending = self.contract.ensure_stage_spawn_pending(
            self._durable_chain_state(), a2
        )
        ack = self.contract.acknowledge_stage_child_started(
            pending["state"],
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
        )
        recovery = self.contract.authorize_stage_trainer_recovery(
            ack["state"],
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
            deterministic_trainer_run_suffix="selfplay-run_A2",
            serialized_trainer=True,
        )

        self.assertTrue(recovery["should_resume_trainer"])
        self.assertEqual(
            recovery["deterministic_trainer_run_suffix"],
            "selfplay-run_A2",
        )
        self.assertEqual(recovery["trainer_recovery_count"], 1)
        self.assertEqual(
            recovery["state"]["stages"]["A2"]["transition_state"],
            "child_started",
        )
        with self.assertRaisesRegex(RuntimeError, "serialized execution"):
            self.contract.authorize_stage_trainer_recovery(
                ack["state"],
                stage_label="A2",
                spawn_claim_id=pending["spawn_claim_id"],
                deterministic_trainer_run_suffix="selfplay-run_A2",
                serialized_trainer=False,
            )
        with self.assertRaisesRegex(RuntimeError, "suffix mismatch"):
            self.contract.authorize_stage_trainer_recovery(
                ack["state"],
                stage_label="A2",
                spawn_claim_id=pending["spawn_claim_id"],
                deterministic_trainer_run_suffix="different_A2",
                serialized_trainer=True,
            )

    def test_duplicate_successor_claim_is_stable_from_a2_through_d8(self):
        schedule = self.contract.build_selfplay8_schedule()
        state = self._durable_chain_state()
        for stage in schedule[1:]:
            pending = self.contract.ensure_stage_spawn_pending(state, stage)
            repeated_pending = self.contract.ensure_stage_spawn_pending(
                pending["state"], stage
            )
            self.assertEqual(
                pending["spawn_claim_id"], repeated_pending["spawn_claim_id"]
            )
            self.assertTrue(repeated_pending["should_spawn"])
            ack = self.contract.acknowledge_stage_child_started(
                repeated_pending["state"],
                stage_label=stage.label,
                spawn_claim_id=pending["spawn_claim_id"],
            )
            duplicate_dispatch = self.contract.ensure_stage_spawn_pending(
                ack["state"], stage
            )
            self.assertFalse(duplicate_dispatch["should_spawn"])
            duplicate_child = self.contract.acknowledge_stage_child_started(
                ack["state"],
                stage_label=stage.label,
                spawn_claim_id=pending["spawn_claim_id"],
            )
            self.assertFalse(duplicate_child["should_train"])
            recovery = self.contract.authorize_stage_trainer_recovery(
                duplicate_child["state"],
                stage_label=stage.label,
                spawn_claim_id=pending["spawn_claim_id"],
                deterministic_trainer_run_suffix=(
                    f"selfplay-run_{stage.label}"
                ),
                serialized_trainer=True,
            )
            self.assertTrue(recovery["should_resume_trainer"])
            state = self.contract.mark_stage_transition_retained(
                recovery["state"],
                stage_label=stage.label,
                spawn_claim_id=pending["spawn_claim_id"],
                retained_payload={"sha256": f"{stage.label.lower()}-sha"},
            )

        self.assertEqual(state["stages"]["D8"]["status"], "retained")

    def test_dispatcher_dedupes_predecessor_but_explicitly_retries_pending(self):
        class Call:
            def __init__(self, object_id):
                self.object_id = object_id

        class Spawn:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Call(f"fc-{len(self.calls)}")

        spawn = Spawn()

        class Train:
            pass

        Train.spawn = spawn
        persisted = []
        namespace = {
            "Path": Path,
            "Any": object,
            "ensure_stage_spawn_pending": (
                self.contract.ensure_stage_spawn_pending
            ),
            "record_stage_spawn_observation": (
                self.contract.record_stage_spawn_observation
            ),
            "_persist_state": lambda root, state: persisted.append(
                copy.deepcopy(state)
            ),
            "train_role_lora_selfplay8_stage": Train,
        }
        _load_functions(
            COORDINATOR_MODULE,
            {"_dispatch_stage_claim"},
            namespace,
        )
        dispatch = namespace["_dispatch_stage_claim"]
        a2 = self.contract.build_selfplay8_schedule()[1]
        first = dispatch(
            Path("/unused"),
            self._durable_chain_state(),
            run_suffix="selfplay-run",
            stage=a2,
        )
        duplicate_parent = dispatch(
            Path("/unused"),
            first["state"],
            run_suffix="selfplay-run",
            stage=a2,
        )
        explicit_retry = dispatch(
            Path("/unused"),
            duplicate_parent["state"],
            run_suffix="selfplay-run",
            stage=a2,
            retry_existing_pending=True,
        )

        self.assertTrue(first["spawned"])
        self.assertFalse(duplicate_parent["spawned"])
        self.assertTrue(explicit_retry["spawned"])
        self.assertEqual(len(spawn.calls), 2)
        self.assertEqual(
            spawn.calls[0]["spawn_claim_id"],
            spawn.calls[1]["spawn_claim_id"],
        )
        ack = self.contract.acknowledge_stage_child_started(
            explicit_retry["state"],
            stage_label="A2",
            spawn_claim_id=explicit_retry["spawn_claim_id"],
        )
        after_ack = dispatch(
            Path("/unused"),
            ack["state"],
            run_suffix="selfplay-run",
            stage=a2,
            retry_existing_pending=True,
        )
        self.assertFalse(after_ack["spawned"])
        self.assertEqual(len(spawn.calls), 2)

    def test_dispatch_failure_preserves_pending_claim_for_recovery(self):
        class FailingSpawn:
            def __call__(self, **kwargs):
                raise ConnectionError("disconnect before call id")

        class Train:
            pass

        Train.spawn = FailingSpawn()
        persisted = []

        class Volume:
            @staticmethod
            def reload():
                return None

        namespace = {
            "Path": Path,
            "Any": object,
            "ensure_stage_spawn_pending": (
                self.contract.ensure_stage_spawn_pending
            ),
            "record_stage_spawn_observation": (
                self.contract.record_stage_spawn_observation
            ),
            "_persist_state": lambda root, state: persisted.append(
                copy.deepcopy(state)
            ),
            "_load_state": lambda root: copy.deepcopy(persisted[-1]),
            "output_vol": Volume,
            "train_role_lora_selfplay8_stage": Train,
        }
        _load_functions(
            COORDINATOR_MODULE,
            {"_dispatch_stage_claim"},
            namespace,
        )
        a2 = self.contract.build_selfplay8_schedule()[1]
        result = namespace["_dispatch_stage_claim"](
            Path("/unused"),
            self._durable_chain_state(),
            run_suffix="selfplay-run",
            stage=a2,
        )

        self.assertFalse(result["spawned"])
        self.assertTrue(result["spawn_attempted"])
        self.assertEqual(result["state"]["status"], "spawn_pending_recovery")
        self.assertEqual(
            result["state"]["stages"]["A2"]["transition_state"],
            "spawn_pending",
        )
        self.assertEqual(
            result["dispatch_error"]["type"], "ConnectionError"
        )
        self.assertEqual(
            persisted[-1]["stages"]["A2"]["spawn_claim_id"],
            result["spawn_claim_id"],
        )

    def test_outer_failure_handler_reloads_and_never_erases_pending_successor(self):
        durable = self._durable_chain_state()
        a2 = self.contract.build_selfplay8_schedule()[1]
        durable = self.contract.ensure_stage_spawn_pending(durable, a2)[
            "state"
        ]
        writes = []

        class Volume:
            reload_count = 0
            commit_count = 0

            @classmethod
            def reload(cls):
                cls.reload_count += 1

            @classmethod
            def commit(cls):
                cls.commit_count += 1

        namespace = {
            "Path": Path,
            "Any": object,
            "output_vol": Volume,
            "_load_state": lambda root: copy.deepcopy(durable),
            "_write_json_atomic": lambda path, state: writes.append(
                copy.deepcopy(state)
            ),
        }
        _load_functions(
            COORDINATOR_MODULE,
            {"_record_stage_failure"},
            namespace,
        )
        stale = self._durable_chain_state()
        namespace["_record_stage_failure"](
            Path("/unused"),
            stale,
            "D1",
            ConnectionError("spawn disconnected"),
        )

        self.assertEqual(Volume.reload_count, 1)
        self.assertEqual(Volume.commit_count, 1)
        self.assertEqual(writes[-1]["status"], "spawn_pending_recovery")
        self.assertEqual(
            writes[-1]["stages"]["A2"]["transition_state"],
            "spawn_pending",
        )

    def test_inner_trainer_unknown_outcome_keeps_child_started_recoverable(self):
        a2 = self.contract.build_selfplay8_schedule()[1]
        pending = self.contract.ensure_stage_spawn_pending(
            self._durable_chain_state(), a2
        )
        durable = self.contract.acknowledge_stage_child_started(
            pending["state"],
            stage_label="A2",
            spawn_claim_id=pending["spawn_claim_id"],
        )["state"]
        writes = []

        class Volume:
            @staticmethod
            def reload():
                return None

            @staticmethod
            def commit():
                return None

        namespace = {
            "Path": Path,
            "Any": object,
            "output_vol": Volume,
            "_load_state": lambda root: copy.deepcopy(durable),
            "_write_json_atomic": lambda path, state: writes.append(
                copy.deepcopy(state)
            ),
        }
        _load_functions(
            COORDINATOR_MODULE,
            {"_record_stage_failure"},
            namespace,
        )
        namespace["_record_stage_failure"](
            Path("/unused"),
            self._durable_chain_state(),
            "A2",
            ConnectionError("nested trainer outcome unknown"),
        )

        self.assertEqual(writes[-1]["status"], "child_started_recovery")
        self.assertEqual(
            writes[-1]["stages"]["A2"]["transition_state"],
            "child_started",
        )
        self.assertEqual(
            writes[-1]["last_trainer_owner_loss"][
                "deterministic_trainer_run_suffix"
            ],
            "selfplay-run_A2",
        )

    def test_retained_before_prune_crash_replays_pruning_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "stage_run"
            ckpt_dir = run_dir / "ckpt"
            final = self._write_checkpoint(
                ckpt_dir, "global_step30_hf", b"retained-adapter"
            )
            self._write_checkpoint(
                ckpt_dir, "global_step10_hf", b"intermediate"
            )
            promoted = self.contract.atomic_copy_population_checkpoint(
                final, root / "population", "A2"
            )
            state = {
                "stages": {
                    "A2": {
                        "status": "retained",
                        "transition_state": "retained",
                        "run_dir": str(run_dir),
                        "population_checkpoint": promoted["path"],
                        "sha256": promoted["sha256"],
                    }
                }
            }
            persisted = []
            namespace = {
                "Path": Path,
                "Any": object,
                "prune_stage_hf_checkpoints": (
                    self.contract.prune_stage_hf_checkpoints
                ),
                "_persist_state": lambda state_root, value: persisted.append(
                    copy.deepcopy(value)
                ),
            }
            _load_functions(
                COORDINATOR_MODULE,
                {"_complete_retained_stage_pruning"},
                namespace,
            )
            recovered = namespace["_complete_retained_stage_pruning"](
                root,
                state,
                stage_label="A2",
            )

            self.assertFalse(final.exists())
            self.assertFalse((ckpt_dir / "global_step10_hf").exists())
            self.assertEqual(
                len(recovered["stages"]["A2"]["pruned_stage_hf_checkpoints"]),
                2,
            )
            replay = namespace["_complete_retained_stage_pruning"](
                root,
                recovered,
                stage_label="A2",
            )
            self.assertEqual(replay, recovered)
            self.assertEqual(len(persisted), 1)

    def test_call_id_disconnect_never_downgrades_a_child_ACK(self):
        persisted = []
        durable = [None]
        contract = self.contract

        class Call:
            @property
            def object_id(self):
                pending_state = persisted[-1]
                claim = pending_state["stages"]["A2"]["spawn_claim_id"]
                durable[0] = contract.acknowledge_stage_child_started(
                    pending_state,
                    stage_label="A2",
                    spawn_claim_id=claim,
                )["state"]
                raise ConnectionError("call id channel disconnected")

        class Spawn:
            @staticmethod
            def __call__(**kwargs):
                return Call()

        class Train:
            pass

        Train.spawn = Spawn()

        class Volume:
            @staticmethod
            def reload():
                return None

        namespace = {
            "Path": Path,
            "Any": object,
            "ensure_stage_spawn_pending": (
                self.contract.ensure_stage_spawn_pending
            ),
            "record_stage_spawn_observation": (
                self.contract.record_stage_spawn_observation
            ),
            "_persist_state": lambda root, state: persisted.append(
                copy.deepcopy(state)
            ),
            "_load_state": lambda root: copy.deepcopy(durable[0]),
            "output_vol": Volume,
            "train_role_lora_selfplay8_stage": Train,
        }
        _load_functions(
            COORDINATOR_MODULE,
            {"_dispatch_stage_claim"},
            namespace,
        )
        result = namespace["_dispatch_stage_claim"](
            Path("/unused"),
            self._durable_chain_state(),
            run_suffix="selfplay-run",
            stage=self.contract.build_selfplay8_schedule()[1],
        )

        self.assertFalse(result["spawned"])
        self.assertEqual(
            result["state"]["stages"]["A2"]["transition_state"],
            "child_started",
        )
        self.assertNotEqual(result["state"]["status"], "failed")

    def test_D1_retained_finish_and_approved_A2_windows_are_resumable(self):
        tree = ast.parse(COORDINATOR_MODULE.read_text(encoding="utf-8"))
        wanted = {
            "_transition_resume_block_reason",
            "resume_role_lora_selfplay8_transition",
        }
        functions = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                copied = copy.deepcopy(node)
                copied.decorator_list = []
                functions.append(copied)
        self.assertEqual({node.name for node in functions}, wanted)

        class Volume:
            @staticmethod
            def reload():
                return None

        schedule = self.contract.build_selfplay8_schedule(2)
        states = []
        reconciled = []
        dispatched = []

        def load_state(root):
            return copy.deepcopy(states[-1])

        def reconcile(state, *, run_suffix, stage):
            reconciled.append(stage.label)
            return {
                "state": state,
                "spawned": True,
                "call_id": "fc-d1-reconcile",
                "spawn_claim_id": state["stages"][stage.label][
                    "spawn_claim_id"
                ],
                "reconcile_only": True,
            }

        def dispatch(root, state, *, run_suffix, stage, **kwargs):
            dispatched.append(stage.label)
            return {
                "state": state,
                "spawned": True,
                "call_id": "fc-a2",
                "spawn_claim_id": "a2-claim",
            }

        namespace = {
            "Any": object,
            "Path": Path,
            "re": __import__("re"),
            "output_vol": Volume,
            "SELFPLAY_ROOT": Path("/unused"),
            "_load_state": load_state,
            "build_selfplay8_schedule": lambda rounds: schedule,
            "_spawn_retained_stage_reconciler": reconcile,
            "_dispatch_stage_claim": dispatch,
            "_population_path": lambda root, label: root / label,
            "train_role_lora_selfplay8_stage": object(),
            "population_labels": self.contract.population_labels,
            "_persist_state": lambda root, state: None,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=functions, type_ignores=[])
                ),
                str(COORDINATOR_MODULE),
                "exec",
            ),
            namespace,
        )
        resume = namespace["resume_role_lora_selfplay8_transition"]

        d1_retained = {
            "status": "running",
            "config": {"rounds": 2},
            "stages": {
                "D1": {
                    "status": "retained",
                    "transition_state": "retained",
                    "spawn_claim_id": "d1-claim",
                }
            },
        }
        states.append(d1_retained)
        result = resume("selfplay-run")
        self.assertTrue(result["spawned"])
        self.assertEqual(reconciled, ["D1"])

        approved_before_a2_claim = copy.deepcopy(d1_retained)
        approved_before_a2_claim["status"] = "awaiting_d1_paired_gate"
        approved_before_a2_claim["d1_paired_promotion"] = {
            "promotion": {"passed": True}
        }
        approved_before_a2_claim["stages"]["D1"][
            "successor_release"
        ] = {"approved": True}
        states.append(approved_before_a2_claim)
        result = resume("selfplay-run")
        self.assertTrue(result["spawned"])
        self.assertEqual(dispatched, ["A2"])

    def test_upstream_patch_forces_final_checkpoint_at_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            actor = upstream / "openrlhf/trainer/ray/ppo_actor.py"
            actor.parent.mkdir(parents=True)
            shutil.copy2(
                UPSTREAM_ROOT / "openrlhf/trainer/ray/ppo_actor.py", actor
            )
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                ROLE_MODULE,
                {"_replace_once", "_patch_upstream_role_early_stopping"},
                namespace,
            )
            namespace["_patch_upstream_role_early_stopping"]()
            source = actor.read_text()

        self.assertIn("ROLE_EARLY_STOP_TRIGGERED", source)
        self.assertIn("if steps % args.save_steps != 0:", source)
        self.assertIn("self._save_checkpoint(args, tag, client_states)", source)
        self.assertIn(
            'early_stop_record["actual_final_step"] = int(steps)', source
        )
        self.assertIn("early_stop_companion_metrics", source)
        self.assertIn("early_stop_companion_bounds", source)
        compile(source, str(actor), "exec")

        tree = ast.parse(source)
        qualifier_nodes = [
            copy.deepcopy(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "early_stop_row_qualifies"
        ]
        self.assertEqual(len(qualifier_nodes), 1)
        qualifier_namespace = {
            "math": math,
            "early_stop_min_steps": 30,
            "early_stop_threshold": 0.95,
            "early_stop_companion_bounds": {
                "defender/overall_harmless_rate": {
                    "direction": "min",
                    "bound": 0.95,
                },
                "defender/cot_format_violation": {
                    "direction": "max",
                    "bound": 0.05,
                },
                "defender/info/generated_harmful_correct_refusal_acc": {
                    "direction": "min",
                    "bound": 0.90,
                },
                "defender/info/generated_benign_correct_refusal_acc": {
                    "direction": "min",
                    "bound": 0.90,
                },
            },
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=qualifier_nodes, type_ignores=[])
                ),
                "early_stop_qualifier.py",
                "exec",
            ),
            qualifier_namespace,
        )
        qualifies = qualifier_namespace["early_stop_row_qualifies"]
        good_metrics = {
            "defender/overall_harmless_rate": 0.97,
            "defender/cot_format_violation": 0.01,
            "defender/info/generated_harmful_correct_refusal_acc": 0.92,
            "defender/info/generated_benign_correct_refusal_acc": 0.94,
        }
        self.assertTrue(
            qualifies({"step": 30, "value": 0.96, "metrics": good_metrics})
        )
        bad_metrics = dict(good_metrics)
        bad_metrics[
            "defender/info/generated_benign_correct_refusal_acc"
        ] = 0.89
        self.assertFalse(
            qualifies({"step": 31, "value": 0.99, "metrics": bad_metrics})
        )

    def test_fixed_opponent_marks_all_prompts_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            for relative in (
                "openrlhf/datasets/prompts_dataset.py",
                "openrlhf/trainer/ppo_utils/language_game.py",
                "openrlhf/trainer/ppo_utils/replay_buffer.py",
                "openrlhf/trainer/ray/ppo_actor.py",
            ):
                destination = upstream / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(UPSTREAM_ROOT / relative, destination)
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                FIXED_MODULE,
                {"_patch_upstream_attacker_only_sampling"},
                namespace,
            )
            role_namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                ROLE_MODULE,
                {
                    "_replace_once",
                    "_patch_upstream_fixed_attacker_lora",
                },
                role_namespace,
            )
            # The imported attacker-only patch uses direct string replacement
            # rather than _replace_once and therefore needs no extra globals.
            namespace["_patch_upstream_attacker_only_sampling"]()
            role_namespace["_patch_upstream_fixed_attacker_lora"]()
            dataset_source = (
                upstream / "openrlhf/datasets/prompts_dataset.py"
            ).read_text()

        self.assertIn("fixed_opponent_generate_all_prompts", dataset_source)
        self.assertIn(
            "percent_generated_harmful, percent_generated_benign = 1.0, 1.0",
            dataset_source,
        )
        compile(dataset_source, "prompts_dataset.py", "exec")

    def test_core_and_coordinator_encode_symmetric_adapter_routing(self):
        role_source = ROLE_MODULE.read_text()
        coordinator_source = (
            REPO_ROOT / "modal_role_lora_selfplay8.py"
        ).read_text()
        self.assertIn("fixed_defender_adapter: str = \"\"", role_source)
        self.assertIn('use_lora="fixed_opponent"', role_source)
        self.assertIn(
            '"fixed_defender_lora_from_actor_vllm"', role_source
        )
        self.assertIn(
            'else "defender/generated_prompts_correct_refusal_acc"',
            role_source,
        )
        self.assertIn("fixed_defender_adapter=(", coordinator_source)
        self.assertIn("fixed_attacker_adapter=(", coordinator_source)
        self.assertIn("actual_final_step", coordinator_source)
        self.assertIn("if label == \"D1\":", coordinator_source)
        self.assertIn(
            'state["status"] = "awaiting_d1_paired_gate"',
            coordinator_source,
        )
        d1_branch = coordinator_source.split('if label == "D1":', 1)[1]
        d1_branch = d1_branch.split("elif not validation", 1)[0]
        self.assertNotIn("train_role_lora_selfplay8_stage.spawn", d1_branch)

    def test_paired_resume_has_no_user_pass_boolean_and_durable_stage_claim(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        resume = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "approve_d1_paired_gate_and_resume_a2"
        )
        argument_names = [argument.arg for argument in resume.args.args]

        self.assertEqual(argument_names, ["run_suffix", "paired_run_suffix"])
        self.assertIn("max_containers=1", source)
        self.assertIn("ensure_stage_spawn_pending", source)
        self.assertIn("acknowledge_stage_child_started", source)
        self.assertIn("mark_stage_transition_retained", source)
        self.assertIn("spawn_claim_id", ast.unparse(resume))
        self.assertNotIn("plan_idempotent_a2_resume", source)
        self.assertNotIn("record_idempotent_a2_spawn", source)
        self.assertNotIn("user_passed", source)

    def test_child_ACK_is_persisted_before_training_and_stage_is_serialized(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stage = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_role_lora_selfplay8_stage"
        )
        rendered = ast.unparse(stage)
        self.assertLess(
            rendered.index("acknowledge_stage_child_started"),
            rendered.index("train_upstream_attacker_lora_fixed_seed.remote"),
        )
        self.assertLess(
            rendered.index("_persist_state(root, state)"),
            rendered.index("train_upstream_attacker_lora_fixed_seed.remote"),
        )
        decorator_text = "\n".join(
            ast.unparse(decorator) for decorator in stage.decorator_list
        )
        self.assertIn("max_containers=1", decorator_text)
        self.assertLess(
            rendered.index("authorize_stage_trainer_recovery"),
            rendered.index("train_upstream_attacker_lora_fixed_seed.remote"),
        )

        trainer_tree = ast.parse(ROLE_MODULE.read_text(encoding="utf-8"))
        trainer = next(
            node
            for node in trainer_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_upstream_attacker_lora_fixed_seed"
        )
        trainer_decorators = "\n".join(
            ast.unparse(decorator) for decorator in trainer.decorator_list
        )
        self.assertIn("max_containers=1", trainer_decorators)
        self.assertIn("run_suffix=trainer_suffix", rendered)

    def test_coordinator_independently_rejects_prompt_label_contamination(self):
        namespace = {"Any": object}
        _load_functions(
            COORDINATOR_MODULE,
            {"_independently_verify_prompt_label_pair_drops"},
            namespace,
        )
        verify = namespace["_independently_verify_prompt_label_pair_drops"]
        equal_none = {
            "prompt_type": "generated_harmful",
            "prompt_harmfulness_mismatch": False,
            "dropped_reason": None,
            "base_arm": {"wildguard": {"prompt_harmfulness": None}},
            "d1_arm": {"wildguard": {"prompt_harmfulness": None}},
        }
        mismatch = {
            "prompt_type": "generated_benign",
            "prompt_harmfulness_mismatch": True,
            "dropped_reason": "wildguard_prompt_harmfulness_mismatch",
            "base_arm": {"wildguard": {"prompt_harmfulness": "harmful"}},
            "d1_arm": {"wildguard": {"prompt_harmfulness": None}},
        }
        result = verify([equal_none, mismatch])
        self.assertEqual(result["counts"], {"overall": 1, "harmful": 0, "benign": 1})
        self.assertEqual(result["rates"]["overall"], 0.5)

        contaminated = copy.deepcopy(mismatch)
        contaminated["d1_arm"]["attacker_raw_reward"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "scored before pair-drop"):
            verify([equal_none, contaminated])

    def test_fresh_defender_stages_keep_role_sft_through_step_30_by_default(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")

        self.assertIn("defender_sft_stop_after_step: int = 30", source)
        self.assertIn('defender_sft_stop_after_step != 30', source)
        self.assertIn(
            'config.get("defender_sft_stop_after_step", 30)',
            source,
        )
        self.assertIn(
            '"defender_sft_stop_after_step": defender_sft_stop_after_step',
            source,
        )

    def test_all_fresh_roles_use_the_frozen_v2_optimizer_and_continuation(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")

        self.assertIn("defender_learning_rate: float = 1e-5", source)
        self.assertNotIn("defender_learning_rate: float = 4e-5", source)
        self.assertIn('actor_lr_scheduler="constant_with_warmup"', source)
        self.assertIn("lr_warmup_ratio=0.05", source)
        self.assertIn("v2_continuation_sft=True", source)
        self.assertNotIn("v2_continuation_sft=is_attacker", source)
        self.assertIn(
            '"Self-play v2 freezes both role learning rates at 1e-5"',
            source,
        )

    def test_training_implementation_hashes_bind_state_stage_and_core_call(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")
        expected_sources = (
            "modal_role_lora_selfplay8.py",
            "modal_upstream_selfredteam_role_lora.py",
            "role_lora_selfplay8.py",
            "modal_upstream_selfredteam_fixed_seed.py",
            "roll/utils/lora_sync_contract.py",
            "roll/third_party/vllm/worker.py",
            "roll/third_party/deepspeed/model_update.py",
        )
        for filename in expected_sources:
            self.assertIn(filename, source)
        tree = ast.parse(source, filename=str(COORDINATOR_MODULE))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_current_training_implementation_hashes"
        )
        frozen_tuple = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Tuple)
            and all(
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                for element in node.elts
            )
            and "modal_role_lora_selfplay8.py"
            in {element.value for element in node.elts}
        )
        self.assertEqual(ast.literal_eval(frozen_tuple), expected_sources)
        self.assertIn('"training_implementation_sha256"', source)
        self.assertIn("_assert_training_implementation_frozen(state)", source)
        self.assertIn("expected_implementation_sha256=(", source)
        self.assertIn("_assert_trainer_manifest_implementation(", source)
        self.assertIn("fresh run_suffix", source)

    def test_training_implementation_freeze_and_manifest_checks_fail_closed(self):
        expected = {
            "modal_role_lora_selfplay8.py": "coordinator-sha",
            "modal_upstream_selfredteam_role_lora.py": "core-sha",
        }
        namespace = {
            "Any": object,
            "Path": Path,
            "json": json,
            "_current_training_implementation_hashes": lambda: dict(expected),
        }
        _load_functions(
            COORDINATOR_MODULE,
            {
                "_assert_training_implementation_frozen",
                "_assert_trainer_manifest_implementation",
            },
            namespace,
        )
        assert_frozen = namespace["_assert_training_implementation_frozen"]
        assert_manifest = namespace["_assert_trainer_manifest_implementation"]
        state = {"config": {"training_implementation_sha256": expected}}
        self.assertEqual(assert_frozen(state), expected)
        with self.assertRaisesRegex(RuntimeError, "fresh run_suffix"):
            assert_frozen({"config": {}})

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest_path = run_dir / "manifest.json"
            with self.assertRaisesRegex(RuntimeError, "Missing or invalid"):
                assert_manifest(run_dir, expected)
            manifest_path.write_text(
                json.dumps(
                    {
                        "implementation_sha256": "core-sha",
                        "expected_implementation_sha256": expected,
                    }
                ),
                encoding="utf-8",
            )
            assert_manifest(run_dir, expected)
            manifest_path.write_text(
                json.dumps(
                    {
                        "implementation_sha256": "old-core-sha",
                        "expected_implementation_sha256": expected,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                assert_manifest(run_dir, expected)

    def test_formal_d_config_registers_update_aware_interim_gate(self):
        source = COORDINATOR_MODULE.read_text(encoding="utf-8")
        self.assertIn("defender_v2_interim_gate_configuration", source)
        self.assertIn('"defender_v2_interim_gate"', source)
        self.assertNotIn("defender_v2_smoke_gate=True", source)

    def test_trainable_and_fixed_defender_use_identical_cot_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            utils_path = upstream / "red_team/utils.py"
            utils_path.parent.mkdir(parents=True)
            shutil.copy2(UPSTREAM_ROOT / "red_team/utils.py", utils_path)
            language_game = (
                upstream / "openrlhf/trainer/ppo_utils/language_game.py"
            )
            language_game.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                UPSTREAM_ROOT
                / "openrlhf/trainer/ppo_utils/language_game.py",
                language_game,
            )
            namespace = {"Path": Path, "UPSTREAM_WORK": upstream}
            _load_functions(
                ROLE_MODULE,
                {
                    "_replace_once",
                    "_patch_upstream_fixed_defender_direct_chat",
                    "_patch_upstream_defender_role_prompt",
                },
                namespace,
            )
            namespace["_patch_upstream_fixed_defender_direct_chat"]()
            namespace["_patch_upstream_defender_role_prompt"]()
            patched_source = utils_path.read_text()

        tree = ast.parse(patched_source)
        wanted = {
            "convert_game_history_to_query",
            "convert_game_history_to_messages",
        }
        functions = [
            copy.deepcopy(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        render_namespace = {
            "ATTACKER_SYSTEM_PROMPT": "attacker-system",
            "ATTACKER_INSTRUCTION_PROMPT_HARMFUL": "harmful {vanilla_prompt}",
            "ATTACKER_INSTRUCTION_PROMPT_BENIGN": "benign {vanilla_prompt}",
            "ATTACKER_INSTRUCTION_COT_PROMPT_NO_SEED": "attacker no seed",
            "ATTACKER_INSTRUCTION_COT_PROMPT_HARMFUL": "harmful cot {vanilla_prompt}",
            "ATTACKER_INSTRUCTION_COT_PROMPT_BENIGN": "benign cot {vanilla_prompt}",
            "DEFENDER_INSTRUCTION_COT_PROMPT": "defender wrapper: {user_query}",
            "ASSISTANT_THINKING_PREFIX": "<think-prefix>",
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=functions, type_ignores=[])
                ),
                "patched_utils.py",
                "exec",
            ),
            render_namespace,
        )

        class Tokenizer:
            @staticmethod
            def apply_chat_template(messages, tokenize, add_generation_prompt):
                return json.dumps(messages, sort_keys=True)

        common = {
            "history": [{"content": "fixed attack"}],
            "player_role": "defender",
            "prompt": "seed",
            "prompt_type": "generated_harmful",
            "tokenizer": Tokenizer(),
        }
        trainable_d = render_namespace["convert_game_history_to_messages"](
            **common,
            custom_configs={
                "no_attacker_turn": True,
                "defender_role_specific_safety_prompt": True,
            },
        )
        fixed_d = render_namespace["convert_game_history_to_messages"](
            **common,
            custom_configs={
                "no_defender_turn": True,
                "fixed_defender_lora_from_actor_vllm": True,
                "defender_role_specific_safety_prompt": True,
            },
        )
        base_d = render_namespace["convert_game_history_to_messages"](
            **common,
            custom_configs={
                "no_defender_turn": True,
                "base_defender_direct_chat_no_cot": True,
            },
        )
        self.assertEqual(trainable_d, fixed_d)
        self.assertTrue(fixed_d.endswith("<think-prefix>"))
        self.assertNotEqual(base_d, fixed_d)
        self.assertFalse(base_d.endswith("<think-prefix>"))


if __name__ == "__main__":
    unittest.main()
