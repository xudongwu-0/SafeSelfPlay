from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ModalRoleLoraDependencyContractTests(unittest.TestCase):
    def test_image_reasserts_roll_compatible_stack_after_vllm(self) -> None:
        source = (REPO_ROOT / "modal_fsp_demo.py").read_text(encoding="utf-8")
        vllm_offset = source.index('.pip_install("vllm==0.10.2")')
        compatibility_offset = source.index("'transformers==4.57.6'")
        mount_offset = source.index(".add_local_dir(")
        self.assertLess(vllm_offset, compatibility_offset)
        self.assertLess(compatibility_offset, mount_offset)
        for requirement in (
            "'numpy==1.26.4'",
            "'transformers==4.57.6'",
            "'accelerate==0.34.2'",
            "'peft==0.12.0'",
            "'trl==0.9.6'",
            "'sacrebleu==2.5.1'",
            "'sentence-transformers==3.4.1'",
            "'cupy-cuda12x==13.6.0'",
            "'opencv-python-headless==4.11.0.86'",
            "'typer==0.16.1'",
        ):
            self.assertEqual(source.count(requirement), 1, requirement)

    def test_existing_runtime_install_is_preinstalled_exactly(self) -> None:
        image_source = (REPO_ROOT / "modal_fsp_demo.py").read_text(
            encoding="utf-8"
        )
        trainer_source = (
            REPO_ROOT / "modal_upstream_selfredteam_role_lora.py"
        ).read_text(encoding="utf-8")
        for requirement in (
            "sacrebleu==2.5.1",
            "sentence-transformers==3.4.1",
        ):
            self.assertIn(requirement, image_source)
            self.assertIn(requirement, trainer_source)

    def test_image_recipe_is_outside_frozen_training_hash_set(self) -> None:
        coordinator = (REPO_ROOT / "modal_role_lora_selfplay8.py").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"def _current_training_implementation_hashes\(\).*?"
            r"for filename in \((.*?)\):",
            coordinator,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        self.assertNotIn("modal_fsp_demo.py", match.group(1))

    def test_all_eight_frozen_training_sources_are_byte_unchanged(self) -> None:
        expected = {
            "modal_role_lora_selfplay8.py": (
                "5eedfda2e111af4b398a117801bcca29064e7a166600a63abb88c8417491c4c2"
            ),
            "modal_upstream_selfredteam_role_lora.py": (
                "d8950d4487dff1df8901ee4ff10542e13249ad8b6aae3dc9e9959f5bb314e340"
            ),
            "role_lora_selfplay8.py": (
                "2098f73699b17497ca9ce113337d47ee0fe699cb71415c1c936da8a776a96ffa"
            ),
            "roll/utils/upstream_v2_payoff.py": (
                "a57552c6d5b42e8fcbdf7ae3cb1beafd53032c36fdd15bc79aa60b440f389b93"
            ),
            "modal_upstream_selfredteam_fixed_seed.py": (
                "72207bbb1c43b644ccd4c6194ca908fdc2c2879eabf76de5acc83b5e51a5b01c"
            ),
            "roll/utils/lora_sync_contract.py": (
                "a730240409baf01639cef68908aac4c90e808d5a611fc13e1bbdee5b147bba6e"
            ),
            "roll/third_party/vllm/worker.py": (
                "f8439c4d4bd7d32d6a76f0cd405e52809665d81ea1b6c7b0df09af74ec620272"
            ),
            "roll/third_party/deepspeed/model_update.py": (
                "90fc3a24b1a123b7aa7b4fbbfab259c48be2424c23a3f6f93d5804c3127f4b21"
            ),
        }
        observed = {
            relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in expected
        }
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
