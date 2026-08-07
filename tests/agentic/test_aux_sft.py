import json

from roll.pipeline.agentic.aux_sft import RoleAuxSFTSampler
from roll.utils.constants import IGNORE_INDEX


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize
        ids = [1]
        for message in messages:
            role_id = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            ids.extend([role_id, *[100 + ord(char) % 50 for char in message["content"]]])
        if add_generation_prompt:
            ids.append(30)
        else:
            ids.append(self.eos_token_id)
        return ids


def test_aux_sft_masks_prompt_and_keeps_assistant_tokens(tmp_path):
    path = tmp_path / "attacker.jsonl"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "rewrite this"},
        {"role": "assistant", "content": "<answer>rewrite</answer>"},
    ]
    path.write_text(json.dumps({"messages": messages}) + "\n")

    sampler = RoleAuxSFTSampler(str(path), seed=42, rank=0)
    batch = sampler.next_batch(FakeTokenizer(), device="cpu", batch_size=1, max_length=256)

    labels = batch["labels"][0].tolist()
    first_target = next(index for index, token in enumerate(labels) if token != IGNORE_INDEX)
    prompt_ids = FakeTokenizer().apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    assert first_target == len(prompt_ids)
    assert labels[first_target:] == batch["input_ids"][0, first_target:].tolist()


def test_aux_sft_accepts_upstream_defender_schema(tmp_path):
    path = tmp_path / "defender.jsonl"
    path.write_text(json.dumps({"vanilla": "hello", "completion": "<answer>hi</answer>"}) + "\n")

    sampler = RoleAuxSFTSampler(str(path), seed=42, rank=0)
    batch = sampler.next_batch(FakeTokenizer(), device="cpu", batch_size=1, max_length=512)

    assert (batch["labels"] != IGNORE_INDEX).any()
