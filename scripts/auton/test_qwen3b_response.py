"""Quick sanity test: load Qwen2.5-3B-Instruct and run prompts matching the actual Kuhn Poker agent format."""
import sys
sys.path.insert(0, "/zfsauton/scratch/wentsec/ROLL")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/zfsauton/scratch/wentsec/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"


def build_env_prompts():
    """Build (system_prompt, user_obs) pairs matching what the env sends to the agent."""
    from roll.pipeline.agentic.env.kuhn_poker.env import KuhnPokerEnv
    env = KuhnPokerEnv(max_steps=2, action_pattern=r"<think>(.*?)</think>\s*<action>(.*?)</action>", opponent_strategy="llm")
    system_prompt = env.get_instructions()

    cases = [
        ("P0 first action — holding King",
         "Your card: King. You are Player 1 (first to act).\nPot size: 2 chips (both antes). Board state: no action yet.\nChoose: Pass or Bet."),
        ("P1 facing a Bet — holding Queen",
         "Your card: Queen. You are Player 2.\nPot size: 3 chips. Board state: Player 1 chose Bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass or Bet."),
        ("P0 responding to bet — holding Jack",
         "Your card: Jack. You are Player 1.\nPot size: 3 chips. Board state: you passed, Player 2 bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass (fold) or Bet (call)."),
    ]
    return system_prompt, cases


def main():
    print(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map}\n")

    system_prompt, cases = build_env_prompts()
    print(f"=== System prompt ===\n{system_prompt}\n")

    for label, obs in cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"=== {label} ===")
        print(f"Obs: {obs}")
        print(f"Response: {response}\n")

    print("Done.")


if __name__ == "__main__":
    main()
