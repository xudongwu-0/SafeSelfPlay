"""Measure how often the thinking_token_budget was hit (thinking force-closed).

Signal: count tokens between start-of-response and the first '</think>'.
- No '</think>' at all: thinking never closed (response hit max_new_tokens mid-thinking).
- thinking_tokens >= budget - slack: budget hit, '</think>' was forced.
- thinking_tokens <  budget - slack: model self-closed thinking.
"""
import json
import os
import sys
from glob import glob

from transformers import AutoTokenizer

SLACK = 8  # tolerance for the "near budget" comparison
MODEL_PATH = "/zfsauton/scratch/wentsec/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def classify_response(response: str, tokenizer, budget: int) -> str:
    idx = response.find("</think>")
    if idx < 0:
        return "no_close"
    thinking_text = response[:idx]
    n_thinking = len(tokenizer.encode(thinking_text, add_special_tokens=False))
    if n_thinking >= budget - SLACK:
        return "budget_hit"
    return "self_close"


def analyze(jsonl: str, tokenizer, budget: int):
    counts = {"self_close": 0, "budget_hit": 0, "no_close": 0}
    thinking_lens = []
    for line in open(jsonl):
        ep = json.loads(line)
        for t in ep["turns"]:
            klass = classify_response(t["response"], tokenizer, budget)
            counts[klass] += 1
            idx = t["response"].find("</think>")
            if idx >= 0:
                thinking_lens.append(len(tokenizer.encode(t["response"][:idx], add_special_tokens=False)))
    total = sum(counts.values())
    return counts, total, thinking_lens


def main():
    sweep_root = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dirs = sorted(glob(os.path.join(sweep_root, "mnt_*")),
                  key=lambda p: int(os.path.basename(p).split("_")[1]))
    print(f"Sweep root: {sweep_root}")
    print(f"{'mnt':>6} {'budget':>7} {'turns':>6} {'self':>6} {'BUDGET':>7} {'no_cls':>7} "
          f"{'hit_rate':>9} {'med_think':>10} {'p90_think':>10}")
    for d in dirs:
        mnt = int(os.path.basename(d).split("_")[1])
        budget = mnt - 128
        jsonl = os.path.join(d, "arena_trajectories.jsonl")
        if not os.path.exists(jsonl):
            continue
        counts, total, thinking_lens = analyze(jsonl, tokenizer, budget)
        hit_rate = (counts["budget_hit"] + counts["no_close"]) / total
        thinking_lens.sort()
        med = thinking_lens[len(thinking_lens) // 2] if thinking_lens else 0
        p90 = thinking_lens[int(0.9 * len(thinking_lens))] if thinking_lens else 0
        marker = "  <-- under 10%" if hit_rate < 0.10 else ""
        print(f"{mnt:>6} {budget:>7} {total:>6} {counts['self_close']:>6} "
              f"{counts['budget_hit']:>7} {counts['no_close']:>7} "
              f"{hit_rate:>8.2%} {med:>10} {p90:>10}{marker}")


if __name__ == "__main__":
    main()
