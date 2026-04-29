"""Analyze arena_trajectories.jsonl across a max_new_tokens sweep to compute truncation rates.

A response is counted as truncated iff the decoded string contains no '</answer>' tag
(i.e., the model did not reach the structured-output stop boundary before hitting
max_new_tokens). Run:

    python analyze_truncation_sweep.py <sweep_root>
"""
import json
import os
import sys
from glob import glob


def compute_truncation(jsonl_path: str) -> tuple[int, int, float, float]:
    n_turns = 0
    n_trunc = 0
    resp_char_lens = []
    for line in open(jsonl_path):
        ep = json.loads(line)
        for turn in ep["turns"]:
            n_turns += 1
            resp = turn["response"]
            resp_char_lens.append(len(resp))
            if "</answer>" not in resp:
                n_trunc += 1
    trunc_rate = n_trunc / n_turns if n_turns else float("nan")
    median_chars = sorted(resp_char_lens)[len(resp_char_lens) // 2] if resp_char_lens else 0
    return n_turns, n_trunc, trunc_rate, median_chars


def main() -> None:
    sweep_root = sys.argv[1]
    dirs = sorted(glob(os.path.join(sweep_root, "mnt_*")),
                  key=lambda p: int(os.path.basename(p).split("_")[1]))
    print(f"Sweep root: {sweep_root}")
    print(f"{'mnt':>6} {'turns':>6} {'trunc':>6} {'rate':>8} {'med_chars':>10}")
    for d in dirs:
        mnt = int(os.path.basename(d).split("_")[1])
        jsonl = os.path.join(d, "arena_trajectories.jsonl")
        if not os.path.exists(jsonl):
            print(f"{mnt:>6} {'--':>6} {'--':>6} {'PENDING':>8} {'--':>10}")
            continue
        n, t, r, mc = compute_truncation(jsonl)
        marker = "  <-- under 10%" if r < 0.10 else ""
        print(f"{mnt:>6} {n:>6} {t:>6} {r:>7.2%} {mc:>10}{marker}")


if __name__ == "__main__":
    main()
