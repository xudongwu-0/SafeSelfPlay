"""Analyze rollout/score/mean across recent runs to recommend fsp_score_threshold.

Reads driver logs from the last N runs, extracts per-step scores and FSP switch
events, computes rolling averages (gen 2+ only), and prints a threshold summary.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

RUNS_DIR = Path("/zfsauton/scratch/wentsec/kuhn_poker_output/runs")
N_RUNS = 10
WINDOW = 10

# ── log parsing ──────────────────────────────────────────────────────────────

METRIC_RE = re.compile(r'"rollout/score/mean":\s*([-\d.e+]+)')
FSP_SWITCH_RE = re.compile(r"FSP: adding LoRA checkpoint")
FSP_TURN_RE = re.compile(r'"fsp/turn":\s*(\d+)')
COLD_START_RE = re.compile(r"FSP cold_start.*?step (\d+)")


def find_driver_log(run_dir: Path) -> Optional[Path]:
    for log in run_dir.glob("logs/*/log_rank_DRIVER_0_1.log"):
        return log
    return None


def parse_run(run_dir: Path) -> list[dict]:
    """Return list of {score, fsp_turn, is_switch} dicts, one per log line with a score."""
    log = find_driver_log(run_dir)
    if log is None:
        return []

    steps = []
    switch_lines: set[int] = set()

    lines = log.read_text(errors="replace").splitlines()

    # First pass: find switch line numbers
    for i, line in enumerate(lines):
        if FSP_SWITCH_RE.search(line):
            switch_lines.add(i)

    # Second pass: extract metric lines
    for i, line in enumerate(lines):
        score_m = METRIC_RE.search(line)
        if not score_m:
            continue
        turn_m = FSP_TURN_RE.search(line)
        fsp_turn = int(turn_m.group(1)) if turn_m else None
        # Check if a switch happened within the next 30 lines after this metric line
        is_switch = any(j in switch_lines for j in range(i, i + 30))
        steps.append({
            "score": float(score_m.group(1)),
            "fsp_turn": fsp_turn,
            "is_switch": is_switch,
        })

    return steps


def rolling_avg(values: list[float], window: int) -> list[Optional[float]]:
    result = []
    for i, _ in enumerate(values):
        if i + 1 < window:
            result.append(None)
        else:
            result.append(sum(values[i + 1 - window: i + 1]) / window)
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_dirs = sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:N_RUNS]

    all_switch_rolling: list[float] = []   # rolling avg at gen 2+ switch points
    all_gen2plus_rolling: list[float] = [] # all rolling avg values in gen 2+

    for run_dir in run_dirs:
        steps = parse_run(run_dir)
        if not steps:
            print(f"  {run_dir.name}: no log found, skipping")
            continue

        scores = [s["score"] for s in steps]
        turns = [s["fsp_turn"] for s in steps]
        is_switch = [s["is_switch"] for s in steps]
        avgs = rolling_avg(scores, WINDOW)

        # Split into generations; skip gen 0 (first gen, vs weak base model)
        print(f"\n{'='*60}")
        print(f"Run: {run_dir.name}  ({len(steps)} steps)")

        for i, (score, turn, avg, sw) in enumerate(zip(scores, turns, avgs, is_switch)):
            if turn is None or turn < 1:
                continue  # skip gen 0
            if avg is not None:
                all_gen2plus_rolling.append(avg)
            if sw and avg is not None:
                all_switch_rolling.append(avg)
                print(f"  step {i:4d}  turn={turn}  score={score:+.3f}  rolling_avg={avg:+.3f}  ← SWITCH")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY  (window={WINDOW}, gen 2+ only)")
    print(f"{'='*60}")

    if not all_switch_rolling:
        print("No gen 2+ switches found.")
        return

    all_switch_rolling.sort()
    print(f"\nRolling-avg score at FSP switch points ({len(all_switch_rolling)} events):")
    print(f"  min    = {min(all_switch_rolling):+.3f}")
    print(f"  median = {sorted(all_switch_rolling)[len(all_switch_rolling)//2]:+.3f}")
    print(f"  mean   = {sum(all_switch_rolling)/len(all_switch_rolling):+.3f}")
    print(f"  max    = {max(all_switch_rolling):+.3f}")
    print(f"  values = {[f'{v:+.3f}' for v in all_switch_rolling]}")

    # How often each candidate threshold would have fired at switch points
    print(f"\nThreshold coverage (fraction of switch points where rolling_avg >= threshold):")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        coverage = sum(1 for v in all_switch_rolling if v >= thresh) / len(all_switch_rolling)
        # False positive rate: fraction of non-switch gen2+ steps where rolling_avg >= threshold
        non_switch_avgs = [v for v in all_gen2plus_rolling if v < 999]  # all gen2+ rolling avgs
        # (false positive = fires when win_rate-based trigger didn't fire)
        print(f"  threshold={thresh:.1f}: coverage={coverage:.0%}")

    print(f"\nRecommendation:")
    # Find threshold that covers ~80% of switch events
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        coverage = sum(1 for v in all_switch_rolling if v >= thresh) / len(all_switch_rolling)
        if coverage >= 0.8:
            print(f"  0.8+ coverage threshold: {thresh:.2f}  (covers {coverage:.0%} of switches)")
            break
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        coverage = sum(1 for v in all_switch_rolling if v >= thresh) / len(all_switch_rolling)
        if coverage >= 0.5:
            print(f"  0.5+ coverage threshold: {thresh:.2f}  (covers {coverage:.0%} of switches)")
            break


if __name__ == "__main__":
    main()
