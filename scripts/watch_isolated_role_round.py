#!/usr/bin/env python3
"""Monitor an isolated attacker-then-defender Modal round through W&B."""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from datetime import datetime

import wandb


PROJECT = "2373025856w-the-university-of-hong-kong/self-play"


ROLE_METRICS = {
    "attacker": (
        "attacker/avg_reward",
        "attacker/info/successful_harmful_request_rate",
        "attacker/revision_faithfulness_rate",
        "attacker/cot_format_violation",
    ),
    "defender": (
        "defender/avg_reward",
        "defender/overall_harmless_rate",
        "defender/correct_refusal_rate",
        "defender/info/generated_harmful_harmlessness_rate",
        "defender/info/generated_benign_harmlessness_rate",
    ),
}


def role_for_name(name: str) -> str | None:
    if "_A20" in name:
        return "attacker"
    if "_D20" in name:
        return "defender"
    return None


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_window(history: list[dict], key: str, size: int = 5) -> float | None:
    values = [float(row[key]) for row in history[-size:] if finite(row.get(key))]
    return sum(values) / len(values) if values else None


def stop_modal_app(app_id: str) -> None:
    if not app_id:
        return
    subprocess.run(["modal", "app", "stop", "-y", app_id], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--stale-minutes", type=int, default=20)
    parser.add_argument("--modal-app-id", default="")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    last_steps: dict[str, int] = {}
    last_progress = time.monotonic()

    while True:
        runs = list(
            api.runs(
                PROJECT,
                filters={"display_name": {"$regex": args.run_suffix}},
                order="created_at",
            )
        )
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"[{timestamp}] runs={len(runs)}", flush=True)

        roles_finished: set[str] = set()
        fatal_reasons: list[str] = []
        saw_progress = False
        for run in runs:
            role = role_for_name(run.name)
            if role is None:
                continue
            history = sorted(
                run.history(samples=2000, pandas=False),
                key=lambda row: row.get("train/global_step", -1),
            )
            step = int(history[-1].get("train/global_step", 0)) if history else 0
            if step > last_steps.get(run.id, -1):
                saw_progress = True
                last_steps[run.id] = step
            if run.state == "finished":
                roles_finished.add(role)
            if run.state in {"crashed", "failed", "killed"}:
                fatal_reasons.append(f"{role} W&B state={run.state}")

            summary = {
                key: mean_window(history, key)
                for key in (*ROLE_METRICS[role], "actor/clipfrac", "actor/kl_loss")
            }
            print(
                f"  {role}: state={run.state} step={step} url={run.url} "
                f"last5={summary}",
                flush=True,
            )

            recent = history[-3:]
            if len(recent) == 3:
                if all(float(row.get("actor/samples_used", 1)) <= 0 for row in recent):
                    fatal_reasons.append(f"{role} used zero samples for 3 steps")
                if all(
                    float(row.get("score/zero_variance_group_frac", 0)) >= 0.999
                    for row in recent
                ):
                    fatal_reasons.append(f"{role} reward variance was zero for 3 steps")
                checked = [
                    row.get(key)
                    for row in recent
                    for key in (*ROLE_METRICS[role], "actor/kl_loss")
                    if row.get(key) is not None
                ]
                if any(not finite(value) for value in checked):
                    fatal_reasons.append(f"{role} produced NaN/Inf metrics")

        if saw_progress:
            last_progress = time.monotonic()
        if fatal_reasons:
            print("FATAL: " + "; ".join(fatal_reasons), flush=True)
            stop_modal_app(args.modal_app_id)
            return 2
        if roles_finished == {"attacker", "defender"}:
            print("Both isolated roles finished.", flush=True)
            return 0
        if runs and time.monotonic() - last_progress > args.stale_minutes * 60:
            print(f"FATAL: no W&B step progress for {args.stale_minutes} minutes", flush=True)
            stop_modal_app(args.modal_app_id)
            return 3
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
