#!/usr/bin/env python3
"""Replay an existing W&B run into the legacy comprehensive dashboard schema.

This script never trains a model. It copies the scalar history already stored
in W&B, adds only mathematically equivalent legacy aliases, and uploads local
sample tables from the same completed run. Metrics that were not logged by the
source trainer are intentionally left absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import wandb
from wandb_workspaces.reports import v2 as wr


DEFAULT_ENTITY = "2373025856w-the-university-of-hong-kong"
DEFAULT_PROJECT = "self-play"
DEFAULT_SOURCE_RUN = "usd1f5c0"
DEFAULT_ARTIFACT_DIR = Path(
    "/home/xudong/work/self_play/checkpoints/"
    "fixedseed_role_lora_learning_20260730"
)


# These aliases preserve the old ROLL panel grouping only when the source
# quantity has the same interpretation. We deliberately do not synthesize
# unavailable entropy, gradient, PPO-ratio, min/max, or throughput metrics.
DIRECT_ALIASES = {
    "actor/pg_loss": "train/policy_loss",
    "actor/lr": "train/actor_lr",
    "actor/kl_loss": "train/kl",
    "critic/kl": "train/kl",
    "critic/returns/mean": "train/return",
    "critic/rewards/mean": "train/reward",
    "critic/score/mean": "train/reward",
    "rollout/score/mean": "train/reward",
    "score/mean": "train/reward",
    "score/attacker_reward": "attacker/avg_reward",
    "rollout/response_length/mean": "train/response_length",
    "rollout/non_prompt_length/mean": "train/response_length",
    "tokens/response_length/mean": "train/response_length",
    "tokens/non_prompt_length/mean": "train/response_length",
    "safety/attack_success": "attacker/request_success_rate",
    "env/RedTeamSafety/attack_success": "attacker/request_success_rate",
    "env/RedTeamSafety/attacker_goal_success": "attacker/request_success_rate",
    "env/RedTeamSafety/attacker_reward": "attacker/avg_reward",
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_csv_table(path: Path) -> wandb.Table:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        data = list(reader)
    return wandb.Table(columns=columns, data=data)


def _load_probe_summary(path: Path) -> tuple[wandb.Table, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["summaries"]
    columns = [
        "model",
        "samples",
        "attack_success_rate",
        "prompt_harmful_rate",
        "defender_harmful_rate",
        "defender_refusal_rate",
        "attacker_format_complete_rate",
    ]
    table = wandb.Table(
        columns=columns,
        data=[[row.get(column) for column in columns] for row in rows],
    )
    return table, payload


def _legacy_aliases(row: dict[str, Any]) -> dict[str, float]:
    aliases: dict[str, float] = {}
    for destination, source in DIRECT_ALIASES.items():
        value = row.get(source)
        if _finite_number(value):
            aliases[destination] = float(value)

    response_length = row.get("train/response_length")
    total_length = row.get("train/total_length")
    if _finite_number(response_length) and _finite_number(total_length):
        prompt_length = float(total_length) - float(response_length)
        aliases["rollout/prompt_length/mean"] = prompt_length
        aliases["tokens/prompt_length/mean"] = prompt_length

    format_violation = row.get("attacker/cot_format_violation")
    if _finite_number(format_violation):
        valid = 1.0 - float(format_violation)
        aliases["quality/action_is_valid"] = valid
        aliases["env/RedTeamSafety/action_is_valid"] = valid

    harmful = row.get("attacker/debug/n_generated_harmful")
    benign = row.get("attacker/debug/n_generated_benign")
    if _finite_number(harmful) and _finite_number(benign):
        count = float(harmful) + float(benign)
        if count > 0:
            aliases["data/attack_seed_harmful"] = float(harmful) / count
            aliases["data/attack_seed_benign"] = float(benign) / count

    return aliases


def _create_report(
    *,
    entity: str,
    project: str,
    source_run_id: str,
    replay_run_name: str,
) -> str:
    runset = wr.Runset(
        entity=entity,
        project=project,
        name="Latest fixed-seed attacker run",
        filters=f"Config('viz_source_run_id') = '{source_run_id}'",
    )

    def grid(*panels: wr.LinePlot | wr.MediaBrowser) -> wr.PanelGrid:
        return wr.PanelGrid(runsets=[runset], panels=list(panels))

    report = wr.Report(
        entity=entity,
        project=project,
        title=f"Fixed-Seed A50 Comprehensive Viz - {source_run_id}",
        description=(
            "No-retraining visualization replay of "
            f"`{replay_run_name}`. Scalar history and sample tables come from "
            f"source run `{source_run_id}`. Legacy aliases are only added when "
            "they are mathematically equivalent; unavailable metrics are not "
            "fabricated."
        ),
        width="fluid",
        blocks=[
            wr.CalloutBlock(
                "This report replays an already completed run. It is not a new "
                "training experiment. Entropy, grad norm, PPO-ratio statistics, "
                "reward min/max, and throughput were not recorded by the source "
                "trainer and therefore cannot be backfilled."
            ),
            wr.H2("Learning Overview"),
            grid(
                wr.LinePlot(
                    title="Reward and return",
                    x="train/global_step",
                    y=["score/mean", "critic/returns/mean"],
                ),
                wr.LinePlot(
                    title="Attack success",
                    x="train/global_step",
                    y=["safety/attack_success"],
                    range_y=(0, 1),
                ),
                wr.LinePlot(
                    title="Faithfulness and valid format",
                    x="train/global_step",
                    y=[
                        "attacker/revision_faithfulness_rate",
                        "quality/action_is_valid",
                    ],
                    range_y=(0, 1),
                ),
            ),
            wr.H2("Optimization"),
            grid(
                wr.LinePlot(
                    title="Policy gradient loss",
                    x="train/global_step",
                    y=["actor/pg_loss"],
                ),
                wr.LinePlot(
                    title="KL to role-start policy",
                    x="train/global_step",
                    y=["actor/kl_loss"],
                ),
                wr.LinePlot(
                    title="Actor learning rate",
                    x="train/global_step",
                    y=["actor/lr"],
                ),
            ),
            wr.H2("Safety Game"),
            grid(
                wr.LinePlot(
                    title="Attacker reward",
                    x="train/global_step",
                    y=["score/attacker_reward"],
                ),
                wr.LinePlot(
                    title="Goal and harmful compliance rates",
                    x="train/global_step",
                    y=[
                        "attacker/request_success_rate",
                        "attacker/info/win_harmful_compliance_rate",
                    ],
                    range_y=(0, 1),
                ),
                wr.LinePlot(
                    title="Request outcome diagnostics",
                    x="train/global_step",
                    y=[
                        "attacker/request_correct_refusal_rate",
                        "attacker/cot_format_violation",
                    ],
                    range_y=(0, 1),
                ),
            ),
            wr.H2("Rewrite Quality and Diversity"),
            grid(
                wr.LinePlot(
                    title="Rewrite faithfulness",
                    x="train/global_step",
                    y=[
                        "attacker/revision_faithfulness_rate",
                        "attacker/info/harmful_revision_faithfulness_rate",
                    ],
                    range_y=(0, 1),
                ),
                wr.LinePlot(
                    title="Inverse self-BLEU",
                    x="train/global_step",
                    y=[
                        "bleu/attacker_inv_self_bleu",
                        "bleu/attacker_thinking_inv_self_bleu",
                        "bleu/attacker_answer_inv_self_bleu",
                    ],
                    range_y=(0, 1),
                ),
                wr.LinePlot(
                    title="Inverse SBERT similarity",
                    x="train/global_step",
                    y=[
                        "sbert/attacker_inv_sbert",
                        "sbert/attacker_thinking_inv_sbert",
                        "sbert/attacker_answer_inv_sbert",
                    ],
                    range_y=(0, 1),
                ),
            ),
            wr.H2("Lengths and Data Flow"),
            grid(
                wr.LinePlot(
                    title="Prompt and response lengths",
                    x="train/global_step",
                    y=[
                        "tokens/prompt_length/mean",
                        "tokens/response_length/mean",
                    ],
                ),
                wr.LinePlot(
                    title="Attacker reasoning and answer lengths",
                    x="train/global_step",
                    y=[
                        "length/attacker_thinking_length",
                        "length/attacker_answer_length",
                    ],
                ),
                wr.LinePlot(
                    title="Samples used and lost",
                    x="train/global_step",
                    y=[
                        "debug/n_samples",
                        "debug/em_lost_samples",
                        "debug/rb_lost_samples",
                    ],
                ),
            ),
            wr.H2("Complete Sample Tables and Independent Probe"),
            grid(
                wr.MediaBrowser(
                    title="Training game conversations",
                    media_keys=["samples/game_log_all"],
                    mode="grid",
                ),
                wr.MediaBrowser(
                    title="Attacker prompts, responses, outcomes, rewards",
                    media_keys=["samples/attacker_response_all"],
                    mode="grid",
                ),
                wr.MediaBrowser(
                    title="Independent paired n=256 evaluation",
                    media_keys=[
                        "evaluation/paired_probe",
                        "evaluation/attack_success_comparison",
                    ],
                    mode="grid",
                ),
            ),
        ],
    )
    report.save()
    return report.url


def replay(args: argparse.Namespace) -> tuple[str, str]:
    api = wandb.Api()
    source_path = f"{args.entity}/{args.project}/{args.source_run_id}"
    source = api.run(source_path)
    history = [
        row
        for row in source.scan_history()
        if _finite_number(row.get("train/global_step"))
    ]
    history.sort(key=lambda row: int(row["train/global_step"]))

    steps = [int(row["train/global_step"]) for row in history]
    if steps != list(range(1, 51)):
        raise RuntimeError(
            f"Expected completed steps 1..50 in {source_path}, found {steps}"
        )

    artifact_dir = args.artifact_dir.resolve()
    game_table = _load_csv_table(artifact_dir / "game_log.csv")
    attacker_table = _load_csv_table(artifact_dir / "attacker_response_log.csv")
    probe_table, probe = _load_probe_summary(
        artifact_dir / "step25_paired_n256" / "summary.json"
    )
    probe_rows = {row["model"]: row for row in probe["summaries"]}
    chart_table = wandb.Table(
        columns=["model", "attack_success_rate"],
        data=[
            ["SFT start", probe_rows["sft_start"]["attack_success_rate"]],
            [
                "Trained step25",
                probe_rows["trained_adapter"]["attack_success_rate"],
            ],
        ],
    )

    replay_run_name = (
        args.name
        or f"VIZ_REPLAY_no_retraining__{args.source_run_id}__legacy_comprehensive"
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id or f"viz{args.source_run_id}",
        name=replay_run_name,
        job_type="visualization-replay",
        tags=[
            "no-retraining",
            "visualization-replay",
            "legacy-dashboard-compatible",
            "fixed-seed",
            "attacker-only",
        ],
        notes=(
            "Visualization-only replay of an already completed run. No model "
            "training or new rollout occurred. Exact legacy aliases are "
            "documented in config; unavailable metrics are omitted."
        ),
        config={
            "viz_source_run_id": args.source_run_id,
            "viz_source_run_name": source.name,
            "viz_source_run_url": source.url,
            "viz_replay_only": True,
            "viz_training_performed": False,
            "viz_history_rows": len(history),
            "viz_alias_version": 1,
            "viz_direct_aliases": DIRECT_ALIASES,
            "viz_unavailable_metrics": [
                "actor/entropy",
                "actor_train/grad_norm",
                "actor/ratio_min",
                "actor/ratio_mean",
                "actor/ratio_max",
                "critic/rewards/min",
                "critic/rewards/max",
                "rollout/score/min",
                "rollout/score/max",
                "system/tps",
                "time/step_total",
            ],
        },
        resume="never",
    )
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")

    for index, source_row in enumerate(history):
        output = {
            key: value
            for key, value in source_row.items()
            if not key.startswith("_") and _finite_number(value)
        }
        output.update(_legacy_aliases(source_row))
        output["train/global_step"] = int(source_row["train/global_step"])
        output["source/runtime_seconds"] = float(source_row.get("_runtime", 0.0))

        if index == len(history) - 1:
            output["samples/game_log_all"] = game_table
            output["samples/attacker_response_all"] = attacker_table
            output["evaluation/paired_probe"] = probe_table
            output["evaluation/attack_success_comparison"] = wandb.plot.bar(
                chart_table,
                "model",
                "attack_success_rate",
                title="Independent paired n=256 attack success",
            )
        run.log(output)

    paired = probe["paired_comparison"]
    run.summary.update(
        {
            "source_run_url": source.url,
            "source_run_state": source.state,
            "source_history_rows": len(history),
            "training_performed_in_this_run": False,
            "independent_eval/sft_start_attack_success": probe_rows["sft_start"][
                "attack_success_rate"
            ],
            "independent_eval/trained_step25_attack_success": probe_rows[
                "trained_adapter"
            ]["attack_success_rate"],
            "independent_eval/improvements": paired[
                "sft_start_failure_trained_success"
            ],
            "independent_eval/regressions": paired[
                "sft_start_success_trained_failure"
            ],
            "independent_eval/exact_p": paired[
                "exact_two_sided_binomial_p"
            ],
        }
    )
    replay_run_url = run.url
    run.finish()

    report_url = _create_report(
        entity=args.entity,
        project=args.project,
        source_run_id=args.source_run_id,
        replay_run_name=replay_run_name,
    )
    return replay_run_url, report_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--name")
    return parser.parse_args()


if __name__ == "__main__":
    replay_run_url, report_url = replay(parse_args())
    print(f"Replay run: {replay_run_url}")
    print(f"Report: {report_url}")
