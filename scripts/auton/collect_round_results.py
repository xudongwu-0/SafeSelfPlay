"""Collect a sweep round's results from wandb and emit a markdown table.

Pulls runs tagged with the round name from the self-play-debug wandb project,
prints a markdown table, and (optionally) appends it to README.md.

Usage:
  python collect_round_results.py --round round1_eager \\
         --variants eager_true eager_false \\
         --param enforce_eager \\
         --append-to examples/agentic_demo/benchmark_grpo_sweep/A6000x4/README.md
"""
import argparse
import sys
from pathlib import Path

import wandb

WANDB_ENTITY = "cwz19"


def fmt_num(v, fmt=".1f"):
    if v is None:
        return "—"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, help="round dir name (e.g. round1_eager)")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="variant names in order (e.g. eager_true eager_false)")
    ap.add_argument("--param", required=True, help="display name for swept parameter column")
    ap.add_argument("--project", default="self-play-debug")
    ap.add_argument("--append-to", type=Path, default=None,
                    help="if set, append the table block to this README.md")
    ap.add_argument("--exp_name_prefix", default="kuhn_a6000",
                    help="Prefix for variant exp_names: <prefix>_<round>_<variant> (default: kuhn_a6000)")
    args = ap.parse_args()

    api = wandb.Api()
    rows = []
    best_tps = -1.0
    best_variant = None
    for variant in args.variants:
        exp_name = f"{args.exp_name_prefix}_{args.round}_{variant}"
        runs = list(api.runs(f"{WANDB_ENTITY}/{args.project}",
                             filters={"display_name": exp_name},
                             order="-created_at"))
        if not runs:
            rows.append({"variant": variant, "missing": True})
            continue
        run = runs[0]
        s = run.summary
        tps = s.get("mean_tps_steady")
        rows.append({
            "variant": variant,
            "tps": tps,
            "rollout_s": s.get("mean_time_step_rollout_steady"),
            "train_s": s.get("mean_time_step_train_steady"),
            "logprobs_s": s.get("mean_time_step_old_log_probs_values_steady"),
            "util": s.get("gpu_util_mean_overall"),
            "util_g0": s.get("gpu_util_mean_g0"),
            "util_g1": s.get("gpu_util_mean_g1"),
            "util_g2": s.get("gpu_util_mean_g2"),
            "util_g3": s.get("gpu_util_mean_g3"),
            "state": run.state,
        })
        if tps and tps > best_tps:
            best_tps = tps
            best_variant = variant

    spread_pct = None
    valid_tps = [r.get("tps") for r in rows if r.get("tps")]
    if len(valid_tps) >= 2:
        spread_pct = (max(valid_tps) - min(valid_tps)) / max(valid_tps) * 100

    no_clear_winner = spread_pct is not None and spread_pct <= 2.0

    lines = []
    lines.append(f"### {args.round}: `{args.param}` sweep")
    lines.append("")
    lines.append(f"| {args.param} | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | "
                 f"GPU util mean (%) | per-GPU util (g0/g1/g2/g3) | state |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['variant']} | — | — | — | — | — | — | NO WANDB RUN |")
            continue
        marker = " **(winner)**" if (not no_clear_winner and r["variant"] == best_variant) else ""
        per_gpu = "/".join(fmt_num(r.get(f"util_g{i}"), ".0f") for i in range(4))
        lines.append(
            f"| {r['variant']}{marker} | {fmt_num(r['rollout_s'])} | {fmt_num(r['train_s'])} | "
            f"{fmt_num(r['logprobs_s'])} | {fmt_num(r['tps'])} | "
            f"{fmt_num(r['util'], '.1f')} | {per_gpu} | {r['state']} |"
        )
    lines.append("")
    if no_clear_winner:
        lines.append(f"**Spread {spread_pct:.1f}% ≤ 2% — no clear winner; keep A40 default.**")
    elif best_variant:
        lines.append(f"**Winner: `{best_variant}` ({fmt_num(best_tps)} tok/s, "
                     f"+{spread_pct:.1f}% spread).**")
    lines.append("")
    block = "\n".join(lines)

    print(block)

    if args.append_to:
        with args.append_to.open("a") as f:
            f.write("\n" + block)
        print(f"\nAppended to {args.append_to}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
