"""Post sweep run summary metrics to wandb.

After a sweep run finishes:
  - Parse gpu_util.csv (nvidia-smi -l 2 output) → mean / max GPU utilization per GPU and overall
  - Find the matching wandb run by exp_name + project
  - Pull per-step system/tps, time/step_*, write summary fields
  - Tag the run with round + variant for easy grouping
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import wandb


def parse_gpu_util(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    per_gpu_util: dict[int, list[float]] = defaultdict(list)
    per_gpu_mem: dict[int, list[float]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
                util = float(row["utilization_gpu_pct"])
                mem = float(row["memory_used_mib"])
            except (ValueError, KeyError):
                continue
            per_gpu_util[idx].append(util)
            per_gpu_mem[idx].append(mem)
    if not per_gpu_util:
        return {}
    out = {}
    all_util = []
    for idx, vals in sorted(per_gpu_util.items()):
        out[f"gpu_util_mean_g{idx}"] = sum(vals) / len(vals)
        out[f"gpu_util_max_g{idx}"] = max(vals)
        out[f"gpu_mem_max_mib_g{idx}"] = max(per_gpu_mem[idx])
        all_util.extend(vals)
    out["gpu_util_mean_overall"] = sum(all_util) / len(all_util)
    out["gpu_util_samples"] = len(all_util)
    return out


WANDB_ENTITY = "cwz19"


def find_run(api: wandb.Api, project: str, exp_name: str):
    path = f"{WANDB_ENTITY}/{project}"
    runs = api.runs(path, filters={"display_name": exp_name}, order="-created_at")
    runs = list(runs)
    if not runs:
        runs = api.runs(path, filters={"config.exp_name": exp_name}, order="-created_at")
        runs = list(runs)
    return runs[0] if runs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--gpu_util_csv", required=True, type=Path)
    ap.add_argument("--slurm_job_id", default="")
    ap.add_argument("--round", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tag", default="sweep_a6000",
                    help="Tag added to the wandb run alongside the round name (default: sweep_a6000)")
    args = ap.parse_args()

    util_metrics = parse_gpu_util(args.gpu_util_csv)
    if not util_metrics:
        print(f"WARN: no GPU util samples parsed from {args.gpu_util_csv}", file=sys.stderr)

    api = wandb.Api()
    run = find_run(api, args.project, args.exp_name)
    if run is None:
        print(f"ERROR: no wandb run named {args.exp_name} in {args.project}", file=sys.stderr)
        return 1

    # `keys=[...]` filter excludes rows missing any key — use samples= and pull keys ourselves.
    history = run.history(samples=200, pandas=False)
    metric_keys = ("system/tps", "system/tps_rollout", "system/tps_train",
                   "time/step_rollout", "time/step_train",
                   "time/step_old_log_probs_values", "time/step_total")
    for k in metric_keys:
        vals = [h.get(k) for h in history if h.get(k) is not None]
        if not vals:
            continue
        steady = vals[1:] if len(vals) > 1 else vals  # drop warmup step 0
        suffix = k.replace("/", "_")
        run.summary[f"mean_{suffix}_steady"] = sum(steady) / len(steady)
        if k == "system/tps":
            run.summary["peak_tps"] = max(vals)
            run.summary["mean_tps_steady"] = sum(steady) / len(steady)

    for k, v in util_metrics.items():
        run.summary[k] = v

    run.summary["sweep_round"] = args.round
    run.summary["sweep_variant"] = args.variant
    if args.slurm_job_id:
        run.summary["slurm_job_id"] = args.slurm_job_id

    run.tags = list(set((run.tags or []) + [args.tag, args.round]))
    run.update()
    print(f"OK: pushed {len(util_metrics)} util metrics + tps summary to {args.project}/{run.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
