#!/bin/bash
# A100x4 single-variant GRPO sweep run (interactive mode).
# Coordinate-descent benchmark against A6000x4 v2 SOTA.
#
# Node allocation (run once per session, max 1h):
#   salloc --mem=240g --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
#     --partition=gpuA100x4-interactive --account=bfoz-delta-gpu \
#     --time=1:00:00 --gpus-per-node=4
#
# Usage:
#   ROUND=<round> VARIANT=<variant> bash scripts/delta/run_grpo_sweep_a100.sh
#
# Session plan (~5 steps per run, ~10 min/run):
#   Session 1 (~60 min): round1_gpu_split (3), round2_train_bs (4)
#   Session 2 (~70 min): round3_rollout_bs (3), round4_group_size (4)
#   Session 3 (~60 min): round5_eager (2), round6_mnt (4), round7_gmu (3)
#
# Constraints (enforced by agentic_config.py):
#   rollout_batch_size == num_env_groups * group_size
#   sum(num_groups_partition) == num_env_groups
# These are kept consistent in every override below.
#
# Batch size approach: gradient_accumulation_steps=1, sweep bs freely (full batch).
# After each round: update BASELINE_OVERRIDES with the winner.

set -ex

ROUND=${ROUND:?Set ROUND env var, e.g. ROUND=round1_gpu_split}
VARIANT=${VARIANT:?Set VARIANT env var, e.g. VARIANT=split_3tr1inf}

# ---- Baseline: A6000 v2 SOTA adapted for A100 40GB ----
# Invariants: rollout_batch_size=256, group_size=4, num_env_groups=64, num_groups_partition=[64].
# Update entries marked "R<N> winner" after each round completes.
BASELINE_OVERRIDES=(
    "actor_train.device_mapping='[0,1,2]'"             # R1 winner (update after round1)
    "actor_infer.device_mapping='[3]'"                  # R1 winner
    "actor_train.training_args.per_device_train_batch_size=4"   # R2 winner (update after round2)
    "actor_train.training_args.gradient_accumulation_steps=1"   # full batch (ga=1)
    "actor_train.infer_batch_size=4"
    "actor_infer.strategy_args.strategy_config.enforce_eager=false"             # R5 winner
    "actor_infer.strategy_args.strategy_config.max_num_batched_tokens=32768"   # R6 winner
    "actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.95"    # R7 winner
    "train_env_manager.num_env_groups=32"               # = rollout_batch_size / group_size = 256/8  (R4 winner)
    "train_env_manager.num_groups_partition=[32]"       # must equal num_env_groups
    "train_env_manager.group_size=8"                    # R4 winner
    "rollout_batch_size=256"                             # R3 winner
    "sequence_length=1024"
    "actor_infer.generating_args.max_new_tokens=512"
)

BENCH_OVERRIDES=(
    "max_steps=5"
    "eval_steps=100"
    "save_steps=10000"
    "fsp_save_steps=0"
    "logging_steps=1"
)

# ---- Per-variant override ----
# Whenever rollout_batch_size or group_size changes, num_env_groups and num_groups_partition
# must also change to satisfy: rollout_batch_size == num_env_groups * group_size.
case "${ROUND}/${VARIANT}" in
    # R1: GPU split — biggest lever per A6000 v2 experience
    # Baseline rb=256, gs=4 → neg=64 stays unchanged across all split variants.
    round1_gpu_split/split_1tr3inf)
        VARIANT_OVERRIDE="actor_train.device_mapping='[0]' actor_infer.device_mapping='[1,2,3]'"
        ;;
    round1_gpu_split/split_2tr2inf)
        VARIANT_OVERRIDE="actor_train.device_mapping='[0,1]' actor_infer.device_mapping='[2,3]'"
        ;;
    round1_gpu_split/split_3tr1inf)
        VARIANT_OVERRIDE="actor_train.device_mapping='[0,1,2]' actor_infer.device_mapping='[3]'"
        ;;

    # R2: per_device_train_batch_size — full batch (ga=1), sweep bs.
    # A100 40GB: expect OOM around bs=32+.
    round2_train_bs/bs4)
        VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=4"
        ;;
    round2_train_bs/bs8)
        VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=8"
        ;;
    round2_train_bs/bs16)
        VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=16"
        ;;
    round2_train_bs/bs32)
        VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=32"
        ;;

    # R3: rollout_batch_size — keep group_size=4, update neg/ngp accordingly.
    round3_rollout_bs/rb64)
        VARIANT_OVERRIDE="rollout_batch_size=64 train_env_manager.num_env_groups=16 train_env_manager.num_groups_partition=[16]"
        ;;
    round3_rollout_bs/rb128)
        VARIANT_OVERRIDE="rollout_batch_size=128 train_env_manager.num_env_groups=32 train_env_manager.num_groups_partition=[32]"
        ;;
    round3_rollout_bs/rb256)
        VARIANT_OVERRIDE="rollout_batch_size=256 train_env_manager.num_env_groups=64 train_env_manager.num_groups_partition=[64]"
        ;;
    round3_rollout_bs/rb512)
        VARIANT_OVERRIDE="rollout_batch_size=512 train_env_manager.num_env_groups=128 train_env_manager.num_groups_partition=[128]"
        ;;

    # R4: group_size — keep rollout_batch_size=256, update neg/ngp accordingly.
    round4_group_size/gs1)
        VARIANT_OVERRIDE="train_env_manager.group_size=1 train_env_manager.num_env_groups=256 train_env_manager.num_groups_partition=[256]"
        ;;
    round4_group_size/gs2)
        VARIANT_OVERRIDE="train_env_manager.group_size=2 train_env_manager.num_env_groups=128 train_env_manager.num_groups_partition=[128]"
        ;;
    round4_group_size/gs4)
        VARIANT_OVERRIDE="train_env_manager.group_size=4 train_env_manager.num_env_groups=64 train_env_manager.num_groups_partition=[64]"
        ;;
    round4_group_size/gs8)
        VARIANT_OVERRIDE="train_env_manager.group_size=8 train_env_manager.num_env_groups=32 train_env_manager.num_groups_partition=[32]"
        ;;

    # R5: enforce_eager
    round5_eager/eager_true)   VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.enforce_eager=true" ;;
    round5_eager/eager_false)  VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.enforce_eager=false" ;;

    # R6: max_num_batched_tokens
    round6_mnt/mnt4096)        VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=4096" ;;
    round6_mnt/mnt8192)        VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=8192" ;;
    round6_mnt/mnt16384)       VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=16384" ;;
    round6_mnt/mnt32768)       VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=32768" ;;

    # R7: gpu_memory_utilization
    round7_gmu/gmu90)          VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.90" ;;
    round7_gmu/gmu92)          VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.92" ;;
    round7_gmu/gmu95)          VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.95" ;;

    *) echo "ERROR: unknown round/variant ${ROUND}/${VARIANT}"; exit 2 ;;
esac

# ---- Paths ----
CONDA_ROOT=/projects/bfoz/wchen11/anaconda3
ROLL_DIR=/u/wchen11/ROLL
EXP_NAME="kuhn_a100_${ROUND}_${VARIANT}"
SWEEP_ROOT=/projects/bfoz/wchen11/kuhn_sweep_a100/${ROUND}/${VARIANT}
mkdir -p ${SWEEP_ROOT}/logs ${SWEEP_ROOT}/render

source ${CONDA_ROOT}/bin/activate
conda activate ${CONDA_ROOT}/envs/roll2

[ -f /projects/bfoz/wchen11/.env_roll ] && source /projects/bfoz/wchen11/.env_roll
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/tmp/kuhn_sweep_$$
export PYTHONPATH=${ROLL_DIR}:${PYTHONPATH}
mkdir -p $TMPDIR

df -h /projects/bfoz/wchen11
nvidia-smi
ray stop --force 2>/dev/null || true
sleep 2

# Background GPU utilization monitor (2 s cadence)
GPU_UTIL_CSV=${SWEEP_ROOT}/logs/gpu_util.csv
echo "timestamp,index,utilization_gpu_pct,memory_used_mib" > ${GPU_UTIL_CSV}
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 2 >> ${GPU_UTIL_CSV} &
NVSMI_PID=$!
trap "kill ${NVSMI_PID} >/dev/null 2>&1 || true; rm -rf $TMPDIR >/dev/null 2>&1 || true" EXIT

# ---- Run ----
cd $ROLL_DIR
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_fsp_train \
    exp_name=${EXP_NAME} \
    output_dir=${SWEEP_ROOT} \
    logging_dir=${SWEEP_ROOT}/logs \
    checkpoint_config.output_dir=${SWEEP_ROOT}/render \
    track_with=wandb \
    tracker_kwargs.project=kuhn-sweep-a100 \
    tracker_kwargs.name=${EXP_NAME} \
    "${BENCH_OVERRIDES[@]}" \
    "${BASELINE_OVERRIDES[@]}" \
    ${VARIANT_OVERRIDE} \
    2>&1 | tee ${SWEEP_ROOT}/logs/run.log
PY_EXIT=${PIPESTATUS[0]}

kill ${NVSMI_PID} 2>/dev/null || true
sleep 2

# ---- Parse GPU utilization ----
echo "=== GPU util summary (${ROUND}/${VARIANT}) ==="
export GPU_UTIL_CSV
python3 - <<'PYEOF'
import csv, sys, os

csv_path = os.environ.get("GPU_UTIL_CSV", "")
if not csv_path or not os.path.exists(csv_path):
    print("gpu_util.csv not found"); sys.exit(0)

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rows.append((int(row["index"]), float(row["utilization_gpu_pct"])))
        except (ValueError, KeyError):
            pass

if not rows:
    print("no data"); sys.exit(0)

from collections import defaultdict
per_gpu = defaultdict(list)
for idx, util in rows:
    per_gpu[idx].append(util)

for idx in sorted(per_gpu):
    vals = per_gpu[idx]
    print(f"  GPU{idx}: mean={sum(vals)/len(vals):.1f}%  max={max(vals):.1f}%  n={len(vals)}")

all_vals = [u for vals in per_gpu.values() for u in vals]
print(f"  Overall mean: {sum(all_vals)/len(all_vals):.1f}%")
PYEOF

echo "===== SWEEP RUN DONE (round=${ROUND} variant=${VARIANT} exit=${PY_EXIT}) ====="
exit ${PY_EXIT}
