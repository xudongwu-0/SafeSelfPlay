#!/bin/bash
# Single-variant A6000 sweep run.
# Launches agent_kuhn_poker_fsp_train.yaml with Hydra CLI overrides:
#   - 5-step bench knobs
#   - A40-winner baseline
#   - one swept hyperparameter (the variant)
#
# Yaml-inheritance composition in the existing setup silently drops parent
# fields (because fsp_train.yaml lacks `# @package _global_`); CLI overrides
# avoid this by mutating the loaded config in place.
#
# Usage:
#   sbatch scripts/auton/run_kuhn_poker_sweep.sh <round> <variant>
# e.g. sbatch scripts/auton/run_kuhn_poker_sweep.sh round1_eager eager_false
#
#SBATCH --job-name=kuhn_sweep
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_sweep_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_sweep_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a6000:4

set -ex

ROUND=${1:?usage: sbatch run_kuhn_poker_sweep.sh <round> <variant>}
VARIANT=${2:?usage: sbatch run_kuhn_poker_sweep.sh <round> <variant>}

# ---------- A40-winner baseline (locked carry-overs into every round) ----------
# R7+ baseline: 3tr+1inf split with DP=3 → ga=3 to keep eff_batch≈36.
# Also bakes R5 winner mnt=4096 (was missed before).
A40_WINNER_OVERRIDES=(
    "actor_train.device_mapping='[0,1,2]'"          # R7 winner (was [0])
    "actor_infer.device_mapping='[3]'"              # R7 winner (was list(range(1,4)))
    "actor_train.training_args.per_device_train_batch_size=4"
    "actor_train.training_args.gradient_accumulation_steps=3"  # tied for DP=3 → eff≈36 (was 8 at DP=1)
    "actor_train.infer_batch_size=4"   # round 4 winner W4
    "actor_infer.strategy_args.strategy_config.enforce_eager=false"
    "actor_infer.strategy_args.strategy_config.max_num_batched_tokens=4096"  # R5 winner W5
    "train_env_manager.num_env_groups=32"
    "train_env_manager.group_size=1"   # round 2 winner W2
    "sequence_length=1024"
    "actor_infer.generating_args.max_new_tokens=512"
    "rollout_batch_size=256"           # R9 winner W9 (868.5 TPS, +9.5% util over rb128)
)

# ---------- Bench knobs (short run, no checkpoints, no FSP snapshots) --------
BENCH_OVERRIDES=(
    "max_steps=5"
    "eval_steps=100"
    "save_steps=10000"
    "fsp_save_steps=0"
    "logging_steps=1"
)

# ---------- Per-variant override: swept hyperparameter for this run ----------
case "${ROUND}/${VARIANT}" in
  round1_eager/eager_true)        VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.enforce_eager=true" ;;
  round1_eager/eager_false)       VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.enforce_eager=false" ;;
  round2_group_size/gs1)          VARIANT_OVERRIDE="train_env_manager.group_size=1" ;;
  round2_group_size/gs2)          VARIANT_OVERRIDE="train_env_manager.group_size=2" ;;
  round2_group_size/gs4)          VARIANT_OVERRIDE="train_env_manager.group_size=4" ;;
  round2_group_size/gs8)          VARIANT_OVERRIDE="train_env_manager.group_size=8" ;;
  round3_train_bs/bs1)            VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=1 actor_train.training_args.gradient_accumulation_steps=32" ;;
  round3_train_bs/bs2)            VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=16" ;;
  round3_train_bs/bs4)            VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=4 actor_train.training_args.gradient_accumulation_steps=8" ;;
  round3_train_bs/bs8)            VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=8 actor_train.training_args.gradient_accumulation_steps=4" ;;
  round4_infer_bs/ibs1)           VARIANT_OVERRIDE="actor_train.infer_batch_size=1" ;;
  round4_infer_bs/ibs4)           VARIANT_OVERRIDE="actor_train.infer_batch_size=4" ;;
  round4_infer_bs/ibs8)           VARIANT_OVERRIDE="actor_train.infer_batch_size=8" ;;
  round4_infer_bs/ibs16)          VARIANT_OVERRIDE="actor_train.infer_batch_size=16" ;;
  round4_infer_bs/ibs32)          VARIANT_OVERRIDE="actor_train.infer_batch_size=32" ;;
  round5_mnt/mnt4096)             VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=4096" ;;
  round5_mnt/mnt8192)             VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=8192" ;;
  round5_mnt/mnt16384)            VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=16384" ;;
  round5_mnt/mnt32768)            VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_num_batched_tokens=32768" ;;
  # 2-train + 2-infer split. ga halved (8 → 4) so effective batch stays 32 (was 1 train GPU × bs4 × ga8).
  round6_gpu_split/train2_infer2) VARIANT_OVERRIDE="actor_train.device_mapping='[0,1]' actor_infer.device_mapping='[2,3]' actor_train.training_args.gradient_accumulation_steps=4" ;;
  # R7: 3-train + 1-infer. ga=3 so eff_batch ≈ 36 (DP=3 × bs=4 × ga=3). 1-GPU vLLM, no TP.
  round7_gpu_split/split_3tr1inf) VARIANT_OVERRIDE="actor_train.device_mapping='[0,1,2]' actor_infer.device_mapping='[3]' actor_train.training_args.gradient_accumulation_steps=3" ;;
  # R8: per_device_train_batch_size at R7 winner. ga=1 throughout — eff_batch varies (24/36/48) but PPO is robust.
  round8_train_bs/bs8)            VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=8 actor_train.training_args.gradient_accumulation_steps=1" ;;
  round8_train_bs/bs12)           VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=12 actor_train.training_args.gradient_accumulation_steps=1" ;;
  round8_train_bs/bs16)           VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=16 actor_train.training_args.gradient_accumulation_steps=1" ;;
  # R9: rollout_batch_size at R7+R8 winners (3tr+1inf, bs=4 ga=3).
  round9_rollout_bs/rb64)         VARIANT_OVERRIDE="rollout_batch_size=64" ;;
  round9_rollout_bs/rb128)        VARIANT_OVERRIDE="rollout_batch_size=128" ;;
  round9_rollout_bs/rb256)        VARIANT_OVERRIDE="rollout_batch_size=256" ;;
  # R10: group_size re-validation at R7+R8+R9 winners.
  round10_group_size/gs1)         VARIANT_OVERRIDE="train_env_manager.group_size=1" ;;
  round10_group_size/gs4)         VARIANT_OVERRIDE="train_env_manager.group_size=4" ;;
  round10_group_size/gs8)         VARIANT_OVERRIDE="train_env_manager.group_size=8" ;;
  *) echo "ERROR: unknown round/variant ${ROUND}/${VARIANT}"; exit 2 ;;
esac

# Round-N-onwards: bake winners from prior rounds into the baseline by editing
# A40_WINNER_OVERRIDES above (a one-line sed in launch_round.sh).

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll2
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

EXP_NAME="kuhn_a6000_${ROUND}_${VARIANT}"
SWEEP_ROOT=/zfsauton/scratch/wentsec/kuhn_sweep_a6000/${ROUND}/${VARIANT}
mkdir -p ${SWEEP_ROOT}/logs ${SWEEP_ROOT}/render

trap "rm -rf ${SWEEP_ROOT}/render/*/checkpoint-* ${SWEEP_ROOT}/render/checkpoint-* ${SWEEP_ROOT}/actor_train-*/checkpoint-* >/dev/null 2>&1 || true" EXIT

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

source /zfsauton/scratch/wentsec/.env_roll

export CUDA_HOME=$ENV_PATH
export CUDA_TARGET_DIR=$ENV_PATH/targets/x86_64-linux
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_TARGET_DIR/include:$CPATH
export LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LIBRARY_PATH
export PYTHONPATH=$ROLL_DIR:$PYTHONPATH
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

ray stop --force 2>/dev/null || true
sleep 2

# Background nvidia-smi sampler (2 s cadence)
GPU_UTIL_CSV=${SWEEP_ROOT}/logs/gpu_util.csv
echo "timestamp,index,utilization_gpu_pct,memory_used_mib" > ${GPU_UTIL_CSV}
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 2 >> ${GPU_UTIL_CSV} &
NVSMI_PID=$!
trap "kill ${NVSMI_PID} >/dev/null 2>&1 || true; rm -rf $TMPDIR >/dev/null 2>&1 || true" EXIT

cd $ROLL_DIR
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_fsp_train \
    exp_name=${EXP_NAME} \
    output_dir=${SWEEP_ROOT} \
    logging_dir=${SWEEP_ROOT}/logs \
    checkpoint_config.output_dir=${SWEEP_ROOT}/render \
    track_with=wandb \
    tracker_kwargs.project=self-play-debug \
    tracker_kwargs.name=${EXP_NAME} \
    "${BENCH_OVERRIDES[@]}" \
    "${A40_WINNER_OVERRIDES[@]}" \
    ${VARIANT_OVERRIDE} \
    2>&1
PY_EXIT=$?

kill ${NVSMI_PID} 2>/dev/null || true
sleep 2

python ${ROLL_DIR}/scripts/auton/post_sweep_to_wandb.py \
    --exp_name "${EXP_NAME}" \
    --project self-play-debug \
    --gpu_util_csv ${GPU_UTIL_CSV} \
    --slurm_job_id ${SLURM_JOB_ID} \
    --round ${ROUND} \
    --variant ${VARIANT} || echo "WARN: post_sweep_to_wandb.py failed"

echo "===== SWEEP RUN DONE (round=${ROUND} variant=${VARIANT} exit=${PY_EXIT}) ====="
exit ${PY_EXIT}
