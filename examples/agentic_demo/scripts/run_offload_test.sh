#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=offload_test
#SBATCH --output=/projects/bfoz/wchen11/offload_test_%j.out

# Usage: sbatch run_offload_test.sh <batch_size> <strategy_config_key> <extra_label> [extra_overrides...]
# e.g.:  sbatch run_offload_test.sh 2 deepspeed_zero2 gc disable_gradient_checkpointing=false
BATCH_SIZE=${1:-2}
STRATEGY=${2:-deepspeed_zero2}
LABEL=${3:-default}
shift 3 || true
EXTRA_OVERRIDES="$*"
set -- # clear positional args so conda doesn't consume them

set -ex

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /projects/bfoz/wchen11/anaconda3/envs/roll2

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/tmp/pip_build_$$
mkdir -p $TMPDIR

cd /u/wchen11/ROLL
export PYTHONPATH=/u/wchen11/ROLL:$PYTHONPATH

GRAD_ACCUM=$((32 / BATCH_SIZE))
EXP="kuhn_bs${BATCH_SIZE}_${LABEL}"

echo "=== Offload test: bs=${BATCH_SIZE}, strategy=${STRATEGY}, label=${LABEL} ==="

ray stop --force 2>/dev/null || true
sleep 2

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_single_rl \
    max_steps=5 \
    eval_steps=100 \
    exp_name="${EXP}" \
    actor_train.training_args.per_device_train_batch_size=${BATCH_SIZE} \
    actor_train.training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
    "actor_train.strategy_args.strategy_config=\${${STRATEGY}}" \
    train_env_manager.num_env_groups=32 \
    train_env_manager.group_size=1 \
    train_env_manager.max_env_num_per_worker=32 \
    "train_env_manager.num_groups_partition=[32]" \
    val_env_manager.num_env_groups=4 \
    val_env_manager.group_size=1 \
    val_env_manager.max_env_num_per_worker=4 \
    "val_env_manager.num_groups_partition=[4]" \
    reward_normalization.grouping=tags \
    +reward_normalization.norm_mean_type=null \
    output_dir=/projects/bfoz/wchen11/kuhn_offload_test/${EXP} \
    logging_dir=/projects/bfoz/wchen11/kuhn_offload_test/${EXP}/logs \
    tracker_kwargs.project=kuhn-offload-test \
    checkpoint_config.output_dir=/projects/bfoz/wchen11/kuhn_offload_test/${EXP}/render \
    ${EXTRA_OVERRIDES} \
    2>&1
