#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bs_sweep
#SBATCH --output=/projects/bfoz/wchen11/bs_sweep_%j.out

BATCH_SIZE=${1:-1}
shift || true

set -ex

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /projects/bfoz/wchen11/anaconda3/envs/roll2

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/tmp/pip_build_$$
mkdir -p $TMPDIR

cd /u/wchen11/ROLL
export PYTHONPATH=/u/wchen11/ROLL:$PYTHONPATH

echo "=== BS sweep: per_device_train_batch_size=${BATCH_SIZE} ==="

ray stop --force 2>/dev/null || true
sleep 2

GRAD_ACCUM=$((32 / BATCH_SIZE))

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_single_rl \
    max_steps=20 \
    eval_steps=100 \
    exp_name="kuhn_bs${BATCH_SIZE}" \
    actor_train.training_args.per_device_train_batch_size=${BATCH_SIZE} \
    actor_train.training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
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
    output_dir=/projects/bfoz/wchen11/kuhn_bs_sweep/bs${BATCH_SIZE} \
    logging_dir=/projects/bfoz/wchen11/kuhn_bs_sweep/bs${BATCH_SIZE}/logs \
    tracker_kwargs.project=kuhn-bs-sweep \
    checkpoint_config.output_dir=/projects/bfoz/wchen11/kuhn_bs_sweep/bs${BATCH_SIZE}/render \
    2>&1
