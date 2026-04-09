#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=grpo_sweep
#SBATCH --output=/projects/bfoz/wchen11/grpo_sweep_%j.out

# Capture arg before sourcing conda (which consumes $1)
GROUP_SIZE=${1:-1}
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

echo "=== GRPO sweep: group_size=${GROUP_SIZE} ==="

ray stop --force 2>/dev/null || true
sleep 2

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_grpo_gs${GROUP_SIZE} \
    2>&1
