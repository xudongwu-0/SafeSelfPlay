#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=28:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=kuhn_poker_single_rl
#SBATCH --output=/projects/bfoz/wchen11/kuhn_poker_single_rl_%j.out

set -ex

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /projects/bfoz/wchen11/anaconda3/envs/roll2

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/tmp/pip_build_$$
mkdir -p $TMPDIR

cd /u/wchen11/ROLL
export PYTHONPATH=/u/wchen11/ROLL:$PYTHONPATH

ray stop --force 2>/dev/null || true
sleep 2

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_single_rl \
    2>&1
