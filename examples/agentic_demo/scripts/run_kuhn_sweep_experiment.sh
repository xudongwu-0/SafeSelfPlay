#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4

set -ex

# Save positional args before sourcing (activate script uses $@)
HYDRA_OVERRIDES=("$@")
set --

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /projects/bfoz/wchen11/anaconda3/envs/roll2

# Restore positional args
set -- "${HYDRA_OVERRIDES[@]}"

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export HF_TOKEN="${HF_TOKEN:-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}"
export TMPDIR=/tmp/pip_build_$$
mkdir -p $TMPDIR

cd /u/wchen11/ROLL
export PYTHONPATH=/u/wchen11/ROLL:$PYTHONPATH

ray stop --force 2>/dev/null || true
sleep 2

# Pass all arguments as Hydra overrides
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_single_rl \
    "$@" \
    2>&1
