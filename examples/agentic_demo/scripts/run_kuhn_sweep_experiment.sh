#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4

set -e

# Save positional args before sourcing (activate script uses $@)
HYDRA_OVERRIDES=("$@")
set --

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /projects/bfoz/wchen11/anaconda3/envs/roll2

# Restore positional args
set -- "${HYDRA_OVERRIDES[@]}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before launching the sweep}"
: "${HF_TOKEN:?Set HF_TOKEN before launching the sweep}"
export WANDB_API_KEY HF_TOKEN
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
