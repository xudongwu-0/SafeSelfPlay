#!/bin/bash
#SBATCH --job-name=kuhn_selfplay_3b
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_selfplay_3b_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_selfplay_3b_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a6000:2

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

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
export ROLL_DISABLE_SLEEP_MODE=1
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

ray stop --force 2>/dev/null || true
sleep 2

mkdir -p /zfsauton/scratch/wentsec/ROLL/logs
OUTPUT_DIR=/zfsauton/scratch/wentsec/kuhn_poker_output/selfplay_3b_${SLURM_JOB_ID}
mkdir -p $OUTPUT_DIR

cd $ROLL_DIR

python examples/start_arena_eval.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_debug \
    --self_play \
    --episodes_per_pair 144 \
    --max_concurrent 8 \
    --output_dir $OUTPUT_DIR \
    --save_trajectories \
    pretrain=Qwen/Qwen2.5-3B-Instruct \
    reward_pretrain=Qwen/Qwen2.5-3B-Instruct
