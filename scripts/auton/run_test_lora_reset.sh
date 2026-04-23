#!/bin/bash
#SBATCH --job-name=lora_reset_test
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/lora_reset_test_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/lora_reset_test_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll2
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
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
mkdir -p $TMPDIR $TRITON_CACHE_DIR

nvidia-smi

python $ROLL_DIR/scripts/auton/test_lora_reset.py

echo "===== LORA RESET TEST DONE ====="
