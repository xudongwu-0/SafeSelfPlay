#!/bin/bash
#SBATCH --job-name=sample_lora
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/sample_lora_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/sample_lora_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a6000:1

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll2
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL
LORA=/zfsauton/scratch/wentsec/kuhn_poker_output/runs/6656_gpu30_73475/render/20260421-205952/checkpoint-50

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH
source /zfsauton/scratch/wentsec/.env_roll

export CUDA_HOME=$ENV_PATH
export CUDA_TARGET_DIR=$ENV_PATH/targets/x86_64-linux
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROLL_DIR:$PYTHONPATH
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
mkdir -p $TMPDIR

python $ROLL_DIR/scripts/auton/sample_lora_responses.py $LORA

rm -rf $TMPDIR
echo "===== SAMPLE DONE ====="
