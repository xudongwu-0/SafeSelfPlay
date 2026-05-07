#!/bin/bash
#SBATCH --job-name=eval_format
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/eval_format_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/eval_format_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --nodelist=gpu27
# No GPU needed — just hitting the vLLM server over HTTP

set -e

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll2
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH
source /zfsauton/scratch/wentsec/.env_roll

export PYTHONPATH=$ROLL_DIR:$PYTHONPATH

python $ROLL_DIR/scripts/auton/eval_action_format.py \
    --host gpu27 \
    --port 8765 \
    --n 50

echo "===== EVAL DONE ====="
