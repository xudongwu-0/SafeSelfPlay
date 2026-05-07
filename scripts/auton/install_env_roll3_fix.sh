#!/bin/bash
#SBATCH --job-name=roll3_fix
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/roll3_fix_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/roll3_fix_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a6000:1

# Pin click to 8.1.x so ray's CLI deepcopy of click commands doesn't hit
# the "is not a valid Sentinel" enum bug introduced in click 8.2.

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

python -c "import click; print(f'click before={click.__version__}')"

pip install --no-cache-dir "click==8.1.8"

python -c "import click; print(f'click after={click.__version__}')"
ray --version
echo "===== ROLL3 FIX DONE ====="
