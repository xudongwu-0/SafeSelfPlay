#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=fsp_rps
#SBATCH --output=output/logs/fsp_rps_%j.out

source /projects/bfoz/wchen11/anaconda3/etc/profile.d/conda.sh
conda activate roll2
cd /u/wchen11/ROLL
export PYTHONPATH=/u/wchen11/ROLL:$PYTHONPATH

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_rps_fsp_train \
    2>&1
