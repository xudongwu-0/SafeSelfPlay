#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=sokoban_5configs
#SBATCH --output=sokoban_5configs_%j.log

set -x

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /projects/bfoz/wchen11/anaconda3/envs/roll2

cd /u/wchen11/ROLL

CONFIGS=(
  agentic_val_sokoban
  agentic_val_sokoban_gigpo
  agentic_val_sokoban_lora
  agentic_val_sokoban_native
  agentic_val_sokoban_sandbox
)

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  echo "===== [$((i+1))/${#CONFIGS[@]}] $cfg ====="
  python /u/wchen11/ROLL/examples/run_sokoban_wrapper.py \
    --config_path qwen2.5-0.5B-agentic \
    --config_name "$cfg" \
    --max_steps 1 \
    --num_gpus 4 \
    --use_deepspeed \
    --smoke_test 2>&1
  echo "===== $cfg DONE ====="
done

echo "===== ALL DONE ====="
