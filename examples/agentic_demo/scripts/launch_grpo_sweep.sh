#!/bin/bash
# Launch GRPO group_size sweep: submit one SLURM job per group_size
set -e

for GS in 1 2 4 8; do
    echo "Submitting group_size=${GS}..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    JOB_ID=$(sbatch --job-name="grpo_gs${GS}" --export=WANDB_API_KEY,HF_TOKEN "${SCRIPT_DIR}/run_grpo_sweep.sh" ${GS} | awk '{print $4}')
    echo "  -> job ${JOB_ID}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Compare: tensorboard --logdir /projects/bfoz/wchen11/kuhn_grpo_sweep"
