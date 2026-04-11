#!/bin/bash
# Launch batch size sweep: submit one SLURM job per batch_size
set -e

for BS in 1 2 4 8; do
    echo "Submitting batch_size=${BS}..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    JOB_ID=$(sbatch --job-name="bs${BS}" --export=WANDB_API_KEY,HF_TOKEN "${SCRIPT_DIR}/run_bs_sweep.sh" ${BS} | awk '{print $4}')
    echo "  -> job ${JOB_ID}"
done

echo ""
echo "Monitor: squeue -u \$USER"
