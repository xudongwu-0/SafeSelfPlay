#!/bin/bash
# Launch vLLM inference parameter sweep: submit one SLURM job per config variant
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for SUFFIX in eager_true eager_false mnt64 mnt128 mnt256 seq512 seq1024 seq2048; do
    echo "Submitting: vllm_${SUFFIX}..."
    JOB_ID=$(sbatch --job-name="vllm_${SUFFIX}" --export=WANDB_API_KEY,HF_TOKEN "${SCRIPT_DIR}/run_vllm_sweep.sh" ${SUFFIX} | awk '{print $4}')
    echo "  -> job ${JOB_ID}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: /projects/bfoz/wchen11/kuhn_vllm_sweep/"
