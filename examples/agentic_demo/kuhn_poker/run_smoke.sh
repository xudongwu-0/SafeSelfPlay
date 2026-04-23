#!/bin/bash
# Smoke check: 3 training steps end-to-end on 4x A6000.
# Validates env/rollout/train pipeline without waiting for a full run.
#
# Usage:
#   bash examples/agentic_demo/kuhn_poker/run_smoke.sh

set -e

ROLL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${ROLL_DIR}/logs"
mkdir -p "${LOGS_DIR}"

echo "Submitting smoke check (3 steps, 4x A6000, ~10-15 min)..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=kuhn_smoke
#SBATCH --output=${LOGS_DIR}/kuhn_smoke_%j.out
#SBATCH --error=${LOGS_DIR}/kuhn_smoke_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a6000:4

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=${ROLL_DIR}

RUN_ID="\${SLURM_JOB_ID:-\$(date +%s)}_\$(hostname -s)_\$\$"
OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/runs/\${RUN_ID}
mkdir -p \${OUTPUT_ROOT}/logs \${OUTPUT_ROOT}/render

trap "rm -rf \${OUTPUT_ROOT}/render/*/checkpoint-* \${OUTPUT_ROOT}/render/checkpoint-* \${OUTPUT_ROOT}/actor_train-*/checkpoint-* 2>/dev/null || true" EXIT

source \${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate \${ENV_PATH}

source /zfsauton/scratch/wentsec/.env_roll

export CUDA_HOME=\${ENV_PATH}
export CUDA_TARGET_DIR=\${ENV_PATH}/targets/x86_64-linux
export PATH=\${CUDA_HOME}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CUDA_TARGET_DIR}/lib:\${CUDA_HOME}/lib:\${LD_LIBRARY_PATH}
export CPATH=\${CUDA_TARGET_DIR}/include:\${CPATH}
export LIBRARY_PATH=\${CUDA_TARGET_DIR}/lib:\${CUDA_HOME}/lib:\${LIBRARY_PATH}
export PYTHONPATH=\${ROLL_DIR}:\${PYTHONPATH}
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_\$\$
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
mkdir -p \${TMPDIR} \${TRITON_CACHE_DIR} \${RAY_TMPDIR}

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

ray stop --force 2>/dev/null || true
sleep 2

cd \${ROLL_DIR}
python examples/start_agentic_pipeline.py \\
    --config_path agentic_demo \\
    --config_name kuhn_poker/smoke \\
    logging_dir=\${OUTPUT_ROOT}/logs \\
    output_dir=\${OUTPUT_ROOT} \\
    checkpoint_config.output_dir=\${OUTPUT_ROOT}/render \\
    2>&1

rm -rf \${TMPDIR}
echo "===== SMOKE DONE (RUN_ID=\${RUN_ID}) ====="
EOF
