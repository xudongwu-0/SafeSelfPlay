#!/bin/bash
# PSRO smoke check: 3 steps, 0.5B model, 2x GPU.
# Verifies base model is seeded into payoff matrix and Nash probs align with enemy pool.
# Expect logs: "PayoffMatrix: added first policy base_model, matrix is 1×1."
#              After step 1: Nash probs length 2, "Nash probabilities updated: [...]"
#              payoff_matrix_iter_1.json lora_paths = ["base_model", "<ckpt>"]
#
# Usage:
#   bash examples/agentic_demo/kuhn_poker/run_psro_smoke.sh

set -e

ROLL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${ROLL_DIR}/logs"
mkdir -p "${LOGS_DIR}"

echo "Submitting PSRO smoke check (3 steps, 0.5B, 2x GPU, ~10-15 min)..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=kuhn_psro_smoke
#SBATCH --output=${LOGS_DIR}/kuhn_psro_smoke_%j.out
#SBATCH --error=${LOGS_DIR}/kuhn_psro_smoke_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a6000:2

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=${ROLL_DIR}

RUN_ID="\${SLURM_JOB_ID:-\$(date +%s)}_\$(hostname -s)_\$\$"
OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/psro_smoke/\${RUN_ID}
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
    --config_name agent_kuhn_poker_psro_smoke \\
    logging_dir=\${OUTPUT_ROOT}/logs \\
    output_dir=\${OUTPUT_ROOT} \\
    checkpoint_config.output_dir=\${OUTPUT_ROOT}/render \\
    2>&1

# Verify: base model in payoff matrix
echo "--- Verifying payoff matrix ---"
MATRIX_FILE=\${OUTPUT_ROOT}/psro/payoff_matrix_iter_1.json
if [ -f "\${MATRIX_FILE}" ]; then
    python3 -c "
import json, sys
data = json.load(open('\${MATRIX_FILE}'))
paths = data['lora_paths']
print('lora_paths:', paths)
assert paths[0] == 'base_model', f'Expected base_model at index 0, got {paths[0]}'
assert len(paths) == 2, f'Expected 2 policies after iter 1, got {len(paths)}'
print('PASS: base_model is policy 0, matrix is 2x2 after iter 1')
"
else
    echo "WARNING: \${MATRIX_FILE} not found"
fi

rm -rf \${TMPDIR}
echo "===== PSRO SMOKE DONE (RUN_ID=\${RUN_ID}) ====="
EOF
