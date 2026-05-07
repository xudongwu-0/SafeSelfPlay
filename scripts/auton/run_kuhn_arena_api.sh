#!/bin/bash
# Usage: set ARENA_API_KEY, ARENA_BASE_URL, ARENA_MODEL before submitting.
#   export ARENA_API_KEY=<key>
#   export ARENA_BASE_URL=<url>
#   export ARENA_MODEL=<model>
#   sbatch scripts/auton/run_kuhn_arena_api.sh
#SBATCH --job-name=kuhn_arena_api
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_arena_api_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_arena_api_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -ex

if [ -z "${ARENA_API_KEY}" ] || [ -z "${ARENA_BASE_URL}" ] || [ -z "${ARENA_MODEL}" ]; then
    echo "ERROR: must set ARENA_API_KEY, ARENA_BASE_URL, ARENA_MODEL before submitting."
    exit 1
fi

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}_$(hostname -s)_$$"
EVAL_OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/eval_api/${RUN_ID}
mkdir -p $EVAL_OUTPUT_ROOT

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

source /zfsauton/scratch/wentsec/.env_roll

export PYTHONPATH=$ROLL_DIR:$PYTHONPATH
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
mkdir -p $TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec

cd $ROLL_DIR
python examples/start_arena_eval.py \
    --mode server_api \
    --api_key "${ARENA_API_KEY}" \
    --base_url "${ARENA_BASE_URL}" \
    --model_name "${ARENA_MODEL}" \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_arena_api \
    --self_play \
    --env_tag KuhnPokerLLMThink \
    --output_dir ${EVAL_OUTPUT_ROOT} \
    --episodes_per_pair 12 \
    --max_concurrent 4 \
    --save_trajectories \
    2>&1

rm -rf $TMPDIR
echo "===== KUHN ARENA API DONE (RUN_ID=${RUN_ID}, OUT=${EVAL_OUTPUT_ROOT}) ====="
