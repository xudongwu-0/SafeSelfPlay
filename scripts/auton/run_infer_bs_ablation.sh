#!/bin/bash
#SBATCH --job-name=kuhn_inferbs
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_inferbs_%x_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_inferbs_%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a6000:4

set -ex

INFER_BS=${1:?"Usage: sbatch --job-name=kuhn_inferbs_<bs> $0 <infer_batch_size>"}

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}_$(hostname -s)_$$"
FSP_OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/runs/inferbs_${INFER_BS}_${RUN_ID}
mkdir -p $FSP_OUTPUT_ROOT/logs $FSP_OUTPUT_ROOT/render

GPU_UTIL_LOG=$FSP_OUTPUT_ROOT/gpu_util.log
GPU_UTIL_PID=""

cleanup() {
    if [[ -n "$GPU_UTIL_PID" ]]; then
        kill "$GPU_UTIL_PID" 2>/dev/null || true
    fi
    rm -rf ${FSP_OUTPUT_ROOT}/render/*/checkpoint-* ${FSP_OUTPUT_ROOT}/render/checkpoint-* ${FSP_OUTPUT_ROOT}/actor_train-*/checkpoint-* 2>/dev/null || true
}
trap cleanup EXIT

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

source /zfsauton/scratch/wentsec/.env_roll

export CUDA_HOME=$ENV_PATH
export CUDA_TARGET_DIR=$ENV_PATH/targets/x86_64-linux
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_TARGET_DIR/include:$CPATH
export LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LIBRARY_PATH
export PYTHONPATH=$ROLL_DIR:$PYTHONPATH
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

ray stop --force 2>/dev/null || true
sleep 2

nvidia-smi dmon -s u -d 1 > "$GPU_UTIL_LOG" &
GPU_UTIL_PID=$!
echo "GPU util sampler PID=$GPU_UTIL_PID, log=$GPU_UTIL_LOG"

cd $ROLL_DIR
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_fsp_4b_kl0p01_cold_sync_gpuutil_baseline \
    logging_dir=${FSP_OUTPUT_ROOT}/logs \
    output_dir=${FSP_OUTPUT_ROOT} \
    checkpoint_config.output_dir=${FSP_OUTPUT_ROOT}/render \
    exp_name=kuhn_inferbs_${INFER_BS} \
    actor_train.infer_batch_size=${INFER_BS} \
    tracker_kwargs.tags="[kuhn_poker,fsp_train,qwen3_5_4b,cold_start,sync,auton,inferbs_ab,bs${INFER_BS}]" \
    2>&1

rm -rf $TMPDIR
echo "===== KUHN INFER-BS ${INFER_BS} DONE (RUN_ID=${RUN_ID}) ====="
