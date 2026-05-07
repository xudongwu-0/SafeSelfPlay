#!/bin/bash
#SBATCH --job-name=kuhn_3b_ibs_sync
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_3b_ibs_sync_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_3b_ibs_sync_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a6000:4

set -ex

IBS="${1:?Usage: sbatch $0 <infer_batch_size>}"

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}_$(hostname -s)_$$"
FSP_OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/runs/kuhn_3b_ibs${IBS}_sync_${RUN_ID}
mkdir -p $FSP_OUTPUT_ROOT/logs $FSP_OUTPUT_ROOT/render

GPU_UTIL_LOG=$FSP_OUTPUT_ROOT/gpu_util.log
GPU_UTIL_PID=""

cleanup() {
    [[ -n "$GPU_UTIL_PID" ]] && kill "$GPU_UTIL_PID" 2>/dev/null || true
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
    --config_name agent_kuhn_poker_fsp_train \
    logging_dir=${FSP_OUTPUT_ROOT}/logs \
    output_dir=${FSP_OUTPUT_ROOT} \
    checkpoint_config.output_dir=${FSP_OUTPUT_ROOT}/render \
    max_steps=5 \
    eval_steps=1000 \
    save_steps=1000 \
    fsp_save_steps=1000 \
    async_generation_ratio=0 \
    exp_name=kuhn_3b_sync_ibs${IBS} \
    tracker_kwargs.project=self-play-debug \
    tracker_kwargs.tags="[kuhn_poker,fsp_train,qwen2_5_3b,cold_start,sync,auton,ibs_sweep,ibs${IBS}]" \
    actor_infer.infer_batch_size=${IBS} \
    2>&1

rm -rf $TMPDIR
echo "===== KUHN 3B SYNC IBS=${IBS} DONE (RUN_ID=${RUN_ID}) ====="
