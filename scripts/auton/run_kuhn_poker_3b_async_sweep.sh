#!/bin/bash
# Qwen2.5-3B async sweep on 4× A6000 (debug partition).
# Baseline: v2 best config (3tr+1inf, bs=4 ga=3, rb=256, gs=4, enforce_eager=false, mnt=4096).
# Sweeps async_generation_ratio and rollout_batch_size.
# Usage: bash scripts/auton/run_kuhn_poker_3b_async_sweep.sh
set -e

ROLL_DIR=/zfsauton/scratch/wentsec/ROLL
SWEEP_TAG="3b_async_$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT=/zfsauton/scratch/wentsec/kuhn_sweep_3b_async/${SWEEP_TAG}
mkdir -p ${SWEEP_ROOT}/logs

# v2 best config baseline overrides
BASELINE=(
    "actor_train.device_mapping='[0,1,2]'"
    "actor_infer.device_mapping='[3]'"
    "actor_train.training_args.per_device_train_batch_size=4"
    "actor_train.training_args.gradient_accumulation_steps=3"
    "actor_train.infer_batch_size=4"
    "actor_infer.strategy_args.strategy_config.enforce_eager=false"
    "actor_infer.strategy_args.strategy_config.max_num_batched_tokens=4096"
    "train_env_manager.num_env_groups=64"
    "train_env_manager.group_size=4"
    "sequence_length=1024"
    "actor_infer.generating_args.max_new_tokens=512"
    "rollout_batch_size=256"
)

BENCH=(
    "max_steps=5"
    "eval_steps=100"
    "save_steps=10000"
    "fsp_save_steps=0"
    "logging_steps=1"
)

VARIANTS=(
    "async0  async_generation_ratio=0"
    "async1  async_generation_ratio=1"
    "async2  async_generation_ratio=2"
    "async1_rb128  async_generation_ratio=1 rollout_batch_size=128 train_env_manager.num_env_groups=32"
    "async1_rb512  async_generation_ratio=1 rollout_batch_size=512 train_env_manager.num_env_groups=128"
)

echo "Sweep tag: ${SWEEP_TAG}"
echo "Sweep root: ${SWEEP_ROOT}"

for ENTRY in "${VARIANTS[@]}"; do
    VNAME=$(echo "$ENTRY" | awk '{print $1}')
    VOVERRIDES=$(echo "$ENTRY" | cut -d' ' -f2-)

    OUT_DIR=${SWEEP_ROOT}/${VNAME}
    mkdir -p ${OUT_DIR}/logs ${OUT_DIR}/render
    JOB_SCRIPT=${OUT_DIR}/launch.sh

    cat > $JOB_SCRIPT << JOBEOF
#!/bin/bash
#SBATCH --job-name=3b_${VNAME}
#SBATCH --output=${OUT_DIR}/slurm_%j.out
#SBATCH --error=${OUT_DIR}/slurm_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a6000:4

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3

source \$CONDA_ROOT/etc/profile.d/conda.sh
conda activate \$ENV_PATH
source /zfsauton/scratch/wentsec/.env_roll

export CUDA_HOME=\$ENV_PATH
export CUDA_TARGET_DIR=\$ENV_PATH/targets/x86_64-linux
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_TARGET_DIR/lib:\$CUDA_HOME/lib:\$LD_LIBRARY_PATH
export CPATH=\$CUDA_TARGET_DIR/include:\$CPATH
export LIBRARY_PATH=\$CUDA_TARGET_DIR/lib:\$CUDA_HOME/lib:\$LIBRARY_PATH
export PYTHONPATH=${ROLL_DIR}:\$PYTHONPATH
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_\$\$
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp_\${SLURM_JOB_ID}
export MULTI_TENANT=1
PORT_OFFSET=\$((SLURM_JOB_ID % 1000))
export MASTER_PORT=\$((6379 + PORT_OFFSET))
export DASHBOARD_PORT=\$((8265 + PORT_OFFSET))
export ROLL_DISABLE_SLEEP_MODE=1
mkdir -p \$TMPDIR \$TRITON_CACHE_DIR \$RAY_TMPDIR

# Background GPU util sampler
GPU_CSV=${OUT_DIR}/logs/gpu_util.csv
echo "timestamp,index,utilization_gpu_pct,memory_used_mib" > \$GPU_CSV
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 2 >> \$GPU_CSV &
NVSMI_PID=\$!
trap "kill \$NVSMI_PID 2>/dev/null || true; rm -rf \$TMPDIR \$RAY_TMPDIR 2>/dev/null || true" EXIT

cd ${ROLL_DIR}
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_qwen25_3b_sync \
    exp_name="3b_${VNAME}" \
    output_dir=${OUT_DIR} \
    logging_dir=${OUT_DIR}/logs \
    checkpoint_config.output_dir=${OUT_DIR}/render \
    track_with=wandb \
    tracker_kwargs.project=self-play-debug \
    tracker_kwargs.name="3b_${VNAME}" \
    ${BENCH[@]} \
    ${BASELINE[@]} \
    ${VOVERRIDES} \
    2>&1

echo "===== ${VNAME} DONE ====="
JOBEOF

    chmod +x $JOB_SCRIPT
    echo "Submitting ${VNAME}..."
    sbatch $JOB_SCRIPT
    sleep 2  # avoid raylet port collision on same node
done

echo ""
echo "All variants submitted. Monitor: squeue -u \$USER"
echo "Results root: ${SWEEP_ROOT}"
