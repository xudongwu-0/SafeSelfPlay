#!/bin/bash
# Sweep max_new_tokens for Qwen3.5-4B Kuhn Poker arena self-play to find <10% truncation threshold.
# Submits one sbatch job per value; trajectories land in $SWEEP_ROOT/mnt_${MNT}/.
set -e

ROLL_DIR=/zfsauton/scratch/wentsec/ROLL
SWEEP_TAG="thinking_budget_$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/eval_sweep/${SWEEP_TAG}
mkdir -p $SWEEP_ROOT

# Sweep values: each needs sequence_length >= MNT + prompt_pad (~512)
MNT_VALUES=(1024 2048 3072 4096 6144)

for MNT in "${MNT_VALUES[@]}"; do
    SEQ_LEN=$((MNT + 512))
    OUT_DIR=${SWEEP_ROOT}/mnt_${MNT}
    mkdir -p $OUT_DIR
    JOB_SCRIPT=${OUT_DIR}/launch.sh

    cat > $JOB_SCRIPT << JOBEOF
#!/bin/bash
#SBATCH --job-name=tbud_mnt${MNT}
#SBATCH --output=${OUT_DIR}/slurm_%j.out
#SBATCH --error=${OUT_DIR}/slurm_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a6000:1

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=${ROLL_DIR}

source \$CONDA_ROOT/etc/profile.d/conda.sh
conda activate \$ENV_PATH
source /zfsauton/scratch/wentsec/.env_roll

export CUDA_HOME=\$ENV_PATH
export CUDA_TARGET_DIR=\$ENV_PATH/targets/x86_64-linux
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_TARGET_DIR/lib:\$CUDA_HOME/lib:\$LD_LIBRARY_PATH
export CPATH=\$CUDA_TARGET_DIR/include:\$CPATH
export LIBRARY_PATH=\$CUDA_TARGET_DIR/lib:\$CUDA_HOME/lib:\$LIBRARY_PATH
export PYTHONPATH=\$ROLL_DIR:\$PYTHONPATH
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_\$\$
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
export ROLL_DISABLE_SLEEP_MODE=1
mkdir -p \$TMPDIR \$TRITON_CACHE_DIR \$RAY_TMPDIR

ray stop --force 2>/dev/null || true
sleep 2

cd \$ROLL_DIR
python examples/start_arena_eval.py \\
    --config_path agentic_demo \\
    --config_name agent_kuhn_poker_fsp_4b_kl0p01_cold_sync \\
    --self_play \\
    --output_dir ${OUT_DIR} \\
    --episodes_per_pair 32 \\
    --max_concurrent 16 \\
    --save_trajectories \\
    num_gpus_per_node=1 \\
    sequence_length=${SEQ_LEN} \\
    max_tokens_per_step=${MNT} \\
    actor_infer.generating_args.max_new_tokens=${MNT} \\
    actor_infer.generating_args.thinking_token_budget=$((MNT - 128)) \\
    2>&1

rm -rf \$TMPDIR
echo "===== MNT=${MNT} DONE (OUT=${OUT_DIR}) ====="
JOBEOF

    chmod +x $JOB_SCRIPT
    echo "Submitting mnt=${MNT} (seq_len=${SEQ_LEN})..."
    sbatch $JOB_SCRIPT
done

echo ""
echo "Sweep tag: ${SWEEP_TAG}"
echo "Results root: ${SWEEP_ROOT}"
echo "Monitor: squeue -u \$USER"
