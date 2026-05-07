#!/bin/bash
#SBATCH --job-name=kuhn_3b_2model_selfplay_ci
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_3b_2model_selfplay_ci_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_3b_2model_selfplay_ci_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=qos_preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a6000:1

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}_$(hostname -s)_$$"
EVAL_OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/eval/${RUN_ID}
mkdir -p $EVAL_OUTPUT_ROOT

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
export VLLM_USE_V1=0
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

ray stop --force 2>/dev/null || true
sleep 2

cd $ROLL_DIR
python examples/start_arena_eval.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_psro_3b_fixSeed_bubbleEval_rst300 \
    --self_play \
    --env_tag KuhnPokerLLMThink \
    --output_dir ${EVAL_OUTPUT_ROOT} \
    --episodes_per_pair 36 \
    --max_concurrent 4 \
    num_gpus_per_node=1 \
    actor_infer.model_args.dtype=fp16 \
    actor_infer.strategy_args.strategy_config.enforce_eager=true \
    actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.85 \
    2>&1

rm -rf $TMPDIR
echo "===== KUHN 3B 2MODEL SELFPLAY CI DONE (RUN_ID=${RUN_ID}, OUT=${EVAL_OUTPUT_ROOT}) ====="
