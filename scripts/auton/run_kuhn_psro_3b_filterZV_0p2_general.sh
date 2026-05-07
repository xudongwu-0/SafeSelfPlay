#!/bin/bash
#SBATCH --job-name=kuhn_psro_3b_filterZV_0p2_general
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_psro_3b_filterZV_0p2_general_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn_psro_3b_filterZV_0p2_general_%j.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --exclude=gpu24,gpu26

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}_$(hostname -s)_$$"
FSP_OUTPUT_ROOT=/zfsauton/scratch/wentsec/kuhn_poker_output/runs/${RUN_ID}
mkdir -p $FSP_OUTPUT_ROOT/logs $FSP_OUTPUT_ROOT/render

trap "rm -rf ${FSP_OUTPUT_ROOT}/render/*/checkpoint-* ${FSP_OUTPUT_ROOT}/render/checkpoint-* ${FSP_OUTPUT_ROOT}/actor_train-*/checkpoint-* 2>/dev/null || true" EXIT

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
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

ray stop --force 2>/dev/null || true
sleep 2

cd $ROLL_DIR
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_psro_3b_fixSeed_bubbleEval_rst300 \
    logging_dir=${FSP_OUTPUT_ROOT}/logs \
    output_dir=${FSP_OUTPUT_ROOT} \
    checkpoint_config.output_dir=${FSP_OUTPUT_ROOT}/render \
    exp_name=kuhn_psro_3b_0p2 \
    filter_zero_variance_groups=true \
    fsp_score_threshold=0.2 \
    2>&1

rm -rf $TMPDIR
echo "===== kuhn_psro_3b_filterZV_0p2_general DONE (RUN_ID=${RUN_ID}) ====="
