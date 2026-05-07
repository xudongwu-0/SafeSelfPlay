#!/bin/bash
#SBATCH --job-name=hold_vllm
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/hold_vllm_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/hold_vllm_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:1

set -e

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll2
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL
MODEL_PATH=/zfsauton/scratch/wentsec/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
PORT=8765

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

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Starting vLLM server on port $PORT ..."
echo "Connect via: $(hostname):$PORT"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --dtype bfloat16 \
    --port $PORT \
    --host 0.0.0.0 \
    --served-model-name qwen2.5-3b \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager false

rm -rf $TMPDIR
