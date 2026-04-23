#!/bin/bash
#SBATCH --job-name=roll3_install
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/roll3_install_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/roll3_install_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a6000:1

# roll3 env for Qwen3.5-4B (needs transformers v5.3+, vllm v0.17+).
# Keeps roll2 intact. Unpins trl/peft/accelerate/datasets for v5 compat.

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

source $CONDA_ROOT/etc/profile.d/conda.sh

if [ ! -d "$ENV_PATH" ]; then
  conda create -y -p $ENV_PATH python=3.10
fi
conda activate $ENV_PATH

# cuda-nvcc 12.8 from conda is required because the system /usr/local/cuda is
# 13 and flash-attn's build asserts torch's CUDA (12.8) matches nvcc's.
conda install -y -p $ENV_PATH -c nvidia/label/cuda-12.8.0 \
  cuda-nvcc cuda-nvvm cuda-cudart-dev cuda-nvrtc-dev cuda-cccl_linux-64 \
  libcublas-dev libcusolver-dev libcusparse-dev libcurand-dev libcufft-dev \
  cuda-profiler-api cuda-driver-dev

export CUDA_HOME=$ENV_PATH
export CUDA_TARGET_DIR=$ENV_PATH/targets/x86_64-linux
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_TARGET_DIR/include:$CPATH
export LIBRARY_PATH=$CUDA_TARGET_DIR/lib:$CUDA_HOME/lib:$LIBRARY_PATH
nvcc --version || true

export TMPDIR=/zfsauton/scratch/wentsec/tmp_pip_$$
mkdir -p $TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.is_available()}')"

# Install requirements but skip the hard-pinned versions that conflict with transformers v5.
# Write the filtered file into ROLL_DIR so the relative `-r requirements_vision.txt` still resolves.
grep -vE '^(trl==|peft==|accelerate==|datasets==|numpy<)' $ROLL_DIR/requirements_common.txt > $ROLL_DIR/requirements_roll3.txt
pip install --no-cache-dir -r $ROLL_DIR/requirements_roll3.txt
rm -f $ROLL_DIR/requirements_roll3.txt

# transformers v5.3+ is the minimum that registers qwen3_5.
# vllm 0.17.1 is the first tag with Qwen3_5ForConditionalGeneration.
# Pull newer peft / trl / accelerate / datasets compatible with transformers v5.
pip install --no-cache-dir "transformers>=5.3,<5.6"
pip install --no-cache-dir "vllm>=0.17,<0.20"
pip install --no-cache-dir "peft>=0.15" "trl>=0.16" "accelerate>=1.0" "datasets>=3.2" "numpy>=1.25,<2.0" "click==8.1.8"
pip install --no-cache-dir deepspeed==0.16.4

CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || true)
if [ -n "$CUDNN_PATH" ]; then
  export CPATH="$CUDNN_PATH/include:$CPATH"
  export LIBRARY_PATH="$CUDNN_PATH/lib:$LIBRARY_PATH"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$LD_LIBRARY_PATH"
fi
export MAX_JOBS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE
pip install --no-cache-dir flash-attn --no-build-isolation

pip install --no-cache-dir -e $ROLL_DIR || true

df -h /zfsauton/scratch /zfsauton2/home/wentsec

python -c "
import torch; print(f'torch={torch.__version__}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import vllm; print(f'vllm={vllm.__version__}')
import transformers; print(f'transformers={transformers.__version__}')
import peft; print(f'peft={peft.__version__}')
import trl; print(f'trl={trl.__version__}')
import accelerate; print(f'accelerate={accelerate.__version__}')
try:
    import roll; print('roll import OK')
except Exception as e: print(f'roll import FAILED: {e}')
try:
    import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except Exception as e: print(f'flash_attn FAILED: {e}')
from transformers import AutoConfig
try:
    c = AutoConfig.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=False)
    print(f'Qwen3.5-4B config.model_type={c.model_type}')
except Exception as e: print(f'Qwen3.5-4B config load FAILED: {e}')
"

rm -rf $TMPDIR
echo '===== ROLL3 SETUP DONE ====='
