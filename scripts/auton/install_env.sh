#!/bin/bash
#SBATCH --job-name=roll_install
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/install_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/install_%j.err
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a6000:1

set -ex

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll2
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

source $CONDA_ROOT/etc/profile.d/conda.sh

if [ ! -d "$ENV_PATH" ]; then
  conda create -y -p $ENV_PATH python=3.10
fi
conda activate $ENV_PATH

export TMPDIR=/zfsauton/scratch/wentsec/tmp_pip_$$
mkdir -p $TMPDIR

df -h /zfsauton/scratch /zfsauton2/home/wentsec

nvidia-smi
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')" 2>/dev/null || echo "torch not yet installed"

pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.is_available()}')"

cd $ROLL_DIR
pip install --no-cache-dir -r requirements_common.txt
pip install --no-cache-dir deepspeed==0.16.4
pip install --no-cache-dir vllm==0.10.2

CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || true)
if [ -n "$CUDNN_PATH" ]; then
  export CPATH="$CUDNN_PATH/include:$CPATH"
  export LIBRARY_PATH="$CUDNN_PATH/lib:$LIBRARY_PATH"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$LD_LIBRARY_PATH"
  echo "Using CUDNN_PATH=$CUDNN_PATH"
else
  echo "cudnn path not resolved; continuing without it"
fi

pip install --no-cache-dir flash-attn --no-build-isolation

pip install --no-cache-dir -e $ROLL_DIR || true

df -h /zfsauton/scratch /zfsauton2/home/wentsec

python -c "
import torch; print(f'torch={torch.__version__}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import vllm; print(f'vllm={vllm.__version__}')
import dacite; print('dacite OK')
try:
    import roll; print('roll OK')
except Exception as e: print(f'roll import: {e}')
try:
    import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except Exception as e: print(f'flash_attn: FAILED ({e})')
"

rm -rf $TMPDIR
echo '===== SETUP DONE ====='
