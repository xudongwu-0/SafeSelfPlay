#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=setup_roll2
#SBATCH --output=setup_roll2_%j.log

set -ex

source /projects/bfoz/wchen11/anaconda3/bin/activate

# Create env under /projects (plenty of quota)
ENV_PATH=/projects/bfoz/wchen11/anaconda3/envs/roll2
if [ ! -d "$ENV_PATH" ]; then
  conda create -y -p $ENV_PATH python=3.10
fi
conda activate $ENV_PATH

export TMPDIR=/tmp/pip_build_$$
mkdir -p $TMPDIR

cd /u/wchen11/ROLL

nvidia-smi
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')" 2>/dev/null || echo "torch not yet installed"

# Install torch 2.8.0 for CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.is_available()}')"

# Install common deps
pip install -r requirements_common.txt

# Install deepspeed + vllm
pip install deepspeed==0.16.4
pip install vllm==0.10.2

# Install flash-attn
pip install flash-attn --no-build-isolation --no-cache-dir

# Try transformer-engine (optional)
CUDNN_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")
export CPATH="$CUDNN_PATH/include:$CPATH"
export LIBRARY_PATH="$CUDNN_PATH/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$LD_LIBRARY_PATH"
pip install "transformer-engine[pytorch]==2.2.0" --no-build-isolation --no-cache-dir || echo "TE install failed (optional)"

# Verify
python -c "
import torch; print(f'torch={torch.__version__}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import vllm; print(f'vllm={vllm.__version__}')
import dacite; print('dacite OK')
import roll; print('roll OK')
try:
    import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except Exception as e: print(f'flash_attn: FAILED ({e})')
try:
    import transformer_engine; print(f'TE={transformer_engine.__version__}')
except: print('TE: FAILED (optional)')
"

echo "===== SETUP DONE ====="
