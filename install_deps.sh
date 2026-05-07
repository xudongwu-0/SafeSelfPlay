#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --job-name=install_deps
#SBATCH --output=install_deps_%j.log

set -x

source /projects/bfoz/wchen11/anaconda3/bin/activate
conda activate /u/wchen11/anaconda3/envs/roll

nvidia-smi

# Avoid cross-device link errors on NFS
export TMPDIR=/tmp/pip_build_$$
mkdir -p $TMPDIR

# Point compiler to cudnn headers/libs
CUDNN_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")
export CPATH="$CUDNN_PATH/include:$CPATH"
export LIBRARY_PATH="$CUDNN_PATH/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$LD_LIBRARY_PATH"

# Reinstall flash-attn (ABI mismatch with current torch)
pip install flash-attn --force-reinstall --no-build-isolation --no-cache-dir
echo "flash-attn install exit code: $?"

# Install transformer-engine
pip install "transformer-engine[pytorch]==2.2.0" --no-build-isolation --no-cache-dir
echo "transformer-engine install exit code: $?"

# Verify
python -c "import flash_attn; print(f'flash_attn={flash_attn.__version__}')" || echo "flash_attn import FAILED"
python -c "import transformer_engine; print(f'TE={transformer_engine.__version__}')" || echo "TE import FAILED"

echo "===== INSTALL DONE ====="
