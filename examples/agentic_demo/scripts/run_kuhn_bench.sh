#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=kuhn_bench
#SBATCH --output=/projects/bfoz/wchen11/kuhn_bench_%j.out

set -ex

source /projects/bfoz/wchen11/anaconda3/etc/profile.d/conda.sh
conda activate roll2

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/tmp/kuhn_bench_$$
mkdir -p $TMPDIR

cd /u/wchen11/ROLL
export PYTHONPATH=/u/wchen11/ROLL:$PYTHONPATH

ray stop --force 2>/dev/null || true
sleep 2

# Args passed via environment
NUM_ENV_GROUPS=${NUM_ENV_GROUPS:-16}
GROUP_SIZE=${GROUP_SIZE:-1}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-64}
GRAD_ACCUM=${GRAD_ACCUM:-16}

echo "=== BENCH: num_env_groups=$NUM_ENV_GROUPS group_size=$GROUP_SIZE batch=$ROLLOUT_BATCH grad_accum=$GRAD_ACCUM ==="

python -c "
import time
from hydra import compose, initialize
from omegaconf import OmegaConf
from dacite import from_dict

initialize(config_path='examples/agentic_demo', job_name='bench')
cfg = compose(config_name='agent_kuhn_poker_bench')

OmegaConf.set_struct(cfg, False)
cfg.train_env_manager.num_env_groups = $NUM_ENV_GROUPS
cfg.train_env_manager.group_size = $GROUP_SIZE
cfg.train_env_manager.num_groups_partition = [$NUM_ENV_GROUPS]
cfg.train_env_manager.max_env_num_per_worker = max($NUM_ENV_GROUPS * $GROUP_SIZE, 16)
cfg.rollout_batch_size = $ROLLOUT_BATCH
cfg.actor_train.training_args.gradient_accumulation_steps = $GRAD_ACCUM
OmegaConf.set_struct(cfg, True)

from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.distributed.scheduler.initialize import init
from roll.utils.import_utils import safe_import_class

ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
init()

pipeline_cls = safe_import_class('roll.pipeline.agentic.agentic_pipeline.AgenticPipeline')
pipeline = pipeline_cls(pipeline_config=ppo_config)
pipeline.run()
" 2>&1

echo "=== BENCH DONE ==="
