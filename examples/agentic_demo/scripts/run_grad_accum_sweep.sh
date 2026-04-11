#!/bin/bash
# Sweep gradient_accumulation_steps with optimal env parallelism (32 groups)
set -e

cd /u/wchen11/ROLL

SWEEP_DIR="/projects/bfoz/wchen11/kuhn_grad_accum_sweep"
mkdir -p $SWEEP_DIR

for GA in 1 2 4 8 16; do
    JOB_SCRIPT="${SWEEP_DIR}/bench_ga${GA}.sh"
    OUT_DIR="${SWEEP_DIR}/ga${GA}"
    mkdir -p $OUT_DIR

    cat > $JOB_SCRIPT << 'JOBEOF'
#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=4
JOBEOF
    echo "#SBATCH --job-name=kuhn_ga_${GA}" >> $JOB_SCRIPT
    echo "#SBATCH --output=${OUT_DIR}/bench_%j.out" >> $JOB_SCRIPT

    cat >> $JOB_SCRIPT << RUNEOF
set -ex

source /projects/bfoz/wchen11/anaconda3/etc/profile.d/conda.sh
conda activate roll2

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export TMPDIR=/tmp/kuhn_ga_${GA}_\$\$
mkdir -p \$TMPDIR
export PYTHONPATH=/u/wchen11/ROLL:\$PYTHONPATH
cd /u/wchen11/ROLL

ray stop --force 2>/dev/null || true
sleep 2

echo "=== BENCH: grad_accum=${GA}, num_env_groups=32 ==="

python -c "
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dacite import from_dict

initialize_config_dir(config_dir='/u/wchen11/ROLL/examples/agentic_demo', job_name='bench_ga${GA}')
cfg = compose(config_name='agent_kuhn_poker_bench')

OmegaConf.set_struct(cfg, False)
cfg.train_env_manager.num_env_groups = 32
cfg.train_env_manager.group_size = 1
cfg.train_env_manager.num_groups_partition = [32]
cfg.train_env_manager.max_env_num_per_worker = 64
cfg.rollout_batch_size = 64
cfg.actor_train.training_args.gradient_accumulation_steps = ${GA}
cfg.output_dir = '${OUT_DIR}'
cfg.logging_dir = '${OUT_DIR}/logs'
cfg.tracker_kwargs.log_dir = '${OUT_DIR}/tensorboard'
OmegaConf.set_struct(cfg, True)

from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.distributed.scheduler.initialize import init
from roll.utils.import_utils import safe_import_class

ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
init()

pipeline_cls = safe_import_class('roll.pipeline.agentic.agentic_pipeline.AgenticPipeline')
pipeline = pipeline_cls(pipeline_config=ppo_config)
pipeline.run()
"

echo "=== BENCH DONE: grad_accum=${GA} ==="
RUNEOF

    chmod +x $JOB_SCRIPT
    echo "Submitting: grad_accum=${GA}"
    sbatch $JOB_SCRIPT
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "Results will be in: ${SWEEP_DIR}/ga{1,2,4,8,16}/"
