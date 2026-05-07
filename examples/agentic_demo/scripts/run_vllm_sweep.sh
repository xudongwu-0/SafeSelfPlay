#!/bin/bash
# Sweep vLLM inference params: enforce_eager, max_new_tokens, sequence_length
set -e

cd /u/wchen11/ROLL

SWEEP_DIR="/projects/bfoz/wchen11/kuhn_vllm_sweep"
mkdir -p $SWEEP_DIR

# Define all sweep variants: name, python override code
declare -A VARIANTS
VARIANTS[eager_true]='cfg.actor_infer.strategy_args.strategy_config.enforce_eager = True'
VARIANTS[eager_false]='cfg.actor_infer.strategy_args.strategy_config.enforce_eager = False'
VARIANTS[mnt64]='cfg.actor_infer.generating_args.max_new_tokens = 64; cfg.max_tokens_per_step = 64'
VARIANTS[mnt128]='cfg.actor_infer.generating_args.max_new_tokens = 128; cfg.max_tokens_per_step = 128'
VARIANTS[mnt256]='cfg.actor_infer.generating_args.max_new_tokens = 256; cfg.max_tokens_per_step = 256'
VARIANTS[seq512]='cfg.sequence_length = 512'
VARIANTS[seq1024]='cfg.sequence_length = 1024'
VARIANTS[seq2048]='cfg.sequence_length = 2048'

for NAME in eager_true eager_false mnt64 mnt128 mnt256 seq512 seq1024 seq2048; do
    OVERRIDE="${VARIANTS[$NAME]}"
    OUT_DIR="${SWEEP_DIR}/${NAME}"
    mkdir -p $OUT_DIR

    JOB_SCRIPT="${SWEEP_DIR}/bench_${NAME}.sh"

    cat > $JOB_SCRIPT << 'JOBEOF'
#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=4
JOBEOF
    echo "#SBATCH --job-name=vllm_${NAME}" >> $JOB_SCRIPT
    echo "#SBATCH --output=${OUT_DIR}/bench_%j.out" >> $JOB_SCRIPT

    cat >> $JOB_SCRIPT << RUNEOF
set -ex

source /projects/bfoz/wchen11/anaconda3/etc/profile.d/conda.sh
conda activate roll2

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/tmp/kuhn_vllm_${NAME}_\$\$
mkdir -p \$TMPDIR
export PYTHONPATH=/u/wchen11/ROLL:\$PYTHONPATH
cd /u/wchen11/ROLL

ray stop --force 2>/dev/null || true
sleep 2

echo "=== BENCH: vllm_${NAME} ==="

python -c "
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dacite import from_dict

initialize_config_dir(config_dir='/u/wchen11/ROLL/examples/agentic_demo', job_name='bench_vllm_${NAME}')
cfg = compose(config_name='agent_kuhn_poker_single_rl')

OmegaConf.set_struct(cfg, False)
# Common sweep settings
cfg.max_steps = 5
cfg.eval_steps = 100
cfg.train_env_manager.num_env_groups = 32
cfg.train_env_manager.group_size = 1
cfg.train_env_manager.num_groups_partition = [32]
cfg.train_env_manager.max_env_num_per_worker = 32
cfg.exp_name = 'kuhn_vllm_${NAME}'
cfg.output_dir = '${OUT_DIR}'
cfg.logging_dir = '${OUT_DIR}/logs'
cfg.tracker_kwargs.project = 'kuhn-vllm-sweep'

# Variant-specific override
${OVERRIDE}

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

echo "=== BENCH DONE: vllm_${NAME} ==="
RUNEOF

    chmod +x $JOB_SCRIPT
    echo "Submitting: vllm_${NAME}"
    sbatch $JOB_SCRIPT
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "Results will be in: ${SWEEP_DIR}/"
