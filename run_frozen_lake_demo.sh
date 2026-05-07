#!/bin/bash
set +x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate roll

export PYTHONPATH="/u/wchen11/ROLL:$PYTHONPATH"
export USE_MODELSCOPE=1

cd /u/wchen11/ROLL

CONFIG_PATH=agentic_demo

python examples/start_agentic_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name agent_val_frozen_lake_single_node_demo
