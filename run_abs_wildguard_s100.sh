#!/usr/bin/env bash
set -euo pipefail

cd /home/xudong/work/self_play/ROLL

export PYTHONUNBUFFERED=1
export ABS_TRAIN_GPU=A100-40GB:4
export ABS_TRAIN_MICRO_BATCH=2
export ABS_GRAD_ACCUM=16
export ABS_TRAIN_INFER_BATCH=2
export ABS_RM_GPU=A10G
export ABS_RM_MAX_CONTAINERS=4
export ABS_EVAL_GPU=A10G:2

modal run modal_abs_benchmark.py \
  --mode all-wildguard \
  --max-steps 100 \
  --limit-eval \
  --tasks "wildjailbreak:harmful,do_anything_now,harmbench,xstest,strongreject:wildguard" \
  --local-output-dir /home/xudong/work/self_play/checkpoints/roll_abs_benchmark_wildguard_s100
