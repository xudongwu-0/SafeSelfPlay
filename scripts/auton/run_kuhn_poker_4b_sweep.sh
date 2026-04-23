#!/bin/bash
# 4B throughput sweep variant. Mirrors run_kuhn_poker_sweep.sh but:
#   - env roll3 (for Qwen3.5-4B v5 transformers / vllm 0.19)
#   - base config is agent_kuhn_poker_fsp_4b.yaml (our 4B yaml)
#   - baked-in baseline is the A6000 3B v2 R10 winner (3tr+1inf, bs=4 ga=3,
#     rb=256, gs=4, enforce_eager=false, mnt=4096, gc=off, bf16, eager_false)
#   - one swept hyperparameter per variant
#
# Usage:
#   sbatch scripts/auton/run_kuhn_poker_4b_sweep.sh <round> <variant>
#
#SBATCH --job-name=kuhn4b_sweep
#SBATCH --output=/zfsauton/scratch/wentsec/ROLL/logs/kuhn4b_sweep_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/ROLL/logs/kuhn4b_sweep_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a6000:4

set -ex

ROUND=${1:?usage: sbatch run_kuhn_poker_4b_sweep.sh <round> <variant>}
VARIANT=${2:?usage: sbatch run_kuhn_poker_4b_sweep.sh <round> <variant>}

# A6000 3B v2 R10 winner — starting baseline for the 4B sweep.
BASELINE_OVERRIDES=(
    "actor_train.device_mapping='[0,1,2]'"                  # 3 train GPUs
    "actor_infer.device_mapping='[3]'"                      # 1 vLLM GPU
    "actor_train.training_args.per_device_train_batch_size=4"
    "actor_train.training_args.gradient_accumulation_steps=3"   # eff_batch ≈ 36 (DP=3 × bs=4 × ga=3)
    "actor_train.infer_batch_size=4"
    "actor_train.model_args.disable_gradient_checkpointing=false"  # gc ON — 4B winner (gc=off OOMs, see A6000x4_qwen3p5_4b README round B)
    "actor_infer.strategy_args.strategy_config.enforce_eager=false"
    "actor_infer.strategy_args.strategy_config.max_num_batched_tokens=4096"
    "actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.92"
    "train_env_manager.num_env_groups=32"
    "train_env_manager.group_size=8"
    "sequence_length=1536"
    "actor_infer.generating_args.max_new_tokens=1024"
    "max_tokens_per_step=1024"
    "rollout_batch_size=256"
)

BENCH_OVERRIDES=(
    "max_steps=5"
    "eval_steps=100"
    "save_steps=10000"
    "fsp_save_steps=0"
    "logging_steps=1"
)

case "${ROUND}/${VARIANT}" in
  # Round A: GPU split — the A6000 v2 headline lever. Re-validate at 4B since
  # bigger activations may shift the optimum.
  roundA_gpu_split/split_3tr1inf)  VARIANT_OVERRIDE="" ;;  # = baseline
  roundA_gpu_split/split_2tr2inf)  VARIANT_OVERRIDE="actor_train.device_mapping='[0,1]' actor_infer.device_mapping='[2,3]' actor_train.training_args.gradient_accumulation_steps=4" ;;
  roundA_gpu_split/split_1tr3inf)  VARIANT_OVERRIDE="actor_train.device_mapping='[0]' actor_infer.device_mapping='[1,2,3]' actor_train.training_args.gradient_accumulation_steps=8" ;;

  # Round B: gradient_checkpointing — biggest memory lever. 4B may need gc=on
  # where 3B didn't. gc_off is the baseline; gc_on is the fallback.
  roundB_gc/gc_off)                VARIANT_OVERRIDE="actor_train.model_args.disable_gradient_checkpointing=true" ;;
  roundB_gc/gc_on)                 VARIANT_OVERRIDE="actor_train.model_args.disable_gradient_checkpointing=false" ;;
  roundB_gc/gc_on_2tr2inf)         VARIANT_OVERRIDE="actor_train.model_args.disable_gradient_checkpointing=false actor_train.device_mapping='[0,1]' actor_infer.device_mapping='[2,3]' actor_train.training_args.gradient_accumulation_steps=4" ;;
  roundB_gc/gc_on_1tr3inf)         VARIANT_OVERRIDE="actor_train.model_args.disable_gradient_checkpointing=false actor_train.device_mapping='[0]' actor_infer.device_mapping='[1,2,3]' actor_train.training_args.gradient_accumulation_steps=8" ;;

  # Round C: enforce_eager — second-biggest perf lever per A6000 R1 (3× rollout).
  # Tested at the 4B winner config (3tr+1inf bs=4 gc=on); baseline already has eager=false.
  roundC_eager/eager_true)         VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.enforce_eager=true actor_train.model_args.disable_gradient_checkpointing=false" ;;
  roundC_eager/eager_false)        VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.enforce_eager=false actor_train.model_args.disable_gradient_checkpointing=false" ;;

  # Round D: train batch size at 3tr+1inf. bs=4 may OOM on 4B with 48 GB.
  roundD_train_bs/bs1)             VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=1 actor_train.training_args.gradient_accumulation_steps=12" ;;
  roundD_train_bs/bs2)             VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=6" ;;
  roundD_train_bs/bs4)             VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=4 actor_train.training_args.gradient_accumulation_steps=3" ;;
  roundD_train_bs/bs2_gcon)        VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=6 actor_train.model_args.disable_gradient_checkpointing=false" ;;
  roundD_train_bs/bs1_gcon)        VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=1 actor_train.training_args.gradient_accumulation_steps=12 actor_train.model_args.disable_gradient_checkpointing=false" ;;
  # With FA2 the O(N) attn memory may re-enable gc=off or larger bs at seq=1536.
  roundD_train_bs/bs4_gcoff_fa2)   VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=4 actor_train.training_args.gradient_accumulation_steps=3 actor_train.model_args.disable_gradient_checkpointing=true" ;;
  roundD_train_bs/bs8_gcon_fa2)    VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=8 actor_train.training_args.gradient_accumulation_steps=2 actor_train.model_args.disable_gradient_checkpointing=false" ;;
  roundD_train_bs/bs8_gcoff_fa2)   VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=8 actor_train.training_args.gradient_accumulation_steps=2 actor_train.model_args.disable_gradient_checkpointing=true" ;;

  # Round E: DeepSpeed stage / CPU offload. ZeRO-3 shards params across DP (frees
  # activation memory on each rank); cpu_offload moves optimizer state to CPU
  # (tiny for LoRA, nearly free). Target: re-enable gc=off or bs=8.
  roundE_zero/zero3_cpuoff_bs4_gcon)  VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero3_cpuoffload}" ;;
  roundE_zero/zero3_cpuoff_bs4_gcoff) VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero3_cpuoffload} actor_train.model_args.disable_gradient_checkpointing=true" ;;
  roundE_zero/zero3_cpuoff_bs8_gcon)  VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero3_cpuoffload} actor_train.training_args.per_device_train_batch_size=8 actor_train.training_args.gradient_accumulation_steps=2" ;;
  roundE_zero/zero3_bs4_gcon)         VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero3}" ;;
  roundE_zero/zero3_bs4_gcoff)        VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero3} actor_train.model_args.disable_gradient_checkpointing=true" ;;

  # Round F: async rollout × train overlap. async_generation_ratio=1 means rollout
  # of step N+1 starts during train of step N, so wall = max(rollout, train).
  roundF_async/async1)        VARIANT_OVERRIDE="+async_generation_ratio=1" ;;
  roundF_async/async2)        VARIANT_OVERRIDE="+async_generation_ratio=2" ;;
  # async shrinks bs to leave headroom for the lookahead rollout buffer
  roundF_async/async1_rb128)  VARIANT_OVERRIDE="+async_generation_ratio=1 rollout_batch_size=128" ;;
  roundF_async/async1_bs2)    VARIANT_OVERRIDE="+async_generation_ratio=1 actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=6" ;;

  # Round G: vLLM rollout-side cheap tweaks (max_loras↓, block_size↑, chunked_prefill=true)
  roundG_vllm/vllm_cheap)  VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_loras=10 actor_infer.strategy_args.strategy_config.block_size=32 +actor_infer.strategy_args.strategy_config.enable_chunked_prefill=true" ;;
  # each individually too, for attribution
  roundG_vllm/max_loras10)   VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_loras=10" ;;
  roundG_vllm/block32)       VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.block_size=32" ;;
  roundG_vllm/chunked)       VARIANT_OVERRIDE="+actor_infer.strategy_args.strategy_config.enable_chunked_prefill=true" ;;

  # Round H: robustness variants — expandable_segments + bs reductions
  roundH_robust/bs2_gcon_fa2) VARIANT_OVERRIDE="actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=6" ;;
  roundH_robust/expandable)   VARIANT_OVERRIDE="" ;;  # same as 5572 but with PYTORCH_ALLOC_CONF (exported below)

  # Round J: within-mini-batch dynamic batching (Fix A — equal-count sorted chunks).
  # Splits each mini-batch into exactly ga_steps sorted chunks with per-chunk narrow.
  # num_microbatches unchanged → DS grad-accum counter stays consistent.
  roundJ_dynbatch/ds_inner) VARIANT_OVERRIDE="+actor_train.use_inner_dynamic_batching_in_train=true +actor_train.sequence_length_round_in_train=64" ;;
  # At seq=1024 the baseline has ~8 GB margin — if this passes it confirms Fix A's
  # implementation is correct and the seq=1536 failures are purely memory-ceiling.
  roundJ_dynbatch/ds_inner_seq1024) VARIANT_OVERRIDE="+actor_train.use_inner_dynamic_batching_in_train=true +actor_train.sequence_length_round_in_train=64 sequence_length=1024 actor_infer.generating_args.max_new_tokens=512 max_tokens_per_step=512" ;;
  # At seq=1536 with bs=2 ga=6: halving bs drops activation memory ~5 GB → Fix A fits.
  # Eff batch stays 36 (2 × 6 × 3 DP). Expected: ~15% slower train than bs=4 but still
  # beats baseline via padding removal.
  roundJ_dynbatch/ds_inner_bs2) VARIANT_OVERRIDE="+actor_train.use_inner_dynamic_batching_in_train=true +actor_train.sequence_length_round_in_train=64 actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=6" ;;

  # Round I: dynamic batching — packs variable-length samples to a token-budget
  # micro-batch. Mean total length ~670 vs padded 1536 → ~56% padding waste today.
  # Budget 6144 ≈ bs=4 × seq=1536 keeps memory roughly equal but ~2× sample throughput.
  roundI_dynbatch/train_6144)          VARIANT_OVERRIDE="+actor_train.use_dynamic_batching_in_train=true +actor_train.max_tokens_per_microbatch_in_train=6144" ;;
  roundI_dynbatch/train_infer_6144)    VARIANT_OVERRIDE="+actor_train.use_dynamic_batching_in_train=true +actor_train.max_tokens_per_microbatch_in_train=6144 +actor_train.use_dynamic_batching_in_infer=true +actor_train.max_tokens_per_microbatch_in_infer=6144" ;;

  # Round N: ZeRO-2 + CPU optimizer offload. Round E only tested ZeRO-3+offload (all
  # OOM, big buffers). ZeRO-2 keeps the base replicated but offloads optimizer fp32
  # state to CPU — could free 200-500 MB/GPU and re-enable gc=off (worth ~40% in train).
  roundN_zero2off/zero2off_gcon_bs4)   VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero2_cpuoffload}" ;;
  roundN_zero2off/zero2off_gcoff_bs4)  VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero2_cpuoffload} actor_train.model_args.disable_gradient_checkpointing=true" ;;
  roundN_zero2off/zero2off_gcon_bs8)   VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero2_cpuoffload} actor_train.training_args.per_device_train_batch_size=8 actor_train.training_args.gradient_accumulation_steps=2" ;;
  roundN_zero2off/zero2off_gcoff_bs2)  VARIANT_OVERRIDE="actor_train.strategy_args.strategy_config=\${deepspeed_zero2_cpuoffload} actor_train.model_args.disable_gradient_checkpointing=true actor_train.training_args.per_device_train_batch_size=2 actor_train.training_args.gradient_accumulation_steps=6" ;;

  # Round O: vLLM gpu_memory_utilization push (0.92 baseline, KV ~31% used → margin)
  roundO_gpumem/gpumem095)             VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.95" ;;
  roundO_gpumem/gpumem097)             VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.97" ;;

  # Round P: rollout_batch_size extension (3B v2 R9 stopped at 256). Bigger rb →
  # more PPO micro-steps → train phase amortizes more rollout work.
  # Constraint (agentic_config.py:323): rollout_batch_size == num_env_groups × group_size.
  roundP_rb/rb384)                     VARIANT_OVERRIDE="rollout_batch_size=384 train_env_manager.num_env_groups=48 train_env_manager.num_groups_partition=[48]" ;;
  roundP_rb/rb512)                     VARIANT_OVERRIDE="rollout_batch_size=512 train_env_manager.num_env_groups=64 train_env_manager.num_groups_partition=[64]" ;;

  # Round Q: max_loras reduction. vLLM cache reserves slots per LoRA. With FSP
  # only the live policy + few enemy slots are active in a given rollout window.
  roundQ_loras/loras5)                 VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_loras=5" ;;
  roundQ_loras/loras2)                 VARIANT_OVERRIDE="actor_infer.strategy_args.strategy_config.max_loras=2" ;;

  # Round R: vLLM enable_prefix_caching. Kuhn prompts share long fixed system+rules
  # prefix — caching skips prefill for the shared part. Free win if compatible w/ LoRA.
  roundR_prefix/prefix_on)             VARIANT_OVERRIDE="+actor_infer.strategy_args.strategy_config.enable_prefix_caching=true" ;;

  *) echo "ERROR: unknown round/variant ${ROUND}/${VARIANT}"; exit 2 ;;
esac

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/roll3
ROLL_DIR=/zfsauton/scratch/wentsec/ROLL

EXP_NAME="kuhn4b_${ROUND}_${VARIANT}"
SWEEP_ROOT=/zfsauton/scratch/wentsec/kuhn_sweep_4b/${ROUND}/${VARIANT}
mkdir -p ${SWEEP_ROOT}/logs ${SWEEP_ROOT}/render

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
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

# Reduce allocator fragmentation on borderline-OOM configs
case "${ROUND}" in
  roundH_robust|roundJ_dynbatch)
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    ;;
esac

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

ray stop --force 2>/dev/null || true
sleep 2

GPU_UTIL_CSV=${SWEEP_ROOT}/logs/gpu_util.csv
echo "timestamp,index,utilization_gpu_pct,memory_used_mib" > ${GPU_UTIL_CSV}
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 2 >> ${GPU_UTIL_CSV} &
NVSMI_PID=$!
trap "kill ${NVSMI_PID} >/dev/null 2>&1 || true; rm -rf $TMPDIR >/dev/null 2>&1 || true" EXIT

cd $ROLL_DIR
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name agent_kuhn_poker_fsp_4b \
    exp_name=${EXP_NAME} \
    output_dir=${SWEEP_ROOT} \
    logging_dir=${SWEEP_ROOT}/logs \
    checkpoint_config.output_dir=${SWEEP_ROOT}/render \
    track_with=wandb \
    tracker_kwargs.project=self-play-debug \
    tracker_kwargs.name=${EXP_NAME} \
    "${BENCH_OVERRIDES[@]}" \
    "${BASELINE_OVERRIDES[@]}" \
    ${VARIANT_OVERRIDE} \
    2>&1
PY_EXIT=$?

kill ${NVSMI_PID} 2>/dev/null || true
sleep 2

# Push run summary (mean_*_steady, gpu_util_*) to wandb so collect_round_results.py
# can build a comparison table. Failures here do not fail the sweep job — wandb
# can be re-pushed manually by re-running this script with the same args.
python ${ROLL_DIR}/scripts/auton/post_sweep_to_wandb.py \
    --exp_name ${EXP_NAME} \
    --project self-play-debug \
    --gpu_util_csv ${GPU_UTIL_CSV} \
    --slurm_job_id ${SLURM_JOB_ID:-} \
    --round ${ROUND} \
    --variant ${VARIANT} \
    --tag sweep_4b || echo "WARN: post_sweep_to_wandb failed (run completed; metrics not pushed)"

echo "===== 4B SWEEP DONE (round=${ROUND} variant=${VARIANT} exit=${PY_EXIT}) ====="
exit ${PY_EXIT}
