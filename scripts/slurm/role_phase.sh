#!/usr/bin/env bash
# Train one role-LoRA oracle on a single Slurm node, without Modal.
#
# Mirrors the argument set that modal_upstream_selfredteam_role_lora_v2.py passes
# to openrlhf.cli.train_ppo_ray, including the postfill-CoT SFT mixture. An
# attacker phase suppresses the defender turn and serves a frozen defender from
# the opponent adapter; a defender phase still runs the attacker turn, and
# optimizer_train_role decides which replay items receive updates.
#
#   ROLE=attacker GEN=1 scripts/slurm/role_phase.sh
#   ROLE=defender GEN=1 OPPONENT=$SSP_ROOT/psro/A1/ckpt/global_step100_hf \
#       scripts/slurm/role_phase.sh
#
# Set DRY_RUN=1 to print the resolved trainer command and exit.
set -eu -o pipefail

# --- workspace ---------------------------------------------------------------
# SSP_ROOT holds work/ (patched upstream tree), hf/ (HF cache), sft_data/ and
# psro/ (outputs). Populate the first two with scripts/slurm/prepare_sources.py
# and scripts/slurm/prepare_sft_data.py.
SSP_ROOT=${SSP_ROOT:?set SSP_ROOT to the workspace directory}
WORK=${SSP_WORK:-$SSP_ROOT/work}
HF_ROOT=${SSP_HF_HOME:-$SSP_ROOT/hf}
SFT_DIR=${SSP_SFT_DIR:-$SSP_ROOT/sft_data}
PSRO=${SSP_OUT:-$SSP_ROOT/psro}
PY=${SSP_PYTHON:-python}
GPUS=${GPUS:-4}

ROLE=${ROLE:?attacker|defender}
GEN=${GEN:-1}
STEPS=${STEPS:-100}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
OPPONENT=${OPPONENT:-}            # frozen opponent adapter; empty means base only
START_ADAPTER=${START_ADAPTER:-}  # warm-start this role instead of cold-starting

# --- memory -------------------------------------------------------------------
# vLLM's gpu_memory_utilization is a PRIVATE per-engine fraction of the whole
# card, not a cap on total card usage. The judge, the trainable engines, the
# frozen-opponent engines and the ZeRO-3 actor must therefore sum below 1.0.
# Measured on H200-141GB with the judge colocated on GPU 0:
#   judge 0.13 + trainable 0.28 + opponent 0.26 + actor (rest) ~= 0.93
# The judge floor is ~0.12; wildguard-7B weights alone are 14.3G.
UTIL=${UTIL:-0.28}
OPP_UTIL=${OPP_UTIL:-0.26}
JUDGE_UTIL=${JUDGE_UTIL:-0.13}

# --- throughput ---------------------------------------------------------------
# Generation is ~93% of a step, so engine-side overhead dominates and spare HBM
# buys nothing (385-token prompts, ~120-token responses). Measured per step on
# 4xH200: 56.1s -> 22.2s by dropping the sleep flags and using TP=1, -> 21.2s
# with prefix caching, -> 18.7s with CUDA graphs.
#
# Do NOT add --vllm_enable_sleep/--deepspeed_enable_sleep here: with the judge
# colocated they cost ~110s/step, because deepspeed offloads optimizer state to
# CPU on every step. They are only worthwhile when the judge has its own GPUs.
ENGINES=${ENGINES:-$GPUS}   # colocated mode asserts actor_gpus == ENGINES*TP
TP=${TP:-1}                 # TP>1 only adds an all-reduce per decode step for 8B
EAGER=${EAGER:-0}           # 1 restores --enforce_eager (disables CUDA graphs)
PREFIX=${PREFIX:-1}         # reuse the shared prompt prefix across calls
MICRO_ROLLOUT=${MICRO_ROLLOUT:-32}
MICRO_TRAIN=${MICRO_TRAIN:-8}          # train_batch_size / GPUS
TRAIN_BATCH=${TRAIN_BATCH:-32}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-128}

ENGINE_PERF=()
[ "$EAGER" = 1 ] && ENGINE_PERF+=(--enforce_eager)
[ "$PREFIX" = 1 ] && ENGINE_PERF+=(--enable_prefix_caching)

# --- role-specific settings, per role_lora_v2.py ------------------------------
case $ROLE in
  attacker)
    LR=${LR:-1e-5}; SFT_STOP=${SFT_STOP:-30}
    SFT_DATA=$SFT_DIR/attacker_rewrite_1180_rl_continuation.jsonl
    SFT_PROBS=1.0; SFT_KEYS=() ;;
  defender)
    LR=${LR:-4e-5}; SFT_STOP=${SFT_STOP:-10}
    SFT_DATA=$SFT_DIR/defender_rl_sft/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl,$SFT_DIR/defender_rl_sft/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl
    SFT_PROBS=0.5,0.5
    SFT_KEYS=(--sft_input_key vanilla --sft_output_key completion) ;;
  *) echo "ROLE must be attacker|defender" >&2; exit 1 ;;
esac
for f in ${SFT_DATA//,/ }; do
  [ -f "$f" ] || { echo "missing SFT file: $f (run prepare_sft_data.py)" >&2; exit 1; }
done

# --target_modules is nargs="*" with a string default, so an unset value is
# iterated character-wise and PEFT looks for modules named {"a","l","-",...}.
LORA_TARGET_MODULES=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)

TAG=${TAG:-${ROLE:0:1}${GEN}}
OUT=$PSRO/$TAG
mkdir -p "$OUT"

# --- environment --------------------------------------------------------------
unset TRANSFORMERS_CACHE            # overrides HF_HOME if a login shell set it
export HF_HOME=$HF_ROOT HF_HUB_CACHE=$HF_ROOT/hub
export TMPDIR=${TMPDIR:-$SSP_ROOT/tmp} TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
# Both are set by role_lora_v2.py immediately before `ray start`: the v1
# EngineCore subprocess exits silently under the LoRA path, and NCCL's cuMem
# allocator conflicts with vLLM's memory pool.
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
# The patched vllm_worker_wrap imports roll.third_party.vllm.TensorLoRARequest;
# Modal mounts the repo for this, so add both trees to the path here.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$WORK:$REPO_ROOT:${PYTHONPATH:-}

BASE=${BASE_MODEL_PATH:-$(ls -d "$HF_ROOT"/hub/models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated/snapshots/*/ | head -1)}
JUDGE=${JUDGE_MODEL_PATH:-$(ls -d "$HF_ROOT"/hub/models--allenai--wildguard/snapshots/*/ | head -1)}
IP=$(hostname -I | awk '{print $1}')
JUDGE_PORT=${JUDGE_PORT:-5100}
RAY_PORT=${RAY_PORT:-6100}
RM_URL=http://$IP:$JUDGE_PORT/classify
cd "$WORK"

CUSTOM=$($PY - "$ROLE" "$SFT_STOP" <<'PY'
import json, sys
role, sft_stop = sys.argv[1], int(sys.argv[2])
print(json.dumps({
    "max_turns": 2, "reward_type": "general_sum", "remove_ties": True,
    "redistribute_after_ties": True,
    "no_defender_turn": role == "attacker",
    "no_attacker_turn": False,
    "optimizer_train_role": role,
    "fixed_defender_from_opponent_vllm": role == "attacker",
    "fixed_attacker_from_opponent_vllm": role == "defender",
    "actor_lr_scheduler": "constant_with_warmup",
    "postfill_cot_stop_after_step": sft_stop,
}))
PY
)

# The frozen opponent always needs a base model; the adapter is layered on from
# the first phase that faces a trained opponent.
OPP_ARGS=(--fixed_opponent_pretrain "$BASE"
          --fixed_opponent_vllm_gpu_memory_utilization "$OPP_UTIL")
[ -n "$OPPONENT" ] && OPP_ARGS+=(--fixed_opponent_lora_path "$OPPONENT"
                                --fixed_opponent_lora_rank "$LORA_RANK")
START_ARGS=(); [ -n "$START_ADAPTER" ] && START_ARGS=(--role_start_adapter "$START_ADAPTER")

TRAIN_CMD=("$PY" -m openrlhf.cli.train_ppo_ray
  --actor_num_nodes 1 --actor_num_gpus_per_node "$GPUS"
  --ref_num_nodes 1 --ref_num_gpus_per_node "$GPUS"
  --remote_rm_url "$RM_URL"
  --vllm_num_engines "$ENGINES" --vllm_tensor_parallel_size "$TP"
  --colocate_all_models --vllm_gpu_memory_utilization "$UTIL"
  --pretrain "$BASE" --save_path "$OUT" --ckpt_path "$OUT/ckpt"
  --save_steps "${SAVE_STEPS:-10}" --save_hf_ckpt --disable_ds_ckpt --max_ckpt_num 1
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA"
  --target_modules "${LORA_TARGET_MODULES[@]}"
  "${OPP_ARGS[@]}" "${START_ARGS[@]}"
  --micro_train_batch_size "$MICRO_TRAIN" --train_batch_size "$TRAIN_BATCH"
  --micro_rollout_batch_size "$MICRO_ROLLOUT" --rollout_batch_size "$ROLLOUT_BATCH"
  --prompt_data "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl"
  --prompt_data_probs "0.5, 0.5"
  --eval_data "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl"
  --max_samples $((STEPS * ROLLOUT_BATCH)) --max_epochs 1
  --prompt_max_len 2048 --generate_max_len 2048
  --zero_stage 3 --num_episodes 1 --bf16 --seed "${TRAINING_SEED:-8888}" --top_p 1.0
  --gradient_checkpointing --gradient_checkpointing_use_reentrant
  --monitor_reference_kl --temperature 1.0 --lr_warmup_ratio "${LR_WARMUP_RATIO:-0.03}"
  --sft_data "$SFT_DATA" --sft_data_probs "$SFT_PROBS"
  --sft_steps 1 --sft_batches_per_step "${SFT_BATCHES_PER_STEP:-1}" "${SFT_KEYS[@]}"
  --actor_learning_rate "$LR" --init_kl_coef 0
  --normalize_reward --packing_samples
  --advantage_estimator reinforce
  --custom_configs "$CUSTOM"
  --actor_loss_coef 1.0 --postfill_cot_loss_coef 1.0
  --eval_steps 100000 --eval_start_steps 100000 --diversity_score_steps 100000
  --vllm_sync_backend nccl "${ENGINE_PERF[@]}" --flash_attn)

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${TRAIN_CMD[@]}"; echo
  exit 0
fi

# --- judge --------------------------------------------------------------------
# The judge shares GPU 0: it is idle for most of a step, so a whole card is
# waste. --max_len must stay at the upstream 8192; at 4096 a long transcript
# returns 500 and takes the whole server down mid-run.
CUDA_VISIBLE_DEVICES=0 RAY_TMPDIR=/tmp/ssp-judge-$JUDGE_PORT \
nohup "$PY" -m openrlhf.cli.serve_wildguard \
  --model_path "$JUDGE" --port "$JUDGE_PORT" --host "$IP" --bf16 \
  --num_gpus 1 --tensor_parallel_size 1 --enforce_eager --seed 42 \
  --max_len 8192 --max_num_seqs 32 --batch_size 32 \
  --gpu_memory_utilization "$JUDGE_UTIL" > "$OUT/judge.log" 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true' EXIT
for i in $(seq 1 90); do
  curl -s -m 5 -X POST "$RM_URL" -H 'Content-Type: application/json' \
    -d '{"queries":[{"prompt":"hi","response":"hello","game_idx":0}],"prompts":null}' \
    2>/dev/null | grep -q labels && break
  kill -0 $JUDGE_PID 2>/dev/null || { echo "judge died" >&2; tail -20 "$OUT/judge.log" >&2; exit 1; }
  sleep 10
done
echo "$TAG judge up after ~$((i * 10))s"

# --- ray ----------------------------------------------------------------------
# One head per node: two concurrent phases on the same node corrupt each other's
# placement-group scheduling and both hang after "Connected to Ray cluster".
export RAY_TMPDIR=/tmp/ssp-ray-$RAY_PORT; rm -rf "$RAY_TMPDIR"
ray start --head --disable-usage-stats --temp-dir="$RAY_TMPDIR" \
  --port="$RAY_PORT" --include-dashboard=false \
  --ray-client-server-port=$((RAY_PORT + 2000)) --num-gpus="$GPUS" >/dev/null
export RAY_ADDRESS=$IP:$RAY_PORT
trap 'kill $JUDGE_PID 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true' EXIT

echo "$TAG start $(date +%H:%M:%S) role=$ROLE lr=$LR steps=$STEPS opponent=${OPPONENT:-none}" >> "$PSRO/state.txt"
"${TRAIN_CMD[@]}"
echo "$TAG done rc=$? $(date +%H:%M:%S)" >> "$PSRO/state.txt"
