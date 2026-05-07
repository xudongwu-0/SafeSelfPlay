# Kuhn Poker PSRO — Getting Started

Policy Space Response Oracle (PSRO) training on Kuhn Poker using **Qwen2.5-3B-Instruct**. Each FSP generation trains against the Nash-weighted pool of past policies; after each generation the payoff matrix is expanded and a new Nash equilibrium is computed.

**Default hardware**: 4× A6000 48GB, single node.  
**GPU split**: 3 train (DeepSpeed ZeRO-2) + 1 vLLM inference.

---

## Prerequisites

- Auton `general` partition access (or a Modal account — see below)
- conda env at `/zfsauton/scratch/wentsec/envs/roll3`
- ROLL repo at `/zfsauton/scratch/wentsec/ROLL`
- `WANDB_API_KEY` and `HF_TOKEN` set in `/zfsauton/scratch/wentsec/.env_roll`

---

## Quick start (Auton)

```bash
# Submit with default config
sbatch scripts/auton/run_kuhn_psro_3b_general.sh

# Pass Hydra overrides after a bare --
sbatch scripts/auton/run_kuhn_psro_3b_general.sh -- exp_name=my_run max_steps=500
```

The script uses config `examples/agentic_demo/kuhn_psro_3b.yaml`. Logs and checkpoints land under `/zfsauton/scratch/wentsec/kuhn_poker_output/runs/<RUN_ID>/`.

**Runtime estimate**: 1000 steps × ~6 min/step ≈ 100 h total. The 48 h `general` wallclock requires 2–3 re-submissions. Set `resume_from_checkpoint: true` on re-runs — checkpoints are saved every `fsp_save_steps` steps.

---

## Quick start (Modal)

```bash
# One-time setup
pip install modal && python -m modal setup
modal secret create roll-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>

# Run PSRO config on Modal (default: A100-40GB × 4)
FSP_CONFIG_NAME=kuhn_psro_3b modal run modal_fsp_demo.py

# With extra Hydra overrides
FSP_CONFIG_NAME=kuhn_psro_3b FSP_EXTRA_OVERRIDES="exp_name=my_run max_steps=200" modal run modal_fsp_demo.py

# Use a different GPU type
FSP_CONFIG_NAME=kuhn_psro_3b FSP_GPU=A100-80GB:4 modal run modal_fsp_demo.py
```

Cost is roughly **$1.68/run** on 4×A10G. First run is slow (~30 min) due to image build; subsequent runs use Modal's layer cache.

---

## Tuning for your hardware

Throughput is sensitive to GPU type and several config knobs. Sweep results live in:

```
examples/agentic_demo/benchmark_grpo_sweep/
├── A40x4/      — 4× A40 40GB  (~530 tok/s on 3B)
├── A6000x4/    — 4× A6000 48GB  (~947 tok/s on 3B)  ← production default
└── A100x4/     — 4× A100
```

If you are running on a different node type, **read the sweep README in the matching subdir** before starting a long run. The key knobs to adjust:

| Knob | Where |
|---|---|
| `rollout_batch_size` | top-level config |
| `sequence_length` | top-level config |
| `actor_train.device_mapping` | GPU indices for training |
| `actor_infer.device_mapping` | GPU indices for vLLM |
| `actor_train.training_args.per_device_train_batch_size` | inner batch size |
| `actor_train.training_args.gradient_accumulation_steps` | accumulation steps |

The biggest single lever found in sweeps: **GPU split**. The default 3-train + 1-vLLM allocation gives +83% throughput vs 1-train + 3-vLLM on A6000. Always check the sweep notes before changing it.

---

## Evaluating two models with `start_arena_eval.py`

`examples/start_arena_eval.py` runs a round-robin tournament, discovers all LoRA checkpoints under a directory, and outputs a payoff matrix and optionally full episode trajectories. Two inference modes are supported:

### Mode 1: `local` — local vLLM (GPU required)

Boots vLLM inference workers on the current node. Use this after a training run to evaluate checkpoints.

```bash
python examples/start_arena_eval.py \
    --mode local \
    --config_name kuhn_psro_3b \
    --checkpoint_dir /zfsauton/scratch/wentsec/kuhn_poker_output/runs/<RUN_ID>/render/ \
    --output_dir ./arena_eval_output \
    --episodes_per_pair 100 \
    --save_trajectories

# or via SLURM (v100, non-A6000):
sbatch scripts/auton/run_kuhn_arena_local.sh
```

### Mode 2: `server_api` — external OpenAI-compatible API (CPU node ok)

Routes inference through an external API server. No GPU needed. API credentials are passed at runtime only — **never stored in config files or code**.

```bash
export ARENA_API_KEY=<your-api-key>
export ARENA_BASE_URL=<your-server-url>   # e.g. https://api.arcee.ai/api/v1
export ARENA_MODEL=<model-name>           # e.g. trinity-mini

python examples/start_arena_eval.py \
    --mode server_api \
    --api_key "$ARENA_API_KEY" --base_url "$ARENA_BASE_URL" --model_name "$ARENA_MODEL" \
    --config_name agent_kuhn_poker_arena_api \
    --self_play --env_tag KuhnPokerLLMThink \
    --output_dir ./arena_eval_output \
    --episodes_per_pair 12 --max_concurrent 4 --save_trajectories

# or via SLURM (cpu partition):
ARENA_API_KEY=<key> ARENA_BASE_URL=<url> ARENA_MODEL=<model> \
    sbatch scripts/auton/run_kuhn_arena_api.sh
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `local` | `local` (vLLM on GPU) or `server_api` (external API, CPU ok) |
| `--api_key` | — | [server_api] API key for external server |
| `--base_url` | — | [server_api] Base URL of external server |
| `--model_name` | — | [server_api] Model name on external server |
| `--config_name` | `agent_kuhn_poker_fsp_train` | Hydra config (`kuhn_psro_3b` for local; `agent_kuhn_poker_arena_api` for server_api) |
| `--checkpoint_dir` | — | Root dir containing `checkpoint-N/` subdirs |
| `--output_dir` | `./arena_eval_output` | Where to write results |
| `--episodes_per_pair` | 16 | Episodes per matchup — 100+ recommended for low variance |
| `--no_base_model` | off | Exclude the base model (no LoRA) from the tournament |
| `--max_checkpoints` | all | Evaluate only the last N checkpoints |
| `--save_trajectories` | off | Save full episode logs as `trajectories.jsonl` |
| `--self_play` | off | Run base model vs itself (no checkpoint dir needed) |
| `--seed` | 12345 | Random seed for episode reproducibility |

**Outputs** (in `--output_dir`):
- `arena_payoff_matrix.json` — `n × n` matrix of expected payoffs
- `arena_trajectories.jsonl` — full episode logs (if `--save_trajectories`)

---

## Key config parameters

| Parameter | Default | Description |
|---|---|---|
| `max_steps` | 1000 | Total training steps |
| `fsp_score_threshold` | 0.15 | Minimum rolling avg reward to trigger FSP switch |
| `fsp_score_timeout` | 300 | Max steps per generation before forcing FSP switch |
| `filter_zero_variance_groups` | true | Skip rollout groups with zero reward variance |
| `cold_start` | true | Reset training LoRA to init weights after each FSP snapshot |
| `psro_episodes_per_pair` | 36 | Payoff matrix eval episodes per (i, j) policy pair |
| `rollout_batch_size` | 264 | Must equal `num_env_groups × group_size` (11 × 24) |
| `sequence_length` | 1024 | Shorter = faster; increase if responses are truncated |
