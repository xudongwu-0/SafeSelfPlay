# Kuhn Poker FSP — Demo

Fictitious Self-Play (FSP) training of **Qwen3.5-4B** on Kuhn Poker using ROLL's agentic pipeline. Each agent plays poker against a frozen pool of its past selves; the pool grows every `fsp_save_steps` steps and the training LoRA is optionally reset (cold-start) to force each generation to learn from scratch.

**Hardware**: 4× A6000 (48 GB) on Auton `general` partition.  
**GPU split**: 3 train (DeepSpeed ZeRO-2, DP=3) + 1 vLLM infer — see [sweep results](#sweep-results) for why.

---

## Files

```
kuhn_poker/
├── train_sync.yaml    # Production: sync rollout+train (simpler, easier to debug)
├── train_async.yaml   # Production: async rollout+train (~1.3–1.6× faster on clean steps)
├── smoke.yaml         # Sanity check: 3 steps, 1 train GPU + 3 vLLM GPUs
├── debug.yaml         # Fast iteration: 2 GPUs, Qwen2.5-0.5B, debug partition
├── run_train.sh       # Launch training (prompts for sync/async)
└── run_smoke.sh       # Launch smoke check
```

---

## Launch training

```bash
# From the ROLL root — prompts "sync or async?" interactively
bash examples/agentic_demo/kuhn_poker/run_train.sh

# Or specify mode directly
bash examples/agentic_demo/kuhn_poker/run_train.sh --mode sync
bash examples/agentic_demo/kuhn_poker/run_train.sh --mode async
```

Pass Hydra overrides after `--`:

```bash
bash examples/agentic_demo/kuhn_poker/run_train.sh --mode sync -- max_steps=100 exp_name=my_run
```

**Sync vs async**:

| | Sync | Async |
|---|---|---|
| `async_generation_ratio` | `0` | `1` |
| Rollout + train | Alternate each step | Overlap — vLLM generates while GPUs 0-2 train |
| Speedup | — | ~1.3× steady-state; ~1.6× on clean steps (see [below](#async-vs-sync)) |
| FSP cold-start | Simple | Requires flush of pre-reset rollouts (handled automatically) |
| Recommended for | Debugging / first runs | Production |

**Full run estimate**: 300 steps × ~6.3 min/step ≈ **31 h**. This exceeds the 24 h `general` wallclock. Use `resume_from_checkpoint: true` and re-submit — FSP checkpoints every 50 steps act as natural resume points.

---

## Run the smoke check

Validates the full env→rollout→train→FSP pipeline in ~10-15 min (3 steps, no eval):

```bash
bash examples/agentic_demo/kuhn_poker/run_smoke.sh
```

The smoke config uses `enforce_eager: true` and 1-GPU training for fast startup. It does **not** run FSP (`fsp_save_steps: 10000`). If smoke passes, the production configs will work.

---

## Debug mode

For rapid code iteration: 2 GPUs, **Qwen2.5-0.5B** (tiny), `debug` partition (1-hour limit, faster queue). Exercises the FSP codepath every step (`fsp_save_steps: 1`).

```bash
# Run directly (no sbatch — useful for interactive nodes)
cd /zfsauton/scratch/wentsec/ROLL
python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name kuhn_poker/debug \
    logging_dir=/zfsauton/scratch/wentsec/kuhn_poker_output/debug/logs \
    output_dir=/zfsauton/scratch/wentsec/kuhn_poker_output/debug \
    checkpoint_config.output_dir=/zfsauton/scratch/wentsec/kuhn_poker_output/debug/render

# Or via sbatch (debug partition)
sbatch --partition=debug --qos=qos_debug --time=01:00:00 --gres=gpu:a6000:2 \
    --wrap="cd /zfsauton/scratch/wentsec/ROLL && \
            source /zfsauton/scratch/wentsec/.env_roll && \
            conda activate /zfsauton/scratch/wentsec/envs/roll2 && \
            python examples/start_agentic_pipeline.py \
                --config_path agentic_demo --config_name kuhn_poker/debug"
```

Debug differences vs production:

| | Debug | Production |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct | Qwen3.5-4B |
| GPUs | 2 (1 train + 1 vLLM) | 4 (3 train + 1 vLLM) |
| Batch | 8 | 264 |
| Steps | 3 | 300 |
| FSP | Every step | Every 50 steps |
| Partition | debug (1 h) | general (24 h) |

---

## Key config parameters

| Parameter | Value | Notes |
|---|---|---|
| `pretrain` | `Qwen/Qwen3.5-4B` | Qwen3.5 requires `dtype: bf16`, `template: qwen3`, `flash_attn: fa2` |
| `rollout_batch_size` | 264 | Must equal `num_env_groups × group_size` (33 × 8) — asserted at startup |
| `sequence_length` | 1536 | Prompt (~240 tok) + max `<think>` (1024 tok) + slack; 1024 causes truncation |
| `fsp_save_steps` | 50 | Save LoRA to enemy pool every 50 steps |
| `cold_start` | `true` | Reset training LoRA to init weights after each FSP snapshot |
| `max_loras` | 7 | `ceil(max_steps / fsp_save_steps) + 1 = ceil(300/50) + 1` — size the LRU cache to fit the full enemy pool |
| `kl_loss_coef` | 0.01 | KL penalty coefficient; set `kl_loss_coef_end ≥ 0` to linearly decay over training |
| `actor_train.device_mapping` | `[0,1,2]` | 3 train GPUs — the main lesson from sweeps |
| `actor_infer.device_mapping` | `[3]` | 1 vLLM GPU — inference is not the bottleneck |

---

## Sweep results

Throughput benchmarks live in [`../benchmark_grpo_sweep/`](../benchmark_grpo_sweep/):

| Directory | Model | GPUs | Best TPS | Notes |
|---|---|---|---|---|
| [`A40x4/`](../benchmark_grpo_sweep/A40x4/) | Qwen2.5-3B | 4× A40 | ~530 tok/s | 6-knob sweep on single-RL task; `sequence_length=1024` was the biggest lever (2.8× baseline) |
| [`A6000x4/`](../benchmark_grpo_sweep/A6000x4/) | Qwen2.5-3B | 4× A6000 | **947 tok/s** (v2) | 10-round coordinate-descent on FSP task; GPU split (1tr+3inf → 3tr+1inf) was the biggest lever (+83%) |
| [`A6000x4_qwen3p5_4b/`](../benchmark_grpo_sweep/A6000x4_qwen3p5_4b/) | Qwen3.5-4B | 4× A6000 | ~456 tok/s | gc=on required (OOM at gc=off); async OOMs; ZeRO-3 hurts LoRA |

### Top-line findings

**GPU split is the biggest lever by far.** The default 1-train + 3-vLLM allocation was wrong — vLLM is massively under-saturated for Kuhn prompts at practical batch sizes. 3-train + 1-vLLM gives +83% TPS on 3B and is the production config for 4B.

**`sequence_length` is the biggest single knob for train time.** Shorter padding means smaller activations and log-prob tensors. On 3B: seq=4096→1024 gave 2.8× TPS. For 4B we need seq=1536 to fit Qwen3.5's `<think>` responses.

**Gradient checkpointing is mandatory for 4B.** Peak VRAM at gc=off is 47.35 GB / 47.4 GB (1.2% margin at any batch size). gc=on costs ~40% train time but is the only option. bs=4, ga=3 (eff_batch=36) is the ceiling.

**Async pipeline validated on 4B**: 1.32× steady-state speedup over a cycle with two FSP cold-starts; 1.58× on clean steps. FSP cold-start + force-sync flush works correctly with no deadlock.

### Per-step bottleneck (4B winner, 3tr+1inf)

| Phase | Time (s) | Share |
|---|---:|---:|
| Train forward+backward | 222 | 59% |
| Rollout (vLLM, 1 GPU) | 112 | 30% |
| Log-probs forward | 41 | 11% |
| **Total** | **~375** | |

The dominant cost is `step_train` on the training GPUs with gc=on recomputation. The vLLM GPU is under-saturated (KV cache ~31% used). The next untested lever is offloading `compute_log_probs` to the vLLM cluster.
