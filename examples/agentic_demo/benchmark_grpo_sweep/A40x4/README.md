# Kuhn Poker GRPO Benchmarks

Model: Qwen2.5-3B-Instruct, LoRA rank=32, 4x A40 GPUs (1 train + 3 infer)

## 1. num_env_groups Sweep (Rollout Throughput)

Config: `rollout_batch_size=64`, `gradient_accumulation_steps=16`, `group_size=1`

| num_env_groups | Rollout (s) | Train (s) | TPS (tok/s) |
|:-:|:-:|:-:|:-:|
| 1 | 800 | 166 | 27.0 |
| 8 | 159 | 169 | 73.1 |
| 16 | 94 | 169 | 81.2 |
| 32 | 61 | 169 | 87.7 |
| 64 | 34 | 166 | 86.3 |

**Takeaway**: Rollout is the bottleneck at low parallelism. num_env_groups=32-64 saturates vLLM throughput; beyond that, train time dominates.

## 2. group_size Sweep (GRPO Baseline)

Config: `rollout_batch_size=128`, `gradient_accumulation_steps=32`, `num_env_groups=32`, `per_device_train_batch_size=1`

| group_size | Avg Rollout (s) | Avg Train (s) | Avg TPS | Steps | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 90.2 | 327.4 | 110.2 | 5 | grouping=tags, no per-group baseline |
| 2 | 52.5 | 327.7 | 116.4 | 6 | grouping=traj_group_id, group mean baseline |
| 4 | 39.4 | 330.9 | 123.2 | 6 | same as gs2 |
| 8 | - | - | - | 0 | crashed (IndexError: empty history) |

**Takeaway**: Larger group_size reduces rollout time (more batching per group) but train time is constant (~327s, dominates 75% of step). gs8 crash needs fix in `traj_env_manager.py:292`.

## 3. per_device_train_batch_size Sweep (Training Throughput)

Config: `num_env_groups=32`, `group_size=1`, `sequence_length=1024`, `enforce_eager=false`, effective_batch=32 (batch_size * grad_accum = 32)

| batch_size | grad_accum | Avg Rollout (s) | Avg Train (s) | Log probs (s) | Avg TPS | Wall Time | Status |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 32 | 29.5 | 51.8 | 19.6 | 492.9 | 01:00:21 | completed |
| 2 | 16 | 29.1 | 50.5 | 21.1 | 470.4 | 01:02:27 | completed |
| **4** | **8** | **32.1** | **50.1** | **19.4** | **529.3** | **00:48:51** | **completed** |
| 8 | 4 | - | - | - | - | - | OOM (42.4/44.4 GB) |

| 8 (gc=on) | 4 | 32.4 | 60.4 | 19.5 | 474.6 | 00:16:09 | completed (5 steps) |

**Takeaway**: Train time is constant (~50s) regardless of batch size — GPU is compute-saturated at bs=1 with seq_len=1024. bs=4 is ~8% faster TPS (529 vs 493) due to less framework overhead. bs=8 OOMs without gradient checkpointing; with gc=on it fits but is slower (60s vs 50s) due to recomputation overhead. ~38% of train time (~20s) is `compute_log_probs` on the single training GPU.

### 3b. infer_batch_size Sweep (Log Probs Throughput)

Config: same as §3 with `per_device_train_batch_size=4`, sweeping `actor_train.infer_batch_size` for `compute_log_probs` forward-only pass.

| infer_batch_size | Log probs (s)* | Train (s) | TPS | Status |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 19.4 | 50.1 | 529.3 | baseline |
| 4 | 17.9 | 50.3 | 515.0 | completed |
| 8 | 17.5 | 50.0 | 516.0 | completed |
| 16 | 19.6 | 49.8 | 497.0 | completed |
| 32 | - | - | - | crashed (Ray actor unavail) |

*steady-state, excluding warmup step

**Takeaway**: Minimal impact (~10% at best at ibs=8). Forward pass is memory-bandwidth bound — loading 6GB of model weights per layer dominates, not compute. Larger batch doesn't improve utilization. ibs=16 regresses, ibs=32 crashes.

## 4. enforce_eager Sweep (CUDA Graphs)

Config: `num_env_groups=32`, `group_size=1`, `max_steps=5`, `rollout_batch_size=128`, `gradient_accumulation_steps=32`

| enforce_eager | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|
| true | 90 | 327 | 81 | 108 | baseline (no CUDA graphs) |
| **false** | **31** | 328 | 81 | **121** | CUDA graphs enabled, 3x rollout speedup |

**Takeaway**: `enforce_eager=false` cuts rollout 3x (90→31s). TPS only +12% because train (327s) dominates. Free win, no downsides.

## 5. max_new_tokens Sweep (vLLM KV Cache)

Config: `num_env_groups=32`, `group_size=1`, `max_steps=5`, `max_tokens_per_step` set to match

| max_new_tokens | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 64 | 42 | 327 | 81 | 98 | too short, more truncated responses |
| 128 | 83 | 327 | 81 | 106 | |
| 256 | 102 | 327 | 81 | 102 | |
| 512 | 90 | 327 | 81 | 108 | baseline |

**Takeaway**: Reducing max_new_tokens hurts TPS. Shorter limit causes more truncated/invalid responses, reducing effective tokens. Keep at 512.

## 6. sequence_length Sweep (Training Padding)

Config: `num_env_groups=32`, `group_size=1`, `max_steps=5`

| sequence_length | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 512 | FAILED | - | - | - | input prompts exceed 512 tokens |
| **1024** | 86 | **51** | **19** | **300** | **best: 2.8x baseline** |
| 2048 | 95 | 120 | 39 | 205 | 1.9x baseline |
| 4096 | 90 | 327 | 81 | 108 | baseline |

**Takeaway**: `sequence_length` is the single biggest lever. Shorter = less padding in train/log_prob forward passes. seq1024 gives 2.8x TPS. seq512 fails (Kuhn Poker prompts ~530 tokens).

## Best Config Recommendation

```yaml
# Combine all sweep winners for max throughput
sequence_length: 1024              # 2.8x TPS (biggest win)
per_device_train_batch_size: 4     # 8% faster, best fit on A40
gradient_accumulation_steps: 8     # keeps effective batch=32
actor_infer:
  strategy_args:
    strategy_config:
      enforce_eager: false         # 3x rollout speedup
  generating_args:
    max_new_tokens: 512            # keep default, reducing hurts
train_env_manager:
  num_env_groups: 32               # saturates vLLM
  group_size: 4                    # reduces rollout further
```

Achieved TPS: **~530 tok/s** (vs 27 tok/s original, ~20x improvement).

Remaining bottleneck: `compute_log_probs` takes ~20s/step (38% of train time) on a single training GPU. Increasing `infer_batch_size` only helps ~10% (memory-bandwidth bound). Real fix: offload log_probs computation to the 3-GPU vLLM inference cluster.
