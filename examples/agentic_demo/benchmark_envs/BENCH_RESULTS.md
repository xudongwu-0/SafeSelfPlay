# Kuhn Poker Parallelism Benchmark

Sweep `num_env_groups` to find optimal GPU saturation for 3x A40 inference GPUs.

**Setup**: Qwen2.5-3B-Instruct, LoRA rank 32, `rollout_batch_size=64`, 4x A40 (1 train + 3 infer), vLLM inference.

## Results

Step 1 (steady-state) where available, step 0 otherwise (marked with \*).

| num_env_groups | rollout (s) | train (s) | tps | kv_cache % | rollout speedup |
|---|---|---|---|---|---|
| 1 | 848.4* | 166.3 | 26.5 | 0.11% | 1.0x |
| 8 | 140.7 | 170.2 | 73.7 | 0.17% | 6.0x |
| 16 | 77.8 | 168.3 | 84.7 | 0.22% | 10.9x |
| **32** | **52.9** | 168.7 | **92.8** | 0.37% | **16.0x** |
| 64 | 37.2* | 166.3 | 78.9 | 0.50% | 22.8x* |

## Takeaways

- **1 env was severely bottlenecked**: rollout (848s) dominated train (166s) by 5:1.
- **32 envs is the sweet spot**: rollout (53s) is 3.2x faster than training (169s), pipeline becomes train-bound. ~6x faster end-to-end.
- **64 envs has lower tps than 32**: thread contention overhead outweighs parallelism gain.
- **KV cache headroom is massive** (<1% used) since Kuhn Poker sequences are short.

## Recommended Config

```yaml
train_env_manager:
  num_env_groups: 32
  group_size: 1
  max_env_num_per_worker: 64
  num_groups_partition: [32]
```

## Next Bottleneck

Training time (168s) now dominates. Options:
- Reduce `gradient_accumulation_steps` (currently 16)
- Enable `async_generation_ratio > 0` to overlap rollout with training
