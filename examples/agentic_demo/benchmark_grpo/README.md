# GRPO Group Size Sweep

Sweeps `group_size` (1, 2, 4, 8) to measure the effect of per-group reward normalization on Kuhn Poker training stability and convergence.

## Background

GRPO computes advantages by normalizing rewards within a group of rollouts from the same prompt. `group_size` controls how many rollouts share a baseline:
- **group_size=1**: No per-group baseline; relies on global reward whitening
- **group_size=2+**: Per-group mean/std normalization; higher values give lower-variance baselines but cost more rollout compute

## Configs

| Config | group_size | Normalization |
|--------|-----------|---------------|
| `agent_kuhn_poker_grpo_sweep.yaml` | (base) | Base sweep config, 100 steps, 32 env groups |
| `agent_kuhn_poker_grpo_gs1.yaml` | 1 | Global whitening only |
| `agent_kuhn_poker_grpo_gs2.yaml` | 2 | Per-group norm |
| `agent_kuhn_poker_grpo_gs4.yaml` | 4 | Per-group norm |
| `agent_kuhn_poker_grpo_gs8.yaml` | 8 | Per-group norm |

## Usage

```bash
bash examples/agentic_demo/scripts/run_grpo_sweep.sh
```

All runs use `num_env_groups=32` (optimal from the [env parallelism benchmark](../benchmark_envs/)).
