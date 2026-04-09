# Environment Parallelism Benchmark

Sweeps `num_env_groups` (1, 8, 16, 32, 64) to find optimal GPU saturation for Kuhn Poker rollouts on 3x A40 inference GPUs.

## What it measures

- **Rollout throughput (tok/s)** as a function of parallel environment groups
- **Rollout vs training time balance** to identify the pipeline bottleneck
- **KV cache pressure** at different parallelism levels

## Configs

| Config | num_env_groups | Description |
|--------|---------------|-------------|
| `agent_kuhn_poker_bench.yaml` | (from base) | Base benchmark, 2 steps, no checkpointing |
| `agent_kuhn_poker_bench_envs1.yaml` | 1 | Baseline, single environment |
| `agent_kuhn_poker_bench_envs8.yaml` | 8 | Low parallelism |
| `agent_kuhn_poker_bench_envs16.yaml` | 16 | Medium parallelism |
| `agent_kuhn_poker_bench_envs32.yaml` | 32 | Optimal (best tok/s) |
| `agent_kuhn_poker_bench_envs64.yaml` | 64 | Over-saturated (thread contention) |

## Usage

```bash
# Run a single config
bash examples/agentic_demo/scripts/run_kuhn_bench.sh

# Run the full sweep
bash examples/agentic_demo/scripts/run_kuhn_bench_sweep.sh
```

See [BENCH_RESULTS.md](BENCH_RESULTS.md) for results and analysis.
