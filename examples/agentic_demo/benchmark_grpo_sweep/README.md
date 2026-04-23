# Kuhn Poker GRPO / FSP throughput benchmarks

Two GPU types, two task variants:

- **[A40x4/](A40x4/)** — original sweep on Delta `gpuA40x4` (4× A40 40GB) against `agent_kuhn_poker_single_rl`. Best ~530 tok/s.
- **[A6000x4/](A6000x4/)** — Auton `general` partition (4× A6000 48GB) against `agent_kuhn_poker_fsp_train` (production FSP task). Coordinate-descent loop starting from A40 winners; per-run GPU utilization recorded.
