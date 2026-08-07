# Strict Self-RedTeam Dual-Full Experiment

## Purpose

This experiment isolates the role-parameterization change from the upstream
Self-RedTeam training pipeline:

- `A`: one full-parameter Llama-3.1 8B attacker policy;
- `D`: one independently initialized full-parameter Llama-3.1 8B defender
  policy;
- only the active role has a DeepSpeed optimizer;
- the opponent is immutable and is used only by a colocated vLLM engine;
- A phase runs before D phase.

The implementation is in `modal_upstream_selfredteam_role_full.py`.

## Model flow

For the default `A50 + D50` run:

1. `A0 = mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`.
2. `D0 = mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`.
3. Train full `A0` for 50 rollout steps against frozen full `D0`; save `A50`.
4. Reinitialize the trainable full defender from `D0`.
5. Train full `D0` for 50 rollout steps against frozen full `A50`; save `D50`.

`D` is not initialized from `A50`, and no optimizer contains opponent
parameters.

## Exact settings retained from upstream

| Setting | Upstream | Dual-full |
|---|---:|---:|
| Base | Llama-3.1-8B-Instruct-abliterated | same |
| Reward | `general_sum` | same |
| Turns | 2 | same |
| Tie removal | enabled | same |
| Advantage | REINFORCE, normalized | same |
| Actor LR | `5e-7` | same |
| Scheduler | cosine with minimum LR | same |
| KL coefficient | `0.01` | same |
| Prompt mixture | harmful/benign `0.5/0.5` | same |
| Rollout batch | 128 | same |
| Micro rollout batch | 8 | same |
| Train batch | 32 | same |
| Micro train batch | 8 | same |
| Max generation | 2048 | same |
| BF16 / ZeRO / packing | BF16 / ZeRO-3 / on | same |
| SFT steps / batches | 1 / 1 | same |
| Postfill SFT coefficient | 1.0 | same |
| GPUs | H200 x4 | same |

## Intentional differences

| Dimension | Upstream shared policy | Dual-full |
|---|---|---|
| Parameters | one shared full 8B policy | two independent full 8B policies |
| Update schedule | A and D samples update the same policy every rollout | A-only block, then D-only block |
| Frozen opponent | none | full opponent in a second vLLM fleet |
| Attacker SFT | defender answer SFT reaches the shared policy | 1,180 cleaned attacker rewrite demonstrations only |
| Defender SFT | two answer files, 29,996 unique rows | the same two answer files only |

The upstream SFT files cannot be literally divided into attacker and defender
subsets: all 29,996 rows contain defender-style `completion` values and zero
rows contain a non-empty `adversarial` field. The role-specific data mapping is
therefore explicit instead of pretending those files include attacker targets.

## Budget interpretation

The default `A50 + D50` uses `100 * 128 = 12,800` environment seed prompts,
equal to an upstream shared-policy 100-step run. This is the controlled sample
budget used for the primary comparison.

The number of optimized role trajectories is not equal to the shared-policy
run. Approximately half of prompts generate attacker turns, so the default run
optimizes about `50 * 64 = 3,200` attacker trajectories and
`50 * 128 = 6,400` defender trajectories before tie removal. A shared policy
at 100 steps sees approximately twice those role-specific counts because both
roles update on every step. Use `STEPS_PER_ROLE=100` for a role-exposure-matched
secondary run; that doubles the environment prompt budget.

## Validation and launch

CPU-only schema and patch validation:

```bash
modal run \
  modal_upstream_selfredteam_role_full.py::validate_strict_dual_full_configuration_entrypoint
```

Detached H200 x4 run:

```bash
./run_strict_selfredteam_dualfull_h200x4.sh
```

Checkpoints are written under:

```text
/output/upstream_selfredteam_role_full/
  strict_dualfull_A50D50_<suffix>/
    A_full_s50_vs_baseD/ckpt/global_step50_hf/
    D_full_s50_vs_A50/ckpt/global_step50_hf/
```
