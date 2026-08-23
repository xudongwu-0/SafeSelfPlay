# Slurm role-LoRA phases (no Modal)

Runs the §2 role-LoRA workflow on a single Slurm node with local GPUs, using the
same patch chain, arguments and SFT mixture as
`modal_upstream_selfredteam_role_lora_v2.py`. Intended for on-prem clusters
where the Modal control plane is unreachable.

This is **latest-opponent self-play**, the same schedule as
`run_selfredteam_lora_next_round_h200x4.sh`. For sequential double-oracle PSRO
use `run_role_lora_cold_psro5_h200x4.sh`.

## Setup

```bash
export SSP_ROOT=/scratch/$USER/safeselfplay-work    # workspace, not the repo

git clone https://github.com/mickelliu/selfplay-redteaming "$SSP_ROOT/upstream"
git -C "$SSP_ROOT/upstream" checkout 0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123

python scripts/slurm/prepare_sources.py     # -> $SSP_ROOT/work  (patched tree)
python scripts/slurm/prepare_sft_data.py    # -> $SSP_ROOT/sft_data

# Base and judge weights into $SSP_ROOT/hf (HF_HOME layout):
#   mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
#   allenai/wildguard   (gated; needs an accepted licence and HF_TOKEN)
```

`SSP_PYTHON` should point at an interpreter with the upstream dependencies. The
combination this was validated against is torch 2.6.0+cu124, **vllm 0.8.2**,
transformers 4.50.0, peft 0.15.2, deepspeed 0.16.5, ray 2.44.0, numpy 1.26.4 and
`setuptools<81` (`pkg_resources` is imported at run time). vllm 0.8.4 raises
`AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'` on
the LoRA path.

## Run

```bash
SSP_ROOT=... scripts/slurm/submit_ladder.sh A1 D1 A2 D2      # chained phases
SSP_ROOT=... ROLE=attacker GEN=1 scripts/slurm/role_phase.sh # one phase
DRY_RUN=1 ... scripts/slurm/role_phase.sh                    # print the command
```

Phases are chained with `--dependency=afterany` and resolve the opponent adapter
at run time, so a phase that stops short still hands its terminal checkpoint to
the next one.

## GPU memory

`gpu_memory_utilization` is a **private per-engine fraction of the whole card**,
not a cap on total card usage. The judge, the trainable engines, the
frozen-opponent engines and the ZeRO-3 actor all draw from the same card and
must sum below 1.0. Working split on 4xH200-141GB with the judge colocated:

| consumer | fraction | note |
| --- | --- | --- |
| wildguard judge | 0.13 | floor ~0.12; 7B weights alone are 14.3G |
| trainable engines | 0.28 | `--vllm_gpu_memory_utilization` |
| frozen opponent | 0.26 | `--fixed_opponent_vllm_gpu_memory_utilization` |
| ZeRO-3 actor | rest | needs ~40G with gradient checkpointing |

Two failure modes follow from getting this wrong: an engine that profiles zero
KV blocks (`No available memory for the cache blocks`), or a `torch.OutOfMemory`
once the actor starts. Neither is fixed by raising a single fraction.

## Throughput

Generation is ~93% of a step (6-8 calls of 64 sequences), so engine overhead
dominates and spare HBM buys nothing: prompts are ~385 tokens and responses
~120, nowhere near KV-bound. Measured per step, 8B rank-64 on 4xH200:

| configuration | s/step |
| --- | --- |
| `--vllm_enable_sleep --deepspeed_enable_sleep` | 170 |
| no sleep flags, 2 engines at TP=2 | 56 |
| no sleep flags, 4 engines at TP=1 | 22.2 |
| + `--enable_prefix_caching` | 21.2 |
| + CUDA graphs (drop `--enforce_eager`) | 18.7 |

The sleep flags are the single largest cost when the judge shares the training
GPUs, because `--deepspeed_enable_sleep` offloads optimizer state to CPU on every
step. They remain worthwhile when the judge has its own GPUs, as on Modal.
`--gradient_checkpointing` is kept: the backward is only ~4s of a step, so it is
cheap memory.

## Constraints

- **One phase per node.** Each phase starts a Ray head; two on one node corrupt
  each other's placement-group scheduling and both hang after
  `Connected to Ray cluster` with the GPUs idle.
- **Do not edit `role_phase.sh` while a phase is running.** Slurm execs it and
  bash reads scripts incrementally, so an edit shifts the file offset and the
  running job dies in the epilogue after training has already finished.
- The judge's `--max_len` must stay at the upstream 8192. At 4096 a long
  transcript returns 500, the judge server shuts down, and the trainer fails
  after 12 retries.
