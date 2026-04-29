# Inner dynamic batching — config guide

A drop-in optimization for the agentic pipeline's training step that removes ~half of the per-batch padding waste. Available **only on the DeepSpeed ZeRO-2 strategy**, gated behind a single yaml flag (default off — zero behavior change unless enabled).

## What it does

For each mini-batch fed to `actor_train.train_step`, the patch:
1. **Sorts** the samples by their (unpadded) sequence length.
2. **Splits** the sorted mini-batch into exactly `gradient_accumulation_steps` equal-count chunks (sample count, not token count).
3. **Narrows** each chunk's tensors to that chunk's actual maximum sequence length, rounded up to a configurable multiple.

Because all samples within a chunk now have similar lengths, padding inside the chunk drops from the global mini-batch max to the chunk-local max — typically ~50–60% reduction in per-chunk padded tokens for our Kuhn Poker workload.

## When to enable it

Use cases:
- Mean trajectory length is well below `sequence_length` (you're paying a lot of padding).
- Training-time is the bottleneck (per-step train > rollout).
- You're on the DeepSpeed ZeRO-2 strategy (FSDP2 / Megatron not supported by this patch).

Skip it when:
- Sequences are already roughly uniform length (no padding to remove).
- You're at the absolute memory ceiling (this patch costs ~7 MB per chunk + a small allocator-fragmentation tax — see "Memory budget" below).

## How to enable

Add three keys under `actor_train` in your yaml:

```yaml
actor_train:
  use_inner_dynamic_batching_in_train: true
  sequence_length_round_in_train: 64    # round chunk's max len up to multiple of N (default 4)
  # max_tokens_per_microbatch_in_train: <unused by this code path — reuse-only field>
```

That's it. The flag is honored only by `deepspeed_strategy.train_step` (other strategies ignore it). No code changes, no environment-variable changes.

### Recommended `sequence_length_round_in_train` values
| Setting | Backend | Notes |
|---|---|---|
| `4` (default) | any | safe, zero kernel-tail penalty on most kernels |
| `64` | bf16 + flash_attn_2 | aligns to fa2's inner tile (head_dim=128 / 256 path), removes "tail" kernel launch — recommended for Qwen3 / Qwen3.5 |
| `128` | aggressive bf16 | only if you've profiled; usually no further gain |

### Per-device batch size constraint

Inner dynbatch yields `gradient_accumulation_steps` chunks per mini-batch. The strategy asserts `mini_steps == gradient_accumulation_steps` (because it preserves DeepSpeed's internal grad-accum counter). This is automatic — the assertion will fail loudly if you've mis-set `per_device_train_batch_size × gradient_accumulation_steps` so that the iterator yields a different `mini_steps`.

In practice: **set `per_device_train_batch_size` and `gradient_accumulation_steps` so that `per_device × ga` is small enough that the largest sample in the mini-batch fits in memory at chunk-of-`per_device` size**.

## Memory budget — when does this fit?

Inner dynbatch costs **~7 MB per chunk** of pre-forward allocation (advanced indexing for the sorted reorder), plus a small allocator-fragmentation tax because chunks have varying shapes.

| seq_length | typical baseline peak (4B + LoRA + ZeRO-2) | margin to A6000 (47.4 GB) | inner dynbatch fits? |
|---:|---:|---:|:---:|
| 1024 | ~38 GB | ~9 GB | yes (trivially) |
| 1536 (bs=4 ga=3) | ~46.8 GB | ~600 MB | **no** — the largest chunk still hits the same peak as baseline, no headroom for the reorder allocation |
| 1536 (bs=2 ga=6) | ~42 GB | ~5 GB | yes |
| 2048 | OOM at any bs | — | no |

Rule of thumb: **drop `per_device_train_batch_size` to halve the peak** before turning this on at high seq lengths. The TPS lost from smaller per_device is more than recovered by the padding removal — measured ~+21% net TPS at 4B seq=1536 with bs=2 ga=6 vs baseline bs=4 ga=3.

## Worked example: Qwen3.5-4B at seq=1536

```yaml
sequence_length: 1536
rollout_batch_size: 256
train_env_manager:
  num_env_groups: 32
  group_size: 8

actor_train:
  device_mapping: "[0,1,2]"             # DP=3
  training_args:
    per_device_train_batch_size: 2      # halved from default 4
    gradient_accumulation_steps: 22     # eff_batch = 2 × 22 × 3 = 132
  model_args:
    flash_attn: fa2
    disable_gradient_checkpointing: false   # gc=on (mandatory at 4B)
    dtype: bf16
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero2}
  use_inner_dynamic_batching_in_train: true
  sequence_length_round_in_train: 64
```

Measured (5-step bench on 4× A6000):
- baseline (bs=4 ga=3, no inner dynbatch): 456 TPS, 6.3 min/step
- this config (bs=2 ga=6 + inner dynbatch): **550 TPS, 5.0 min/step (+21%)**

## How it works internally

| File | Function | Role |
|---|---|---|
| `roll/configs/worker_config.py` | field `use_inner_dynamic_batching_in_train` | the gate flag (default `False`) |
| `roll/utils/dynamic_batching.py` | `split_mini_batch_sorted_chunks_narrowed` | generator: yields one sorted+narrowed chunk at a time |
| `roll/distributed/strategy/deepspeed_strategy.py` | `train_step` (head) | branches to the helper when the flag is on |

The strategy's loop body (`backward → step`) is unchanged — DS's internal `gradient_accumulation_steps` counter still drives `optimizer.step()` exactly once per mini-batch, because `num_microbatches == gradient_accumulation_steps` always.

## What this patch does NOT do

- **No across-mini-batch reordering.** Each mini-batch is sorted independently. GRPO groups stay together because `rollout_batch_size == num_env_groups × group_size` ensures each mini-batch contains complete groups (this is asserted in `agentic_config.py`).
- **No reduction in optimizer step count.** Same `effective_batch = per_device × ga × DP` semantics as fixed-chunk training.
- **No support for variable `num_microbatches`.** If you want token-budget packing (e.g. "fit ≤ 6144 tokens per micro-batch"), use ROLL's full-batch `use_dynamic_batching_in_train` — but it currently only works with the Megatron strategy, not DeepSpeed.
- **No FSDP2 or Megatron support.** This patch is DeepSpeed-only. Megatron has its own dynbatch path (`use_dynamic_batching_in_train`).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OOM at modeling_qwen3_5.py linear-attention forward` | Baseline already at memory ceiling; inner dynbatch's ~7 MB overhead tips it over | Drop `per_device_train_batch_size` and double `gradient_accumulation_steps` to keep `eff_batch` constant |
| `AssertionError: expected mini_steps == gradient_accumulation_steps` | Your mini-batch arithmetic doesn't match | Ensure `rollout_batch_size / DP / per_device_train_batch_size == gradient_accumulation_steps × N` for some integer N |
| `TypeError: can't convert cuda:0 device type tensor to numpy` | Stale build (this was a pre-release bug in `dynamic_batching.py`) | Pull latest; the helper now uses CPU indices |
| TPS no improvement at all | Sequences in your workload are already uniform | Inner dynbatch can only remove padding; if there's no padding, no gains |

## See also

- Round J results in [`benchmark_grpo_sweep/A6000x4_qwen3p5_4b/README.md`](./benchmark_grpo_sweep/A6000x4_qwen3p5_4b/README.md) — full attribution of why bs=2 was needed at 4B.
- `agent_kuhn_poker_fsp_4b.yaml` — production config that uses this feature.
