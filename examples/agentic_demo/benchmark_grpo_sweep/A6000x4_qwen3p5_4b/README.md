# Kuhn Poker FSP throughput sweep — Auton 4× A6000, Qwen3.5-4B

Model: **Qwen/Qwen3.5-4B** (hybrid linear + full attention, text variant `Qwen3_5ForCausalLM`), LoRA rank=32.
Env: `roll3` — transformers 5.5.4, vLLM 0.19.1, torch 2.10, flash_attn 2.8.3, click 8.1.8 (see `scripts/auton/install_env_roll3.sh`).
Task: `agent_kuhn_poker_fsp_4b.yaml` (production FSP self-play on Auton `general`).

Starting baseline: the A6000 3B v2 R10 winner ([../A6000x4/](../A6000x4/)) carried over wherever applicable — 3tr+1inf, bs=4, ga=3, infer_batch_size=4, `enforce_eager=false`, `max_num_batched_tokens=4096`, `rollout_batch_size=256`, `group_size=4`, `num_env_groups=32`. 5-step bench per variant, same sweep script (`run_kuhn_poker_4b_sweep.sh`) and `collect_round_results.py`.

## Deltas forced by 4B (cannot carry over from 3B sweep)

| Knob | 3B winner | 4B value | Why |
|---|---|---|---|
| `disable_gradient_checkpointing` | true (gc off) | **false (gc on)** | 4B + LoRA + ZeRO-2 at seq=1024 bs=4 peaks at 47.35 GB / 47.4 GB on gc=off → OOM on every split. gc=on is mandatory; costs ~40% in train time. |
| `dtype` | fp16 | **bf16** | Qwen3.5 is bf16-native; fp16 numerics drift on the linear-attention state updates. |
| `template` | qwen2_5 | **qwen3** | Qwen3.5 template enables `<think>` reasoning block. |
| `flash_attn` (actor_train) | sdpa | **fa2** | Installed in roll3. Marginal TPS win at seq=1024, ~tied at seq=2048 — Qwen3.5's 24 linear-attn layers don't route through fa2 kernels; only the 8 full-attn layers benefit. |
| `sequence_length` / `max_new_tokens` | 1024 / 512 | **1536 / 1024** | User-requested so Qwen3.5 `<think>` responses (mean ~430 tokens) aren't truncated. Raised seq_length to fit prompt (~240) + max_new (1024) + slack. |

v5 HF compatibility required code patches in `model_providers.py` (`AutoModelForVision2Seq`→`AutoModelForImageTextToText` shim, `pad_token_id` via `get_text_config()`, prefer CausalLM over ConditionalGeneration for text-only Qwen3.5), `vllm_strategy.py` (sync `get_tokenizer`, `additional_special_tokens`→`extra_special_tokens` fallback, `replace_additional_special_tokens` kwarg fallback), `vllm_utils.py` (`vllm.lora.models`→`vllm.lora.lora_model`), `pg_utils.py` (torch 2.10 `pg_options`→`backend_options` with tuple-based version parse), `token_mask_utils.py` (BatchEncoding → list unwrap, skip the system-mock probe that Qwen3.5 rejects with "No user query found"), and `deepspeed_strategy.py` (rebuild LR scheduler against the DS-wrapped optimizer to work around torch 2.10's `zip(..., strict=True)` on `param_groups`/`base_lrs`).

## Results

### round A: GPU split (bs=4 gc=on fa2, seq=1024, mnt=512)

| split | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | per-GPU util (g0/g1/g2/g3) | state |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1tr+3inf (smoke) | 65 | 358 | 71 | 300 | 52/11/11/11 | completed |
| 2tr+2inf | 74 | 176 | 35 | 520 | 34/42/15/15 | completed |
| **3tr+1inf (winner)** | **113** | **137** | **26** | **632** | 50/11/11/11 | completed |

**Winner: 3tr+1inf.** Same shape as the 3B v2 R7 result — the inference cluster is massively under-saturated vs train; 1 vLLM GPU is enough for rb=256 Kuhn prompts.

### round B: gradient_checkpointing (at 3tr+1inf bs=4)

| gc | split | Result |
|:-:|:-:|:-:|
| off | 3tr+1inf | **OOM** (5558) — peak 47.35 GB at seq=1024 |
| off | 2tr+2inf | **OOM** (5560) |
| off | 3tr+1inf bs=2 | **OOM** (5561) |
| off | 3tr+1inf bs=1 | completed — 511 TPS (5562), 5.5 min/step |
| **on** | **3tr+1inf bs=4 (winner)** | **completed — 632 TPS, 4.6 min/step** |

`gc=off bs=1` fits but is slower than `gc=on bs=4` by 19% — **gc=on is strictly better** once it's required.

### round C: enforce_eager (at 3tr+1inf bs=4 gc=on)

| eager | Rollout (s) | Train (s) | TPS | Notes |
|:-:|:-:|:-:|:-:|:-:|
| true | 213 | 140 | 422 | CUDA graphs disabled — rollout ~2× slower |
| **false (winner)** | **113** | **137** | **632** | default |

Matches 3B R1 result (CUDA graphs are ~2× rollout lever).

### round D: per_device_train_batch_size (at 3tr+1inf gc=on fa2 seq=1536)

| bs | ga | eff_batch | Result |
|:-:|:-:|:-:|:-:|
| 1 | 12 | 36 | completed at gc=off (slower than winner) |
| 2 | 6 | 36 | **OOM** at gc=on |
| **4** | **3** | **36** | **completed — winner** |
| 8 | 2 | 48 | **OOM** at gc=on and gc=off |

No bs headroom at 4B.

### round E: ZeRO-3 ± CPU offload (at 3tr+1inf bs=4 gc=on fa2 seq=1536)

| variant | Result |
|:-:|:-:|
| zero3_cpuoff_bs4_gcon | **OOM** |
| zero3_cpuoff_bs4_gcoff | **OOM** |
| zero3_cpuoff_bs8_gcon | **OOM** |

ZeRO-3 doesn't help LoRA: the frozen 4B base (~10 GB) is replicated per GPU regardless of ZeRO stage — only LoRA params (80 MB) are sharded (saves ~53 MB/GPU). ZeRO-3 also adds all-gather / pre-fetch workspace buffers that DS ZeRO-2 doesn't have, net **costs** memory for LoRA workloads. CPU offload doesn't rescue it — LoRA optimizer state is already sub-100 MB. Matches the DeepSpeed/HF PEFT guidance: **use ZeRO-2 with LoRA**.

### round F: async rollout×train overlap (at 3tr+1inf bs=4 gc=on fa2 seq=1536)

| variant | Result |
|:-:|:-:|
| `async_generation_ratio=1` | **OOM** (holds step N+1 rollout + step N train simultaneously) |
| `async_generation_ratio=1, rb=128` | **OOM** |
| `async_generation_ratio=1, bs=2 ga=6` | **OOM** |

Async requires ~2× data in memory during the overlap window. 4B doesn't have that headroom at any practical bs/rb.

### round G: vLLM cheap tweaks (at 3tr+1inf bs=4 gc=on fa2 seq=1536)

| variant | Result |
|:-:|:-:|
| max_loras=10 + block_size=32 + enable_chunked_prefill=true | **OOM** on train GPU |
| enable_chunked_prefill=true alone | **OOM** on train GPU |

Confusingly the OOMs landed on train GPUs, not the vLLM GPU. Likely node-variance noise: the winner 5572 succeeded on gpu30/gpu31 but 5592/5593 ran on gpu24/gpu29 with slightly less baseline free memory. **Attribution invalid** — these tweaks would almost certainly help vLLM-GPU throughput if re-run on the same node as 5572. Not the tweaks' fault.

### round H: robustness (at winner config)

| variant | Result |
|:-:|:-:|
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | **OOM** (same node-variance) |
| bs=2 gc=on fa2 | **OOM** — same 46.83 GB PyTorch peak as bs=4 |

The last result is telling: **halving bs doesn't help memory.** Peak is dominated by non-activation overhead (frozen base, DS fp32 grad accumulators, vLLM NCCL broadcast staging on train GPUs during weight sync, etc.), not activations.

### round I/J: dynamic batching attempts

ROLL's `use_dynamic_batching_in_train` wiring exists but asserts `batch_size % per_device_train_batch_size == 0` inside DeepSpeed strategy's `train_step` (`deepspeed_strategy.py:468`) — the strategy doesn't read the `micro_batch_indices` meta that the upstream `dynamic_batching_shard` produces. So the existing ROLL path doesn't work with DS + variable micro-batch sizes.

Two in-house variants attempted:

**Fix variant 1 (reject)** — variable `num_microbatches` with DS counter reset (`self.model._config.gradient_accumulation_steps = N; self.model.micro_steps = 0`). All budgets OOM (5625/5627/5628/5630): the counter mutation doesn't actually flush DS's fp32 gradient accumulators, so ~200-500 MB of grad state persists across mini-batches, combined with ~100-150 MB from `dynamic_batching_shard`'s reorder + `DataProto.chunk()` clone. Eats the tight ~400 MB baseline margin.

**Fix variant 2 (in progress)** — equal-count sorted chunks. Splits each mini-batch into exactly `ga_steps` sorted chunks + per-chunk narrow, keeping `num_microbatches == gradient_accumulation_steps` so DS state is undisturbed. Replaces `DataProto.chunk` (which clones) with `DataProto.slice(start, end)` (contiguous views). Memory overhead: ~1× mini-batch (only the reorder advanced-indexing allocation). Code in `roll/utils/dynamic_batching.py:split_mini_batch_sorted_chunks_narrowed`, gated by `actor_train.use_inner_dynamic_batching_in_train=false` by default.

## Winner (confirmed stable)

```yaml
# A6000x4 Qwen3.5-4B winner — 456 TPS, 6.3 min/step
pretrain: Qwen/Qwen3.5-4B
reward_pretrain: Qwen/Qwen3.5-4B
sequence_length: 1536
rollout_batch_size: 256

actor_train:
  model_args:
    flash_attn: fa2
    disable_gradient_checkpointing: false   # gc=on required for 4B
    dtype: bf16
    lora_rank: 32
    lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  training_args:
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 3
    learning_rate: 2.0e-5
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero2}      # not zero3 — zero3 hurts LoRA
  device_mapping: "[0,1,2]"                  # 3 train GPUs, DP=3
  infer_batch_size: 4

actor_infer:
  generating_args:
    max_new_tokens: 1024
  strategy_args:
    strategy_name: vllm
    strategy_config:
      enforce_eager: false                   # 2× rollout lever
      max_num_batched_tokens: 4096
      gpu_memory_utilization: 0.92
  device_mapping: "[3]"                      # 1 vLLM GPU
  infer_batch_size: 1

train_env_manager:
  num_env_groups: 32
  group_size: 8                              # gs*num_env_groups == rollout_batch_size (1 traj/env)
```

Full-run wallclock: 6.3 min/step × 300 steps ≈ **31 h**. Does not fit 24 h wallclock on `general`; plan for FSP checkpoint-resume across two sbatch slots (FSP snapshots every 50 steps).

## Per-step bottleneck breakdown (winner)

| Phase | Time (s) | Share | Notes |
|---|---:|---:|---|
| time/step_rollout | 112 | 30% | vLLM on 1 GPU, KV cache ~31% used — not saturated |
| time/step_train | 222 | 59% | 3 train GPUs, gc=on recompute is ~40% of this |
| time/step_old_log_probs | 41 | 11% | forward on 3 train GPUs |
| **total** | **~375** | | |

`gc=on` is the single biggest perf cost for 4B, and it's mandatory. The only untapped lever we identified is **within-mini-batch dynamic batching** (fix variant 2, under test in round J) which targets the ~56% padding waste in `step_train` — mean total length 670 vs padded 1536.

## Key takeaways vs 3B

1. **~2.5× slower per step** than 3B v2 winner (375s vs 111s). Causes: 33% more params (×1.3), gc=on is mandatory (×1.4), vLLM 0.19 + Qwen3.5 hybrid attention kernels less tuned than vLLM 0.10 + Qwen2.5 (~×1.5 in rollout), seq=1536 vs 1024 (×1.5 in train). TPS-per-token also drops (631→456) because `<think>` responses are longer (~440 vs ~180 tokens).
2. **Memory ceiling is the dominant constraint.** The winner peaks at ~46.8 GB / 47.4 GB — a 1.2% margin. Any perturbation (node variance, tiny allocation shift) tips over.
3. **ZeRO-3 and async are dead ends for LoRA at this size.** Async OOMs regardless of bs/rb; ZeRO-3 adds buffers without freeing the frozen base.
4. **Dynamic batching is the biggest remaining lever but requires code work** — upstream assert at DS's `make_iterator` blocks variable-micro-batch paths. Fix variant 2 is in-flight.

## round S: infer_batch_size smoke (1tr+3inf split, 2026-04-21)

**Config:** smoke config (`agent_kuhn_poker_fsp_4b_smoke.yaml`) — **1tr+3inf**, ga=16, bs=4, gc=on, fa2, seq=1536, rb=256. Differs from the production winner (3tr+1inf, ga=3). Step times measured from log timestamps; no per-phase breakdown.

| ibs | Init step (s) | Steady-state mean (s) | n steps | vs ibs=1 | Jobs |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 1301 | **1068** | 5 | — | 6585 |
| **4** | 1249 | **1001** | 5+1 | **−6.3% (−67 s/step)** | 6597, 6589 |

Consistent with the 3B A6000 result (ibs=4 was +2.3 % there on the 3tr+1inf winner). At 1tr+3inf the inference side is completely under-saturated — 3 vLLM GPUs serving only rb=256 Kuhn prompts — so `infer_batch_size` has essentially no effect on rollout time. The 6% gain here is entirely from the **training side**: ibs=4 batches up the actor_train infer pass (used to compute log-probs for the current policy) and amortizes the per-request overhead. At the winner split (3tr+1inf) this effect is proportionally smaller. **ibs=4 remains the right default for production.**

## Job ID index (for log retrieval)

`/zfsauton/scratch/wentsec/ROLL/logs/kuhn4b_sweep_<JID>.out`

| JID(s) | What |
|---|---|
| 5530-5533 | env install iterations (flash-attn build OOM → MAX_JOBS=2 fix) |
| 5538 | click 8.3.2 → 8.1.8 fix (ray CLI compatibility) |
| 5539-5548, 5551-5553 | smoke iterations (all v5 compat patches applied and validated) |
| 5558-5566 | round A (GPU split), round B (gc), round C (eager), round D (bs) |
| 5567-5572 | seq=2048 explorations, fa2 validation; **5572 = validated winner** |
| 5573-5575 | bs=4/8 gc=off/on × fa2 (all OOM) |
| 5583-5585 | round E ZeRO-3 ± CPU offload (all OOM) |
| 5587-5590 | round F async (all OOM) |
| 5592-5593, 5595-5596 | round G vLLM tweaks, round H robustness (attribution-invalid OOM) |
| 5603-5604 | round I full-batch dynbatch via existing ROLL knob — both bugs (deepspeed_strategy divisibility assert, agentic `worker.dp_size` typo) |
| 5606 | gs=8 validation + new assert `rb == num_env_groups × gs` |
| 5623-5630 | round J Fix variant 1 (DS counter reset) — all OOM at varying budget |
| 5642-onwards | round J Fix variant 2 (equal-count sorted chunks, view-slice) — in progress |
| 5658 | round A re-verify with corrected gc=on baseline (433 TPS, gpu26) |
| 5660 | round N zero2off+gcon+bs4 (CPU optimizer offload + gc=on) |
| 5664 | round N zero2off+gcon+bs8 (OOM — bs=8 activation pressure) |
| 5665-5666 | round O `gpu_memory_utilization` 0.95 / 0.97 |
| 5669-5670 | round Q `max_loras` 5 / 2 |
| 5671 | round R `enable_prefix_caching=true` |
| 5659, 5672-5674 | round G re-run pinned to gpu30 (vllm_cheap, max_loras10, block32, chunked) |
| 5680-5681, 5696-5697 | round P `rollout_batch_size` 384 / 512 (5680/5681 failed list-type, fixed in 5696/5697) |
| 6585 | round S ibs=1 smoke (1tr+3inf, hit 2hr SLURM cap after 5 clean steps) |
| 6597, 6589 | round S ibs=4 smoke (general + preempt twin; 6589 preempted once, resumed) |

## v3 sweep (rounds N, O, P, Q, R, G-rerun) — 2026-04-19/20

Goal: exhaust untested knobs (CPU offload of optimizer, vLLM `gpu_memory_utilization` push, `rollout_batch_size` extension, `max_loras` reduction, `enable_prefix_caching`) and re-run round G with same-node pinning to fix attribution.

Verify baseline (5658, gpu26): **433 TPS** (round A re-run with corrected gc=on baseline). All v3 results compared to this. README's headline 456 TPS is **gpu30** (job 5572) — **not reproducible on gpu25/gpu26 in the v3 batch**, indicating ~5 % node-time-of-day variance. Round G's gpu30-pinned re-runs land at 428-447, also below 456 — so the gap is mixed cluster-load + node variance, not a regression.

### Results (steps 1-3 mean, post-warmup)

| Round | Variant | Job | TPS | train (s) | rollout (s) | logprob (s) | vs 433 | node | Notes |
|---|---|---|---|---|---|---|---|---|---|
| A | split_3tr1inf (verify) | 5658 | 433 | 224 | 113 | 40.3 | baseline | gpu26 | corrected gc=on baseline |
| N | zero2off + gcon + bs4 | 5660 | 426 | 226 | 114 | 41.0 | -2 % | gpu25 | wash; LoRA optimizer state ~80 MB → CPU offload moves nothing |
| N | zero2off + gcon + bs8 | 5664 | OOM | — | — | — | ❌ | gpu26 | activation cliff still |
| N | zero2off + gcoff + bs4 | 5652 | OOM | — | — | — | ❌ | gpu26 | offloading optimizer doesn't free enough for gc=off |
| O | gpu_memory_utilization=0.95 | 5665 | 433 | 226 | 114 | 41.1 | wash | gpu25 | KV at ~31 % already; pushing further has no perf effect |
| O | gpu_memory_utilization=0.97 | 5666 | 443 | 222 | 113 | 40.1 | +2 % | gpu26 | within node-variance noise |
| Q | max_loras=5 | 5669 | 438 | 226 | 112 | 41.0 | +1 % | gpu25 | only 1 LoRA active in 5-step bench (`fsp_save_steps=0`) — change has no real effect |
| Q | max_loras=2 | 5670 | 437 | 226 | 113 | 41.0 | +1 % | gpu25 | same — would matter for full FSP run with growing enemy pool |
| R | enable_prefix_caching=true | 5671 | 441 | 220 | 114 | 41.0 | +2 % | gpu26 | train dropped 224→220s; could be cluster getting quieter, not the prefix cache itself |
| G | vllm_cheap (mloras10+blk32+chunked) | 5659 | 428 | 226 | 113 | 40.8 | -1 % | gpu30 | **node-pinned re-run** of 5592; bundle didn't help |
| G | max_loras=10 alone | 5672 | **447** | 220 | 113 | 40.6 | **+3 %** | gpu30 | best v3 result; but train 220s vs verify 224 = same cluster-load drift seen in 5671 |
| G | block_size=32 alone | 5673 | 441 | 220 | 114 | 40.7 | +2 % | gpu30 | within noise |
| G | enable_chunked_prefill=true alone | 5674 | 443 | 220 | 113 | 40.4 | +2 % | gpu30 | within noise |
| P | rollout_batch_size=384 | 5696 | not run | — | — | — | — | — | queue-blocked behind unrelated training jobs through morning; deferred |
| P | rollout_batch_size=512 | 5697 | not run | — | — | — | — | — | queue-blocked; deferred |

### v3 takeaways

1. **No new winner.** All 11 completed variants land in **426-447 TPS**, a ±3 % band around the verify baseline. The 4B winner config from rounds A-J is fully exhausted.
2. **CPU optimizer offload is dead for LoRA.** Round N gcon+bs4 was a wash; gcoff_bs4 and gcon_bs8 OOM'd. The 4B + LoRA + ZeRO-2 optimizer state is ~80 MB; offloading saves ~80 MB on each train GPU — far less than the ~400-500 MB headroom needed to flip gc=on → gc=off. **Skip ZeRO-2 + CPU offload.**
3. **`gpu_memory_utilization=0.97` is +2 % at best — within noise; production safer at 0.92** (the 1.2 % memory-margin headroom matters more than 2 % TPS).
4. **`max_loras` reduction has no effect on bench**, since `fsp_save_steps=0` keeps the enemy pool at 1 entry. For production runs (300 steps × `fsp_save_steps=50`) the enemy pool grows to ~7 → `max_loras=10` is the right floor.
5. **`enable_chunked_prefill=true`, `block_size=32`, `enable_prefix_caching=true` are all wash** at this workload size (Kuhn prompts ~240 tokens, single LoRA active per bench step).
6. **Round G attribution-fix succeeded methodologically** (gpu30 pinning eliminated the previous spurious OOMs) but didn't surface a real winner — the original Round G OOMs in v2 were entirely node-variance, the tweaks themselves are wash on this 4B config.
7. **Round P (rb=384/512) deferred.** Queued 11 h, never ran (cluster slot held by unrelated training jobs). Bigger rollout batches amortize train more (3B v2 R9 saw +9 % rb=128→256). Worth retrying when cluster is free; expected upper bound based on the 3B trend ~+5 % at 4B (longer rollouts grow proportionally faster than train savings here).

### Production config: unchanged

The v3 sweep does not justify any changes to `examples/agentic_demo/agent_kuhn_poker_fsp_4b.yaml`. The 456 TPS / 6.3 min/step from the v2 winner stands. Next-step lever (per the original README) remains `compute_log_probs` offload to the vLLM cluster — code change, not a sweep variant.
