# Kuhn Poker FSP throughput sweep — Auton 4× A6000

Model: Qwen2.5-3B-Instruct, LoRA rank=32, **4× A6000 (48 GB)** on Auton `general` partition.
Task: `agent_kuhn_poker_fsp_train.yaml` (production FSP self-play).
Method: coordinate descent — one hyperparameter per round, lock the round's winner into the next round. Round-1 baseline = A40 winners (see [../A40x4/](../A40x4/)).

5 steps per run; per-step `system/tps` from wandb, GPU utilization from background `nvidia-smi -l 2` sampler. Tie rule: if a round's TPS spread is ≤ 2 %, treat as no clear winner and keep the A40 default.

## How runs are launched

Each variant is launched via [`scripts/auton/run_kuhn_poker_sweep.sh`](../../../scripts/auton/run_kuhn_poker_sweep.sh) which loads `agent_kuhn_poker_fsp_train.yaml` and applies Hydra CLI overrides:

1. **Bench knobs:** `max_steps=5`, `eval_steps=100`, `save_steps=10000`, `fsp_save_steps=0`.
2. **A40-winner baseline** (locked carry-overs into every round, updated as later rounds pick winners):
   - `sequence_length=1024`, `max_new_tokens=512`
   - `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` (eff_batch = 32, until round 3)
   - `enforce_eager=false` (until round 1 confirms)
   - `num_env_groups=32`, `group_size=4` (until round 2)
   - `infer_batch_size=1` (until round 4)
   - `max_num_batched_tokens=16384` (until round 5)
3. **Variant override:** the one swept hyperparameter (full mapping in the script's `case` block).

Yaml-inheritance was not used: `agent_kuhn_poker_fsp_train.yaml` lacks `# @package _global_`, which silently drops parent fields when included via `defaults:`. CLI overrides mutate the loaded config in place, sidestepping that.

After each run the GPU-util CSV is written to `${SWEEP_ROOT}/logs/gpu_util.csv` and `scripts/auton/post_sweep_to_wandb.py` pushes mean / max GPU util plus per-step `system/tps`, `time/step_*` summary fields back to the wandb run (project `self-play-debug`).

## Results

Per-round results are appended below by `scripts/auton/collect_round_results.py` after each round finishes.

### round1_eager: `enforce_eager` sweep

| enforce_eager | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | GPU util mean (%) | per-GPU util (g0/g1/g2/g3) | state |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| eager_true | 20.5 | 44.1 | 17.9 | 374.5 | 16.5 | 51/5/5/5 | finished |
| eager_false **(winner)** | 8.0 | 44.1 | 17.9 | 471.9 | 19.2 | 57/7/7/6 | finished |

**Winner: `eager_false` (471.9 tok/s, +20.6% spread).**

### round2_group_size: `group_size` sweep

| group_size | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | GPU util mean (%) | per-GPU util (g0/g1/g2/g3) | state |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| gs1 **(winner)** | 18.9 | 44.1 | 17.8 | 436.6 | 21.9 | 53/12/11/11 | finished |
| gs2 | 12.4 | 44.9 | 18.3 | 436.0 | 19.1 | 54/7/7/8 | finished |
| gs4 | 9.3 | 45.1 | 18.5 | 433.7 | 17.2 | 53/5/6/5 | finished |
| gs8 | 8.9 | 45.2 | 18.6 | 403.6 | 17.5 | 53/6/6/5 | finished |

**Winner: `gs1` (436.6 tok/s, +7.6% spread).**

### round3_train_bs: `per_device_train_batch_size` sweep

| per_device_train_batch_size | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | GPU util mean (%) | per-GPU util (g0/g1/g2/g3) | state |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| bs1 | 17.9 | 48.6 | 18.2 | 408.2 | 20.8 | 52/10/11/10 | finished |
| bs2 | 17.6 | 45.6 | 18.2 | 422.8 | 21.0 | 52/10/11/11 | finished |
| bs4 **(winner)** | 18.1 | 44.7 | 18.3 | 425.1 | 21.3 | 54/10/11/10 | finished |
| bs8 | — | — | — | — | — | —/—/—/— | failed |

**Winner: `bs4` (425.1 tok/s, +4.0% spread).**

### round4_infer_bs: `infer_batch_size` sweep

| infer_batch_size | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | GPU util mean (%) | per-GPU util (g0/g1/g2/g3) | state |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ibs1 | 18.3 | 44.9 | 18.3 | 430.3 | 21.1 | 52/11/11/11 | finished |
| ibs4 **(winner)** | 18.4 | 44.7 | 16.6 | 440.4 | 22.0 | 53/11/12/11 | finished |
| ibs8 | 17.0 | 44.8 | 16.3 | 433.3 | 21.4 | 52/11/11/11 | finished |
| ibs16 | 18.0 | 44.9 | 16.4 | 439.5 | 20.7 | 51/11/10/10 | finished |
| ibs32 | — | — | — | — | — | —/—/—/— | failed |

**Winner: `ibs4` (440.4 tok/s, +2.3% spread).**

### round5_mnt: `max_num_batched_tokens` sweep

| max_num_batched_tokens | Rollout (s) | Train (s) | Log probs (s) | TPS (tok/s) | GPU util mean (%) | per-GPU util (g0/g1/g2/g3) | state |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| mnt4096 **(winner)** | 18.6 | 44.7 | 16.5 | 452.0 | 20.7 | 50/11/11/11 | finished |
| mnt8192 | 18.4 | 44.7 | 16.6 | 436.4 | 20.1 | 49/11/10/10 | finished |
| mnt16384 | 18.3 | 44.7 | 16.5 | 434.7 | 21.3 | 52/12/11/11 | finished |
| mnt32768 | 19.1 | 44.7 | 16.6 | 435.7 | 20.9 | 49/12/11/12 | finished |

**Winner: `mnt4096` (452.0 tok/s, +3.8% spread).**

## Best-Combined Config Recommendation

```yaml
# A6000x4 winners after all 5 rounds (combine into agent_kuhn_poker_fsp_train.yaml)
sequence_length: 1024                           # locked from A40 winner
actor_train:
  training_args:
    per_device_train_batch_size: 4              # round 3 winner W3 (425 tok/s, +4% over bs1)
    gradient_accumulation_steps: 8              # tied so eff_batch = 32
  infer_batch_size: 4                           # round 4 winner W4 (440 tok/s, +2% over ibs1)
actor_infer:
  generating_args:
    max_new_tokens: 512                         # locked from A40 winner
  strategy_args:
    strategy_config:
      enforce_eager: false                      # round 1 winner W1 (472 vs 375 tok/s — biggest win)
      max_num_batched_tokens: 4096              # round 5 winner W5 (452 tok/s, +4% over 16384)
train_env_manager:
  num_env_groups: 32                            # locked from A40 winner
  group_size: 1                                 # round 2 winner W2 — but see caveat below
```

**Combined result (round 5 mnt4096 — uses all winners W1..W4 + best W5): 452.0 tok/s, ~21% mean GPU util.**

## Per-step bottleneck (4× A6000)

| Phase | Time / step (s, eager_false ibs4 baseline) | Share | Notes |
|---|---|---|---|
| `time/step_train` | 44.7 | ~46 % | 1 train GPU, deepspeed_zero2, bs=4 ga=8 — single-GPU compute bound |
| `time/step_old_log_probs_values` | 16.5 | ~17 % | 1 train GPU, log-probs forward pass — memory-bandwidth bound |
| `time/step_rollout` | 18.3 | ~19 % | 3 vLLM GPUs, well below saturation |
| other (model_update, advantages, eval) | ~17 | ~18 % | |
| **Total** | **~96.5** | | |

The training GPU (g0) sits at ~50 % util; the three inference GPUs (g1/g2/g3) at ~10 % util — same bottleneck the A40 sweep called out (`compute_log_probs` is single-train-GPU). **Real fix would be offloading log_probs forward to the vLLM cluster**, not more sweeping.

## Caveats / known interactions

- **Round 2 picked `group_size=1`** (436.6 vs gs4=433.7 — within noise, +0.7 %). All subsequent rounds inherited gs=1, which moved rollout time from ~9 s (gs4) up to ~18 s (gs1). Combined-winners TPS is **452**, below the round-1 best of **472** (eager_false @ gs=4). Coordinate descent's known weakness — within-round local winners can be globally suboptimal.
  - Practical takeaway: `group_size=4` is a safer choice for production FSP runs (matches A40 winner, gives more vLLM headroom).
- `bs8` and `ibs32` OOM on A6000 (48 GB) just like on A40 (40 GB) — the limit is the log-probs forward pass, not VRAM size per se.
- `ibs16` failed once on `EADDRINUSE` (vLLM port still held by previous job on the same node); a retry on the same script succeeded. If a sweep variant fails for an obviously-infra reason, retry it once before treating it as failed.

## Bonus: GPU split sweep (round 6)

Holding all 5 round-winners locked, swap GPU allocation from **1 train + 3 vLLM infer** (the FSP default) to **2 train + 2 infer**, with `gradient_accumulation_steps=4` so effective batch stays 32 (DP=2 × bs=4 × ga=4).

| split | Rollout (s) | Train (s) | Log probs (s) | TPS | util mean | per-GPU util |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1tr + 3inf (round 5 best) | 18.3 | 44.7 | 16.5 | 452.0 | 21 % | 50/11/11/11 |
| **2tr + 2inf** | 18.6 | **24.3** | **8.5** | **673.0** | **26.6 %** | 34/42/15/15 |

**+49 % TPS over the round-5 best.** The bottleneck wasn't vLLM throughput (rollout time barely changed going 3→2 inference GPUs — vLLM is not bandwidth-bound at this batch size for Kuhn prompts). It was the train GPU doing serial work alone. DeepSpeed ZeRO-2 with DP=2 cuts train + log_probs time roughly in half.

Inference GPUs (g2, g3) at ~15 % util suggest a third move worth trying: **3 train + 1 infer** — at the cost of losing vLLM tensor parallelism and likely slowing rollout, but freeing more compute for train.

## v2 sweep (rounds 7–10) — maximize TPS + util

Coordinate descent restart with GPU split as round 7's variable. For R7's `1tr+3inf` and `2tr+2inf` rows we reuse round 5's `mnt4096` and round 6's `train2_infer2` data points (same locked baseline).

### R7: GPU split

| split | Rollout (s) | Train (s) | Log probs (s) | Total (s) | TPS | Util mean | per-GPU |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1tr + 3inf (= R5 mnt4096) | 18.3 | 44.7 | 16.5 | 96.5 | 452.0 | 21 % | 50/11/11/11 |
| 2tr + 2inf (= R6 train2_infer2) | 18.6 | 24.3 | 8.5 | 80.8 | 673.0 | 26.6 % | 34/42/15/15 |
| **3tr + 1inf (winner)** | 20.8 | 21.0 | 6.6 | 78.0 | **826.8** | **29.9 %** | 30/35/37/18 |

**Winner: `3tr + 1inf` (826.8 tok/s, +23 % over 2tr+2inf, +83 % over the 1tr+3inf default).** With DP=3 we use `gradient_accumulation_steps=3` so effective batch ≈ 36 (was 32 at DP=1 — minor deviation, accepted). vLLM rollout barely slowed going 3 GPUs → 1 GPU (+13 %), which is the headline finding: the inference cluster was massively under-saturated. The 4th GPU is much more valuable doing training.

### R8: per_device_train_batch_size at 3tr+1inf (`ga=1`, eff varies)

| per_device_train_batch_size | ga | eff_batch | TPS | Util mean | state |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **bs=4 (= R7 baseline winner)** | 3 | 36 | **826.8** | 29.9 % | finished |
| bs=8 | 1 | 24 | — | — | OOM at activation alloc (45 GB used, +4.6 GB needed) |
| bs=16 | 1 | 48 | — | — | OOM (same root cause) |

**No new winner — bs=4 stays.** ZeRO-2 only shards optimizer state; per-GPU activations scale with `per_device_train_batch_size` and are the actual ceiling at 48 GB. To push bs higher would need ZeRO-3 or activation checkpointing (`disable_gradient_checkpointing: false`) — both add their own perf cost; out of scope here.

### R9: `rollout_batch_size` at 3tr+1inf, bs=4 ga=3

| rollout_batch_size | Rollout (s) | Train (s) | Log probs (s) | Total (s) | TPS | Util mean | per-GPU |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| rb64 | 11.6 | 10.8 | 3.6 | 51.9 | 730.3 | 25.1 % | — |
| rb128 | 20.9 | 21.7 | 6.7 | 81.5 | 792.7 | 27.7 % | — |
| **rb256 (winner)** | 39.5 | 42.7 | 12.7 | 129.7 | **868.5** | **36.3 %** | 38/42/42/23 |

**Winner: `rb=256` (868.5 tok/s, +18.9 % over rb64; +9.5 % util over rb128).** Bigger rollout batch → more PPO micro-steps → train phase amortizes better. Notably this is the first round where util jumped meaningfully (27 → 36 %); rb=512 might push further but is out of scope.

### R10: `group_size` re-validation at 3tr+1inf, bs=4 ga=3, rb=256

| group_size | Rollout (s) | Train (s) | Log probs (s) | Total (s) | TPS | Util mean | per-GPU |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| gs1 | 39.9 | 43.1 | 12.6 | 129.2 | 867.1 | 36.0 % | — |
| **gs4 (winner)** | 20.6 | 43.6 | 12.8 | 111.5 | **947.2** | **37.5 %** | — |
| gs8 | 19.9 | 44.3 | 13.4 | 113.8 | 938.1 | 35.6 % | 40/46/44/12 |

**Winner: `gs4` (947.2 tok/s, +9.2 % over gs1).** gs8 is within 1 % of gs4 — gs4 picked on tie. Bigger groups again help rollout (39.9s → 20.6s for gs1→gs4) by improving vLLM batching.

## v2 Best Config (combined R7–R10 winners)

```yaml
# A6000x4 v2 — overrides on top of agent_kuhn_poker_fsp_train.yaml
actor_train:
  device_mapping: "[0,1,2]"                       # R7: 3 train GPUs (DP=3)
  training_args:
    per_device_train_batch_size: 4                # R7/R8: bs=4 (bs8/16 OOM at DP=3)
    gradient_accumulation_steps: 3                # tied so eff_batch ≈ 36
  infer_batch_size: 4                             # R4 (carried, not re-swept)
actor_infer:
  device_mapping: "[3]"                           # R7: 1 vLLM GPU (was 3 — vLLM was massively under-saturated)
  generating_args:
    max_new_tokens: 512                           # locked
  strategy_args:
    strategy_config:
      enforce_eager: false                        # R1 (carried)
      max_num_batched_tokens: 4096                # R5 (carried)
rollout_batch_size: 256                           # R9 winner (+18.9 % TPS, +9.5 % util over rb=128)
sequence_length: 1024                             # locked
train_env_manager:
  num_env_groups: 32                              # locked
  group_size: 4                                   # R10 winner (+9.2 % over gs1; halves rollout time)
```

**Combined v2 result (round 10 gs4 — all winners locked together): 947.2 tok/s, 37.5 % mean GPU util.**

| Metric | v1 best (R5 mnt4096) | v2 best (R10 gs4) | Δ |
|:-:|:-:|:-:|:-:|
| TPS | 452.0 | **947.2** | **+110 %** |
| Util mean | 21.0 % | **37.5 %** | **+79 %** |
| total (s) | 96.5 | 111.5 | +15.5 % (more work per step) |
| rollout (s) | 18.3 | 20.6 | barely changed (1 vLLM GPU on rb=256 vs 3 GPUs on rb=128) |
| train (s) | 44.7 | 43.6 | flat (3-GPU DP × bs=4 ga=3 ≈ 1-GPU bs=4 ga=8) |
| log_probs (s) | 16.5 | 12.8 | -22 % |

## Key takeaways from v2

1. **The biggest single lever was the GPU split, by far** (R7: 1.83× TPS going 1tr+3inf → 3tr+1inf). The default 1-train + 3-vLLM allocation was wrong for this workload — vLLM was massively under-saturated. Hardware budget should follow the bottleneck, and the bottleneck is training, not inference.
2. **Bigger `rollout_batch_size` is the main util lever** (R9: 27% → 36% util going rb=128 → rb=256). vLLM scheduler + train loop both amortize overhead better with larger batches.
3. **Re-validating `group_size` mattered**: gs=4 beats gs=1 by 9% TPS once `rb=256` and 3tr+1inf are locked in (whereas in v1 with rb=128 / 1tr+3inf, gs values were within noise). Coordinate descent is path-dependent — one round's "tied" winner may stop being tied after the next round shifts the regime.
4. **Persistent bottleneck is still single-GPU compute_log_probs / train fwd+bwd.** 3 train GPUs at ~40 % util means we're ~bound by the per-GPU dense matmul rate at fp16, not by parallelism. Further gains would need either ZeRO-3 (allow bs > 4), activation checkpointing (allow bs > 4 at memory cost), async generation (overlap rollout with train), or moving log_probs to vLLM (out of scope, but still on the table).

## Async pipeline sanity test — Qwen3.5-4B, SOTA config, 5-step FSP smoke

Goal: validate the async pipeline (`async_generation_ratio: 1`) end-to-end on the **4B** model with **v2-SOTA hyperparams** and measure the wall-clock speedup vs sync (`async_generation_ratio: 0`). Also exercises the FSP cold-start path (`reset_lora_weights` + `force-sync flush` at `fsp_save_steps=2`).

Configs: [`agent_kuhn_poker_fsp_async_4b_smoke.yaml`](../../agent_kuhn_poker_fsp_async_4b_smoke.yaml) / [`agent_kuhn_poker_fsp_sync_4b_smoke.yaml`](../../agent_kuhn_poker_fsp_sync_4b_smoke.yaml). Both inherit from `agent_kuhn_poker_fsp_4b_smoke.yaml` and apply the v2 winner set, with one deviation: `num_env_groups=64, group_size=4` (vs sweep's 32/4) to satisfy the post-sweep `rollout_batch_size == num_env_groups * group_size` assertion at rb=256. 5 steps, fsp_save_steps=2, so FSP cold-start fires at steps 2 and 4.

### Per-step wall clock (4× A6000, 3 train + 1 vLLM)

| step | async (5648) | sync (5649) | speedup | notes |
|:-:|---:|---:|---:|---|
| 0 (warmup) | 701 s | 723 s | 1.03× | vLLM CUDA-graph compile + first rollout ramp |
| 1 | **203 s** | **321 s** | **1.58×** | clean steady state |
| 2 (FSP cold-start) | 312 s | 494 s | 1.58× | LoRA reset at step 2 |
| 3 (force-sync flush) | 337 s | 320 s | 0.95× | async drains pre-reset rollouts |
| 4 (FSP cold-start) | 421 s | 551 s | 1.31× | LoRA reset at step 4 |
| **total (0–4)** | **1974 s** | **2409 s** | **1.22×** | |
| **steady (1–4)** | **1273 s** | **1686 s** | **1.32×** | excludes warmup |

### Why 1.58× on clean steps, not 2×

Ceiling is bounded by `(rollout + train_serial) / max(rollout, train_serial)`. On step 1:

- sync step = 321 s (rollout + trainer serial)
- async step = 203 s (trainer serial only; rollout fully hidden)
- implied rollout ≈ **118 s**, trainer serial ≈ **203 s**
- ratio train/rollout ≈ **1.72×** — trainer still dominates, but by less than on 3B
- ceiling = 321 / 203 = **1.58×** — hit exactly on step 1

On 3B the same ratio was ~2.7× (trainer serial 56 s vs rollout 21 s from R10), giving a ceiling of ~1.23×. Moving to 4B slowed the rollout side (1-GPU vLLM on bigger model, rb=256, max_new=512) proportionally more than the train side, tightening the ratio and lifting the async ceiling.

### FSP cold-start path validated end-to-end

The `async_pipeline` + FSP cold-start + `flush_pending` sequence fired cleanly:

- step 2: `FSP cold_start: resetting training LoRA weights at step 2` → step finished with no deadlock
- step 3: `FSP force-sync: flushing pre-reset rollouts at step 3` → step finished within 5-6 min, pipeline continued normally

Arena eval (48 episodes, 4 snapshots, 4 per pair) completed on both runs.

### Cost of the flush

Step 3 is the only row where sync beats async (0.95×). This is the force-sync flush paying for correctness: rollouts generated before the LoRA reset are drained rather than trained on with mismatched weights. Over a full `fsp_save_steps` cycle (here: 2) this costs ~1 step of async advantage per cold-start; over longer cycles (e.g. `fsp_save_steps=50` as in production) the amortized cost is negligible.

### Takeaways

1. **Async pipeline works on 4B-SOTA** — no deadlock, correct FSP semantics, 1.32× steady-state speedup over a cycle that includes two cold-starts.
2. **Async speedup scales favorably with model size** — 4B gets 1.58× clean-step, 3B projects ~1.23×, because larger models shift more cost into rollout relative to train when rollout is serialized to 1 GPU.
3. **Raising `async_generation_ratio > 1` would not help here** — train is still the bottleneck on clean steps, and flush cost scales with queue depth. Stick with `async_generation_ratio=1`.
4. **Honest v2-SOTA baseline**: the non-SOTA smoke runs (rb=16, eager=true, mnt=16384, max_new=1024) gave a "larger looking" 1.44× speedup but at ~17× lower tok/s — not a real performance setting. Any speedup quote should cite the SOTA numbers.
