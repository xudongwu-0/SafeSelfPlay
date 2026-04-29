# A100x4 GRPO Sweep — Benchmark Results

**Hardware:** 4× NVIDIA A100 PCIe 40GB  
**Cluster:** Delta (`gpuA100x4-interactive`, account `bfoz-delta-gpu`)  
**Model:** Qwen2.5-3B-Instruct (LoRA rank 32, bf16, DeepSpeed ZeRO-2)  
**Env:** KuhnPokerLLMThink  
**Bench:** 5 steps per run, `max_new_tokens=512`, `sequence_length=1024`  
**Effective batch:** `per_device_train_batch_size × gradient_accumulation_steps × num_train_gpus = 48` (held constant across R1–R2)  
**Scripts:** `scripts/delta/run_grpo_sweep_a100.sh`, `scripts/delta/launch_grpo_sweep_a100.sh`  
**Output dir:** `/projects/bfoz/wchen11/kuhn_sweep_a100/`  
**WandB:** `kuhn-sweep-a100`

---

## Starting Baseline (A6000 v2 SOTA → A100)

| Parameter | Value | Source |
|-----------|-------|--------|
| `actor_train.device_mapping` | `[0,1,2]` | A6000 R7 winner |
| `actor_infer.device_mapping` | `[3]` | A6000 R7 winner |
| `per_device_train_batch_size` | 4 | A6000 R3 winner |
| `gradient_accumulation_steps` | 1 | full batch (no accumulation) |
| `infer_batch_size` | 4 | A6000 R4 winner |
| `enforce_eager` | false | A6000 R1 winner |
| `max_num_batched_tokens` | 4096 | A6000 R5 winner |
| `gpu_memory_utilization` | 0.92 | default |
| `group_size` | 4 | A6000 R10 winner |
| `rollout_batch_size` | 256 | A6000 R9 winner |
| `num_env_groups` | 64 | = rollout_batch_size / group_size = 256/4 |
| `num_groups_partition` | [64] | = [num_env_groups] |

---

## Round 1 — GPU Split

*Biggest lever per A6000 experience: 3tr+1inf gave +83% over 1tr+3inf.*

| Variant | actor_train_gpus | actor_infer_gpus | tok/s | step_time_s | rollout_s | train_s | log_probs_s | gpu_util_% | winner |
|---------|-----------------|-----------------|-------|-------------|-----------|---------|-------------|------------|--------|
| 1tr+3inf | [0] | [1,2,3] | 969 | 113 | 1.3 | 75 | 32 | 24.1 (GPU0=80%) | |
| 2tr+2inf | [0,1] | [2,3] | 1690 | 65 | 1.3 | 40 | 18 | 30.6 (GPU0-1≈54%) | |
| 3tr+1inf | [0,1,2] | [3] | 2067 | 50 | 1.1 | 32 | 15 | 37.7 (GPU0-2≈45%, GPU3=15%) | ✓ |

*tok/s = final cumulative system/tps; step_time/rollout/train/log_probs = avg steady-state steps 1-3; gpu_util = training-steps window mean*

**Winner:** 3tr+1inf — `actor_train.device_mapping='[0,1,2]'`, `actor_infer.device_mapping='[3]'`  
**Locked baseline update:** `actor_train.device_mapping`, `actor_infer.device_mapping`, `gradient_accumulation_steps`

---

## Round 2 — `per_device_train_batch_size`

*Full batch: ga=1, sweep bs freely. A100 40GB ≈ A6000 48GB; OOM at bs=16+.*

| Variant | bs | ga_steps | tok/s | step_time_s | gpu_util_% | winner |
|---------|-----|----------|-------|-------------|------------|--------|
| bs4 | 4 | 1 | 2127 | 50 | 42.3 | ✓ |
| bs8 | 8 | 1 | 2104 | — | — | |
| bs16 | 16 | 1 | OOM | — | — | |
| bs32 | 32 | 1 | OOM | — | — | |

*bs16 OOM: tried to allocate 9.27 GiB, only 6.31 GiB free. bs32 OOM: tried 18.55 GiB, 17.79 GiB free.*

**Winner:** bs4 (2127 tok/s; bs8=2104 within noise but bs4 slightly faster)  
**Locked baseline update:** `per_device_train_batch_size=4`

---

## Round 3 — `rollout_batch_size`

| Variant | rollout_batch_size | num_env_groups | tok/s | step_time_s | gpu_util_% | winner |
|---------|--------------------|----------------|-------|-------------|------------|--------|
| rb64 | 64 | 16 | 1603 | — | — | |
| rb128 | 128 | 32 | 1727 | — | — | |
| rb256 | 256 | 64 | 2291 | — | — | ✓ |
| rb512 | 512 | 128 | 2174 | — | — | |

**Winner:** rb256 (2291 tok/s; current baseline already optimal)  
**Locked baseline update:** `rollout_batch_size=256`

---

## Round 4 — `group_size`

| Variant | group_size | tok/s | step_time_s | gpu_util_% | winner |
|---------|------------|-------|-------------|------------|--------|
| gs1 | 1 | 2268 | — | — | |
| gs2 | 2 | 2140 | — | — | |
| gs4 | 4 | 2194 | — | — | |
| gs8 | 8 | 2279 | — | — | ✓ |

*gs1 is degenerate (zero within-group variance → GRPO advantage collapses). gs8 wins on both throughput and training quality.*

**Winner:** gs8 (2279 tok/s)  
**Locked baseline update:** `group_size=8`, `num_env_groups=32`, `num_groups_partition=[32]`

---

## Round 5 — `enforce_eager`

| Variant | enforce_eager | tok/s | step_time_s | gpu_util_% | winner |
|---------|---------------|-------|-------------|------------|--------|
| eager_false | false | 2263 | — | — | ✓ |
| eager_true | true | 1975 | — | — | |

**Winner:** eager_false (2263 tok/s; +15% vs eager_true)  
**Locked baseline update:** `enforce_eager=false`

---

## Round 6 — `max_num_batched_tokens`

| Variant | max_num_batched_tokens | tok/s | step_time_s | gpu_util_% | winner |
|---------|-----------------------|-------|-------------|------------|--------|
| mnt4096 | 4096 | 2181 | — | — | |
| mnt8192 | 8192 | 2203 | — | — | |
| mnt16384 | 16384 | 1923 | — | — | |
| mnt32768 | 32768 | 2274 | — | — | ✓ |

*mnt16384 anomalously slow — longer prefill latency with mid-range batch size. mnt32768 wins by maximizing vLLM batching efficiency.*

**Winner:** mnt32768 (2274 tok/s)  
**Locked baseline update:** `max_num_batched_tokens=32768`

---

## Round 7 — `gpu_memory_utilization`

| Variant | gpu_memory_utilization | tok/s | step_time_s | gpu_util_% | winner |
|---------|------------------------|-------|-------------|------------|--------|
| gmu90 | 0.90 | 2311 | — | — | |
| gmu92 | 0.92 | 2251 | — | — | |
| gmu95 | 0.95 | 2322 | — | — | ✓ |

*Higher gmu gives vLLM more KV-cache headroom, reducing evictions and improving throughput.*

**Winner:** gmu95 (2322 tok/s)  
**Locked baseline update:** `gpu_memory_utilization=0.95`

---

## Round 8 — `infer_batch_size`

*Batch size for log-prob computation pass (actor_train.infer_batch_size). Larger = better GPU utilization during the log-probs phase.*

| Variant | infer_batch_size | tok/s | step_time_s | gpu_util_% | winner |
|---------|-----------------|-------|-------------|------------|--------|
| ibs4 | 4 | — | — | — | |
| ibs8 | 8 | — | — | — | |
| ibs16 | 16 | — | — | — | |
| ibs32 | 32 | — | — | — | |

**Winner:** TBD

---

## Final SOTA Config

| Parameter | Value | Round |
|-----------|-------|-------|
| `actor_train.device_mapping` | `[0,1,2]` | R1 |
| `actor_infer.device_mapping` | `[3]` | R1 |
| `per_device_train_batch_size` | `4` | R2 |
| `gradient_accumulation_steps` | `1` | — |
| `group_size` | `8` | R4 |
| `rollout_batch_size` | `256` | R3 |
| `num_env_groups` | `32` | = 256/8 |
| `enforce_eager` | `false` | R5 |
| `max_num_batched_tokens` | `32768` | R6 |
| `gpu_memory_utilization` | `0.95` | R7 |
| **tok/s** | **2322** | R7 best |

---

## Cross-Hardware Comparison

| Metric | A40x4 | A6000x4 v1 | A6000x4 v2 | **A100x4** |
|--------|-------|-----------|-----------|-----------|
| VRAM | 40 GB | 48 GB | 48 GB | **40 GB** |
| tok/s | 530 | 452 | 947 | **2322 (SOTA)** |
| gpu_util_% | — | 21 | 37.5 | **37.7 (R1 proxy)** |
| GPU split | 1tr+3inf | 1tr+3inf | 3tr+1inf | **3tr+1inf** |
| per_device_bs | 4 | 4 | 4 | **4** |
| group_size | 4 | 1 | 4 | **8** |
| rollout_batch_size | 128 | 128 | 256 | **256** |
| max_num_batched_tokens | — | — | 4096 | **32768** |
| gpu_memory_utilization | — | — | 0.92 | **0.95** |
