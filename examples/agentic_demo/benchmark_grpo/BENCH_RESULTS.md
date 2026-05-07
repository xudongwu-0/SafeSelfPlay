# Gradient Accumulation Steps Sweep

Sweep `gradient_accumulation_steps` (1, 2, 4, 8, 16) to maximize tok/s by balancing training batch size vs step count.

**Setup**: Qwen2.5-3B-Instruct, LoRA rank 32, `rollout_batch_size=64`, `num_env_groups=32`, 4x A40 (1 train + 3 infer), vLLM inference, 2 steps per run.

**Fixed from env sweep**: `num_env_groups=32` (optimal), rollout ~53s/step.

## Results

Step 1 (steady-state) where available, step 0 otherwise (marked with \*).

| grad_accum | train_iters | rollout (s) | train (s) | log_probs (s) | tps | step total (s) | notes |
|---|---|---|---|---|---|---|---|
| 1 | 64 | | | | | | |
| 2 | 32 | | | | | | |
| 4 | 16 | | | | | | |
| 8 | 8 | | | | | | |
| 16 | 4 | | | | | | |

`train_iters` = `rollout_batch_size / (per_device_train_batch_size * grad_accum)` = `64 / (1 * grad_accum)`

## Analysis

<!-- Fill after jobs complete -->

## Recommended Config

```yaml
actor_train:
  training_args:
    gradient_accumulation_steps: TBD
```

## Jobs

| grad_accum | job_id | output |
|---|---|---|
| 1 | 17437245 | `/projects/bfoz/wchen11/kuhn_grad_accum_sweep/ga1/` |
| 2 | 17437246 | `/projects/bfoz/wchen11/kuhn_grad_accum_sweep/ga2/` |
| 4 | 17437247 | `/projects/bfoz/wchen11/kuhn_grad_accum_sweep/ga4/` |
| 8 | 17437248 | `/projects/bfoz/wchen11/kuhn_grad_accum_sweep/ga8/` |
| 16 | 17437249 | `/projects/bfoz/wchen11/kuhn_grad_accum_sweep/ga16/` |
