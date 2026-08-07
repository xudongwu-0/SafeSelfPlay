# Dynamic-SFT Dual-LoRA r64

This experiment preserves the proven sequential A200 -> D200 configuration and
changes only the trainable parameter scope from two full 8B policies to two
independent LoRA adapters.

## Configuration

| Setting | Value |
|---|---:|
| Base model | `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` |
| Attacker start | base model + fresh LoRA |
| Defender start | base model + fresh LoRA |
| LoRA | rank 64, alpha 64 |
| Target modules | all attention and MLP projections |
| Schedule | A200, then D200 against frozen A200 |
| Prompt distribution | 50% official harmful, 50% official benign |
| Rollout batch | 128 |
| Micro rollout batch | 8 |
| Train batch | 32 |
| Micro train batch | 8 |
| Learning rate | `1e-6`, cosine with minimum LR |
| Reward | upstream Self-RedTeam `general_sum` |
| KL loss coefficient | 0 |
| KL diagnostic | enabled against each role's base start |
| Attacker auxiliary SFT | coefficient 1 through step 30, then 0 |
| Defender auxiliary SFT | coefficient 1 through step 10, then 0 |
| GPU default | 4x H200 |

Attacker and defender checkpoints are independent. D starts from the base model,
not from A weights; A200 is loaded only as D's frozen opponent.

## Launch

```bash
./run_dynamic_sft_duallora_r64_h200x4.sh
```

The launcher is detached, so the Modal coordinator continues after the local
SSH session disconnects. Override settings with environment variables only when
running an explicit ablation, for example `STEPS_PER_ROLE=50`.

The existing full-parameter and strict rank-32 launchers remain unchanged.
