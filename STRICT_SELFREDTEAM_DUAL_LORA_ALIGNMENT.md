# Strict Self-RedTeam / Dual-LoRA Alignment

This launcher is a controlled comparison against the successful upstream
Self-RedTeam run. It keeps the upstream two-turn game, data, reward, online
SFT, optimizer settings, and generation settings. The intended method changes
are limited to independent role adapters and phased optimization.

## Canonical references

- Upstream repository: `mickelliu/selfplay-redteaming`
- Upstream training script: `../selfplay-redteaming/scripts/red_team_game_reinforce_8b.sh`
- Successful upstream W&B run: `51b2b7ca` (`selfredteam_official_repp_success`)
- Strict launcher: `run_strict_selfredteam_duallora_h200x4.sh`
- Implementation: `modal_upstream_selfredteam_role_lora.py`

## Role schedule

### Attacker phase (A50)

1. Start a fresh rank-32 attacker LoRA on the official Llama base.
2. Preserve the upstream 50/50 generated-versus-vanilla prompt marking.
3. Run the complete upstream two-turn protocol with the current attacker LoRA
   and the frozen base model as defender.
4. Compute the original game outcomes and metrics for both roles.
5. After tie removal, retain only attacker trajectories for advantage
   normalization and RL optimization. The defender receives no gradients.
6. Apply the same upstream online auxiliary-SFT batch to the trainable attacker
   policy, because the official shared policy receives that batch every step.
7. Save A50.

### Defender phase (D50)

1. Start a fresh rank-32 defender LoRA on the same official Llama base.
2. Preserve the same prompt marking and full two-turn protocol.
3. Generated rows use frozen A50 for the attacker turn. Vanilla rows use the
   original seed directly, exactly as upstream. The current defender LoRA
   produces every defender response.
4. Compute original game outcomes and metrics for both roles.
5. After tie removal, retain only defender trajectories for advantage
   normalization and RL optimization. A50 receives no gradients.
6. Apply the same upstream online auxiliary-SFT batch to the defender LoRA.
7. Save D50.

## Exactly aligned settings

| Setting | Upstream successful run | Strict dual-LoRA run |
|---|---:|---:|
| Base model | `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` | Same |
| Prompt data | upstream vanilla harmful + benign JSONL | Same files |
| Prompt probabilities | `0.5, 0.5` | Same |
| Generated fraction within each label | `0.5` | Same |
| Attacker/defender prompts | upstream hidden-CoT prompts | Same, unpatched |
| Turns per game | 2 | 2 |
| Reward | upstream `general_sum` | Same implementation and coefficients |
| Tie handling | remove ties | Same |
| Invalid/parsing handling | upstream behavior | Same; no rewrite filter |
| Advantage estimator | REINFORCE | Same |
| Reward normalization | enabled | Same |
| PPO actor loss coefficient | 1.0 | Same |
| Initial KL coefficient | 0.01 | Same |
| Reference policy | frozen base model | Same |
| Learning rate | `5e-7` | Same for both LoRAs |
| LR scheduler | cosine with 3% warmup, min LR = 0.1x | Same |
| Rollout batch | 128 | Same |
| Micro rollout batch | 8 | Same |
| Train batch | 32 | Same |
| Micro train batch | 8 | Same |
| Samples per prompt | 1 | Same |
| PPO epochs per rollout | 1 | Same |
| Prompt max length | 2048 | Same |
| Generation max length | 2048 | Same |
| Sampling | temperature 1.0, top-p 1.0 | Same |
| Precision / attention | BF16 / FlashAttention | Same |
| ZeRO stage | 3 | Same |
| Adam offload | disabled | Same |
| Gradient checkpointing | enabled, non-reentrant | Same |
| Packing | enabled | Same |
| Actor/ref/vLLM layout | 4 GPUs, colocated, 4 TP1 engines | Same |
| vLLM memory utilization | 0.7 | Same |
| vLLM/DeepSpeed sleep | enabled | Same |
| SFT datasets and probabilities | upstream two files, `0.5,0.5` | Same |
| SFT steps / batches per step | 1 / 1 | Same |
| Postfill-CoT loss coefficient | 1.0 | Same |
| Seed | 8888 | Same |
| Holdout eval | official harmful holdout, every 10 from step 50 | Same |
| Diversity interval | 5 | Same |
| Hardware launcher | 4 GPUs | H200 x4 |

## Intentional and unavoidable differences

| Area | Upstream Self-RedTeam | Our controlled method | Consequence |
|---|---|---|---|
| Policy parameters | One shared full-parameter 8B policy | Independent A/D LoRA r32, alpha32 over attention and MLP linear layers | Parameter count and effective update scale differ |
| Role update | Attacker and defender trajectories update the same policy every step | A50 updates only A; D50 updates only D | No cross-role RL gradient leakage |
| Opponent | The same policy changes while playing both roles | Opponent is frozen within each phase | Stationary best-response environment |
| Initialization | One shared base policy | A and D each cold-start from base | No transfer from A weights into D weights |
| RL samples optimized per rollout | Generated attacker plus all defender trajectories | A keeps generated attacker trajectories; D keeps defender trajectories | Fewer PPO minibatches per phase is a direct result of freezing one role |
| Online SFT destination | Shared policy | Current role LoRA | Same SFT batch, different parameter destination |
| Schedule | Up to 200 shared-policy steps | 50 A steps then 50 D steps | Required role switch boundary |
| Max samples | 40000 plus stop at step 200 | `128 * 50` per role | Encodes the phase boundary |
| Checkpoint interval | 100 | 50 | Saves each role endpoint |
| W&B | One shared-policy run | One run per role plus coordinator manifest | Role attribution is explicit |
| LoRA runtime | Not used (`lora_rank=0`) | Current and frozen opponent adapters routed in vLLM | Required for independent policies |

The effective role sample counts are not manually rebalanced or duplicated.
With the upstream 50% generation rule, a 128-prompt rollout contains roughly 64
attacker trajectories and 128 defender trajectories before tie removal. The
upstream shared policy can optimize both sets; A-only and D-only phases retain
only their selected role. Repeating samples to equalize update counts would be
an additional algorithmic change and is therefore disabled.

## Explicitly disabled experimental changes

- Optimized jailbreak rewrite prompt
- Attacker-only 100% generation
- Direct-chat frozen defender prompt
- Role-specific defender system prompt
- Malformed-CoT privacy rewrite
- Invalid-request/rewrite filtering
- Hard-negative replay balancing
- Global replay redistribution after tie removal
- Constant or enlarged learning rate
- KL coefficient zero
- Attacker SFT checkpoint initialization
- Reduced generation length

## Safety assertions

The strict launcher rejects a run unless the official model, datasets, batch
geometry, LR, KL, scheduler, auxiliary SFT, prompt profile, and upstream reward
handling are selected. Replay asserts that every optimized trajectory belongs
to the selected role. During D50, generation explicitly routes frozen A50 to
the attacker and current D to the defender; this fixes the previous fallback
that could route both turns through the frozen attacker adapter.

