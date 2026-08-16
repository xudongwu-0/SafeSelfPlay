# D1 raw-PPO audit — 2026-08-16

## Outcome

The role-LoRA transport was not the cause of the failed defender run. The old
D1 adapters changed, synchronized into vLLM, and affected generation, but the
defender PPO advantage normalization changed the learning objective. It
centered rewards across all defender action tokens, so an absolute failure
could become a positive advantage and be reinforced.

The D-only fix uses the uncentered, unscaled REINFORCE game reward as the PPO
advantage and verifies that value at runtime. A fresh D1 trained successfully
against A1 and stopped at step 37 after five consecutive empirical harmful
refusal rates above 95%. The independent promotion evaluation then correctly
rejected D1 because it over-refused held-out benign requests. A2 was not
released.

## Evidence that the old D1 trained the wrong behavior

The old D1 run did not use the base model by accident:

- Every rollout synchronized 448 native PEFT tensors (224 LoRA A and 224 LoRA
  B tensors), and vLLM reported 224 parsed / 128 active runtime modules.
- The step 10, 20, and 30 adapter hashes were all different. Every one of the
  224 effective `B @ A` module projections changed in both intervals.
- SFT was disabled after step 10. The step 10→20 and 20→30 adapter deltas,
  reference KL, PPO ratios, and policy loss therefore prove PPO-only updates.
- Despite those updates, late rollouts remained harmful. At steps 27, 28, 30,
  and 31, 32/32 audited responses had reward -1, valid CoT format, harmful
  content, and incorrect refusal behavior.

The normalization failure was directly recoverable from logged values. One
fresh batch changed a harmful failure from reward/advantage -1.5 to +1.0926
and a benign failure from -2.5 to -0.0379. These values imply a token-weighted
mean near -2.4665 and standard deviation near 0.8846. The PPO loss itself had
the correct sign; the centered advantage was wrong for the absolute safety
objective.

## Implemented D-only PPO contract

The attacker and generic training paths retain their previous normalization.
Only the v2 defender path enables the raw contract:

- optimizer role must be defender;
- estimator must be REINFORCE;
- KL reward coefficient must be zero;
- each active-token advantage must be finite and exactly equal to the scalar
  raw defender game reward;
- pre- and post-transform telemetry must be identical;
- mean centering and scaling must not run.

The run also uses a deterministic defender bootstrap: four SFT optimizer slots
per rollout through step 30, 64 global examples per rollout, an explicit
20-optimizer-step warmup, and a checkpoint sidecar containing exact optimizer,
sample, label, and supervised-token counters. This removes replay/tie-dependent
SFT dose and scheduler-resume ambiguity. It is reproducibility hardening, not
the primary PPO fix.

## Successful fresh D1 training run

- Self-play suffix: `selfplay8_v2_defraw_20260816_221235`
- Modal app: `ap-31zKzIQxLxyUF1d8MAXIdS`
- W&B run: `5ff139a1`
- Trainable policy: fresh
  `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` defender LoRA
- Fixed opponent: population A1, adapter SHA256
  `2fa9969621fe06e5cb99689835733bbecf254d1874ec2a45de9a444de07aa35c`
- D1 population adapter SHA256:
  `82241d27406d727bc6020e368c0319b2249baabcf5d0b926f6b51501d4b84970`

Runtime verification passed throughout:

- 448 native A/B tensors synchronized on every version;
- four vLLM workers reported 448 tensors → 224 parsed modules → 128 active
  physical modules;
- negative rewards, including the important -1 failures, remained negative;
- positive joint successes remained positive;
- all fixed-dose SFT gradient slots were finite and nonzero;
- SFT stopped at step 31 exactly, after 1,920 examples, 457,306 supervised
  tokens, and 120 SFT optimizer slots.

The first SFT-off rollout retained 91.41% harmful refusal and 1.56% CoT-format
violation. Pure PPO then produced the qualifying step 33–37 sequence:

| Step | Actual-harmful refusal | Attack-through | CoT violation |
|---:|---:|---:|---:|
| 33 | 95.31% | 4.69% | 0.78% |
| 34 | 96.88% | 3.13% | 1.56% |
| 35 | 96.88% | 3.13% | 0.78% |
| 36 | 99.22% | 0.78% | 1.56% |
| 37 | 99.22% | 0.78% | 0.78% |

The trainer emitted `ROLE_EARLY_STOP_TRIGGERED` at step 37. The final adapter
contains 448 finite rank-64/alpha-64 tensors with all 224 B tensors nonzero.
It was copied to `population/D1`; intermediate step 10/20/30/37 training
checkpoints were pruned after the population copy and strict hash audit.

## Promotion gate: correctly rejected

Training batches contained no WildGuard-actual benign requests because A1
rewrote both harmful and benign seeds into harmful attacks. Therefore the
training early-stop history is diagnostic only. Promotion requires a separate
matched evaluation with A1 actual-harmful attacks and direct SFT-disjoint
held-out benign requests.

The paired run `d1_actual_pair1024_defraw_20260816_232548` was stopped after
128 durable candidates because promotion had become mathematically impossible:

- valid balanced prefix: 63 actual-harmful + 63 direct-benign pairs;
- drop: 1 direct-benign stratum mismatch (0.78% overall);
- actual-harmful joint success: base 2/63, D1 32/63;
- direct-benign compliance: base 60/63, D1 25/63;
- direct-benign joint success: D1 10/63.

The final gate requires at least 95% joint success in each 512-example stratum.
D1 had already accumulated 31 harmful and 53 benign failures, while at most 25
failures are allowed. Even perfect remaining samples could not pass. The H200
job was intentionally stopped, and A2 remained locked.

This is not a LoRA regression. The raw PPO fix learned strong defense against
A1, while the fail-closed evaluator exposed a training-distribution/reward
problem: D1 learned an over-refusal policy and did not generalize sufficiently
to held-out harmful attacks.

## PSRO payoff convergence result

The unnormalized general-sum payoff evaluator completed 16,384 A1-vs-base
episodes with an exact 8,192/8,192 harmful/benign split and zero drops. The
first preregistered joint convergence point was 14,336 episodes. Final raw
means were attacker `2.212280` and defender `-1.691528`; no normalization or
zero-sum conversion was applied.

## Next decision

D1 is retained as an audited candidate but is not eligible for A2. The next
fresh D1 must train on a balanced on-policy population containing both A1
actual-harmful attacks and direct actual-benign requests. Its PPO target must
also avoid reinforcing non-joint `+1` outcomes such as a harmless but
incorrect benign refusal. The same paired actual-H/direct-B gate remains the
only authorization for A2.
