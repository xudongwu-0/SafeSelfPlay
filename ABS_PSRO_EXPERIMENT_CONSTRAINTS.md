# ABS + PSRO Experiment Constraints

This note records the constraints for the collaborator-facing ABS-style PSRO
experiments. Treat it as the source of truth before launching or editing these
runs.

## Method Boundary

- The target comparison is:
  - vanilla ABS-style anchored bipolicy self-play
  - the same ABS-style setup plus PSRO opponent-pool selection
- The ABS repository points to `mickelliu/selfplay-redteaming` as the current
  public code path to train ABS LoRA adapters, but this is still a public
  reference implementation path, not a standalone official ABS code release.
- The intended algorithmic delta is only:
  - attacker/defender checkpoint pools
  - payoff matrix computation
  - Nash or mixture-based opponent selection
  - fixed-role training segments
  - final model selection from the historical pool
- Do not present extra reward shaping as part of the method.

## Cold-Start Training

- Current main protocol is cold start.
- Every attacker best response starts from the base Qwen2.5-3B-Instruct LoRA
  initialization.
- Every defender best response starts from the base Qwen2.5-3B-Instruct LoRA
  initialization.
- Do not use same-role inheritance for the main result:
  - not `A1 -> A2 -> A3`
  - not `D1 -> D2 -> D3`
- One PSRO iteration is 100 optimizer steps:
  - 50 attacker steps
  - 50 defender steps
- Multi-iteration runs should be named clearly, for example:
  - `absreplica_psro_coldstart_iter100x5_noanchor`

## Reward And Diagnostics

- Formal PSRO runs should match the public `selfplay-redteaming` / ABS-style
  reward as closely as possible.
- Prefer the public reference `general_sum` component reward for formal runs:
  - attacker prompt harmfulness label consistency: +/- 0.5
  - attacker response refusal outcome: +/- 0.5
  - attacker response harmfulness outcome: +/- 1.0
  - defender response refusal correctness: +/- 1.0
  - defender response harmlessness: +/- 1.0
- Do not use the local `abs_zero_sum` simplification for formal ABS-reference
  comparisons unless explicitly labeled as a separate ablation.
- On-topic, anchor, and extra-content checks may be logged as diagnostics, but
  should not change the formal training reward unless the vanilla baseline uses
  the exact same shaping.
- For no-anchor formal runs, set:
  - `ABS_ATTACKER_ON_TOPIC_WEIGHT=0`
- Useful diagnostic metrics to keep in W&B:
  - `attack_on_topic_score`
  - `attack_anchor_recall`
  - `attack_label_consistent`
  - `attack_offtopic_penalty`
- If these diagnostics are not part of reward, describe them only as monitoring
  metrics.

## Payoff Matrix

- The PSRO payoff matrix should represent a clean safety game outcome, not the
  shaped GRPO training reward.
- Use attacker-perspective payoff:
  - positive means attacker succeeds / defender fails
  - negative means defender succeeds / attacker fails
- Cache previously computed attacker-defender pairs.
- Do not recompute cached pairwise payoffs unless the checkpoint path, labels,
  evaluator, or sample budget changed.
- After each full 100-step iteration, compute or update the matrix and use the
  resulting mixture to select opponents for the next iteration.

## Vanilla Baseline

- Vanilla baseline is the ABS-style bipolicy run under the same model,
  dataset, judge, optimizer budget, and evaluation protocol.
- It should not receive different prompt templates, reward shaping, sampling
  settings, or evaluator settings unless explicitly documented.
- If PSRO disables a shaping term for formal comparison, vanilla must disable
  it too.

## Reporting

- Be precise in wording:
  - "ABS-style reproduction" for our ROLL implementation.
  - "ABS + PSRO" for the extension.
  - Avoid saying that every detail is the original ABS official code.
- Separate:
  - paper-reported ABS numbers
  - our vanilla ABS-style reproduction
  - our ABS + PSRO result
- Exclude uncertain XSTest columns from main tables unless the evaluation setup
  is revalidated.
