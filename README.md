# SafeSelfPlay

SafeSelfPlay trains separate attacker and defender policies for safety
self-play. It uses the public
[Self-RedTeam](https://github.com/mickelliu/selfplay-redteaming) game and reward
pipeline, PEFT/LLaMA-Factory LoRA adapters, and PSRO-style checkpoint pools.
Training and evaluation run remotely on Modal.

## New-machine setup

Clone this repository and prepare the pinned upstream source as a sibling
directory. The helper creates `../selfplay-redteaming` at the revision expected
by the launchers and tests:

```bash
git clone git@github.com:xudongwu-0/SafeSelfPlay.git
cd SafeSelfPlay
./setup_selfredteam_upstream.sh
python -m pip install modal
modal setup
modal secret create roll-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>
```

The Hugging Face account behind `HF_TOKEN` must be able to download the gated
models used by the selected run. When the new machine logs into the same Modal
workspace, existing cloud volumes and detached runs remain available; the main
ones are `roll-abs-benchmark-output`, `selfredteam-official-output`, and
`roll-hf-cache`. A different Modal workspace starts with empty volumes, so use
the published `xudongwu/SafeSelfPlay-checkpoints` artifacts or copy the required
checkpoints explicitly.

Local `checkpoints/`, model caches, W&B caches, PDFs, and untracked experiment
launchers are not part of Git. Modal builds the CUDA/Python runtime remotely,
so a local GPU is not required for the documented commands.

## 1. Reproduce Self-RedTeam

Prepare the exact upstream source revision:

```bash
./setup_selfredteam_upstream.sh
modal secret create roll-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>
modal deploy modal_selfredteam_wildguard.py
modal deploy modal_selfredteam_official_h200.py
python launch_selfredteam_official_h200.py
```

The reproduced 200-step checkpoint is stored at:

```text
Modal volume selfredteam-official-output:
selfredteam_official/selfredteam_official_repp_fullptx_sft_meta_llama_31_8b_instruct_abliterated_h200x4_s200_20260803_101623/ckpt/global_step200_hf

Hugging Face:
xudongwu/SafeSelfPlay-checkpoints/self-redteam-reproduction/step200
```

The authors' released checkpoints remain in their
[official Hugging Face collection](https://huggingface.co/collections/mickelliu/self-redteam-68f72b48c4beea864617fe4c);
we link to those weights rather than relabeling or mirroring them.

This is our run of upstream commit
`0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123`, not an author-released
checkpoint. Load it as a normal Transformers model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "xudongwu/SafeSelfPlay-checkpoints"
model = AutoModelForCausalLM.from_pretrained(
    repo, subfolder="self-redteam-reproduction/step200"
)
tokenizer = AutoTokenizer.from_pretrained(
    repo, subfolder="self-redteam-reproduction/step200"
)
```

### Defender results

The first two rows are reported by Self-RedTeam; the remaining rows use our
released evaluator with the same `llama3_cot` template. They are separated
because paper-reported and locally measured numbers are not a paired
evaluation. A dash means the paper did not report that supplemental column.

| Model | Source | WG adv ASR ↓ | WG vanilla ASR ↓ | WJB harmful ASR ↓ | DAN ASR ↓ | HarmBench adv ASR ↓ | OR-Bench RTA ↑ | XSTest RTA ↑ | WJB benign comply ↑ | XSTest benign comply ↑ | OR-Bench benign comply ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-abliterated | Paper | 0.478 | 0.553 | 0.991 | 0.937 | 0.654 | 0.014 | 0.290 | 0.992 | 0.988 | - |
| Self-RedTeam + SFT, step 200 | Paper | 0.138 | 0.019 | 0.240 | 0.396 | 0.221 | 0.846 | 0.814 | 0.806 | 0.920 | - |
| Llama-3.1-8B-Instruct-abliterated | Our evaluation | 0.424 | 0.454 | 0.938 | 0.910 | 0.575 | 0.099 | 0.415 | 0.932 | 0.940 | 0.972 |
| Self-RedTeam reproduction, step 200 | Our evaluation | 0.095 | 0.019 | 0.159 | 0.400 | 0.196 | 0.925 | 0.835 | 0.780 | 0.880 | 0.365 |
| SafeSelfPlay D1, step 100 (no-reward-hacking) | Our evaluation | 0.065 | 0.019 | 0.378 | 0.280 | 0.121 | 0.896 | 0.870 | 0.908 | 0.932 | 0.469 |
| SafeSelfPlay D2, step 100 (no-reward-hacking) | Our evaluation | 0.074 | 0.019 | 0.272 | 0.377 | 0.126 | 0.860 | 0.895 | 0.820 | 0.952 | 0.637 |
| SafeSelfPlay D3, step 100 (no-reward-hacking) | Our evaluation | 0.024 | 0.022 | 0.135 | 0.083 | 0.087 | 0.841 | 0.870 | 0.856 | 0.988 | 0.588 |
| SafeSelfPlay D1, step 100(reward-hacking) | Our evaluation | 0.036 | 0.007 | 0.145 | 0.200 | 0.066 | 0.916 | 0.945 | 0.612 | 0.836 | 0.367 |
| SafeSelfPlay D2, step 80 (reward-hacking) | Our evaluation | 0.015 | 0.024 | 0.093 | 0.423 | 0.084 | 0.802 | 0.670 | 0.584 | 0.984 | 0.510 |
| SafeSelfPlay D3, step 80 (reward-hacking) | Our evaluation | 0.003 | 0.000 | 0.031 | 0.090 | 0.031 | 0.860 | 0.780 | 0.604 | 0.980 | 0.430 |

The five ASR columns contain harmful prompts and are lower-is-better. OR-Bench
RTA and XSTest RTA also contain harmful or contrast prompts, but report the
safe-response rate (`1 - ASR`), so they are higher-is-better. WJB adversarial
benign, XSTest vanilla benign, and OR-Bench hard-1k measure whether the model
still complies with benign requests; lower values on these columns indicate
more over-refusal. The released evaluator does not include StrongREJECT.

The current-pipeline D1 row evaluates the rank-64, 100-step checkpoint from
`cold_psro_capdrop_A100D100_r64_20260819`. Its exact result manifest is
`/output/upstream_selfredteam_role_full_eval/cold_psro_capdrop_A100D100_r64_20260819_D1_full_eval_20260820/comparison.json`.
The base values are the previous released-evaluator run rather than a fresh
paired rerun. D1 substantially improves harmful-prompt safety, but the
OR-Bench benign compliance drop from 0.972 to 0.469 indicates over-refusal.

The current-pipeline D2 row is the 100-step latest-opponent continuation from
D1. Its exact result manifest is
`/output/upstream_selfredteam_role_full_eval/naive_latest_A2_to_D5_s100_20260820_D2_full_eval_20260820/comparison.json`.
Relative to D1, D2 improves WildJailbreak harmful ASR, XSTest RTA, and
OR-Bench benign compliance, while DAN ASR and WildJailbreak benign compliance
regress; it is therefore a mixed rather than uniformly dominant update.

The current-pipeline D3 row is the next 100-step latest-opponent continuation.
Its exact result manifest is
`/output/upstream_selfredteam_role_full_eval/naive_latest_A2_to_D5_s100_20260820_D3_full_eval_20260820/comparison.json`.
Relative to D2, D3 substantially improves four of the five harmful ASR
benchmarks and improves WildJailbreak and XSTest benign compliance. OR-Bench
benign compliance and the two RTA columns regress, so later generations remain
subject to the same safety-versus-over-refusal evaluation.

The reward-hacking D1-D3 rows use the same rank-64 role-LoRA parameterization but
different step budgets, opponents, and pre-fix training pipeline; they document
completed generations and should not be read as a controlled inter-generation
ablation against the current cold restart.

## 2. Train Role LoRAs

Cold-start PSRO A1 then D1:

```bash
./run_selfredteam_lora_a1d1_h200x4.sh
```

The training reward remains the original Self-RedTeam general-sum reward with
its existing CoT/SFT shaping. For generated attacks whose WildGuard prompt
label flips relative to the seed, attacker reward is capped at zero and the
corresponding defender-training game is omitted from replay. This policy does
not alter the separately computed zero-sum PSRO matrix reward.

Continue through warm-start, latest-opponent naive self-play. The canonical
launcher reads the verified A1/D1 population from the cold-run state and, by
default, trains A2, D2, ..., A5, D5 for 100 steps each:

```bash
SOURCE_RUN_SUFFIX=cold_psro_capdrop_A100D100_r64_20260819 \
./run_selfredteam_lora_next_round_h200x4.sh
```

Each attacker inherits the previous attacker and trains against the previous
defender; each defender inherits the previous defender and trains against the
attacker just produced in the same generation. Every phase uses a new
optimizer, exact 50/50 harmful/benign input seeds, and the same asymmetric
label-drift policy as A1/D1. State and adapter hashes are persisted between
phases so a failed chain can be relaunched without repeating completed roles.

Recover only a failed defender phase without retraining its attacker:

```bash
ATTACKER_ADAPTER=/output/.../A3/ckpt/global_step80_hf \
DEFENDER_START_ADAPTER=/output/.../D2/ckpt/global_step80_hf \
TARGET_GENERATION=3 \
./run_selfredteam_lora_defender_only_h200x4.sh
```

Canonical settings are H200x4, LoRA `r64/alpha64`, rollout/train/micro batch
`128/32/8`, attacker LR `1e-5`, defender LR `4e-5`, no KL penalty, and a
5% warmup followed by constant LR. Checkpoints are saved every 10 steps.

Published adapters live under `lora/A1` through `lora/D3` in
[`xudongwu/SafeSelfPlay-checkpoints`](https://huggingface.co/xudongwu/SafeSelfPlay-checkpoints).
Load one adapter with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
)
model = PeftModel.from_pretrained(
    base, "xudongwu/SafeSelfPlay-checkpoints", subfolder="lora/D3"
)
```

## 3. Evaluate

Deploy the evaluator once, then evaluate any full checkpoint or PEFT adapter
stored on the Modal output volume:

```bash
modal deploy modal_selfredteam_official_eval.py
CHECKPOINT_PATH=/output/.../D3/ckpt/global_step80_hf \
OUTPUT_SLUG=D3_full_eval \
TRAINED_LABEL=our_D3 \
./run_selfredteam_eval.sh
```

The evaluator runs WG adversarial/vanilla, WJB harmful, DAN, HarmBench,
OR-Bench toxic, XSTest, WJB benign, and OR-Bench hard-1k. It reports harmful
ASR (lower is better), harmful RTA (higher is better), and benign compliance
(higher is better). Both aggregate and per-example results are written under
`/output/upstream_selfredteam_role_full_eval/<OUTPUT_SLUG>`.

## 4. PSRO

The current Llama role-LoRA launcher trains each new policy against the latest
opponent. Evaluate one frozen adapter pair with the adapter-aware payoff code:

```bash
modal run --detach \
  modal_upstream_v2_payoff.py::upstream_v2_payoff_convergence \
  --attacker-adapter /output/.../A3/ckpt/global_step80_hf \
  --defender-adapter /output/.../D3/ckpt/global_step80_hf \
  --episodes 4096
```

The evaluator and convergence cache are implemented in
`modal_upstream_v2_payoff.py` and `roll/utils/upstream_v2_payoff.py`.

The older ROLL/Qwen path contains the complete payoff-matrix, Nash-mixture,
and PSRO orchestration entrypoint in `modal_abs_benchmark.py`:

```bash
modal run modal_abs_benchmark.py \
  --mode payoff-matrix \
  --payoff-attacker-paths /output/.../A1,/output/.../A2 \
  --payoff-defender-paths /output/.../D1,/output/.../D2

modal run --detach modal_abs_benchmark.py \
  --mode asym-psro-train \
  --asym-iterations 5 \
  --asym-role-steps 50
```

Pairwise results are cached, so unchanged checkpoint pairs are not evaluated
again. This complete orchestrator is not yet wired to the current Llama
rank-64 role-LoRA launcher; current A1-A3/D1-D3 checkpoints therefore represent
sequential latest-opponent self-play, not Nash-mixture PSRO.

## Attribution

This repository is a research fork of
[ROLL](https://github.com/alibaba/ROLL). The language game, upstream optimizer,
and general-sum reward follow
[selfplay-redteaming](https://github.com/mickelliu/selfplay-redteaming). The
separate role adapters, sequential best-response training, checkpoint lineage,
and PSRO integration are SafeSelfPlay additions.
