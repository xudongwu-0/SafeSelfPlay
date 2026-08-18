# SafeSelfPlay

SafeSelfPlay trains separate attacker and defender policies for safety
self-play. It uses the public
[Self-RedTeam](https://github.com/mickelliu/selfplay-redteaming) game and reward
pipeline, PEFT/LLaMA-Factory LoRA adapters, and PSRO-style checkpoint pools.
Training and evaluation run remotely on Modal.

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

All metrics are attack success rates, so lower is better. The first two rows
are reported by Self-RedTeam; the remaining rows use our released evaluator
with the same `llama3_cot` template. They are separated because paper-reported
and locally measured numbers are not a paired evaluation.

| Model | Source | WG adv ASR ↓ | WG vanilla ASR ↓ | WJB harmful ASR ↓ | DAN ASR ↓ | HarmBench adv ASR ↓ |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-abliterated | Paper | 0.478 | 0.553 | 0.991 | 0.937 | 0.654 |
| Self-RedTeam + SFT, step 200 | Paper | 0.138 | 0.019 | 0.240 | 0.396 | 0.221 |
| Llama-3.1-8B-Instruct-abliterated | Our evaluation | 0.424 | 0.454 | 0.938 | 0.910 | 0.575 |
| Self-RedTeam reproduction, step 200 | Our evaluation | 0.095 | 0.019 | 0.159 | 0.400 | 0.196 |
| SafeSelfPlay D1, LoRA generation 1 step 100 | Our evaluation | 0.036 | 0.007 | 0.145 | 0.200 | 0.066 |
| SafeSelfPlay D2, LoRA generation 2 step 80 | Our evaluation | 0.015 | 0.024 | 0.093 | 0.423 | 0.084 |

The D1 and D2 rows use the same rank-64 role-LoRA parameterization but
different step budgets and opponents; they document completed generations and
should not be read as a controlled D1-vs-D2 ablation.

## 2. Train Role LoRAs

Cold-start A1 then D1:

```bash
./run_selfredteam_lora_a1d1_h200x4.sh
```

Continue one complete generation, for example A2/D2 to A3/D3:

```bash
ATTACKER_START_ADAPTER=/output/.../A2/ckpt/global_step80_hf \
DEFENDER_START_ADAPTER=/output/.../D2/ckpt/global_step80_hf \
SOURCE_GENERATION=2 \
./run_selfredteam_lora_next_round_h200x4.sh
```

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

The evaluator reports WG adversarial, WG vanilla, WJB harmful, DAN, and
HarmBench ASR. All are lower-is-better. Results are written under
`/output/upstream_selfredteam_role_full_eval/<OUTPUT_SLUG>`.

## 4. PSRO

The payoff evaluator and cache are implemented in
`modal_upstream_v2_payoff.py` and `roll/utils/upstream_v2_payoff.py`. The ROLL
PSRO orchestration entrypoint is `modal_abs_benchmark.py`:

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
again. The resulting payoff matrix determines the opponent mixture for the
next fixed-role best response.

## Attribution

This repository is a research fork of
[ROLL](https://github.com/alibaba/ROLL). The language game, upstream optimizer,
and general-sum reward follow
[selfplay-redteaming](https://github.com/mickelliu/selfplay-redteaming). The
separate role adapters, sequential best-response training, checkpoint lineage,
and PSRO integration are SafeSelfPlay additions.
