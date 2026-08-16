# Role-specific LoRA v2 successful reproduction (2026-08-16)

## Result

The repaired Llama-3.1-8B attacker LoRA completed 100 optimizer steps on four
Modal H200 GPUs and reproduced the historical v2 learning transition.

- Modal training app: `ap-hZRSNwYh8hjSH3axNYBLZW`
- Modal function call: `fc-01M04MG0JV2GJDP02DN4FWM8ZR`
- W&B run: [af7e646b](https://wandb.ai/2373025856w-the-university-of-hong-kong/self-play/runs/af7e646b)
- Final step 100: reward `2.45`, request success `1.000`
- First joint target at step 65: reward `2.51`, request success `0.969`
- Best reward at step 88: reward `2.59`, request success `1.000`
- Last-ten median: reward `2.505`, request success `1.000`
- Final harmful and benign request success: both `1.000`
- Final CoT-format violation rate: `0`

The authoritative run directory on Modal Volume `roll-abs-benchmark-output` is:

```text
/output/upstream_selfredteam_role_lora/upstream_selfredteam_v2repro_meta_llama_31_8b_instruct_abliterated_attacker_lora_r64_a64_fromBase_vs_base_prompt_optimized_normalmix_s100_rb128_mb8_tb32_lr1e-5_kl0_warm0.05_const_auxsft_upstreaminvalid_lora_v2_repaired_20260816_1435
```

`run_status.json` records `return_code=0` and `completed=true`. The independent
checkpoint validator records `expected_step=final_step=100`.

## What was repaired

The historical failure mode was real: registering a fixed LoRA ID without
evicting it lets vLLM's LRU cache retain stale adapter weights. The repaired
path now removes the old fixed slot before every add and fails closed if the
adapter request or worker registration is missing.

The complete runtime contract now also:

- synchronizes the initial PEFT adapter before the first rollout and after
  every optimizer step;
- transfers all 448 native PEFT A/B tensors rather than merged dense weights;
- preserves canonical PEFT names and verifies their mapping to active vLLM
  modules;
- verifies 224 A/B pairs over 32 layers and seven target modules;
- propagates an explicit per-request LoRA selector, while defender inference
  explicitly uses base weights;
- rejects partial, duplicate, rank-mismatched, non-finite, or unregistered
  adapters rather than silently generating with the base model.

The v2 reproduction additionally restored the successful training recipe:

- LoRA rank/alpha `64/64`, targets `q/k/v/o/gate/up/down_proj`;
- LR `1e-5`, `constant_with_warmup`, warmup ratio `0.05`;
- rollout/micro-rollout `128/8`, train/micro-train `32/8`;
- optimized harmful/benign prompts with a 50/50 source mixture;
- exact rendered rollout prefix plus continuation-only attacker SFT;
- auxiliary SFT through step 30, disabled from step 31;
- REINFORCE with normalized reward, KL coefficient zero, reference KL
  monitoring only;
- generation length 2048, checkpoint/evaluation cadence 10, seed 8888.

The rendered 1,180-row continuation dataset has SHA-256
`05c101e5ed0c5cb4e6d554fa8706e1cede0ddffbd41e25179bc7d829dcda8019`.
All 1,180 prompt/completion boundaries were checked against the real
Llama-3.1 tokenizer. The source dataset SHA-256 is
`11b860bee147d668ad3645a8c757bdab6b2fbcaeed8e0ac5e2acd108ce13c233`.

One documented implementation difference remains: the historical run used a
separate fixed-opponent vLLM engine, whereas this run used the same vLLM base
weights with the LoRA request explicitly disabled for defender generation.
The attacker/base request routing was independently exercised before the long
run, and the full learning curve below confirms that updated adapter weights
were used for attacker rollouts.

## Runtime evidence

Before the first rollout, every vLLM worker accepted the adapter and reported:

```text
Verified training LoRA registration: 448 tensors -> 224 parsed modules -> 128 active runtime modules
Verified LoRA sync version 1: 448 native PEFT A/B tensors
```

The final update reported sync version 101. The persisted `training.log`
contains exactly the contiguous versions 1 through 101, with no traceback,
missing-LoRA request, base fallback, or registration failure.

Selected learning points:

| Step | Reward | Request success | CoT violation | SFT coefficient |
|---:|---:|---:|---:|---:|
| 1 | -1.24 | 0.117 | 0.875 | 1 |
| 10 | -0.289 | 0.117 | 0.367 | 1 |
| 20 | 0.805 | 0.281 | 0.0156 | 1 |
| 30 | 0.625 | 0.188 | 0 | 1 |
| 31 | 0.570 | 0.188 | 0 | 0 |
| 40 | 1.00 | 0.297 | 0 | 0 |
| 50 | 1.44 | 0.484 | 0 | 0 |
| 60 | 2.21 | 0.891 | 0 | 0 |
| 63 | 2.48 | 0.984 | 0.00781 | 0 |
| 65 | 2.51 | 0.969 | 0.00781 | 0 |
| 66 | 2.52 | 1.000 | 0 | 0 |
| 88 | 2.59 | 1.000 | 0 | 0 |
| 100 | 2.45 | 1.000 | 0 | 0 |

The last ten rewards were `2.57, 2.52, 2.46, 2.48, 2.55, 2.49, 2.56,
2.55, 2.41, 2.45`; their median is `2.505`. The corresponding request-success
median is `1.000`.

## Persisted checkpoints

All expected checkpoints exist and have distinct adapter-weight hashes:

| Step | SHA-256 |
|---:|---|
| 10 | `979a98cb3f2638ede47e92861dfc15966357586e09a8ba4df2aecfbf8fb253d9` |
| 20 | `c6fa4e9862d4ac2ba11f83a75b6a3d1621180acfba7d03860cab91d374546d23` |
| 30 | `0bd941e9ecb78228d0c0c6b366f043fd2c5068ed9ebdf435f73434fa7902ccaa` |
| 40 | `075f3ae81a667986138d3af2d2ec861c3eaa92cd7b5b109528101b20ca92f4a1` |
| 50 | `6d008eeaa21b1bf50b708c01a9917bbc4844e16e3894070d1f1b9b68c3d2d325` |
| 60 | `eaec8a58cbae2c11246a5346ac5a8c0b8016621ec28c7e4a0549be4703a527ef` |
| 70 | `d57896cf66da398699254da0f5dbe6e46b01bdc352208fbb60438dcc49dc24cd` |
| 80 | `3a4a3312912c0e2aa678d140cf639492b2a852076efef1ffffa16a1814aa707b` |
| 90 | `5470ac5233682ef52ecb5fe3dfc785ca05e2da012c0cbb40a06acfd0190b6f3c` |
| 100 | `2fa9969621fe06e5cb99689835733bbecf254d1874ec2a45de9a444de07aa35c` |

The strict step-100 numerical audit passed:

- total LoRA tensors: 448;
- A tensors: 224/224 nonzero and finite;
- B tensors: 224/224 nonzero and finite;
- rank/alpha: 64/64;
- A squared L2 norm: `4808.182231903076`;
- B squared L2 norm: `7.694841127842665`;
- final adapter hash: `2fa9969621fe06e5cb99689835733bbecf254d1874ec2a45de9a444de07aa35c`.

## Reproduce and audit

With the `marl4llms` Modal profile active and `roll-secrets` configured:

```bash
modal run --detach \
  modal_upstream_selfredteam_role_lora.py::role_lora_v2_reproduction \
  --run-suffix <unique-suffix>
```

The entrypoint pins the validated 100-step recipe. To audit a completed final
checkpoint:

```bash
modal run \
  modal_upstream_selfredteam_role_lora.py::audit_lora_checkpoint \
  --checkpoint-path /output/<run-dir>/ckpt/global_step100_hf \
  --expected-tensor-count 448
```

The strict audit is enabled by default and fails unless the complete Llama v2
448/224/224/nonzero/finite contract holds.

## Local regression evidence

The focused offline suite has 21 passing tests across
`tests.test_upstream_role_lora_sync` and `tests.test_role_lora_v2_recipe`.
It covers initial/per-step synchronization, fixed-ID eviction and reload,
request routing, PEFT/vLLM name mapping, A/B tensor contracts, exact
continuation boundaries, the scheduler recipe, strict checkpoint cadence, and
the final numerical audit contract. The changed Python files also pass
`py_compile` and `git diff --check`.
