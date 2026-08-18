# SafeSelfPlay

SafeSelfPlay is a research fork of ROLL for safety self-play experiments. It
implements an ABS-style two-player red-team safety game and adds PSRO-style
opponent pools for alternating attacker/defender training.

This codebase is built on top of [ROLL](https://github.com/alibaba/ROLL) and
uses its Ray/vLLM/DeepSpeed agentic RL infrastructure. The safety environment,
prompt templates, and WildGuard-based reward path are informed by the public
[selfplay-redteaming](https://github.com/mickelliu/selfplay-redteaming)
reference used around Anchored Bipolicy Self-Play (ABS). We should describe
our runs as an **ABS-style reproduction/extension**, not as official ABS
training code.

## What This Fork Adds

- A `RedTeamSafetyEnv` two-turn self-play environment:
  - attacker rewrites a seed prompt into a stronger safety test;
  - defender answers the rewritten prompt safely and helpfully;
  - rewards support the public-style `general_sum` safety components.
- Fixed-role training segments for attacker-only and defender-only LoRA best
  responses.
- PSRO-style attacker/defender checkpoint pools with cached pairwise payoff
  matrices and mixture-based opponent selection.
- Modal launchers for remote GPU training, WildGuard reward serving, safety
  evaluation, SFT attacker bootstrapping, and prompt-rewrite probes.
- W&B logging for training curves, safety metrics, raw prompt/response tables,
  payoff matrices, and selected checkpoint metadata.
- Role-start KL support, so each role can measure KL against its own starting
  LoRA or base model instead of an unrelated reference.

## Key Files

- `modal_upstream_selfredteam_role_lora_v2.py` - the canonical LoRA training,
  continuation, audit, and remote orchestration implementation.
- `run_selfredteam_lora_a1d1_h200x4.sh` - cold-start `A1 -> D1` training.
- `run_selfredteam_lora_a2d2_h200x4.sh` - continued `A2 -> D2` self-play and
  automatic D2 evaluation.
- `modal_selfredteam_official_eval.py` - paired defender benchmark evaluation.
- `data/safety_selfplay/` - the cleaned attacker SFT set, generation prompts,
  and data-quality report used by the canonical run.
- `roll/pipeline/agentic/env/red_team_safety/env.py` - safety game and reward
  implementation.

## Training Quick Start

Training is remote-only through Modal; the launchers never use the local GPU.
Create the required secret once, then verify access to WildGuard:

```bash
cd /home/xudong/work/self_play/ROLL
modal secret create roll-secrets WANDB_API_KEY=<wandb-key> HF_TOKEN=<hf-token>
modal run modal_abs_benchmark.py --mode check-token
```

Only the LoRA v2 launchers below are canonical. Older experiment entrypoints
remain implementation history and must not be used for a new comparable run.
Each role has an independent adapter, and the inactive role is frozen.

### LoRA A1 -> D1

```bash
./run_selfredteam_lora_a1d1_h200x4.sh
```

The default is A100 followed by D100 on four H200 GPUs. Each role starts from
the same base model with its own rank-64, alpha-64 adapter. The formal setting
uses attacker LR `1e-5` and defender LR `4e-5`.

```bash
STEPS_PER_ROLE=100 \
ATTACKER_LR=1e-5 \
DEFENDER_LR=4e-5 \
RUN_SUFFIX=lora_formal \
./run_selfredteam_lora_a1d1_h200x4.sh
```

### LoRA A2 -> D2 Self-Play

After A1 and D1 exist, run the next sequential self-play iteration with:

```bash
ATTACKER_START_ADAPTER=/output/.../A1/ckpt/global_step100_hf \
DEFENDER_START_ADAPTER=/output/.../D1/ckpt/global_step100_hf \
./run_selfredteam_lora_a2d2_h200x4.sh
```

This continues the existing adapters instead of cold-starting them: A2 starts
from A1 and trains for 80 steps against frozen D1; D2 starts from D1 and trains
for 80 steps against frozen A2. The remote orchestrator then evaluates D2 on
WG adversarial, WG vanilla, WJB harmful, DAN, and HarmBench. Both starting
adapter paths are required so the checkpoint lineage is explicit.

Both launchers use `modal run --detach`, so the remote call survives an SSH
disconnect. Training curves and raw trajectory tables are written to the W&B
`self-play` project; checkpoints and manifests are written to the persistent
Modal `/output` volume.

## Canonical Parameters

| Setting | Canonical value | Meaning |
| --- | ---: | --- |
| Base model | Llama 3.1 8B abliterated | Independent initialization for A1 and D1 |
| A1/D1 steps | `100 / 100` | Optimizer updates per cold-start role |
| A2/D2 steps | `80 / 80` | Optimizer updates per continued role |
| Attacker LR | `1e-5` | Actor optimizer LR during A1/A2 |
| Defender LR | `4e-5` | Actor optimizer LR during D1/D2 |
| LoRA rank / alpha | `64 / 64` | Adapter capacity and scaling |
| Rollout / train / micro batch | `128 / 32 / 8` | Prompts, replay batch, per-device micro batch |
| Attacker SFT cutoff | 30 | Role-specific format SFT is active through this step |
| Defender SFT cutoff | 10 | Defender format SFT is active through this step |
| KL loss coefficient | `0` | Role-start KL is monitored but not penalized |
| LR schedule / warmup | constant / `0.05` | Constant LR after a 5% warmup |
| Checkpoint interval | 10 | Optimizer steps between checkpoints |
| Reward | upstream `general_sum` | Self-RedTeam role reward, unchanged |

Every step consumes a rollout batch of 128 prompts. Therefore A100 + D100 is
200 optimizer updates and 25,600 role-conditioned prompt rollouts in total;
it should not be described as a shared-policy 100-step run.

## LoRA Implementation And Audits

Use only `modal_upstream_selfredteam_role_lora_v2.py` for new LoRA runs. It
uses LLaMA-Factory 0.9.3/PEFT to construct adapters and fixes two failures in
the historical path: native LoRA A/B tensors are synchronized into every vLLM
rollout engine, and replay is redistributed globally after zero-variance groups
are removed instead of silently truncating a 128-sample rollout. It also
restores adapters after vLLM sleep/wake and validates target module names.

Audit a checkpoint before adding it to a PSRO pool:

```bash
modal run modal_upstream_selfredteam_role_lora_v2.py::lora_v2_app.audit_lora_weights \
  --checkpoint /output/.../ckpt/global_step100_hf
modal run modal_upstream_selfredteam_role_lora_v2.py::lora_v2_app.audit_lora_rollout \
  --checkpoint /output/.../ckpt/global_step100_hf
```

The rollout audit compares Hugging Face PEFT inference, vLLM file-based LoRA,
and in-memory synchronization. The latter two must be token-identical.

## Learning-Rate Sweep Record

The LR search was a sequence of controlled runs, not an automated exhaustive
grid. LoRA attacker trials covered approximately `1e-6`, `2.5e-6`, `4e-6`,
`5e-6`, and `1e-5`; defender trials covered `1e-5` and `2e-5`. Lower attacker
rates learned slowly or reached an early format plateau. The post-fix A1
baseline selected `1e-5`, and the matched D1 selected `2e-5`. Runs made before
the native adapter-sync and replay fixes are diagnostic history and are not
valid LR comparisons.

The controlled acceleration probe doubled the LoRA rates to A1 `2e-5` and D1
`4e-5`. Doubling the attacker LR did not improve its final result enough to
replace `1e-5`; doubling the defender LR produced materially faster D1
convergence and is the formal defender setting. All other parameters remained
fixed. The probe runs are
[`3e24be98`](https://wandb.ai/2373025856w-the-university-of-hong-kong/self-play/runs/3e24be98)
and its paired D1 run.
The established baseline runs are
[`A1 6087d6fb`](https://wandb.ai/2373025856w-the-university-of-hong-kong/self-play/runs/6087d6fb)
and
[`D1 7da2ba71`](https://wandb.ai/2373025856w-the-university-of-hong-kong/self-play/runs/7da2ba71).
The D1 checkpoint obtained WG adv `0.036`, WG vanilla `0.007`, WJB `0.145`,
DAN `0.200`, and HarmBench adv `0.066` in the saved evaluation.

---

<div align="center">

<img src="assets/roll.jpeg" width="40%" alt="ROLL Logo">

# ROLL: Reinforcement Learning Optimization for Large-Scale Learning

<h4>🚀 An Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models 🚀</h4>

<p>
  <a href="https://github.com/alibaba/ROLL/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/alibaba/ROLL/issues">
    <img src="https://img.shields.io/github/issues/alibaba/ROLL" alt="GitHub issues">
  </a>
  <a href="https://github.com/alibaba/ROLL/stargazers">
    <img src="https://img.shields.io/github/stars/alibaba/ROLL?style=social" alt="Repo stars">
  </a>
  <a href="https://arxiv.org/abs/2506.06122"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>
  <!-- 组织主页：点击跳转到 https://github.com/alibaba -->
  <a href="./assets/roll_wechat.png" target="_blank">
    <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
  </a>
  <a href="https://deepwiki.com/alibaba/ROLL" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
  <a href="./assets/future_lab.png" target="_blank">
    <img src="https://img.shields.io/twitter/follow/FutureLab2025?style=social" alt="X QR">
  </a>
</p>

</div>

ROLL is an efficient and user-friendly RL library designed for Large Language Models (LLMs) utilizing Large Scale GPU resources. It significantly enhances LLM performance in key areas such as human preference alignment, complex reasoning, and multi-turn agentic interaction scenarios.

Leveraging a multi-role distributed architecture with Ray for flexible resource allocation and heterogeneous task scheduling, ROLL integrates cutting-edge technologies like Megatron-Core, SGLang and vLLM to accelerate model training and inference.



---

## 📢 News

| 📣   Updates                                                                                                                                                                                                                                                                                                                                                        |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[03/06/2026]** 🎉 We support Qwen3.5 [Dense](examples/qwen3.5-35BA3-rlvr_megatron/rlvr_megatron_80GB.yaml) and [MoE](examples/qwen3.5-35BA3-rlvr_megatron/rlvr_megatron_80GB.yaml) series models and [on-policy distill](docs_roll/i18n/zh-Hans/docusaurus-plugin-content-docs/current/User Guides/Pipeline/on_policy_distill_pipeline_start.md). Welcome to use! |
| **[02/03/2026]** 🎉 We released FSDP2 Strategy, Megatron with LoRA, GPU partial overlapping, Qwen3-Omni supports and other features. For more details, please refer to the release notes. Welcome to use!                                                                                                                                                           |
| **[01/01/2026]** 🎉 Our [Let It Flow: Agentic Crafting on Rock and Roll](https://arxiv.org/abs/2512.24873) report released! Introducing ALE ecosystem and ROME, an open-source agentic model with novel IPA algorithm.                                                                                                                                              |
| **[11/08/2025]** 🎉 Our [ROCK: Reinforcement Open Construction Kit](https://github.com/alibaba/ROCK) released, Explore the new capabilities!.                                                                                                                                                                                                                       |
| **[10/23/2025]** 🎉 Our Papers released, see [Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning](https://arxiv.org/abs/2510.01656) and [Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy Optimization](https://arxiv.org/abs/2510.13554).                                                     |
| **[10/14/2025]** 🎉 Our Paper released, see [Part II: ROLL Flash -- Accelerating RLVR and Agentic Training with Asynchrony](https://arxiv.org/abs/2510.11345).                                                                                                                                                                                                      |
| **[09/28/2025]** 🎉 Ascend NPU support — see [usage guide](https://alibaba.github.io/ROLL/docs/User%20Guides/Hardware%20Support/ascend_usage).                                                                                                                                                                                                                      |
| **[09/25/2025]** 🎉 Our Paper released, see [RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training](https://arxiv.org/abs/2509.21009)                                                                                                                                                                                                    |
| **[09/24/2025]** 🎉 Support [Wan2_2 Reward FL pipeline](examples/wan2.2-14B-reward_fl_ds/reward_fl_config.yaml). Explore the new capabilities!                                                                                                                                                                                                                      |
| **[09/23/2025]** 🎉 ROLL aligns with GEM environment definition, providing agentic Tool Use training capabilities, [ToolUse docs](docs_roll/docs/English/UserGuide/agentic/Tool_Use.md).                                                                                                                                                                            |
| **[09/16/2025]** 🎉 Qwen3-Next model training is supported, refer to [configuration](examples/qwen3-next-80BA3B-rlvr_megatron/rlvr_config.yaml).                                                                                                                                                                                                                    |
| **[09/04/2025]** 🎉 ROLL supports vLLM dynamic FP8 rollout and remove_padding for acceleration.                                                                                                                                                                                                                                                                     |
| **[08/28/2025]** 🎉 ROLL supports SFT pipeline, refer to [configuration](examples/qwen2.5-7B-sft_megatron/sft_config.yaml).                                                                                                                                                                                                                                         |
| **[08/13/2025]** 🎉 ROLL supports AMD GPUs with out-of-box image docker and Dockerfile and specific yamls under `examples/` directory. Please refer to [Installation](https://alibaba.github.io/ROLL/docs/Getting%20Started/Installation/).                                                                                                                         |
| **[08/11/2025]** 🎉 Our Paper released, see [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning](https://arxiv.org/abs/2508.08221).                                                                                                                                                                                                                     |
| **[08/10/2025]** 🎉 Agentic RL supports [stepwise learning](examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_gigpo.yaml), like [GiGPO](https://arxiv.org/abs/2505.10978); Distill supports [VLM](examples/qwen2.5-vl-7B-distill/distill_vl_megatron.yaml). Explore the new capabilities!                                                                         |
| **[08/06/2025]** 🎉 ROLL PPT is now available, [Slides](assets/ROLL%20高效且用户友好的大模型RL训练框架.pdf).                                                                                                                                                                                                                                                                       |
| **[07/31/2025]** 🎉 Refactor agentic rl design. Support agentic rl [async training](examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_async.yaml). Explore the new capabilities!                                                                                                                                                                                  |
| **[07/31/2025]** 🎉 Support [DistillPipeline](examples/qwen2.5-7B-distill_megatron/run_distill_pipeline.sh)/[DpoPipeline](examples/dpo_examples/run_dpo_pipeline.sh). Support [lora](examples/qwen2.5-7B-rlvr_megatron/rlvr_lora_zero3.yaml). Support [GSPO](https://arxiv.org/abs/2507.18071)                                                                      |
| **[06/25/2025]** 🎉 Support thread env for env scaling and support [qwen2.5 VL agentic pipeline](examples/qwen2.5-vl-3B-agentic/agentic_val_sokoban.yaml).                                                                                                                                                                                                          |
| **[06/13/2025]** 🎉 Support [Qwen2.5 VL rlvr pipeline](examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml) and upgrade mcore to 0.12 version.                                                                                                                                                                                                                           |
| **[06/09/2025]** 🎉 ROLL tech report is now available! Access the report [here](https://arxiv.org/abs/2506.06122).                                                                                                                                                                                                                                                  |
| **[06/08/2025]** 🎉Supports  Qwen3([8B](examples/qwen3-8B-rlvr_megatron/rlvr_config.yaml)/14B/32B), Qwen3-MoE([30A3](examples/qwen3-30BA3B-rlvr_megatron/rlvr_config.yaml)/[235A22](examples/qwen3-235BA22B-rlvr_megatron/rlvr_config.yaml)), Qwen2.5([7B](examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml)/14B/32B/72B) LLM models.                             |
| **[05/30/2025]** 🎉 Training [RLVR](examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml) and [Agentic RL](examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml) with ROLL is now available! Explore the new capabilities.                                                                                                                                        |
---


## 🚀 Get Started

[Documents](https://alibaba.github.io/ROLL/)

### Quick Start
[Installation](https://alibaba.github.io/ROLL/docs/Getting%20Started/Installation/)  
[Config System Explanation](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/config_system)  
[Debugging Guide](https://alibaba.github.io/ROLL/docs/Getting%20Started/Debugging%20Guide/debug_guide)  
[Trackers and Metrics](https://alibaba.github.io/ROLL/docs/User%20Guides/Tracker%20&%20Metrics/trackers_and_metrics)  
[Checkpoint Saving and Resuming Guide](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/checkpoint_and_resume)  
[Converting MCoreAdapter Models to Hugging Face Format](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/megatron_convert_2_hf)  
[Quick Start: Single-Node Deployment Guide](https://alibaba.github.io/ROLL/docs/Getting%20Started/Quick%20Start/single_node_quick_start)  
[Quick Start: Multi-Node Deployment Guide](https://alibaba.github.io/ROLL/docs/Getting%20Started/Quick%20Start/multi_nodes_quick_start)  
[Quick Start: Using Alibaba Cloud Function Compute DevPod for Rapid Development](https://alibaba.github.io/ROLL/docs/Getting%20Started/Quick%20Start/aliyun_serverless_devpod_quick_start)
[Frequently Asked Questions](https://alibaba.github.io/ROLL/docs/Getting%20Started/FAQ/qa_issues)

### UserGuide

#### Pipeline Step by Step
[RLVR Pipeline](https://alibaba.github.io/ROLL/docs/User%20Guides/Pipeline/rlvr_pipeline_start)  
[Agentic Pipeline](https://alibaba.github.io/ROLL/docs/User%20Guides/Pipeline/agentic_pipeline_start)  
[Agentic Comprehensive Guide](https://alibaba.github.io/ROLL/docs/User%20Guides/Pipeline/agent_pipeline_start)  
[Distill Pipeline](https://alibaba.github.io/ROLL/docs/User%20Guides/Pipeline/distill_pipeline_start)

#### Algorithms
[Reinforce++](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/Reinforce_Plus_Plus)  
[TOPR](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/TOPR)  
[GiGPO](https://alibaba.github.io/ROLL/docs/User%20Guides/Agentic/agentic_GiGPO)  
[PPO](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/PPO)  
[Lite PPO](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/LitePPO)  
[GRPO](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/GRPO)  
[GSPO](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/GSPO)  
[RAFT++](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/RAFT_Plus_Plus)  
[StarPO](https://alibaba.github.io/ROLL/docs/User%20Guides/Agentic/agentic_StarPO)   
[RewardFL](https://alibaba.github.io/ROLL/docs/User%20Guides/Algorithms/Reward_FL)

#### Backend
[DeepSeed](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/deepspeed)  
[Megatron](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/megatron)   
[vLLM](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/vllm)  
[SGLang](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/sglang)

#### Advanced Features
[Asynchronous Parallel Rollout](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/async_parallel_rollout)  
[Asynchronous Training Feature](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/async_training)  

#### Performance Optimization & Resource Management 
[Resource Config](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/device_mapping)   
[GPU Time-Division Multiplexing Control](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/offload_reload_control)  

#### ROLL x Ascend
[Ascend Usage Guide](https://alibaba.github.io/ROLL/docs/User%20Guides/Hardware%20Support/ascend_usage)

---

## ✨ Key Features
*   **Multi-task RL Training (RLVR):** Covers mathematics, coding, general reasoning, open-ended Q&A, instruction following, etc.
    *   Flexible `domain_batch_size` distribution control.
    *   **Sample-level asynchronous parallel Rollout**, asynchronous reward calculation, and dynamic sampling.
    *   Asynchronous training under implementation.
*   **Agentic RL:** Multi-turn interaction capabilities for games, multi-turn dialogues, tool use, etc.
    *   Environment-level **asynchronous parallel rollout**.
    *   Supports **asynchronous training**.
    *   Multi-turn interaction rollout supports **local debugging**, improving multi-turn interaction business development efficiency.
    *   Supports **TrajectoryWise (StartPO)** and **StepWise (GiGPO)** training paradigms.
*   **Algorithm-Friendly:** Provides flexible and rich RL strategy configurations by default.
    *   Over 20 rich reinforcement learning strategy options, such as reward normalization, reward clipping, various advantage estimation methods, etc.
    *   Out-of-the-box support for reinforcement learning algorithms, such as **PPO, GRPO, Reinforce++, TOPR, RAFT++, GSPO**, etc.
*   **Rich Training and Inference Engine:** Ray-based multi-role distributed architecture; Strategy abstraction unifies various backends, enabling easy operation from single machines to thousands-of-GPU clusters.
    *   Inference/Generation supports vLLM, SGLang.
    *   Training supports DeepSpeed (ZeRO), Megatron-LM 5D parallelism (mcore-adapter, dp/tp/pp/cp/ep), FSDP under implementation.
    *   Extreme offload/reload capabilities.
    *   Supports [LoRA](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/lora) training.
    *   Supports FP8 rollout (FP8 inference for LLM as judge, FP8 rollout with BF16 training under development).
*   **AutoDeviceMapping:** Supports custom device mapping for different roles, flexibly managing colocated and disaggregated deployments.
*   **Observability:** Integrated with SwanLab / WandB / TensorBoard, tracking of performance for each domain and reward type.
*   **Rich Post-training Technical Support:**
    *   Agentic RL LLM & VLM
    *   RLVR LLM & VLM
    *   Distill Pipeline LLM & VLM
    *   DPO Pipeline
    *   SFT Pipeline under development

---

## 🏆 Notable work based on ROLL
- [ComplementaryRL](https://arxiv.org/abs/2603.17621): Complementary RL is a learning framework that enables agents to effectively learn from experience through the seamless co-evolution of an experience extractor and a policy actor within the RL optimization loop.
- [RLix](https://github.com/rlops/rlix): RLix is an RL job manager that lets more RL jobs run concurrently with less waiting by sharing GPU capacity across jobs, while preserving each pipeline’s training behavior and improving GPU utilization.
- [TurningPoint-GRPO](https://arxiv.org/abs/2602.06422): A GRPO framework for Flow Matching models in text-to-image generation that alleviates step-wise reward sparsity by modeling step-level incremental rewards and explicitly captures long-term effects via turning points detection, providing dense learning signals for each denoising action.
- [STAgent](https://arxiv.org/abs/2512.24957): An agentic LLM specialized for spatio-temporal understanding and complex tasks like constrained POI discovery and itinerary planning, featuring hierarchical data curation with 1:10,000 filter ratio and cascaded training (seed SFT + difficulty-aware SFT + RL), achieving strong performance on TravelBench while preserving general capabilities.
- [IPRO](https://arxiv.org/abs/2510.14255): A novel video diffusion framework using reinforcement learning to enhance identity preservation in human-centric I2V generation, optimizing diffusion models with face identity scorer and KL-divergence regularization.
- [TaoSR-SHE](https://arxiv.org/abs/2510.07972): Stepwise Hybrid Examination Reinforcement Learning Framework for Taobao Search Relevance, with SRPO (hybrid reward model + offline verifier), diversified data filtering, and multi-stage curriculum learning.
- [EARL](https://arxiv.org/abs/2510.05943): Efficient Agentic RL Systems for LLMs, introducing a dynamic parallelism selector and a layout-aware data dispatcher to boost throughput, reduce memory and data movement bottlenecks, enabling stable large-scale agentic RL without hard context-length limits.
- [LiveThinking](https://arxiv.org/abs/2510.07685): Real-time reasoning for AI-powered livestreaming by distilling a 670B teacher LLM to a 30B MoE (3B active) via Rejection Sampling Fine-Tuning, then compressing reasoning with GRPO; delivers sub-second latency and ~30x compute reduction, with gains in response correctness (3.3%), helpfulness (21.8%), and GMV in Taobao Live.
- [TaoSR-AGRL](https://www.arxiv.org/abs/2510.08048): Adaptive Guided Reinforcement Learning for LLM-based e-commerce relevance, introducing Rule-aware Reward Shaping and Adaptive Guided Replay to improve long-horizon reasoning, rule adherence, and training stability in Taobao Search; deployed in main search handling hundreds of millions of users.
- [RecGPT](https://www.arxiv.org/abs/2507.22879): a next-generation, LLM-driven framework that places user intent at the core of recommender systems, fostering a more sustainable and mutually beneficial ecosystem.
- [TaoSR1](https://arxiv.org/abs/2508.12365): A novel LLM framework directly deploying Chain-of-Thought (CoT) reasoning for e-commerce query-product relevance prediction, overcoming deployment challenges for superior performance.
- [AIGB-Pearl](https://www.arxiv.org/abs/2509.15927): a novel auto-bidding method that integrates generative planning and policy optimization, utilizing an LLM-enhanced trajectory evaluator to iteratively refine bidding strategies for state-of-the-art advertising performance.
-----

## 🙏 Citation and Acknowledgement

ROLL is inspired by the design of OpenRLHF, VeRL, Nemo-Aligner, and RAGEN.
The project is developed by Alibaba TAOBAO & TMALL Group and Alibaba Group. The code is distributed under the Apache License (Version 2.0). This product contains various third-party components under other open-source licenses. See the `NOTICE` file for more information.

The following repositories have been used in ROLL, either in their close-to-original form or as an inspiration:

  * [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  * [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
  * [sgl-project/sglang](https://github.com/sgl-project/sglang)
  * [vllm-project/vllm](https://github.com/vllm-project/vllm)
  * [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

If you use ROLL in your research or project, please consider citing us:

```bibtex
@article{wang2025reinforcement,
  title={Reinforcement Learning Optimization for Large-Scale Learning: An Efficient and User-Friendly Scaling Library},
  author={Wang, Weixun and Xiong, Shaopan and Chen, Gengru and Gao, Wei and Guo, Sheng and He, Yancheng and Huang, Ju and Liu, Jiaheng and Li, Zhendong and Li, Xiaoyang and others},
  journal={arXiv preprint arXiv:2506.06122},
  year={2025}
}
```



-----

## 🤝 About [ROCK & ROLL Team]
ROLL is a project jointly developed by Taotian Future Living Lab and Alibaba AI Engine Team, with a strong emphasis on pioneering the future of Reinforcement Learning (RL). Our mission is to explore and shape innovative forms of future living powered by advanced RL technologies. If you are passionate about the future of RL and want to be part of its evolution, we warmly welcome you to join us!👇

<a href="./assets/roll_wechat.png" target="_blank">
  <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
</a>
<a href="./assets/future_lab.png" target="_blank">
  <img src="https://img.shields.io/twitter/follow/FutureLab2025?style=social" alt="X QR">
</a>

-----
We are HIRING! 
- Post Training Infra 研发工程师 [JD link](https://talent-holding.alibaba.com/off-campus/position-detail?lang=zh&positionId=7000016304)
- 大模型训练专家： 
  - （社招）[JD link](https://talent.taotian.com/off-campus/position-detail?lang=zh&positionId=7000024203)
  - （校招）[JD link](https://talent.taotian.com/campus/position-detail?positionId=199900140053)
- Infra 研究型实习生 [JD link](https://talent-holding.alibaba.com/campus/position-detail?lang=zh&positionId=59900004115)

-----

<div align="center">
We welcome contributions from the community! 🤝
</div>
