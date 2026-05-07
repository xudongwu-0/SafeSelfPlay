"""Standalone arena evaluation for FSP two-player games.

Two modes:
  local       Boot local vLLM inference workers (requires GPU).
  server_api  Use an external OpenAI-compatible API server (CPU node OK).
              Requires --api_key, --base_url, and --model_name at runtime.
              Credentials are NEVER stored in config files.

Usage:
    # Local vLLM mode
    python examples/start_arena_eval.py --mode local \
        --config_name agent_kuhn_poker_fsp_train \
        --checkpoint_dir /path/to/render/timestamp/ \
        --output_dir ./arena_eval_output \
        --episodes_per_pair 16 --save_trajectories

    # External API mode (no GPU needed)
    python examples/start_arena_eval.py --mode server_api \
        --api_key <key> --base_url <url> --model_name <model> \
        --config_name agent_kuhn_poker_arena_api \
        --self_play --env_tag KuhnPokerLLMThink \
        --output_dir ./arena_eval_output \
        --episodes_per_pair 12 --save_trajectories
"""

import argparse
import os
import re
from typing import List, Optional

import ray
from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.distributed.scheduler.router import RouterManager
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.arena_eval import (
    log_payoff_matrix,
    run_arena_evaluation,
    save_payoff_matrix,
    save_trajectories_jsonl,
)
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()


def discover_checkpoints(
    checkpoint_dir: str, include_base: bool = True, max_n: Optional[int] = None,
) -> List[Optional[str]]:
    """Find LoRA checkpoint dirs (containing adapter_config.json), sorted by checkpoint number."""
    checkpoints = []
    for dirpath, _, filenames in os.walk(checkpoint_dir):
        if "adapter_config.json" in filenames:
            checkpoints.append(dirpath)

    def _sort_key(path: str) -> int:
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=_sort_key)
    if max_n is not None:
        checkpoints = checkpoints[-max_n:]

    lora_paths: List[Optional[str]] = []
    if include_base:
        lora_paths.append(None)
    lora_paths.extend(checkpoints)
    return lora_paths


@ray.remote
def _download_models(model_name_or_paths: set):
    from concurrent import futures
    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures.wait([executor.submit(download_model, p) for p in model_name_or_paths])


def main():
    parser = argparse.ArgumentParser(description="Standalone arena evaluation for FSP models")
    parser.add_argument("--mode", choices=["local", "server_api"], default="local",
                        help="'local': use local vLLM (GPU). 'server_api': use external OpenAI-compatible API (CPU ok).")
    # server_api credentials — never stored in config files
    parser.add_argument("--api_key", default=None, help="[server_api] API key for external inference server.")
    parser.add_argument("--base_url", default=None, help="[server_api] Base URL of external inference server.")
    parser.add_argument("--model_name", default=None, help="[server_api] Model name on external server.")
    parser.add_argument("--config_path", default="agentic_demo")
    parser.add_argument("--config_name", default="agent_kuhn_poker_fsp_train")
    parser.add_argument("--checkpoint_dir", default=None, help="Dir to search for LoRA checkpoints")
    parser.add_argument("--output_dir", default="./arena_eval_output")
    parser.add_argument("--episodes_per_pair", type=int, default=16)
    parser.add_argument("--max_concurrent", type=int, default=32)
    parser.add_argument("--no_base_model", action="store_true", help="Exclude base model from eval")
    parser.add_argument("--max_checkpoints", type=int, default=None)
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--env_tag", default=None, help="Override env tag (default: first in config)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--self_play", action="store_true",
                        help="Run base model vs itself (diagnostic, no checkpoints needed)")
    args, overrides = parser.parse_known_args()

    if args.mode == "server_api":
        missing = [f for f, v in [("--api_key", args.api_key), ("--base_url", args.base_url), ("--model_name", args.model_name)] if not v]
        if missing:
            parser.error(f"server_api mode requires: {', '.join(missing)}")

    # --- Load config ---
    initialize(config_path=args.config_path, job_name="arena_eval")
    cfg = compose(config_name=args.config_name, overrides=overrides)
    pipeline_config: AgenticConfig = from_dict(
        data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True),
    )

    # Override: use all GPUs for inference (recalculate world_size)
    num_gpus = pipeline_config.num_gpus_per_node
    pipeline_config.actor_infer.device_mapping = list(range(num_gpus))
    num_gpus_per_worker = pipeline_config.actor_infer.num_gpus_per_worker or 1
    pipeline_config.actor_infer.world_size = num_gpus // num_gpus_per_worker
    # Arena eval is single-node and only uses actor_infer; ignore train device_mapping
    pipeline_config.num_nodes = 1

    # --- Discover checkpoints ---
    if args.self_play:
        lora_paths: List[Optional[str]] = [None, None]
    else:
        if not args.checkpoint_dir:
            logger.error("--checkpoint_dir is required unless --self_play is set")
            return
        lora_paths = discover_checkpoints(
            args.checkpoint_dir, include_base=not args.no_base_model, max_n=args.max_checkpoints,
        )
    labels = ["base_model" if p is None else os.path.basename(p) for p in lora_paths]
    logger.info(f"Discovered {len(lora_paths)} models: {labels}")
    if len(lora_paths) < 2:
        logger.error("Need at least 2 models for arena evaluation")
        return

    # Load tokenizer (always needed for prompt encoding / response decoding)
    tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_infer.model_args)

    if args.mode == "server_api":
        # Inject API credentials into pipeline config at runtime — not stored in any file
        from roll.pipeline.agentic.agentic_config import LLMProxyConfig
        api_proxy = LLMProxyConfig(
            proxy_type="openai",
            proxy_config={
                "base_url": args.base_url,
                "api_key": args.api_key,
                "model_name": args.model_name,
                "timeout": 60,
                "max_retries": 3,
                "retry_delay": 2,
            },
        )
        pipeline_config.train_env_manager.llm_proxy = api_proxy
        pipeline_config.val_env_manager.llm_proxy = api_proxy
        logger.info(f"server_api mode: base_url={args.base_url}, model={args.model_name}")
        generate_scheduler = None
    else:
        # --- Local mode: init Ray + vLLM inference cluster ---
        init()

        resource_manager = ResourceManager(
            num_nodes=pipeline_config.num_nodes, num_gpus_per_node=num_gpus,
        )

        actor_infer = Cluster(
            name=pipeline_config.actor_infer.name,
            worker_cls=pipeline_config.actor_infer.worker_cls,
            resource_manager=resource_manager,
            worker_config=pipeline_config.actor_infer,
        )

        # Download base model
        model_names = set()
        if pipeline_config.actor_infer.model_args.model_name_or_path:
            model_names.add(pipeline_config.actor_infer.model_args.model_name_or_path)
        if model_names:
            for pg_list in actor_infer.placement_groups:
                if pg_list:
                    pg = pg_list[0]["placement_group"]
                    ray.get(
                        _download_models.options(
                            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
                        ).remote(model_name_or_paths=model_names)
                    )
                    break

        # Initialize vLLM workers
        logger.info("Initializing vLLM inference workers...")
        ray.get(actor_infer.initialize(pipeline_config=pipeline_config, blocking=False))

        # Create RouterManager directly (generate_scheduler for arena env managers)
        generate_scheduler = ray.remote(RouterManager).options(
            name="RouterManager-arena-eval",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False,
            ),
            max_concurrency=args.max_concurrent + 1,
        ).remote(
            actor_cluster=actor_infer,
            router_args=pipeline_config.router_args,
            num_gpus_per_node=num_gpus,
        )
        ray.get(generate_scheduler.initialize.remote())
        ray.get(generate_scheduler.resume.remote())

    # --- Run arena evaluation ---
    env_tag = args.env_tag or list(pipeline_config.custom_envs.keys())[0]
    logger.info(f"Starting arena evaluation: env={env_tag}, episodes_per_pair={args.episodes_per_pair}")

    result = run_arena_evaluation(
        lora_paths=lora_paths,
        generate_scheduler=generate_scheduler,
        pipeline_config=pipeline_config,
        tokenizer=tokenizer,
        env_tag=env_tag,
        episodes_per_pair=args.episodes_per_pair,
        max_concurrent=args.max_concurrent,
        seed_base=args.seed,
        save_trajectories=args.save_trajectories,
    )

    if args.save_trajectories:
        payoff_matrix, trajectories = result
        save_trajectories_jsonl(trajectories, args.output_dir)
    else:
        payoff_matrix = result

    # --- Save results ---
    log_payoff_matrix(payoff_matrix, lora_paths)
    save_payoff_matrix(payoff_matrix, lora_paths, args.output_dir)

    logger.info("Arena evaluation complete!")

    # Shutdown
    if generate_scheduler is not None:
        ray.shutdown()


if __name__ == "__main__":
    main()
