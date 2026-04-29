"""Standalone arena evaluation for FSP two-player games.

Boots only vLLM inference workers (no training), discovers LoRA checkpoints,
runs pairwise arena matches, and saves payoff matrix + trajectories.

Usage:
    python examples/start_arena_eval.py \
        --config_name agent_kuhn_poker_fsp_train \
        --checkpoint_dir /path/to/render/timestamp/ \
        --output_dir ./arena_eval_output \
        --episodes_per_pair 16 \
        --save_trajectories
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
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.resource_manager import ResourceManager
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

    # --- Init Ray + inference cluster ---
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

    # Create RequestScheduler directly (no RolloutScheduler needed)
    generate_scheduler = RequestScheduler.options(
        name="RequestScheduler-arena-eval",
        get_if_exists=True,
        namespace=RAY_NAMESPACE,
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False,
        ),
        max_concurrency=args.max_concurrent + 1,
    ).remote(
        infer_cluster=actor_infer,
        pipeline_config=pipeline_config,
        resource_manager=resource_manager,
    )
    ray.get(generate_scheduler.resume.remote())

    # Load tokenizer
    tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_infer.model_args)

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
    ray.shutdown()


if __name__ == "__main__":
    main()
