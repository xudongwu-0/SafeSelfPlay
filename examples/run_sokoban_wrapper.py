"""Wrapper to run agentic pipeline with GPU/step overrides."""
import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.utils.import_utils import safe_import_class
from roll.utils.str_utils import print_pipeline_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config")
    parser.add_argument("--config_name", default="sppo_config")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--use_deepspeed", action="store_true",
                        help="Switch from megatron_train to deepspeed_train (avoids TE dependency)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Reduce batch sizes for quick smoke test")
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    # Apply overrides
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.num_gpus is not None:
        n = args.num_gpus
        device_map = f"list(range(0,{n}))"
        cfg.num_gpus_per_node = n
        cfg.actor_train.device_mapping = device_map
        cfg.actor_infer.device_mapping = device_map
        cfg.reference.device_mapping = device_map

    # Switch to deepspeed strategy (like the working demo config)
    if args.use_deepspeed:
        OmegaConf.set_struct(cfg, False)
        cfg.actor_train.strategy_args.strategy_name = "deepspeed_train"
        cfg.actor_train.strategy_args.strategy_config = cfg.deepspeed_zero2
        # Use sdpa attention (no flash-attn package needed)
        cfg.actor_train.model_args.attn_implementation = "sdpa"
        cfg.actor_infer.model_args.attn_implementation = "sdpa"
        cfg.reference.model_args.attn_implementation = "sdpa"
        # Use fp16 like the demo
        cfg.actor_train.model_args.dtype = "fp16"
        cfg.actor_infer.model_args.dtype = "fp16"
        cfg.reference.model_args.dtype = "fp16"
        OmegaConf.set_struct(cfg, True)

    # Reduce batch sizes for quick smoke test
    if args.smoke_test:
        OmegaConf.set_struct(cfg, False)
        cfg.rollout_batch_size = 16
        cfg.val_batch_size = 16
        # Keep group_size from config, just reduce num_env_groups
        train_tags = list(cfg.train_env_manager.tags)
        cfg.train_env_manager.num_env_groups = len(train_tags)
        cfg.train_env_manager.num_groups_partition = [1] * len(train_tags)
        cfg.train_env_manager.group_size = 1
        # Minimal val
        val_tags = list(cfg.train_env_manager.tags)
        cfg.val_env_manager.tags = val_tags
        cfg.val_env_manager.num_env_groups = len(val_tags)
        cfg.val_env_manager.num_groups_partition = [1] * len(val_tags)
        cfg.val_env_manager.group_size = 1
        cfg.actor_train.training_args.gradient_accumulation_steps = 1
        cfg.eval_steps = 100000  # skip eval to save time
        cfg.save_steps = 100000  # skip checkpoint to avoid disk errors
        OmegaConf.set_struct(cfg, True)

    # Override paths that don't exist on this cluster
    output_base = "/u/wchen11/ROLL/output"
    cfg.tracker_kwargs.log_dir = f"{output_base}/tensorboard/{args.config_name}"
    cfg.checkpoint_config.output_dir = f"{output_base}/checkpoints/{args.config_name}"

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    print_pipeline_config(ppo_config)

    pipeline_cls = getattr(cfg, "pipeline_cls", "roll.pipeline.agentic.agentic_pipeline.AgenticPipeline")
    if isinstance(pipeline_cls, str):
        pipeline_cls = safe_import_class(pipeline_cls)

    pipeline = pipeline_cls(pipeline_config=ppo_config)
    pipeline.run()


if __name__ == "__main__":
    main()
