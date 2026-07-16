"""Arena evaluation: pairwise payoff matrix for LoRA models after FSP training."""

import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import ray
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.two_player_traj_env_manager import TwoPlayerTrajEnvManager
from roll.utils.logging import get_logger

logger = get_logger()

ARENA_SRC_RANK_BASE = 500000


def _get_model_label(lora_path: Optional[str]) -> str:
    return "base_model" if lora_path is None else os.path.basename(lora_path)


def _build_trajectory(
    env_manager: TwoPlayerTrajEnvManager,
    tokenizer: PreTrainedTokenizer,
    player_i_lora: Optional[str],
    player_j_lora: Optional[str],
    seed: int,
    payoff: float,
) -> dict:
    """Build trajectory dict from env_manager state after an episode, matching training log format."""
    history = env_manager.rollout_cache.history
    turns = []
    scores = []
    for entry in history:
        if "prompt_ids" not in entry or "response_ids" not in entry:
            continue
        prompt = tokenizer.decode(entry["prompt_ids"], skip_special_tokens=False)
        response = tokenizer.decode(entry["response_ids"], skip_special_tokens=False)
        reward = entry.get("reward", 0.0)
        turns.append({"prompt": prompt, "response": response})
        scores.append(reward)

    # Extract behavioral metrics from completed history entries
    episode_metrics = {}
    for entry in history:
        if "metrics" in entry:
            episode_metrics = entry["metrics"]

    return {
        "player_i": _get_model_label(player_i_lora),
        "player_j": _get_model_label(player_j_lora),
        "seed": seed,
        "payoff": payoff,
        "turns": turns,
        "episode_score": sum(scores),
        "step_scores": scores,
        "metrics": episode_metrics,
    }


def _compute_safety_game_payoff(env_manager: TwoPlayerTrajEnvManager, default_payoff: float) -> float:
    """Return a zero-sum safety-game payoff for RedTeamSafety arena episodes.

    RedTeamSafety's training reward is role-specific and general-sum-ish. PSRO,
    however, needs a pairwise "who won this matchup?" signal from player_i's
    perspective. The env records agent_role for player_i and terminal safety
    metrics in the final info dict, so we remap:

    - player_i as defender: +1 if defender_success else -1
    - player_i as attacker: -1 if the rewrite violates the seed label;
      otherwise -1 if defender_success else +1

    Other environments keep their original payoff semantics.
    """
    env = getattr(env_manager, "env", None)
    if env is None or env.__class__.__name__ != "RedTeamSafetyEnv":
        return default_payoff

    final_info = {}
    for entry in getattr(env_manager.rollout_cache, "history", []):
        if "agent_role" in entry:
            final_info = entry

    role = final_info.get("agent_role", getattr(env, "agent_role", None))
    metrics = final_info.get("metrics", {}) or {}

    if role == "defender":
        if "defender_success" in metrics:
            return 1.0 if float(metrics["defender_success"]) > 0.5 else -1.0
    elif role == "attacker":
        if "attack_label_consistent" in metrics and float(metrics["attack_label_consistent"]) <= 0.5:
            return -1.0
        if "defender_success" in metrics:
            return -1.0 if float(metrics["defender_success"]) > 0.5 else 1.0
        if "attack_success" in metrics:
            return 1.0 if float(metrics["attack_success"]) > 0.5 else -1.0

    return default_payoff


def _create_arena_env_manager(
    pipeline_config: AgenticConfig,
    env_tag: str,
    tokenizer: PreTrainedTokenizer,
    generate_scheduler,
    env_id: int = 0,
    env_config_overrides: Optional[Dict] = None,
) -> TwoPlayerTrajEnvManager:
    """Create a lightweight TwoPlayerTrajEnvManager for arena evaluation."""
    cfg_template = pipeline_config.custom_envs[env_tag]
    config = dict(cfg_template.get("env_config", {}))
    if env_config_overrides:
        config.update(env_config_overrides)
    env_config = DictConfig({
        **cfg_template,
        "tag": env_tag,
        "group_id": 0,
        "env_id": env_id,
        "config": config,
        "env_class": cfg_template.env_type,
        "env_manager_cls": cfg_template.get(
            "env_manager_cls",
            "roll.pipeline.agentic.env_manager.two_player_traj_env_manager.TwoPlayerTrajEnvManager",
        ),
        "group_seed": 0,
    })
    # Remove nested env_config to avoid duplication (already flattened into "config")
    if "env_config" in env_config:
        del env_config["env_config"]

    worker_config = pipeline_config.train_env_manager
    return TwoPlayerTrajEnvManager(
        worker_config=worker_config,
        pipeline_config=pipeline_config,
        env_config=env_config,
        tokenizer=copy.deepcopy(tokenizer),
        generate_scheduler=generate_scheduler,
        output_queue=None,  # Not used — we bypass reset()
        thread_lock=threading.Lock(),
        mode="val",
    )


def _handle_arena_opponent_first(
    env_manager: TwoPlayerTrajEnvManager,
    opponent_lora: Optional[str],
    generate_scheduler,
    tokenizer: PreTrainedTokenizer,
    pipeline_config: AgenticConfig,
    worker_config: EnvManagerConfig,
    src_rank_base: int,
) -> None:
    """Generate opponent's first action when env signals opponent_first=True."""
    obs_for_opponent = env_manager.rollout_cache.history[-1]["observation"]

    opponent_lm_input = env_manager._format_opponent_messages(obs_for_opponent)
    opponent_input_ids = opponent_lm_input.batch["input_ids"]
    opponent_messages = [m for step in env_manager.opponent_history for m in (step.get("messages") or [])]

    max_new_tokens = min(
        env_manager.env_config["max_tokens_per_step"],
        worker_config.generating_args.max_new_tokens,
        pipeline_config.sequence_length - opponent_input_ids.shape[1],
    )
    generation_config = worker_config.generating_args.to_dict()
    generation_config["max_new_tokens"] = min(max_new_tokens, pipeline_config.sequence_length)
    opponent_lm_input.meta_info["src_rank"] = src_rank_base + 1
    opponent_lm_input.meta_info["lora_name"] = opponent_lora

    opponent_output: DataProto = env_manager.llm_proxy.generate(
        messages=opponent_messages, lm_input=opponent_lm_input, generation_config=generation_config
    )

    if opponent_output is None:
        env_manager.rollout_cache.terminated = True
        return

    opponent_responses = tokenizer.batch_decode(
        opponent_output.batch['responses'], skip_special_tokens=False
    )
    opponent_action = opponent_responses[0]

    opponent_response_ids = opponent_output.batch['responses'][0].tolist()
    if env_manager.opponent_history:
        env_manager.opponent_history[-1]["response_ids"] = opponent_response_ids
        env_manager.opponent_history[-1]["messages"].append({
            "role": "assistant",
            "content": tokenizer.decode(opponent_response_ids, skip_special_tokens=True),
        })

    obs_for_agent, reward, terminated, truncated, info = env_manager.env.step(action=opponent_action)

    env_manager.rollout_cache.history[-1]["observation"] = obs_for_agent
    if "opponent_first" in env_manager.rollout_cache.history[-1]:
        del env_manager.rollout_cache.history[-1]["opponent_first"]

    if terminated:
        env_manager.rollout_cache.terminated = True


def play_episode(
    env_manager: TwoPlayerTrajEnvManager,
    player_i_lora: Optional[str],
    player_j_lora: Optional[str],
    generate_scheduler,
    tokenizer: PreTrainedTokenizer,
    pipeline_config: AgenticConfig,
    worker_config: EnvManagerConfig,
    seed: int,
    src_rank_base: int,
    save_trajectory: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """Play one episode between player_i and player_j. Returns payoff (EV) from player_i's perspective.

    Reuses format_messages() and step() from TwoPlayerTrajEnvManager for identical
    prompt formatting and opponent generation logic as during rollout.
    """
    max_steps = env_manager.env_config.max_steps

    # Reset env_manager state (bypass output_queue-based reset)
    env_manager.current_opponent_lora = player_j_lora
    env_manager.opponent_history = []
    env_manager.rollout_cache = RolloutCache(env_id=0, group_id=0, tag="arena")
    observation, info = env_manager.env.reset(seed=seed)
    env_manager.rollout_cache.history = [{
        "observation": observation,
        "actions_left": max_steps - env_manager.rollout_cache.step,
        "messages": None,
        **info,
    }]

    # Handle opponent-first (e.g., agent assigned as second mover in Kuhn Poker)
    if info.get("opponent_first", False):
        _handle_arena_opponent_first(
            env_manager, player_j_lora, generate_scheduler, tokenizer,
            pipeline_config, worker_config, src_rank_base,
        )

    while not env_manager.rollout_cache.terminated:
        # --- Agent 0 (player_i): format_messages (identical to rollout) ---
        lm_input = env_manager.format_messages(env_manager.rollout_cache)
        input_ids = lm_input.batch["input_ids"]
        agent_messages = [m for step in env_manager.rollout_cache.history for m in (step.get("messages") or [])]

        max_new_tokens = min(
            env_manager.env_config["max_tokens_per_step"],
            worker_config.generating_args.max_new_tokens,
            pipeline_config.sequence_length - input_ids.shape[1],
        )
        generation_config = worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, pipeline_config.sequence_length)

        lm_input.meta_info["src_rank"] = src_rank_base
        lm_input.meta_info["lora_name"] = player_i_lora  # KEY: explicit LoRA for agent 0

        lm_output: DataProto = env_manager.llm_proxy.generate(
            messages=agent_messages, lm_input=lm_input, generation_config=generation_config
        )

        if lm_output is None:
            break

        # Post-process: store response_ids + messages (same as make_decision)
        response_ids = lm_output.batch['responses'][0].tolist()
        content = env_manager.rollout_cache.history[-1]
        content["response_ids"] = response_ids
        content["messages"].append({
            "role": "assistant",
            "content": tokenizer.decode(response_ids, skip_special_tokens=True),
        })
        lm_output.meta_info["stop_reason"] = "finish"

        # --- Agent 1 (player_j): handled by step() with current_opponent_lora ---
        rollout_cache = env_manager.step(lm_output)

        if rollout_cache.terminated:
            break

    default_payoff = env_manager.env.final_reward if env_manager.env.step_count > 0 else 0.0
    payoff = _compute_safety_game_payoff(env_manager, default_payoff)

    if save_trajectory:
        traj = _build_trajectory(env_manager, tokenizer, player_i_lora, player_j_lora, seed, payoff)
        return payoff, traj
    return payoff


def run_arena_evaluation(
    lora_paths: List[Optional[str]],
    generate_scheduler,
    pipeline_config: AgenticConfig,
    tokenizer: PreTrainedTokenizer,
    env_tag: str,
    episodes_per_pair: int = 4,
    max_concurrent: int = 32,
    seed_base: int = 12345,
    save_trajectories: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[dict]]]:
    """Run pairwise matches between all LoRA models and return payoff matrix.

    Args:
        lora_paths: List of LoRA checkpoint paths. None = base model.
        generate_scheduler: Ray actor handle to RequestScheduler.
        pipeline_config: Pipeline configuration.
        tokenizer: Tokenizer for prompt encoding.
        env_tag: Which custom_env config to use.
        episodes_per_pair: Number of episodes per (i, j) matchup.
        max_concurrent: Max concurrent episodes.
        seed_base: Base seed for reproducibility.

    Returns:
        N×N numpy array where M[i][j] = payoff (expected chips won) of model i against model j.
    """
    n = len(lora_paths)
    payoff_matrix = np.full((n, n), 0.0)  # diagonal = 0.0 (self-play EV)
    worker_config = pipeline_config.train_env_manager

    # Pre-create env_managers (one per concurrent slot)
    logger.info(f"Arena: creating {max_concurrent} env managers for concurrent evaluation...")
    env_managers = []
    for idx in range(max_concurrent):
        em = _create_arena_env_manager(
            pipeline_config, env_tag, tokenizer, generate_scheduler, env_id=idx,
        )
        env_managers.append(em)

    # Assert full state coverage if env exposes NUM_START_STATES
    _num_start_states = getattr(env_managers[0].env, "NUM_START_STATES", None)
    if _num_start_states is not None and episodes_per_pair % _num_start_states != 0:
        logger.warning(
            f"episodes_per_pair ({episodes_per_pair}) not divisible by "
            f"NUM_START_STATES ({_num_start_states}); state coverage will be uneven."
        )

    # Build task list: (i, j, episode_idx)
    tasks = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for ep in range(episodes_per_pair):
                tasks.append((i, j, ep))

    logger.info(f"Arena: {n} models, {len(tasks)} total episodes, {episodes_per_pair} per pair")

    # Results accumulator: {(i, j): [win_rates]}
    results: Dict[tuple, list] = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                results[(i, j)] = []

    all_trajectories: List[dict] = []

    # Run episodes concurrently
    failure_count = 0
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Map from future to (i, j, ep, env_manager_idx)
        future_to_info = {}
        em_available = list(range(max_concurrent))
        pending_tasks = list(tasks)

        def submit_batch():
            while pending_tasks and em_available:
                i, j, ep = pending_tasks.pop(0)
                em_idx = em_available.pop(0)
                seed = seed_base + ep
                src_rank = ARENA_SRC_RANK_BASE + em_idx * 2
                future = executor.submit(
                    play_episode,
                    env_managers[em_idx],
                    lora_paths[i],
                    lora_paths[j],
                    generate_scheduler,
                    tokenizer,
                    pipeline_config,
                    worker_config,
                    seed,
                    src_rank,
                    save_trajectories,
                )
                future_to_info[future] = (i, j, ep, em_idx)

        submit_batch()

        completed = 0
        total = len(tasks)
        while future_to_info:
            done_futures = []
            for future in as_completed(future_to_info):
                done_futures.append(future)
                break  # process one at a time to resubmit

            for future in done_futures:
                i, j, ep, em_idx = future_to_info.pop(future)
                try:
                    result = future.result()
                    if save_trajectories:
                        payoff, traj = result
                        all_trajectories.append(traj)
                    else:
                        payoff = result
                    results[(i, j)].append(payoff)
                except Exception as e:
                    logger.exception(f"Arena episode ({i},{j}) ep={ep} failed: {e}")
                    results[(i, j)].append(0.0)  # fallback
                    failure_count += 1

                em_available.append(em_idx)
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Arena: {completed}/{total} episodes completed")

                submit_batch()

    if failure_count:
        logger.warning(
            f"Arena: {failure_count}/{total} episodes failed and were filled with 0.0; "
            f"payoff matrix may be biased."
        )

    # Compute mean payoffs
    for (i, j), payoffs in results.items():
        payoff_matrix[i][j] = np.mean(payoffs) if payoffs else 0.0

    if pipeline_config.arena_log_ci_table:
        log_payoff_ci_table(results, lora_paths)

    if save_trajectories:
        return payoff_matrix, all_trajectories
    return payoff_matrix


def _compute_95ci(payoffs: List[float]) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using normal approximation."""
    n = len(payoffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(payoffs, dtype=float)
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    margin = 1.96 * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def log_payoff_ci_table(
    raw_results: Dict[Tuple[int, int], List[float]],
    lora_paths: List[Optional[str]],
) -> None:
    """Print a table of per-pair #runs and 95% CI for the arena payoff."""
    labels = [_get_model_label(p) for p in lora_paths]
    col_w = max(max(len(l) for l in labels), 8)

    logger.info("=" * 80)
    logger.info("Arena 95% CI Table (normal approximation, per matchup pair)")
    logger.info("=" * 80)
    header = f"{'player_i':<{col_w}}  {'player_j':<{col_w}}  {'#runs':>6}  {'mean':>8}  {'95% CI'}"
    logger.info(header)
    logger.info("-" * 80)
    for (i, j) in sorted(raw_results.keys()):
        payoffs = raw_results[(i, j)]
        mean, ci_low, ci_high = _compute_95ci(payoffs)
        n = len(payoffs)
        logger.info(
            f"{labels[i]:<{col_w}}  {labels[j]:<{col_w}}  {n:>6}  {mean:>+8.3f}  [{ci_low:+.3f}, {ci_high:+.3f}]"
        )
    logger.info("=" * 80)


def log_payoff_matrix(payoff_matrix: np.ndarray, lora_paths: List[Optional[str]]) -> None:
    """Pretty-print the payoff matrix."""
    labels = []
    for p in lora_paths:
        if p is None:
            labels.append("base")
        else:
            labels.append(os.path.basename(p))

    col_width = max(len(l) for l in labels) + 2
    col_width = max(col_width, 8)

    header = " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
    logger.info("=" * 60)
    logger.info("Arena Evaluation Payoff Matrix (row i vs col j = payoff EV of i)")
    logger.info("=" * 60)
    logger.info(header)
    for i, row in enumerate(payoff_matrix):
        row_str = f"{labels[i]:>{col_width}}" + "".join(f"{v:>{col_width}.3f}" for v in row)
        logger.info(row_str)
    logger.info("=" * 60)


def save_payoff_matrix(
    payoff_matrix: np.ndarray,
    lora_paths: List[Optional[str]],
    output_dir: str,
) -> str:
    """Save payoff matrix to JSON file. Returns the file path."""
    labels = []
    for p in lora_paths:
        if p is None:
            labels.append("base_model")
        else:
            labels.append(os.path.basename(p))

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "arena_payoff_matrix.json")
    data = {
        "labels": labels,
        "lora_paths": [p if p is not None else "base_model" for p in lora_paths],
        "payoff_matrix": payoff_matrix.tolist(),
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Arena payoff matrix saved to {filepath}")
    return filepath


def tracker_log_payoff_matrix(
    payoff_matrix: np.ndarray,
    lora_paths: List[Optional[str]],
    tracker,
    step: int,
) -> None:
    """Log payoff matrix to tracker: scalars for all trackers, Table for wandb."""
    from roll.utils.tracking import WandbTracker

    n = len(lora_paths)
    labels = [_get_model_label(p) for p in lora_paths]

    metrics = {}
    for i, label in enumerate(labels):
        off_diag = [payoff_matrix[i][j] for j in range(n) if i != j]
        metrics[f"arena/mean_payoff/{label}"] = float(np.mean(off_diag)) if off_diag else 0.0

    if isinstance(tracker, WandbTracker):
        import wandb
        columns = ["model"] + labels
        data = [[labels[i]] + [round(float(payoff_matrix[i][j]), 4) for j in range(n)] for i in range(n)]
        metrics["arena/payoff_matrix"] = wandb.Table(columns=columns, data=data)

    tracker.log(values=metrics, step=step)


def tracker_log_psro(
    payoff_matrix: np.ndarray,
    lora_paths: List[Optional[str]],
    nash_probs: Optional[np.ndarray],
    tracker,
    step: int,
) -> None:
    """Log PSRO payoff matrix and Nash opponent sampling probabilities to wandb as Tables."""
    from roll.utils.tracking import WandbTracker

    if not isinstance(tracker, WandbTracker):
        return

    import wandb
    n = len(lora_paths)
    labels = [_get_model_label(p) for p in lora_paths]

    metrics = {}
    columns = ["model"] + labels
    data = [[labels[i]] + [round(float(payoff_matrix[i][j]), 4) for j in range(n)] for i in range(n)]
    metrics["psro/payoff_matrix"] = wandb.Table(columns=columns, data=data)

    if nash_probs is not None and len(nash_probs) == n:
        prob_data = [[labels[i], round(float(nash_probs[i]), 4)] for i in range(n)]
        metrics["psro/opponent_prob"] = wandb.Table(columns=["policy", "prob"], data=prob_data)

    tracker.log(values=metrics, step=step)


def save_trajectories_jsonl(trajectories: List[dict], output_dir: str) -> str:
    """Save trajectories as JSONL (one JSON per line per episode). Returns file path."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "arena_trajectories.jsonl")
    with open(filepath, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    logger.info(f"Arena trajectories saved to {filepath} ({len(trajectories)} episodes)")
    return filepath
