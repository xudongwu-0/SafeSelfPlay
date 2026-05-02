import copy
import re
import random
from typing import Optional, List

import numpy as np

import ray
import torch
from tensordict import TensorDict
from threading import Lock

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, compute_conversation_end_token_id
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field


class TwoPlayerTrajEnvManager(TrajEnvManager):
    """Env manager for two-player zero-sum games with fictitious self-play.

    Extends TrajEnvManager to query the same vLLM server twice per round:
    - Agent 0 (training agent): uses current LoRA adapter
    - Agent 1 (opponent): sampled from enemy pool (base model or past LoRA checkpoints)

    Enemy pool starts as [None] (base model only). The pipeline adds past LoRA
    checkpoint paths to the pool at fsp_save_steps intervals.

    Only agent 0's trajectory data goes to training.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("opponent_generate_scheduler", None)
        super().__init__(*args, **kwargs)
        self.logger = get_logger()
        self.opponent_history: list[dict] = []
        # Fictitious self-play enemy pool: None = base model, str = LoRA checkpoint path
        self.enemy_pool: List[Optional[str]] = [None]
        self.current_opponent_lora: Optional[str] = None
        self.nash_probabilities: Optional[np.ndarray] = None

    def reset(self) -> Optional[RolloutCache]:
        result = super().reset()
        self.opponent_history = []
        # Sample opponent from enemy pool for this episode.
        # If nash_probabilities are set (PSRO mode), use np.random.choice with those weights.
        # Otherwise fall back to mode-based sampling (uniform / linear / exponential).
        n = len(self.enemy_pool)
        if self.nash_probabilities is not None and len(self.nash_probabilities) == n:
            idx = np.random.choice(n, p=self.nash_probabilities)
            self.current_opponent_lora = self.enemy_pool[idx]
        else:
            mode = getattr(self.pipeline_config, "fsp_opponent_weight_mode", "uniform")
            if n <= 1 or mode == "uniform":
                self.current_opponent_lora = random.choice(self.enemy_pool)
            else:
                if mode == "linear":
                    weights = [i + 1 for i in range(n)]
                elif mode == "exponential":
                    weights = [2 ** i for i in range(n)]
                else:
                    weights = [1] * n
                self.current_opponent_lora = random.choices(self.enemy_pool, weights=weights, k=1)[0]

        if result is not None and self.rollout_cache.history[-1].get("opponent_first", False):
            self._handle_opponent_first_move()

        return result

    def update_enemy_pool(self, lora_path: str):
        """Add a LoRA checkpoint to the enemy pool."""
        self.enemy_pool.append(lora_path)
        self.logger.info(f"Enemy pool updated: {len(self.enemy_pool)} opponents (added {lora_path})")

    def set_nash_probabilities(self, probs) -> None:
        """Set Nash mixed-strategy weights for opponent sampling. None reverts to mode-based sampling."""
        self.nash_probabilities = np.asarray(probs, dtype=float) if probs is not None else None
        if self.nash_probabilities is not None:
            self.logger.info(f"Nash probabilities updated: {self.nash_probabilities.tolist()}")
        else:
            self.logger.info("Nash probabilities cleared; reverting to mode-based sampling.")

    def _maybe_prepend_opponent_reasoning(self, observation: str, opponent_response: str) -> str:
        """If include_opponent_reasoning is enabled, prepend opponent's <think> block to observation."""
        if not getattr(self.worker_config, "include_opponent_reasoning", False):
            return observation
        match = re.search(r'<think>.*?</think>', opponent_response, re.DOTALL)
        if match:
            return f"{match.group(0)}\n{observation}"
        return observation

    def _normalize_opponent_action(self, action: str) -> str:
        """If opponent's action has invalid format, default to Pass."""
        action_info = self.env.parse_action(action)
        if action_info["action"] is None:
            return "<action>Pass</action>"
        return action

    def _get_opponent_action_override(self) -> Optional[str]:
        """Return a fixed action string for smoke testing, or None to use vLLM."""
        strategy = getattr(self.worker_config, "smoke_opponent_strategy", None)
        if strategy == "always_bet":
            return "<answer>Bet</answer>"
        return None

    def _handle_opponent_first_move(self):
        """Generate opponent's first action when the env signals opponent_first=True.

        Used for games where the agent is randomly assigned as the second mover
        (e.g., poker-P1 in Kuhn Poker). The opponent's first action is generated
        during reset so the agent's first format_messages sees the result.
        """
        obs_for_opponent = self.rollout_cache.history[-1]["observation"]

        override = self._get_opponent_action_override()
        if override is not None:
            opponent_action = override
        else:
            # Generate opponent's action via vLLM
            opponent_lm_input = self._format_opponent_messages(obs_for_opponent)
            opponent_input_ids = opponent_lm_input.batch["input_ids"]

            max_new_tokens = min(
                self.env_config["max_tokens_per_step"],
                self.worker_config.generating_args.max_new_tokens,
                self.pipeline_config.sequence_length - opponent_input_ids.shape[1],
            )
            generation_config = self.worker_config.generating_args.to_dict()
            generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
            opponent_lm_input.meta_info["src_rank"] = self.env_config["env_id"] + 100000
            opponent_lm_input.meta_info["lora_name"] = self.current_opponent_lora

            opponent_output: DataProto = self.llm_proxy.generate(
                messages=None, lm_input=opponent_lm_input, generation_config=generation_config
            )

            if opponent_output is None:
                self.rollout_cache.terminated = True
                return

            # Decode and store opponent's response
            opponent_responses = self.tokenizer.batch_decode(
                opponent_output.batch['responses'], skip_special_tokens=False
            )
            opponent_action = opponent_responses[0]

            opponent_response_ids = opponent_output.batch['responses'][0].tolist()
            if self.opponent_history:
                self.opponent_history[-1]["response_ids"] = opponent_response_ids
                self.opponent_history[-1]["messages"].append({
                    "role": "assistant",
                    "content": self.tokenizer.decode(opponent_response_ids, skip_special_tokens=True),
                })

        # Step env with opponent's first action (default to Pass if invalid format)
        opponent_action = self._normalize_opponent_action(opponent_action)
        with self.thread_lock, self.env_step_limiter:
            obs_for_agent, reward, terminated, truncated, info = self.env.step(action=opponent_action)

        obs_for_agent = self._maybe_prepend_opponent_reasoning(obs_for_agent, opponent_action)

        # Update rollout_cache so agent sees the post-opponent observation
        self.rollout_cache.history[-1]["observation"] = obs_for_agent
        if "opponent_first" in self.rollout_cache.history[-1]:
            del self.rollout_cache.history[-1]["opponent_first"]
        self.rollout_cache.history[-1]["actions_left"] = self.env_config.max_steps - self.rollout_cache.step

        if terminated:
            # Game ended on opponent's first action. Don't set terminated here —
            # let the agent still act once so formulate_rollouts has valid data.
            # env.step() will return GAME_OVER with stored reward.
            pass

    def step(self, llm_output: DataProto) -> RolloutCache:
        """Two-player step: agent 0 acts, then opponent acts, round resolves."""
        # --- Agent 0's action (already decided in make_decision) ---
        responses = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)
        agent_action = responses[0]

        with self.thread_lock, self.env_step_limiter:
            obs_for_opponent, agent_reward, agent_terminated, agent_truncated, agent_step_info = self.env.step(action=agent_action)

        # Store agent 0's response in rollout_cache (will get reward after round resolves)
        self.rollout_cache.history[-1]['llm_response'] = agent_action

        # Capture agent's action_is_valid *now*, before the (optional) opponent env.step
        # at line 215-217 would otherwise overwrite 'metrics' with opponent's.
        if agent_step_info is not None:
            merged_metrics = dict(self.rollout_cache.history[-1].get("metrics", {}))
            merged_metrics.update(agent_step_info.get("metrics", {}))
            merged_agg = dict(self.rollout_cache.history[-1].get("metrics_agg_mode", {}))
            merged_agg.update(agent_step_info.get("metrics_agg_mode", {}))
            self.rollout_cache.history[-1]["metrics"] = merged_metrics
            self.rollout_cache.history[-1]["metrics_agg_mode"] = merged_agg
            for k, v in agent_step_info.items():
                if k not in ("metrics", "metrics_agg_mode"):
                    self.rollout_cache.history[-1][k] = v

        # Game ended on agent's action (e.g., agent is P1 and action resolves the game)
        if agent_terminated:
            self.rollout_cache.step += 1
            self.rollout_cache.terminated = True
            self.rollout_cache.truncated = agent_truncated
            self.rollout_cache.history[-1]['reward'] = agent_reward
            self.rollout_cache.history.append({
                "observation": "",
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
                "messages": None,
            })
            return self.rollout_cache

        # --- Opponent's turn ---
        override = self._get_opponent_action_override()
        if override is not None:
            opponent_action = override
        else:
            opponent_lm_input = self._format_opponent_messages(obs_for_opponent)
            opponent_input_ids = opponent_lm_input.batch["input_ids"]

            max_new_tokens = min(
                self.env_config["max_tokens_per_step"],
                self.worker_config.generating_args.max_new_tokens,
                self.pipeline_config.sequence_length - opponent_input_ids.shape[1],
            )
            generation_config = self.worker_config.generating_args.to_dict()
            generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
            opponent_lm_input.meta_info["src_rank"] = self.env_config["env_id"] + 100000  # distinct src_rank
            opponent_lm_input.meta_info["lora_name"] = self.current_opponent_lora  # None=base, str=checkpoint path

            opponent_output: DataProto = self.llm_proxy.generate(
                messages=None, lm_input=opponent_lm_input, generation_config=generation_config
            )

            if opponent_output is None:
                self.rollout_cache.history[-1]['reward'] = 0.0
                self.rollout_cache.terminated = True
                return self.rollout_cache

            opponent_responses = self.tokenizer.batch_decode(opponent_output.batch['responses'], skip_special_tokens=False)
            opponent_action = opponent_responses[0]

            # Store opponent's response in opponent_history for future prompts
            opponent_response_ids = opponent_output.batch['responses'][0].tolist()
            if self.opponent_history:
                self.opponent_history[-1]["response_ids"] = opponent_response_ids
                self.opponent_history[-1]["messages"].append({
                    "role": "assistant",
                    "content": self.tokenizer.decode(opponent_response_ids, skip_special_tokens=True),
                })

        # --- Resolve round: env.step with opponent's action (default to Pass if invalid format) ---
        opponent_action = self._normalize_opponent_action(opponent_action)
        with self.thread_lock, self.env_step_limiter:
            obs_for_agent, reward, terminated, truncated, info = self.env.step(action=opponent_action)

        obs_for_agent = self._maybe_prepend_opponent_reasoning(obs_for_agent, opponent_action)

        # Store resolved reward and info for agent 0
        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True

        self.rollout_cache.history[-1]['reward'] = reward
        if info is not None:
            # Merge opponent's env.step info without clobbering agent-side metrics
            # recorded earlier. 'action_is_valid' is re-keyed as 'opponent/action_is_valid'
            # because the agent-side one (already stored) is the training signal we want.
            # Terminal outcome metrics (win_rate, success, reward-derived) always overwrite
            # the non-terminal 0.0 placeholders written by the agent's non-final env step.
            TERMINAL_METRICS = {"win_rate", "success", "bluff", "value_bet", "agent_bet",
                                 "faced_bet", "fold_vs_bet"}
            opp_metrics = info.get("metrics", {}) or {}
            opp_agg = info.get("metrics_agg_mode", {}) or {}
            merged_metrics = dict(self.rollout_cache.history[-1].get("metrics", {}))
            merged_agg = dict(self.rollout_cache.history[-1].get("metrics_agg_mode", {}))
            for k, v in opp_metrics.items():
                target = f"opponent/{k}" if k == "action_is_valid" else k
                if k == "action_is_valid" or target not in merged_metrics or k in TERMINAL_METRICS:
                    merged_metrics[target] = v
            for k, v in opp_agg.items():
                target = f"opponent/{k}" if k == "action_is_valid" else k
                if k == "action_is_valid" or target not in merged_agg or k in TERMINAL_METRICS:
                    merged_agg[target] = v
            self.rollout_cache.history[-1]["metrics"] = merged_metrics
            self.rollout_cache.history[-1]["metrics_agg_mode"] = merged_agg
            for k, v in info.items():
                if k not in ("metrics", "metrics_agg_mode"):
                    self.rollout_cache.history[-1][k] = v

        # Next observation for agent 0
        next_entry: dict = {
            "observation": obs_for_agent,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,
        }
        self.rollout_cache.history.append(next_entry)

        return self.rollout_cache

    def formulate_rollouts(self, rollout_cache):
        lm_input = super().formulate_rollouts(rollout_cache)
        if lm_input is not None:
            opponent_id = str(self.current_opponent_lora) if self.current_opponent_lora is not None else "base"
            lm_input.non_tensor_batch["opponent_id"] = np.array(
                [opponent_id] * lm_input.batch.batch_size[0], dtype=object
            )
        return lm_input

    def _format_opponent_messages(self, observation: str) -> DataProto:
        """Format prompt for the opponent using the same template as agent 0."""
        messages = []
        user_content = ""

        is_first_turn = len(self.opponent_history) == 0
        if is_first_turn:
            messages.append({"role": "system", "content": self.agent_system_template})
            user_content = f"{self.env.get_instructions()}\n"

        render_dict = {"observation": observation}
        opponent_step = len(self.opponent_history) + 1
        if contains_renderable_field(self.agent_template, "turn_idx"):
            render_dict["turn_idx"] = opponent_step
        if contains_renderable_field(self.agent_template, "actions_left"):
            render_dict["actions_left"] = self.env_config.max_steps - (self.rollout_cache.step if self.rollout_cache else 0)
        if contains_renderable_field(self.agent_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        if contains_renderable_field(self.agent_template, "suffix"):
            render_dict["suffix"] = ""
        user_content += self.agent_template.format(**render_dict)
        messages.append({"role": "user", "content": user_content})

        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True, template_name=self.pipeline_config.actor_infer.data_args.template)

        history_token_ids = []
        for items in self.opponent_history:
            if "prompt_ids" in items and "response_ids" in items:
                history_token_ids.extend(items["prompt_ids"])
                history_token_ids.extend(items["response_ids"])

        if len(history_token_ids):
            prompt_ids = compute_conversation_end_token_id(self.tokenizer) + prompt_ids

        input_ids = history_token_ids + prompt_ids

        # Store for opponent history
        self.opponent_history.append({
            "prompt_ids": prompt_ids,
            "response_ids": [],  # filled after opponent generates
            "messages": messages,
        })

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1) - 1

        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        return lm_input
