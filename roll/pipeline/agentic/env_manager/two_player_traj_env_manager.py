import copy
from typing import Optional

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
    """Env manager for two-player zero-sum games.

    Extends TrajEnvManager to query the same vLLM server twice per round:
    - Agent 0 (training agent): uses LoRA adapter (default lora_name)
    - Agent 1 (opponent): uses base model (lora_name=None)

    Per round:
    1. Agent 0 sees observation, makes decision (parent's make_decision)
    2. env.step(agent_0_action) → observation for agent 1
    3. Agent 1 sees its observation, makes decision via base model
    4. env.step(agent_1_action) → resolves round, returns obs + reward for agent 0

    Only agent 0's trajectory data goes to training.
    """

    def __init__(self, *args, **kwargs):
        # Remove opponent_generate_scheduler if passed (backward compat)
        kwargs.pop("opponent_generate_scheduler", None)
        super().__init__(*args, **kwargs)
        self.logger = get_logger()
        # Track opponent's conversation history for prompt formatting
        self.opponent_history: list[dict] = []

    def reset(self) -> Optional[RolloutCache]:
        result = super().reset()
        self.opponent_history = []
        return result

    def step(self, llm_output: DataProto) -> RolloutCache:
        """Two-player step: agent 0 acts, then opponent acts, round resolves."""
        # --- Agent 0's action (already decided in make_decision) ---
        responses = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)
        agent_action = responses[0]

        with self.thread_lock, self.env_step_limiter:
            obs_for_opponent, _, _, _, agent_step_info = self.env.step(action=agent_action)

        # Store agent 0's response in rollout_cache (will get reward after round resolves)
        self.rollout_cache.history[-1]['llm_response'] = agent_action

        # --- Opponent's turn (base model, lora_name=None) ---
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
        opponent_lm_input.meta_info["generation_config"] = generation_config
        opponent_lm_input.meta_info["pad_to_seq_len"] = False
        opponent_lm_input.meta_info["lora_name"] = None  # base model, no LoRA

        opponent_output: DataProto = ray.get(
            self.generate_scheduler.generate_one_request.remote(data=opponent_lm_input)
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

        # --- Resolve round: env.step with opponent's action ---
        with self.thread_lock, self.env_step_limiter:
            obs_for_agent, reward, terminated, truncated, info = self.env.step(action=opponent_action)

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
            self.rollout_cache.history[-1].update(info)

        # Next observation for agent 0
        self.rollout_cache.history.append({
            "observation": obs_for_agent,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,
        })

        return self.rollout_cache

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

        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)

        # Build history token_ids from opponent's previous turns
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
