import copy
from typing import Optional

import ray
import torch
from tensordict import TensorDict
from threading import Lock

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, compute_conversation_end_token_id
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig, LLMProxyConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.logging import get_logger


class TwoPlayerTrajEnvManager(TrajEnvManager):
    """Env manager for two-player zero-sum games.

    Extends TrajEnvManager to query two LLM servers per round:
    - Agent 0 (training agent): uses self.llm_proxy (same as parent)
    - Agent 1 (opponent): uses self.opponent_llm_proxy (frozen model)

    Per round:
    1. Agent 0 sees observation, makes decision (parent's make_decision)
    2. env.step(agent_0_action) → observation for agent 1
    3. Agent 1 sees its observation, makes decision via opponent proxy
    4. env.step(agent_1_action) → resolves round, returns obs + reward for agent 0

    Only agent 0's trajectory data goes to training.
    """

    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue,
                 thread_lock: Lock,
                 mode: str = 'train',
                 opponent_generate_scheduler=None,
                 *args, **kwargs):
        super().__init__(
            worker_config=worker_config,
            pipeline_config=pipeline_config,
            env_config=env_config,
            tokenizer=tokenizer,
            generate_scheduler=generate_scheduler,
            output_queue=output_queue,
            thread_lock=thread_lock,
            mode=mode,
            *args, **kwargs,
        )
        self.logger = get_logger()

        if opponent_generate_scheduler is None:
            raise ValueError("TwoPlayerTrajEnvManager requires opponent_generate_scheduler")

        self.opponent_llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=opponent_generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env,
        )

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

        # Store agent 0's response in rollout_cache (will get reward later)
        self.rollout_cache.history[-1]['llm_response'] = agent_action
        # Don't store agent_step_info metrics yet — round not resolved

        # --- Opponent's turn ---
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

        opponent_output: DataProto = ray.get(
            self.opponent_llm_proxy.generate_scheduler.generate_one_request.remote(data=opponent_lm_input)
        )

        if opponent_output is None:
            # Opponent failed — treat as invalid action
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

        # Now store the resolved reward and info for agent 0
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

        # Use the same agent_template
        render_dict = {"observation": observation}
        opponent_step = len(self.opponent_history) + 1
        from roll.utils.str_utils import contains_renderable_field
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
