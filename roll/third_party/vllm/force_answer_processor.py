"""Per-request vLLM logits processor that forces `<answer>` immediately after `</think>`.

Pairs with `SamplingParams.thinking_token_budget`: once the tokenizer's
`</think>` sequence has been emitted (either naturally or because the budget
forced it), this processor masks the next len(<answer>) sampling steps to
the `<answer>` token sequence, then releases control. Generation then
continues normally — usually `Pass` or `Bet` followed by `</answer>`.

Activated per-request when both:
  * vllm_config.reasoning_config is set, AND
  * the request's SamplingParams.thinking_token_budget is not None.

Wire via AsyncEngineArgs(logits_processors=[FQCN]) — see
`actor_infer.strategy_args.strategy_config.logits_processors` in agentic yamls.
"""

from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


_NEG_INF = float("-inf")


class ForceAnswerAfterThinkProcessor(AdapterLogitsProcessor):
    """Forces `<answer>` after `</think>` per request."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        reasoning_config = getattr(vllm_config, "reasoning_config", None)
        self.enabled = reasoning_config is not None
        if not self.enabled:
            self.close_think_ids: list[int] = []
            self.answer_ids: list[int] = []
            return

        tokenizer = cached_tokenizer_from_config(model_config=vllm_config.model_config)
        end_ids = getattr(reasoning_config, "reasoning_end_token_ids", None)
        if end_ids:
            self.close_think_ids = list(end_ids)
        else:
            self.close_think_ids = list(
                tokenizer.encode("</think>", add_special_tokens=False)
            )
        self.answer_ids = list(tokenizer.encode("<answer>", add_special_tokens=False))

        if not self.close_think_ids or not self.answer_ids:
            self.enabled = False

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams) -> Optional[RequestLogitsProcessor]:
        if not self.enabled or params.thinking_token_budget is None:
            return None

        close_ids = self.close_think_ids
        answer_ids = self.answer_ids
        n_close = len(close_ids)
        n_answer = len(answer_ids)
        # Captured in the closure so per-request state lives across calls.
        state = {"matched_at": None}

        def per_request(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            if state["matched_at"] is None:
                if len(output_ids) >= n_close and list(output_ids[-n_close:]) == close_ids:
                    state["matched_at"] = len(output_ids)
            if state["matched_at"] is not None:
                forced_idx = len(output_ids) - state["matched_at"]
                if 0 <= forced_idx < n_answer:
                    next_token = answer_ids[forced_idx]
                    logits.fill_(_NEG_INF)
                    logits[next_token] = 0.0
            return logits

        return per_request
