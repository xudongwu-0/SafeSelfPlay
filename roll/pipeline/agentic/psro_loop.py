"""PSROLoop: incremental PayoffMatrix expansion + Nash computation for PSRO.

Instantiated once in AgenticPipeline when psro_mode=True. Call on_policy_added()
after each FSP checkpoint is saved to expand the payoff matrix and return Nash
mixed-strategy weights for opponent sampling in the next generation.
"""

import os
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.empirical_game import PayoffMatrix
from roll.pipeline.agentic.meta_solver import compute_nash
from roll.utils.logging import get_logger

logger = get_logger()


class PSROLoop:
    """Manages PSRO iterations on top of the AgenticPipeline FSP mechanism.

    Call on_policy_added() after each FSP checkpoint is saved. Returns the Nash
    mixed strategy (p2_strategy) for opponent pool sampling in the next generation.
    The generate_scheduler must be resumed by the caller before calling
    on_policy_added(), as PayoffMatrix runs arena episodes through it.
    """

    def __init__(
        self,
        generate_scheduler,
        pipeline_config: AgenticConfig,
        tokenizer: PreTrainedTokenizer,
        env_tag: str,
    ) -> None:
        self._iteration: int = 0
        self.payoff_matrix = PayoffMatrix(
            generate_scheduler=generate_scheduler,
            pipeline_config=pipeline_config,
            tokenizer=tokenizer,
            env_tag=env_tag,
            episodes_per_pair=pipeline_config.psro_episodes_per_pair,
            max_concurrent=pipeline_config.psro_max_concurrent_eval,
            seed_base=pipeline_config.psro_seed_base,
        )
        logger.info(
            f"PSROLoop initialized: env_tag={env_tag}, "
            f"episodes_per_pair={pipeline_config.psro_episodes_per_pair}, "
            f"max_concurrent={pipeline_config.psro_max_concurrent_eval}"
        )

    def on_policy_added(
        self,
        new_policy: Optional[str],
        output_dir: str,
    ) -> Optional[np.ndarray]:
        """Expand the payoff matrix with new_policy and return Nash opponent weights.

        Must be called while the generate_scheduler is resumed (caller's responsibility).
        Ordering relative to fsp_checkpoints.append() in AgenticPipeline does not matter:
        PayoffMatrix maintains its own policies list independently.

        Args:
            new_policy: LoRA checkpoint path (or None for base model).
            output_dir: Root output dir; matrix saved under output_dir/psro/.

        Returns:
            p2_strategy — Nash probs over the updated opponent pool — or None if only
            one policy exists (trivial, uniform sampling implied).
        """
        existing_policies: List[Optional[str]] = list(self.payoff_matrix.policies)
        logger.info(
            f"PSROLoop iter {self._iteration}: expanding {len(existing_policies)}x{len(existing_policies)} "
            f"matrix, new_policy={new_policy!r}"
        )

        try:
            updated_matrix = self.payoff_matrix.expand_matrix(new_policy, existing_policies)
        except ValueError as exc:
            logger.error(f"PSROLoop: expand_matrix failed: {exc}")
            return None

        self._iteration += 1
        self.payoff_matrix.log()

        psro_dir = os.path.join(output_dir, "psro")
        self.payoff_matrix.save(psro_dir, f"payoff_matrix_iter_{self._iteration}.json")
        self.payoff_matrix.save(psro_dir, "payoff_matrix_latest.json")

        if len(self.payoff_matrix.policies) <= 1:
            logger.info("PSROLoop: single policy, skipping Nash computation.")
            return None

        _, p2_strategy = compute_nash(updated_matrix)
        logger.info(
            "PSROLoop iter %d: Nash p2_strategy = [%s]",
            self._iteration,
            ", ".join(f"{p:.4f}" for p in p2_strategy),
        )
        return p2_strategy
