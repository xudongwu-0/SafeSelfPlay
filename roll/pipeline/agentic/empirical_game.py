"""Empirical game (PSRO): incremental payoff matrix expansion for two-player LLM games."""

import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.arena_eval import (
    ARENA_SRC_RANK_BASE,
    _create_arena_env_manager,
    _get_model_label,
    log_payoff_matrix,
    play_episode,
    save_payoff_matrix,
)
from roll.utils.logging import get_logger

logger = get_logger()

_SEED_STRIDE = 1_000_000


class PayoffMatrix:
    """Incremental empirical payoff matrix for PSRO.

    Grows by one row and column each time a new policy is added via expand_matrix(),
    running only the matches involving the new policy rather than recomputing the full N×N.

    M[i][j] = average payoff (expected chips won) of policy i against policy j.
    Diagonal entries are always 0.0 (self-play EV = 0 in zero-sum games).
    """

    def __init__(
        self,
        generate_scheduler,
        pipeline_config: AgenticConfig,
        tokenizer: PreTrainedTokenizer,
        env_tag: str,
        episodes_per_pair: int = 4,
        max_concurrent: int = 32,
        seed_base: int = 12345,
    ) -> None:
        self._matrix: np.ndarray = np.empty((0, 0), dtype=float)
        self.policies: List[Optional[str]] = []
        self._iteration: int = 0

        self._generate_scheduler = generate_scheduler
        self._pipeline_config = pipeline_config
        self._tokenizer = tokenizer
        self._env_tag = env_tag
        self._episodes_per_pair = episodes_per_pair
        self._max_concurrent = max_concurrent
        self._seed_base = seed_base
        self._worker_config = pipeline_config.train_env_manager

        logger.info(f"PayoffMatrix: creating {max_concurrent} env managers...")
        self._env_managers = [
            _create_arena_env_manager(pipeline_config, env_tag, tokenizer, generate_scheduler, env_id=idx)
            for idx in range(max_concurrent)
        ]

    def expand_matrix(
        self,
        new_policy: Optional[str],
        existing_policies: List[Optional[str]],
    ) -> np.ndarray:
        """Add new_policy to the matrix, running K matches against all existing policies.

        Args:
            new_policy: LoRA checkpoint path (or None for base model) to add.
            existing_policies: Ordered list of policies already in the matrix. Must
                match self.policies; if it differs a warning is emitted and self.policies
                is used as truth.

        Returns:
            Copy of the updated (n+1)×(n+1) payoff matrix.

        Raises:
            ValueError: If new_policy is already present in self.policies.
        """
        if new_policy in self.policies:
            raise ValueError(f"Policy already in matrix: {new_policy!r}")

        if existing_policies != self.policies:
            logger.warning(
                "expand_matrix: existing_policies differs from self.policies; using self.policies as truth."
            )

        n = len(self.policies)

        # First policy: no matches needed, just initialise the 1×1 matrix.
        if n == 0:
            self._matrix = np.array([[0.0]], dtype=float)
            self.policies.append(new_policy)
            self._iteration += 1
            logger.info(f"PayoffMatrix: added first policy {_get_model_label(new_policy)}, matrix is 1×1.")
            return self._matrix.copy()

        seed_k = self._seed_base + self._iteration * _SEED_STRIDE

        # Build tasks: (i_idx, j_idx, ep, matchup_pos).
        # matchup_pos drives seed derivation and must be stable within the call.
        tasks: List[Tuple[int, int, int, int]] = []
        matchup_pos = 0
        for j in range(n):
            # new (row n) vs existing[j]
            for ep in range(self._episodes_per_pair):
                tasks.append((n, j, ep, matchup_pos))
            # existing[j] vs new (col n)
            matchup_pos += 1
            for ep in range(self._episodes_per_pair):
                tasks.append((j, n, ep, matchup_pos))
            matchup_pos += 1

        results: Dict[Tuple[int, int], List[float]] = {}
        for j in range(n):
            results[(n, j)] = []
            results[(j, n)] = []

        logger.info(
            f"PayoffMatrix: expanding {n}×{n} → {n+1}×{n+1}, "
            f"new policy={_get_model_label(new_policy)}, "
            f"{len(tasks)} episodes total."
        )

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
            em_available = list(range(self._max_concurrent))
            pending = list(tasks)
            future_to_info: Dict = {}

            def _submit_batch() -> None:
                while pending and em_available:
                    i_idx, j_idx, ep, mpos = pending.pop(0)
                    em_idx = em_available.pop(0)
                    seed = seed_k + mpos * self._episodes_per_pair + ep
                    src_rank = ARENA_SRC_RANK_BASE + em_idx * 2

                    lora_i = new_policy if i_idx == n else self.policies[i_idx]
                    lora_j = new_policy if j_idx == n else self.policies[j_idx]

                    future = executor.submit(
                        play_episode,
                        self._env_managers[em_idx],
                        lora_i,
                        lora_j,
                        self._generate_scheduler,
                        self._tokenizer,
                        self._pipeline_config,
                        self._worker_config,
                        seed,
                        src_rank,
                    )
                    future_to_info[future] = (i_idx, j_idx, ep, em_idx)

            _submit_batch()
            completed = 0
            total = len(tasks)

            while future_to_info:
                for future in as_completed(future_to_info):
                    i_idx, j_idx, ep, em_idx = future_to_info.pop(future)
                    try:
                        payoff: float = future.result()
                    except Exception as exc:
                        logger.error(f"PayoffMatrix: episode ({i_idx},{j_idx}) ep={ep} failed: {exc}")
                        payoff = 0.0
                    results[(i_idx, j_idx)].append(payoff)
                    em_available.append(em_idx)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"PayoffMatrix: {completed}/{total} episodes done.")
                    _submit_batch()
                    break  # re-enter as_completed with updated future_to_info

        # Expand matrix: copy existing block, fill new row/col, diagonal = 0.0.
        new_matrix = np.full((n + 1, n + 1), 0.0, dtype=float)
        new_matrix[:n, :n] = self._matrix
        for (i_idx, j_idx), payoffs in results.items():
            new_matrix[i_idx][j_idx] = np.mean(payoffs) if payoffs else 0.0

        self._matrix = new_matrix
        self.policies.append(new_policy)
        self._iteration += 1

        logger.info(f"PayoffMatrix: expansion complete, matrix is now {n+1}×{n+1}.")
        return self._matrix.copy()

    def log(self) -> None:
        """Pretty-print the current payoff matrix."""
        log_payoff_matrix(self._matrix, self.policies)

    def save(self, output_dir: str, filename: str = "empirical_game_matrix.json") -> str:
        """Serialize the matrix to JSON. Returns the filepath."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        data = {
            "labels": [_get_model_label(p) for p in self.policies],
            "lora_paths": [p if p is not None else "base_model" for p in self.policies],
            "payoff_matrix": self._matrix.tolist(),
            "iteration": self._iteration,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"PayoffMatrix saved to {filepath}")
        return filepath

    @classmethod
    def load(
        cls,
        path: str,
        generate_scheduler,
        pipeline_config: AgenticConfig,
        tokenizer: PreTrainedTokenizer,
        env_tag: str,
        episodes_per_pair: int = 4,
        max_concurrent: int = 32,
        seed_base: int = 12345,
    ) -> "PayoffMatrix":
        """Restore a PayoffMatrix from a JSON file produced by save().

        The evaluation context (generate_scheduler, pipeline_config, etc.) must be
        provided because it cannot be serialised.
        """
        with open(path) as f:
            data = json.load(f)

        pm = cls(
            generate_scheduler=generate_scheduler,
            pipeline_config=pipeline_config,
            tokenizer=tokenizer,
            env_tag=env_tag,
            episodes_per_pair=episodes_per_pair,
            max_concurrent=max_concurrent,
            seed_base=seed_base,
        )
        pm._matrix = np.array(data["payoff_matrix"], dtype=float)
        pm.policies = [None if p == "base_model" else p for p in data["lora_paths"]]
        pm._iteration = data.get("iteration", len(pm.policies))
        logger.info(f"PayoffMatrix loaded from {path} (iteration={pm._iteration}, n={len(pm.policies)}).")
        return pm
