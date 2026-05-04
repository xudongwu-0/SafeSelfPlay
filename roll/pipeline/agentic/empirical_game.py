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
        if self._pipeline_config.psro_bubble_eval_episodes > 0:
            self._online_sum: np.ndarray = np.zeros((1, 1))
            self._online_count: np.ndarray = np.zeros((1, 1), dtype=int)
            self._bubble_episode_counter: int = 0

        logger.info(f"PayoffMatrix: creating {max_concurrent} env managers...")
        self._env_managers = [
            _create_arena_env_manager(pipeline_config, env_tag, tokenizer, generate_scheduler, env_id=idx)
            for idx in range(max_concurrent)
        ]
        if self._pipeline_config.psro_bubble_eval_episodes > 0:
            logger.info(f"PayoffMatrix: creating {max_concurrent} dedicated bubble eval env managers...")
            self._bubble_env_managers = [
                _create_arena_env_manager(
                    pipeline_config, env_tag, tokenizer, generate_scheduler,
                    env_id=max_concurrent + idx,
                    env_config_overrides={"debug_mode": True},
                )
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

        # Build tasks: (i_idx, j_idx, ep).
        # Both (n,j) and (j,n) share seeds keyed by existing policy index j,
        # so they play identical game states and positional/card variance cancels.
        tasks: List[Tuple[int, int, int]] = []
        for j in range(n):
            # new (row n) vs existing[j]
            for ep in range(self._episodes_per_pair):
                tasks.append((n, j, ep))
            # existing[j] vs new (col n)
            for ep in range(self._episodes_per_pair):
                tasks.append((j, n, ep))

        results: Dict[Tuple[int, int], List[float]] = {}
        episode_log: List[Tuple[int, int, int, float]] = []  # (i, j, seed, payoff)
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
                    i_idx, j_idx, ep = pending.pop(0)
                    em_idx = em_available.pop(0)
                    # Shared seed for (n,j) and (j,n): keyed by existing policy index
                    existing_idx = j_idx if i_idx == n else i_idx
                    seed = seed_k + existing_idx * self._episodes_per_pair + ep
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
                    future_to_info[future] = (i_idx, j_idx, em_idx, seed)

            _submit_batch()
            completed = 0
            total = len(tasks)

            while future_to_info:
                for future in as_completed(future_to_info):
                    i_idx, j_idx, em_idx, seed = future_to_info.pop(future)
                    try:
                        payoff: float = future.result()
                    except Exception as exc:
                        logger.error(f"PayoffMatrix: episode ({i_idx},{j_idx}) failed: {exc}")
                        payoff = 0.0
                    results[(i_idx, j_idx)].append(payoff)
                    episode_log.append((i_idx, j_idx, seed, payoff))
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
        if self._pipeline_config.psro_bubble_eval_episodes > 0:
            new_sum = np.zeros((n + 1, n + 1))
            new_sum[:n, :n] = self._online_sum
            self._online_sum = new_sum
            new_count = np.zeros((n + 1, n + 1), dtype=int)
            new_count[:n, :n] = self._online_count
            self._online_count = new_count

        logger.info(f"PayoffMatrix: expansion complete, matrix is now {n+1}×{n+1}.")
        self._log_per_state_breakdown(episode_log, n, new_policy)
        return self._matrix.copy()

    def _log_per_state_breakdown(
        self,
        episode_log: List[Tuple[int, int, int, float]],
        n: int,
        new_policy: Optional[str],
    ) -> None:
        """Log per-state payoff breakdown when debug_mode is enabled on the env."""
        env = self._env_managers[0].env
        if not getattr(env, "debug_mode", False):
            return
        num_states = getattr(env, "NUM_START_STATES", None)
        state_names = getattr(env, "_ALL_STATES", None)
        if num_states is None or state_names is None:
            return

        card_name = {0: "J", 1: "Q", 2: "K"}
        per_state: Dict[Tuple[int, int, int], List[float]] = {}
        for ii, jj, seed, pv in episode_log:
            key = (ii, jj, seed % num_states)
            per_state.setdefault(key, []).append(pv)

        new_label = _get_model_label(new_policy)
        for j in range(n):
            j_label = _get_model_label(self.policies[j])
            for agent_idx, opp_idx in [(n, j), (j, n)]:
                agent_label = new_label if agent_idx == n else j_label
                opp_label = j_label if agent_idx == n else new_label
                rows = []
                for si in range(num_states):
                    p0c, p1c, is_p0 = state_names[si]
                    vals = per_state.get((agent_idx, opp_idx, si), [])
                    mean_str = f"{np.mean(vals):.3f}" if vals else "N/A"
                    rows.append(
                        f"  state{si:2d} p0={card_name[p0c]} p1={card_name[p1c]}"
                        f" agentIsP0={is_p0}: {mean_str} {vals}"
                    )
                logger.info(
                    f"PerState [{agent_label} vs {opp_label}]:\n" + "\n".join(rows)
                )

    def get_online_matrix(self) -> np.ndarray:
        """Return payoff matrix with online estimates overlaid where available."""
        result = self._matrix.copy()
        mask = self._online_count > 0
        result[mask] = self._online_sum[mask] / self._online_count[mask]
        return result

    def run_bubble_eval(
        self,
        stop_event: threading.Event,
        episodes_per_step: int,
    ) -> Optional[Dict[Tuple[int, int], List[float]]]:
        """Run background eval episodes during training idle time.

        Args:
            stop_event: Set by the main thread when training finishes.
            episodes_per_step: Total episodes; groups of 12 assigned to least-visited pairs.

        Returns:
            Dict of results keyed by (i,j) pair, or None if interrupted early.
        """
        n = len(self.policies)
        if n < 2:
            return None

        n_groups = max(1, episodes_per_step // 12)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

        # Greedy selection: pick least-visited pair for each group.
        local_counts = self._online_count.copy()
        group_targets: List[Tuple[int, int]] = []
        for _ in range(n_groups):
            best = min(pairs, key=lambda p: (local_counts[p[0], p[1]], p[0], p[1]))
            group_targets.append(best)
            local_counts[best[0], best[1]] += 12

        # Build all tasks; seed uniquely per (group, ep).
        counter_base = self._bubble_episode_counter
        self._bubble_episode_counter += n_groups
        tasks: List[Tuple[int, int, int, int]] = []  # (i, j, ep, group_idx)
        for g, (i, j) in enumerate(group_targets):
            for ep in range(12):
                seed = self._seed_base + 2_000_000_000 + (counter_base + g) * 13 + ep
                tasks.append((i, j, ep, seed))

        results: Dict[Tuple[int, int], List[float]] = {}
        for (i, j) in set(group_targets):
            results[(i, j)] = []

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
            em_available = list(range(self._max_concurrent))
            pending = list(tasks)
            future_to_info: Dict = {}

            def _submit_batch() -> None:
                while pending and em_available:
                    i_idx, j_idx, _ep, seed = pending.pop(0)
                    em_idx = em_available.pop(0)
                    src_rank = ARENA_SRC_RANK_BASE + (self._max_concurrent + em_idx) * 2
                    future = executor.submit(
                        play_episode,
                        self._bubble_env_managers[em_idx],
                        self.policies[i_idx],
                        self.policies[j_idx],
                        self._generate_scheduler,
                        self._tokenizer,
                        self._pipeline_config,
                        self._worker_config,
                        seed,
                        src_rank,
                    )
                    future_to_info[future] = (i_idx, j_idx, em_idx)

            _submit_batch()
            n_completed = 0
            total = len(tasks)

            while future_to_info:
                for future in as_completed(future_to_info):
                    i_idx, j_idx, em_idx = future_to_info.pop(future)
                    try:
                        payoff: float = future.result()
                    except Exception as exc:
                        logger.debug(f"PayoffMatrix bubble_eval ({i_idx},{j_idx}) aborted/failed: {exc}")
                        payoff = None
                    if payoff is not None:
                        results[(i_idx, j_idx)].append(payoff)
                    em_available.append(em_idx)
                    n_completed += 1
                    _submit_batch()
                    if n_completed >= 36 and stop_event.is_set():
                        logger.info("PayoffMatrix bubble_eval: stop_event set, interrupting after %d/%d.", n_completed, total)
                        return None
                    break  # re-enter as_completed with updated future_to_info

        logger.info("PayoffMatrix bubble_eval: all %d episodes complete.", total)
        return results

    def commit_bubble_eval(self, results: Dict[Tuple[int, int], List[float]]) -> None:
        """Accumulate bubble eval results into online estimator."""
        for (i, j), payoffs in results.items():
            self._online_sum[i, j] += sum(payoffs)
            self._online_count[i, j] += len(payoffs)

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
