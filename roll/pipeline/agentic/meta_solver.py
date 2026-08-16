"""Meta-strategy solver: Nash Equilibrium for 2-player zero-sum meta-games.

Uses Multiplicative Weights Update (Hedge) — numpy-only, no scipy required.
Time-averaged strategies converge to Nash for 2-player zero-sum games.

Input: (m, n) payoff matrix where A[i,j] is the payoff to the row player.
Output: Nash mixed strategies for each player.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Hedge regret bound is O(sqrt(T log n)); 50k iters is enough for ~1e-3 accuracy on
# the < 20-strategy meta-games we hit. eta is the canonical sqrt(log n / T) step.
_HEDGE_NUM_ITERATIONS: int = 50_000
_HEDGE_ETA_SCALE: float = 2.0


def compute_nash(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Nash Equilibrium mixed strategies for a 2-player zero-sum game.

    Uses Multiplicative Weights Update (Hedge). For the small policy pools
    typical in PSRO/FSP (< 20 policies), convergence is fast and accurate.

    Args:
        payoff_matrix: 2-D array of shape (m, n). ``A[i, j]`` is the payoff to
            the row player when P1 plays strategy ``i`` and P2 plays strategy ``j``.

    Returns:
        ``(p1_strategy, p2_strategy)`` — each a 1-D float64 ndarray summing to 1.

    Raises:
        ValueError: if ``payoff_matrix`` is not 2-D or has zero rows/columns.
    """
    A = np.atleast_2d(np.asarray(payoff_matrix, dtype=np.float64))
    if A.ndim != 2:
        raise ValueError(f"payoff_matrix must be 2-D, got shape {A.shape}")
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError(f"payoff_matrix must have at least one row and column, got ({m}, {n})")

    if m == 1 and n == 1:
        return np.array([1.0]), np.array([1.0])
    if m == 1:
        # The row player has no choice, so every minimizing column is a best
        # response. Mix uniformly only across tied minimizers.
        best_columns = A[0] == np.min(A[0])
        p2_strategy = best_columns.astype(np.float64)
        p2_strategy /= p2_strategy.sum()
        return np.array([1.0]), p2_strategy
    if n == 1:
        # The column player has no choice, so every maximizing row is a best
        # response. Mix uniformly only across tied maximizers.
        best_rows = A[:, 0] == np.max(A[:, 0])
        p1_strategy = best_rows.astype(np.float64)
        p1_strategy /= p1_strategy.sum()
        return p1_strategy, np.array([1.0])

    eta = _HEDGE_ETA_SCALE * np.sqrt(np.log(max(m, n) + 1) / _HEDGE_NUM_ITERATIONS)

    p = np.ones(m) / m
    q = np.ones(n) / n
    avg_p = np.zeros(m)
    avg_q = np.zeros(n)

    for _ in range(_HEDGE_NUM_ITERATIONS):
        avg_p += p
        avg_q += q
        # P1 (maximiser): payoff for each row = A @ q
        p = p * np.exp(eta * (A @ q))
        p /= p.sum()
        # P2 (minimiser): payoff for each col = -(A.T @ p)
        q = q * np.exp(eta * (-(A.T @ p)))
        q /= q.sum()

    return avg_p / _HEDGE_NUM_ITERATIONS, avg_q / _HEDGE_NUM_ITERATIONS
