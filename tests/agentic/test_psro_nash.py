"""Tests for PSRO Nash computation correctness.

Covers:
1. compute_nash correctness on canonical zero-sum games
2. Nash best-response conditions
3. PayoffMatrix antisymmetry bug: M[i][j] records payoff of i *as first player*,
   so M[i][j] + M[j][i] != 0 for positional games like Kuhn Poker.
   The meta-game matrix must be antisymmetrized before Nash computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from roll.pipeline.agentic.meta_solver import compute_nash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_response_gap(A: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Max unilateral improvement either player can achieve (should be ~0 at Nash)."""
    val = float(p1 @ A @ p2)
    p1_gap = max(float((A @ p2)[i]) - val for i in range(len(p1)))
    p2_gap = max(val - float((p1 @ A)[j]) for j in range(len(p2)))
    return max(p1_gap, p2_gap)


def is_nash(A: np.ndarray, p1: np.ndarray, p2: np.ndarray, tol: float = 5e-3) -> bool:
    return best_response_gap(A, p1, p2) <= tol


def antisymmetrize(M: np.ndarray) -> np.ndarray:
    """Return 0.5*(M - M.T): the correct zero-sum meta-game payoff for role-averaged play."""
    return 0.5 * (M - M.T)


# ---------------------------------------------------------------------------
# Section 1: compute_nash correctness on canonical games
# ---------------------------------------------------------------------------

class TestComputeNashCanonical:
    """compute_nash must converge to the known Nash on standard zero-sum games."""

    def test_1x1(self):
        p1, p2 = compute_nash(np.array([[5.0]]))
        assert np.allclose(p1, [1.0]) and np.allclose(p2, [1.0])

    def test_matching_pennies(self):
        """Nash = [0.5, 0.5] for both players."""
        A = np.array([[1., -1.], [-1., 1.]])
        p1, p2 = compute_nash(A)
        assert np.allclose(p1, [0.5, 0.5], atol=1e-2), f"p1={p1}"
        assert np.allclose(p2, [0.5, 0.5], atol=1e-2), f"p2={p2}"

    def test_rock_paper_scissors(self):
        """Nash = uniform for both players."""
        A = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
        p1, p2 = compute_nash(A)
        assert np.allclose(p1, [1/3]*3, atol=1e-2), f"p1={p1}"
        assert np.allclose(p2, [1/3]*3, atol=1e-2), f"p2={p2}"

    def test_dominant_strategy(self):
        """P1 has a strictly dominant strategy; Nash must be the pure strategy."""
        # Row 0 dominates row 1: [3,3] > [1,1] entrywise.
        A = np.array([[3., 3.], [1., 1.]])
        p1, _ = compute_nash(A)
        assert p1[0] > 0.99, f"Expected pure strategy 0, got p1={p1}"

    def test_known_value_bimatrix(self):
        """3x2 game: Nash satisfies best-response conditions."""
        A = np.array([[1., -5.], [0., 0.], [-3., 2.]])
        p1, p2 = compute_nash(A)
        assert is_nash(A, p1, p2), f"BR gap={best_response_gap(A, p1, p2):.4f}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_antisymmetric_matrix(self, seed: int):
        """For any antisymmetric (zero-sum) game, Nash must satisfy best-response."""
        rng = np.random.default_rng(seed)
        n = 4
        raw = rng.uniform(-1, 1, (n, n))
        A = 0.5 * (raw - raw.T)  # antisymmetric -> zero-sum
        p1, p2 = compute_nash(A)
        gap = best_response_gap(A, p1, p2)
        assert gap < 0.02, f"BR gap={gap:.4f} for seed={seed}"


# ---------------------------------------------------------------------------
# Section 2: Nash best-response conditions on known game instances
# ---------------------------------------------------------------------------

class TestNashBestResponseConditions:
    """The returned Nash must satisfy best-response for both players."""

    def test_matching_pennies_br(self):
        A = np.array([[1., -1.], [-1., 1.]])
        p1, p2 = compute_nash(A)
        assert is_nash(A, p1, p2), f"BR gap={best_response_gap(A, p1, p2):.4f}"

    def test_rps_br(self):
        A = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
        p1, p2 = compute_nash(A)
        assert is_nash(A, p1, p2), f"BR gap={best_response_gap(A, p1, p2):.4f}"

    def test_strategies_are_valid_distributions(self):
        A = np.array([[1., -1.], [-1., 1.]])
        p1, p2 = compute_nash(A)
        assert abs(p1.sum() - 1.0) < 1e-6 and (p1 >= -1e-9).all()
        assert abs(p2.sum() - 1.0) < 1e-6 and (p2 >= -1e-9).all()


# ---------------------------------------------------------------------------
# Section 3: PayoffMatrix antisymmetry bug
# ---------------------------------------------------------------------------

class TestPayoffMatrixStructure:
    """
    PayoffMatrix stores M[i][j] = payoff of policy-i when it plays as
    *first player* (P1 in Kuhn Poker) versus policy-j as second player (P2).

    M[j][i] = payoff of policy-j when *it* plays as first player vs policy-i as P2.

    These are two DIFFERENT role assignments, so M is NOT antisymmetric:
        M[i][j] + M[j][i]  !=  0  in general.

    compute_nash(M) therefore computes Nash for the wrong (non-zero-sum) game.

    The fix: antisymmetrize before calling compute_nash.
        M_meta[i][j] = 0.5*(M[i][j] - M[j][i])
    This represents policy-i's advantage over policy-j *averaged over both roles*.
    """

    # --- 3a. The matrix is not antisymmetric when P1 has positional bias ----------

    def test_matrix_not_antisymmetric_with_positional_bias(self):
        """
        In Kuhn Poker, Nash-vs-Nash EV is -1/18 for P1 (structural disadvantage).
        If both policies play Nash-like strategies:
          M[0][1] = -1/18  (policy-0 as P1, disadvantaged by position)
          M[1][0] = -1/18  (policy-1 as P1, same disadvantage)
        => M[0][1] + M[1][0] = -2/18 != 0. Matrix is NOT antisymmetric.
        """
        p1_bias = -1.0 / 18.0
        M = np.array([[0.0, p1_bias], [p1_bias, 0.0]])
        assert not np.allclose(M + M.T, 0.0), (
            f"Matrix should NOT be antisymmetric: M[0][1]+M[1][0]={M[0][1]+M[1][0]:.4f}"
        )

    # --- 3b. compute_nash on the raw non-antisymmetric matrix gives wrong value ---

    def test_raw_matrix_nash_value_nonzero_for_equal_skill(self):
        """
        For an equal-skill game with positional bias, the true meta-game value is 0
        (neither policy has a skill advantage). But compute_nash on the raw matrix
        returns a nonzero game value (~p1_bias/2) because the raw matrix is not
        a valid zero-sum meta-game — it encodes positional bias as a spurious shift.

        M_raw = [[0, p1_bias], [p1_bias, 0]]  (symmetric, equal-skill)
        Nash: uniform [0.5, 0.5] (by symmetry)
        Nash value = p1_bias/2 ~= -0.028  (NOT zero — wrong)

        Correct meta-game M_meta = antisymm(M_raw) = [[0,0],[0,0]] -> value = 0.
        """
        p1_bias = -1.0 / 18.0
        M_raw = np.array([[0.0, p1_bias], [p1_bias, 0.0]])

        p1, p2 = compute_nash(M_raw)
        nash_val_raw = float(p1 @ M_raw @ p2)

        # Antisymmetrized matrix for equal-skill game is identically zero.
        M_correct = antisymmetrize(M_raw)
        assert np.allclose(M_correct, 0.0), f"M_correct should be zero: {M_correct}"

        # Raw Nash value is ~p1_bias/2, NOT zero. This is the bug:
        # the raw matrix produces a nonzero game value for equal-skill policies.
        expected_raw_val = p1_bias / 2.0  # ~= -0.0278 (uniform Nash on symmetric matrix)
        assert abs(nash_val_raw - expected_raw_val) < 0.01, (
            f"Raw Nash val={nash_val_raw:.4f}, expected ~{expected_raw_val:.4f}"
        )
        assert abs(nash_val_raw) > 0.01, (
            f"Raw Nash val should be nonzero (positional bias artifact), got {nash_val_raw:.4f}"
        )

    # --- 3c. Concrete example: raw vs antisym give clearly different Nash ---------

    def test_asymmetric_skill_raw_vs_antisym_give_different_nash(self):
        """
        M_raw = [[0, 0.3], [0.1, 0]] where:
          M[0][1] = 0.3: policy-0 earns 0.3 chips as P1 vs policy-1 as P2
          M[1][0] = 0.1: policy-1 earns 0.1 chips as P1 vs policy-0 as P2
          (both positive because P1 has a structural advantage in Kuhn Poker)

        NOT antisymmetric: M[0][1] + M[1][0] = 0.4 != 0.
        M_meta = antisymm = [[0, 0.1], [-0.1, 0]]  (policy-0 net advantage = 0.1)

        Analytical raw Nash: p2_raw ~= [0.75, 0.25]  (mixed — a/(a+b) = 0.3/0.4)
        Antisym Nash: row-0 dominates row-1 -> p2_meta ~= [1, 0]

        The correct opponent weights (antisym) say: use the stronger opponent (policy-0).
        The buggy opponent weights (raw) say: mix 75/25, diluting with the weaker policy.
        """
        M_raw = np.array([[0.0, 0.3], [0.1, 0.0]])
        M_meta = antisymmetrize(M_raw)

        assert np.allclose(M_meta, [[0.0, 0.1], [-0.1, 0.0]], atol=1e-9)

        _, p2_raw = compute_nash(M_raw)
        _, p2_meta = compute_nash(M_meta)

        # Raw Nash: mixed ~[0.75, 0.25] (analytical: a/(a+b) = 0.3/0.4)
        assert abs(p2_raw[0] - 0.75) < 0.05, (
            f"Raw p2[0] should be ~0.75, got {p2_raw[0]:.4f}"
        )

        # Meta Nash: row-0 dominates -> p2 concentrates on col-0 (use policy-0)
        assert p2_meta[0] > 0.9, (
            f"Meta p2[0] should be >0.9 (policy-0 dominates), got {p2_meta[0]:.4f}"
        )

        # The two p2 strategies differ significantly.
        assert not np.allclose(p2_raw, p2_meta, atol=0.05), (
            f"p2 should differ: raw={p2_raw}, meta={p2_meta}"
        )

    # --- 3d. 3-policy pool shows the structural issue at scale -------------------

    def test_3x3_antisymmetry_check(self):
        """3-policy pool: raw M is not antisymmetric due to positional bias."""
        p1_bias = -1.0 / 18.0
        skills = [0.0, 0.2, 0.5]

        n = 3
        M_raw = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    M_raw[i][j] = (skills[i] - skills[j]) + p1_bias

        assert not np.allclose(M_raw + M_raw.T, 0.0), (
            "Expected non-antisymmetric matrix due to positional bias."
        )

        M_meta = antisymmetrize(M_raw)
        expected_meta = np.array([
            [0.0, skills[0] - skills[1], skills[0] - skills[2]],
            [skills[1] - skills[0], 0.0, skills[1] - skills[2]],
            [skills[2] - skills[0], skills[2] - skills[1], 0.0],
        ])
        assert np.allclose(M_meta, expected_meta, atol=1e-9)

        # Nash on M_meta should favor the strongest policy (policy-2, skill=0.5).
        p1_meta, p2_meta = compute_nash(M_meta)
        assert p1_meta[2] > p1_meta[1] > p1_meta[0], (
            f"Nash p1 should rank policies by skill, got {p1_meta}"
        )
        assert is_nash(M_meta, p1_meta, p2_meta), (
            f"Nash BR gap={best_response_gap(M_meta, p1_meta, p2_meta):.4f}"
        )

    # --- 3e. Fix: antisymmetrize before Nash, then check BR ----------------------

    def test_antisymmetrized_matrix_satisfies_nash_br(self):
        """After antisymmetrizing, compute_nash must satisfy BR conditions."""
        rng = np.random.default_rng(42)
        n = 4
        skill = rng.uniform(0, 1, n)
        p1_bias = -1.0 / 18.0
        M_raw = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    M_raw[i][j] = (skill[i] - skill[j]) + p1_bias

        M_meta = antisymmetrize(M_raw)
        p1, p2 = compute_nash(M_meta)

        assert is_nash(M_meta, p1, p2), (
            f"BR gap={best_response_gap(M_meta, p1, p2):.4f} on antisymmetrized matrix"
        )

    # --- 3f. PSROLoop uses raw matrix (documents the bug) -----------------------

    def test_psro_p2_strategy_should_use_antisymmetric_matrix(self):
        """
        PSROLoop.on_policy_added() calls compute_nash(updated_matrix) where
        updated_matrix is the RAW (non-antisymmetric) matrix.

        This test shows that raw vs antisym p2_strategy differ on the concrete
        example M_raw=[[0, 0.3],[0.1, 0]]. The correct opponent weights (antisym)
        concentrate on the stronger policy-0; the buggy weights (raw) give a spurious
        mixed strategy that over-weights the weaker policy-1.
        """
        M_raw = np.array([[0.0, 0.3], [0.1, 0.0]])
        M_meta = antisymmetrize(M_raw)

        _, p2_current = compute_nash(M_raw)   # current (buggy) behavior
        _, p2_correct = compute_nash(M_meta)  # correct behavior

        # Raw: ~[0.75, 0.25] — gives 25% weight to weaker policy-1
        assert p2_current[1] > 0.2, (
            f"Raw p2[1] should be >0.2 (spurious weight on weaker policy), got {p2_current[1]:.4f}"
        )

        # Correct: ~[1, 0] — concentrates on stronger policy-0
        assert p2_correct[0] > 0.9, (
            f"Correct p2[0] should be >0.9 (dominant policy-0), got {p2_correct[0]:.4f}"
        )

        # The two p2_strategy values differ significantly.
        assert not np.allclose(p2_current, p2_correct, atol=0.05), (
            f"Expected p2 to differ: current={p2_current}, correct={p2_correct}"
        )
