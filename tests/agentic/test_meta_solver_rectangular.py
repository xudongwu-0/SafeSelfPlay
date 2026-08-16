"""Focused regression tests for rectangular degenerate zero-sum games."""

from __future__ import annotations

import numpy as np

from roll.pipeline.agentic.meta_solver import compute_nash


def test_one_by_one_remains_deterministic() -> None:
    p1, p2 = compute_nash(np.array([[7.0]]))

    np.testing.assert_array_equal(p1, np.array([1.0]))
    np.testing.assert_array_equal(p2, np.array([1.0]))


def test_single_row_chooses_unique_minimizing_column() -> None:
    p1, p2 = compute_nash(np.array([[3.0, -2.0, 1.0]]))

    np.testing.assert_array_equal(p1, np.array([1.0]))
    np.testing.assert_array_equal(p2, np.array([0.0, 1.0, 0.0]))


def test_single_row_is_uniform_over_tied_minimizing_columns() -> None:
    _, p2 = compute_nash(np.array([[4.0, -2.0, 3.0, -2.0]]))

    np.testing.assert_array_equal(p2, np.array([0.0, 0.5, 0.0, 0.5]))


def test_single_column_chooses_unique_maximizing_row() -> None:
    p1, p2 = compute_nash(np.array([[-2.0], [5.0], [1.0]]))

    np.testing.assert_array_equal(p1, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(p2, np.array([1.0]))


def test_single_column_is_uniform_over_tied_maximizing_rows() -> None:
    p1, _ = compute_nash(np.array([[3.0], [-1.0], [3.0], [2.0]]))

    np.testing.assert_array_equal(p1, np.array([0.5, 0.0, 0.5, 0.0]))


def test_general_matrix_path_still_solves_matching_pennies() -> None:
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])

    p1, p2 = compute_nash(payoff)

    np.testing.assert_allclose(p1, np.array([0.5, 0.5]), atol=1e-2)
    np.testing.assert_allclose(p2, np.array([0.5, 0.5]), atol=1e-2)
