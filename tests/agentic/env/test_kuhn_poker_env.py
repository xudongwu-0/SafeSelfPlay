"""Unit tests for KuhnPokerEnv terminal-state reward payouts.

All 5 game branches × 6 card permutations × 2 agent positions (60 cases),
plus zero-sum invariant and edge-case tests.

Card values: J=0, Q=1, K=2.
Branches:
  1. Pass / Pass              → showdown, no bet   → ±1
  2. Pass / Bet / Pass        → P0 folds            → P1 wins ±1
  3. Pass / Bet / Bet         → showdown, with bet  → ±2
  4. Bet  / Pass              → P1 folds            → P0 wins ±1
  5. Bet  / Bet               → showdown, with bet  → ±2
"""

import pytest

from roll.pipeline.agentic.env.kuhn_poker.env import KuhnPokerEnv

# J=0, Q=1, K=2
ALL_CARD_PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def _play(p0_card: int, p1_card: int, agent_is_p0: bool, actions: list[str]) -> float:
    """Run a scripted game and return the terminal reward seen by the agent."""
    env = KuhnPokerEnv(max_steps=2, action_pattern=r"<answer>(.*?)</answer>", opponent_strategy="llm")
    env._init_game_state()
    env.cards = {0: p0_card, 1: p1_card}
    env.agent_is_p0 = agent_is_p0

    reward = 0.0
    for action in actions:
        _, reward, terminated, _, _ = env.step(f"<answer>{action}</answer>")
        if terminated:
            break
    return reward


def _p0_wins(p0: int, p1: int) -> bool:
    return p0 > p1


# ---------------------------------------------------------------------------
# Branch 1: Pass / Pass → showdown, no bet (±1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch1_pass_pass_agent_p0(p0: int, p1: int) -> None:
    expected = 1.0 if _p0_wins(p0, p1) else -1.0
    assert _play(p0, p1, True, ["Pass", "Pass"]) == expected


@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch1_pass_pass_agent_p1(p0: int, p1: int) -> None:
    expected = -1.0 if _p0_wins(p0, p1) else 1.0
    assert _play(p0, p1, False, ["Pass", "Pass"]) == expected


# ---------------------------------------------------------------------------
# Branch 2: Pass / Bet / Pass → P0 folds, P1 wins (all ±1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch2_p0_folds_agent_p0(p0: int, p1: int) -> None:
    assert _play(p0, p1, True, ["Pass", "Bet", "Pass"]) == -1.0


@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch2_p0_folds_agent_p1(p0: int, p1: int) -> None:
    assert _play(p0, p1, False, ["Pass", "Bet", "Pass"]) == 1.0


# ---------------------------------------------------------------------------
# Branch 3: Pass / Bet / Bet → showdown, with bet (±2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch3_call_showdown_agent_p0(p0: int, p1: int) -> None:
    expected = 2.0 if _p0_wins(p0, p1) else -2.0
    assert _play(p0, p1, True, ["Pass", "Bet", "Bet"]) == expected


@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch3_call_showdown_agent_p1(p0: int, p1: int) -> None:
    expected = -2.0 if _p0_wins(p0, p1) else 2.0
    assert _play(p0, p1, False, ["Pass", "Bet", "Bet"]) == expected


# ---------------------------------------------------------------------------
# Branch 4: Bet / Pass → P1 folds, P0 wins (all ±1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch4_p1_folds_agent_p0(p0: int, p1: int) -> None:
    assert _play(p0, p1, True, ["Bet", "Pass"]) == 1.0


@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch4_p1_folds_agent_p1(p0: int, p1: int) -> None:
    assert _play(p0, p1, False, ["Bet", "Pass"]) == -1.0


# ---------------------------------------------------------------------------
# Branch 5: Bet / Bet → showdown, with bet (±2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch5_bet_showdown_agent_p0(p0: int, p1: int) -> None:
    expected = 2.0 if _p0_wins(p0, p1) else -2.0
    assert _play(p0, p1, True, ["Bet", "Bet"]) == expected


@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_branch5_bet_showdown_agent_p1(p0: int, p1: int) -> None:
    expected = -2.0 if _p0_wins(p0, p1) else 2.0
    assert _play(p0, p1, False, ["Bet", "Bet"]) == expected


# ---------------------------------------------------------------------------
# Zero-sum invariant: P0 reward + P1 reward == 0 for every terminal state
# ---------------------------------------------------------------------------

_BET_SHOWDOWN_BRANCHES = [
    ("Pass", "Bet", "Bet"),   # branch 3
    ("Bet", "Bet"),           # branch 5
]
_ALL_BRANCHES = [
    ("Pass", "Pass"),
    ("Pass", "Bet", "Pass"),
    ("Pass", "Bet", "Bet"),
    ("Bet", "Pass"),
    ("Bet", "Bet"),
]


@pytest.mark.parametrize("actions", _ALL_BRANCHES)
@pytest.mark.parametrize("p0,p1", ALL_CARD_PAIRS)
def test_zero_sum(p0: int, p1: int, actions: tuple[str, ...]) -> None:
    r_p0 = _play(p0, p1, True, list(actions))
    r_p1 = _play(p0, p1, False, list(actions))
    assert r_p0 + r_p1 == 0.0, f"Zero-sum broken: P0={r_p0}, P1={r_p1} for cards=({p0},{p1}) actions={actions}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_game_over_noop() -> None:
    """Extra step after terminal must return same reward and terminated=True."""
    env = KuhnPokerEnv(max_steps=2, opponent_strategy="llm")
    env._init_game_state()
    env.cards = {0: 1, 1: 0}  # Q vs J
    env.agent_is_p0 = True
    env.step("<answer>Bet</answer>")
    _, reward, terminated, _, _ = env.step("<answer>Pass</answer>")  # P1 folds → +1
    assert terminated and reward == 1.0
    _, reward2, terminated2, _, _ = env.step("<answer>Bet</answer>")  # no-op
    assert terminated2 and reward2 == 1.0


def test_invalid_action_defaults_to_pass_with_penalty() -> None:
    """Invalid format defaults to Pass and applies format_penalty."""
    env = KuhnPokerEnv(max_steps=2, format_penalty=-0.5, opponent_strategy="llm")
    env._init_game_state()
    env.cards = {0: 2, 1: 0}  # K vs J
    env.agent_is_p0 = True
    _, _, _, _, info = env.step("garbage text")
    assert info["metrics"]["action_is_valid"] is False
    assert info["metrics"]["format_penalty"] == -0.5


def test_reward_magnitude_no_bet_is_one() -> None:
    """No-bet showdown rewards are exactly ±1."""
    for p0, p1 in ALL_CARD_PAIRS:
        r = _play(p0, p1, True, ["Pass", "Pass"])
        assert abs(r) == 1.0


def test_reward_magnitude_bet_showdown_is_two() -> None:
    """Bet showdown rewards are exactly ±2 (not ±4)."""
    for p0, p1 in ALL_CARD_PAIRS:
        for actions in [["Pass", "Bet", "Bet"], ["Bet", "Bet"]]:
            r = _play(p0, p1, True, actions)
            assert abs(r) == 2.0, f"Expected ±2, got {r} for cards=({p0},{p1}) actions={actions}"
