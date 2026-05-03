#!/usr/bin/env python
"""Debug PSRO arena state coverage and per-state payoffs — no GPU needed.

Usage:
    python examples/debug_arena.py

Verifies:
1. debug_mode=True cycles through all 12 Kuhn Poker states with seeds 0-11.
2. Per-state payoffs under various fixed strategies to diagnose why the raw
   payoff matrix has many negative entries.
"""

import sys
import random
from typing import Callable

sys.path.insert(0, "/zfsauton/scratch/wentsec/ROLL")

from roll.pipeline.agentic.env.kuhn_poker.env import KuhnPokerEnv

CARD = {0: "J", 1: "Q", 2: "K"}
ACT = "<action>{}</action>"


# ── strategies ────────────────────────────────────────────────────────────────

def make_fixed(action: str) -> Callable:
    """Always play the same action."""
    def strategy(env: KuhnPokerEnv) -> str:
        return ACT.format(action)
    strategy.__name__ = f"always_{action.lower()}"
    return strategy


def nash(env: KuhnPokerEnv) -> str:
    """Kuhn Poker Nash equilibrium (alpha=1/3).

    Reads the CURRENT PLAYER's card via game_state so it works correctly
    for both agent and opponent calls.
    """
    state = env.game_state
    # Determine which player is currently acting and read their card
    if state in (KuhnPokerEnv.P0_ACTING, KuhnPokerEnv.P0_RESPONDING):
        card = env.cards[0]
    else:  # P1_ACTING
        card = env.cards[1]

    if state == KuhnPokerEnv.P0_ACTING:
        # P0's opening bet
        if card == 2:    return ACT.format("Bet")   # K: always bet
        elif card == 1:  return ACT.format("Pass")  # Q: always pass
        else:            return ACT.format("Bet") if random.random() < 1/3 else ACT.format("Pass")  # J: bluff 1/3

    elif state == KuhnPokerEnv.P1_ACTING:
        if env.p0_action == "Bet":
            # P0 bet → P1 calls or folds
            if card == 2:   return ACT.format("Bet")  # K: always call
            elif card == 1: return ACT.format("Bet") if random.random() < 1/3 else ACT.format("Pass")  # Q: call 1/3
            else:           return ACT.format("Pass")  # J: always fold
        else:
            # P0 passed → P1 can check or bet
            if card == 2:   return ACT.format("Bet")   # K: always bet
            elif card == 1: return ACT.format("Bet") if random.random() < 1/3 else ACT.format("Pass")  # Q: bet 1/3
            else:           return ACT.format("Pass")  # J: never bet into pass

    else:  # P0_RESPONDING to P1's bet
        if card == 2:   return ACT.format("Bet")   # K: always call
        else:           return ACT.format("Pass")  # J/Q: fold

nash.__name__ = "nash"


def random_action(env: KuhnPokerEnv) -> str:
    return ACT.format(random.choice(["Bet", "Pass"]))

random_action.__name__ = "random"


# ── game runner ───────────────────────────────────────────────────────────────

def play_game(env: KuhnPokerEnv, agent_strat: Callable, opponent_strat: Callable, seed: int) -> float:
    """Play one game. Agent=player_i, Opponent=player_j. Returns agent's reward."""
    obs, info = env.reset(seed=seed)

    if info.get("opponent_first"):
        # Opponent is P0 (first mover) — generate opponent's action
        opp_action = opponent_strat(env)
        obs, _, done, _, _ = env.step(opp_action)
        if done:
            return env.final_reward

    done = False
    while not done:
        # Agent acts
        action = agent_strat(env)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

        # Opponent acts (P0 responding to P1 bet, or normal P1 response)
        opp_action = opponent_strat(env)
        obs, _, done, _, _ = env.step(opp_action)

    return env.final_reward


# ── main ──────────────────────────────────────────────────────────────────────

def run_state_coverage_check():
    """Verify seeds 0-11 map to all 12 distinct states."""
    env = KuhnPokerEnv(debug_mode=True, action_pattern=r"<action>(.*?)</action>")
    print("=== STATE COVERAGE (seeds 0-11) ===")
    seen_states = set()
    for seed in range(12):
        _, info = env.reset(seed=seed)
        state_id = env.get_init_state_id()
        seen_states.add(state_id)
        print(f"  seed={seed:2d}: p0={CARD[env.cards[0]]} p1={CARD[env.cards[1]]}"
              f" agentIsP0={env.agent_is_p0}  ({'first' if env.agent_is_p0 else 'second'} mover)")
    print(f"\n  Unique states visited: {len(seen_states)} / 12")
    assert len(seen_states) == 12, "Not all states covered!"
    print("  PASS: all 12 states covered exactly once.\n")


def run_per_state_payoffs(agent_name: str, agent_strat: Callable, opp_name: str, opp_strat: Callable,
                          episodes_per_state: int = 100):
    """For each of the 12 states, run N games and show mean payoff."""
    env = KuhnPokerEnv(debug_mode=True, action_pattern=r"<action>(.*?)</action>")
    print(f"=== PER-STATE PAYOFFS: agent={agent_name} vs opponent={opp_name} ({episodes_per_state} ep/state) ===")
    total = []
    for si in range(12):
        payoffs = []
        for ep in range(episodes_per_state):
            seed = si + ep * 12  # si%12==si for any such seed, cycles through state si
            payoffs.append(play_game(env, agent_strat, opp_strat, seed=seed))
        mean = sum(payoffs) / len(payoffs)
        total.extend(payoffs)
        p0c = CARD[env._ALL_STATES[si][0]]
        p1c = CARD[env._ALL_STATES[si][1]]
        is_p0 = env._ALL_STATES[si][2]
        role = "agent=P0" if is_p0 else "agent=P1"
        print(f"  state{si:2d} p0={p0c} p1={p1c} {role}: mean={mean:+.3f}")
    overall = sum(total) / len(total)
    print(f"  Overall mean: {overall:+.3f}\n")


def run_matrix_vs_fixed(strategy_pairs, episodes_per_state: int = 200):
    """Show M[i][j] + M[j][i] sum to check zero-sum property."""
    env = KuhnPokerEnv(debug_mode=True, action_pattern=r"<action>(.*?)</action>")
    print("=== ZERO-SUM CHECK: M[i][j] + M[j][i] should ≈ 0 with shared seeds ===")
    for (an, a_strat), (bn, b_strat) in strategy_pairs:
        payoffs_ab, payoffs_ba = [], []
        for si in range(12):
            for ep in range(episodes_per_state):
                seed = si + ep * 12
                payoffs_ab.append(play_game(env, a_strat, b_strat, seed=seed))
                payoffs_ba.append(play_game(env, b_strat, a_strat, seed=seed))
        mab = sum(payoffs_ab) / len(payoffs_ab)
        mba = sum(payoffs_ba) / len(payoffs_ba)
        print(f"  M[{an} vs {bn}]={mab:+.3f}  M[{bn} vs {an}]={mba:+.3f}  sum={mab+mba:+.3f}")
    print()


if __name__ == "__main__":
    random.seed(42)

    run_state_coverage_check()

    strategies = [
        ("always_bet",  make_fixed("Bet")),
        ("always_pass", make_fixed("Pass")),
        ("random",      random_action),
        ("nash",        nash),
    ]

    # Per-state payoffs for each agent strategy against always_bet opponent
    for name, strat in strategies:
        run_per_state_payoffs(name, strat, "always_bet", make_fixed("Bet"), episodes_per_state=50)

    # Also show nash vs nash (should be ~0 overall)
    run_per_state_payoffs("nash", nash, "nash", nash, episodes_per_state=200)

    # Zero-sum check
    pairs = [
        (("always_bet", make_fixed("Bet")), ("always_pass", make_fixed("Pass"))),
        (("always_bet", make_fixed("Bet")), ("nash", nash)),
        (("random", random_action), ("nash", nash)),
    ]
    run_matrix_vs_fixed(pairs, episodes_per_state=200)
