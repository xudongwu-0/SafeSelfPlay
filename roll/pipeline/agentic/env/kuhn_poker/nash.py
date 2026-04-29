"""Kuhn Poker Nash reference and derived-metric helpers.

Nash family is parameterized by alpha in [0, 1/3]; we use alpha=1/3 as the
reference strategy (all other alphas give the same game value -1/18 to P1).

Info-set keys are short strings "{card}_{role}" where card in {J,Q,K} and
role identifies where the agent's turn falls in the decision tree:
  P1_open    : agent is first mover, nobody has acted yet
  P1_vs_bet  : agent was first mover, passed, opponent bet back at them
  P2_vs_bet  : agent is second mover, opponent bet
  P2_vs_pass : agent is second mover, opponent passed

For each info set we publish the Nash Bet probability as a reference.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, Optional


ALPHA = 1.0 / 3.0

# Nash P(Bet) at alpha=1/3. See Kuhn 1950.
NASH_BET: Dict[str, float] = {
    "J_P1_open":    ALPHA,              # bluff Jack
    "Q_P1_open":    0.0,                # never bet Queen
    "K_P1_open":    3 * ALPHA,          # always bet King
    "J_P1_vs_bet":  0.0,                # fold Jack to a raise
    "Q_P1_vs_bet":  ALPHA + 1.0 / 3.0,  # mostly call Queen to raise
    "K_P1_vs_bet":  1.0,                # always call King
    "J_P2_vs_bet":  0.0,                # fold Jack facing bet
    "Q_P2_vs_bet":  1.0 / 3.0,          # mixed call Queen facing bet
    "K_P2_vs_bet":  1.0,                # always call King
    "J_P2_vs_pass": 1.0 / 3.0,          # mixed bluff Jack after check
    "Q_P2_vs_pass": 0.0,                # check Queen behind
    "K_P2_vs_pass": 1.0,                # bet King behind
}

ALL_INFO_SETS = list(NASH_BET.keys())

# Map card int (0=J, 1=Q, 2=K) to the key character used here.
CARD_CHR = {0: "J", 1: "Q", 2: "K"}

# Card rank for showdown comparison (higher wins).
RANK = {"J": 1, "Q": 2, "K": 3}


# -------- Exact game tree EV (exploitability) --------

def _game_ev_to_p1(p1_pol: Dict[str, float], p2_pol: Dict[str, float], c1: str, c2: str) -> float:
    """Expected value to P1 for a single deal (c1, c2), given both stochastic policies.

    Policies are dicts: {info_set_key: P(Bet)}. Missing keys default to 0 (Pass).
    """
    def pb(pol: Dict[str, float], key: str) -> float:
        return pol.get(key, 0.0)

    ev = 0.0
    # P1 open
    p_bet_p1 = pb(p1_pol, f"{c1}_P1_open")
    for a1, w1 in [("Bet", p_bet_p1), ("Pass", 1.0 - p_bet_p1)]:
        if w1 == 0.0:
            continue
        if a1 == "Bet":
            p_bet_p2 = pb(p2_pol, f"{c2}_P2_vs_bet")
            for a2, w2 in [("Bet", p_bet_p2), ("Pass", 1.0 - p_bet_p2)]:
                if w2 == 0.0:
                    continue
                if a2 == "Bet":
                    ev += w1 * w2 * (2.0 if RANK[c1] > RANK[c2] else -2.0)
                else:
                    ev += w1 * w2 * 1.0  # P2 folds, P1 wins 1 ante
        else:  # P1 passed
            p_bet_p2 = pb(p2_pol, f"{c2}_P2_vs_pass")
            for a2, w2 in [("Bet", p_bet_p2), ("Pass", 1.0 - p_bet_p2)]:
                if w2 == 0.0:
                    continue
                if a2 == "Bet":
                    p_bet_p1_again = pb(p1_pol, f"{c1}_P1_vs_bet")
                    for a1b, w1b in [("Bet", p_bet_p1_again), ("Pass", 1.0 - p_bet_p1_again)]:
                        if w1b == 0.0:
                            continue
                        if a1b == "Bet":
                            ev += w1 * w2 * w1b * (2.0 if RANK[c1] > RANK[c2] else -2.0)
                        else:
                            ev += w1 * w2 * w1b * (-1.0)  # P1 folds, P2 wins 1 ante
                else:  # both pass, showdown over antes
                    ev += w1 * w2 * (1.0 if RANK[c1] > RANK[c2] else -1.0)
    return ev


def expected_value_p1(p1_pol: Dict[str, float], p2_pol: Dict[str, float]) -> float:
    """Expected value to P1 over uniform card dealing (no duplicates)."""
    total = 0.0
    n = 0
    for c1, c2 in product("JQK", repeat=2):
        if c1 == c2:
            continue
        total += _game_ev_to_p1(p1_pol, p2_pol, c1, c2)
        n += 1
    return total / n


def _best_response_p2_value(p1_pol: Dict[str, float]) -> float:
    """Max EV to P2 (i.e. -P1 EV) over deterministic P2 policies.

    Enumerates 2**6 = 64 deterministic policies over P2's six info sets.
    """
    p2_info_sets = [f"{c}_{role}" for c in "JQK" for role in ("P2_vs_bet", "P2_vs_pass")]
    best_ev_p2 = -1e18
    for bits in range(1 << len(p2_info_sets)):
        p2_pol = {}
        for i, iset in enumerate(p2_info_sets):
            p2_pol[iset] = 1.0 if (bits >> i) & 1 else 0.0
        ev_p2 = -expected_value_p1(p1_pol, p2_pol)
        if ev_p2 > best_ev_p2:
            best_ev_p2 = ev_p2
    return best_ev_p2


def _best_response_p1_value(p2_pol: Dict[str, float]) -> float:
    """Max EV to P1 over deterministic P1 policies (2**6 enumeration)."""
    p1_info_sets = [f"{c}_{role}" for c in "JQK" for role in ("P1_open", "P1_vs_bet")]
    best_ev_p1 = -1e18
    for bits in range(1 << len(p1_info_sets)):
        p1_pol = {}
        for i, iset in enumerate(p1_info_sets):
            p1_pol[iset] = 1.0 if (bits >> i) & 1 else 0.0
        ev_p1 = expected_value_p1(p1_pol, p2_pol)
        if ev_p1 > best_ev_p1:
            best_ev_p1 = ev_p1
    return best_ev_p1


def exploitability(policy: Dict[str, float]) -> float:
    """Exploitability in chips per hand.

    The input policy is a single full strategy covering both P1 and P2 info sets —
    i.e. what the agent would play regardless of seat assignment. We compute:

        exp = 0.5 * (best_response_p2_gain + best_response_p1_gain)

    where best_response_*_gain is how much a perfect opponent earns above Nash
    when the agent plays 'policy' in the corresponding seat. Returns >=0.
    """
    nash_value_p1 = -1.0 / 18.0
    nash_value_p2 = +1.0 / 18.0
    br_p2_ev = _best_response_p2_value(policy)       # policy plays P1, BR plays P2
    br_p1_ev = _best_response_p1_value(policy)       # policy plays P2, BR plays P1
    return 0.5 * ((br_p2_ev - nash_value_p2) + (br_p1_ev - nash_value_p1))


# -------- Derived-metric computation for pipeline integration --------

def _extract_empirical_policy(
    metrics: Dict[str, float], env_tag: str
) -> Dict[str, float]:
    """Recover P(Bet | info_set) from mean-aggregated 'visit' + 'bet' indicators.

    Expects metric names:
        env/{env_tag}/kuhn_visit/{info_set}
        env/{env_tag}/kuhn_bet/{info_set}

    visit_mean = fraction of rollouts that visited this info set.
    bet_mean   = fraction of rollouts that visited AND bet.
    P(Bet | visit) = bet_mean / visit_mean.
    """
    policy: Dict[str, float] = {}
    for iset in ALL_INFO_SETS:
        vkey = f"env/{env_tag}/kuhn_visit/{iset}"
        bkey = f"env/{env_tag}/kuhn_bet/{iset}"
        if vkey not in metrics or bkey not in metrics:
            continue
        visit_mean = metrics[vkey]
        bet_mean = metrics[bkey]
        if visit_mean < 1e-9:
            continue
        policy[iset] = max(0.0, min(1.0, bet_mean / visit_mean))
    return policy


def compute_derived_metrics(
    metrics: Dict[str, float], env_tag: str = "KuhnPokerLLMThink"
) -> Dict[str, float]:
    """Given aggregated env metrics, emit Nash-distance and exploitability metrics.

    Returns a dict of new metrics (prefixed 'nash/' and 'kuhn/'). Empty if no
    Kuhn Poker metrics are present.
    """
    policy = _extract_empirical_policy(metrics, env_tag)
    if not policy:
        return {}

    derived: Dict[str, float] = {}
    total_l1 = 0.0
    n = 0
    for iset, p_bet in policy.items():
        nash_p = NASH_BET[iset]
        delta = abs(p_bet - nash_p)
        derived[f"nash/p_bet/{iset}"] = p_bet
        derived[f"nash/l1/{iset}"] = delta
        total_l1 += delta
        n += 1
        _eps = 1e-9
        entropy = -p_bet * math.log(p_bet + _eps) - (1 - p_bet) * math.log(1 - p_bet + _eps)
        derived[f"kuhn/entropy/{iset}"] = entropy

    if n > 0:
        avg_l1 = total_l1 / n
        derived["nash/l1_per_infoset"] = avg_l1
        derived["nash/tv_per_infoset"] = avg_l1 / 2.0  # TV distance for binary action
        derived["nash/infoset_coverage"] = n / float(len(ALL_INFO_SETS))
        derived["kuhn/entropy/mean"] = sum(derived[f"kuhn/entropy/{k}"] for k in policy) / n

    # Exploitability: needs all 12 info sets with defined probabilities.
    # For missing info sets fall back to Nash reference (best-case assumption).
    full_policy = dict(NASH_BET)
    full_policy.update(policy)
    derived["nash/exploitability"] = exploitability(full_policy)

    return derived
