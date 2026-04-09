"""
Kuhn Poker Environment

## Description
Two-player Kuhn Poker: 3-card deck (Jack < Queen < King), each player antes 1,
dealt 1 card. Player 0 acts first. Supports two-player mode with random position assignment.

## Action Space
- Pass
- Bet

Format: <answer>Pass</answer> or <answer>Bet</answer>

## Rewards (from agent's perspective)
- Showdown (no bet): +1 win, -1 lose
- Fold: +1 opponent folds, -1 agent folds
- Showdown (with bet): +2 win, -2 lose
"""

from .env import KuhnPokerEnv

__all__ = ["KuhnPokerEnv"]
