"""
Rock Paper Scissors Environment

## Description
The agent plays Rock Paper Scissors against a random opponent for multiple rounds.
The goal is to learn to predict and counter opponent patterns.

## Action Space
- Rock
- Paper
- Scissors

Format: <answer>Rock</answer>

## Rewards
- Win: +1
- Draw: 0
- Lose: -1
"""

from .env import RockPaperScissorsEnv

__all__ = ["RockPaperScissorsEnv"]
