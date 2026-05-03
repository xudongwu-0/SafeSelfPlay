#!/usr/bin/env python
"""Payoff mean/std analysis for Kuhn Poker to inform psro_episodes_per_pair choice.

Runs three matchup types and reports mean ± std of per-episode payoff, plus
standard error at N=12/36/60 so the user can judge how noisy the matrix entry is.

Analytical strategies (random, nash, always_bet) run without GPU.
Base model (Qwen2.5-3B-Instruct) requires a GPU — pass --no-base-model to skip.

Usage:
    python examples/eval_count_analysis.py
    python examples/eval_count_analysis.py --no-base-model
    python examples/eval_count_analysis.py --n-cycles 60
"""

import sys
import re
import random
import argparse
from typing import Callable

import numpy as np

sys.path.insert(0, "/zfsauton/scratch/wentsec/ROLL")

from roll.pipeline.agentic.env.kuhn_poker.env import KuhnPokerEnv
from examples.debug_arena import make_fixed, nash, random_action

MODEL_PATH = (
    "/zfsauton/scratch/wentsec/hf_cache/hub"
    "/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
    "/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)
ACTION_PATTERN = r"<action>(.*?)</action>"
_ACT_RE = re.compile(ACTION_PATTERN, re.IGNORECASE | re.DOTALL)

# System prompt that asks for <action> tags (used for base-model inference).
_BASE_SYSTEM_PROMPT = (
    "As an expert poker strategist, analyze the following Kuhn Poker hand.\n"
    "Kuhn Poker: 3-card deck (Jack < Queen < King). Each player antes 1 chip.\n"
    "Only two actions exist: Pass or Bet (+1 chip). There is NO raise or fold action.\n"
    "Pass = check/fold depending on context. Bet = bet/call depending on context.\n\n"
    "Game flow:\n"
    "  P1: Pass or Bet\n"
    "  P2: Pass or Bet\n"
    "  If P1 passed and P2 bet → P1 gets one more chance: Pass(fold) or Bet(call)\n"
    "Showdown: higher card wins the pot.\n\n"
    "Reply ONLY in this exact format (nothing else):\n"
    "<action>Pass</action>  or  <action>Bet</action>"
)


# ── strategy adapters ────────────────────────────────────────────────────────

Strategy = Callable[["KuhnPokerEnv", str], str]


def wrap(strat: Callable) -> Strategy:
    """Adapt a strategy(env) → action to strategy(env, obs) → action."""
    return lambda env, obs: strat(env)


def make_llm_strategy(model, tokenizer) -> Strategy:
    """Return an LLM strategy that calls model.generate() for each decision."""
    import torch

    def strategy(env: KuhnPokerEnv, obs: str) -> str:
        messages = [
            {"role": "system", "content": _BASE_SYSTEM_PROMPT},
            {"role": "user", "content": obs},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        m = _ACT_RE.search(response)
        if m:
            return f"<action>{m.group(1)}</action>"
        if "bet" in response.lower():
            return "<action>Bet</action>"
        return "<action>Pass</action>"

    return strategy


# ── game runner ───────────────────────────────────────────────────────────────

def play_game(env: KuhnPokerEnv, agent: Strategy, opp: Strategy, seed: int) -> float:
    """Play one episode; strategies receive (env, obs). Returns agent payoff."""
    obs, info = env.reset(seed=seed)

    if info.get("opponent_first"):
        opp_action = opp(env, obs)
        obs, _, done, _, _ = env.step(opp_action)
        if done:
            return env.final_reward

    done = False
    while not done:
        action = agent(env, obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
        opp_action = opp(env, obs)
        obs, _, done, _, _ = env.step(opp_action)

    return env.final_reward


def collect_payoffs(agent: Strategy, opp: Strategy, n_cycles: int) -> list[float]:
    """Collect n_cycles × 12 payoffs, cycling through all 12 start states."""
    env = KuhnPokerEnv(debug_mode=True, action_pattern=ACTION_PATTERN)
    return [
        play_game(env, agent, opp, seed=si + ep * 12)
        for ep in range(n_cycles)
        for si in range(12)
    ]


# ── reporting ─────────────────────────────────────────────────────────────────

def report(name: str, payoffs: list[float]) -> None:
    arr = np.array(payoffs)
    mean = arr.mean()
    std = arr.std()
    print(
        f"  {name:<38} N={len(arr):4d} | "
        f"mean={mean:+.4f}  std={std:.4f} | "
        f"SE@12={std/np.sqrt(12):.4f}  "
        f"SE@36={std/np.sqrt(36):.4f}  "
        f"SE@60={std/np.sqrt(60):.4f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cycles", type=int, default=30,
                        help="Analytical: cycles of 12 states (total = N×12, default 30→360)")
    parser.add_argument("--base-cycles", type=int, default=10,
                        help="Base model: cycles of 12 states (default 10→120; GPU is slow)")
    parser.add_argument("--no-base-model", action="store_true",
                        help="Skip base-model section (no GPU available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    hdr = f"{'Matchup':<38} {'N':>5}   {'mean':>8}  {'std':>7}   {'SE@12':>7}  {'SE@36':>7}  {'SE@60':>7}"

    print(f"\nKuhn Poker Payoff Analysis  (debug_mode=True, {args.n_cycles * 12} ep/matchup for analytical)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    # ── Analytical (no GPU) ──────────────────────────────────────────────────
    always_bet = wrap(make_fixed("Bet"))
    always_pass = wrap(make_fixed("Pass"))
    random_strat = wrap(random_action)
    nash_strat = wrap(nash)

    analytical = [
        ("random vs random",       random_strat, random_strat),
        ("random vs nash",         random_strat, nash_strat),
        ("random vs always_bet",   random_strat, always_bet),
        ("nash vs nash",           nash_strat,   nash_strat),
        ("nash vs always_bet",     nash_strat,   always_bet),
    ]

    for name, agent, opp in analytical:
        payoffs = collect_payoffs(agent, opp, n_cycles=args.n_cycles)
        report(name, payoffs)

    # ── Base model (GPU) ─────────────────────────────────────────────────────
    if not args.no_base_model:
        print(f"\n  [Base model: Qwen2.5-3B-Instruct — {args.base_cycles * 12} ep/matchup]")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"  Loading model from {MODEL_PATH} ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
            )
            model.eval()
            print("  Model loaded.\n")

            llm = make_llm_strategy(model, tokenizer)

            bm_matchups = [
                ("base_model vs base_model",  llm,        llm),
                ("base_model vs nash",         llm,        nash_strat),
                ("base_model vs always_bet",   llm,        always_bet),
            ]
            for name, agent, opp in bm_matchups:
                payoffs = collect_payoffs(agent, opp, n_cycles=args.base_cycles)
                report(name, payoffs)

        except Exception as exc:
            print(f"  ERROR loading/running base model: {exc}")

    print()
    print("  SE@N = std / sqrt(N).  Current production: N=36.")
    print("  Guideline: SE < 0.05 chips ≈ reliable enough for PSRO Nash computation.")


if __name__ == "__main__":
    main()
