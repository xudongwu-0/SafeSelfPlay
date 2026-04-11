import random
import re
from typing import Optional

import gem
from gem import Env
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func


CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}


class KuhnPokerEnv(Env):
    """Kuhn Poker environment for two-player self-play.

    3-card deck (J < Q < K). Each player antes 1, dealt 1 card.
    Poker-P0 acts first: Pass or Bet.
      - P0 Pass → P1: Pass (showdown) or Bet
        - P1 Bet → P0: Pass (fold) or Bet (call)
      - P0 Bet → P1: Pass (fold) or Bet (call)

    Two-player mode (opponent_strategy == "llm"):
        On reset(), agent is randomly assigned poker-P0 or poker-P1.
        When agent is poker-P1, info["opponent_first"]=True signals the
        manager to generate the opponent's first action before the agent acts.

    Rewards from agent's perspective:
        Showdown no bet: +1 (win) / -1 (lose)
        Fold: +1 (opponent folds) / -1 (agent folds)
        Showdown with bet: +2 (win) / -2 (lose)
    """

    ACTION_LOOKUP = {0: "Pass", 1: "Bet"}

    # Game states
    P0_ACTING = "p0_acting"
    P1_ACTING = "p1_acting"
    P0_RESPONDING = "p0_responding"
    GAME_OVER = "game_over"

    def __init__(
        self,
        max_steps: int = 2,
        format_penalty: float = 0.0,
        action_pattern: str = r"<answer>(.*?)</answer>",
        special_token_list: tuple = ("<|im_start|>", "<|im_end|>"),
        env_instruction: Optional[str] = None,
        opponent_strategy: str = "llm",
        **kwargs,
    ):
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.opponent_strategy = opponent_strategy
        self.two_player = opponent_strategy == "llm"

        import re
        self.think_mode = re.compile(action_pattern).groups > 1

        if env_instruction is not None:
            self.env_instruction = env_instruction
        elif self.think_mode:
            self.env_instruction = (
                "Kuhn Poker: 3 cards (Jack < Queen < King), each player antes 1 chip.\n"
                "Only two actions exist: Pass or Bet (+1 chip). There is NO raise or fold action.\n"
                "Pass = check/fold depending on context. Bet = bet/call depending on context.\n\n"
                "Game flow:\n"
                "  P1: Pass or Bet\n"
                "  P2: Pass or Bet\n"
                "  If P1 passed and P2 bet → P1 gets one more chance: Pass(fold) or Bet(call)\n"
                "Showdown: higher card wins the pot.\n\n"
                "Reply ONLY in this format (nothing after the answer tag):\n"
                "<think>your reasoning</think><answer>Pass</answer>\n"
                "or\n"
                "<think>your reasoning</think><answer>Bet</answer>"
            )
        else:
            self.env_instruction = (
                "Kuhn Poker: 3 cards (Jack < Queen < King), each player antes 1 chip.\n"
                "Actions: Pass or Bet (+1 chip). No raise exists.\n\n"
                "Game: P1 acts → P2 acts → (if P1 passed and P2 bet) P1 responds.\n"
                "Showdown: higher card wins. Fold: lose your ante.\n\n"
                "Format: <answer>Pass</answer> or <answer>Bet</answer>"
            )

        self.rng = random.Random()
        self._init_game_state()

    def _init_game_state(self):
        self.cards: dict[int, int] = {}  # player_idx -> card_value (0=J, 1=Q, 2=K)
        self.agent_is_p0: bool = True
        self.game_state: str = self.P0_ACTING
        self.p0_action: Optional[str] = None
        self.p1_action: Optional[str] = None
        self.p0_response: Optional[str] = None
        self.final_reward: float = 0.0
        self.step_count: int = 0
        self.wins: int = 0

    def get_instructions(self) -> str:
        action_str = "\nYour available actions are:\n" + ", ".join(self.ACTION_LOOKUP.values())
        return self.env_instruction + action_str

    def reset(self, seed: Optional[int] = None):
        Env.reset(self, seed)
        self.rng = random.Random(seed)
        self._init_game_state()

        # Deal cards: sample 2 from {0,1,2} without replacement
        deck = [0, 1, 2]
        self.rng.shuffle(deck)
        self.cards = {0: deck[0], 1: deck[1]}  # poker-P0, poker-P1

        # Randomly assign agent position
        self.agent_is_p0 = self.rng.choice([True, False])

        info = {"env_instruction": self.get_instructions()}

        if self.agent_is_p0:
            # Agent is poker-P0 (first mover) — normal flow
            obs = self._render_p0_first_action()
            info["opponent_first"] = False
        else:
            # Agent is poker-P1 — opponent (poker-P0) should act first
            obs = self._render_p0_first_action()  # obs for opponent (P0)
            info["opponent_first"] = True

        return obs, info

    def step(self, action: str):
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "win_rate": "last",
            "format_penalty": "mean",
        }

        # Game already over — return stored result
        if self.game_state == self.GAME_OVER:
            info = self._make_info(True, metrics_agg_mode, "Game already over.")
            return "", self.final_reward, True, False, info

        action_info = self.parse_action(action)
        is_valid = action_info["action"] is not None

        # Invalid action → default to Pass (fold/check) + penalty
        if not is_valid:
            action_info["action"] = 0  # Pass

        action_name = self.ACTION_LOOKUP[action_info["action"]]

        if self.game_state == self.P0_ACTING:
            return self._handle_p0_acting(action_name, is_valid, action_info, metrics_agg_mode)
        elif self.game_state == self.P1_ACTING:
            return self._handle_p1_acting(action_name, is_valid, action_info, metrics_agg_mode)
        elif self.game_state == self.P0_RESPONDING:
            return self._handle_p0_responding(action_name, is_valid, action_info, metrics_agg_mode)

        # Should never reach here
        info = self._make_info(True, metrics_agg_mode, "Unexpected state.")
        return "", 0.0, True, False, info

    def _handle_p0_acting(self, action_name: str, is_valid: bool, action_info: dict, agg: dict):
        """Poker-P0's first action: Pass or Bet."""
        self.p0_action = action_name
        self.game_state = self.P1_ACTING
        obs = self._render_p1_action()
        desc = f"Player 1 (first) chose {action_name}."
        info = self._make_info(is_valid, agg, desc)
        info.update(action_info)
        return obs, 0.0, False, False, info

    def _handle_p1_acting(self, action_name: str, is_valid: bool, action_info: dict, agg: dict):
        """Poker-P1's action after seeing P0's choice."""
        self.p1_action = action_name

        if self.p0_action == "Pass" and action_name == "Pass":
            # Both pass → showdown
            return self._resolve_showdown(bet=False, is_valid=is_valid, action_info=action_info, agg=agg)
        elif self.p0_action == "Pass" and action_name == "Bet":
            # P0 passed, P1 bets → P0 must respond
            self.game_state = self.P0_RESPONDING
            obs = self._render_p0_response()
            desc = "Player 2 bet. Player 1 must respond."
            info = self._make_info(is_valid, agg, desc)
            info.update(action_info)
            return obs, 0.0, False, False, info
        elif self.p0_action == "Bet" and action_name == "Pass":
            # P0 bet, P1 folds → P0 wins
            return self._resolve_fold(folder=1, is_valid=is_valid, action_info=action_info, agg=agg)
        else:
            # P0 bet, P1 calls → showdown
            return self._resolve_showdown(bet=True, is_valid=is_valid, action_info=action_info, agg=agg)

    def _handle_p0_responding(self, action_name: str, is_valid: bool, action_info: dict, agg: dict):
        """Poker-P0 responds to P1's bet (P0 had passed, P1 bet)."""
        self.p0_response = action_name

        if action_name == "Pass":
            # P0 folds → P1 wins
            return self._resolve_fold(folder=0, is_valid=is_valid, action_info=action_info, agg=agg)
        else:
            # P0 calls → showdown
            return self._resolve_showdown(bet=True, is_valid=is_valid, action_info=action_info, agg=agg)

    def _resolve_showdown(self, bet: bool, is_valid: bool, action_info: dict, agg: dict):
        """Resolve by comparing cards. Higher card wins."""
        self.game_state = self.GAME_OVER
        self.step_count = 1

        p0_wins = self.cards[0] > self.cards[1]
        stake = 2 if bet else 1  # pot contribution per player beyond ante

        # Reward from agent's perspective
        if self.agent_is_p0:
            self.final_reward = float(stake) if p0_wins else float(-stake)
        else:
            self.final_reward = float(-stake) if p0_wins else float(stake)

        self.wins = 1 if self.final_reward > 0 else 0

        winner = "Player 1" if p0_wins else "Player 2"
        desc = (f"Showdown: {CARD_NAMES[self.cards[0]]} vs {CARD_NAMES[self.cards[1]]}. "
                f"{winner} wins! Reward: {self.final_reward:+.0f}")
        info = self._make_info(is_valid, agg, desc)
        info.update(action_info)
        return "", self.final_reward, True, False, info

    def _resolve_fold(self, folder: int, is_valid: bool, action_info: dict, agg: dict):
        """Resolve by fold. The non-folding player wins the pot."""
        self.game_state = self.GAME_OVER
        self.step_count = 1

        # Folder loses 1 (their ante), winner gains 1
        if self.agent_is_p0:
            self.final_reward = 1.0 if folder == 1 else -1.0
        else:
            self.final_reward = -1.0 if folder == 1 else 1.0

        self.wins = 1 if self.final_reward > 0 else 0

        folder_name = "Player 1" if folder == 0 else "Player 2"
        desc = f"{folder_name} folds. Reward: {self.final_reward:+.0f}"
        info = self._make_info(is_valid, agg, desc)
        info.update(action_info)
        return "", self.final_reward, True, False, info

    def _make_info(self, is_valid: bool, agg: dict, desc: str) -> dict:
        win_rate = self.wins / self.step_count if self.step_count > 0 else 0.0

        # Behavioral metrics (only meaningful at game end)
        agent_card = self.cards[0 if self.agent_is_p0 else 1]
        agent_first_action = self.p0_action if self.agent_is_p0 else self.p1_action

        # Did agent face a bet?
        facing_bet = False
        folded = False
        if self.agent_is_p0 and self.p1_action == "Bet":
            facing_bet = True
            folded = self.p0_response == "Pass"
        elif not self.agent_is_p0 and self.p0_action == "Bet":
            facing_bet = True
            folded = self.p1_action == "Pass"

        return {
            "metrics": {
                "action_is_valid": is_valid,
                "success": win_rate > 0.5,
                "win_rate": win_rate,
                "format_penalty": 0.0 if is_valid else self.format_penalty,
                "agent_bet": 1.0 if agent_first_action == "Bet" else 0.0,
                "bluff": 1.0 if (agent_card == 0 and agent_first_action == "Bet") else 0.0,
                "value_bet": 1.0 if (agent_card == 2 and agent_first_action == "Bet") else 0.0,
                "faced_bet": 1.0 if facing_bet else 0.0,
                "fold_vs_bet": 1.0 if (facing_bet and folded) else 0.0,
            },
            "metrics_agg_mode": {
                **agg,
                "agent_bet": "mean",
                "bluff": "mean",
                "value_bet": "mean",
                "faced_bet": "mean",
                "fold_vs_bet": "mean",
            },
            "action_desc": desc,
        }

    # --- Observation rendering ---

    def _render_p0_first_action(self) -> str:
        """Observation for poker-P0's first action."""
        card = CARD_NAMES[self.cards[0]]
        return (f"Your card: {card}. You are Player 1 (first to act).\n"
                f"Choose: Pass or Bet.")

    def _render_p1_action(self) -> str:
        """Observation for poker-P1 after P0's action."""
        card = CARD_NAMES[self.cards[1]]
        return (f"Your card: {card}. You are Player 2.\n"
                f"Opponent (Player 1) chose: {self.p0_action}.\n"
                f"Choose: Pass or Bet.")

    def _render_p0_response(self) -> str:
        """Observation for poker-P0 responding to P1's bet."""
        card = CARD_NAMES[self.cards[0]]
        return (f"Your card: {card}. You are Player 1.\n"
                f"You passed, opponent (Player 2) bet.\n"
                f"Choose: Pass (fold) or Bet (call).")

    def render(self, mode: str = "text") -> str:
        if self.game_state == self.P0_ACTING:
            return self._render_p0_first_action()
        elif self.game_state == self.P1_ACTING:
            return self._render_p1_action()
        elif self.game_state == self.P0_RESPONDING:
            return self._render_p0_response()
        return "Game over."

    def parse_action(self, text: str) -> dict:
        result = default_parser_action_func(text, self.action_pattern, self.ACTION_LOOKUP, self.special_token_list)
        if result["action"] is None:
            cleaned = text
            for token in self.special_token_list:
                cleaned = cleaned.replace(token, "")
            rev = {v.lower(): k for k, v in self.ACTION_LOOKUP.items()}
            # "Pass (fold)" or "Bet (call)" inside answer tags
            m = re.search(r'<answer>\s*(Pass|Bet)\s*\(', cleaned, re.IGNORECASE)
            if not m:
                # "answer: Bet" / "answer:Pass" (model forgot angle brackets)
                m = re.search(r'answer\s*[:=]\s*(Pass|Bet)', cleaned, re.IGNORECASE)
            if m:
                result = {"action": rev.get(m.group(1).strip().lower()), "action_content": m.group(1).strip(), "think_content": ""}
        return result

    def sample_random_action(self) -> str:
        return random.choice(list(self.ACTION_LOOKUP.values()))

    def close(self):
        pass


if __name__ == "__main__":
    def play_game(cards: list[int], agent_is_p0: bool, actions: list[str]) -> float:
        """Helper: play a full game with given cards and action sequence."""
        env = KuhnPokerEnv(max_steps=2, action_pattern=r"<answer>(.*?)</answer>", opponent_strategy="llm")
        env.rng = random.Random(0)
        env._init_game_state()
        env.cards = {0: cards[0], 1: cards[1]}
        env.agent_is_p0 = agent_is_p0

        action_idx = 0
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(f"<answer>{a}</answer>")
            print(f"  step({a}): reward={reward}, terminated={terminated}, desc={info['action_desc']}")
            if terminated:
                break
        return reward

    # Test all 5 game branches
    print("=== Branch 1: P0=Pass, P1=Pass → Showdown ===")
    print("Agent=P0, cards: Q vs J → agent wins +1")
    r = play_game([1, 0], True, ["Pass", "Pass"])
    assert r == 1.0, f"Expected 1.0, got {r}"

    print("\n=== Branch 2: P0=Pass, P1=Bet, P0=Pass(fold) → P1 wins ===")
    print("Agent=P0, cards: J vs K → agent folds -1")
    r = play_game([0, 2], True, ["Pass", "Bet", "Pass"])
    assert r == -1.0, f"Expected -1.0, got {r}"

    print("\n=== Branch 3: P0=Pass, P1=Bet, P0=Bet(call) → Showdown ===")
    print("Agent=P0, cards: K vs J → agent wins +2")
    r = play_game([2, 0], True, ["Pass", "Bet", "Bet"])
    assert r == 2.0, f"Expected 2.0, got {r}"

    print("\n=== Branch 4: P0=Bet, P1=Pass(fold) → P0 wins ===")
    print("Agent=P0, cards: J vs Q → opponent folds, agent wins +1")
    r = play_game([0, 1], True, ["Bet", "Pass"])
    assert r == 1.0, f"Expected 1.0, got {r}"

    print("\n=== Branch 5: P0=Bet, P1=Bet(call) → Showdown ===")
    print("Agent=P0, cards: Q vs K → agent loses -2")
    r = play_game([1, 2], True, ["Bet", "Bet"])
    assert r == -2.0, f"Expected -2.0, got {r}"

    # Test agent as P1 (rewards flipped)
    print("\n=== Agent=P1 tests ===")
    print("Branch 1: P0=Pass, P1=Pass → Showdown. Agent=P1, cards: J vs K → agent(P1) wins +1")
    r = play_game([0, 2], False, ["Pass", "Pass"])
    assert r == 1.0, f"Expected 1.0, got {r}"

    print("\nBranch 4: P0=Bet, P1=Pass(fold). Agent=P1, cards: K vs J → agent(P1) folds -1")
    r = play_game([2, 0], False, ["Bet", "Pass"])
    assert r == -1.0, f"Expected -1.0, got {r}"

    print("\nBranch 5: P0=Bet, P1=Bet(call). Agent=P1, cards: J vs K → agent(P1) wins +2")
    r = play_game([0, 2], False, ["Bet", "Bet"])
    assert r == 2.0, f"Expected 2.0, got {r}"

    # Test GAME_OVER no-op
    print("\n=== GAME_OVER no-op test ===")
    env = KuhnPokerEnv(max_steps=2, opponent_strategy="llm")
    env._init_game_state()
    env.cards = {0: 1, 1: 0}
    env.agent_is_p0 = True
    env.step("<answer>Bet</answer>")
    obs, reward, terminated, _, info = env.step("<answer>Pass</answer>")  # P1 folds
    print(f"  Game ended: reward={reward}, terminated={terminated}")
    obs, reward, terminated, _, info = env.step("<answer>Bet</answer>")  # no-op
    print(f"  No-op call: reward={reward}, terminated={terminated}")
    assert terminated is True

    # Test invalid action
    print("\n=== Invalid action test ===")
    env = KuhnPokerEnv(max_steps=2, format_penalty=-0.01, opponent_strategy="llm")
    env._init_game_state()
    env.cards = {0: 2, 1: 0}
    env.agent_is_p0 = True
    obs, reward, terminated, _, info = env.step("garbage text")
    print(f"  Invalid action defaults to Pass: valid={info['metrics']['action_is_valid']}, penalty={info['metrics']['format_penalty']}")
    assert info["metrics"]["action_is_valid"] is False

    # Test reset with position randomization
    print("\n=== Reset position randomization ===")
    env = KuhnPokerEnv(max_steps=2, opponent_strategy="llm")
    positions = []
    for i in range(20):
        obs, info = env.reset(seed=i)
        positions.append(info.get("opponent_first", False))
    p1_count = sum(positions)
    print(f"  Agent as P1 (opponent_first): {p1_count}/20")
    assert 0 < p1_count < 20, "Position should be randomized"

    print("\nAll tests passed!")
