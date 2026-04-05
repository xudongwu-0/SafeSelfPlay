import random
from typing import Optional

import gem
from gem import Env
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func


# Win lookup: action -> what it beats
WIN_LOOKUP = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}


class RockPaperScissorsEnv(Env):
    """Rock Paper Scissors environment supporting both single-player and two-player modes.

    Single-player mode (opponent_strategy != "llm"):
        step(action) plays against a hardcoded opponent strategy.

    Two-player mode (opponent_strategy == "llm"):
        step() is called twice per round — once per player.
        Player 0 acts first, then player 1. Player 1 does NOT see player 0's choice.
        Round resolves after player 1's step, returning reward from player 0's perspective.
    """

    ACTION_LOOKUP = {0: "Rock", 1: "Paper", 2: "Scissors"}

    def __init__(
        self,
        max_steps: int = 10,
        format_penalty: float = 0.0,
        action_pattern: str = r"<answer>(.*?)</answer>",
        special_token_list: tuple = ("<|im_start|>", "<|im_end|>"),
        env_instruction: Optional[str] = None,
        opponent_strategy: str = "random",
        **kwargs,
    ):
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.opponent_strategy = opponent_strategy
        self.two_player = opponent_strategy == "llm"
        self.env_instruction = env_instruction or (
            "You are playing Rock Paper Scissors against an opponent. "
            "Each round, choose one action. Rock beats Scissors, Scissors beats Paper, Paper beats Rock. "
            "Try to win as many rounds as possible.\n\n"
            "You MUST respond with exactly one of the following three formats (use angle brackets, not parentheses):\n"
            "<answer>Rock</answer>\n"
            "<answer>Paper</answer>\n"
            "<answer>Scissors</answer>\n\n"
            "Do NOT output anything else. Do NOT use parentheses like (answer)."
        )

        self.step_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.history: list[dict] = []
        self.rng = random.Random()

        # Two-player state
        self.current_player = 0
        self.pending_move: Optional[str] = None  # player 0's move waiting for player 1

    def get_instructions(self) -> str:
        action_str = "\nYour available actions are:\n" + ", ".join(self.ACTION_LOOKUP.values())
        return self.env_instruction + action_str

    def reset(self, seed: Optional[int] = None):
        Env.reset(self, seed)
        self.rng = random.Random(seed)
        self.step_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.history = []
        self.current_player = 0
        self.pending_move = None
        obs = self._render_for_player(0)
        return obs, {"env_instruction": self.get_instructions()}

    def step(self, action: str):
        if self.two_player:
            return self._step_two_player(action)
        return self._step_single_player(action)

    def _step_single_player(self, action: str):
        """Original single-player step logic."""
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "win_rate": "last",
            "format_penalty": "mean",
        }

        self.step_count += 1
        action_info = self.parse_action(action)

        # Invalid action
        if action_info["action"] is None:
            obs = self._render_for_player(0)
            action_desc = f"Round {self.step_count}: Invalid action, no move made."
            metrics = {
                "action_is_valid": False,
                "success": self.wins > (self.step_count // 2),
                "win_rate": self.wins / self.step_count if self.step_count > 0 else 0.0,
                "format_penalty": self.format_penalty,
            }
            info = {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode, "action_desc": action_desc}
            info.update(action_info)
            truncated = self.step_count >= self.max_steps
            return obs, self.format_penalty, truncated, truncated, info

        # Opponent move
        opponent_move = self._opponent_action()
        player_move = self.ACTION_LOOKUP[action_info["action"]]

        # Determine outcome
        reward, result = self._resolve_round(player_move, opponent_move)

        self.history.append({"round": self.step_count, "player": player_move, "opponent": opponent_move, "result": result})

        terminated = self.step_count >= self.max_steps
        obs = self._render_for_player(0)
        action_desc = f"Round {self.step_count}: You played {player_move}, opponent played {opponent_move}. {result}!"

        win_rate = self.wins / self.step_count if self.step_count > 0 else 0.0
        metrics = {
            "action_is_valid": True,
            "success": win_rate > 0.5,
            "win_rate": win_rate,
            "format_penalty": 0.0,
        }
        info = {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode, "action_desc": action_desc}
        info.update(action_info)

        if terminated:
            reward = win_rate

        return obs, reward, terminated, False, info

    def _step_two_player(self, action: str):
        """Two-player step: called once per player per round."""
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "win_rate": "last",
            "format_penalty": "mean",
        }

        action_info = self.parse_action(action)

        if self.current_player == 0:
            # Player 0's turn: store move, return obs for player 1
            if action_info["action"] is None:
                self.pending_move = None
            else:
                self.pending_move = self.ACTION_LOOKUP[action_info["action"]]

            self.current_player = 1
            obs = self._render_for_player(1)
            # No reward yet — round not resolved
            info = {
                "metrics": {"action_is_valid": action_info["action"] is not None, "format_penalty": 0.0 if action_info["action"] is not None else self.format_penalty},
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": f"Player 0 has chosen.",
            }
            info.update(action_info)
            return obs, 0.0, False, False, info

        else:
            # Player 1's turn: resolve round
            self.step_count += 1

            if action_info["action"] is None:
                opponent_move = None
            else:
                opponent_move = self.ACTION_LOOKUP[action_info["action"]]

            player_move = self.pending_move

            # Handle invalid moves
            if player_move is None or opponent_move is None:
                self.current_player = 0
                self.pending_move = None
                obs = self._render_for_player(0)
                action_desc = f"Round {self.step_count}: Invalid action(s)."
                metrics = {
                    "action_is_valid": opponent_move is not None,
                    "success": self.wins > (self.step_count // 2),
                    "win_rate": self.wins / self.step_count if self.step_count > 0 else 0.0,
                    "format_penalty": self.format_penalty if player_move is None else 0.0,
                }
                truncated = self.step_count >= self.max_steps
                info = {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode, "action_desc": action_desc}
                info.update(action_info)
                return obs, self.format_penalty if player_move is None else 0.0, truncated, truncated, info

            # Resolve round (reward from player 0's perspective)
            reward, result = self._resolve_round(player_move, opponent_move)

            self.history.append({"round": self.step_count, "player": player_move, "opponent": opponent_move, "result": result})

            terminated = self.step_count >= self.max_steps
            self.current_player = 0
            self.pending_move = None
            obs = self._render_for_player(0)
            action_desc = f"Round {self.step_count}: You played {player_move}, opponent played {opponent_move}. {result}!"

            win_rate = self.wins / self.step_count if self.step_count > 0 else 0.0
            metrics = {
                "action_is_valid": True,
                "success": win_rate > 0.5,
                "win_rate": win_rate,
                "format_penalty": 0.0,
            }
            info = {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode, "action_desc": action_desc}
            info.update(action_info)

            if terminated:
                reward = win_rate

            return obs, reward, terminated, False, info

    def _resolve_round(self, player_move: str, opponent_move: str) -> tuple[float, str]:
        """Resolve a round, update stats. Returns (reward, result_str) from player 0's perspective."""
        if player_move == opponent_move:
            self.draws += 1
            return 0.0, "Draw"
        elif WIN_LOOKUP[player_move] == opponent_move:
            self.wins += 1
            return 1.0, "Win"
        else:
            self.losses += 1
            return -1.0, "Lose"

    def _opponent_action(self) -> str:
        """Generate opponent's move based on strategy (single-player mode only)."""
        if self.opponent_strategy == "random":
            return self.rng.choice(list(self.ACTION_LOOKUP.values()))
        elif self.opponent_strategy == "rock":
            return "Rock"
        elif self.opponent_strategy == "paper":
            return "Paper"
        elif self.opponent_strategy == "scissors":
            return "Scissors"
        elif self.opponent_strategy == "cycle":
            return self.ACTION_LOOKUP[self.step_count % 3]
        return self.rng.choice(list(self.ACTION_LOOKUP.values()))

    def _render_for_player(self, player: int) -> str:
        """Render observation for a specific player."""
        if not self.history:
            round_num = 1
            lines = [f"Rock Paper Scissors - Round {round_num}/{self.max_steps}"]
            if player == 0:
                lines.append("No rounds played yet. Make your first move!")
            else:
                lines.append("Your opponent has made their choice. Now make yours!")
            return "\n".join(lines)

        next_round = self.step_count + 1
        lines = [f"Rock Paper Scissors - Round {next_round}/{self.max_steps}"]
        lines.append(f"Score: W={self.wins} D={self.draws} L={self.losses}")
        lines.append("")
        lines.append("History:")
        for h in self.history[-5:]:  # Show last 5 rounds
            if player == 0:
                lines.append(f"  R{h['round']}: You={h['player']} vs Opp={h['opponent']} -> {h['result']}")
            else:
                # Swap perspective for player 1
                swapped_result = {"Win": "Lose", "Lose": "Win", "Draw": "Draw"}[h["result"]]
                lines.append(f"  R{h['round']}: You={h['opponent']} vs Opp={h['player']} -> {swapped_result}")

        if player == 1 and self.pending_move is not None:
            lines.append("")
            lines.append("Your opponent has made their choice for this round. Now make yours!")

        return "\n".join(lines)

    def render(self, mode: str = "text") -> str:
        return self._render_for_player(0)

    def parse_action(self, text: str) -> dict:
        return default_parser_action_func(text, self.action_pattern, self.ACTION_LOOKUP, self.special_token_list)

    def sample_random_action(self) -> str:
        return random.choice(list(self.ACTION_LOOKUP.values()))

    def close(self):
        pass


if __name__ == "__main__":
    # Test single-player mode
    print("=== Single Player Mode ===")
    env: RockPaperScissorsEnv = gem.make(env_id="rock_paper_scissors", max_steps=3)
    obs, info = env.reset(seed=42)
    print(obs)
    for _ in range(3):
        action_text = f"<answer>{env.sample_random_action()}</answer>"
        print(f"\nAction: {action_text}")
        obs, reward, terminated, truncated, info = env.step(action_text)
        print(obs, f"reward={reward}", info["action_desc"])
        if terminated:
            break

    # Test two-player mode
    print("\n=== Two Player Mode ===")
    env2: RockPaperScissorsEnv = gem.make(env_id="rock_paper_scissors", max_steps=3, opponent_strategy="llm")
    obs, info = env2.reset(seed=42)
    print(f"Player 0 sees:\n{obs}")
    for _ in range(3):
        # Player 0
        p0_action = f"<answer>{env2.sample_random_action()}</answer>"
        print(f"\nPlayer 0 action: {p0_action}")
        obs, reward, terminated, truncated, info = env2.step(p0_action)
        print(f"Player 1 sees:\n{obs}")
        if terminated:
            break
        # Player 1
        p1_action = f"<answer>{env2.sample_random_action()}</answer>"
        print(f"Player 1 action: {p1_action}")
        obs, reward, terminated, truncated, info = env2.step(p1_action)
        print(f"Player 0 sees:\n{obs}, reward={reward}")
        if terminated:
            break
