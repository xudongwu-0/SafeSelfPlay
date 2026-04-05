import random
from typing import Optional

import gem
from gem import Env
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func


# Win lookup: action -> what it beats
WIN_LOOKUP = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}


class RockPaperScissorsEnv(Env):
    """Rock Paper Scissors environment where an LLM agent plays against a random opponent."""

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
        obs = self.render()
        return obs, {"env_instruction": self.get_instructions()}

    def step(self, action: str):
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
            obs = self.render()
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
        if player_move == opponent_move:
            result = "Draw"
            reward = 0.0
            self.draws += 1
        elif WIN_LOOKUP[player_move] == opponent_move:
            result = "Win"
            reward = 1.0
            self.wins += 1
        else:
            result = "Lose"
            reward = -1.0
            self.losses += 1

        self.history.append({"round": self.step_count, "player": player_move, "opponent": opponent_move, "result": result})

        terminated = self.step_count >= self.max_steps
        obs = self.render()
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

        # Final reward at episode end: bonus based on overall win rate
        if terminated:
            reward = win_rate

        return obs, reward, terminated, False, info

    def _opponent_action(self) -> str:
        """Generate opponent's move based on strategy."""
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

    def parse_action(self, text: str) -> dict:
        return default_parser_action_func(text, self.action_pattern, self.ACTION_LOOKUP, self.special_token_list)

    def render(self, mode: str = "text") -> str:
        if not self.history:
            return f"Rock Paper Scissors - Round 1/{self.max_steps}\nNo rounds played yet. Make your first move!"

        lines = [f"Rock Paper Scissors - Round {self.step_count + 1}/{self.max_steps}"]
        lines.append(f"Score: W={self.wins} D={self.draws} L={self.losses}")
        lines.append("")
        lines.append("History:")
        for h in self.history[-5:]:  # Show last 5 rounds
            lines.append(f"  R{h['round']}: You={h['player']} vs Opp={h['opponent']} -> {h['result']}")
        return "\n".join(lines)

    def sample_random_action(self) -> str:
        return random.choice(list(self.ACTION_LOOKUP.values()))

    def close(self):
        pass


if __name__ == "__main__":
    env: RockPaperScissorsEnv = gem.make(env_id="rock_paper_scissors", max_steps=5)
    obs, info = env.reset(seed=42)
    print(info["env_instruction"])
    print(obs)
    for _ in range(5):
        action_text = f"<answer>{env.sample_random_action()}</answer>"
        print(f"\nAction: {action_text}")
        obs, reward, terminated, truncated, info = env.step(action_text)
        print(obs, f"reward={reward}", info["action_desc"])
        if terminated:
            break
