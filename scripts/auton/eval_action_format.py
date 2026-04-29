"""Sample the live vLLM server N times per case and report valid <action> format ratio."""
import re
import argparse
from openai import OpenAI

CASES = [
    ("P0_King_open",
     "Your card: King. You are Player 1 (first to act).\nPot size: 2 chips (both antes). Board state: no action yet.\nChoose: Pass or Bet."),
    ("P1_Queen_vs_bet",
     "Your card: Queen. You are Player 2.\nPot size: 3 chips. Board state: Player 1 chose Bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass or Bet."),
    ("P0_Jack_vs_bet",
     "Your card: Jack. You are Player 1.\nPot size: 3 chips. Board state: you passed, Player 2 bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass (fold) or Bet (call)."),
]

SYSTEM_PROMPT = (
    "As an expert poker strategist, analyze the following Kuhn Poker hand.\n"
    "Kuhn Poker: 3-card deck (Jack < Queen < King). Each player antes 1 chip.\n"
    "Only two actions exist: Pass or Bet (+1 chip). There is NO raise or fold action.\n"
    "Pass = check/fold depending on context. Bet = bet/call depending on context.\n\n"
    "Game flow:\n"
    "  P1: Pass or Bet\n"
    "  P2: Pass or Bet\n"
    "  If P1 passed and P2 bet → P1 gets one more chance: Pass(fold) or Bet(call)\n"
    "Showdown: higher card wins the pot.\n\n"
    "For each decision: note the pot size and board state, calculate pot odds, then choose.\n"
    "Reply ONLY in this format (nothing after the action tag):\n"
    "<think>your reasoning</think><action>Pass</action>\n"
    "or\n"
    "<think>your reasoning</think><action>Bet</action>\n"
    "Your available actions are:\n"
    "Pass, Bet"
)

VALID_RE = re.compile(r"<think>[\s\S]*?</think>\s*<action>(Pass|Bet)</action>", re.IGNORECASE)


def is_valid(text: str) -> bool:
    return bool(VALID_RE.search(text))


def extract_action(text: str) -> str:
    m = VALID_RE.search(text)
    return m.group(1) if m else "INVALID"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="gpu27")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n", type=int, default=50, help="samples per case")
    parser.add_argument("--temperature", type=float, default=0.99)
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="none")

    print(f"Sampling {args.n} times per case from {args.host}:{args.port}\n")

    total_valid = 0
    total_samples = 0

    for label, obs in CASES:
        valid_count = 0
        action_counts = {"Pass": 0, "Bet": 0, "INVALID": 0}

        for i in range(args.n):
            resp = client.chat.completions.create(
                model="qwen2.5-3b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs},
                ],
                max_tokens=512,
                temperature=args.temperature,
                top_p=0.99,
            )
            text = resp.choices[0].message.content
            action = extract_action(text)
            action_counts[action] = action_counts.get(action, 0) + 1
            if action != "INVALID":
                valid_count += 1

        total_valid += valid_count
        total_samples += args.n
        print(f"[{label}]")
        print(f"  valid format: {valid_count}/{args.n} ({100*valid_count/args.n:.1f}%)")
        print(f"  actions: Pass={action_counts['Pass']}  Bet={action_counts['Bet']}  INVALID={action_counts['INVALID']}\n")

    print(f"Overall valid format: {total_valid}/{total_samples} ({100*total_valid/total_samples:.1f}%)")


if __name__ == "__main__":
    main()
