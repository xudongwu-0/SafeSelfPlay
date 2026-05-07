import random
import re

from roll.utils.logging import get_logger

logger = get_logger()


def parse_chance_action(action_content: str, action_lookup: dict) -> tuple:
    """Parse mixed-strategy format: 'Pass 1/3 Bet 2/3' or 'Bet 2/3 Pass 1/3'.

    Primary format is n/m fractions; decimals accepted as fallback.
    Returns (sampled_action_idx, probs_dict) or (None, {}) if not recognized.
    """
    if not action_content:
        return None, {}

    rev = {v.lower(): k for k, v in action_lookup.items()}

    def _parse_prob(s: str) -> float:
        s = s.strip().rstrip("%")
        if "/" in s:
            num, denom = s.split("/", 1)
            n, d = float(num), float(denom)
            if d == 0:
                return 0.0 if n == 0 else float("inf")
            return n / d
        return float(s)

    actions = list(action_lookup.values())
    pattern = r"({actions})\s+([\d./]+%?)\s+({actions})\s+([\d./]+%?)".format(
        actions="|".join(re.escape(a) for a in actions)
    )
    m = re.search(pattern, action_content, re.IGNORECASE)
    if m:
        a1, p1_str, a2, p2_str = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            p1, p2 = _parse_prob(p1_str), _parse_prob(p2_str)
        except (ValueError, ZeroDivisionError):
            return None, {}

        # Handle inf: one action dominates → deterministic.
        if p1 == float("inf") or p2 == float("inf"):
            if p1 == float("inf") and p2 != float("inf"):
                p1, p2 = 1.0, 0.0
            elif p2 == float("inf") and p1 != float("inf"):
                p1, p2 = 0.0, 1.0
            else:
                return None, {}
        total = p1 + p2
        if total <= 0:
            return None, {}
        p1, p2 = p1 / total, p2 / total

        idx1, idx2 = rev.get(a1.lower()), rev.get(a2.lower())
        if idx1 is None or idx2 is None:
            return None, {}

        probs = {idx1: p1, idx2: p2}
        sampled = random.choices([idx1, idx2], weights=[p1, p2])[0]
        return sampled, probs

    # Single-action format: "Bet 1/1" or "Pass 1" — treat as deterministic.
    single_pattern = r"^\s*({actions})\s+([\d./]+%?)\s*$".format(
        actions="|".join(re.escape(a) for a in actions)
    )
    m2 = re.search(single_pattern, action_content.strip(), re.IGNORECASE)
    if m2:
        a, p_str = m2.group(1), m2.group(2)
        try:
            p = _parse_prob(p_str)
        except (ValueError, ZeroDivisionError):
            return None, {}
        if p <= 0:
            return None, {}
        idx = rev.get(a.lower())
        if idx is None:
            return None, {}
        other_idx = next(k for k in action_lookup if k != idx)
        probs = {idx: 1.0, other_idx: 0.0}
        return idx, probs

    return None, {}


def default_parser_action_func(text, action_pattern, action_lookup, special_token_list):
    if special_token_list is not None:
        for special_token in special_token_list:
            text = text.replace(special_token, "").strip()

    action = None
    match = re.search(action_pattern, text, re.DOTALL)
    if not match:
        action_info = {
            "action": action,
            "action_content": "",
            "think_content": ""
        }
        return action_info
    try:
        if len(match.groups()) == 1:
            think_content, action_content = "", match.group(1).strip()
        else:
            think_content, action_content = match.group(1).strip(), match.group(2).strip()
        action_content = action_content.strip()
        think_content = think_content.strip()

        action = action_content
        if action_lookup is not None and len(action_lookup) > 0:
            action = None
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            if action_content.lower() in rev_action_lookup:
                action = rev_action_lookup[action_content.lower()]

        action_info = {
            "action": action,
            "action_content": action_content,
            "think_content": think_content,
        }
        return action_info
    except Exception:
        logger.exception(f"default_parser_action_func: failed to parse text={text!r}")
        return {
            "action": action,
            "action_content": "",
            "think_content": "",
        }