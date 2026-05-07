import random
import re
from typing import Optional

import gem
from gem import Env
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func, parse_chance_action


CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}
CARD_STRENGTH = {0: "lowest", 1: "middle", 2: "highest"}


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
    NUM_START_STATES = 12  # 6 card pairs × 2 agent positions

    # Game states
    P0_ACTING = "p0_acting"
    P1_ACTING = "p1_acting"
    P0_RESPONDING = "p0_responding"
    GAME_OVER = "game_over"

    def __init__(
        self,
        max_steps: int = 2,
        format_penalty: float = 0.0,
        reasoning_reward: float = 0.0,
        action_pattern: str = r"<answer>(.*?)</answer>",
        special_token_list: tuple = ("<|im_start|>", "<|im_end|>"),
        env_instruction: Optional[str] = None,
        opponent_strategy: str = "llm",
        stochastic_policy: bool = False,
        debug_mode: bool = False,
        **kwargs,
    ):
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.reasoning_reward = reasoning_reward
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.opponent_strategy = opponent_strategy
        self.two_player = opponent_strategy == "llm"
        self.stochastic_policy = stochastic_policy
        self.debug_mode = debug_mode

        import re
        self.think_mode = re.compile(action_pattern).groups > 1

        if env_instruction is not None:
            self.env_instruction = env_instruction
        elif self.think_mode and stochastic_policy:
            self.env_instruction = (
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
                "Output a MIXED STRATEGY — probabilities for Pass and Bet that sum to 1.\n"
                "Reply ONLY in this format:\n"
                "<think>your reasoning</think><action>Pass N1/D Bet N2/D</action>\n"
                "where N1/D + N2/D = 1  (e.g., <action>Pass 1/3 Bet 2/3</action>)"
            )
        elif self.think_mode:
            self.env_instruction = (
                "As an expert poker strategist, analyze the following Kuhn Poker hand.\n"
                "Kuhn Poker: 3-card deck (Jack < Queen < King). Each player antes 1 chip.\n"
                "Only two actions exist: Pass or Bet (+1 chip). There is NO raise or fold action.\n"
                "Pass = check/fold depending on context. Bet = bet/call depending on context.\n\n"
                "Game flow:\n"
                "  P1: Pass or Bet\n"
                "  P2: Pass or Bet\n"
                "  If P1 passed and P2 bet → P1 gets one more chance: Pass(fold) or Bet(call)\n"
                "Showdown: higher card wins the pot.\n\n"
                "For each decision: consider your card strength and the expected value of "
                "betting vs passing, then choose.\n"
                "Reply ONLY in this format (nothing after the action tag):\n"
                "<think>your reasoning</think><action>Pass</action>\n"
                "or\n"
                "<think>your reasoning</think><action>Bet</action>"
            )
        else:
            self.env_instruction = (
                "As an expert poker strategist, analyze the following Kuhn Poker hand.\n"
                "Kuhn Poker: 3-card deck (Jack < Queen < King). Each player antes 1 chip.\n"
                "Actions: Pass or Bet (+1 chip). No raise exists.\n\n"
                "Game: P1 acts → P2 acts → (if P1 passed and P2 bet) P1 responds.\n"
                "Showdown: higher card wins. Fold: lose your ante.\n\n"
                "For each decision: note the pot size and board state, calculate pot odds, then choose.\n"
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
        self._last_think: str = ""
        self._last_action_probs: dict = {}
        self._last_action_taken: Optional[str] = None
        self._opponent_action_at_think: Optional[str] = None

    def get_instructions(self) -> str:
        action_str = "\nYour available actions are:\n" + ", ".join(self.ACTION_LOOKUP.values())
        return self.env_instruction + action_str

    def get_init_state_id(self) -> str:
        """Return canonical ID for the initial game state.

        Based only on env-visible state at reset (card deal + agent position),
        not on opponent actions. Call only after reset().
        """
        assert len(self.cards) == 2, "get_init_state_id() must be called after reset()"
        return f"p0card={self.cards[0]}_p1card={self.cards[1]}_agentisP0={self.agent_is_p0}"

    # All 12 start states: (p0_card, p1_card, agent_is_p0)
    _ALL_STATES = [
        (p0, p1, pos)
        for pos in [True, False]
        for p0, p1 in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
    ]

    def reset(self, seed: Optional[int] = None):
        Env.reset(self, seed)
        self._init_game_state()

        if self.debug_mode and seed is not None:
            # Cycle deterministically through all 12 start states
            state = self._ALL_STATES[seed % self.NUM_START_STATES]
            self.cards = {0: state[0], 1: state[1]}
            self.agent_is_p0 = state[2]
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random(seed)
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
        self._last_think = action_info.get("think_content") or ""
        self._last_action_probs = action_info.get("action_probs", {})

        # Invalid action → terminate with lowest reward (-2)
        if not is_valid:
            self.game_state = self.GAME_OVER
            self.final_reward = -2.0
            self.wins = 0
            info = self._make_info(False, metrics_agg_mode, "Invalid action format. Reward: -2")
            info.update(action_info)
            return "", self.final_reward, True, False, info

        action_name = self.ACTION_LOOKUP[action_info["action"]]
        self._last_action_taken = action_name
        if self.game_state == self.P1_ACTING:
            self._opponent_action_at_think = self.p0_action
        elif self.game_state == self.P0_RESPONDING:
            self._opponent_action_at_think = self.p1_action
        else:
            self._opponent_action_at_think = None

        handler_args = (action_name, is_valid, action_info, metrics_agg_mode)
        if self.game_state == self.P0_ACTING:
            obs, reward, terminated, truncated, info = self._handle_p0_acting(*handler_args)
        elif self.game_state == self.P1_ACTING:
            obs, reward, terminated, truncated, info = self._handle_p1_acting(*handler_args)
        elif self.game_state == self.P0_RESPONDING:
            obs, reward, terminated, truncated, info = self._handle_p0_responding(*handler_args)
        else:
            info = self._make_info(True, metrics_agg_mode, "Unexpected state.")
            return "", 0.0, True, False, info

        reward += info.pop("reasoning_bonus", 0.0)
        length_bonus = self.reasoning_reward * (len(action) / 4000)  # 4000 chars ≈ 1000 tokens ≈ 1 good-pattern unit
        reward += length_bonus
        info["metrics"]["reasoning/length_bonus"] = length_bonus
        return obs, reward, terminated, truncated, info

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
        win_stake = 2 if bet else 1
        lose_stake = 2 if bet else 1

        # Reward from agent's perspective
        if self.agent_is_p0:
            self.final_reward = float(win_stake) if p0_wins else float(-lose_stake)
        else:
            self.final_reward = float(-lose_stake) if p0_wins else float(win_stake)

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

    # Nash prescribes a deterministic action (P=0 or P=1) at these info-sets.
    # Mixed info-sets (J_P1_open, Q_P1_vs_bet, J_P2_vs_pass, Q_P2_vs_bet) are excluded.
    _NASH_DETERMINISTIC = {
        "Q_P1_open": "Pass",   # never bet Queen open
        "K_P1_open": "Bet",    # always bet King open
        "J_P1_vs_bet": "Pass", # always fold Jack to a raise
        "K_P1_vs_bet": "Bet",  # always call King
        "J_P2_vs_bet": "Pass", # always fold Jack facing bet
        "K_P2_vs_bet": "Bet",  # always call King
        "Q_P2_vs_pass": "Pass",# never bet Queen behind
        "K_P2_vs_pass": "Bet", # always bet King behind
    }

    def _make_info(self, is_valid: bool, agg: dict, desc: str) -> dict:
        win_rate = self.wins / self.step_count if self.step_count > 0 else 0.0

        # Behavioral metrics (only meaningful at game end)
        agent_card = self.cards[0 if self.agent_is_p0 else 1]

        agent_first_action = self.p0_action if self.agent_is_p0 else self.p1_action
        facing_bet = False
        folded = False
        if self.agent_is_p0 and self.p1_action == "Bet":
            facing_bet = True
            folded = self.p0_response == "Pass"
        elif not self.agent_is_p0 and self.p0_action == "Bet":
            facing_bet = True
            folded = self.p1_action == "Pass"

        # Per-info-set visit/bet indicators. Each game visits at most 2 agent info
        # sets (first decision, plus optionally P1_vs_bet if P1 passed and P2 bet).
        # Indicators are mean-aggregated across the rollout batch; the downstream
        # nash.compute_derived_metrics() recovers P(Bet | info set) as bet/visit
        # and computes Nash distance + exploitability.
        info_set_metrics = self._info_set_indicators()

        # --- Reasoning quality metrics ---
        think = getattr(self, "_last_think", "")

        think_present = 1.0 if think.strip() else 0.0

        _card_name = CARD_NAMES[agent_card].lower()
        _card_strength = CARD_STRENGTH[agent_card].lower()
        mentions_card = 1.0 if (
            re.search(rf'\b{_card_name}\b', think, re.IGNORECASE) or
            re.search(rf'\b{_card_strength}\b', think, re.IGNORECASE)
        ) else 0.0

        # J/Q claiming "highest/strongest/best card"; K claiming "lowest/weakest/worst card"
        _claims_highest = bool(re.search(r'\b(highest|strongest|best)\b.{0,30}\bcard\b', think, re.IGNORECASE))
        _claims_lowest = bool(re.search(r'\b(lowest|weakest|worst)\b.{0,30}\bcard\b', think, re.IGNORECASE))
        card_strength_error = 1.0 if (
            (_claims_highest and agent_card != 2) or (_claims_lowest and agent_card != 0)
        ) else 0.0

        # Explicit wrong card ordering: J beats Q/K, Q beats K, or J>Q/J>K/Q>K notation
        _rank = r'\b({a})\b.{{0,40}}\b(beats|beat|higher|stronger|better|above|greater)\b.{{0,40}}\b({b})\b'
        card_rank_error = 1.0 if any([
            re.search(_rank.format(a="jack",  b="queen"), think, re.IGNORECASE),
            re.search(_rank.format(a="jack",  b="king"),  think, re.IGNORECASE),
            re.search(_rank.format(a="queen", b="king"),  think, re.IGNORECASE),
            re.search(r'\bj\s*>\s*[qk]\b|\bq\s*>\s*k\b', think, re.IGNORECASE),
        ]) else 0.0

        # Opponent cannot hold the same card as the agent (one copy of each card in the deck).
        # Fire on positive claims ("opponent might/could have Jack") but not on negations
        # ("opponent can't have Jack").
        _opp_ref = r'(?:opponent|they|enemy|player\s*(?:1|2|one|two))'
        _poss_claim = bool(re.search(
            rf'\b{_opp_ref}\b.{{0,60}}\b(?:might|could|may|possibly|perhaps)\b.{{0,60}}\b{_card_name}\b',
            think, re.IGNORECASE))
        _direct_claim = bool(re.search(
            rf'\b{_opp_ref}\b.{{0,40}}\b(?:has|have|holds|hold|is\s+holding)\b.{{0,30}}\b{_card_name}\b',
            think, re.IGNORECASE))
        _neg_words = r"(?:not|no|never|doesn't|don't|can't|cannot|won't|impossible)"
        _neg_claim = bool(re.search(
            rf"\b{_opp_ref}\b.{{0,80}}\b{_neg_words}\b.{{0,40}}\b{_card_name}\b",
            think, re.IGNORECASE))
        impossible_opponent_card = 1.0 if ((_poss_claim or _direct_claim) and not _neg_claim) else 0.0

        # Concepts absent from Kuhn Poker: multi-card community games, compound raise types, blinds
        wrong_poker_rules = 1.0 if bool(re.search(
            r'\b(river|flop|turn|community\s+card|straight|flush|full\s+house|two\s+pair'
            r'|three\s+of\s+a\s+kind|royal\s+flush|re.?raise|big\s+blind|small\s+blind'
            r'|second\s+card|kicker)\b',
            think, re.IGNORECASE,
        )) else 0.0

        # Opponent action visible: P1 always sees P0's action; P0 sees P1's bet when responding
        _opp_action = getattr(self, "_opponent_action_at_think", None)
        hallucinated_opponent_action = 0.0
        if _opp_action is not None:
            _opp_word = _opp_action.lower()
            _general = bool(re.search(r'\b(opponent|they|enemy|player\s*(?:1|2|one|two))\b', think, re.IGNORECASE))
            _action_ref = bool(re.search(rf'\b{re.escape(_opp_word)}\b', think, re.IGNORECASE))
            mentions_opp_action = 1.0 if (_general or _action_ref) else 0.0
            # Detect claims that opponent did the opposite of what actually happened
            _wrong_verbs = (
                r'(?:pass(?:ed)?|fold(?:ed)?|check(?:ed)?)' if _opp_action == "Bet" else r'(?:bet(?:s|ted)?)'
            )
            _subj = r'(?:opponent|player\s*(?:1|2|one|two)|they)'
            _m = re.search(rf'\b{_subj}\b.{{0,30}}\b{_wrong_verbs}\b', think, re.IGNORECASE)
            if _m:
                _pre = think[max(0, _m.start() - 60):_m.start()]
                _hyp_words = r'\b(?:if|when|would|could|might|should|suppose|imagine|assuming)\b'
                if not re.search(_hyp_words, _pre, re.IGNORECASE):
                    hallucinated_opponent_action = 1.0

        # Reasoning-action consistency: only scored when think contains an unambiguous directive
        _action_taken = getattr(self, "_last_action_taken", None)
        _concludes_bet = bool(re.search(
            r'\b(should\s+bet|will\s+bet|i\s+(?:will\s+)?bet|best\s+to\s+bet|choose\s+bet|going\s+to\s+bet)\b',
            think, re.IGNORECASE))
        _concludes_pass = bool(re.search(
            r'\b(should\s+(?:pass|fold)|will\s+(?:pass|fold)|i\s+(?:will\s+)?(?:pass|fold)'
            r'|best\s+to\s+(?:pass|fold)|choose\s+pass|going\s+to\s+(?:pass|fold))\b',
            think, re.IGNORECASE))

        # nash_action_correct: at deterministic-Nash info-sets, did the agent play correctly?
        # Returns 1.0 (correct), 0.0 (wrong), or is omitted for mixed info-sets.
        from roll.pipeline.agentic.env.kuhn_poker.nash import CARD_CHR
        card_chr = CARD_CHR[agent_card]
        if self.agent_is_p0:
            iset_open = f"{card_chr}_P1_open"
            iset_response = f"{card_chr}_P1_vs_bet" if (self.p0_action == "Pass" and self.p1_action == "Bet") else None
            response_action = self.p0_response
        else:
            role = "P2_vs_bet" if self.p0_action == "Bet" else "P2_vs_pass"
            iset_open = f"{card_chr}_{role}"
            iset_response = None
            response_action = None

        nash_correct_vals = []
        for iset, action_taken in [(iset_open, agent_first_action), (iset_response, response_action)]:
            if iset is None or action_taken is None:
                continue
            nash_action = self._NASH_DETERMINISTIC.get(iset)
            if nash_action is not None:
                nash_correct_vals.append(1.0 if action_taken == nash_action else 0.0)

        base_metrics = {
            "action_is_valid": is_valid,
            "success": win_rate > 0.5,
            "win_rate": win_rate,
            "format_penalty": 0.0 if is_valid else self.format_penalty,
            "agent_bet": 1.0 if agent_first_action == "Bet" else 0.0,
            "bluff": 1.0 if (agent_card == 0 and agent_first_action == "Bet") else 0.0,
            "value_bet": 1.0 if (agent_card == 2 and agent_first_action == "Bet") else 0.0,
            "faced_bet": 1.0 if facing_bet else 0.0,
            "fold_vs_bet": 1.0 if (facing_bet and folded) else 0.0,
            "reasoning/think_present":            think_present,
            "reasoning/mentions_card":            mentions_card,
            "reasoning/card_strength_error":      card_strength_error,
            "reasoning/card_rank_error":          card_rank_error,
            "reasoning/impossible_opponent_card": impossible_opponent_card,
            "reasoning/wrong_poker_rules":        wrong_poker_rules,
        }
        if _opp_action is not None:
            base_metrics["reasoning/mentions_opponent_action"] = mentions_opp_action
            base_metrics["reasoning/hallucinated_opponent_action"] = hallucinated_opponent_action
        if _action_taken is not None and (_concludes_bet ^ _concludes_pass):
            base_metrics["reasoning/action_consistent"] = 1.0 if (
                (_concludes_bet and _action_taken == "Bet") or
                (_concludes_pass and _action_taken == "Pass")
            ) else 0.0

        if nash_correct_vals:
            base_metrics["nash_action_correct"] = sum(nash_correct_vals) / len(nash_correct_vals)

        probs = self._last_action_probs
        if probs:
            base_metrics["pass_prob"] = probs.get(0, 0.0)
            base_metrics["bet_prob"] = probs.get(1, 0.0)
        else:
            base_metrics["pass_prob"] = 0.0 if agent_first_action == "Bet" else 1.0
            base_metrics["bet_prob"] = 1.0 if agent_first_action == "Bet" else 0.0

        base_metrics.update(info_set_metrics)

        base_agg = {
            **agg,
            "agent_bet": "mean",
            "bluff": "mean",
            "value_bet": "mean",
            "faced_bet": "mean",
            "fold_vs_bet": "mean",
            "nash_action_correct": "mean",
            "pass_prob": "mean",
            "bet_prob": "mean",
            "reasoning/think_present":            "mean",
            "reasoning/mentions_card":            "mean",
            "reasoning/card_strength_error":      "mean",
            "reasoning/card_rank_error":          "mean",
            "reasoning/impossible_opponent_card": "mean",
            "reasoning/wrong_poker_rules":        "mean",
            "reasoning/mentions_opponent_action":      "mean",
            "reasoning/hallucinated_opponent_action":   "mean",
            "reasoning/action_consistent":              "mean",
        }
        base_agg.update({k: "mean" for k in info_set_metrics})
        base_agg["reasoning/reasoning_bonus"] = "mean"
        base_agg["reasoning/length_bonus"] = "mean"

        num_good = (
            think_present
            + mentions_card
            + base_metrics.get("reasoning/mentions_opponent_action", 0.0)
            + base_metrics.get("reasoning/action_consistent", 0.0)
        )
        num_errors = (
            card_strength_error
            + card_rank_error
            + impossible_opponent_card
            + wrong_poker_rules
            + hallucinated_opponent_action
        )
        reasoning_bonus = self.reasoning_reward * (num_good - num_errors)
        base_metrics["reasoning/reasoning_bonus"] = reasoning_bonus
        base_metrics["reasoning/length_bonus"] = 0.0  # filled in by step()

        return {
            "metrics": base_metrics,
            "metrics_agg_mode": base_agg,
            "action_desc": desc,
            "reasoning_bonus": reasoning_bonus,
        }

    def _info_set_indicators(self) -> dict:
        """0/1 indicators for each info set: did this game visit it, and did the agent bet?

        Mean-aggregated across a batch, the ratio bet/visit recovers P(Bet|info set).
        """
        from roll.pipeline.agentic.env.kuhn_poker.nash import ALL_INFO_SETS, CARD_CHR

        out = {f"kuhn_visit/{iset}": 0.0 for iset in ALL_INFO_SETS}
        out.update({f"kuhn_bet/{iset}": 0.0 for iset in ALL_INFO_SETS})

        agent_card_chr = CARD_CHR[self.cards[0 if self.agent_is_p0 else 1]]

        if self.agent_is_p0:
            # Agent acted first. p0_action is the open action.
            if self.p0_action is not None:
                key = f"{agent_card_chr}_P1_open"
                out[f"kuhn_visit/{key}"] = 1.0
                out[f"kuhn_bet/{key}"] = 1.0 if self.p0_action == "Bet" else 0.0
            # If P0 passed and P1 bet, agent had a second turn (P1_vs_bet).
            if self.p0_action == "Pass" and self.p1_action == "Bet" and self.p0_response is not None:
                key = f"{agent_card_chr}_P1_vs_bet"
                out[f"kuhn_visit/{key}"] = 1.0
                out[f"kuhn_bet/{key}"] = 1.0 if self.p0_response == "Bet" else 0.0
        else:
            # Agent acted second. p1_action is the response to p0's open.
            if self.p1_action is not None and self.p0_action is not None:
                role = "P2_vs_bet" if self.p0_action == "Bet" else "P2_vs_pass"
                key = f"{agent_card_chr}_{role}"
                out[f"kuhn_visit/{key}"] = 1.0
                out[f"kuhn_bet/{key}"] = 1.0 if self.p1_action == "Bet" else 0.0

        return out

    # --- Observation rendering ---

    def _render_p0_first_action(self) -> str:
        """Observation for poker-P0's first action."""
        card = CARD_NAMES[self.cards[0]]
        strength = CARD_STRENGTH[self.cards[0]]
        return (f"Your card: {card} ({strength} card). You are Player 1 (first to act).\n"
                f"Pot size: 2 chips (both antes). Board state: no action yet.\n"
                f"Choose: Pass or Bet.")

    def _render_p1_action(self) -> str:
        """Observation for poker-P1 after P0's action."""
        card = CARD_NAMES[self.cards[1]]
        strength = CARD_STRENGTH[self.cards[1]]
        pot = 3 if self.p0_action == "Bet" else 2
        call_cost = 1 if self.p0_action == "Bet" else 0
        pot_odds_str = (
            f"Pot odds: {pot}:{call_cost} (call {call_cost} to win {pot})."
            if call_cost else "No bet to call."
        )
        return (f"Your card: {card} ({strength} card). You are Player 2.\n"
                f"Pot size: {pot} chips. Board state: Player 1 chose {self.p0_action}.\n"
                f"{pot_odds_str}\n"
                f"Choose: Pass or Bet.")

    def _render_p0_response(self) -> str:
        """Observation for poker-P0 responding to P1's bet."""
        card = CARD_NAMES[self.cards[0]]
        strength = CARD_STRENGTH[self.cards[0]]
        return (f"Your card: {card} ({strength} card). You are Player 1.\n"
                f"Pot size: 3 chips. Board state: you passed, Player 2 bet.\n"
                f"Pot odds: 3:1 (call 1 to win 3).\n"
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

        # Stochastic policy: try chance action on extracted action_content first.
        if self.stochastic_policy and result["action_content"]:
            chance_idx, probs = parse_chance_action(result["action_content"], self.ACTION_LOOKUP)
            if chance_idx is not None:
                return {
                    "action": chance_idx,
                    "action_content": result["action_content"],
                    "think_content": result["think_content"],
                    "action_probs": probs,
                }

        if result["action"] is None:
            cleaned = text
            for token in self.special_token_list:
                cleaned = cleaned.replace(token, "")
            rev = {v.lower(): k for k, v in self.ACTION_LOOKUP.items()}

            # Stochastic policy: try chance format directly on full text (model forgot tags).
            if self.stochastic_policy:
                chance_idx, probs = parse_chance_action(cleaned, self.ACTION_LOOKUP)
                if chance_idx is not None:
                    return {
                        "action": chance_idx,
                        "action_content": cleaned,
                        "think_content": result["think_content"],
                        "action_probs": probs,
                    }

            # Accept <action> or legacy <answer> tags as fallbacks.
            m = re.search(r'<action>\s*(Pass|Bet)\s*</action>', cleaned, re.IGNORECASE)
            if not m:
                m = re.search(r'<answer>\s*(Pass|Bet)\s*</answer>', cleaned, re.IGNORECASE)
            if not m:
                # "Pass (fold)" or "Bet (call)" inside action/answer tags
                m = re.search(r'<(?:action|answer)>\s*(Pass|Bet)\s*\(', cleaned, re.IGNORECASE)
            if not m:
                # "action: Bet" / "answer:Pass" (model forgot angle brackets)
                m = re.search(r'(?:action|answer)\s*[:=]\s*(Pass|Bet)', cleaned, re.IGNORECASE)
            if m:
                _content = m.group(1).strip()
                result = {
                    "action": rev.get(_content.lower()),
                    "action_content": _content,
                    "think_content": "",
                }
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
    _m = info['metrics']
    print(f"  Invalid action defaults to Pass: valid={_m['action_is_valid']}, penalty={_m['format_penalty']}")
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
