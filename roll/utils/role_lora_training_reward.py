"""Role-specific label-drift policy for Self-RedTeam LoRA training."""

from __future__ import annotations

from typing import Literal


TRAINING_LABEL_DRIFT_POLICY = "asymmetric-attacker-cap-zero-defender-drop-v1"

LabelDriftAction = Literal["keep", "cap_zero", "drop"]


def label_drift_training_action(
    *,
    optimizer_train_role: str,
    prompt_type: str,
    prompt_harmfulness: str | None,
    policy: str | None,
) -> LabelDriftAction:
    """Return the training action for one generated attack.

    Vanilla prompts are not attacker rewrites and therefore cannot drift.
    A missing WildGuard prompt label is left to the established tie/filtering
    behavior; only an explicit flip between harmful and unharmful is handled.
    """

    if optimizer_train_role not in {"attacker", "defender"}:
        raise ValueError(
            f"invalid optimizer_train_role: {optimizer_train_role!r}"
        )
    if policy in {None, ""}:
        return "keep"
    if policy != TRAINING_LABEL_DRIFT_POLICY:
        raise ValueError(f"unsupported label-drift training policy: {policy!r}")
    if prompt_harmfulness not in {"harmful", "unharmful", None}:
        raise ValueError(
            "prompt_harmfulness must be harmful, unharmful, or None, got "
            f"{prompt_harmfulness!r}"
        )

    expected = {
        "generated_harmful": "harmful",
        "generated_benign": "unharmful",
    }.get(prompt_type)
    if expected is None or prompt_harmfulness is None:
        return "keep"
    if prompt_harmfulness == expected:
        return "keep"
    return "cap_zero" if optimizer_train_role == "attacker" else "drop"


def cap_label_drift_attacker_reward(reward: float) -> float:
    """Prevent label drift from yielding a positive attacker reward."""

    return min(float(reward), 0.0)
