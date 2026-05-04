"""Tests for LR scheduler monotonicity within PSRO iterations.

Verifies that _compute_scheduler_steps produces a num_training_steps large enough
that the cosine curve never completes (and therefore never rises) within one generation.
No Ray, DeepSpeed, or GPU required.
"""
import math
import torch
from transformers import get_cosine_schedule_with_warmup


def _make_scheduler(num_training_steps: int, num_warmup_steps: int = 2, lr: float = 3e-4):
    opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=lr)
    return opt, get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def _collect_lrs(scheduler, opt, n_steps: int) -> list[float]:
    lrs = []
    for _ in range(n_steps):
        scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


def _compute_scheduler_steps(
    num_pipeline_steps: int,
    rollout_batch_size: int,
    dp_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    ppo_epochs: int,
) -> int:
    backward_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    rollout_per_rank = rollout_batch_size // dp_size
    backward_steps_per_global = max(1, rollout_per_rank * ppo_epochs // backward_batch_size)
    return max(1, num_pipeline_steps * backward_steps_per_global)


# bubble144 config values
CONFIG = dict(
    rollout_batch_size=264,
    dp_size=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=11,
    ppo_epochs=1,
)
GENERATION_STEPS = 150   # fsp_score_timeout
FSP_SAVE_STEPS = 50      # iterations actually end here (fsp_save_steps)


def test_backward_steps_per_global():
    """There should be 2 scheduler steps per global pipeline step."""
    total = _compute_scheduler_steps(1, **CONFIG)
    assert total == 2, f"expected 2 backward steps per global step, got {total}"


def test_old_formula_oscillates():
    """Old formula (// dp_size) causes LR to oscillate within one iteration."""
    old_steps = max(1, GENERATION_STEPS // CONFIG["dp_size"])  # = 50
    # Simulate FSP_SAVE_STEPS=50 pipeline steps × 2 scheduler steps each = 100
    scheduler_steps_in_iteration = FSP_SAVE_STEPS * 2
    assert scheduler_steps_in_iteration > old_steps, "precondition: old formula too small"

    _, sched = _make_scheduler(old_steps)
    lrs = _collect_lrs(sched, _, scheduler_steps_in_iteration)

    # After the cosine completes (progress > 1), LR should rise. Check it actually does.
    post_minimum = lrs[old_steps:]  # steps past num_training_steps
    rose = any(post_minimum[i + 1] > post_minimum[i] for i in range(len(post_minimum) - 1))
    assert rose, "Expected old formula to cause LR to rise past minimum"


def test_fixed_formula_monotone_full_generation():
    """Fixed formula: LR monotonically non-increasing over a full generation."""
    total_steps = _compute_scheduler_steps(GENERATION_STEPS, **CONFIG)  # = 300
    opt, sched = _make_scheduler(total_steps)
    lrs = _collect_lrs(sched, opt, total_steps)

    # After warmup (2 steps), every step must be ≤ previous.
    post_warmup = lrs[2:]
    for i in range(len(post_warmup) - 1):
        assert post_warmup[i + 1] <= post_warmup[i] + 1e-10, (
            f"LR rose at scheduler step {i + 3}: {post_warmup[i]:.6f} → {post_warmup[i+1]:.6f}"
        )


def test_fixed_formula_monotone_early_end():
    """Fixed formula: LR monotonically non-increasing even when iteration ends early."""
    total_steps = _compute_scheduler_steps(GENERATION_STEPS, **CONFIG)  # = 300
    early_steps = FSP_SAVE_STEPS * 2  # 100 steps (iteration ends at fsp_save_steps=50)

    opt, sched = _make_scheduler(total_steps)
    lrs = _collect_lrs(sched, opt, early_steps)

    post_warmup = lrs[2:]
    for i in range(len(post_warmup) - 1):
        assert post_warmup[i + 1] <= post_warmup[i] + 1e-10, (
            f"LR rose at step {i + 3}: {post_warmup[i]:.6f} → {post_warmup[i+1]:.6f}"
        )


def test_reset_starts_at_max():
    """After early-end + reset, new scheduler starts at max LR (after warmup)."""
    total_steps = _compute_scheduler_steps(GENERATION_STEPS, **CONFIG)  # = 300
    early_steps = FSP_SAVE_STEPS * 2  # 100

    lr = 3e-4
    opt, sched = _make_scheduler(total_steps, lr=lr)
    _collect_lrs(sched, opt, early_steps)  # partial first generation

    # Reset: new scheduler for the next generation
    opt2, sched2 = _make_scheduler(total_steps, lr=lr)
    lrs2 = _collect_lrs(sched2, opt2, total_steps)

    # Peak LR (end of warmup) should equal the configured learning rate
    peak = max(lrs2[:5])
    assert abs(peak - lr) < 1e-8, f"After reset, peak LR {peak} ≠ configured lr {lr}"


def test_initial_and_reset_match():
    """Initial generation and subsequent resets have the same LR schedule."""
    total_steps = _compute_scheduler_steps(GENERATION_STEPS, **CONFIG)
    lr = 3e-4

    opt1, sched1 = _make_scheduler(total_steps, lr=lr)
    opt2, sched2 = _make_scheduler(total_steps, lr=lr)

    lrs1 = _collect_lrs(sched1, opt1, total_steps)
    lrs2 = _collect_lrs(sched2, opt2, total_steps)

    for i, (a, b) in enumerate(zip(lrs1, lrs2)):
        assert abs(a - b) < 1e-10, f"Step {i}: initial={a} reset={b}"


if __name__ == "__main__":
    test_backward_steps_per_global()
    print("✓ backward_steps_per_global = 2")

    test_old_formula_oscillates()
    print("✓ old formula (//dp_size) causes oscillation")

    test_fixed_formula_monotone_full_generation()
    print("✓ fixed formula: LR monotone over full generation (300 steps)")

    test_fixed_formula_monotone_early_end()
    print("✓ fixed formula: LR monotone over early-end (100 steps)")

    test_reset_starts_at_max()
    print("✓ after reset, LR peaks at configured max")

    test_initial_and_reset_match()
    print("✓ initial and reset schedulers are identical")

    print("\nAll tests passed.")
