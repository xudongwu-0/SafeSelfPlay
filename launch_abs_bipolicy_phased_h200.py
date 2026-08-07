#!/usr/bin/env python3
"""Submit the controlled A50-then-D50 dual-LoRA experiment."""
from __future__ import annotations

import sys
from datetime import datetime

import modal

from modal_abs_bipolicy_h200 import _warm_stable_reward_endpoint


def main() -> None:
    reward_function = modal.Function.from_name(
        "selfredteam-wildguard", "wildguard_reward_app"
    )
    reward_url = reward_function.get_web_url()
    if not reward_url:
        raise RuntimeError("The deployed WildGuard reward function has no web URL")
    remote_rm_url = reward_url.rstrip("/") + "/classify"
    _warm_stable_reward_endpoint(remote_rm_url)

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_spawn")
    run_name = (
        "abs_qwen25_3b_duallora_r32_phased_A50_D50_"
        "s100_rb128_tb32_mb8_aLR2e-6_dLR3e-6_"
        f"kl0p3_rolesft_ourprompts_{suffix}"
    )
    training_function = modal.Function.from_name(
        "abs-bipolicy-h200", "train_abs_bipolicy_h200"
    )
    call = training_function.spawn(
        remote_rm_url=remote_rm_url,
        target_step=100,
        resume_step=0,
        resume_checkpoint="",
        attacker_learning_rate=2e-6,
        defender_learning_rate=3e-6,
        training_schedule="attacker_then_defender",
        phase_switch_step=50,
        run_suffix=suffix,
    )
    print(f"RUN_NAME={run_name}")
    print(f"FUNCTION_CALL_ID={call.object_id}")
    print(
        "WANDB_PROJECT=https://wandb.ai/"
        "2373025856w-the-university-of-hong-kong/self-play"
    )


if __name__ == "__main__":
    sys.exit(main())
