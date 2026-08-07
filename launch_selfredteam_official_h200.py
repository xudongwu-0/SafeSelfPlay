#!/usr/bin/env python3
"""Launch the deployed Self-RedTeam H200 job as an independent Modal call."""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import modal
import requests


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parent / "selfplay-redteaming"
UPSTREAM_COMMIT = "0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123"
DEFAULT_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"


def main() -> None:
    local_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=UPSTREAM,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected upstream commit {UPSTREAM_COMMIT}, found {local_commit}"
        )

    reward_function = modal.Function.from_name(
        "selfredteam-wildguard", "wildguard_reward_app"
    )
    reward_url = reward_function.get_web_url()
    if not reward_url:
        raise RuntimeError("The deployed WildGuard reward function has no web URL")
    remote_rm_url = reward_url.rstrip("/") + "/classify"
    warmup_payload = {
        "queries": [
            {
                "game_idx": 0,
                "prompt": "How can I stay safe while hiking?",
                "response": "Share your route and check the weather.",
            }
        ]
    }

    def warm_reward_worker(_: int) -> None:
        last_error: BaseException | None = None
        for attempt in range(6):
            try:
                response = requests.post(
                    remote_rm_url,
                    json=warmup_payload,
                    timeout=900,
                )
                response.raise_for_status()
                return
            except requests.RequestException as error:
                last_error = error
                if attempt < 5:
                    time.sleep(min(60, 2 ** (attempt + 1)))
        assert last_error is not None
        raise last_error

    # The upstream trainer sends one reward request from each of four ranks.
    # Warm four single-input containers before allocating the H200 node.
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(warm_reward_worker, range(4)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        "selfredteam_official_repp_fullptx_sft_"
        "meta_llama_31_8b_instruct_abliterated_h200x4_s200_" + timestamp
    )
    wandb_run_id = uuid.uuid4().hex[:8]
    training_function = modal.Function.from_name(
        "selfredteam-official-h200", "run_official_selfredteam"
    )
    call = training_function.spawn(
        run_name=run_name,
        wandb_run_id=wandb_run_id,
        remote_rm_url=remote_rm_url,
        model_path=DEFAULT_MODEL,
        stop_at_step=200,
    )
    print(f"RUN_NAME={run_name}")
    print(f"WANDB_RUN_ID={wandb_run_id}")
    print(f"FUNCTION_CALL_ID={call.object_id}")
    print(
        "WANDB_URL=https://wandb.ai/2373025856w-the-university-of-hong-kong/"
        f"self-play/runs/{wandb_run_id}"
    )


if __name__ == "__main__":
    sys.exit(main())
