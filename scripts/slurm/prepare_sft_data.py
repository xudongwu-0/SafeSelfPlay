"""Build the postfill-CoT SFT files that role_lora_v2 mixes into RL.

Upstream reads these from a Modal volume. This repoints the module constants at
the in-repo sources and calls the same converters, so the on-disk format stays
theirs. Run after prepare_sources.py.

    SSP_ROOT=/path/to/workspace python scripts/slurm/prepare_sft_data.py

Writes $SSP_SFT_DIR (default $SSP_ROOT/sft_data) and prints the resolved
attacker and defender paths.
"""

from __future__ import annotations

import os
import pathlib
import sys


def main() -> int:
    root = os.environ.get("SSP_ROOT")
    if not root:
        print("set SSP_ROOT to the workspace directory", file=sys.stderr)
        return 2

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import prepare_sources  # noqa: F401  - installs the modal stub

    prepare_sources._stub_modal()
    repo = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    import modal_upstream_selfredteam_role_lora_v2 as v2

    work = pathlib.Path(os.environ.get("SSP_WORK", f"{root}/work"))
    out = pathlib.Path(os.environ.get("SSP_SFT_DIR", f"{root}/sft_data"))
    out.mkdir(parents=True, exist_ok=True)

    v2.UPSTREAM_WORK = work
    v2.ATTACKER_SFT_DATA = str(repo / "data/safety_selfplay/attacker_rewrite_1180.jsonl")
    v2.ATTACKER_RL_SFT_DATA = str(out / "attacker_rewrite_1180_rl_continuation.jsonl")
    v2.DEFENDER_RL_SFT_ROOT = out / "defender_rl_sft"

    attacker = v2._prepare_attacker_rl_sft_data()
    defender = v2._prepare_defender_rl_sft_data()
    print(f"attacker: {attacker}")
    for path in defender:
        print(f"defender: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
