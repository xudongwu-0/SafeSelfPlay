#!/bin/bash
# Launch a full session of A100x4 GRPO sweep variants sequentially.
# Run this after allocating an interactive node.
#
# Usage: SESSION=<1|2|3|4> bash scripts/delta/launch_grpo_sweep_a100.sh
#
# Session 1 (~60 min): round1_gpu_split (3 variants) + round2_train_bs (3 variants)
# Session 2 (~70 min): round3_rollout_bs (3 variants) + round4_group_size (4 variants)
# Session 3 (~90 min): round5_eager (2) + round6_mnt (4) + round7_gmu (3)
# Session 4 (~70 min): round8_infer_bs (4 variants)
#
# After each session: inspect /projects/bfoz/wchen11/kuhn_sweep_a100/ logs,
# update BASELINE_OVERRIDES in run_grpo_sweep_a100.sh with winners, then run next session.

set -e

SESSION=${SESSION:?Set SESSION=1, 2, or 3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_grpo_sweep_a100.sh"

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

run_variant() {
    local round=$1 variant=$2
    echo ""
    echo "##################################################"
    echo "# START: ${round}/${variant}  $(date)"
    echo "##################################################"
    ROUND=${round} VARIANT=${variant} bash ${SWEEP_SCRIPT}
    echo "##################################################"
    echo "# END: ${round}/${variant}  $(date)"
    echo "##################################################"
}

case "${SESSION}" in
    1)
        run_variant round1_gpu_split split_1tr3inf
        run_variant round1_gpu_split split_2tr2inf
        run_variant round1_gpu_split split_3tr1inf
        # --- Update BASELINE_OVERRIDES with R1 winner before continuing ---
        run_variant round2_train_bs bs4
        run_variant round2_train_bs bs8
        run_variant round2_train_bs bs16
        run_variant round2_train_bs bs32
        ;;
    2)
        run_variant round3_rollout_bs rb64
        run_variant round3_rollout_bs rb128
        run_variant round3_rollout_bs rb256
        run_variant round3_rollout_bs rb512
        # --- Update BASELINE_OVERRIDES with R3 winner before continuing ---
        run_variant round4_group_size gs1
        run_variant round4_group_size gs2
        run_variant round4_group_size gs4
        run_variant round4_group_size gs8
        ;;
    3)
        run_variant round5_eager eager_true
        run_variant round5_eager eager_false
        run_variant round6_mnt mnt4096
        run_variant round6_mnt mnt8192
        run_variant round6_mnt mnt16384
        run_variant round6_mnt mnt32768
        run_variant round7_gmu gmu90
        run_variant round7_gmu gmu92
        run_variant round7_gmu gmu95
        ;;
    4)
        run_variant round8_infer_bs ibs4
        run_variant round8_infer_bs ibs8
        run_variant round8_infer_bs ibs16
        run_variant round8_infer_bs ibs32
        ;;
    *)
        echo "ERROR: SESSION must be 1, 2, 3, or 4"; exit 2
        ;;
esac

echo ""
echo "===== SESSION ${SESSION} COMPLETE ====="
echo "Results: /projects/bfoz/wchen11/kuhn_sweep_a100/"
echo "WandB:   https://wandb.ai/kuhn-sweep-a100"
