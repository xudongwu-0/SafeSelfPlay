#!/bin/bash
# Submit a sweep round's variants serially via SLURM --dependency.
# Usage: ./launch_round.sh <round_dir> <variant1> <variant2> ...
# e.g.:  ./launch_round.sh round1_eager eager_true eager_false

set -e

ROUND_DIR=${1:?usage: launch_round.sh <round_dir> <variant1> [variant2 ...]}
shift

ROLL_DIR=/zfsauton/scratch/wentsec/ROLL
SBATCH_SCRIPT=${ROLL_DIR}/scripts/auton/run_kuhn_poker_sweep.sh

PREV_JOB=""
for VARIANT in "$@"; do
    if [ -z "${PREV_JOB}" ]; then
        JOB=$(sbatch --parsable --job-name="kuhn_sweep_${VARIANT}" \
              ${SBATCH_SCRIPT} ${ROUND_DIR} ${VARIANT})
    else
        JOB=$(sbatch --parsable --job-name="kuhn_sweep_${VARIANT}" \
              --dependency=afterany:${PREV_JOB} \
              ${SBATCH_SCRIPT} ${ROUND_DIR} ${VARIANT})
    fi
    echo "Submitted ${VARIANT}: job ${JOB}"
    PREV_JOB=${JOB}
done

echo ""
echo "Last job in chain: ${PREV_JOB}"
echo "Watch:  squeue -u \$USER"
