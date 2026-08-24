#!/usr/bin/env bash
# Submit an alternating attacker/defender role-LoRA ladder to Slurm.
#
# Each phase trains against the previous phase's terminal adapter, so the phases
# are chained with --dependency=afterany and resolve the opponent path at run
# time rather than assuming a step count. This is latest-opponent self-play; for
# double-oracle PSRO see run_role_lora_cold_psro5_h200x4.sh.
#
#   SSP_ROOT=/path/to/workspace scripts/slurm/submit_ladder.sh A1 D1 A2 D2
#
# One phase runs at a time by construction: two concurrent phases on one node
# each start a Ray head and corrupt each other's placement-group scheduling.
set -eu -o pipefail

SSP_ROOT=${SSP_ROOT:?set SSP_ROOT to the workspace directory}
PSRO=${SSP_OUT:-$SSP_ROOT/psro}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ $# -gt 0 ]; then PHASES=("$@"); else PHASES=(A1 D1 A2 D2); fi

GPUS=${GPUS:-4}
STEPS=${STEPS:-100}
PARTITION=${PARTITION:-}
QOS=${QOS:-}
GRES=${GRES:-gpu:$GPUS}
TIME_LIMIT=${TIME_LIMIT:-03:00:00}
CPUS=${CPUS:-40}
MEM=${MEM:-440G}
LOGDIR=${LOGDIR:-$SSP_ROOT/logs}
mkdir -p "$LOGDIR" "$PSRO"

prev_tag=""
prev_job=""
port=5100
for tag in "${PHASES[@]}"; do
  case $tag in
    A*) role=attacker ;;
    D*) role=defender ;;
    *)  echo "phase must look like A1 or D2, got: $tag" >&2; exit 1 ;;
  esac
  gen=${tag//[!0-9]/}

  script=$(mktemp "$LOGDIR/${tag}_XXXX.sbatch")
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=ssp_$tag"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --gres=$GRES"
    echo "#SBATCH --cpus-per-task=$CPUS"
    echo "#SBATCH --mem=$MEM"
    echo "#SBATCH --time=$TIME_LIMIT"
    echo "#SBATCH --output=$LOGDIR/${tag}_%j.out"
    echo "#SBATCH --error=$LOGDIR/${tag}_%j.err"
    [ -n "$PARTITION" ] && echo "#SBATCH --partition=$PARTITION"
    [ -n "$QOS" ] && echo "#SBATCH --qos=$QOS"
    echo
    echo "export SSP_ROOT=$SSP_ROOT"
    [ -n "${SSP_WORK:-}" ]     && echo "export SSP_WORK=$SSP_WORK"
    [ -n "${SSP_HF_HOME:-}" ]  && echo "export SSP_HF_HOME=$SSP_HF_HOME"
    [ -n "${SSP_SFT_DIR:-}" ]  && echo "export SSP_SFT_DIR=$SSP_SFT_DIR"
    [ -n "${SSP_PYTHON:-}" ]   && echo "export SSP_PYTHON=$SSP_PYTHON"
    echo "export ROLE=$role GEN=$gen TAG=$tag STEPS=$STEPS GPUS=$GPUS"
    echo "export JUDGE_PORT=$port RAY_PORT=$((port + 1000))"
    if [ -n "$prev_tag" ]; then
      # Resolve at run time: the predecessor may stop short of STEPS.
      echo "export OPPONENT=\$(ls -d $PSRO/$prev_tag/ckpt/global_step*_hf 2>/dev/null | sort -V | tail -1)"
      echo "[ -n \"\$OPPONENT\" ] || { echo 'no $prev_tag checkpoint to train against' >&2; exit 1; }"
    fi
    echo "exec bash $HERE/role_phase.sh"
  } > "$script"

  if [ -n "$prev_job" ]; then
    job=$(sbatch --parsable --dependency=afterany:"$prev_job" "$script")
  else
    job=$(sbatch --parsable "$script")
  fi
  echo "$tag -> job $job${prev_tag:+ (after $prev_tag)}"
  prev_tag=$tag
  prev_job=$job
  port=$((port + 100))
done
