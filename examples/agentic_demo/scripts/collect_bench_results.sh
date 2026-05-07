#!/bin/bash
# Collect timing results from benchmark sweep
SWEEP_DIR="/projects/bfoz/wchen11/kuhn_bench_sweep"

echo "=== Kuhn Poker Parallelism Benchmark Results ==="
echo "Date: $(date)"
echo ""
printf "%-12s  %-15s  %-15s  %-15s\n" "num_envs" "rollout(s)" "train(s)" "total(s)"
printf "%-12s  %-15s  %-15s  %-15s\n" "--------" "---------" "--------" "--------"

for NUM_ENVS in 1 8 16 32 64; do
    LOG=$(ls -t ${SWEEP_DIR}/envs${NUM_ENVS}/bench_*.out 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        printf "%-12s  %-15s  %-15s  %-15s\n" "$NUM_ENVS" "N/A" "N/A" "N/A"
        continue
    fi

    # Extract step 1 timings (step 0 includes init overhead)
    ROLLOUT=$(grep "time/step_rollout" "$LOG" | tail -1 | grep -oP '[\d.]+' | tail -1)
    TRAIN=$(grep "time/step_train" "$LOG" | tail -1 | grep -oP '[\d.]+' | tail -1)
    TOTAL=$(grep "time/step_total" "$LOG" | tail -1 | grep -oP '[\d.]+' | tail -1)

    printf "%-12s  %-15s  %-15s  %-15s\n" "$NUM_ENVS" "${ROLLOUT:-N/A}" "${TRAIN:-N/A}" "${TOTAL:-N/A}"
done

echo ""
echo "Log files:"
for NUM_ENVS in 1 8 16 32 64; do
    LOG=$(ls -t ${SWEEP_DIR}/envs${NUM_ENVS}/bench_*.out 2>/dev/null | head -1)
    [ -n "$LOG" ] && echo "  envs${NUM_ENVS}: $LOG"
done
