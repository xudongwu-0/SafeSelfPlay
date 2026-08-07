#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

STEPS_PER_ROLE="${STEPS_PER_ROLE:-50}"
RUN_SUFFIX="${RUN_SUFFIX:-dualfull_$(date +%Y%m%d_%H%M%S)}"

# A50 + D50 consumes the same 100 * 128 environment-prompt budget as the
# upstream 100-step shared-policy run. Both phases use full 8B parameters.
exec modal run --detach \
  modal_upstream_selfredteam_role_full.py::strict_upstream_aligned_dual_full_round \
  --steps-per-role "$STEPS_PER_ROLE" \
  --run-suffix "$RUN_SUFFIX" \
  --detach
