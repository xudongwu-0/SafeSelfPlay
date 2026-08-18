#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SELFREDTEAM_ROOT:-${ROOT}/../selfplay-redteaming}"
REPOSITORY="https://github.com/mickelliu/selfplay-redteaming.git"
COMMIT="0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123"

if [[ ! -d "${TARGET}/.git" ]]; then
  git clone "${REPOSITORY}" "${TARGET}"
fi

git -C "${TARGET}" fetch origin "${COMMIT}"
git -C "${TARGET}" checkout --detach "${COMMIT}"
echo "Self-RedTeam ready at ${TARGET} (${COMMIT})"
