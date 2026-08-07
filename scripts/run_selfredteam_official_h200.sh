#!/usr/bin/env bash
set -euo pipefail

cd /home/xudong/work/self_play/ROLL
modal deploy modal_selfredteam_wildguard.py
modal deploy modal_selfredteam_official_h200.py
exec python launch_selfredteam_official_h200.py "$@"
