#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/motion-agent
# Official Motion-Agent checkpoint URL from szqwu/Motion-Agent.
gdown --no-cookies --continue 1Tagt2xUwv_h0JNMtrM_Ty1rWemkLF5jH -O models/motion-agent/motion_agent.zip
unzip -n models/motion-agent/motion_agent.zip -d models/motion-agent
