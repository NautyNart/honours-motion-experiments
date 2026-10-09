#!/usr/bin/env bash
# Run from the repository root with bash; safe to resume after interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p models .cache/huggingface
export HF_HOME="$PWD/.cache/huggingface"
# Limit concurrent transfers on connections where Xet transfers stall.
export HF_HUB_DISABLE_XET=1
status_file="$PWD/models/setup-download.status"
trap 'echo "FAILED at line $LINENO; see models/setup-download.log" > "$status_file"' ERR
echo "RUNNING: GloVe and evaluator checkpoints" > "$status_file"
if [[ ! -f glove/our_vab_data.npy ]]; then
    bash prepare/download_glove.sh
fi
if [[ ! -f checkpoints/.extractors-complete ]]; then
    bash prepare/download_extractor.sh
    touch checkpoints/.extractors-complete
fi
echo "RUNNING: Motion-Agent base checkpoint" > "$status_file"
bash prepare/download_base_model.sh
echo "RUNNING: IRG-MotionLLM stage 1–3 checkpoints" > "$status_file"
python prepare/download_irg_models.py
python prepare/link_assets.py
echo "COMPLETE: public downloads extracted and linked; Gemma access and HumanML3D data still required" > "$status_file"
