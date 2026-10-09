#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo -e "Downloading glove (in use by the evaluators)"
gdown --no-cookies --continue https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing

unzip -n glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"
