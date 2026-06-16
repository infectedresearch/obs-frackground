#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${repo_root}/models"

mkdir -p "${model_dir}"

curl -L \
  --output "${model_dir}/rvm_mobilenetv3_fp16.onnx" \
  "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx"

printf 'Downloaded %s\n' "${model_dir}/rvm_mobilenetv3_fp16.onnx"
