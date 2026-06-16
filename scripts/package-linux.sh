#!/usr/bin/env bash
set -euo pipefail

version="${1:-v0.1.0-alpha.1}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build-release"
dist_dir="${repo_root}/dist"
package_name="obs-frackground-${version}-linux-x86_64"
stage_dir="${dist_dir}/${package_name}"
package_root="${stage_dir}/obs-frackground"
model_path="${repo_root}/models/rvm_mobilenetv3_fp16.onnx"

if [[ ! -f "${model_path}" ]]; then
  bash "${repo_root}/scripts/download-rvm-models.sh"
fi

cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "${build_dir}"

rm -rf "${stage_dir}"
mkdir -p "${package_root}"

cmake --install "${build_dir}" --prefix "${package_root}"

mkdir -p "${package_root}/share/doc/obs-frackground"
install -m 0644 \
  "${repo_root}/README.md" \
  "${repo_root}/RELEASE.md" \
  "${repo_root}/LICENSE" \
  "${repo_root}/THIRD_PARTY_NOTICES.md" \
  "${package_root}/share/doc/obs-frackground/"

tar -C "${stage_dir}" -czf "${dist_dir}/${package_name}.tar.gz" "obs-frackground"

printf 'Created %s\n' "${dist_dir}/${package_name}.tar.gz"
