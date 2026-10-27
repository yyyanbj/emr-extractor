#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
environment="${1:-$bundle_root/.venv}"
python_bin="${PYTHON_BIN:-python3.12}"

if [[ -e "$environment" ]]; then
  echo "refusing to overwrite existing environment: $environment" >&2
  exit 1
fi

"$python_bin" -m venv "$environment"
"$environment/bin/python" -m pip install \
  --no-index \
  --find-links "$bundle_root/wheels" \
  --requirement "$bundle_root/requirements.txt"

project_wheel="$(find "$bundle_root/wheels" -maxdepth 1 -name 'emr_extractor-*.whl' -print -quit)"
if [[ -z "$project_wheel" ]]; then
  echo "emr-extractor wheel not found in $bundle_root/wheels" >&2
  exit 1
fi
"$environment/bin/python" -m pip install --no-index --no-deps "$project_wheel"
"$environment/bin/python" -m pip check

backend="$(tr -d '[:space:]' < "$bundle_root/BACKEND")"
echo "Installed $backend deployment into $environment"
echo "Model home: $bundle_root/models"
echo "Run commands with --model-home '$bundle_root/models' --offline"
