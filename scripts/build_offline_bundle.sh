#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
bundle="${2:-offline-bundle-${backend}}"

if [[ "$backend" != "paddleocr" && "$backend" != "mineru" ]]; then
  echo "usage: $0 paddleocr|mineru [output-directory]" >&2
  exit 2
fi
if [[ -e "$bundle" ]]; then
  echo "refusing to overwrite existing path: $bundle" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p "$bundle/wheels" "$bundle/models"
bundle="$(cd "$bundle" && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT

# shortcut: wheel bundles target the builder OS/architecture, rebuild for every target platform.
extra="ocr"
if [[ "$backend" == "mineru" ]]; then
  extra="mineru"
fi

uv build --wheel --out-dir "$bundle/wheels"
uv export \
  --frozen \
  --no-dev \
  --extra llm \
  --extra "$extra" \
  --no-emit-project \
  --no-hashes \
  --output-file "$bundle/requirements.txt"
uv run --with pip python -m pip wheel \
  --wheel-dir "$bundle/wheels" \
  --requirement "$bundle/requirements.txt"

if [[ "$backend" == "paddleocr" ]]; then
  uv run --with pip python -m pip wheel \
    --only-binary=:all: \
    --no-deps \
    --wheel-dir "$bundle/wheels" \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    paddlepaddle==3.2.0
  printf '\npaddlepaddle==3.2.0\n' >> "$bundle/requirements.txt"
fi

uv venv --python "$(tr -d '[:space:]' < .python-version)" "$temporary/venv"
uv pip install \
  --python "$temporary/venv/bin/python" \
  --no-index \
  --find-links "$bundle/wheels" \
  --requirement "$bundle/requirements.txt"
project_wheel="$(find "$bundle/wheels" -maxdepth 1 -name 'emr_extractor-*.whl' -print -quit)"
uv pip install --python "$temporary/venv/bin/python" --no-deps "$project_wheel"

"$temporary/venv/bin/emr-extractor" download-models \
  --backend "$backend" \
  --model-home "$bundle/models"

cp scripts/install_offline.sh docs/deployment.md "$bundle/"
printf '%s\n' "$backend" > "$bundle/BACKEND"
echo "Offline bundle ready: $bundle"
