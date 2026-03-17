#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f "asset_links.env" ]; then
  echo "Missing asset_links.env in project root."
  exit 1
fi

source asset_links.env

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found."
  exit 1
fi

python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("gdown") is None:
    print("Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
PY

download_if_missing() {
  local file_id="$1"
  local output_path="$2"
  local label="$3"

  if [ -z "${file_id:-}" ] || [ "$file_id" = "PUT_${label}_FILE_ID_HERE" ]; then
    echo "Missing Google Drive file ID for $label in asset_links.env"
    exit 1
  fi

  if [ -f "$output_path" ]; then
    echo "$label archive already exists: $output_path"
    return
  fi

  echo "Downloading $label archive..."
  python3 -m gdown --id "$file_id" -O "$output_path"
}

mkdir -p downloads

download_if_missing "$DATA_FILE_ID" "downloads/data_assets.tar.gz" "DATA"
download_if_missing "$MODELS_FILE_ID" "downloads/model_assets.tar.gz" "MODELS"

extract_if_needed() {
  local archive="$1"
  local target_check="$2"
  local label="$3"

  if [ -e "$target_check" ]; then
    echo "$label already extracted."
    return
  fi

  echo "Extracting $label..."
  tar -xzf "$archive" -C .
}

extract_if_needed "downloads/data_assets.tar.gz" "data/food-101/images" "data"
extract_if_needed "downloads/model_assets.tar.gz" "backend/models/weights" "models"

echo "Downloads and extraction complete."
