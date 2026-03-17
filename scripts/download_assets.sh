#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_RETRIES=5
RETRY_DELAY=5

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

mkdir -p downloads

validate_archive() {
  local archive_path="$1"
  if [ ! -f "$archive_path" ]; then
    return 1
  fi

  if tar -tzf "$archive_path" >/dev/null 2>&1; then
    return 0
  fi

  return 1
}

download_with_retries() {
  local file_id="$1"
  local output_path="$2"
  local label="$3"

  if [ -z "${file_id:-}" ]; then
    echo "Missing Google Drive file ID for $label in asset_links.env"
    exit 1
  fi

  if validate_archive "$output_path"; then
    echo "$label archive already exists and is valid: $output_path"
    return 0
  fi

  if [ -f "$output_path" ]; then
    echo "$label archive exists but is invalid/corrupted. Removing it..."
    rm -f "$output_path"
  fi

  local attempt=1
  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    echo "Downloading $label archive (attempt $attempt/$MAX_RETRIES)..."

    rm -f "$output_path"

    if python3 -m gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"; then
      if validate_archive "$output_path"; then
        echo "$label archive downloaded and verified successfully."
        return 0
      else
        echo "$label archive downloaded but failed validation. Removing bad file..."
        rm -f "$output_path"
      fi
    else
      echo "$label archive download failed on attempt $attempt."
      rm -f "$output_path"
    fi

    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
      echo "Retrying $label in ${RETRY_DELAY}s..."
      sleep "$RETRY_DELAY"
    fi

    attempt=$((attempt + 1))
  done

  echo
  echo "Automatic download failed for $label after $MAX_RETRIES attempts."
  echo "Manual fallback:"
  echo "1. Download the archive in your browser."
  echo "2. Save it as:"
  echo "   $output_path"
  echo "3. Run ./setup_and_run.sh again"
  echo
  exit 1
}

extract_if_needed() {
  local archive="$1"
  local target_check="$2"
  local label="$3"

  if [ -e "$target_check" ]; then
    echo "$label already extracted."
    return 0
  fi

  if ! validate_archive "$archive"; then
    echo "$label archive is missing or invalid: $archive"
    exit 1
  fi

  echo "Extracting $label..."
  tar -xzf "$archive" -C .
}

download_with_retries "$DATA_FILE_ID" "downloads/data_assets.tar.gz" "DATA"
download_with_retries "$MODELS_FILE_ID" "downloads/model_assets.tar.gz" "MODELS"

extract_if_needed "downloads/data_assets.tar.gz" "data/food-101/images" "data"
extract_if_needed "downloads/model_assets.tar.gz" "backend/models/weights" "models"

echo "Downloads and extraction complete."
