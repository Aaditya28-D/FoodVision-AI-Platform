#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_RETRIES=5
RETRY_DELAY=5

# =========================
# Hardcoded Google Drive IDs
# =========================
DATA_PART_00_FILE_ID="1un7DLYEXcM0_FP3F3sa5pC6_yCLYkAsK"
DATA_PART_01_FILE_ID="1Cgy8WCJ3Sftrv-Hu1YvXVS4LD9ihlCnV"
DATA_PART_02_FILE_ID="1QyPJrzubfLHrXcfB31VK7DBAl_QlSb9X"
DATA_PART_03_FILE_ID="1Ckd8-jZv-eUmLOEfzIV4Gr4I4BVZs3fh"
DATA_PART_04_FILE_ID="1gFpaoirgmRvUKzv3cs91C7RFjXy_sqxt"
MODELS_FILE_ID="1X7_yQn5A9KxGHeyFfTfD0wgfuQ2k8LWf"

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

mkdir -p downloads/data_parts

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
    echo "Missing Google Drive file ID for $label."
    exit 1
  fi

  if [ -f "$output_path" ] && [ -s "$output_path" ]; then
    echo "$label already exists: $output_path"
    return 0
  fi

  local attempt=1
  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    echo "Downloading $label (attempt $attempt/$MAX_RETRIES)..."
    rm -f "$output_path"

    if python3 -m gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"; then
      if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "$label downloaded successfully."
        return 0
      fi
    fi

    echo "$label download failed on attempt $attempt."
    rm -f "$output_path"

    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
      echo "Retrying $label in ${RETRY_DELAY}s..."
      sleep "$RETRY_DELAY"
    fi

    attempt=$((attempt + 1))
  done

  echo
  echo "Automatic download failed for $label after $MAX_RETRIES attempts."
  echo "Manual fallback:"
  echo "1. Download the file in your browser."
  echo "2. Save it as:"
  echo "   $output_path"
  echo "3. Run ./setup_and_run.sh again"
  echo
  exit 1
}

download_data_parts() {
  download_with_retries "${DATA_PART_00_FILE_ID}" "downloads/data_parts/data_assets.part_00" "DATA part 00"
  download_with_retries "${DATA_PART_01_FILE_ID}" "downloads/data_parts/data_assets.part_01" "DATA part 01"
  download_with_retries "${DATA_PART_02_FILE_ID}" "downloads/data_parts/data_assets.part_02" "DATA part 02"
  download_with_retries "${DATA_PART_03_FILE_ID}" "downloads/data_parts/data_assets.part_03" "DATA part 03"
  download_with_retries "${DATA_PART_04_FILE_ID}" "downloads/data_parts/data_assets.part_04" "DATA part 04"
}

join_data_parts() {
  local joined_archive="downloads/data_assets.tar.gz"

  if validate_archive "$joined_archive"; then
    echo "Joined data archive already exists and is valid: $joined_archive"
    return 0
  fi

  echo "Joining data archive parts..."
  rm -f "$joined_archive"
  cat \
    downloads/data_parts/data_assets.part_00 \
    downloads/data_parts/data_assets.part_01 \
    downloads/data_parts/data_assets.part_02 \
    downloads/data_parts/data_assets.part_03 \
    downloads/data_parts/data_assets.part_04 \
    > "$joined_archive"

  if ! validate_archive "$joined_archive"; then
    echo "Joined data archive is invalid after joining. Removing it..."
    rm -f "$joined_archive"
    exit 1
  fi

  echo "Joined data archive verified successfully."
}

download_models_archive() {
  local models_archive="downloads/model_assets.tar.gz"

  if validate_archive "$models_archive"; then
    echo "MODELS archive already exists and is valid: $models_archive"
    return 0
  fi

  if [ -f "$models_archive" ]; then
    echo "MODELS archive exists but is invalid/corrupted. Removing it..."
    rm -f "$models_archive"
  fi

  download_with_retries "${MODELS_FILE_ID}" "$models_archive" "MODELS archive"

  if ! validate_archive "$models_archive"; then
    echo "MODELS archive failed validation after download."
    rm -f "$models_archive"
    exit 1
  fi
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

download_data_parts
join_data_parts
download_models_archive

extract_if_needed "downloads/data_assets.tar.gz" "data/food-101/images" "data"
extract_if_needed "downloads/model_assets.tar.gz" "backend/models/weights" "models"

echo "Downloads and extraction complete."
