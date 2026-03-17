#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Checking prerequisites..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found."
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required but not found."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not found."
  exit 1
fi

echo "Creating backend virtual environment if needed..."
cd backend

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  echo "Installing backend Python dependencies..."
  pip install -r requirements.txt
else
  echo "Warning: backend/requirements.txt not found. Skipping pip install."
fi

cd "$ROOT_DIR"

if [ -f "package.json" ]; then
  echo "Installing root npm dependencies..."
  npm install
fi

if [ -f "web/package.json" ]; then
  echo "Installing frontend npm dependencies..."
  cd web
  npm install
  cd "$ROOT_DIR"
fi

echo "Downloading and extracting assets..."
./scripts/download_assets.sh

echo "Verifying assets..."
python3 scripts/verify_assets.py

echo "Setup complete."
echo "Launching app..."
./run_app.sh
