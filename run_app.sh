#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d "backend/.venv" ]; then
  echo "backend/.venv not found. Run ./setup_and_run.sh first."
  exit 1
fi

python3 scripts/verify_assets.py

echo "Starting app..."
cd backend
source .venv/bin/activate
npm run dev
