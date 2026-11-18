#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
.venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}" 2>&1 | tee results/api_server.log
