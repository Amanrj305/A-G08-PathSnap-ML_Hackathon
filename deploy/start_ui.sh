#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m http.server "${PORT:-8501}" --bind 0.0.0.0 --directory ui
