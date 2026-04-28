#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"
export PYTHONUNBUFFERED=1
exec python -m optimization.bo_loop --config config.example.yaml "$@"
