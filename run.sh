#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
ls data/processed/*.parquet >/dev/null 2>&1 || "$PY" scripts/download_processed.py
[ -f models/metrics.json ] || "$PY" train.py
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
