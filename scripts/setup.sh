#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
[[ -d .venv ]] || python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import cv2, fastapi, uvicorn; print('Backend dependencies OK')"
