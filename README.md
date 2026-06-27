# Mapping Engine

Realtime RTSP frame distribution foundation. One `FrameProvider` owns the
camera connection and decoded frames; local consumers subscribe asynchronously.

## Setup

```bash
./scripts/setup.sh
cd frontend && npm install
```

## Run

```bash
./scripts/run_backend.sh
./scripts/run_frontend.sh
```

Open <http://localhost:5173>. Override the camera with `RTSP_URL` and the
browser backend address with `VITE_BACKEND_URL`.

## Test

```bash
.venv/bin/pytest
```

