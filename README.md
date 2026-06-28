# Mapping Engine

Realtime RTSP perception foundation. One `FrameProvider` owns the camera
connection and decoded frames; the `VisionPipeline` subscribes to it and
publishes structured ORB results for the React canvas overlay.

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

Open <http://localhost:5173>. The default source is
`rtsp://admin:zaq1xsw2@192.168.1.81:5543/live/channel0`; override it with
`RTSP_URL`. Override the browser backend address with `VITE_BACKEND_URL`.

## Test

```bash
.venv/bin/pytest
```
