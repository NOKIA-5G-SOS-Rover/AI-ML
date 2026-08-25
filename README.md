# AI/ML Rover Module

Python software running on the Arduino UNO Q (Linux side) of the rover: camera capture, person detection, cloud alerts, motor control, SignalR communication, battery telemetry, and an optional autonomy mode.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Connects to the SignalR hub, drives motors via `arduino.app_utils.Bridge`, runs the autonomy state machine and telemetry loop, and supervises `ml.py` as a child process. |
| `ml.py` | **Current** camera pipeline (dual camera, ONNX person detection, one shared inference thread, local HTTP API, relays frames + alerts to the backend). |
| `ml_RecordAndDetect.py` | Older camera pipeline, kept only for reference. Not used by `main.py`. |
| `yolo26n_img480_int8.onnx` | ONNX model used by `ml.py` (480x480 input). |
| `requirements.txt` | Python dependencies. |
| `Dockerfile` |  |
| `.github/workflows/docker-publish.yml` | Builds/pushes the image to `ghcr.io/nokia-5g-sos-rover/rover-ai-module`. |
| `.env.example` | Example Cloud endpoint and rover/camera config. |

## Architecture

```
main.py
 ├── SignalR connection to DashboardHub
 ├── Bridge motor commands
 ├── manual control loop, watchdog, autonomy loop, telemetry loop
 └── spawns ml.py
       ├── cam1 capture ─┐
       ├── cam2 capture ─┤
       │                 └── shared ONNX inference thread
       ├── cam1 encoder / relay / HTTP API :8081
       └── cam2 encoder / relay / HTTP API :8082
```

Boots in **MANUAL** mode. Switch to autonomous with the `set-mode-autonomous` command. There is no distance/obstacle sensor — only cameras — so autonomy is a mitigation, not real obstacle avoidance. Always supervise and keep manual control available.

## Requirements

`requirements.txt`: `numpy`, `opencv-python-headless`, `signalrcore`, `requests`, `onnxruntime`.

Additionally, `main.py` needs `arduino.app_utils` (`App`, `Bridge`) — this is provided by the UNO Q board runtime, **not** installable via pip.

## Running

Normal launch (on the board):
```bash
arduino-app-cli app start ~/ArduinoApps/castel
```
This runs `main.py`, which starts `ml.py` automatically (`START_ML=1` by default). Sync the whole folder, not just `main.py` — the supervisor needs `ml.py` alongside it, and restarts it 5s after any crash.

To run the camera process separately:
```bash
START_ML=0 python main.py
python ml.py
```

## Configuration (environment variables)

**Cloud / identity**

| Variable | Default |
| --- | --- |
| `API_BASE` | `http://92.87.91.146:5000` |
| `ROVER_ID` | `ROVER-Q1` |
| `SESSION_ID` | auto-generated (UTC timestamp) |
| `START_ML` | `1` |
| `ENABLE_TELEMETRY` | `False` |

**Cameras** (`ml.py`; `main.py` reads/forwards the same device + port vars)

| Variable | Default |
| --- | --- |
| `CAM1_DEVICE` / `CAM2_DEVICE` | board-specific V4L2 paths |
| `CAM1_PORT` / `CAM2_PORT` | `8081` / `8082` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640` / `480` |
| `CAPTURE_FPS` | `25` |
| `CAMERA_FOURCC` | `MJPG` |
| `MODEL_PATH` | `yolo26n_img480_int8.onnx` |
| `INFERENCE_SIZE` | `480` |
| `CONFIDENCE_THRESHOLD` | `0.45` |
| `WEB_FPS` / `PUSH_FPS` | `8` / `5` |
| `JPEG_QUALITY` / `ALERT_JPEG_QUALITY` | `35` / `55` |
| `ALERT_CONFIRM_FRAMES` | `3` |
| `ALERT_CLEAR_FRAMES` | `3` |

Keep camera IDs, device paths, and ports in sync between `main.py` and `ml.py`. `main.py`'s autonomy code polls `cam1` at `http://localhost:8081`.

## Local Camera API

Each camera exposes its own port:

- `GET /` — identity + endpoint list
- `GET /api/detections` — latest detections (JSON)
- `GET /snapshot` — latest annotated JPEG (`503` before first frame)
- `GET /video_feed` — MJPEG stream

```bash
curl http://localhost:8081/api/detections
curl http://localhost:8081/snapshot --output cam1.jpg
```

## Detection & Alerts

Keeps only COCO class `0` (person) and filters detections using `CONFIDENCE_THRESHOLD`. A person alert is triggered after the person is detected in `ALERT_CONFIRM_FRAMES` consecutive inference frames (default 3). After an alert the detector must observe `ALERT_CLEAR_FRAMES` consecutive frames without a person (default 3) before another alert can be triggered. 

```json
{
  "alertType": "Human Detected",
  "source": "YOLOv26-Camera",
  "motorHaltRequested": true,
  "injuryClass": "Unknown"
}
```
POSTed to `${API_BASE}/events`, then the frame is uploaded to `/events/{id}/image`. Alert HTTP uploads run on a separate thread, so backend latency does not block inference. Frames are separately relayed to `${API_BASE}/stream/{camera_id}/frame`.

Autonomy can fire a second, distinct alert on arrival: `alertType: "PERSON_REACHED"`, `source: "autonomy"`.

⚠️ `ml.py` expects an end-to-end ONNX model, output shape `(N, 6)` = `[x1, y1, x2, y2, confidence, class_id]`. `ml_RecordAndDetect.py` expects a different (YOLOv8 class-score) format. Don't swap models/scripts without matching the format and `INFERENCE_SIZE`.

## Manual Control

SignalR `ReceiveCommand` payload: `roverId`, `command`, optional `speed` (0–100%, converted to 0–255 PWM; default 150 if omitted), optional `degrees`.

Commands: `forward`, `backward`, `stop`, `turn-left`, `turn-right`, `arc-left`, `arc-right`, `turn-degrees`, `set-mode-manual`, `set-mode-autonomous`, `set-speed`.

Latest-state-wins (no queue). A moving command with no update for 10s triggers the watchdog and stops. SignalR errors/disconnects also stop the motors immediately.

## Autonomy

1. **SCANNING** — rotate in short bursts looking for a person (confidence ≥ 0.6).
2. **APPROACHING** — center on the person, pulse forward.
3. **HOLDING** — stop once the person's bounding box covers ≥45% of the frame; hold 120s, then resume scanning.

Safety tuning in `main.py`: 3s lost-target timeout, 0.15 center dead zone, 110 max approach speed, short pulses. Camera-only — not real obstacle avoidance.

## Telemetry

Off by default (`ENABLE_TELEMETRY=False`). When enabled: reads `get_battery_raw` via Bridge, converts to % using voltage-divider constants in `main.py` (`BATTERY_VREF=3.3`, 10k/10k resistors, 3.0V–4.2V range — **provisional, calibrate against real hardware**), POSTs to `${API_BASE}/telemetry` every 5s. For multi-cell packs, scale the empty/full voltages by cell count.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ml.py` not found | Sync the whole folder, not just `main.py` |
| Model not found | `MODEL_PATH` points to the `.onnx` file next to the script |
| Camera won't open | V4L2 path, permissions, same mapping in `main.py` and `ml.py` |
| No video on dashboard | Local `/snapshot` + `/api/detections` first, then relay errors / `API_BASE` |
| No alert | Confirm 3 consecutive person detections, confidence threshold, backend reachability |
| Rover won't move | SignalR connectivity, exact rover ID/group (`ROVER-Q1`), `Bridge` availability |
| Telemetry fails | Disabled by default; validate backend + battery calibration before enabling |

## Known Issues



## Handoff Checklist

- [ ] Sync `main.py`, `ml.py`, model, and `requirements.txt` together
- [ ] Confirm UNO Q runtime + Bridge commands are available
- [ ] Confirm camera paths and 8081/8082 ports
- [ ] Verify model output format matches the detector in use
- [ ] Set `API_BASE`, `ROVER_ID`, `SESSION_ID` for the target deployment
- [ ] Test local detections, snapshot, video feed
- [ ] Test Cloud relay + event/image upload
- [ ] Test SignalR reconnect + motor-stop behavior
- [ ] Calibrate battery constants before enabling telemetry
- [ ] Test autonomy only under supervision, in open space
- [ ] Fix the Dockerfile before relying on CI/CD
