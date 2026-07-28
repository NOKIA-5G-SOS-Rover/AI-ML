#!/usr/bin/env python3
"""
YOLO person detection + MJPEG stream for Arduino UNO Q.

On UNO Q:
  pip install -r requirements.txt
  python person_detector_server.py --host 0.0.0.0 --port 8080

On PC browser:
  http://<UNO-Q-IP>:8080
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Missing ultralytics. Run: pip install ultralytics") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("person-detector")

PERSON_CLASS_ID = 0  # COCO "person"


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = True

    latest_frame: Optional[np.ndarray] = None
    display_frame: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)

    fps_capture: float = 0.0
    fps_inference: float = 0.0
    fps_stream: float = 0.0


def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device index {device}")

    # Keep latency low: don't accumulate old frames in driver buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Camera ready: %dx%d @ %.1f FPS", actual_w, actual_h, actual_fps)
    return cap


def capture_worker(cap: cv2.VideoCapture, state: AppState) -> None:
    count = 0
    t0 = time.perf_counter()

    while state.running:
        ok, frame = cap.read()
        if not ok:
            log.warning("Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        count += 1
        now = time.perf_counter()
        if now - t0 >= 1.0:
            with state.lock:
                state.fps_capture = count / (now - t0)
            count = 0
            t0 = now

        with state.lock:
            state.latest_frame = frame


def inference_worker(
    model: YOLO,
    state: AppState,
    imgsz: int,
    conf: float,
    device: str,
) -> None:
    count = 0
    t0 = time.perf_counter()

    while state.running:
        loop_start = time.perf_counter()

        with state.lock:
            frame = state.latest_frame
            if frame is None:
                frame_copy = None
            else:
                frame_copy = frame.copy()

        if frame_copy is None:
            time.sleep(0.01)
            continue

        results = model.predict(
            source=frame_copy,
            imgsz=imgsz,
            conf=conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            device=device,
        )

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0])
                detections.append(Detection(x1, y1, x2, y2, score))

        count += 1
        now = time.perf_counter()
        if now - t0 >= 1.0:
            with state.lock:
                state.fps_inference = count / (now - t0)
            count = 0
            t0 = now

        with state.lock:
            state.detections = detections

        # Optional tiny sleep so inference thread doesn't spin at 100% CPU when model is very fast
        elapsed = time.perf_counter() - loop_start
        if elapsed < 0.005:
            time.sleep(0.005)


def draw_overlay(frame: np.ndarray, detections: List[Detection], hud: str) -> np.ndarray:
    out = frame.copy()

    for det in detections:
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"person {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y = max(th + 8, det.y1)
        cv2.rectangle(out, (det.x1, y - th - 8), (det.x1 + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(out, label, (det.x1 + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def render_worker(state: AppState) -> None:
    """Compose smooth video using latest camera frame + last known detections."""
    while state.running:
        with state.lock:
            frame = state.latest_frame
            detections = list(state.detections)
            fps_cap = state.fps_capture
            fps_inf = state.fps_inference

        if frame is not None:
            hud = f"capture {fps_cap:.1f} fps | infer {fps_inf:.1f} fps | persons {len(detections)}"
            display = draw_overlay(frame, detections, hud)
            with state.lock:
                state.display_frame = display

        time.sleep(1.0 / 30.0)  # target ~30 FPS UI stream


def mjpeg_stream(state: AppState, jpeg_quality: int):
    """multipart/x-mixed-replace MJPEG generator."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    count = 0
    t0 = time.perf_counter()

    while state.running:
        with state.lock:
            frame = state.display_frame
            if frame is None:
                frame = state.latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue

        count += 1
        now = time.perf_counter()
        if now - t0 >= 1.0:
            with state.lock:
                state.fps_stream = count / (now - t0)
            count = 0
            t0 = now

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        time.sleep(1.0 / 30.0)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>UNO Q Person Detector</title>
  <style>
    body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
    h1 { margin-top: 0; }
    .panel { max-width: 980px; margin: 0 auto; }
    img { width: 100%; max-width: 960px; border: 2px solid #333; border-radius: 8px; background: #000; }
    .stats { margin-top: 12px; font-size: 18px; }
    .ok { color: #7CFC00; }
    .warn { color: #FFD700; }
  </style>
</head>
<body>
  <div class="panel">
    <h1>Person Detection Stream</h1>
    <img id="video" src="/video_feed" alt="MJPEG stream" />
    <div class="stats" id="stats">Connecting...</div>
  </div>

  <script>
    async function pollStats() {
      try {
        const res = await fetch("/api/detections", { cache: "no-store" });
        const data = await res.json();
        const el = document.getElementById("stats");
        el.innerHTML =
          `Capture: <span class="ok">${data.fps_capture.toFixed(1)} FPS</span> | ` +
          `Inference: <span class="ok">${data.fps_inference.toFixed(1)} FPS</span> | ` +
          `Stream: <span class="ok">${data.fps_stream.toFixed(1)} FPS</span> | ` +
          `Persons: <span class="warn">${data.persons.length}</span>`;
      } catch (err) {
        document.getElementById("stats").textContent = "Stats unavailable";
      }
    }
    setInterval(pollStats, 500);
    pollStats();
  </script>
</body>
</html>
"""


def create_flask_app(state: AppState, jpeg_quality: int) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.get("/video_feed")
    def video_feed():
        return Response(
            mjpeg_stream(state, jpeg_quality=jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/detections")
    def detections_api():
        with state.lock:
            payload = {
                "fps_capture": state.fps_capture,
                "fps_inference": state.fps_inference,
                "fps_stream": state.fps_stream,
                "persons": [
                    {
                        "x1": d.x1,
                        "y1": d.y1,
                        "x2": d.x2,
                        "y2": d.y2,
                        "confidence": d.confidence,
                    }
                    for d in state.detections
                ],
            }
        return jsonify(payload)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO person detector + MJPEG stream")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 camera index, usually 0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", default="yolov8n.pt", help="Use yolov8n.pt for speed")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO input size (320 is a good balance)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="cpu is safest on UNO Q")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading YOLO model: %s", args.model)
    model = YOLO(args.model)

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    state = AppState()

    workers = [
        threading.Thread(target=capture_worker, args=(cap, state), name="capture", daemon=True),
        threading.Thread(
            target=inference_worker,
            args=(model, state, args.imgsz, args.conf, args.device),
            name="inference",
            daemon=True,
        ),
        threading.Thread(target=render_worker, args=(state,), name="render", daemon=True),
    ]

    for worker in workers:
        worker.start()
        log.info("Started thread: %s", worker.name)

    app = create_flask_app(state, jpeg_quality=args.jpeg_quality)
    log.info("Open on PC: http://<UNO-Q-IP>:%d", args.port)

    try:
        # threaded=True allows multiple browser clients for MJPEG + /api/detections
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        state.running = False
        cap.release()
        log.info("Stopped.")


if __name__ == "__main__":
    main()