#!/usr/bin/env python3
"""
Dual-camera YOLO person detection + side-by-side MJPEG stream.

Example:
    python test_2_cameras.py \
        --camera-left 2 \
        --camera-right 4 \
        --host 0.0.0.0 \
        --port 8080

Open in a browser:
    http://<DEVICE-IP>:8080
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
log = logging.getLogger("dual-person-detector")

PERSON_CLASS_ID = 0  # COCO class ID for "person"


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for notifying another Linux process."""

    enabled: bool
    target_pid: Optional[int]
    signal_number: int
    signal_name: str
    detection_env: str
    status_file: Optional[str]


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class CameraData:
    index: int
    name: str
    latest_frame: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)
    fps_capture: float = 0.0
    read_failures: int = 0


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = True

    left: CameraData = field(default_factory=lambda: CameraData(0, "LEFT"))
    right: CameraData = field(default_factory=lambda: CameraData(1, "RIGHT"))

    display_frame: Optional[np.ndarray] = None
    fps_inference: float = 0.0
    fps_stream: float = 0.0

    person_present: bool = False
    person_count: int = 0
    signal_count: int = 0
    last_signal_error: Optional[str] = None
    signal_target_pid: Optional[int] = None


def resolve_camera_backend(name: str) -> int:
    """Resolve an OpenCV capture backend from a portable command-line name."""
    normalized = name.lower()
    if normalized == "auto":
        # DirectShow is generally more reliable than MSMF for two webcams on Windows.
        return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2

    backends = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }
    return backends[normalized]


def open_camera(
    device: int,
    width: int,
    height: int,
    fps: int,
    camera_name: str,
    backend_name: str,
    fourcc: str,
) -> cv2.VideoCapture:
    backend = resolve_camera_backend(backend_name)
    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {camera_name} camera device index {device} "
            f"with backend {backend_name}"
        )

    # MJPG significantly reduces USB bandwidth when two UVC cameras are connected.
    if fourcc:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*fourcc.upper()),
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Confirm that the camera can return a frame before starting worker threads.
    first_frame_ok = False
    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            first_frame_ok = True
            break
        time.sleep(0.05)

    if not first_frame_ok:
        cap.release()
        raise RuntimeError(
            f"{camera_name} camera index {device} opened but returned no frames. "
            "Try another camera index, backend, USB port, or lower resolution."
        )

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(
        chr((actual_fourcc_value >> (8 * i)) & 0xFF) for i in range(4)
    ).strip("\x00")
    log.info(
        "%s camera %d ready: %dx%d @ %.1f FPS, backend=%s, FOURCC=%s",
        camera_name,
        device,
        actual_w,
        actual_h,
        actual_fps,
        backend_name,
        actual_fourcc or "unknown",
    )
    return cap


def capture_worker(
    cap: cv2.VideoCapture,
    state: AppState,
    camera_side: str,
) -> None:
    """Continuously capture frames from one camera."""
    count = 0
    t0 = time.perf_counter()

    while state.running:
        ok, frame = cap.read()
        if not ok:
            with state.lock:
                camera = getattr(state, camera_side)
                camera.read_failures += 1
            log.warning("%s camera read failed, retrying...", camera_side.upper())
            time.sleep(0.05)
            continue

        count += 1
        now = time.perf_counter()

        with state.lock:
            camera = getattr(state, camera_side)
            camera.latest_frame = frame

            if now - t0 >= 1.0:
                camera.fps_capture = count / (now - t0)
                count = 0
                t0 = now


def write_detection_status(
    status_file: Optional[str],
    detection_env: str,
    person_present: bool,
    person_count: int,
) -> None:
    """Update this process's environment and an optional shared status file."""
    value = "1" if person_present else "0"
    os.environ[detection_env] = value
    os.environ[f"{detection_env}_COUNT"] = str(person_count)

    if not status_file:
        return

    status_path = os.path.abspath(status_file)
    status_dir = os.path.dirname(status_path)
    os.makedirs(status_dir, exist_ok=True)

    temporary_path = f"{status_path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(f"{detection_env}={value}\n")
        handle.write(f"{detection_env}_COUNT={person_count}\n")
        handle.write(f"PERSON_DETECTOR_PID={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, status_path)


def notify_detection_change(
    state: AppState,
    config: SignalConfig,
    person_count: int,
) -> None:
    """
    Update the aggregate detection state.

    A signal is sent only when the aggregate state changes from no person in
    either camera to at least one person in one or both cameras.
    """
    person_present = person_count > 0

    with state.lock:
        previous_person_present = state.person_present
        previous_person_count = state.person_count
        state.person_present = person_present
        state.person_count = person_count

    if (
        person_present != previous_person_present
        or person_count != previous_person_count
    ):
        try:
            write_detection_status(
                status_file=config.status_file,
                detection_env=config.detection_env,
                person_present=person_present,
                person_count=person_count,
            )
        except OSError as exc:
            log.error("Could not update detection status: %s", exc)

    # Rising edge only: no person in either camera -> person in at least one.
    if not person_present or previous_person_present:
        return

    if not config.enabled:
        return

    if config.target_pid is None:
        log.warning(
            "Person detected, but no target PID was configured; %s was not sent.",
            config.signal_name,
        )
        return

    try:
        os.kill(config.target_pid, config.signal_number)
    except ProcessLookupError:
        message = f"Target PID {config.target_pid} does not exist"
        with state.lock:
            state.last_signal_error = message
        log.error("%s; could not send %s.", message, config.signal_name)
    except PermissionError:
        message = f"Permission denied for target PID {config.target_pid}"
        with state.lock:
            state.last_signal_error = message
        log.error("%s; could not send %s.", message, config.signal_name)
    except OSError as exc:
        message = str(exc)
        with state.lock:
            state.last_signal_error = message
        log.error(
            "Could not send %s to PID %d: %s",
            config.signal_name,
            config.target_pid,
            exc,
        )
    else:
        with state.lock:
            state.signal_count += 1
            state.last_signal_error = None
        log.info(
            "Person detected: sent %s to PID %d.",
            config.signal_name,
            config.target_pid,
        )


def result_to_detections(result) -> List[Detection]:
    """Convert one Ultralytics result into Detection objects."""
    detections: List[Detection] = []

    if result is None or result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        detections.append(Detection(x1, y1, x2, y2, score))

    return detections


def inference_worker(
    model: YOLO,
    state: AppState,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    signal_config: SignalConfig,
) -> None:
    """Run one YOLO batch containing the newest frame from each camera."""
    completed_images = 0
    t0 = time.perf_counter()

    while state.running:
        loop_start = time.perf_counter()

        with state.lock:
            left_frame = (
                None
                if state.left.latest_frame is None
                else state.left.latest_frame.copy()
            )
            right_frame = (
                None
                if state.right.latest_frame is None
                else state.right.latest_frame.copy()
            )

        frames: List[Tuple[str, np.ndarray]] = []

        if left_frame is not None:
            frames.append(("left", left_frame))

        if right_frame is not None:
            frames.append(("right", right_frame))

        if not frames:
            time.sleep(0.01)
            continue

        new_detections: Dict[str, List[Detection]] = {}

        # Process each camera separately because the ONNX model accepts batch size 1.
        for side, frame in frames:
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                device=device,
                stream=False,
            )

            result = results[0] if results else None
            new_detections[side] = result_to_detections(result)
            completed_images += 1
        now = time.perf_counter()

        with state.lock:
            if "left" in new_detections:
                state.left.detections = new_detections["left"]
            if "right" in new_detections:
                state.right.detections = new_detections["right"]

            if now - t0 >= 1.0:
                # This is processed camera images per second. Divide by two to
                # estimate complete left+right pairs per second.
                state.fps_inference = completed_images / (now - t0)
                completed_images = 0
                t0 = now

            total_persons = (
                len(state.left.detections) + len(state.right.detections)
            )

        notify_detection_change(
            state=state,
            config=signal_config,
            person_count=total_persons,
        )

        elapsed = time.perf_counter() - loop_start
        if elapsed < 0.005:
            time.sleep(0.005)


def draw_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    camera_name: str,
    camera_index: int,
    fps_capture: float,
    fps_inference_images: float,
) -> np.ndarray:
    out = frame.copy()

    for det in detections:
        cv2.rectangle(
            out,
            (det.x1, det.y1),
            (det.x2, det.y2),
            (0, 255, 0),
            2,
        )
        label = f"person {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_y = max(th + 8, det.y1)
        cv2.rectangle(
            out,
            (det.x1, label_y - th - 8),
            (det.x1 + tw + 4, label_y),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            out,
            label,
            (det.x1 + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

    title = f"{camera_name} camera {camera_index}"
    hud = (
        f"capture {fps_capture:.1f} fps | "
        f"infer {fps_inference_images:.1f} img/s | "
        f"persons {len(detections)}"
    )

    cv2.putText(
        out,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        out,
        hud,
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 255),
        2,
    )
    return out


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Resize while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if height == target_height:
        return frame

    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height))


def waiting_frame(
    width: int,
    height: int,
    label: str,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"Waiting for {label} camera...",
        (25, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    return frame


def render_worker(state: AppState, fallback_width: int, fallback_height: int) -> None:
    """Draw both camera overlays and combine them side-by-side."""
    while state.running:
        with state.lock:
            left_frame = (
                None
                if state.left.latest_frame is None
                else state.left.latest_frame.copy()
            )
            right_frame = (
                None
                if state.right.latest_frame is None
                else state.right.latest_frame.copy()
            )
            left_detections = list(state.left.detections)
            right_detections = list(state.right.detections)
            left_fps = state.left.fps_capture
            right_fps = state.right.fps_capture
            inference_fps = state.fps_inference
            left_index = state.left.index
            right_index = state.right.index

        if left_frame is None:
            left_frame = waiting_frame(fallback_width, fallback_height, "LEFT")
        if right_frame is None:
            right_frame = waiting_frame(fallback_width, fallback_height, "RIGHT")

        left_display = draw_overlay(
            left_frame,
            left_detections,
            "LEFT",
            left_index,
            left_fps,
            inference_fps,
        )
        right_display = draw_overlay(
            right_frame,
            right_detections,
            "RIGHT",
            right_index,
            right_fps,
            inference_fps,
        )

        common_height = min(left_display.shape[0], right_display.shape[0])
        left_display = resize_to_height(left_display, common_height)
        right_display = resize_to_height(right_display, common_height)

        separator = np.zeros((common_height, 6, 3), dtype=np.uint8)
        combined = cv2.hconcat([left_display, separator, right_display])

        with state.lock:
            state.display_frame = combined

        time.sleep(1.0 / 30.0)


def mjpeg_stream(state: AppState, jpeg_quality: int):
    """multipart/x-mixed-replace MJPEG generator."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    count = 0
    t0 = time.perf_counter()

    while state.running:
        with state.lock:
            frame = (
                None
                if state.display_frame is None
                else state.display_frame.copy()
            )

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
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        time.sleep(1.0 / 30.0)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dual Camera Person Detector</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #111;
      color: #eee;
      margin: 0;
      padding: 20px;
    }
    h1 { margin-top: 0; }
    .panel { max-width: 1500px; margin: 0 auto; }
    img {
      width: 100%;
      border: 2px solid #333;
      border-radius: 8px;
      background: #000;
    }
    .stats { margin-top: 12px; font-size: 18px; line-height: 1.5; }
    .ok { color: #7CFC00; }
    .warn { color: #FFD700; }
  </style>
</head>
<body>
  <div class="panel">
    <h1>Dual-Camera Person Detection</h1>
    <img id="video" src="/video_feed" alt="Dual MJPEG stream" />
    <div class="stats" id="stats">Connecting...</div>
  </div>

  <script>
    async function pollStats() {
      try {
        const res = await fetch("/api/detections", { cache: "no-store" });
        const data = await res.json();
        const el = document.getElementById("stats");
        el.innerHTML =
          `LEFT camera ${data.left.camera}: ` +
          `<span class="ok">${data.left.fps_capture.toFixed(1)} capture FPS</span>, ` +
          `<span class="warn">${data.left.person_count} person(s)</span><br>` +
          `RIGHT camera ${data.right.camera}: ` +
          `<span class="ok">${data.right.fps_capture.toFixed(1)} capture FPS</span>, ` +
          `<span class="warn">${data.right.person_count} person(s)</span><br>` +
          `Inference: <span class="ok">${data.fps_inference.toFixed(1)} images/s</span> | ` +
          `Stream: <span class="ok">${data.fps_stream.toFixed(1)} FPS</span> | ` +
          `Aggregate detections: <span class="warn">${data.person_count}</span>`;
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


def detection_to_dict(detection: Detection) -> dict:
    return {
        "x1": detection.x1,
        "y1": detection.y1,
        "x2": detection.x2,
        "y2": detection.y2,
        "confidence": detection.confidence,
    }


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
            left_persons = [detection_to_dict(d) for d in state.left.detections]
            right_persons = [detection_to_dict(d) for d in state.right.detections]

            payload = {
                "fps_inference": state.fps_inference,
                "fps_stream": state.fps_stream,
                "detector_pid": os.getpid(),
                "signal_target_pid": state.signal_target_pid,
                "person_present": state.person_present,
                "person_count": state.person_count,
                "signals_sent": state.signal_count,
                "last_signal_error": state.last_signal_error,
                "left": {
                    "camera": state.left.index,
                    "fps_capture": state.left.fps_capture,
                    "read_failures": state.left.read_failures,
                    "person_count": len(left_persons),
                    "persons": left_persons,
                },
                "right": {
                    "camera": state.right.index,
                    "fps_capture": state.right.fps_capture,
                    "read_failures": state.right.read_failures,
                    "person_count": len(right_persons),
                    "persons": right_persons,
                },
                # Compatibility-friendly combined list. Coordinates are local
                # to the source camera image, not to the combined browser image.
                "persons": [
                    {"camera": "left", **person} for person in left_persons
                ]
                + [
                    {"camera": "right", **person} for person in right_persons
                ],
            }

        return jsonify(payload)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-camera YOLO person detector + side-by-side MJPEG stream"
    )
    parser.add_argument(
        "--camera-left",
        type=int,
        default=2,
        help="Left camera index (default: 2)",
    )
    parser.add_argument(
        "--camera-right",
        type=int,
        default=4,
        help="Right camera index (default: 4)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "any", "dshow", "msmf", "v4l2"),
        default="auto",
        help="OpenCV camera backend; auto uses DSHOW on Windows and V4L2 on Linux",
    )
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="Camera capture FOURCC. MJPG is recommended for two USB cameras.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n_416.onnx",
        help="YOLO model path",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="YOLO input size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu or 0",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=70)

    parser.add_argument(
        "--signal-pid",
        type=int,
        default=None,
        help="PID that receives SIGUSR when a person first appears",
    )
    parser.add_argument(
        "--signal-pid-env",
        default="ROVER_CONTROL_PID",
        help=(
            "Environment variable containing the target PID when "
            "--signal-pid is not supplied"
        ),
    )
    parser.add_argument(
        "--signal",
        choices=("SIGUSR1", "SIGUSR2"),
        default="SIGUSR1",
        help="User signal sent to the target process",
    )
    parser.add_argument(
        "--detection-env",
        default="PERSON_DETECTED",
        help="Process-local environment variable set to 1 or 0",
    )
    parser.add_argument(
        "--status-file",
        default=(
            f"/tmp/person_detector_status_{os.getuid()}.env"
            if os.name == "posix"
            else ""
        ),
        help=(
            "Shared status file for the process receiving SIGUSR. "
            "It is enabled by default on Linux and disabled by default "
            "on Windows. Pass an empty string to disable it."
        ),
    )
    return parser.parse_args()


def resolve_target_pid(args: argparse.Namespace) -> Optional[int]:
    if args.signal_pid is not None:
        if args.signal_pid <= 0:
            raise ValueError("--signal-pid must be a positive integer")
        return args.signal_pid

    raw_pid = os.environ.get(args.signal_pid_env)
    if raw_pid is None or not raw_pid.strip():
        return None

    try:
        target_pid = int(raw_pid)
    except ValueError as exc:
        raise ValueError(
            f"{args.signal_pid_env} must contain an integer PID, got {raw_pid!r}"
        ) from exc

    if target_pid <= 0:
        raise ValueError(f"{args.signal_pid_env} must contain a positive PID")

    return target_pid


def make_signal_config(args: argparse.Namespace) -> SignalConfig:
    target_pid = resolve_target_pid(args)

    # SIGUSR1/SIGUSR2 are POSIX signals and are not available on Windows.
    if os.name != "posix":
        if target_pid is not None:
            log.warning("Ignoring --signal-pid because SIGUSR is unavailable on Windows.")
        return SignalConfig(
            enabled=False,
            target_pid=None,
            signal_number=0,
            signal_name="disabled",
            detection_env=args.detection_env,
            status_file=None,
        )

    signal_number = getattr(signal, args.signal)
    return SignalConfig(
        enabled=True,
        target_pid=target_pid,
        signal_number=signal_number,
        signal_name=args.signal,
        detection_env=args.detection_env,
        status_file=args.status_file or None,
    )


def main() -> None:
    args = parse_args()

    if args.camera_left == args.camera_right:
        raise ValueError("Left and right camera indices must be different")

    signal_config = make_signal_config(args)

    detector_pid = os.getpid()
    os.environ["PERSON_DETECTOR_PID"] = str(detector_pid)
    try:
        write_detection_status(
            status_file=signal_config.status_file,
            detection_env=signal_config.detection_env,
            person_present=False,
            person_count=0,
        )
    except OSError as exc:
        log.error("Could not initialize detection status file: %s", exc)

    log.info("Detector PID: %d", detector_pid)
    if not signal_config.enabled:
        log.info("POSIX SIGUSR notification is disabled on this platform.")
    elif signal_config.target_pid is None:
        log.warning(
            "No signal target configured. Use --signal-pid PID or set %s.",
            args.signal_pid_env,
        )
    else:
        log.info(
            "A new person detection in either camera will send %s to PID %d.",
            signal_config.signal_name,
            signal_config.target_pid,
        )

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}. Pass the full path with --model."
        )

    log.info("Loading YOLO model: %s", model_path)
    model = YOLO(str(model_path))

    left_cap = open_camera(
        args.camera_left,
        args.width,
        args.height,
        args.fps,
        "LEFT",
        args.camera_backend,
        args.fourcc,
    )

    try:
        right_cap = open_camera(
            args.camera_right,
            args.width,
            args.height,
            args.fps,
            "RIGHT",
            args.camera_backend,
            args.fourcc,
        )
    except Exception:
        left_cap.release()
        raise

    state = AppState(
        left=CameraData(args.camera_left, "LEFT"),
        right=CameraData(args.camera_right, "RIGHT"),
        signal_target_pid=signal_config.target_pid,
    )

    workers = [
        threading.Thread(
            target=capture_worker,
            args=(left_cap, state, "left"),
            name="capture-left",
            daemon=True,
        ),
        threading.Thread(
            target=capture_worker,
            args=(right_cap, state, "right"),
            name="capture-right",
            daemon=True,
        ),
        threading.Thread(
            target=inference_worker,
            args=(
                model,
                state,
                args.imgsz,
                args.conf,
                args.iou,
                args.device,
                signal_config,
            ),
            name="inference",
            daemon=True,
        ),
        threading.Thread(
            target=render_worker,
            args=(state, args.width, args.height),
            name="render",
            daemon=True,
        ),
    ]

    for worker in workers:
        worker.start()
        log.info("Started thread: %s", worker.name)

    app = create_flask_app(state, jpeg_quality=args.jpeg_quality)
    local_host = "127.0.0.1" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    log.info("Open on this computer: http://%s:%d", local_host, args.port)

    try:
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            use_reloader=False,
        )
    finally:
        state.running = False

        # Let capture/inference threads leave their loops before releasing cameras.
        # Releasing a VideoCapture while another thread is inside cap.read() can
        # produce a native OpenCV segmentation fault on shutdown.
        for worker in workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                log.warning("Thread %s did not stop within 2 seconds.", worker.name)

        left_cap.release()
        right_cap.release()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
        log.info("Stopped.")


if __name__ == "__main__":
    main()