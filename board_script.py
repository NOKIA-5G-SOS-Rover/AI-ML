#!/usr/bin/env python3
"""
Usage:
    python3 board_stream.py \
        --camera-id 0 \
        --laptop-url http://127.0.0.1:5050/frame
"""

import argparse
import threading
import time
import sys

import cv2
import numpy as np
import requests


PERSON_CLASS_ID = 0  # COCO class index for "person"


def log(*args, **kwargs):
    """print() that always flushes immediately, so output shows up in
    real time even when stdout is buffered (e.g. piped, redirected,
    or in some SSH/terminal setups)."""
    print(*args, **kwargs, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Board-side USB webcam -> YOLOv8n (ONNX) person detector -> Flask streamer"
    )
    parser.add_argument("--laptop-url", default="http://127.0.0.1:5000/frame",
                        help="URL of the Flask /frame endpoint on the laptop "
                             "(via SSH tunnel)")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="USB webcam device index, e.g. 0 for /dev/video0 (default 0)")
    parser.add_argument("--reconnect-delay", type=float, default=2.0,
                        help="Seconds to wait before retrying if the webcam "
                             "stream disconnects (default 2.0)")
    parser.add_argument("--onnx-model", default="yolov8n.onnx",
                        help="Path to the exported YOLOv8n ONNX model (default yolov8n.onnx)")
    parser.add_argument("--model-size", type=int, default=160,
                        help="Input size the ONNX model was exported with, e.g. "
                             "160 for imgsz=160 (default 160). Must match the export size.")
    parser.add_argument("--conf-threshold", type=float, default=0.35,
                        help="Minimum confidence to keep a detection (default 0.35)")
    parser.add_argument("--nms-threshold", type=float, default=0.45,
                        help="IoU threshold for non-max suppression (default 0.45)")
    parser.add_argument("--jpeg-quality", type=int, default=80,
                        help="JPEG quality 1-100 for the frame sent over HTTP (default 80)")
    parser.add_argument("--fps-limit", type=float, default=15.0,
                        help="Max frames per second to process/send (default 15). "
                             "Set to 0 for no limit (process as fast as possible).")
    parser.add_argument("--post-timeout", type=float, default=1.0,
                        help="Timeout in seconds for each HTTP POST to the laptop (default 1.0)")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="Seconds between FPS stats log lines (default 5.0)")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Number of CPU threads OpenCV should use for inference "
                             "(default 4, matches the board's 4 cores)")
    parser.add_argument("--skip-grayscale-for-detection", action="store_true",
                        help="Feed the original color frame to the detector instead of "
                             "converting to grayscale first. Often improves both speed "
                             "(skips a conversion) and accuracy. The displayed/sent frame "
                             "is still grayscale either way.")
    return parser.parse_args()


class FrameGrabber:
    """Continuously reads frames from a video source in a background
    thread, always keeping only the most recent frame available.
    This decouples camera read speed from the detection loop, so
    detection always works on the freshest frame rather than processing
    a backlog of stale ones."""

    def __init__(self, camera_id, reconnect_delay):
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def _open(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            return None
        # Minimize internal V4L2 driver buffering to prevent stale frames/latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        log(f"[INFO] Connecting to USB webcam (id={self.camera_id})...")
        self._cap = self._open()
        if self._cap is None:
            log(f"[ERROR] Could not open USB webcam at index {self.camera_id}")
            log("[ERROR] Check that the webcam is plugged in and recognized (e.g., via 'ls /dev/video*').")
            sys.exit(1)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                log("[WARN] Failed to read frame from USB webcam, reconnecting...")
                self._cap.release()
                time.sleep(self.reconnect_delay)
                self._cap = self._open()
                continue

            with self._lock:
                self._latest_frame = frame

    def get_latest(self):
        """Returns the most recent frame, or None if none available yet."""
        with self._lock:
            return self._latest_frame

    def stop(self):
        self._running = False
        if self._cap is not None:
            self._cap.release()


class FrameSender:
    """Sends the latest annotated frame to the laptop in a background
    thread, decoupled from the detection loop. This means a slow or
    spiky POST (common on flaky WiFi) never blocks frame capture or
    inference — the detection loop just drops off whatever's newest
    and moves on, and this thread sends as fast as the network
    allows, always sending the most recent frame rather than queuing
    up a backlog of stale ones.

    Uses a single persistent requests.Session so the underlying TCP
    connection to the tunnel is reused across POSTs instead of being
    torn down and re-established every time, which matters a lot on
    WiFi links with erratic per-connection latency spikes."""

    def __init__(self, url, post_timeout):
        self.url = url
        self.post_timeout = post_timeout
        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._running = False
        self._thread = None
        self.sent_count = 0
        self.dropped_count = 0
        self.failed_count = 0

    def submit(self, jpeg_bytes):
        """Called from the detection loop. Overwrites whatever frame
        was pending (if the sender hasn't gotten to it yet, it's
        dropped in favor of the newer one)."""
        with self._lock:
            if self._latest_frame_pending():
                self.dropped_count += 1
            self._latest_jpeg = jpeg_bytes
        self._new_frame_event.set()

    def _latest_frame_pending(self):
        return self._latest_jpeg is not None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        session = requests.Session()
        while self._running:
            self._new_frame_event.wait(timeout=0.5)
            self._new_frame_event.clear()

            with self._lock:
                jpeg_bytes = self._latest_jpeg
                self._latest_jpeg = None

            if jpeg_bytes is None:
                continue

            try:
                session.post(
                    self.url,
                    data=jpeg_bytes,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=self.post_timeout,
                )
                self.sent_count += 1
            except requests.exceptions.RequestException as e:
                self.failed_count += 1
                log(f"[WARN] Failed to POST frame: {e}")

    def stop(self):
        self._running = False


def load_onnx_model(path, num_threads):
    cv2.setNumThreads(num_threads)
    net = cv2.dnn.readNetFromONNX(path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect_people(net, source_img, model_size, conf_threshold, nms_threshold):
    """Runs YOLOv8n ONNX inference on source_img (any size/aspect —
    blobFromImage handles the resize to model_size x model_size
    internally, avoiding a separate cv2.resize call) and returns a
    list of (x, y, w, h) boxes for person-class detections only, in
    source_img's original pixel coordinate space."""

    blob = cv2.dnn.blobFromImage(
        source_img, scalefactor=1 / 255.0, size=(model_size, model_size),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    output = net.forward()  # shape (1, 84, N)

    # Transpose to (N, 84): columns 0-3 are box [cx, cy, w, h] in
    # model input pixel space, columns 4-83 are per-class scores.
    predictions = output[0].transpose()

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Keep only person-class detections above the confidence threshold.
    keep_mask = (class_ids == PERSON_CLASS_ID) & (confidences >= conf_threshold)
    boxes_raw = predictions[keep_mask, :4]
    scores_kept = confidences[keep_mask]

    if len(boxes_raw) == 0:
        return []

    # Convert (cx, cy, w, h) -> (x, y, w, h) for cv2.dnn.NMSBoxes.
    boxes_xywh = []
    for cx, cy, w, h in boxes_raw:
        x = cx - w / 2
        y = cy - h / 2
        boxes_xywh.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_kept.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    # Scale boxes from model input space (model_size x model_size)
    # back to source_img's actual pixel space.
    scale_x = source_img.shape[1] / model_size
    scale_y = source_img.shape[0] / model_size

    final_boxes = []
    for i in indices:
        x, y, w, h = boxes_xywh[i]
        final_boxes.append((
            int(x * scale_x), int(y * scale_y),
            int(w * scale_x), int(h * scale_y)
        ))
    return final_boxes


def main():
    args = parse_args()

    log(f"[INFO] Loading ONNX model: {args.onnx_model} (threads={args.num_threads})")
    try:
        net = load_onnx_model(args.onnx_model, args.num_threads)
    except cv2.error as e:
        log(f"[ERROR] Failed to load {args.onnx_model}: {e}")
        log("[ERROR] Did you run the export step? See the script docstring for the command.")
        sys.exit(1)

    grabber = FrameGrabber(args.camera_id, args.reconnect_delay)
    grabber.start()

    sender = FrameSender(args.laptop_url, args.post_timeout)
    sender.start()

    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0

    log(f"[INFO] Streaming to {args.laptop_url}")
    log("[INFO] Press Ctrl+C to stop.")

    last_processed_frame_id = None
    frames_processed = 0
    stats_window_start = time.time()

    try:
        while True:
            loop_start = time.time()

            frame = grabber.get_latest()
            if frame is None:
                time.sleep(0.05)
                continue

            # Skip re-processing the exact same frame object if the
            # grabber hasn't produced a new one yet.
            frame_id = id(frame)
            if frame_id == last_processed_frame_id:
                time.sleep(0.01)
                continue
            last_processed_frame_id = frame_id

            # Downscale to a square matching the model's input size,
            # since that's what we'll run inference on directly.
            small = cv2.resize(frame, (args.model_size, args.model_size),
                                interpolation=cv2.INTER_AREA)

            # Grayscale
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # YOLO expects 3-channel input; convert grayscale back to
            # 3-channel (all channels equal) purely to satisfy the
            # model's expected input shape. Visual content stays
            # grayscale.
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            boxes = detect_people(
                net, gray_3ch, args.model_size,
                args.conf_threshold, args.nms_threshold
            )

            annotated = gray_3ch.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                annotated, f"People: {len(boxes)}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

            frames_processed += 1

            # Encode as JPEG
            ok, jpeg_buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            )
            if not ok:
                log("[WARN] JPEG encoding failed, skipping frame")
                continue

            # Hand off to the background sender thread. This never
            # blocks on the network — if the sender is still busy
            # with the previous frame (e.g. a slow POST due to WiFi
            # latency spikes), this new frame just replaces whatever
            # was pending, and detection keeps running at full speed.
            sender.submit(jpeg_buf.tobytes())

            # Periodic FPS stats.
            now = time.time()
            elapsed_window = now - stats_window_start
            if elapsed_window >= args.stats_interval:
                fps = frames_processed / elapsed_window
                sent_fps = sender.sent_count / elapsed_window
                log(f"[STATS] processed {fps:.1f} fps, sent {sent_fps:.1f} fps, "
                    f"dropped {sender.dropped_count}, failed {sender.failed_count}")
                frames_processed = 0
                sender.sent_count = 0
                sender.dropped_count = 0
                sender.failed_count = 0
                stats_window_start = now

            # Throttle to fps_limit.
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log("\n[INFO] Stopping stream.")
    finally:
        grabber.stop()
        sender.stop()


if __name__ == "__main__":
    main()
