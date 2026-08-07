#!/usr/bin/env python3
"""
Person detection pe 2 camere USB (Arduino Uno Q / Debian) folosind un model
YOLO26 in format ONNX (onnxruntime), cu trimitere in timp real a frame-urilor
procesate + metadate JSON catre un server REST, folosind multipart/form-data.

Fiecare camera ruleaza pe thread-ul ei propriu (captura -> inferenta -> POST),
independent de cealalta, ca sa nu se blocheze una pe alta.

=== CE TREBUIE SA MODIFICI TU ===
  1. SERVER_URL         -> IP-ul / URL-ul serverului tau
  2. MODEL_PATH         -> calea catre fisierul .onnx
  3. CAMERAS            -> device-urile camerelor (deja puse /dev/video0 si /dev/video2)
  4. INPUT_WIDTH, INPUT_HEIGHT, CLASS_ID_PERSON, CONF_THRES -> daca modelul tau difera

=== FORMAT MODEL PRESUPUS (YOLO26, export standard Ultralytics) ===
YOLO26 e NMS-free by design -- Ultralytics exporta in ONNX cu NMS deja integrat
in graph ("end-to-end"), deci output-ul e deja filtrat, NU mai trebuie NMS manual.
  - input:  (1, 3, H, W) float32, normalizat [0,1], RGB. Aici H=480, W=640
            (modelul e "img480", non-patrat, deci NU se face letterbox cu padding,
            ci resize direct pe (W,H) -- vezi preprocess()).
  - output: (1, num_detections, 6) = [x1, y1, x2, y2, confidence, class_id]
            deja in coordonate ale input-ului resized (fara padding de scazut).
Daca modelul tau difera (alt shape de output), vezi functia `postprocess()` mai jos.
"""

import os
import io
import cv2
import time
import json
import queue
import signal
import logging
import threading
import numpy as np
import onnxruntime as ort
import requests

# ============================== CONFIG ====================================

# --- Server ---
SERVER_URL = "http://92.87.91.146:5000"   # <-- PUNE AICI IP-ul tau
REQUEST_TIMEOUT = 5          # secunde, timeout per request HTTP
HTTP_RETRIES = 1             # cate reincercari la fail (0 = fara retry)

# --- Model ---
MODEL_PATH = "yolo26n_img480_int8.onnx"   # <-- calea catre modelul tau .onnx
INPUT_WIDTH = 640              # latimea pe care o asteapta modelul (yolo26n_img480 -> 640x480)
INPUT_HEIGHT = 480             # inaltimea pe care o asteapta modelul
CONF_THRES = 0.45             # prag de confidenta
CLASS_ID_PERSON = 0           # in COCO, "person" e clasa 0. Schimba daca modelul tau difera.
# Nota: YOLO26 are NMS integrat in graph-ul ONNX (export standard Ultralytics),
# deci nu mai exista un IOU_THRES de configurat aici -- NMS-ul e deja aplicat de model.

# --- Camere ---
# device_path -> ce apare in /dev; camera_id -> folosit ca identificator in JSON/nume fisier
CAMERAS = [
    {"camera_id": "cam0", "device": "/dev/video0"},
    {"camera_id": "cam1", "device": "/dev/video2"},
]
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30

JPEG_QUALITY = 85              # calitate compresie JPEG (0-100) pentru frame-ul trimis

# Downscale explicit: indiferent la ce rezolutie da camera frame-uri (ex: 1080p),
# fortam resize la aceasta rezolutie inainte de inferenta/desenare/trimitere.
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
log = logging.getLogger(__name__)

# Flag global pentru shutdown gracios (Ctrl+C)
stop_event = threading.Event()


# ============================== MODEL WRAPPER ==============================

class Yolo26OnnxDetector:
    """
    Wrapper simplu si thread-safe pentru un model YOLO26 exportat in ONNX
    (export standard Ultralytics, cu NMS end-to-end integrat in graph).

    (O sesiune onnxruntime per thread e mai sigura, dar sesiunile onnxruntime
    suporta de fapt apeluri concurente de pe mai multe thread-uri pe aceeasi
    sesiune -- deci folosim UNA singura, partajata, cu un lock pentru siguranta.)
    """

    def __init__(self, model_path: str, input_width: int = 640, input_height: int = 480):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Nu gasesc modelul ONNX la: {model_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Foloseste CPU by default; daca ai un provider de accelerare disponibil
        # pe placa (ex: CUDA, OpenVINO, etc.), il poti adauga aici in lista.
        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        log.info(f"Model YOLO26 incarcat cu providers: {self.session.get_providers()}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_width = input_width
        self.input_height = input_height

        # Lock pentru siguranta -- unele build-uri onnxruntime nu sunt 100% thread-safe
        # pentru inferenta concurenta pe aceeasi sesiune. Cu 2 camere e ieftin sa serializam.
        self._lock = threading.Lock()

    def preprocess(self, frame_bgr: np.ndarray):
        """
        Resize direct (fara letterbox/padding) la (input_width, input_height),
        pentru ca modelul YOLO26 e exportat non-patrat (640x480), la fel cum
        vine si frame-ul de la camera dupa downscale -- deci scale-ul pe X si Y
        poate diferi usor daca rezolutia sursa are alt aspect ratio.
        """
        h0, w0 = frame_bgr.shape[:2]
        target_w, target_h = self.input_width, self.input_height

        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1,3,H,W)

        # scale separat pe X si Y, ca sa deconvertim corect boxele inapoi la rezolutia originala
        meta = {
            "scale_x": target_w / w0,
            "scale_y": target_h / h0,
            "orig_w": w0,
            "orig_h": h0,
        }
        return tensor, meta

    def infer(self, tensor: np.ndarray):
        with self._lock:
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return outputs

    def postprocess(self, outputs, meta, conf_thres=0.45, class_id_person=0):
        """
        Output YOLO26 (export Ultralytics end-to-end, NMS deja integrat in graph):
          shape (1, num_detections, 6) = [x1, y1, x2, y2, confidence, class_id]
        Deja filtrat/final -- NU se mai aplica NMS manual aici.

        Returneaza lista de detectii doar pentru clasa "person":
          [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        in coordonate ale imaginii ORIGINALE (nu ale input-ului resized).
        """
        pred = outputs[0]  # (1, num_detections, 6)
        pred = np.squeeze(pred, axis=0)  # (num_detections, 6)

        if pred.size == 0:
            return []

        boxes_xyxy = pred[:, :4]
        confidences = pred[:, 4]
        class_ids = pred[:, 5]

        mask = (class_ids.astype(np.int32) == class_id_person) & (confidences >= conf_thres)
        boxes_xyxy = boxes_xyxy[mask]
        confidences = confidences[mask]

        if len(boxes_xyxy) == 0:
            return []

        # Deconverteste din coordonate ale input-ului resized -> coordonate imagine originala.
        # Fara padding de scazut (nu am facut letterbox), doar impartim la scale pe fiecare axa.
        scale_x, scale_y = meta["scale_x"], meta["scale_y"]
        orig_w, orig_h = meta["orig_w"], meta["orig_h"]

        boxes_xyxy = boxes_xyxy.copy()
        boxes_xyxy[:, [0, 2]] /= scale_x
        boxes_xyxy[:, [1, 3]] /= scale_y

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        detections = []
        for box, conf in zip(boxes_xyxy, confidences):
            detections.append({
                "bbox": [round(float(v), 1) for v in box],  # [x1,y1,x2,y2]
                "confidence": round(float(conf), 4),
            })
        return detections

    def detect(self, frame_bgr: np.ndarray, conf_thres=None, class_id_person=None):
        tensor, meta = self.preprocess(frame_bgr)
        outputs = self.infer(tensor)
        return self.postprocess(
            outputs, meta,
            conf_thres=conf_thres if conf_thres is not None else CONF_THRES,
            class_id_person=class_id_person if class_id_person is not None else CLASS_ID_PERSON,
        )


def draw_detections(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Deseneaza bounding box-uri + confidence pe frame. Returneaza frame-ul modificat."""
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


# ============================== POST / UPLOAD ===============================

def post_frame(camera_id: str, frame_bgr: np.ndarray, detections: list, timestamp: float):
    """
    Trimite multipart/form-data:
      - field "image": JPEG binar (frame procesat, cu bounding boxes desenate)
      - field "metadata": JSON string cu detaliile detectiei
    """
    ok, jpeg_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        log.warning(f"[{camera_id}] Nu am putut encoda frame-ul in JPEG, skip.")
        return

    metadata = {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "person_count": len(detections),
        "detections": detections,  # listă de {"bbox": [x1,y1,x2,y2], "confidence": float}
    }

    files = {
        "image": (f"{camera_id}_{int(timestamp * 1000)}.jpg", jpeg_buf.tobytes(), "image/jpeg"),
    }
    data = {
        "metadata": json.dumps(metadata),
    }

    attempt = 0
    while attempt <= HTTP_RETRIES:
        try:
            resp = requests.post(SERVER_URL+"/api/video", files=files, data=data, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                log.warning(f"[{camera_id}] Server a raspuns cu status {resp.status_code}: {resp.text[:200]}")
            else:
                log.debug(f"[{camera_id}] Trimis OK, {len(detections)} persoane detectate.")
            return
        except requests.exceptions.RequestException as e:
            attempt += 1
            log.warning(f"[{camera_id}] Eroare la trimiterea POST (attempt {attempt}): {e}")
            if attempt > HTTP_RETRIES:
                log.error(f"[{camera_id}] Frame pierdut dupa {HTTP_RETRIES + 1} incercari.")
                return
            time.sleep(0.2)


# ============================== CAMERA WORKER ================================

def camera_worker(camera_cfg: dict, detector: Yolo26OnnxDetector):
    """
    Bucla principala pentru o singura camera, ruleaza intr-un thread dedicat:
      captura frame -> detectie YOLO -> deseneaza boxes -> trimite POST
    Rezistenta la reconectare: daca device-ul cade, reincearca sa se reconecteze.
    """
    camera_id = camera_cfg["camera_id"]
    device = camera_cfg["device"]

    threading.current_thread().name = camera_id

    cap = None

    def open_capture():
        c = cv2.VideoCapture(device, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        c.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        # buffer mic ca sa nu acumulam frame-uri vechi (latenta mica, "real-time")
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c

    cap = open_capture()
    if not cap.isOpened():
        log.error(f"[{camera_id}] Nu pot deschide device-ul {device}.")

    frame_count = 0
    last_fps_log = time.time()

    while not stop_event.is_set():
        if not cap.isOpened():
            log.warning(f"[{camera_id}] Camera neconectata, reincerc in 1s...")
            time.sleep(1)
            cap.release()
            cap = open_capture()
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning(f"[{camera_id}] Frame invalid / camera deconectata, reincerc...")
            cap.release()
            time.sleep(1)
            cap = open_capture()
            continue

        # Downscale explicit la 640x480, indiferent de rezolutia nativa a camerei.
        if frame.shape[1] != OUTPUT_WIDTH or frame.shape[0] != OUTPUT_HEIGHT:
            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)

        timestamp = time.time()

        try:
            detections = detector.detect(frame)
        except Exception as e:
            log.error(f"[{camera_id}] Eroare in inferenta YOLO: {e}")
            continue

        processed_frame = draw_detections(frame, detections)

        # Trimitem sincron in acest thread -- ok pentru rate "real-time" pe 2 camere;
        # daca vrei sa nu blochezi bucla de captura in caz de retea lenta, poti trimite
        # postarea intr-un thread-pool separat. Aici o pastram simplu si per-camera.
        post_frame(camera_id, processed_frame, detections, timestamp)

        frame_count += 1
        now = time.time()
        if now - last_fps_log >= 5.0:
            fps = frame_count / (now - last_fps_log)
            log.info(f"[{camera_id}] ~{fps:.1f} FPS (captura+inferenta+POST), {len(detections)} persoane in ultimul frame")
            frame_count = 0
            last_fps_log = now

    if cap is not None:
        cap.release()
    log.info(f"[{camera_id}] Thread oprit.")


# ============================== MAIN ==========================================

def main():
    log.info("Initializare model YOLO26 ONNX...")
    detector = Yolo26OnnxDetector(MODEL_PATH, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)

    def handle_sigint(signum, frame):
        log.info("Semnal de oprire primit (Ctrl+C), inchidere gracioasa...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    threads = []
    for cam_cfg in CAMERAS:
        t = threading.Thread(target=camera_worker, args=(cam_cfg, detector), daemon=True)
        t.start()
        threads.append(t)
        log.info(f"Thread pornit pentru {cam_cfg['camera_id']} ({cam_cfg['device']})")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    for t in threads:
        t.join(timeout=5)

    log.info("Toate thread-urile s-au oprit. Bye.")


if __name__ == "__main__":
    main()