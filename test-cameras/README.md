# Multi-Camera Person Detection with YOLO26 ONNX

This project performs **real-time person detection** from two USB cameras using a **YOLO26 ONNX** model and sends the processed frames together with detection metadata to a REST server.

Each camera runs independently in its own thread, allowing simultaneous capture, inference, and upload without blocking the other camera.

---

## Features

- Real-time person detection using **YOLO26** exported to **ONNX**
- Supports multiple USB cameras (default: 2)
- Independent processing thread per camera
- ONNX Runtime inference
- Automatic camera reconnection
- Bounding box visualization
- JPEG frame compression
- REST API upload using `multipart/form-data`
- JSON metadata containing detections and timestamps
- Graceful shutdown with `Ctrl+C`

---

## Project Files

| File | Description |
|------|-------------|
| `ML_Record_And_Detect.py` | Main application for camera capture, inference, and REST upload |
| `yolo26n_img480_int8.onnx` | Quantized YOLO26 ONNX model used for inference |

---

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- ONNX Runtime
- Requests

Install dependencies:

```bash
pip install opencv-python numpy onnxruntime requests
```

---

## Configuration

Before running the application, modify the configuration section in `ML_Record_And_Detect.py`.

### REST Server

```python
SERVER_URL = "http://<SERVER_IP>:5000"
```

Set the address of your REST server.

---

### Model

```python
MODEL_PATH = "yolo26n_img480_int8.onnx"
```

Adjust the following parameters if your model differs:

- `INPUT_WIDTH`
- `INPUT_HEIGHT`
- `CONF_THRES`
- `CLASS_ID_PERSON`

---

### Cameras

Default configuration:

```python
CAMERAS = [
    {"camera_id": "cam0", "device": "/dev/video0"},
    {"camera_id": "cam1", "device": "/dev/video2"},
]
```

Update the device paths if your cameras are connected differently.

---

## Model Format

The application expects an **Ultralytics YOLO26 ONNX export** with **end-to-end NMS** already integrated.

### Input

```
(1, 3, 480, 640)
```

- RGB
- Float32
- Normalized to `[0,1]`

Frames are resized directly to **640×480** without letterboxing.

### Output

```
(1, N, 6)

[x1, y1, x2, y2, confidence, class_id]
```

Because NMS is already part of the exported model, **no additional Non-Maximum Suppression is performed** in the application.

---

## Processing Pipeline

Each camera follows the same processing loop:

```
Camera
   │
   ▼
Capture Frame
   │
   ▼
Resize (640×480)
   │
   ▼
YOLO26 ONNX Inference
   │
   ▼
Filter Person Detections
   │
   ▼
Draw Bounding Boxes
   │
   ▼
JPEG Encoding
   │
   ▼
POST to REST Server
```

Each camera operates in its own thread, allowing continuous processing even if another camera experiences delays.

---

## REST Upload

Each processed frame is uploaded using `multipart/form-data`.

### Image

```
image
```

JPEG encoded frame containing the rendered detections.

### Metadata

```json
{
  "camera_id": "cam0",
  "timestamp": 1712345678.123,
  "person_count": 2,
  "detections": [
    {
      "bbox": [120.4, 80.3, 250.8, 420.6],
      "confidence": 0.94
    }
  ]
}
```

---

## Running

```bash
python3 ML_Record_And_Detect.py
```

The application will:

1. Load the YOLO26 ONNX model.
2. Open both cameras.
3. Start one processing thread per camera.
4. Detect people in real time.
5. Draw detections on each frame.
6. Upload frames and metadata to the configured REST server.

Terminate with:

```
Ctrl+C
```

which performs a graceful shutdown of all camera threads.

---

## Notes

- Designed for Linux systems using V4L2 (`/dev/video*`).
- The default implementation uses the CPU execution provider for ONNX Runtime.
- CUDA or other hardware providers can be enabled if available.
- Camera reconnection is handled automatically if a device becomes unavailable.
- The application is optimized for low-latency processing using a small capture buffer.

---


## Changes needed

- Traba sa pa pe placa video
- Traba sa vad daca modelul e pe float32 sau pe int8 cum ar trebui
- Traba sa gasesc unde fac Request-ul de POST
