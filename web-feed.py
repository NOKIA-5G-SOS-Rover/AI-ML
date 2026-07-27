#!/usr/bin/env python3
"""
laptop_server.py
------------------
Runs on your Fedora laptop.

Hosts a Flask app with:
  - POST /frame       -> board pushes each annotated JPEG frame here
  - GET  /video_feed   -> MJPEG stream your browser can view
  - GET  /             -> simple HTML page embedding the stream

The board reaches this app through an SSH reverse tunnel (see the
tunnel command below), so this app just needs to listen on
127.0.0.1:5000 (or 0.0.0.0:5000 if you also want to view it from
another device on your LAN).

Requirements (install on the laptop, Fedora):
    pip install flask

Usage:
    python3 laptop_server.py
    # then open http://127.0.0.1:5000 in a browser

--- SSH reverse tunnel (run this from your laptop, or add -R to your
existing ssh session into the board) ---

    ssh -R 5000:127.0.0.1:5000 user@<board-ip>

This forwards port 5000 *on the board* back to port 5000 *on your
laptop*. So on the board, POSTing to http://127.0.0.1:5000/frame
actually lands on this Flask app running on your laptop.

If you're already SSH'd into the board, you don't need a new
connection: instead, close and reconnect with the -R flag added,
e.g.:

    ssh -R 5000:127.0.0.1:5000 user@<board-ip>

Then run board_stream.py on the board (in that same SSH session, or
another one), and run this script on your laptop.
"""

import threading
import time

from flask import Flask, request, Response, render_template_string

app = Flask(__name__)

# Shared state for the latest frame, protected by a lock since Flask
# handles requests concurrently (POST writer thread vs GET readers).
_latest_frame = None
_frame_lock = threading.Lock()
_last_update_ts = 0.0

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Board Webcam Feed</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; text-align: center; }
        img { max-width: 90vw; border: 2px solid #444; margin-top: 20px; }
        #status { margin-top: 10px; color: #888; }
    </style>
</head>
<body>
    <h1>Live YOLOv8n Feed (Arduino Uno Q)</h1>
    <img src="/video_feed" alt="video stream">
    <div id="status">Waiting for frames...</div>
</body>
</html>
"""


@app.route("/frame", methods=["POST"])
def receive_frame():
    """Board POSTs a raw JPEG byte body here."""
    global _latest_frame, _last_update_ts

    jpeg_bytes = request.get_data()
    if not jpeg_bytes:
        return "empty body", 400

    with _frame_lock:
        _latest_frame = jpeg_bytes
        _last_update_ts = time.time()

    return "ok", 200


def _mjpeg_generator():
    """Yields the latest frame repeatedly in multipart/x-mixed-replace format."""
    boundary = "frame"
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is not None:
            yield (
                f"--{boundary}\r\n"
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode() + frame + b"\r\n"

        # Cap the stream rate; new frames arrive as fast as the board
        # sends them, this just controls how often we re-serve the
        # latest one to connected browser clients.
        time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with _frame_lock:
        has_frame = _latest_frame is not None
        last_ts = _last_update_ts

    age = time.time() - last_ts if last_ts else None
    return {
        "has_frame": has_frame,
        "seconds_since_last_frame": age,
    }


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    # Listen on all interfaces so it's reachable via the SSH tunnel
    # (the tunnel connects to 127.0.0.1:5000 on this machine, which
    # 0.0.0.0 binding also satisfies).
    app.run(host="0.0.0.0", port=5040, threaded=True)


