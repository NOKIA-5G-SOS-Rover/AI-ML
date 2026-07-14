from tkinter import Image

import cv2
from ultralytics import YOLO
from PIL import Image

cap = cv2.VideoCapture(0)
model = YOLO('yolo11s.pt')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    results = model(frame, classes=[0])
    frame = results[0].plot()


    cv2.imshow('WebCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
