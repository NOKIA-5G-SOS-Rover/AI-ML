FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY ml.py .
COPY yolo26n_img480_int8.onnx .

ENV PYTHONPATH="/opt/arduino/app_utils:${PYTHONPATH}"

CMD ["python", "main.py"]
