FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY main.py .
COPY ml.py .
COPY ml_RecordAndDetect.py .
COPY yolo26n_img480_int8.onnx .

CMD ["python", "main.py"]
