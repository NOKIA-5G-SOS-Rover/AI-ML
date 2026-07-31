# 1. Folosim o versiune de Linux cu Python gata instalat
FROM python:3.10-slim

# 2. Setam folderul de lucru in interiorul containerului
WORKDIR /app

# 3. Instalam pachetele de sistem necesare pentru OpenCV pe Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiem scripturile noastre in container
COPY person_detector_server.py .
COPY send_detection.py .

# 5. Instalam bibliotecile de Python pe care le-am folosit si local
RUN pip install --no-cache-dir requests python-dotenv opencv-python ultralytics flask

# 6. Expunem portul pe care emite Flask-ul stream-ul video
EXPOSE 8080

# 7. Comanda de start a containerului
# In productie pe Rover (care ruleaza Linux), AI-ul se va descurca perfect 
CMD ["python", "person_detector_server.py", "--camera", "0"]