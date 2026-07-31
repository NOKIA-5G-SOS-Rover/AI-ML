# 1. Folosim o versiune de Linux cu Python gata instalat
FROM python:3.10-slim

# 2. Setam folderul de lucru in interiorul containerului
WORKDIR /app

# 3. Copiem scripturile noastre in container
COPY person_detector_server.py .
COPY send_detection.py .

# 4. Instalam bibliotecile de Python. 
# Folosim opencv-python-headless care nu necesita dependente de sistem Linux!
RUN pip install --no-cache-dir requests python-dotenv opencv-python-headless ultralytics flask

# 5. Expunem portul pe care emite Flask-ul stream-ul video
EXPOSE 8080

# 6. Comanda de start a containerului
CMD ["python", "person_detector_server.py", "--camera", "0"]