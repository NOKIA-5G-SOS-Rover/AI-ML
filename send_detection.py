import os
import signal
import sys
import time
import requests
import platform
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://92.87.91.146:5000/events")
ROVER_ID = os.getenv("ROVER_ID", "ROVER-Q1")
CAMERA_ID = os.getenv("CAMERA_ID", "CAM-01")

def trimite_alerta_in_cloud(person_data):
    """Formateaza si trimite JSON-ul catre backend-ul C#"""
    
    # 1. Calculam dimensiunile
    raw_width = float(person_data["x2"] - person_data["x1"])
    raw_height = float(person_data["y2"] - person_data["y1"])
    
    # 2. Fortam limitele pentru a trece de atributele [Range] din C#
    box_width = max(raw_width, 0.1)   # Trebuie sa fie > 0
    box_height = max(raw_height, 1.0) # Trebuie sa fie minim 1
    
    # 3. Asiguram ca scorul de incredere este intre 0 si 1
    conf = float(person_data["confidence"])
    if conf > 1.0:
        conf = conf / 100.0
        
    payload = {
        "roverId": ROVER_ID,
        "sessionId": "Misiune-Auto",
        "alertType": "Human Detected",
        "source": "YOLOv8-Camera",
        "detectedAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "locationX": max(float(person_data.get("x1", 0.0)), 0.0), # Range >= 0
        "locationY": max(float(person_data.get("y1", 0.0)), 0.0), # Range >= 0
        "boundingBoxWidth": box_width,
        "boundingBoxHeight": box_height,
        "confidenceScore": conf,
        "motorHaltRequested": True,
        "injuryClass": "Unknown",
        "cameraId": CAMERA_ID,
        "status": "NEW"
    }
    
    try:
        res = requests.post(BACKEND_URL, json=payload, timeout=5)
        if res.status_code in (200, 201):
            # .NET Core returneaza JSON-ul cu camelCase implicit
            print(f"Alerta expediata in Cloud! ID Server: {res.json().get('id', 'N/A')}")
        else:
            print(f"Backend-ul a respins alerta. Status code: {res.status_code}")
            # Adaugam res.text ca sa vedem exact ce camp pica la validare in C#
            print(f"Motiv: {res.text}") 
    except Exception as e:
        print(f"Eroare la trimiterea catre cloud: {e}")

def proceseaza_detectia(sursa="Semnal"):
    print(f"\n[!] Declanșare prin {sursa}! O persoană a apărut pe ecran.")
    try:
        response = requests.get("http://localhost:8080/api/detections", timeout=2)
        if response.status_code == 200:
            data = response.json()
            persons = data.get("persons", [])
            if persons:
                trimite_alerta_in_cloud(persons[0])
            else:
                print("Date primite, dar nicio persoana in cadru.")
    except requests.exceptions.RequestException as e:
        print(f"Eroare API local: {e}")

# Helper pentru declansarea standard de pe robot (Linux)
def handle_sigusr1(signum, frame):
    proceseaza_detectia("Linux SIGUSR1")

if __name__ == "__main__":
    print("==================================================")
    print(f"Comms Manager pornit! PID: {os.getpid()}")
    print("==================================================")
    
    if platform.system() == "Windows":
        print("Sistem Windows detectat. Mod de testare local (Polling) activat...")
        last_person_state = False # Tine minte daca la pasul anterior era un om pe ecran
        try:
            while True:
                try:
                    res = requests.get("http://localhost:8080/api/detections", timeout=1)
                    if res.status_code == 200:
                        data = res.json()
                        current_person_state = data.get("person_present", False)
                        
                        # Daca inainte nu era nimeni (False) si acum a aparut cineva (True) -> Trimitem alerta!
                        if current_person_state and not last_person_state:
                            if data.get("persons"):
                                proceseaza_detectia("Windows Polling")
                        
                        # Actualizam starea pentru secunda urmatoare
                        last_person_state = current_person_state
                        
                except requests.exceptions.RequestException:
                    pass # Ignoram erorile cand AI-ul nu este inca pornit
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInchidere...")
            sys.exit(0)
    else:
        print("Sistem Linux detectat. Astept semnalul SIGUSR1 de la AI...")
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInchidere...")
            sys.exit(0)