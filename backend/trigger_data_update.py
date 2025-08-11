# trigger_data_update.py
import requests
import time
import random
from base64 import b64encode
import io
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any
from pydantic import BaseModel # Import BaseModel

# --- Data Models (Pydantic - duplicated for self-contained script) ---
class ImageInfo(BaseModel):
    id: str
    type: str
    quality: str
    base64_data: str

class EyeDiagnosis(BaseModel):
    糖网: bool = False
    青光眼: bool = False
    AMD: bool = False
    病理性近视: bool = False
    RVO: bool = False
    RAO: bool = False
    视网膜脱离: bool = False
    其它黄斑病变: bool = False
    其它眼病变: bool = False
    正常: bool = False

class PatientData(BaseModel):
    patient_id: str
    eye_images: List[ImageInfo]
    diagnosis_results: Dict[str, EyeDiagnosis]

# URL of your FastAPI backend endpoint to add new patient data
BACKEND_URL = "http://127.0.0.1:8000/api/add_new_patient"

# --- Helper Functions (Frontend-agnostic image generation) ---
def generate_random_image_base64_external(text: str, color: tuple) -> str:
    """Generates a simple colored square image with text and returns its Base64 string.
       This is for the external script, keeping it self-contained."""
    width, height = 200, 200 # Keeping smaller for real-scenario simulation
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    text_color = (255, 255, 255) if sum(color) < 300 else (0, 0, 0)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = (width - text_width) / 2
    text_y = (height - text_height) / 2

    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return b64encode(buffered.getvalue()).decode("utf-8")

def create_single_dummy_patient_data() -> PatientData:
    """Creates dummy data for a single new patient for the external trigger."""
    patient_id = f"external_trigger_{int(time.time())}_{random.randint(100, 999)}"
    
    eye_images: List[ImageInfo] = []
    image_types = ['左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照', 'OCT', 'FFA']
    image_qualities = ['Good', 'Usable', 'Bad']
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(6)]
    
    # Ensure at least 4 images are generated, randomly choosing types
    generated_types = set()
    for i in range(6): # Generate a fixed number of images, or adjust as needed
        img_type = random.choice(image_types)
        img_quality = random.choice(image_qualities)
        img_color = colors[i] # Use pre-defined random colors
        base64_data = generate_random_image_base64_external(f"{img_type}\nQ:{img_quality}\nImg {i+1}", img_color)
        eye_images.append(ImageInfo(
            id=f"img_{patient_id}_{i}",
            type=img_type,
            quality=img_quality,
            base64_data=base64_data
        ))
    
    # Simulate random diagnosis results
    left_diagnosis = EyeDiagnosis(**{
        disease: random.choice([True, False]) for disease in EyeDiagnosis.__fields__.keys() if disease != '其它眼病变'
    })
    # Manually assign '其它眼病变' as it's not a direct boolean in frontend, but bool in backend
    left_diagnosis.其它眼病变 = random.choice([True, False])

    right_diagnosis = EyeDiagnosis(**{
        disease: random.choice([True, False]) for disease in EyeDiagnosis.__fields__.keys() if disease != '其它眼病变'
    })
    right_diagnosis.其它眼病变 = random.choice([True, False])
    
    return PatientData(
        patient_id=patient_id,
        eye_images=eye_images,
        diagnosis_results={
            "left_eye": left_diagnosis,
            "right_eye": right_diagnosis
        }
    )

def send_new_patient_data():
    """Generates a new patient's data and sends it to the backend."""
    patient_data = create_single_dummy_patient_data()
    print(f"Generating new patient data for ID: {patient_data.patient_id}")
    print(f"Attempting to send data to backend at {BACKEND_URL}...")
    try:
        response = requests.post(BACKEND_URL, json=patient_data.dict()) # Send Pydantic model as dict
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        print(f"Backend response: {response.json().get('status', 'Success')}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to the backend. Is FastAPI running at {BACKEND_URL}?")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred during the request: {e}")

if __name__ == "__main__":
    print("This script will send new patient data to the FastAPI backend.")
    print("Press Enter to send a new patient's data (or type 'exit' to quit).")
    while True:
        user_input = input("Action (press Enter or type 'exit'): ")
        if user_input.lower() == 'exit':
            print("Exiting trigger script.")
            break
        else:
            send_new_patient_data()
            print("\nWaiting for next patient data input...")
