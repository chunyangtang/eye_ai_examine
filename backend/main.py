# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from base64 import b64encode
import io
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Dict, Any
import threading # Import threading for Lock

app = FastAPI()

# Configure CORS to allow requests from your React frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models (Pydantic) ---
class ImageInfo(BaseModel):
    id: str
    type: str  # e.g., '左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照'
    quality: str # e.g., 'Good', 'Usable', 'Bad'
    base64_data: str # Base64 encoded image

class EyeDiagnosis(BaseModel):
    # Dictionary where keys are disease names and values are boolean (detected/not detected)
    糖网: bool = False
    青光眼: bool = False
    AMD: bool = False
    病理性近视: bool = False
    RVO: bool = False
    RAO: bool = False
    视网膜脱离: bool = False
    其它黄斑病变: bool = False
    其它眼病变: bool = False
    正常: bool = False # 'Normal'

class PatientData(BaseModel):
    patient_id: str
    eye_images: List[ImageInfo]
    diagnosis_results: Dict[str, EyeDiagnosis] # 'left_eye', 'right_eye'

class SubmitDiagnosisRequest(BaseModel):
    patient_id: str
    diagnosis_results: Dict[str, EyeDiagnosis]
    eye_images: List[ImageInfo]

# --- Simulated Data Storage (In-memory) ---
patients_data: Dict[str, PatientData] = {}
patient_ids_order: List[str] = []
current_patient_index = 0 # This will now point to the "latest" patient added or navigated to
patient_data_lock = threading.Lock() # Lock for thread-safe access to global data

# --- Helper Functions (Backend-specific) ---
def generate_random_image_base64_backend(text: str, color: tuple) -> str:
    """Generates a simple colored square image with text and returns its Base64 string.
       For backend use, keeping original smaller size."""
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

def create_initial_dummy_patient_data_batch(num_patients: int = 5) -> None:
    """Creates an initial batch of dummy data on server startup."""
    global patients_data, patient_ids_order, current_patient_index
    with patient_data_lock:
        patients_data.clear()
        patient_ids_order.clear()
        
        for i in range(num_patients):
            patient_id = f"startup_patient_{random.randint(1000, 9999)}"
            eye_images: List[ImageInfo] = []
            image_types = ['左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照', 'OCT', 'FFA']
            image_qualities = ['Good', 'Usable', 'Bad']
            colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (50, 200, 200), (200, 50, 200)]

            for img_idx in range(6): # Each patient gets 6 dummy images
                img_type = random.choice(image_types)
                img_quality = random.choice(image_qualities)
                img_color = random.choice(colors)
                base64_data = generate_random_image_base64_backend(f"{img_type}\nQ:{img_quality}\nImg {img_idx+1}", img_color)
                eye_images.append(ImageInfo(
                    id=f"img_{patient_id}_{img_idx}",
                    type=img_type,
                    quality=img_quality,
                    base64_data=base64_data
                ))
            
            # Simulate random diagnosis results
            left_diagnosis = EyeDiagnosis(**{
                disease: random.choice([True, False]) for disease in EyeDiagnosis.__fields__.keys()
            })
            right_diagnosis = EyeDiagnosis(**{
                disease: random.choice([True, False]) for disease in EyeDiagnosis.__fields__.keys()
            })
            
            new_patient = PatientData(
                patient_id=patient_id,
                eye_images=eye_images,
                diagnosis_results={
                    "left_eye": left_diagnosis,
                    "right_eye": right_diagnosis
                }
            )
            patients_data[new_patient.patient_id] = new_patient
            patient_ids_order.append(new_patient.patient_id)
        
        # Set the current patient to the first one generated initially
        if patient_ids_order:
            current_patient_index = 0
        print(f"Initialized {len(patients_data)} dummy patients on startup.")
        print(f"Patient IDs: {patient_ids_order}")

# Initialize dummy data on server startup
create_initial_dummy_patient_data_batch()

# --- API Endpoints ---
@app.get("/api/patients/current")
async def get_current_patient_data():
    """Returns the data for the current patient."""
    global current_patient_index
    with patient_data_lock:
        if not patient_ids_order:
            raise HTTPException(status_code=404, detail="No patient data available")
        
        patient_id = patient_ids_order[current_patient_index]
        print(f"Serving data for current patient: {patient_id}")
        return patients_data[patient_id]

@app.get("/api/patients/next")
async def get_next_patient_data():
    """Returns the data for the next patient in sequence."""
    global current_patient_index
    with patient_data_lock:
        if not patient_ids_order:
            raise HTTPException(status_code=404, detail="No patient data available")
        
        current_patient_index = (current_patient_index + 1) % len(patient_ids_order)
        patient_id = patient_ids_order[current_patient_index]
        print(f"Serving data for next patient: {patient_id}")
        return patients_data[patient_id]

@app.get("/api/patients/previous")
async def get_previous_patient_data():
    """Returns the data for the previous patient in sequence."""
    global current_patient_index
    with patient_data_lock:
        if not patient_ids_order:
            raise HTTPException(status_code=404, detail="No patient data available")
        
        current_patient_index = (current_patient_index - 1 + len(patient_ids_order)) % len(patient_ids_order)
        patient_id = patient_ids_order[current_patient_index]
        print(f"Serving data for previous patient: {patient_id}")
        return patients_data[patient_id]

@app.post("/api/submit_diagnosis")
async def submit_diagnosis(request: SubmitDiagnosisRequest):
    """
    Receives updated diagnosis results and image info from the frontend and "saves" them.
    """
    with patient_data_lock:
        print(f"Received diagnosis and image info submission for patient: {request.patient_id}")
        if request.patient_id in patients_data:
            patients_data[request.patient_id].diagnosis_results = request.diagnosis_results
            patients_data[request.patient_id].eye_images = request.eye_images
            print(f"Diagnosis and image info updated for {request.patient_id}")
            return {"status": "Diagnosis and image info submitted successfully!"}
        else:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found")

@app.post("/api/add_new_patient")
async def add_new_patient_data(patient_data: PatientData):
    """
    Adds a new patient's data received from an external source (e.g., trigger script)
    and sets them as the current patient.
    """
    global patients_data, patient_ids_order, current_patient_index
    with patient_data_lock:
        if patient_data.patient_id in patients_data:
            print(f"Patient {patient_data.patient_id} already exists. Overwriting.")
            # Remove old entry from order if exists
            if patient_data.patient_id in patient_ids_order:
                patient_ids_order.remove(patient_data.patient_id)
        
        patients_data[patient_data.patient_id] = patient_data
        patient_ids_order.append(patient_data.patient_id)
        
        # Set the newly added patient as the current one
        current_patient_index = len(patient_ids_order) - 1 

        print(f"New patient {patient_data.patient_id} added and set as current.")
        return {"status": f"Patient {patient_data.patient_id} added successfully and set as current."}


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
