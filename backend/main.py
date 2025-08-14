# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Dict
import threading # Import threading for Lock

from datatype import ImageInfo, EyeDiagnosis, PatientData, SubmitDiagnosisRequest, EyePrediction, EyePredictionThresholds
from patientdataio import load_batch_patient_data, create_batch_dummy_patient_data

app = FastAPI()

# Configure CORS to allow requests from your React frontend
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simulated Data Storage (In-memory) ---
patients_data: Dict[str, PatientData] = {}
patient_ids_order: List[str] = []
current_patient_index = 0 # This will now point to the "latest" patient added or navigated to

patient_data_lock = threading.Lock() # Lock for thread-safe access to global data


# Initialize data on server startup
# initial_patients = create_batch_dummy_patient_data()
initial_patients = load_batch_patient_data()
with patient_data_lock:
    for patient in initial_patients:
        patients_data[patient.patient_id] = patient
        patient_ids_order.append(patient.patient_id)
    if patient_ids_order:
        current_patient_index = 0

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
        print(f"Received diagnosis submission for patient: {request.patient_id}")
        patient = patients_data.get(request.patient_id)
        if patient:
            patient.diagnosis_results = request.diagnosis_results
            # Update image type/quality if provided
            if hasattr(request, "image_updates") and request.image_updates:
                for update_img in request.image_updates:
                    for img in patient.eye_images:
                        if img.id == update_img["id"]:
                            img.type = update_img["type"]
                            img.quality = update_img["quality"]
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
