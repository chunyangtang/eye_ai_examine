# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Dict
import threading # Import threading for Lock

from datatype import ImageInfo, EyeDiagnosis, PatientData, SubmitDiagnosisRequest, EyePrediction, EyePredictionThresholds, ManualDiagnosisData
from patientdataio import load_batch_patient_data, create_batch_dummy_patient_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Storage (In-memory cache) ---
patients_data_cache: Dict[str, PatientData] = {}
# Storage for manual diagnosis data (separate from AI predictions)
manual_diagnosis_storage: Dict[str, ManualDiagnosisData] = {}
patient_data_lock = threading.Lock() # Lock for thread-safe access to cached data

# No preloading - data will be loaded on-demand

# --- API Endpoints ---
@app.get("/api/patients/{ris_exam_id}")
async def get_patient_by_id(ris_exam_id: str):
    """Returns the data for a specific patient by ris_exam_id (patient_id)."""
    import time
    start_time = time.time()
    
    with patient_data_lock:
        # Check cache first
        if ris_exam_id in patients_data_cache:
            print(f"Serving cached data for patient: {ris_exam_id} (took {time.time() - start_time:.2f}s)")
            return patients_data_cache[ris_exam_id]
        
        # Load from data source
        try:
            print(f"Loading patient {ris_exam_id} from data source...")
            from patientdataio import load_single_patient_data
            patient_data = load_single_patient_data("../data/inference_results.json", ris_exam_id)
            # Cache the loaded data
            patients_data_cache[ris_exam_id] = patient_data
            elapsed = time.time() - start_time
            print(f"Loaded and cached data for patient: {ris_exam_id} (took {elapsed:.2f}s)")
            return patient_data
        except KeyError:
            print(f"Patient {ris_exam_id} not found (took {time.time() - start_time:.2f}s)")
            raise HTTPException(status_code=404, detail=f"Patient {ris_exam_id} not found")
        except Exception as e:
            print(f"Error loading patient {ris_exam_id}: {e} (took {time.time() - start_time:.2f}s)")
            raise HTTPException(status_code=500, detail="Failed to load patient data")

@app.get("/api/patients")
async def get_available_patient_ids():
    """Returns a list of available patient IDs from the data source."""
    try:
        import json
        # Only load the JSON keys, not the full data
        with open("../data/inference_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        patient_ids = list(data.keys())
        print(f"Found {len(patient_ids)} available patients")
        return {"patient_ids": patient_ids}
    except Exception as e:
        print(f"Error loading patient IDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to load patient IDs")

@app.post("/api/submit_diagnosis")
async def submit_diagnosis(request: SubmitDiagnosisRequest):
    """
    Receives manual diagnosis data and image info updates from the frontend and stores them.
    """
    with patient_data_lock:
        print(f"Received diagnosis submission for patient: {request.patient_id}")
        
        # Check if patient exists in cache
        patient = patients_data_cache.get(request.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found in cache. Load patient data first.")
        
        # Store manual diagnosis data
        if request.manual_diagnosis or request.custom_diseases or request.diagnosis_notes:
            from datatype import ManualDiagnosisData, CustomDiseases
            
            manual_data = ManualDiagnosisData(
                manual_diagnosis=request.manual_diagnosis or {},
                custom_diseases=request.custom_diseases or CustomDiseases(),
                diagnosis_notes=request.diagnosis_notes or ""
            )
            
            manual_diagnosis_storage[request.patient_id] = manual_data
            print(f"Manual diagnosis data stored for patient: {request.patient_id}")
            print(f"Manual diagnosis: {manual_data.manual_diagnosis}")
            print(f"Custom diseases: {manual_data.custom_diseases}")
            print(f"Diagnosis notes: {manual_data.diagnosis_notes}")
        
        # Update image type/quality if provided
        if request.image_updates:
            for update_img in request.image_updates:
                for img in patient.eye_images:
                    if img.id == update_img["id"]:
                        img.type = update_img["type"]
                        img.quality = update_img["quality"]
            print(f"Image info updated for {request.patient_id}")
        
        return {"status": "Manual diagnosis and image info submitted successfully!"}

@app.get("/api/manual_diagnosis/{ris_exam_id}")
async def get_manual_diagnosis(ris_exam_id: str):
    """
    Returns the manual diagnosis data for a specific patient.
    """
    with patient_data_lock:
        manual_data = manual_diagnosis_storage.get(ris_exam_id)
        if manual_data:
            return manual_data
        else:
            # Return empty structure if no manual diagnosis exists yet
            from datatype import ManualDiagnosisData, CustomDiseases
            return ManualDiagnosisData(
                manual_diagnosis={},
                custom_diseases=CustomDiseases(),
                diagnosis_notes=""
            )

@app.get("/api/manual_diagnoses")
async def get_all_manual_diagnoses():
    """
    Returns all stored manual diagnosis data for debugging purposes.
    """
    with patient_data_lock:
        return {
            "stored_diagnoses": list(manual_diagnosis_storage.keys()),
            "data": manual_diagnosis_storage
        }

@app.post("/api/add_new_patient")
async def add_new_patient_data(patient_data: PatientData):
    """
    Adds a new patient's data received from an external source (e.g., trigger script)
    and caches it for future access.
    """
    with patient_data_lock:
        if patient_data.patient_id in patients_data_cache:
            print(f"Patient {patient_data.patient_id} already exists in cache. Overwriting.")
        
        patients_data_cache[patient_data.patient_id] = patient_data
        print(f"New patient {patient_data.patient_id} added to cache.")
        return {"status": f"Patient {patient_data.patient_id} added successfully to cache."}


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
