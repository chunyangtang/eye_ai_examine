# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Dict, Any
import threading # Import threading for Lock

from datatype import ImageInfo, EyeDiagnosis, PatientData, SubmitDiagnosisRequest, EyePrediction, EyePredictionThresholds, ManualDiagnosisData, UpdateSelectionRequest
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
# Cache of raw per-image probabilities to avoid disk I/O on reselection
raw_probs_cache: Dict[str, Dict[str, Any]] = {}
RAW_JSON_PATH = "../data/inference_results.json"
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
            # Warm raw prob cache if missing
            if ris_exam_id not in raw_probs_cache:
                try:
                    def _parse_ts(p: str) -> int:
                        try:
                            base = os.path.basename(p)
                            name, _ = os.path.splitext(base)
                            parts = name.split('_')
                            cand = parts[-1] if parts else name
                            digits = ''.join(ch for ch in cand if ch.isdigit())
                            return int(digits) if digits else 0
                        except Exception:
                            return 0
                    import json, os
                    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
                        all_json = json.load(f)
                    raw_patient = all_json.get(ris_exam_id, {})
                    image_type_mapping = {
                        "右眼眼底": "右眼CFP",
                        "左眼眼底": "左眼CFP",
                        "右眼外观": "右眼外眼照",
                        "左眼外观": "左眼外眼照",
                    }
                    raw_by_type: Dict[str, list] = {}
                    raw_by_id: Dict[str, Dict] = {}
                    for img in raw_patient.get("images", []):
                        mapped_type = image_type_mapping.get(img.get("eye", ""), "")
                        raw_by_type.setdefault(mapped_type, []).append(img)
                        img_id = f"img_{ris_exam_id}_{img.get('img_path', '')}"
                        raw_by_id[img_id] = img.get("probs", {})
                    # sort lists once, keep only needed order
                    for k in list(raw_by_type.keys()):
                        raw_by_type[k] = sorted(raw_by_type[k], key=lambda im: _parse_ts(im.get("img_path", "")), reverse=True)
                    raw_probs_cache[ris_exam_id] = {"by_type": raw_by_type, "by_id": raw_by_id}
                except Exception:
                    pass
            return patients_data_cache[ris_exam_id]
        
        # Load from data source
        try:
            print(f"Loading patient {ris_exam_id} from data source...")
            from patientdataio import load_single_patient_data
            patient_data = load_single_patient_data(RAW_JSON_PATH, ris_exam_id)
            # Cache the loaded data
            patients_data_cache[ris_exam_id] = patient_data
            # Build raw prob cache for this patient (warm for reselection)
            try:
                def _parse_ts(p: str) -> int:
                    try:
                        base = os.path.basename(p)
                        name, _ = os.path.splitext(base)
                        parts = name.split('_')
                        cand = parts[-1] if parts else name
                        digits = ''.join(ch for ch in cand if ch.isdigit())
                        return int(digits) if digits else 0
                    except Exception:
                        return 0
                import json, os
                with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
                    all_json = json.load(f)
                raw_patient = all_json.get(ris_exam_id, {})
                image_type_mapping = {
                    "右眼眼底": "右眼CFP",
                    "左眼眼底": "左眼CFP",
                    "右眼外观": "右眼外眼照",
                    "左眼外观": "左眼外眼照",
                }
                raw_by_type: Dict[str, list] = {}
                raw_by_id: Dict[str, Dict] = {}
                for img in raw_patient.get("images", []):
                    mapped_type = image_type_mapping.get(img.get("eye", ""), "")
                    raw_by_type.setdefault(mapped_type, []).append(img)
                    img_id = f"img_{ris_exam_id}_{img.get('img_path', '')}"
                    raw_by_id[img_id] = img.get("probs", {})
                for k in list(raw_by_type.keys()):
                    raw_by_type[k] = sorted(raw_by_type[k], key=lambda im: _parse_ts(im.get("img_path", "")), reverse=True)
                raw_probs_cache[ris_exam_id] = {"by_type": raw_by_type, "by_id": raw_by_id}
            except Exception:
                pass
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


@app.post("/api/update_selection")
async def update_selection(request: UpdateSelectionRequest):
    """
    Recompute prediction_results and diagnosis_results based on the selected image IDs.
    Rules:
    - For each eye, prefer CFP probs when a matching CFP image is selected; cataract prefers 外眼 if an external-eye image is selected for that eye.
    - If selected images don't include a type, fallback to any available for that type; else use zeros for missing.
    - Update thresholds for cataract when 外眼 is used.
    """
    with patient_data_lock:
        patient = patients_data_cache.get(request.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found in cache. Load patient data first.")

        # Build maps for quick lookup
        selected_set = set(request.selected_image_ids or [])
        all_images = list(patient.eye_images)

        # Helper to find first image by type among selected, else among all
        def find_image_by_type(img_type: str):
            for im in all_images:
                if im.type == img_type and im.id in selected_set:
                    return im
            for im in all_images:
                if im.type == img_type:
                    return im
            return None

        # We need access to original JSON probs to recompute.
        # Prefer in-memory cache to avoid disk IO; ensure it's present.
        if request.patient_id not in raw_probs_cache:
            # Try to warm the cache quickly from disk
            try:
                def _parse_ts(p: str) -> int:
                    try:
                        base = os.path.basename(p)
                        name, _ = os.path.splitext(base)
                        parts = name.split('_')
                        cand = parts[-1] if parts else name
                        digits = ''.join(ch for ch in cand if ch.isdigit())
                        return int(digits) if digits else 0
                    except Exception:
                        return 0
                import json, os
                with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
                    all_json = json.load(f)
                raw_patient = all_json.get(request.patient_id)
                if raw_patient is None:
                    raise HTTPException(status_code=404, detail="Patient raw data not found")
                image_type_mapping = {
                    "右眼眼底": "右眼CFP",
                    "左眼眼底": "左眼CFP",
                    "右眼外观": "右眼外眼照",
                    "左眼外观": "左眼外眼照",
                }
                raw_by_type: Dict[str, list] = {}
                raw_by_id: Dict[str, Dict] = {}
                for img in raw_patient.get("images", []):
                    mapped_type = image_type_mapping.get(img.get("eye", ""), "")
                    raw_by_type.setdefault(mapped_type, []).append(img)
                    img_id = f"img_{request.patient_id}_{img.get('img_path', '')}"
                    raw_by_id[img_id] = img.get("probs", {})
                for k in list(raw_by_type.keys()):
                    raw_by_type[k] = sorted(raw_by_type[k], key=lambda im: _parse_ts(im.get("img_path", "")), reverse=True)
                raw_probs_cache[request.patient_id] = {"by_type": raw_by_type, "by_id": raw_by_id}
            except Exception:
                raise HTTPException(status_code=501, detail="Recompute not supported without source data")

        raw_by_type = raw_probs_cache[request.patient_id]["by_type"]
        raw_by_id = raw_probs_cache[request.patient_id]["by_id"]

        # Fallback: get the latest image's probs for a given type
        def extract_probs_by_type(desired_type: str):
            lst = raw_by_type.get(desired_type, [])
            return (lst[0].get("probs", {}) if lst else {})

        diagnosis_mapping = {
            "青光眼": "青光眼",
            "糖尿病性视网膜病变": "糖网",
            "年龄相关性黄斑变性": "AMD",
            "病理性近视": "病理性近视",
            "视网膜静脉阻塞（RVO）": "RVO",
            "视网膜动脉阻塞（RAO）": "RAO",
            "视网膜脱离（RD）": "视网膜脱离",
            "其他视网膜病": "其它视网膜病",
            "其他黄斑病变": "其它黄斑病变",
            "白内障": "白内障",
            "正常": "正常"
        }

        from datatype import CATARACT_EXTERNAL_THRESHOLD

        prediction_thresholds = EyePredictionThresholds(**patient.prediction_thresholds.dict())
        prediction_results = {}
        diagnosis_results = {}

        ext_cataract_used_overall = False
        debug_used = {"left_eye": {}, "right_eye": {}}
        for eye in ["left_eye", "right_eye"]:
            cfp_type = "左眼CFP" if eye == "left_eye" else "右眼CFP"
            ext_type = "左眼外眼照" if eye == "left_eye" else "右眼外眼照"

            # Determine selected image presence per type
            selected_cfp = find_image_by_type(cfp_type)
            selected_ext = find_image_by_type(ext_type)

            # Extract probs: prefer exact selected image's probs via image ID; fallback to latest-by-type
            if selected_cfp and selected_cfp.id in raw_by_id:
                raw_cfp = raw_by_id[selected_cfp.id]
                debug_used[eye]["cfp"] = selected_cfp.id
            else:
                raw_cfp = extract_probs_by_type(cfp_type)
                debug_used[eye]["cfp"] = f"latest_by_type:{cfp_type}"
            if selected_ext and selected_ext.id in raw_by_id:
                raw_ext = raw_by_id[selected_ext.id]
                debug_used[eye]["ext"] = selected_ext.id
            else:
                raw_ext = extract_probs_by_type(ext_type)
                debug_used[eye]["ext"] = f"latest_by_type:{ext_type}"

            # Normalize mapped probs
            def map_probs(raw):
                return {diagnosis_mapping.get(k, k): v for k, v in (raw or {}).items()}

            cfp_probs = map_probs(raw_cfp)
            ext_probs = map_probs(raw_ext)

            # Start from zeros if missing
            def zero_probs():
                return {k: 0.0 for k in diagnosis_mapping.values()}

            probs = zero_probs()

            # Use CFP probs when available (prefer selected CFP if exists; but probabilities are same per type here)
            if cfp_probs:
                probs.update(cfp_probs)

            # If an external-eye image is selected for this eye and it has cataract probability, override cataract
            ext_used = False
            if selected_ext and ("白内障" in ext_probs):
                probs["白内障"] = ext_probs["白内障"]
                ext_used = True

            # Save predictions
            prediction_results[eye] = EyePrediction(**probs)

            # Thresholds and diagnoses
            eye_thresholds = prediction_thresholds.dict()
            if ext_used:
                eye_thresholds["白内障"] = CATARACT_EXTERNAL_THRESHOLD
                ext_cataract_used_overall = True
            # Build diagnosis using possibly adjusted threshold
            diag = {}
            for disease, threshold in eye_thresholds.items():
                diag[disease] = getattr(prediction_results[eye], disease, 0.0) >= threshold
            diagnosis_results[eye] = EyeDiagnosis(**diag)

        # If any eye used ext cataract, reflect that in thresholds object returned (for UI remap); keep cache thresholds unchanged otherwise
        if ext_cataract_used_overall:
            setattr(prediction_thresholds, "白内障", CATARACT_EXTERNAL_THRESHOLD)

        # Update cache object
        patient.prediction_results = prediction_results
        patient.diagnosis_results = diagnosis_results
        # Note: prediction_thresholds may be eye-dependent for cataract; keeping patient-wide as prior

        return {
            "status": "Selection updated",
            "prediction_results": prediction_results,
            "diagnosis_results": diagnosis_results,
            "prediction_thresholds": prediction_thresholds,
            "debug_used_images": debug_used,
        }

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
