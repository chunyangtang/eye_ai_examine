# main.py

import os
import json
import asyncio
import datetime
from dotenv import load_dotenv
import httpx
import threading
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from pathlib import Path


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from datatype import ImageInfo, EyeDiagnosis, PatientData, SubmitDiagnosisRequest, EyePrediction, EyePredictionThresholds, ManualDiagnosisData, UpdateSelectionRequest
from patientdataio import load_batch_patient_data, create_batch_dummy_patient_data

import logging

# 在文件开头添加日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

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
INFERENCE_RESULTS_PATH = RAW_JSON_PATH
# Path to questionnaire data from the other project
CONSULTATION_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../eye_ai_consultation/data/questionnaire_data.json")
)

# Lock for thread-safe access to cached data
patient_data_lock = threading.Lock()
_infer_lock = threading.Lock()
consultation_data_lock = threading.Lock()


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


# --- Consultation Data Models (accept free text and arrays) ---
class EyeSymptomData(BaseModel):
    mainSymptom: Optional[str] = None
    onsetMethod: Optional[str] = None
    onsetTime: Optional[str] = None
    accompanyingSymptoms: Optional[Union[List[str], str]] = None  # allow string or list
    medicalHistory: Optional[str] = None
    mainSymptomOther: Optional[str] = None

    class Config:
        extra = 'allow'  # tolerate extra keys from UI

class ConsultationData(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    affectedArea: Optional[List[str]] = None
    leftEye: Optional[EyeSymptomData] = None
    rightEye: Optional[EyeSymptomData] = None
    bothEyes: Optional[EyeSymptomData] = None
    submissionTime: Optional[str] = None

    class Config:
        extra = 'allow'  # tolerate extra keys

class SaveConsultationRequest(BaseModel):
    ris_exam_id: str  # 修改为ris_exam_id
    consultation_data: ConsultationData



# --- Helpers for consultation file I/O ---
def load_consultation_data() -> List[Dict[str, Any]]:
    if not os.path.exists(CONSULTATION_DATA_PATH):
        return []
    with open(CONSULTATION_DATA_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # If file accidentally not a list, coerce to list
            return [data]
        except json.JSONDecodeError:
            return []

def save_consultation_data(data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(CONSULTATION_DATA_PATH), exist_ok=True)
    with open(CONSULTATION_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def _to_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        return "、".join([str(x) for x in v])
    return str(v)

def normalize_consultation_text(entry: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure eye sections store free text where needed
    for eye_field in ("leftEye", "rightEye", "bothEyes"):
        eye = entry.get(eye_field)
        if isinstance(eye, dict):
            if "accompanyingSymptoms" in eye:
                eye["accompanyingSymptoms"] = _to_text(eye.get("accompanyingSymptoms"))
    return entry

def _parse_iso_to_aware_dt(s: Optional[str]) -> Optional[datetime.datetime]:
    """
    Parse ISO string into a timezone-aware datetime.
    - Accepts 'Z' suffix or explicit offsets.
    - If naive (no tz), assume local timezone.
    """
    if not s:
        return None
    try:
        dt = datetime.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except Exception:
        return None
    if dt.tzinfo is None:
        try:
            local_tz = datetime.datetime.now().astimezone().tzinfo
        except Exception:
            local_tz = datetime.timezone.utc
        dt = dt.replace(tzinfo=local_tz)
    return dt

# --- Correct matching: submissionTime should be BEFORE or equal to examineTime, choose closest ---
def find_best_matching_consultation(all_data: List[Dict[str, Any]], patient_name: Optional[str], exam_time: Optional[str]):
    if not patient_name:
        return None

    # Parse exam_time -> aware (UTC for comparison)
    exam_dt = _parse_iso_to_aware_dt(exam_time)
    exam_dt_utc = exam_dt.astimezone(datetime.timezone.utc) if exam_dt else None

    # Filter by name
    same_name = [x for x in all_data if x.get("name") == patient_name]
    if not same_name:
        return None

    if not exam_dt_utc:
        # If no exam time, return the last record by submissionTime if present
        def time_key(x):
            st_dt = _parse_iso_to_aware_dt(x.get("submissionTime"))
            return (st_dt.astimezone(datetime.timezone.utc) if st_dt else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))
        return sorted(same_name, key=time_key)[-1]

    # Choose the entry with submissionTime <= exam_time and closest to exam_time
    candidates = []
    for x in same_name:
        st_dt = _parse_iso_to_aware_dt(x.get("submissionTime"))
        if not st_dt:
            continue
        st_dt_utc = st_dt.astimezone(datetime.timezone.utc)
        if st_dt_utc <= exam_dt_utc:
            candidates.append((x, st_dt_utc))
    if candidates:
        candidates.sort(key=lambda t: (exam_dt_utc - t[1]))
        return candidates[0][0]

    # Fallback: if none before exam, return the earliest after exam (closest)
    after = []
    for x in same_name:
        st_dt = _parse_iso_to_aware_dt(x.get("submissionTime"))
        if not st_dt:
            continue
        st_dt_utc = st_dt.astimezone(datetime.timezone.utc)
        if st_dt_utc > exam_dt_utc:
            after.append((x, st_dt_utc))
    if after:
        after.sort(key=lambda t: (t[1] - exam_dt_utc))
        return after[0][0]

    return None

# --- Helpers for patient context ---
def _get_patient_context(ris_exam_id: str) -> Dict[str, Optional[str]]:
    """
    Returns {'name': str|None, 'examineTime': str|None} from cache, else from RAW_JSON_PATH.
    """
    ctx = {"name": None, "examineTime": None}
    try:
        patient = patients_data_cache.get(ris_exam_id)
        if patient:
            ctx["name"] = getattr(patient, "name", None)
            ctx["examineTime"] = getattr(patient, "examine_time", None)
            if ctx["name"] or ctx["examineTime"]:
                return ctx
    except Exception:
        pass

    # Fallback to read raw inference file
    try:
        with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
            all_json = json.load(f)
        raw = all_json.get(ris_exam_id, {})
        if isinstance(raw, dict):
            ctx["name"] = raw.get("name")
            ctx["examineTime"] = raw.get("examineTime")
    except Exception:
        pass
    return ctx

def _find_best_matching_index(all_data: List[Dict[str, Any]], patient_name: Optional[str], exam_time: Optional[str]) -> Optional[int]:
    """
    Same selection rule as find_best_matching_consultation, but returns index in list.
    Prefers submissionTime <= examineTime (closest), else earliest after.
    """
    if not patient_name or not isinstance(all_data, list) or len(all_data) == 0:
        return None

    exam_dt = _parse_iso_to_aware_dt(exam_time)
    exam_utc = exam_dt.astimezone(datetime.timezone.utc) if exam_dt else None

    # filter name
    name_idxs = [(i, x) for i, x in enumerate(all_data) if isinstance(x, dict) and x.get("name") == patient_name]
    if not name_idxs:
        return None

    if not exam_utc:
        # no exam time -> pick latest by submissionTime
        def time_key(x):
            st = _parse_iso_to_aware_dt(x.get("submissionTime"))
            return (st.astimezone(datetime.timezone.utc) if st else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))
        name_idxs.sort(key=lambda t: time_key(t[1]))
        return name_idxs[-1][0]

    # before or equal to exam (closest)
    candidates = []
    for i, x in name_idxs:
        st = _parse_iso_to_aware_dt(x.get("submissionTime"))
        if not st:
            continue
        st_utc = st.astimezone(datetime.timezone.utc)
        if st_utc <= exam_utc:
            candidates.append((i, st_utc))
    if candidates:
        candidates.sort(key=lambda t: (exam_utc - t[1]))
        return candidates[0][0]

    # earliest after exam
    after = []
    for i, x in name_idxs:
        st = _parse_iso_to_aware_dt(x.get("submissionTime"))
        if not st:
            continue
        st_utc = st.astimezone(datetime.timezone.utc)
        if st_utc > exam_utc:
            after.append((i, st_utc))
    if after:
        after.sort(key=lambda t: (t[1] - exam_utc))
        return after[0][0]
    return None

# --- API: get/save consultation ---
@app.get("/api/consultation/{ris_exam_id}")
async def get_consultation_info(ris_exam_id: str, patient_name: Optional[str] = None):
    """
    获取问诊信息
    - 如果提供patient_name参数，优先使用该名称搜索问诊信息
    - 否则使用ris_exam_id对应的患者名称搜索
    """
    # 获取患者上下文
    ctx = _get_patient_context(ris_exam_id)
    
    # 决定使用哪个患者姓名进行搜索
    search_name = patient_name if patient_name else ctx.get("name")
    examine_time = ctx.get("examineTime")

    with consultation_data_lock:
        all_consultations = load_consultation_data()
        
        # 如果提供了patient_name，返回所有同名的问诊记录供选择
        if patient_name:
            same_name_consultations = []
            for i, consultation in enumerate(all_consultations):
                if consultation.get("name") == patient_name:
                    # 构建显示信息
                    base_info = {
                        "index": i,
                        "name": consultation.get("name"),
                        "age": consultation.get("age"),
                        "gender": consultation.get("gender"),
                        "phone": consultation.get("phone"),
                        "submissionTime": consultation.get("submissionTime"),
                        "hasRefined": "refined" in consultation and isinstance(consultation["refined"], dict)
                    }
                    
                    # 如果有refined数据，也添加refined的信息
                    if base_info["hasRefined"]:
                        refined = consultation["refined"]
                        refined_info = base_info.copy()
                        refined_info.update({
                            "age": refined.get("age", base_info["age"]),
                            "gender": refined.get("gender", base_info["gender"]),
                            "phone": refined.get("phone", base_info["phone"]),
                            "submissionTime": refined.get("refinedTime", refined.get("submissionTime", base_info["submissionTime"])),
                            "isRefined": True
                        })
                        same_name_consultations.append(refined_info)
                    else:
                        base_info["isRefined"] = False
                        same_name_consultations.append(base_info)
            
            # 按提交时间排序
            same_name_consultations.sort(
                key=lambda x: _parse_iso_to_aware_dt(x.get("submissionTime")) or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
                reverse=True
            )
            
            return {
                "consultation_data": None,
                "status": "multiple_matches",
                "same_name_consultations": same_name_consultations,
                "search_name": patient_name,
                "source": "questionnaire_data.json",
                "path": CONSULTATION_DATA_PATH,
            }
        
        # 原有逻辑：根据名称和时间匹配最佳问诊记录
        idx = _find_best_matching_index(all_consultations, search_name, examine_time)
        if idx is None:
            return {
                "consultation_data": None,
                "status": "no_match",
                "source": "questionnaire_data.json",
                "path": CONSULTATION_DATA_PATH,
            }

        base = all_consultations[idx]
        refined = base.get("refined")
        # 优先使用refined数据
        if isinstance(refined, dict):
            result = dict(refined)
            result.setdefault("name", base.get("name"))
            # 显示refinedTime作为submissionTime
            if "refinedTime" in result:
                result.setdefault("submissionTime", result.get("refinedTime"))
            else:
                result.setdefault("submissionTime", base.get("submissionTime"))
            return {
                "consultation_data": result,
                "status": "success_refined",
                "source": f"questionnaire_data.json[{idx}].refined",
                "path": CONSULTATION_DATA_PATH
            }
        else:
            # 返回原始数据
            minimal = {
                "name": base.get("name"),
                "age": base.get("age"),
                "gender": base.get("gender"),
                "phone": base.get("phone"),
                "affectedArea": base.get("affectedArea"),
                "leftEye": base.get("leftEye"),
                "rightEye": base.get("rightEye"),
                "bothEyes": base.get("bothEyes"),
                "submissionTime": base.get("submissionTime"),
            }
            return {
                "consultation_data": minimal,
                "status": "success_original",
                "source": f"questionnaire_data.json[{idx}]",
                "path": CONSULTATION_DATA_PATH
            }

### 2. 修改根据索引获取特定问诊记录的API

@app.get("/api/consultation/{ris_exam_id}/by_index/{consultation_index}")
async def get_consultation_by_index(ris_exam_id: str, consultation_index: int, use_refined: bool = True):
    """
    根据索引获取特定的问诊记录
    """
    with consultation_data_lock:
        all_consultations = load_consultation_data()
        
        if consultation_index < 0 or consultation_index >= len(all_consultations):
            raise HTTPException(status_code=404, detail="Consultation index out of range")
        
        base = all_consultations[consultation_index]
        
        if use_refined and "refined" in base and isinstance(base["refined"], dict):
            # 使用refined数据
            result = dict(base["refined"])
            result.setdefault("name", base.get("name"))
            if "refinedTime" in result:
                result.setdefault("submissionTime", result.get("refinedTime"))
            else:
                result.setdefault("submissionTime", base.get("submissionTime"))
            return {
                "consultation_data": result,
                "status": "success_refined",
                "source": f"questionnaire_data.json[{consultation_index}].refined",
                "path": CONSULTATION_DATA_PATH
            }
        else:
            # 使用原始数据
            minimal = {
                "name": base.get("name"),
                "age": base.get("age"),
                "gender": base.get("gender"),
                "phone": base.get("phone"),
                "affectedArea": base.get("affectedArea"),
                "leftEye": base.get("leftEye"),
                "rightEye": base.get("rightEye"),
                "bothEyes": base.get("bothEyes"),
                "submissionTime": base.get("submissionTime"),
            }
            return {
                "consultation_data": minimal,
                "status": "success_original",
                "source": f"questionnaire_data.json[{consultation_index}]",
                "path": CONSULTATION_DATA_PATH
            }

# 保存问诊信息的API也需要修改参数名
@app.post("/api/consultation")
async def save_consultation_info(request: SaveConsultationRequest):
    # 需要修改SaveConsultationRequest中的字段名
    with consultation_data_lock:
        all_consultations = load_consultation_data()

        # Build normalized refined payload
        refined_payload = request.consultation_data.dict(exclude_none=True)
        refined_payload = normalize_consultation_text(refined_payload)

        # Ensure name if missing
        if not refined_payload.get("name"):
            ctx = _get_patient_context(request.ris_exam_id)  # 修改为ris_exam_id
            if ctx.get("name"):
                refined_payload["name"] = ctx["name"]

        # Timestamp for refined save (tz-aware)
        refined_time = datetime.datetime.now().astimezone().isoformat()
        refined_payload["refinedTime"] = refined_time
        refined_payload["_exam_ris_exam_id"] = request.ris_exam_id  # 修改字段名

        # Locate original questionnaire item by name + examineTime alignment
        ctx = _get_patient_context(request.ris_exam_id)  # 修改为ris_exam_id
        idx = _find_best_matching_index(all_consultations, ctx.get("name"), ctx.get("examineTime"))

        if idx is not None:
            # Overwrite previous refined content
            all_consultations[idx]["refined"] = refined_payload
        else:
            # Fallback: create a minimal container with refined inside
            container = {
                "name": refined_payload.get("name"),
                "submissionTime": refined_time,
                "refined": refined_payload,
            }
            all_consultations.append(container)

        save_consultation_data(all_consultations)

    return {
        "status": "Consultation data refined and saved successfully",
        "refined": refined_payload,
        "path": CONSULTATION_DATA_PATH
    }


# --- LLM prompts config (provider/base/model stay in .env) ---
LLM_PROMPTS_PATH = os.getenv(
    "LLM_PROMPTS_PATH",
    os.path.join(os.path.dirname(__file__), "config", "llm_prompts.json")
)

def load_llm_prompts() -> Dict[str, Any]:
    defaults = {
        "system_prompt": "",
        "update_prompt": "请基于最新问诊信息、AI预测与人工复检结果，生成简要且可操作的临床意见摘要。",
        "include_patient_context": True,
        "context_template": "患者：{patient_name}（{age}岁，{gender_zh}），检查时间：{examine_time}\n问诊要点：{consultation_summary}\nAI预测（左眼）：{ai_left_summary}\nAI预测（右眼）：{ai_right_summary}\n人工复核：{manual_summary}\n阈值：{threshold_summary}"
    }
    try:
        p = Path(LLM_PROMPTS_PATH)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {**defaults, **data}
    except Exception:
        pass
    return defaults

@app.get("/api/llm_config")
async def get_llm_config():
    cfg = load_llm_prompts()
    # Only prompts/flags; provider/base/model remain in .env
    return {
        "system_prompt": cfg.get("system_prompt", ""),
        "update_prompt": cfg.get("update_prompt", ""),
        "include_patient_context": cfg.get("include_patient_context", True),
        "has_context_template": bool(cfg.get("context_template"))
    }

def _zh_gender(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    s = str(g).strip().lower()
    if s == "male": return "男"
    if s == "female": return "女"
    if s == "other": return "其他"
    # 处理中文输入
    if s in ["男", "女性", "女", "其他"]:
        return s if s in ["男", "女", "其他"] else "女"
    return g

def _format_prob(p: Optional[float]) -> str:
    try:
        return f"{float(p):.2f}"
    except Exception:
        return "-"

def _summarize_consultation(cons: Optional[Dict[str, Any]]) -> str:
    if not cons or not isinstance(cons, dict):
        return "无"
    map_area = lambda a: "左眼" if a=="left" else ("右眼" if a=="right" else "双眼")
    areas = [map_area(a) for a in (cons.get("affectedArea") or [])]
    parts = []
    if areas: parts.append(f"受累部位：{'、'.join(areas)}")
    def eye_part(label, obj):
        if not obj: return None
        ms = obj.get("mainSymptom")
        om = obj.get("onsetMethod")
        ot = obj.get("onsetTime")
        ac = obj.get("accompanyingSymptoms")
        mh = obj.get("medicalHistory")
        segs = []
        if ms: segs.append(f"主要：{ms}")
        if om or ot: segs.append(f"起病：{(om or '')} {(' '+ot) if ot else ''}".strip())
        if ac: segs.append(f"伴随：{ac}")
        if mh: segs.append(f"病史：{mh}")
        return f"{label}（" + "；".join(segs) + "）" if segs else None
    for label, key in (("左眼","leftEye"),("右眼","rightEye"),("双眼","bothEyes")):
        x = eye_part(label, cons.get(key))
        if x: parts.append(x)
    return "；".join(parts) if parts else "无"

def _summarize_ai(patient: Optional[PatientData], eye_key: str) -> str:
    if not patient or not getattr(patient, "prediction_results", None):
        return "无"
    preds = getattr(patient.prediction_results, eye_key, None) if hasattr(patient.prediction_results, eye_key) else patient.prediction_results.get(eye_key)  # type: ignore
    if hasattr(preds, "dict"):
        preds = preds.dict()  # type: ignore
    if not isinstance(preds, dict):
        return "无"
    # Try to read thresholds
    th = getattr(patient, "prediction_thresholds", None)
    thd = th.dict() if hasattr(th, "dict") else (th or {})
    # Rank top-3
    items = []
    for k, v in preds.items():
        try:
            items.append((k, float(v), float(thd.get(k, 0.5))))
        except Exception:
            continue
    items.sort(key=lambda t: t[1], reverse=True)
    top = items[:3]
    if not top:
        return "无"
    parts = [f"{k} P={_format_prob(p)} (T={_format_prob(t)})" for k, p, t in top]
    return "，".join(parts)

def _summarize_thresholds(patient: Optional[PatientData]) -> str:
    """总结阈值设置"""
    if not patient:
        return "使用默认阈值"
    
    th = getattr(patient, "prediction_thresholds", None)
    if not th:
        return "AMD:0.30；青光眼:0.60；糖网:0.50；白内障:0.60"
    
    thd = th.dict() if hasattr(th, "dict") else th
    if not isinstance(thd, dict):
        return "AMD:0.30；青光眼:0.60；糖网:0.50；白内障:0.60"
    
    items = []
    disease_mapping = {
        "年龄相关性黄斑变性": "AMD",
        "青光眼": "青光眼", 
        "糖尿病性视网膜病变": "糖网",
        "白内障": "白内障"
    }
    
    for disease, short_name in disease_mapping.items():
        threshold = thd.get(disease, 0.5)
        items.append(f"{short_name}:{threshold:.2f}")
    
    return "；".join(items)

def _summarize_manual(patient_id: str) -> str:
    md = manual_diagnosis_storage.get(patient_id)
    if not md:
        return "无"
    try:
        md_dict = md.dict()
    except Exception:
        md_dict = {
            "manual_diagnosis": getattr(md, "manual_diagnosis", {}),
            "custom_diseases": getattr(md, "custom_diseases", {}),
            "diagnosis_notes": getattr(md, "diagnosis_notes", "")
        }
    pos = [k for k, v in (md_dict.get("manual_diagnosis") or {}).items() if v]
    extra = md_dict.get("custom_diseases") or {}
    notes = (md_dict.get("diagnosis_notes") or "").strip()
    segs = []
    if pos: segs.append("人工判断：" + "、".join(pos))
    if getattr(extra, "left_eye", None) or getattr(extra, "right_eye", None):
        # pydantic object support
        l = getattr(extra, "left_eye", "")
        r = getattr(extra, "right_eye", "")
        if str(l).strip(): segs.append(f"左眼附加：{l}")
        if str(r).strip(): segs.append(f"右眼附加：{r}")
    elif isinstance(extra, dict):
        if extra.get("left_eye"): segs.append(f"左眼附加：{extra.get('left_eye')}")
        if extra.get("right_eye"): segs.append(f"右眼附加：{extra.get('right_eye')}")
    if notes: segs.append(f"备注：{notes}")
    return "；".join(segs) if segs else "无"

def _summarize_ai_detailed(patient: Optional[PatientData], eye_key: str) -> str:
    """改进的AI预测总结，直接显示疾病可能性判断结果"""
    if not patient:
        return "无预测数据"
    
    # 获取预测结果和阈值
    prediction_results = getattr(patient, "prediction_results", None)
    prediction_thresholds = getattr(patient, "prediction_thresholds", None)
    diagnosis_results = getattr(patient, "diagnosis_results", None)
    
    if not prediction_results:
        # 如果没有prediction_results，尝试从原始数据重新计算
        logger.warning(f"No prediction_results for patient {patient.patient_id}, trying to use raw data")
        return _fallback_ai_summary(patient, eye_key)
    
    # 获取该眼的预测概率
    eye_probs = None
    if hasattr(prediction_results, eye_key):
        eye_probs = getattr(prediction_results, eye_key)
    elif isinstance(prediction_results, dict):
        eye_probs = prediction_results.get(eye_key)
    
    if hasattr(eye_probs, "dict"):
        eye_probs = eye_probs.dict()
    elif hasattr(eye_probs, "__dict__"):
        eye_probs = eye_probs.__dict__
    
    if not isinstance(eye_probs, dict) or not eye_probs:
        logger.warning(f"No eye_probs for {eye_key}, trying fallback")
        return _fallback_ai_summary(patient, eye_key)
    
    # 获取阈值
    thresholds = {}
    if prediction_thresholds:
        if hasattr(prediction_thresholds, "dict"):
            thresholds = prediction_thresholds.dict()
        elif hasattr(prediction_thresholds, "__dict__"):
            thresholds = prediction_thresholds.__dict__
        elif isinstance(prediction_thresholds, dict):
            thresholds = prediction_thresholds
    
    # 默认阈值
    default_thresholds = {
        "年龄相关性黄斑变性": 0.30,
        "青光眼": 0.60,
        "糖尿病性视网膜病变": 0.50,
        "白内障": 0.60,
        "其他黄斑病变": 0.50,
        "其他视网膜病": 0.50,
        "视网膜静脉阻塞（RVO）": 0.50,
        "视网膜动脉阻塞（RAO）": 0.50,
        "病理性近视": 0.50,
        "视网膜脱离（RD）": 0.50,
        "正常": 0.50
    }
    
    # 合并阈值，优先使用设定的阈值
    final_thresholds = {**default_thresholds, **thresholds}
    
    # 获取诊断结果
    eye_diagnosis = {}
    if diagnosis_results:
        if hasattr(diagnosis_results, eye_key):
            diag_data = getattr(diagnosis_results, eye_key)
            if hasattr(diag_data, "dict"):
                eye_diagnosis = diag_data.dict()
            elif hasattr(diag_data, "__dict__"):
                eye_diagnosis = diag_data.__dict__
            elif isinstance(diag_data, dict):
                eye_diagnosis = diag_data
        elif isinstance(diagnosis_results, dict):
            diag_data = diagnosis_results.get(eye_key, {})
            if hasattr(diag_data, "dict"):
                eye_diagnosis = diag_data.dict()
            elif isinstance(diag_data, dict):
                eye_diagnosis = diag_data
    
    # 按预测值降序排序，只显示前几个主要疾病
    try:
        sorted_diseases = sorted(eye_probs.items(), key=lambda x: float(x[1]), reverse=True)
    except Exception:
        logger.error(f"Error sorting diseases for {eye_key}: {eye_probs}")
        return "预测数据格式错误"
    
    items = []
    for disease, prob in sorted_diseases[:5]:  # 只显示前5个疾病
        try:
            prob_val = float(prob)
            threshold = float(final_thresholds.get(disease, 0.5))
            is_positive = eye_diagnosis.get(disease, prob_val > threshold)
            
            # 根据是否超过阈值显示不同格式
            if is_positive:
                items.append(f"{disease}: **阳性** (预测值{prob_val:.3f} > 阈值{threshold:.3f})")
            else:
                items.append(f"{disease}: 阴性 (预测值{prob_val:.3f} ≤ 阈值{threshold:.3f})")
        except Exception as e:
            logger.warning(f"Error processing disease {disease}: {e}")
            continue
    
    return "；".join(items) if items else "无有效预测"

def _fallback_ai_summary(patient: Optional[PatientData], eye_key: str) -> str:
    """当无法从patient对象获取预测结果时的fallback方法"""
    if not patient:
        return "无预测数据"
    
    patient_id = getattr(patient, "patient_id", None)
    if not patient_id:
        return "无患者ID"
    
    # 尝试从原始缓存数据中获取
    try:
        raw_cache = raw_probs_cache.get(patient_id)
        if not raw_cache:
            return "无原始预测缓存"
        
        # 根据eye_key确定查找的图像类型
        cfp_type = "左眼CFP" if eye_key == "left_eye" else "右眼CFP"
        
        # 获取该类型的最新图像数据
        raw_by_type = raw_cache.get("by_type", {})
        images_of_type = raw_by_type.get(cfp_type, [])
        
        if not images_of_type:
            return f"无{cfp_type}数据"
        
        # 使用最新的图像预测结果
        latest_image = images_of_type[0]  # 已经按时间排序
        probs = latest_image.get("probs", {})
        
        if not probs:
            return "无概率数据"
        
        # 疾病名称映射
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
        
        # 转换疾病名称并排序
        mapped_probs = {}
        for orig_name, prob in probs.items():
            mapped_name = diagnosis_mapping.get(orig_name, orig_name)
            mapped_probs[mapped_name] = float(prob)
        
        sorted_diseases = sorted(mapped_probs.items(), key=lambda x: x[1], reverse=True)
        
        # 使用默认阈值
        default_thresholds = {
            "AMD": 0.30,
            "青光眼": 0.60,
            "糖网": 0.50,
            "白内障": 0.60,
        }
        
        items = []
        for disease, prob in sorted_diseases[:5]:
            threshold = default_thresholds.get(disease, 0.5)
            is_positive = prob > threshold
            
            if is_positive:
                items.append(f"{disease}: **阳性** (预测值{prob:.3f} > 阈值{threshold:.3f})")
            else:
                items.append(f"{disease}: 阴性 (预测值{prob:.3f} ≤ 阈值{threshold:.3f})")
        
        return "；".join(items) if items else "无有效预测"
        
    except Exception as e:
        logger.error(f"Fallback AI summary failed: {e}")
        return f"获取预测数据失败: {str(e)}"

def _build_context_placeholders(ris_exam_id: Optional[str]) -> Dict[str, str]:
    p: Optional[PatientData] = patients_data_cache.get(ris_exam_id or "")
    
    # 从患者数据中获取基本信息
    name = getattr(p, "name", None) if p else None
    age = getattr(p, "age", None) if p and hasattr(p, "age") else None
    gender = getattr(p, "gender", None) if p and hasattr(p, "gender") else None
    examine_time = getattr(p, "examine_time", None) if p else None
    
    # 如果患者数据中没有年龄和性别，尝试从推理结果文件中获取
    if not age or not gender or not name:
        try:
            with open(RAW_JSON_PATH, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if ris_exam_id and ris_exam_id in raw_data:
                    patient_info = raw_data[ris_exam_id]
                    if not age and "age" in patient_info:
                        age = patient_info["age"]
                    if not gender and "gender" in patient_info:
                        gender = patient_info["gender"]
                    if not name and "name" in patient_info:
                        name = patient_info["name"]
                    if not examine_time and "examineTime" in patient_info:
                        examine_time = patient_info["examineTime"]
        except Exception as e:
            logger.warning(f"Failed to load additional patient info: {str(e)}")

    # 从问诊数据中获取年龄和性别信息（优先级更高，因为这是用户填写的最新信息）
    with consultation_data_lock:
        cons_all = load_consultation_data()
        ctx = _get_patient_context(ris_exam_id or "")
        idx = _find_best_matching_index(cons_all, ctx.get("name") or name, ctx.get("examineTime") or examine_time)
        cons = None
        
        if idx is not None:
            base = cons_all[idx]
            refined = base.get("refined")
            # 优先使用refined数据，否则使用原始数据
            if isinstance(refined, dict):
                cons = refined
            else:
                cons = base
        
        # 从问诊数据中提取年龄和性别，如果存在的话
        if cons:
            if not age and cons.get("age"):
                age = cons.get("age")
            if not gender and cons.get("gender"):
                gender = cons.get("gender")
            if not name and cons.get("name"):
                name = cons.get("name")

    gender_zh = _zh_gender(gender)

    consultation_summary = _summarize_consultation(cons or {})
    # 使用改进的AI预测总结
    ai_left_summary = _summarize_ai_detailed(p, "left_eye")
    ai_right_summary = _summarize_ai_detailed(p, "right_eye")
    threshold_summary = _summarize_thresholds(p)
    manual_summary = _summarize_manual(ris_exam_id or "")

    return {
        "patient_name": name or "患者",
        "age": str(age or ""),
        "gender_zh": gender_zh or "",
        "examine_time": examine_time or "",
        "consultation_summary": consultation_summary,
        "ai_left_summary": ai_left_summary,
        "ai_right_summary": ai_right_summary,
        "manual_summary": manual_summary,
        "threshold_summary": threshold_summary,
    }

def _fill_template(template: str, mapping: Dict[str, str]) -> str:
    # Simple placeholder replace without raising on missing keys
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v or "")
    return out


def load_inference_map() -> Dict[str, Any]:
    if not os.path.exists(INFERENCE_RESULTS_PATH):
        return {}
    try:
        with open(INFERENCE_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_inference_map(d: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(INFERENCE_RESULTS_PATH), exist_ok=True)
    with open(INFERENCE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def persist_llm_context(
    patient_id: str,
    *,
    system_prompt: str,
    context_text: str,
    user_prompt: str,
    assistant_text: str,
    reset: bool,
) -> Dict[str, Any]:
    with _infer_lock:
        d = load_inference_map()
        rec = d.get(patient_id)
        if not isinstance(rec, dict):
            rec = {}
        # fresh context when reset; else append to existing
        base_ctx = {} if reset or not isinstance(rec.get("llm_context"), dict) else rec["llm_context"]
        history: List[Dict[str, str]] = []

        if reset:
            # put current system/context at top
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})
            if context_text.strip():
                history.append({"role": "system", "content": f"[患者上下文]\n{context_text}"})
        else:
            old_hist = base_ctx.get("history")
            if isinstance(old_hist, list):
                history.extend([x for x in old_hist if isinstance(x, dict) and "role" in x and "content" in x])

        # append the new turn
        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": assistant_text})

        rec["llm_context"] = {
            "history": history,
            "last_updated": datetime.datetime.now().astimezone().isoformat(),
            "system_prompt": system_prompt,
            "context_text": context_text,
            "version": 1,
        }
        d[patient_id] = rec
        save_inference_map(d)
        return rec["llm_context"]

@app.get("/api/llm_context/{patient_id}")
async def get_llm_context(patient_id: str):
    with _infer_lock:
        d = load_inference_map()
        rec = d.get(patient_id) or {}
        ctx = rec.get("llm_context") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        return {"status": "ok", "patient_id": patient_id, "llm_context": ctx, "path": INFERENCE_RESULTS_PATH}

@app.delete("/api/llm_context/{patient_id}")
async def delete_llm_context(patient_id: str):
    with _infer_lock:
        d = load_inference_map()
        rec = d.get(patient_id)
        if isinstance(rec, dict) and "llm_context" in rec:
            rec.pop("llm_context", None)
            d[patient_id] = rec
            save_inference_map(d)
    return {"status": "deleted", "patient_id": patient_id}

# --- LLM chat (streaming) ---
class LLMChatRequest(BaseModel):
    patient_id: Optional[str] = None
    messages: List[Dict[str, str]]  # [{role:'user'|'assistant'|'system', content:str}]
    persist: Optional[bool] = True     # persist to inference_results.json
    reset: Optional[bool] = False      # overwrite previous on this request


def _llm_env():
    # OLLAMA本地部署配置
    return {
        "base": os.getenv("LLM_API_BASE", "http://10.138.6.3:50201"),
        "model": os.getenv("LLM_MODEL", "DeepSeek-3.1:latest"),
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
    }

def format_llm_output_for_frontend(text: str) -> str:
    """
    将LLM输出转换为前端友好的格式
    支持基本的Markdown到HTML转换和换行处理
    """
    if not text:
        return text
    
    # 处理换行符
    text = text.replace('\n', '<br>')
    
    # 处理加粗 **text** -> <strong>text</strong>
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # 处理斜体 *text* -> <em>text</em>  
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # 处理标题 ## text -> <h3>text</h3>
    text = re.sub(r'^### (.*?)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    
    # 处理列表项 - text -> <li>text</li>
    lines = text.split('<br>')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('- ') or line.startswith('• '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{line[2:].strip()}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(line)
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return '<br>'.join(formatted_lines)

@app.post("/api/llm_chat_stream")
async def llm_chat_stream(req: LLMChatRequest):
    env = _llm_env()
    cfg = load_llm_prompts()
    
    # 构建消息列表
    messages_to_send: List[Dict[str, str]] = []
    sys_prompt = cfg.get("system_prompt", "") or ""
    
    # 检查是否是首次更新推理请求（只有一个user消息且内容是update_prompt）
    is_initial_update = (
        len(req.messages or []) == 1 and 
        req.messages[0].get("role") == "user" and
        req.messages[0].get("content", "").strip() == cfg.get("update_prompt", "").strip()
    )
    
    if is_initial_update:
        # 对于初次更新推理请求，将所有信息合并为一个user prompt
        context_text = ""
        if cfg.get("include_patient_context", True):
            placeholders = _build_context_placeholders(req.patient_id)
            context_text = _fill_template(cfg.get("context_template", ""), placeholders)
        
        # 构建完整的单个user prompt
        combined_prompt_parts = []
        
        # 添加系统角色说明
        if sys_prompt:
            combined_prompt_parts.append(f"角色要求：{sys_prompt}")
        
        # 添加患者上下文
        if context_text.strip():
            combined_prompt_parts.append(f"患者信息：\n{context_text}")
        
        # 添加任务要求
        update_prompt = cfg.get("update_prompt", "基于提供的患者信息，直接给出临床分析和建议，分为两部分：1）临床推理过程 2）意见摘要")
        combined_prompt_parts.append(f"任务要求：{update_prompt}")
        
        # 合并为单个user消息
        combined_content = "\n\n".join(combined_prompt_parts)
        messages_to_send.append({"role": "user", "content": combined_content})
        
    else:
        # 对于后续对话，也使用user方式传递上下文信息
        context_text = ""
        # 构建上下文信息时使用ris_exam_id
        if cfg.get("include_patient_context", True):
            placeholders = _build_context_placeholders(req.patient_id)  # 这里req.patient_id实际上是ris_exam_id
            context_text = _fill_template(cfg.get("context_template", ""), placeholders)
        
        # 构建包含上下文的user消息作为第一条消息
        context_prompt_parts = []
        
        # 添加系统角色说明
        if sys_prompt:
            context_prompt_parts.append(f"角色要求：{sys_prompt}")
        
        # 添加患者上下文
        if context_text.strip():
            context_prompt_parts.append(f"患者信息：\n{context_text}")
        
        # 如果有上下文信息，将其作为第一条user消息
        if context_prompt_parts:
            context_content = "\n\n".join(context_prompt_parts)
            messages_to_send.append({"role": "user", "content": context_content})
            # 添加一个简短的assistant确认消息
            messages_to_send.append({"role": "assistant", "content": "已了解患者信息和角色要求。"})

        # 加载历史对话记录
        if req.patient_id:
            try:
                with _infer_lock:
                    d = load_inference_map()
                    rec = d.get(req.patient_id, {})
                    llm_context = rec.get("llm_context", {})
                    history = llm_context.get("history", [])
                    
                    if isinstance(history, list) and history:
                        # 只保留user/assistant对话，跳过任何system消息
                        conversation_history = [
                            msg for msg in history 
                            if isinstance(msg, dict) and 
                            msg.get("role") in ["user", "assistant"] and 
                            msg.get("content", "").strip()
                        ]
                        
                        # 添加历史对话
                        messages_to_send.extend(conversation_history)
                        logger.info(f"Loaded {len(conversation_history)} historical messages for patient {req.patient_id}")
                    else:
                        logger.info(f"No conversation history found for patient {req.patient_id}")
            except Exception as e:
                logger.warning(f"Failed to load conversation history for patient {req.patient_id}: {e}")

        # 添加当前请求的新消息
        current_messages = req.messages or []
        # 只添加user消息，因为assistant消息应该来自历史记录
        new_user_messages = [
            msg for msg in current_messages 
            if msg.get("role") == "user"
        ]
        messages_to_send.extend(new_user_messages)

    # === 添加DEBUG日志 ===
    logger.info("=" * 80)
    logger.info(f"LLM Chat Request for Patient: {req.patient_id} ({'INITIAL_UPDATE' if is_initial_update else 'FOLLOW_UP'})")
    logger.info("=" * 80)
    logger.info(f"Environment: {env}")
    logger.info(f"Messages count: {len(messages_to_send)}")
    
    # for i, msg in enumerate(messages_to_send):
    #     role = msg.get("role", "unknown")
    #     content = msg.get("content", "")
    #     logger.info(f"Message [{i+1}] ({role}):")
    #     logger.info("-" * 40)
    #     if len(content) > 500:
    #         logger.info(f"{content[:500]}...\n[TRUNCATED - Total length: {len(content)} chars]")
    #     else:
    #         logger.info(content)
    #     logger.info("-" * 40)
    
    # logger.info("=" * 80)

    # 获取最后一个用户消息用于持久化
    last_user = ""
    if is_initial_update:
        last_user = cfg.get("update_prompt", "")
    else:
        # 从当前请求中获取最新的用户消息
        for m in reversed(req.messages or []):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

    # 缓存助手回复内容
    assistant_buf: List[str] = []

    async def ollama_stream():
        # OLLAMA API端点
        url = f"{env['base'].rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": env["model"],
            "messages": messages_to_send,
            "stream": True
        }
        
        logger.info(f"Sending request to OLLAMA: {url}")
        logger.info(f"Payload model: {payload['model']}")
        logger.info(f"Payload stream: {payload['stream']}")
        logger.info(f"Payload messages: {payload['messages']}")
        logger.info(f"Request type: {'COMBINED_USER_PROMPT' if is_initial_update else 'USER_CONTEXT_PROMPT'}")
        
        timeout = httpx.Timeout(120.0, read=120.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as r:
                    logger.info(f"OLLAMA Response Status: {r.status_code}")
                    
                    if r.status_code != 200:
                        error_text = await r.atext()
                        logger.error(f"OLLAMA Error Response: {error_text}")
                        yield f"\n[OLLAMA error] HTTP {r.status_code}: {error_text}\n"
                        return
                    
                    response_chunks = 0
                    async for raw_line in r.aiter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        
                        response_chunks += 1
                        if response_chunks == 1:
                            logger.info("Started receiving OLLAMA streaming response...")
                        
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON line: {line[:100]}...")
                            continue
                        
                        if not isinstance(obj, dict):
                            continue
                            
                        if "error" in obj:
                            logger.error(f"OLLAMA Stream Error: {obj['error']}")
                            yield f"\n[OLLAMA error] {obj['error']}\n"
                            break
                        
                        message = obj.get("message", {})
                        content = message.get("content", "")
                        
                        if content:
                            assistant_buf.append(content)
                            yield content
                        
                        if obj.get("done", False):
                            logger.info(f"OLLAMA response completed. Total chunks: {response_chunks}")
                            logger.info(f"Assistant response length: {len(''.join(assistant_buf))} chars")
                            break
                            
        except httpx.TimeoutException:
            logger.error("OLLAMA request timeout")
            yield "\n[OLLAMA **错误**] 请求超时，请检查OLLAMA服务是否正常运行\n"
        except httpx.ConnectError:
            logger.error(f"Cannot connect to OLLAMA service: {env['base']}")
            yield f"\n[OLLAMA **错误**] 无法连接到OLLAMA服务 {env['base']}\n"
        except Exception as e:
            logger.error(f"OLLAMA request failed: {str(e)}")
            yield f"\n[OLLAMA **错误**] {str(e)}\n"
        finally:
            if req.persist and req.patient_id and assistant_buf:
                full_response = "".join(assistant_buf)
                logger.info(f"Persisting LLM context for patient {req.patient_id}")
                
                persist_llm_context(
                    req.patient_id,
                    system_prompt="",  # 不再使用system_prompt持久化
                    context_text="",   # 不再单独存储context_text
                    user_prompt=last_user,
                    assistant_text=full_response,
                    reset=bool(req.reset),
                )

    return StreamingResponse(ollama_stream(), media_type="text/plain; charset=utf-8")

# 同样更新格式化版本
@app.post("/api/llm_chat_formatted")
async def llm_chat_formatted(req: LLMChatRequest):
    """
    提供格式化的LLM响应（非流式），支持HTML格式，统一使用user方式传递上下文
    """
    env = _llm_env()
    cfg = load_llm_prompts()
    
    # 构建消息（统一使用user方式）
    messages_to_send: List[Dict[str, str]] = []
    sys_prompt = cfg.get("system_prompt", "") or ""
    
    # 构建上下文信息
    context_text = ""
    if cfg.get("include_patient_context", True):
        placeholders = _build_context_placeholders(req.patient_id)
        context_text = _fill_template(cfg.get("context_template", ""), placeholders)
    
    # 构建包含上下文的user消息
    context_prompt_parts = []
    
    # 添加系统角色说明
    if sys_prompt:
        context_prompt_parts.append(f"角色要求：{sys_prompt}")
    
    # 添加患者上下文
    if context_text.strip():
        context_prompt_parts.append(f"患者信息：\n{context_text}")
    
    # 如果有上下文信息，将其作为第一条user消息
    if context_prompt_parts:
        context_content = "\n\n".join(context_prompt_parts)
        messages_to_send.append({"role": "user", "content": context_content})
        # 添加一个简短的assistant确认消息
        messages_to_send.append({"role": "assistant", "content": "已了解患者信息和角色要求。"})

    # 添加当前请求的消息
    messages_to_send.extend(req.messages or [])
    
    # === 添加DEBUG日志 ===
    logger.info("=" * 80)
    logger.info(f"LLM Chat Formatted Request for Patient: {req.patient_id}")
    logger.info("=" * 80)
    logger.info(f"Environment: {env}")
    logger.info(f"Messages count: {len(messages_to_send)}")
    
    for i, msg in enumerate(messages_to_send):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.info(f"Message [{i+1}] ({role}):")
        logger.info("-" * 40)
        if len(content) > 500:
            logger.info(f"{content[:500]}...\n[TRUNCATED - Total length: {len(content)} chars]")
        else:
            logger.info(content)
        logger.info("-" * 40)
    
    logger.info("=" * 80)
    
    last_user = ""
    for m in reversed(req.messages or []):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    # 非流式请求
    url = f"{env['base'].rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": env["model"],
        "messages": messages_to_send,
        "stream": False
    }
    
    logger.info(f"Sending formatted request to OLLAMA: {url}")
    logger.info(f"Payload model: {payload['model']}")
    logger.info(f"Payload stream: {payload['stream']}")
    
    timeout = httpx.Timeout(120.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            logger.info(f"OLLAMA Formatted Response Status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"OLLAMA Formatted Error: {error_text}")
                return {"error": f"HTTP {response.status_code}: {error_text}"}
            
            result = response.json()
            if "error" in result:
                logger.error(f"OLLAMA Formatted Response Error: {result['error']}")
                return {"error": result["error"]}
            
            message = result.get("message", {})
            content = message.get("content", "")
            
            if content:
                logger.info(f"OLLAMA Formatted Response Length: {len(content)} chars")
                logger.info(f"OLLAMA Formatted Response Preview: {content[:200]}...")
                
                # 格式化输出
                formatted_content = format_llm_output_for_frontend(content)
                
                # 持久化
                if req.persist and req.patient_id:
                    logger.info(f"Persisting formatted LLM context for patient {req.patient_id}")
                    persist_llm_context(
                        req.patient_id,
                        system_prompt="",  # 不再使用system_prompt
                        context_text="",   # 不再单独存储context_text
                        user_prompt=last_user,
                        assistant_text=content,
                        reset=bool(req.reset),
                    )
                
                return {
                    "content": content,
                    "formatted_content": formatted_content,
                    "status": "success"
                }
            else:
                logger.warning("Empty response from OLLAMA formatted endpoint")
                return {"error": "Empty response from OLLAMA"}
            
    except Exception as e:
        logger.error(f"OLLAMA formatted request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
