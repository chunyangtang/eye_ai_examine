# main.py

import os
import json
import asyncio
import datetime
from dotenv import load_dotenv
import httpx
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel
from pathlib import Path


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from datatype import ImageInfo, EyeDiagnosis, PatientData, SubmitDiagnosisRequest, EyePrediction, EyePredictionThresholds, ManualDiagnosisData, UpdateSelectionRequest, AlterThresholdRequest, CustomDiseases
from patientdataio import load_batch_patient_data, create_batch_dummy_patient_data, load_patient_from_record

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
manual_diagnosis_file_lock = threading.Lock()
def _load_manual_diagnosis_store() -> Dict[str, Any]:
    """Read persisted manual diagnosis information keyed by patient id."""
    if not os.path.exists(EXAMINE_RESULTS_PATH):
        return {}
    try:
        with open(EXAMINE_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.warning("Manual diagnosis file %s did not contain a dict; resetting", EXAMINE_RESULTS_PATH)
    except json.JSONDecodeError:
        logger.error("Manual diagnosis file %s is not valid JSON; ignoring", EXAMINE_RESULTS_PATH)
    except Exception as exc:
        logger.error("Failed to load manual diagnosis file %s: %s", EXAMINE_RESULTS_PATH, exc)
    return {}


def _save_manual_diagnosis_store(serializable: Dict[str, Any]) -> None:
    """Persist manual diagnosis storage to JSON file."""
    os.makedirs(os.path.dirname(EXAMINE_RESULTS_PATH), exist_ok=True)
    tmp_path = EXAMINE_RESULTS_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, EXAMINE_RESULTS_PATH)
    except Exception as exc:
        logger.error("Failed to save manual diagnosis data to %s: %s", EXAMINE_RESULTS_PATH, exc)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _preload_manual_diagnoses() -> None:
    """Populate in-memory manual diagnosis cache from persisted JSON file."""
    try:
        persisted = _load_manual_diagnosis_store()
    except Exception as exc:
        logger.error("Unable to preload manual diagnoses: %s", exc)
        persisted = {}

    if not isinstance(persisted, dict):
        logger.warning("Preloaded manual diagnosis data is not a dict; skipping preload")
        return

    for patient_id, payload in persisted.items():
        if not isinstance(payload, dict):
            continue
        try:
            manual_diagnosis_storage[patient_id] = ManualDiagnosisData(**payload)
        except Exception:
            try:
                from datatype import CustomDiseases
                manual_diagnosis_storage[patient_id] = ManualDiagnosisData(
                    manual_diagnosis=payload.get("manual_diagnosis", {}),
                    custom_diseases=payload.get("custom_diseases") or CustomDiseases(),
                    diagnosis_notes=payload.get("diagnosis_notes", "")
                )
            except Exception as inner_exc:
                logger.error("Failed to deserialize manual diagnosis for %s: %s", patient_id, inner_exc)


_preload_manual_diagnoses()

# Limit the amount of conversation history we replay to the model each turn
MAX_CHAT_HISTORY_MESSAGES = int(os.getenv("LLM_MAX_HISTORY_MESSAGES", "20"))

# Ollama connection resilience tuning
OLLAMA_MAX_RETRIES = max(1, int(os.getenv("LLM_MAX_RETRIES", "2")))
OLLAMA_RETRY_DELAY_SECONDS = max(0.0, float(os.getenv("LLM_RETRY_DELAY_SECONDS", "1.0")))
OLLAMA_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "30m")

# Get data paths from environment variables
RAW_JSON_ROOT_ENV = os.getenv("RAW_JSON_PATH", "../data")
if not os.path.isabs(RAW_JSON_ROOT_ENV):
    RAW_JSON_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), RAW_JSON_ROOT_ENV))
else:
    RAW_JSON_ROOT = RAW_JSON_ROOT_ENV

RAW_JSON_MAX_DAYS = max(1, int(os.getenv("RAW_JSON_MAX_DAYS", "21")))

EXAMINE_RESULTS_PATH_ENV = os.getenv("EXAMINE_RESULTS_PATH", "../data/examine_results.json")
if not os.path.isabs(EXAMINE_RESULTS_PATH_ENV):
    EXAMINE_RESULTS_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), EXAMINE_RESULTS_PATH_ENV)
    )
else:
    EXAMINE_RESULTS_PATH = EXAMINE_RESULTS_PATH_ENV

INFERENCE_RESULTS_PATH = EXAMINE_RESULTS_PATH

# --- Inference data aggregation (latest dated folders) ---
_inference_cache_lock = threading.Lock()
_inference_patient_records: Dict[str, Dict[str, Any]] = {}
_inference_patient_base_dirs: Dict[str, str] = {}
_inference_loaded_files: List[str] = []


def _resolve_inference_sources() -> List[tuple]:
    """Return a list of (json_path, base_dir, label) sorted by newest first."""
    if os.path.isfile(RAW_JSON_ROOT):
        return [(RAW_JSON_ROOT, os.path.dirname(RAW_JSON_ROOT), os.path.basename(RAW_JSON_ROOT))]

    if not os.path.isdir(RAW_JSON_ROOT):
        logger.error("RAW_JSON_ROOT does not exist or is not accessible: %s", RAW_JSON_ROOT)
        return []

    entries: List[tuple] = []
    try:
        for name in os.listdir(RAW_JSON_ROOT):
            full_dir = os.path.join(RAW_JSON_ROOT, name)
            if not os.path.isdir(full_dir):
                continue
            label = name.strip()
            if len(label) == 8 and label.isdigit():
                candidate = os.path.join(full_dir, "inference_results.json")
                if os.path.isfile(candidate):
                    entries.append((candidate, full_dir, label))
    except Exception as exc:
        logger.error("Failed to enumerate inference directories under %s: %s", RAW_JSON_ROOT, exc)
        return []

    if not entries:
        # Fallback: look for inference_results.json directly under root for backwards compatibility
        fallback = os.path.join(RAW_JSON_ROOT, "inference_results.json")
        if os.path.isfile(fallback):
            return [(fallback, RAW_JSON_ROOT, os.path.basename(RAW_JSON_ROOT))]
        logger.warning("No dated inference folders found under %s", RAW_JSON_ROOT)
        return []

    # Sort by label descending (newest date first)
    entries.sort(key=lambda item: item[2], reverse=True)
    return entries[:RAW_JSON_MAX_DAYS]


def _refresh_inference_cache(force: bool = False) -> None:
    global _inference_patient_records, _inference_patient_base_dirs, _inference_loaded_files
    sources = _resolve_inference_sources()
    file_keys = [path for path, _, _ in sources]

    with _inference_cache_lock:
        if not force and file_keys == _inference_loaded_files and _inference_patient_records:
            return

        combined: Dict[str, Dict[str, Any]] = {}
        base_dirs: Dict[str, str] = {}

        for path, base_dir, label in sources:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.error("Failed to load inference file %s: %s", path, exc)
                continue

            if not isinstance(data, dict):
                logger.warning("Inference file %s did not contain a JSON object", path)
                continue

            for pid, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                combined[pid] = payload
                base_dirs[pid] = base_dir

        _inference_patient_records = combined
        _inference_patient_base_dirs = base_dirs
        _inference_loaded_files = file_keys

        logger.info(
            "Loaded %s inference file(s) covering %s patient(s)",
            len(file_keys),
            len(combined)
        )


def get_inference_patient_ids() -> List[str]:
    _refresh_inference_cache()
    with _inference_cache_lock:
        return list(_inference_patient_records.keys())


def get_inference_record(patient_id: str) -> Optional[Dict[str, Any]]:
    _refresh_inference_cache()
    with _inference_cache_lock:
        record = _inference_patient_records.get(patient_id)
        if not record:
            return None
        # Return a shallow copy to prevent accidental mutation
        return dict(record)


def get_inference_base_dir(patient_id: str) -> Optional[str]:
    _refresh_inference_cache()
    with _inference_cache_lock:
        return _inference_patient_base_dirs.get(patient_id)


def get_full_inference_map() -> Dict[str, Dict[str, Any]]:
    _refresh_inference_cache()
    with _inference_cache_lock:
        return {pid: dict(payload) for pid, payload in _inference_patient_records.items()}

# Path to questionnaire data from the other project
CONSULTATION_DATA_PATH_ENV = os.getenv("CONSULTATION_DATA_PATH")
if CONSULTATION_DATA_PATH_ENV:
    # If path is relative, make it relative to the backend directory
    if not os.path.isabs(CONSULTATION_DATA_PATH_ENV):
        CONSULTATION_DATA_PATH = os.path.normpath(
            os.path.join(os.path.dirname(__file__), CONSULTATION_DATA_PATH_ENV)
        )
    else:
        CONSULTATION_DATA_PATH = CONSULTATION_DATA_PATH_ENV
else:
    # Default path
    CONSULTATION_DATA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../eye_ai_consultation/data/questionnaire_data.json")
    )

# Lock for thread-safe access to cached data
patient_data_lock = threading.Lock()
consultation_data_lock = threading.Lock()

# --- New helpers for RAW v2 structure ---
def _extract_image_type(img: Dict[str, Any]) -> Optional[str]:
    """
    Map RAW v2 eye_classification.class to internal types:
      Left fundus -> 左眼CFP
      Right fundus -> 右眼CFP
      Left outer eye -> 左眼外眼照
      Right outer eye -> 右眼外眼照
    """
    try:
        cls = str(img.get("eye_classification", {}).get("class", "")).strip().lower()
    except Exception:
        cls = ""
    if cls == "left fundus":
        return "左眼CFP"
    if cls == "right fundus":
        return "右眼CFP"
    if cls == "left outer eye":
        return "左眼外眼照"
    if cls == "right outer eye":
        return "右眼外眼照"
    return None

def _extract_disease_probs(img: Dict[str, Any]) -> Dict[str, float]:
    """
    Use fundus_classification.diseases as disease probs if present.
    If missing or invalid, return {}.
    """
    try:
        diseases = img.get("fundus_classification", {}).get("diseases", {})
        if isinstance(diseases, dict):
            # ensure float
            return {str(k): float(v) for k, v in diseases.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}

def _parse_ts_from_imgpath(p: str) -> int:
    """
    Keep the original timestamp/digits extraction to sort 'latest' images.
    """
    try:
        base = os.path.basename(p or "")
        name, _ = os.path.splitext(base)
        parts = (name or "").split('_')
        cand = parts[-1] if parts else name
        digits = ''.join(ch for ch in (cand or "") if ch.isdigit())
        return int(digits) if digits else 0
    except Exception:
        return 0

def _warm_raw_cache_from_raw_json(ris_exam_id: str) -> None:
    """Build raw_probs_cache[ris_exam_id] from aggregated inference records."""
    record = get_inference_record(ris_exam_id)
    if record is None:
        raise KeyError(f"Patient {ris_exam_id} not found in inference cache")

    images = record.get("images", [])
    if not isinstance(images, list):
        images = []

    raw_by_type: Dict[str, list] = {}
    raw_by_id: Dict[str, Dict] = {}

    for img in images:
        if not isinstance(img, dict):
            continue
        img_type = _extract_image_type(img)
        if not img_type:
            continue

        probs = _extract_disease_probs(img)

        # cache by id
        img_id = f"img_{ris_exam_id}_{img.get('img_path', '')}"
        raw_by_id[img_id] = probs

        # push into type list; augment with 'probs' for fallback users
        img_copy = dict(img)
        img_copy["probs"] = probs
        raw_by_type.setdefault(img_type, []).append(img_copy)

    # sort lists latest-first by parsed digits from img_path
    for k in list(raw_by_type.keys()):
        raw_by_type[k] = sorted(
            raw_by_type[k],
            key=lambda im: _parse_ts_from_imgpath(im.get("img_path", "")),
            reverse=True
        )

    raw_probs_cache[ris_exam_id] = {"by_type": raw_by_type, "by_id": raw_by_id}

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
                    _warm_raw_cache_from_raw_json(ris_exam_id)
                except Exception:
                    pass
            return patients_data_cache[ris_exam_id]
        
        # Load from data source
        try:
            print(f"Loading patient {ris_exam_id} from data source...")
            record = get_inference_record(ris_exam_id)
            if record is None:
                raise KeyError
            base_dir = get_inference_base_dir(ris_exam_id) or RAW_JSON_ROOT
            patient_data = load_patient_from_record(ris_exam_id, record, base_dir)
            # Cache the loaded data
            patients_data_cache[ris_exam_id] = patient_data
            # Build raw prob cache for this patient (warm for reselection)
            try:
                _warm_raw_cache_from_raw_json(ris_exam_id)
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
        ids = get_inference_patient_ids()
        print(f"Found {len(ids)} available patients")
        return {"patient_ids": ids}
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
            # Convert raw dict to ManualEyeDiagnosis objects
            processed_manual_diagnosis = {}
            if request.manual_diagnosis:
                from datatype import ManualEyeDiagnosis
                for eye_key, eye_data in request.manual_diagnosis.items():
                    if isinstance(eye_data, dict):
                        # Create ManualEyeDiagnosis from dict, filling missing fields with False
                        processed_manual_diagnosis[eye_key] = ManualEyeDiagnosis(**eye_data)
                    else:
                        processed_manual_diagnosis[eye_key] = eye_data
            
            manual_data = ManualDiagnosisData(
                manual_diagnosis=processed_manual_diagnosis,
                custom_diseases=request.custom_diseases or CustomDiseases(),
                diagnosis_notes=request.diagnosis_notes or ""
            )
            
            manual_diagnosis_storage[request.patient_id] = manual_data
            with manual_diagnosis_file_lock:
                persisted = _load_manual_diagnosis_store()
                try:
                    payload = manual_data.dict()
                except Exception:
                    payload = {
                        "manual_diagnosis": manual_data.manual_diagnosis,
                        "custom_diseases": getattr(manual_data, "custom_diseases", {}),
                        "diagnosis_notes": getattr(manual_data, "diagnosis_notes", "")
                    }
                persisted[request.patient_id] = payload
                _save_manual_diagnosis_store(persisted)
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
            # Try to warm the cache quickly from disk (RAW v2)
            try:
                _warm_raw_cache_from_raw_json(request.patient_id)
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


@app.post("/api/alter_threshold")
async def alter_threshold(request: AlterThresholdRequest):
    """
    Cycles between different threshold sets for a patient and recomputes diagnosis results.
    """
    with patient_data_lock:
        patient = patients_data_cache.get(request.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found in cache.")
        
        # Cycle to the next threshold set (0 -> 1, 1 -> 0)
        current_set = getattr(patient, 'active_threshold_set', 0)
        next_set = 1 if current_set == 0 else 0
        
        # Get the new threshold values
        new_thresholds = EyePredictionThresholds.get_threshold_set(next_set)
        
        # Update patient's threshold set and active set index
        patient.prediction_thresholds = new_thresholds
        patient.active_threshold_set = next_set
        
        # Recompute diagnosis results with new thresholds
        diagnosis_results = {}
        for eye in ["left_eye", "right_eye"]:
            if eye not in patient.prediction_results:
                continue
                
            diag = {}
            prediction = patient.prediction_results[eye]
            
            for disease in ["青光眼", "糖网", "AMD", "病理性近视", "RVO", "RAO", "视网膜脱离", "其它视网膜病", "其它黄斑病变", "白内障", "正常"]:
                threshold = getattr(new_thresholds, disease, 0.5)
                prob = getattr(prediction, disease, 0.0)
                diag[disease] = prob >= threshold
            
            diagnosis_results[eye] = EyeDiagnosis(**diag)
        
        # Update patient's diagnosis results
        patient.diagnosis_results = diagnosis_results
        
        return {
            "status": "Threshold altered successfully",
            "active_threshold_set": next_set,
            "new_thresholds": new_thresholds.dict(),
            "updated_diagnosis_results": diagnosis_results
        }


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


def _find_consultation_by_exam_id(all_data: List[Dict[str, Any]], ris_exam_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Locate a consultation entry (refined preferred) linked to a specific ris_exam_id.

    Returns (matched_consultation, base_entry) where base_entry is the original list item
    containing the consultation (useful for fallback metadata such as submissionTime).
    """
    if not ris_exam_id:
        return None, None

    for entry in all_data or []:
        if not isinstance(entry, dict):
            continue

        refined = entry.get("refined")
        if isinstance(refined, dict) and refined.get("_exam_ris_exam_id") == ris_exam_id:
            return refined, entry

        if entry.get("_exam_ris_exam_id") == ris_exam_id:
            return entry, entry

    return None, None

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
    Returns {'name': str|None, 'examineTime': str|None} from cache, else from aggregated inference data.
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

    # Fallback to aggregated inference cache
    try:
        record = get_inference_record(ris_exam_id)
        if isinstance(record, dict):
            ctx["name"] = record.get("name")
            ctx["examineTime"] = record.get("examineTime")
    except Exception:
        pass

    if not ctx["name"] or not ctx["examineTime"]:
        try:
            with consultation_data_lock:
                all_cons = load_consultation_data()
                matched, base_entry = _find_consultation_by_exam_id(all_cons, ris_exam_id)
            source = matched or base_entry
            if isinstance(source, dict):
                if not ctx["name"] and source.get("name"):
                    ctx["name"] = source.get("name")
                if not ctx["examineTime"]:
                    ctx["examineTime"] = source.get("examineTime") or source.get("submissionTime")
        except Exception:
            pass
    return ctx

def _resolve_consultation_time(entry: Dict[str, Any]) -> Optional[datetime.datetime]:
    """Prefer refined timestamps, fallback to base submissionTime."""
    if not isinstance(entry, dict):
        return None

    refined = entry.get("refined")
    if isinstance(refined, dict):
        for key in ("refinedTime", "submissionTime"):
            dt = _parse_iso_to_aware_dt(refined.get(key))
            if dt:
                return dt.astimezone(datetime.timezone.utc)

    dt = _parse_iso_to_aware_dt(entry.get("submissionTime"))
    if dt:
        return dt.astimezone(datetime.timezone.utc)
    return None


def _find_best_matching_index(all_data: List[Dict[str, Any]], patient_name: Optional[str], exam_time: Optional[str]) -> Optional[int]:
    """
    Same selection rule as find_best_matching_consultation, but returns index in list.
    Prefers submission/refined time <= examTime (closest), else earliest after.
    Matches strictly by the provided patient_name (including refined name field).
    """
    normalized_name = (patient_name or "").strip()
    if not normalized_name or not isinstance(all_data, list) or len(all_data) == 0:
        return None

    exam_dt = _parse_iso_to_aware_dt(exam_time)
    exam_utc = exam_dt.astimezone(datetime.timezone.utc) if exam_dt else None

    name_idxs: List[Tuple[int, Dict[str, Any]]] = []
    for i, entry in enumerate(all_data):
        if not isinstance(entry, dict):
            continue
        entry_names: List[str] = []
        base_name = entry.get("name")
        if base_name:
            entry_names.append(str(base_name).strip())
        refined = entry.get("refined")
        if isinstance(refined, dict):
            refined_name = refined.get("name")
            if refined_name:
                entry_names.append(str(refined_name).strip())

        if normalized_name and normalized_name in entry_names:
            name_idxs.append((i, entry))

    if not name_idxs:
        return None

    if not exam_utc:
        # no exam time -> pick latest by available timestamps
        def time_key(entry: Dict[str, Any]):
            dt = _resolve_consultation_time(entry)
            return dt if dt else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

        name_idxs.sort(key=lambda t: time_key(t[1]))
        return name_idxs[-1][0]

    # before or equal to exam (closest)
    candidates: List[Tuple[int, datetime.datetime]] = []
    for idx, entry in name_idxs:
        st_utc = _resolve_consultation_time(entry)
        if not st_utc:
            continue
        if st_utc <= exam_utc:
            candidates.append((idx, st_utc))
    if candidates:
        candidates.sort(key=lambda t: (exam_utc - t[1]))
        return candidates[0][0]

    # earliest after exam
    after: List[Tuple[int, datetime.datetime]] = []
    for idx, entry in name_idxs:
        st_utc = _resolve_consultation_time(entry)
        if not st_utc:
            continue
        if st_utc > exam_utc:
            after.append((idx, st_utc))
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
LLM_PROMPTS_PATH_ENV = os.getenv("LLM_PROMPTS_PATH", "config/llm_prompts.json")
if not os.path.isabs(LLM_PROMPTS_PATH_ENV):
    LLM_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), LLM_PROMPTS_PATH_ENV)
else:
    LLM_PROMPTS_PATH = LLM_PROMPTS_PATH_ENV

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
    
    # 如果患者数据中没有年龄和性别，尝试从推理结果缓存中获取
    if not age or not gender or not name or not examine_time:
        try:
            record = get_inference_record(ris_exam_id or "")
            if isinstance(record, dict):
                if not age and record.get("age") is not None:
                    age = record.get("age")
                if not gender and record.get("gender") is not None:
                    gender = record.get("gender")
                if not name and record.get("name"):
                    name = record.get("name")
                if not examine_time and record.get("examineTime"):
                    examine_time = record.get("examineTime")
        except Exception as e:
            logger.warning(f"Failed to load additional patient info: {str(e)}")

    ctx = _get_patient_context(ris_exam_id or "")

    # 从问诊数据中获取年龄和性别信息（优先级更高，因为这是用户填写的最新信息）
    cons: Optional[Dict[str, Any]] = None
    base_entry: Optional[Dict[str, Any]] = None
    search_name = (ctx.get("name") or name or "").strip()
    search_time = ctx.get("examineTime") or examine_time

    def _merge_consultation(base: Optional[Dict[str, Any]], refined: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        keys_to_preserve = [
            "name",
            "age",
            "gender",
            "phone",
            "affectedArea",
            "leftEye",
            "rightEye",
            "bothEyes",
            "submissionTime",
            "examineTime",
        ]
        merged: Dict[str, Any] = {}
        if isinstance(base, dict):
            for key in keys_to_preserve:
                if base.get(key) is not None:
                    merged[key] = base.get(key)
        if isinstance(refined, dict):
            merged.update(refined)
        if search_name and not merged.get("name"):
            merged["name"] = search_name
        return merged

    with consultation_data_lock:
        cons_all = load_consultation_data()
        if search_name:
            idx = _find_best_matching_index(cons_all, search_name, search_time)
            if idx is not None:
                base_entry = cons_all[idx]
                refined = base_entry.get("refined") if isinstance(base_entry, dict) else None
                cons = _merge_consultation(base_entry, refined)

    if isinstance(cons, dict):
        if not age and cons.get("age"):
            age = cons.get("age")
        if not gender and cons.get("gender"):
            gender = cons.get("gender")
        if not name and cons.get("name"):
            name = cons.get("name")
        if not examine_time:
            examine_time = cons.get("examineTime") or cons.get("submissionTime")

    if (not examine_time) and isinstance(base_entry, dict):
        examine_time = (
            base_entry.get("examineTime")
            or base_entry.get("submissionTime")
            or examine_time
        )

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


# --- LLM chat (streaming) ---
class LLMChatRequest(BaseModel):
    patient_id: Optional[str] = None
    messages: List[Dict[str, str]]  # [{role:'user'|'assistant'|'system', content:str}]


def _llm_env():
    # OLLAMA本地部署配置
    return {
        "base": os.getenv("LLM_API_BASE", "http://10.138.6.3:50201"),
        "model": os.getenv("LLM_MODEL", "DeepSeek-3.1:latest"),
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
    }

@app.post("/api/llm_chat_stream")
async def llm_chat_stream(req: LLMChatRequest):
    env = _llm_env()
    cfg = load_llm_prompts()
    
    # 构建消息列表
    messages_to_send: List[Dict[str, str]] = []
    # Use original system prompt from config, but keep simple fallback
    sys_prompt = cfg.get("system_prompt", "") or "你是一个有用的AI助手。"
    
    # 检查是否是首次更新推理请求（只有一个user消息且内容是update_prompt）
    is_initial_update = (
        len(req.messages or []) == 1 and 
        req.messages[0].get("role") == "user" and
        req.messages[0].get("content", "").strip() == cfg.get("update_prompt", "").strip()
    )

    context_text = ""
    if cfg.get("include_patient_context", True):
        placeholders = _build_context_placeholders(req.patient_id)
        context_text = _fill_template(cfg.get("context_template", ""), placeholders)

        if is_initial_update:
            logger.info(f"Initial context text length: {len(context_text)} characters")
        else:
            logger.info(f"Follow-up context text length: {len(context_text)} characters")

        if len(context_text) > 1000:
            context_text = context_text[:1000] + "...[TRUNCATED FOR TESTING]"
            logger.info(f"Truncated context to {len(context_text)} characters")

    if sys_prompt:
        messages_to_send.append({"role": "system", "content": sys_prompt})

    if context_text.strip():
        if len(context_text) > 2000:
            context_text = context_text[:2000] + "...[内容过长已截断]"
            logger.info(f"Trimmed context to {len(context_text)} characters before sending")
        messages_to_send.append({"role": "user", "content": "/no_think " + context_text})

    if not is_initial_update and req.messages:
        for msg in req.messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            content = msg.get("content")

            if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
                logger.warning(
                    "Invalid message format: role='%s', content type=%s, content length=%s",
                    role,
                    type(content),
                    len(str(content or ""))
                )
                continue

            cleaned = content.strip()
            if role == "user":
                if not cleaned.startswith("/no_think"):
                    cleaned = "/no_think " + cleaned
                messages_to_send.append({"role": "user", "content": cleaned})
            elif role == "assistant":
                messages_to_send.append({"role": "assistant", "content": cleaned})
            elif role == "system":
                # Rare but allow explicit system injections from client
                messages_to_send.append({"role": "system", "content": cleaned})
            else:
                logger.warning("Unsupported message role '%s' ignored", role)

    # 验证消息列表不为空
    if not messages_to_send:
        logger.error("No valid messages to send to OLLAMA")
        async def empty_response():
            yield "\n[错误] 没有有效的消息可发送\n"
        return StreamingResponse(empty_response(), media_type="text/plain; charset=utf-8")

    # === 添加DEBUG日志 ===
    logger.info("=" * 80)
    logger.info(f"LLM Chat Request for Patient: {req.patient_id} ({'INITIAL_UPDATE' if is_initial_update else 'FOLLOW_UP'})")
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

    # 缓存助手回复内容
    assistant_buf: List[str] = []

    async def ollama_stream():
        # OLLAMA API端点
        url = f"{env['base'].rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": env["model"],
            "messages": messages_to_send,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE
        }
        
        logger.info(f"Sending request to OLLAMA: {url}")
        logger.info(f"Payload model: {payload['model']}")
        logger.info(f"Payload stream: {payload['stream']}")
        logger.info(f"Payload messages: {payload['messages']}")
        logger.info(f"Payload keep_alive: {payload['keep_alive']}")
        logger.info(f"Request type: {'COMBINED_USER_PROMPT' if is_initial_update else 'USER_CONTEXT_PROMPT'}")
        
        timeout = httpx.Timeout(120.0, read=120.0)
        last_exception: Optional[Exception] = None

        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
                logger.info(
                    "Starting OLLAMA streaming attempt %s/%s for patient %s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    req.patient_id
                )

                response_chunks = 0
                pending_text = ""
                max_buffer_len = 16384

                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async with client.stream("POST", url, headers=headers, json=payload) as r:
                            logger.info(f"OLLAMA Response Status: {r.status_code}")

                            if r.status_code != 200:
                                error_text = await r.atext()
                                last_exception = RuntimeError(f"HTTP {r.status_code}: {error_text}")
                                logger.error("OLLAMA Error Response (attempt %s): %s", attempt, error_text)
                                continue

                            async for raw_line in r.aiter_lines():
                                if raw_line is None:
                                    continue

                                line = raw_line.strip()
                                if not line:
                                    continue

                                if line.startswith(":"):
                                    continue

                                if line.startswith("event:"):
                                    logger.debug(f"Skipping SSE event label: {line}")
                                    continue

                                if line.startswith("data:"):
                                    line = line[len("data:"):].strip()
                                    if not line:
                                        continue

                                if line in ("[DONE]", '"[DONE]"'):
                                    logger.info("OLLAMA stream signaled completion token [DONE].")
                                    break

                                pending_text += line

                                try:
                                    obj = json.loads(pending_text)
                                    pending_text = ""
                                except json.JSONDecodeError:
                                    if len(pending_text) > max_buffer_len:
                                        logger.warning(
                                            "Streaming buffer exceeded %s characters; resetting to avoid runaway growth.",
                                            max_buffer_len
                                        )
                                        pending_text = ""
                                    else:
                                        logger.debug(
                                            "Waiting for additional stream data to complete JSON frame (buffer length=%s).",
                                            len(pending_text)
                                        )
                                    continue

                                if not isinstance(obj, dict):
                                    logger.debug(f"Ignoring non-dict stream payload: {obj}")
                                    continue

                                response_chunks += 1
                                if response_chunks == 1:
                                    logger.info("Started receiving OLLAMA streaming response...")

                                logger.info(f"Parsed response chunk #{response_chunks}: {obj}")

                                if "error" in obj:
                                    err_text = obj.get("error")
                                    last_exception = RuntimeError(err_text)
                                    logger.error("OLLAMA Stream Error (attempt %s): %s", attempt, err_text)
                                    break

                                message = obj.get("message", {})
                                content = message.get("content", "")

                                if content:
                                    assistant_buf.append(content)
                                    yield content

                                if obj.get("done", False):
                                    logger.info(
                                        "OLLAMA response completed. Total chunks: %s, response length: %s chars",
                                        response_chunks,
                                        len(''.join(assistant_buf))
                                    )
                                    break

                except httpx.TimeoutException as e:
                    last_exception = e
                    logger.warning(
                        "OLLAMA request timeout on attempt %s/%s",
                        attempt,
                        OLLAMA_MAX_RETRIES,
                        exc_info=e
                    )
                except httpx.ConnectError as e:
                    last_exception = e
                    logger.warning(
                        "OLLAMA connection error on attempt %s/%s: %s",
                        attempt,
                        OLLAMA_MAX_RETRIES,
                        e
                    )
                except httpx.RequestError as e:
                    last_exception = e
                    logger.warning(
                        "OLLAMA request error on attempt %s/%s: %s",
                        attempt,
                        OLLAMA_MAX_RETRIES,
                        e
                    )
                except Exception as e:
                    last_exception = e
                    logger.error(
                        "OLLAMA request failed on attempt %s/%s: %s",
                        attempt,
                        OLLAMA_MAX_RETRIES,
                        e
                    )

                if assistant_buf:
                    logger.info("OLLAMA streaming succeeded on attempt %s", attempt)
                    return

                if attempt < OLLAMA_MAX_RETRIES:
                    logger.info(
                        "Retrying OLLAMA stream (next attempt %s/%s) after %.2f seconds",
                        attempt + 1,
                        OLLAMA_MAX_RETRIES,
                        OLLAMA_RETRY_DELAY_SECONDS
                    )
                    await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)

        if not assistant_buf:
            if last_exception:
                logger.error(
                    "OLLAMA streaming failed after %s attempts: %s",
                    OLLAMA_MAX_RETRIES,
                    last_exception
                )
                yield f"\n[OLLAMA **错误**] {str(last_exception)}\n"
            else:
                logger.warning(
                    "OLLAMA streaming produced no content after %s attempts without explicit exception",
                    OLLAMA_MAX_RETRIES
                )
                yield "\n[LLM warning] 模型未返回内容，请稍后重试或检查日志。\n"
            return

    return StreamingResponse(ollama_stream(), media_type="text/plain; charset=utf-8")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
